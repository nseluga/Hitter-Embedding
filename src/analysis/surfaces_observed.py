"""
Observed counterpart of the V.9 location surfaces (model_visualization_anchors.run_v9).

The model surface is q = E[wOBA points | ball in play] by plate location, anchor minus a
league average. This script computes the same thing from what actually happened: every
measured ball in play is given the SAME value the model's q is built from (the outcome
table looked up at the ball's observed exit-velocity, launch-angle and spray bins), then
averaged per location cell, anchor minus all hitters, against each pitcher hand.

Two differences from the model surface that the comparison cannot remove:
  - the model scores every hitter on one fixed pool of pitches; the observed side can only
    use the pitches each hitter actually put in play, so pitch mix differs by hitter;
  - the model's league is a 100-hitter average; the observed league is every ball in play.

Rows are written for the training seasons (what the model saw) and for the held-out season
(2024 for the selection model), so the figure can show the first and the text can report
how well the surface carries to a season the model never saw. 2025 is never read.

Run:  PYTHONPATH=. python -m src.analysis.surfaces_observed
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.model_visualization_anchors import (
    EVAL_SEASON_WEIGHTS, LOCATION_X_EDGES, LOCATION_Z_EDGES, bin_index)
from src.analysis.model_visualization_stats import ANCHORS
from src.data import eval_targets
from src.model import loader, query, query_tables as qt

MIN_BIP = 20   # cells with fewer balls in play are too noisy to draw or to score


def observed_cells(df, anchors=ANCHORS):
    """
    df: one row per ball in play with batter, p_throws, x_bin, z_bin, value.
    Returns per (anchor, hand, cell): n, anchor mean, league mean, delta, standard error.
    """
    keys = ["p_throws", "x_bin", "z_bin"]
    league = df.groupby(keys)["value"].mean().rename("league_value")
    mine = df[df["batter"].isin(anchors)].groupby(["batter"] + keys)["value"].agg(
        n="size", anchor_value="mean", sd="std").reset_index()
    out = mine.join(league, on=keys)
    out["delta"] = out["anchor_value"] - out["league_value"]
    out["se"] = out["sd"] / np.sqrt(out["n"])
    out["anchor_name"] = out["batter"].map(anchors)
    return out.rename(columns={"batter": "anchor"}).drop(columns="sd")


def agreement(model, observed, min_bip=MIN_BIP):
    """Correlation of model delta with observed delta over cells with enough balls in play."""
    keys = ["anchor", "p_throws", "x_bin", "z_bin"]
    m = model.merge(observed[observed["n"] >= min_bip], on=keys, suffixes=("_model", "_obs"))
    w = m["n"].to_numpy()
    x, y = m["delta_model"].to_numpy(), m["delta_obs"].to_numpy()
    xc, yc = x - np.average(x, weights=w), y - np.average(y, weights=w)
    weighted = float((w * xc * yc).sum() / np.sqrt((w * xc ** 2).sum() * (w * yc ** 2).sum()))
    return {"n_cells": int(len(m)), "n_bip": int(w.sum()),
            "r": float(np.corrcoef(x, y)[0, 1]), "r_weighted_by_bip": weighted}


def main():
    parser = argparse.ArgumentParser(description="Observed location surfaces for the anchors.")
    parser.add_argument("--out-dir", default="results/v_chain_clean")
    parser.add_argument("--data-dir", default="data/processed/phase_d5_clean")
    parser.add_argument("--pitch-events", default="data/processed/pitch_events_labeled.parquet")
    parser.add_argument("--eval-targets", default="data/processed/eval_targets_pa.parquet")
    parser.add_argument("--heldout-season", type=int, default=2024)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    tensors, manifest = loader.load_tensors(args.data_dir)
    train_seasons = manifest["train_seasons"]
    assert args.heldout_season not in train_seasons and args.heldout_season <= 2024, \
        "held-out season must be outside training and 2025 must never be read"

    frame = qt.align_pitch_frame(args.pitch_events, args.eval_targets, tensors["season"],
                                 career_pitchers=manifest.get("career_pitchers_excluded", False))
    tables = query.build_tables(frame, tensors, manifest, pd.read_parquet(args.eval_targets))
    weights = eval_targets.load_weights()[EVAL_SEASON_WEIGHTS]
    # the same (ev, la, spray) -> wOBA points table the model's q is composed from
    points = qt.woba_points_table(tables["outcome"], weights)

    ev, la, spray = (tensors[k].numpy() for k in ("ev", "la", "spray"))
    measured = (ev != -1) & (la != -1) & (spray != -1)
    df = pd.DataFrame({
        "batter": frame["batter"].to_numpy(), "season": frame["season"].to_numpy(),
        "p_throws": frame["p_throws"].to_numpy(),
        "x_bin": bin_index(frame["plate_x"].to_numpy(), LOCATION_X_EDGES),
        "z_bin": bin_index(frame["plate_z"].to_numpy(), LOCATION_Z_EDGES),
    })[measured]
    df["value"] = points[ev[measured], la[measured], spray[measured]]

    periods = {"train": df[df["season"].isin(train_seasons)],
               "heldout": df[df["season"] == args.heldout_season]}
    observed = pd.concat([observed_cells(part).assign(period=name)
                          for name, part in periods.items()], ignore_index=True)
    observed.to_csv(out_dir / "surfaces_location_observed.csv", index=False)

    model = pd.read_csv(out_dir / "surfaces_location.csv")
    model = model[model["quantity"] == "q"]
    report = {"min_bip": MIN_BIP, "train_seasons": [int(s) for s in train_seasons],
              "heldout_season": args.heldout_season,
              **{name: agreement(model, observed[observed["period"] == name])
                 for name in periods}}
    (out_dir / "surfaces_location_agreement.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print(observed.groupby(["period", "anchor_name", "p_throws"])["n"].agg(["sum", "min", "median"]))


if __name__ == "__main__":
    main()
