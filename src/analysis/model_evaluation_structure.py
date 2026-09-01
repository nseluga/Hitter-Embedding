"""
Phase E.10 — is the count chain ITSELF biased toward walks?

E.6 exonerated the swing head. E.7-E.9 measured both channels through which the pitch
resampler could distort the walk rate and found they cancel to -0.00003, against a residual
excess of +0.00226. Neither named suspect owns it, so this asks whether the composition's
own STRUCTURE does.

The chain treats a plate appearance as independent draws from a (pitcher, stand, count)
cell. Real plate appearances are not independent draws: a pitcher who has just thrown two
balls is not a random pitcher at 2-0, and how he attacks depends on the hitter in a way the
marginal cell rate averages away. If that dependence matters, a chain built from PERFECTLY
observed transition frequencies will still miss the observed outcome rates.

That is a model-free test, and it is the point of this module. Every transition here is a
counted frequency from real pitches. No network is loaded, no embedding is read. Whatever
gap survives belongs to the composition, not to anything Phase D trained.

Two aggregations are reported because the composition uses the second:
  pooled       one league chain from pooled frequencies
  per-pitcher  a chain per pitcher, W(0,0) averaged with the composition's own weights
The difference between them is the Jensen term the composition already carries.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import claim1_eval
from src.analysis.model_evaluation_price_draw import league_chain, to_aggregates, SWING_MISS, FOUL, STRIKE_LIKE
from src.model import query, query_tables as qt

CHANNELS = ("ball", "strike", "hbp", "foul", "bip")


def _channel_masks(description, keep):
    masks = {
        "ball": np.isin(description, qt.BALL_DESCRIPTIONS),
        "hbp": description == "hit_by_pitch",
        "foul": np.isin(description, FOUL),
        "bip": description == qt.IN_PLAY_DESCRIPTION,
    }
    masks["strike"] = np.isin(description, STRIKE_LIKE) | np.isin(description, SWING_MISS)
    # checked on the KEPT rows only: `intent_ball` is deliberately in no channel, because
    # an intentional ball sits outside the wOBA denominator the chain covers
    stacked = sum(masks.values())[keep]
    assert stacked.max() == 1 and stacked.min() == 1, \
        "pitch descriptions do not partition the outcome space exactly"
    return masks


def per_pitcher_chains(frame, keep, min_pitches=200):
    """
    One (4, 3) transition table per pitcher, then absorbing rates per pitcher.

    A pitcher with an empty cell cannot finish a plate appearance, so his cell backs off to
    the league rate at that count -- the same failure the composition handles by marking him
    unusable, handled here by substitution so the population stays whole rather than
    silently selecting toward high-volume pitchers.

    Built with one groupby rather than a mask per (pitcher, count): 2,000 pitchers times 12
    counts is 24,000 full scans of a 7.3M-row column, which is an afternoon.
    """
    masks = _channel_masks(frame["description"].to_numpy(), keep)
    sub = frame.loc[keep, ["pitcher", "balls", "strikes"]].copy()
    for name in CHANNELS:
        sub[name] = masks[name][keep].astype("float64")

    pooled_counts = sub.groupby(["balls", "strikes"])[list(CHANNELS)].sum()
    pooled = {name: np.zeros((qt.N_BALLS, qt.N_STRIKES)) for name in CHANNELS}
    for (b, s), row in pooled_counts.iterrows():
        total = row.sum()
        for name in CHANNELS:
            pooled[name][b, s] = row[name] / total

    grouped = sub.groupby(["pitcher", "balls", "strikes"])[list(CHANNELS)].sum()
    totals = grouped.sum(axis=1)
    shares = grouped.div(totals, axis=0)
    per_pitcher_n = sub.groupby("pitcher").size()

    rows, weights = [], []
    for one, n_pitches in per_pitcher_n.items():
        if n_pitches < min_pitches:
            continue
        table = {name: pooled[name].copy() for name in CHANNELS}
        mine = shares.loc[one]
        for (b, s), cell in mine.iterrows():
            for name in CHANNELS:
                table[name][b, s] = cell[name]
        rates = query.absorbing_rates(to_aggregates(table))
        rows.append({key: float(rates[key][0, 0]) for key in query.ABSORBING_KEYS})
        weights.append(float(n_pitches))
    return pd.DataFrame(rows), np.asarray(weights)


def main():
    parser = argparse.ArgumentParser(
        description="Phase E.10 — structural bias of the independent-pitch count chain.")
    parser.add_argument("--eval-season", type=int, default=2024)
    parser.add_argument("--final-run", action="store_true")
    parser.add_argument("--data-dir", default="data/processed/phase_d")
    parser.add_argument("--pitch-events", default="data/processed/pitch_events_labeled.parquet")
    parser.add_argument("--eval-targets", default="data/processed/eval_targets_pa.parquet")
    parser.add_argument("--out-dir", default="results/model_evaluation")
    args = parser.parse_args()

    claim1_eval.assert_not_test_season(args.eval_season, final_run=args.final_run)
    out_dir = Path(args.out_dir)
    manifest = json.loads((Path(args.data_dir) / "manifest.json").read_text())
    season = np.load(Path(args.data_dir) / "season.npy", mmap_mode="r")
    frame = qt.align_pitch_frame(args.pitch_events, args.eval_targets, np.asarray(season))
    pa_df = pd.read_parquet(args.eval_targets)
    seasons = manifest["train_seasons"]
    keep = (np.isin(frame["season"].to_numpy(), seasons)
            & ~frame["description"].isin(qt.EXCLUDED_DESCRIPTIONS).to_numpy())

    # the observed side is restricted to the SAME plate appearances the pitch frame covers,
    # by (game_pk, at_bat_number). Comparing against the shipped all-PA reference instead
    # would reintroduce exactly the population mismatch E.1 was written to remove.
    covered = frame.loc[keep, qt.JOIN_KEYS].drop_duplicates()
    window = pa_df[np.isin(pa_df["season"].to_numpy(), seasons)
                   & pa_df["in_denominator"].to_numpy()]
    window = window.merge(covered, on=qt.JOIN_KEYS, how="inner")
    events = window["events"].to_numpy()
    observed = {key: float(np.isin(events, names).sum()) / len(window)
                for key, names in query.ABSORBING_EVENTS.items()}
    observed["bip"] = 1.0 - sum(observed.values())

    shares = query.handedness_shares(pa_df, seasons)
    pooled = query.absorbing_rates(to_aggregates(league_chain(frame, keep, shares)))
    pooled = {key: float(pooled[key][0, 0]) for key in query.ABSORBING_KEYS}

    per_pitcher, weights = per_pitcher_chains(frame, keep)
    averaged = {key: float(np.average(per_pitcher[key], weights=weights))
                for key in query.ABSORBING_KEYS}

    table = pd.DataFrame([
        {"aggregation": name, **{f"rate_{key}": rates[key] for key in query.ABSORBING_KEYS},
         **{f"relerr_{key}": rates[key] / observed[key] - 1.0
            for key in query.ABSORBING_KEYS}}
        for name, rates in (("observed", observed), ("pooled_chain", pooled),
                            ("per_pitcher_chain", averaged))])
    table.to_csv(out_dir / "structure.csv", index=False)

    residual = 0.08538 - 0.08312     # E.1's population-matched walk excess
    summary = {
        "n_pa": int(len(window)), "n_pitchers": int(len(per_pitcher)),
        "observed_bb": observed["bb"], "pooled_chain_bb": pooled["bb"],
        "per_pitcher_chain_bb": averaged["bb"],
        "structural_bb_bias_pooled": pooled["bb"] - observed["bb"],
        "structural_bb_bias_per_pitcher": averaged["bb"] - observed["bb"],
        "jensen_term_bb": averaged["bb"] - pooled["bb"],
        "structural_bb_bias_relative": averaged["bb"] / observed["bb"] - 1.0,
        "residual_walk_excess": residual,
        "share_of_residual_explained": (averaged["bb"] - observed["bb"]) / residual,
        "structural_k_bias_relative": averaged["k"] / observed["k"] - 1.0,
    }
    (out_dir / "structure_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n-- observed vs a chain built from perfectly observed transitions --")
    print(table.to_string(index=False))
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
