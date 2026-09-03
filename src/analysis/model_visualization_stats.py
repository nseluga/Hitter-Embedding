"""
Phase V observable-stat panel (docs/phase-v-spec.md §1, §2 -- the CSV every V.3-V.5
loadings figure joins against). Read-only: no model, scorer, or loss code runs here.

Everything is computed on TRAINING seasons only (2015-2023), never on 2024/2025 --
this module has no eval_season argument and no path reads a later season.

`pull_rate` reuses the `spray` label already produced by `src.data.labels` rather
than re-deriving it from `hc_x`/`hc_y`: `spray` is the horizontal launch angle off
home plate (origin `HOME_PLATE_HC_X`, `HOME_PLATE_HC_Y` = 125.42, 198.27, the
Statcast community-convention origin), mirrored by `stand` so POSITIVE ALREADY
MEANS PULL FOR BOTH HANDS (see `labels.add_contact_quality_labels`). A ball is
"pulled" here when that mirrored spray angle exceeds `PULL_THRESHOLD_DEG` (15
degrees toward the batter's pull side), over the same in-play, coordinate-valid
population the label itself defines (|spray| <= 90, hit coordinates present).

`prior_pa_L`/`prior_pa_R` and `n_pa_L`/`n_pa_R` are the SAME quantity (denominator
PA against that hand, summed over 2015-2023): `claim1_eval.prior_exposure` with
eval_season=2024 sums exactly the seasons strictly before 2024, which is 2015-2023
given the data's earliest season is 2015 (asserted below). They are exposed under
both names because the spec lists them for two different readers -- V.2's log-PA
exposure axis, and V.3/V.5's platoon differential -- not because they differ.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.analysis.claim1_eval import assign_stratum, prior_exposure
from src.data.eval_targets import aggregate, drop_pitcher_batters

TRAIN_SEASONS = (2015, 2023)
EVAL_SEASON_FOR_PRIOR = TRAIN_SEASONS[1] + 1  # 2024: prior_exposure sums seasons < this

# Shared across every Phase V script that names the five anchor hitters (anchor swapped
# Duvall -> Bohm 2026-09-02, see decision log). Short names derive from the full name's
# first token, matching the prior per-module ANCHOR_IDS convention.
ANCHORS = {
    545361: "Mike Trout",
    665742: "Juan Soto",
    592626: "Joc Pederson",
    664761: "Alec Bohm",
    656941: "Kyle Schwarber",
}
ANCHOR_SHORT_NAMES = {batter: name.split()[-1] for batter, name in ANCHORS.items()}

PITCH_PARQUET = "data/processed/pitch_events_labeled.parquet"
EVAL_TARGETS_PARQUET = "data/processed/eval_targets_pa.parquet"
MANIFEST_PATH = "data/processed/phase_d5/manifest.json"

CHASE_ZONES = (11, 12, 13, 14)
IN_ZONE_ZONES = tuple(range(1, 10))
PULL_THRESHOLD_DEG = 15.0

PITCH_COLUMNS = ["batter", "stand", "season", "swing", "contact", "zone", "ev", "la",
                  "spray", "bat_speed"]


def load_pitch_frame(path=PITCH_PARQUET):
    """
    Read only the columns this module needs (pyarrow column projection) and restrict
    to training seasons 2015-2023. Asserts no other season survives the filter.
    """
    df = pq.read_table(path, columns=PITCH_COLUMNS).to_pandas()
    df = df[df["season"].between(*TRAIN_SEASONS)]
    assert set(df["season"].unique()) <= set(range(TRAIN_SEASONS[0], TRAIN_SEASONS[1] + 1)), \
        "a season outside 2015-2023 survived the filter"
    return df


def per_batter_pitch_stats(df):
    """
    Swing/contact/quality/pull rates per batter over training-season pitches.
    df: pitch_events_labeled rows already filtered to seasons 2015-2023.
    Returns one row per batter with n_pitches, the rate columns, and majority stand.
    """
    grouped = df.groupby("batter")
    swings = df[df["swing"] == 1]
    chase = df[df["zone"].isin(CHASE_ZONES)]
    in_zone = df[df["zone"].isin(IN_ZONE_ZONES)]
    batted_ev = df[df["ev"].notna()]
    batted_la = df[df["la"].notna()]
    batted_spray = df[df["spray"].notna()]
    tracked_bat_speed = df[df["bat_speed"].notna()]

    stats = pd.DataFrame({
        "n_pitches": grouped.size(),
        "swing_rate": grouped["swing"].mean(),
        "stand": grouped["stand"].agg(lambda values: values.mode().iat[0]),
    })
    stats["whiff_rate"] = 1 - swings.groupby("batter")["contact"].mean()
    stats["contact_rate"] = swings.groupby("batter")["contact"].mean()
    stats["chase_rate"] = chase.groupby("batter")["swing"].mean()
    stats["zone_swing_rate"] = in_zone.groupby("batter")["swing"].mean()
    stats["ev_mean"] = batted_ev.groupby("batter")["ev"].mean()
    stats["ev_p90"] = batted_ev.groupby("batter")["ev"].quantile(0.9)
    stats["la_mean"] = batted_la.groupby("batter")["la"].mean()
    stats["bat_speed_mean"] = tracked_bat_speed.groupby("batter")["bat_speed"].mean()
    stats["pull_rate"] = batted_spray.groupby("batter")["spray"].apply(
        lambda values: (values > PULL_THRESHOLD_DEG).mean())
    return stats.reset_index()


def per_batter_exposure_and_platoon(pa_df):
    """
    Training-season (2015-2023) wOBA level, side-specific exposure, and the
    observed platoon differential per batter, from eval_targets_pa.
    pa_df: the raw PA-level eval-target table (any seasons); filtered internally.
    """
    pa_df = drop_pitcher_batters(pa_df)
    train_pa = pa_df[pa_df["season"].between(*TRAIN_SEASONS)]
    assert set(train_pa["season"].unique()) <= set(range(TRAIN_SEASONS[0], TRAIN_SEASONS[1] + 1))

    level = aggregate(train_pa, by=("batter",))[["batter", "woba"]] \
        .rename(columns={"woba": "woba_level"})

    prior = prior_exposure(pa_df, eval_season=EVAL_SEASON_FOR_PRIOR)
    prior_wide = (prior.pivot(index="batter", columns="p_throws", values="prior_pa")
                  .reindex(columns=["L", "R"]).fillna(0.0))
    prior_wide.columns = ["prior_pa_L", "prior_pa_R"]
    prior_wide = prior_wide.reset_index()
    prior_wide["n_pa_L"] = prior_wide["prior_pa_L"]
    prior_wide["n_pa_R"] = prior_wide["prior_pa_R"]
    prior_wide["log_prior_pa"] = np.log1p(prior_wide["prior_pa_L"] + prior_wide["prior_pa_R"])
    prior_wide["stratum"] = assign_stratum(
        prior_wide[["prior_pa_L", "prior_pa_R"]].min(axis=1))

    side_woba = aggregate(train_pa, by=("batter", "p_throws"))[["batter", "p_throws", "woba"]]
    side_wide = side_woba.pivot(index="batter", columns="p_throws", values="woba") \
        .reindex(columns=["L", "R"])
    side_wide.columns = ["woba_vs_L", "woba_vs_R"]
    side_wide = side_wide.reset_index()

    out = level.merge(prior_wide, on="batter", how="outer").merge(side_wide, on="batter", how="outer")
    # LHB: better vs opposite hand (RHP) is positive; RHB: better vs opposite hand (LHP)
    stand = df_majority_stand(pa_df, train_pa)
    out = out.merge(stand, on="batter", how="left")
    is_lhb = out["stand"] == "L"
    out["obs_platoon_diff"] = np.where(
        is_lhb, out["woba_vs_R"] - out["woba_vs_L"], out["woba_vs_L"] - out["woba_vs_R"])
    return out.drop(columns=["woba_vs_L", "woba_vs_R", "stand"])


def df_majority_stand(pa_df, train_pa):
    """Majority batting stand per batter over training-season PA, for platoon orientation."""
    return (train_pa.groupby("batter")["stand"].agg(lambda values: values.mode().iat[0])
            .rename("stand").reset_index())


def load_vocabulary(path=MANIFEST_PATH):
    """
    Batter -> embedding_index for every hitter in the frozen phase_d5 vocabulary
    (the join spine: every vocabulary hitter gets a row, even with all-NaN stats).
    """
    manifest = json.loads(Path(path).read_text())
    vocab = manifest["vocabulary"]
    return pd.DataFrame({"batter": [int(batter) for batter in vocab],
                          "embedding_index": list(vocab.values())})


def build_hitter_stats(pitch_path=PITCH_PARQUET, eval_targets_path=EVAL_TARGETS_PARQUET,
                        manifest_path=MANIFEST_PATH):
    """
    Assemble the full per-hitter observable-stat panel, joined onto the phase_d5
    vocabulary. Returns one row per vocabulary hitter (NaN stats if unobserved).
    """
    pitch_df = load_pitch_frame(pitch_path)
    pitch_stats = per_batter_pitch_stats(pitch_df)

    pa_df = pd.read_parquet(eval_targets_path)
    platoon_stats = per_batter_exposure_and_platoon(pa_df)

    vocab = load_vocabulary(manifest_path)
    out = vocab.merge(pitch_stats, on="batter", how="left").merge(platoon_stats, on="batter", how="left")
    assert len(out) == len(vocab), "vocabulary join changed row count"
    return out


def main():
    parser = argparse.ArgumentParser(description="Phase V per-hitter observable-stat panel.")
    parser.add_argument("--out-dir", default="results/model_visualization")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = build_hitter_stats()
    out_path = out_dir / "hitter_stats.csv"
    stats.to_csv(out_path, index=False)
    print(f"rows written: {len(stats)}")
    print(f"output: {out_path}")


if __name__ == "__main__":
    main()
