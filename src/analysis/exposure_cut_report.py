"""
Exposure-cut report -- the real, rerunnable numbers behind "Where the Exposure Cut
Falls" (a prior ad-hoc session, unlogged, visible only in a Claude artifact and
handoff prose). Reuses claim1_eval's stratum machinery and stabilization.py's
variance-components estimator; recomputes nothing that already has an answer.

NOT a claim-1 re-score. The sealed 2025 test season is used only to count the
population by prior-exposure band (the same stratification claim1_eval already
does before scoring), never to score predictions -- the one-pass rule is not
implicated.

What it computes (decision-log 2026-09-08, "Decisive-stratum floor confirmed..."):
  1. Pearson r between per-hitter pitch count and PA count vs LHP (train seasons)
     -- confirms the ~0.997 figure on committed data.
  2. pitches-per-PA ratio vs LHP, recomputed directly (~3.90 claimed).
  3. An independent PITCH-unit stabilization point: swing rate vs LHP (the natural
     per-pitch metric stabilization.py already reports) through the same
     variance-components estimator STABILIZATION_N_STAR came from, compared to
     STABILIZATION_N_STAR (PA-based, wOBA) converted to pitches via the ratio in
     (2). These are two different metrics on two different units, not the same n*
     measured twice -- the comparison is a sanity check on the exposure-unit
     conversion, not a claim they must coincide.
  4. The 2025 platoon-frame population by prior-PA-vs-LHP band (train-only prior
     exposure, hitters only) -- a population count, mirroring the stratification
     step inside claim1_eval.build_eval_frame without scoring anything.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import claim1_eval as evaluation
from src.analysis import stabilization as stab
from src.data.eval_targets import aggregate, drop_pitcher_batters

OUT_DIR = Path("results/exposure_cut")
DEFAULT_MAX_TRAIN_SEASON = 2023  # matches stabilization.py's own train cutoff
DEFAULT_EVAL_SEASON = 2025  # sealed test season -- population count only, never scored


def pitch_pa_correlation(labeled, pa_df, max_train_season):
    """Per-hitter (pitches, PA) vs LHP on train seasons: Pearson r and the pitches/PA ratio."""
    pitches = (labeled[(labeled["season"] <= max_train_season) & (labeled["p_throws"] == "L")]
               .groupby("batter").size().rename("pitches"))
    pa = (pa_df[(pa_df["season"] <= max_train_season) & (pa_df["p_throws"] == "L")
                & pa_df["in_denominator"]]
          .groupby("batter").size().rename("pa"))
    matched = pd.concat([pitches, pa], axis=1, join="inner")
    r = float(np.corrcoef(matched["pitches"], matched["pa"])[0, 1])
    ratio = float(matched["pitches"].sum() / matched["pa"].sum())
    return r, ratio, len(matched)


def pitch_unit_stabilization(labeled, max_train_season, seed):
    """Independent pitch-unit n*: swing rate vs LHP through the variance-components estimator."""
    pop = labeled[(labeled["season"] <= max_train_season) & (labeled["p_throws"] == "L")]
    stats = stab.hitter_stats(pop, "batter", "swing")
    point, lo, hi = stab.stabilization_ci(stats, threshold=0.5, seed=seed)
    return point, lo, hi, len(stats)


def population_table(pa_df, eval_season, boundaries=evaluation.STRATUM_BOUNDARIES):
    """2025 platoon-frame population by prior-PA-vs-LHP band -- count only, not scored."""
    pa_df = drop_pitcher_batters(pa_df)
    actuals = aggregate(pa_df, by=("batter", "season", "p_throws"))
    actuals = actuals[(actuals["season"] == eval_season) & (actuals["p_throws"] == "L")]

    prior = evaluation.prior_exposure(pa_df, eval_season)
    prior = prior[prior["p_throws"] == "L"][["batter", "prior_pa"]]
    frame = actuals[["batter"]].merge(prior, on="batter", how="left")
    frame["prior_pa"] = frame["prior_pa"].fillna(0.0)
    frame["stratum"] = evaluation.assign_stratum(frame["prior_pa"], boundaries)

    counts = frame["stratum"].value_counts().reindex(evaluation.STRATUM_NAMES, fill_value=0)
    return {
        "eval_season": int(eval_season),
        "n_hitters": int(len(frame)),
        "n_above_low_cut": int(frame["prior_pa"].ge(boundaries[0]).sum()),
        "by_stratum": {k: int(v) for k, v in counts.items()},
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Exposure-cut report: pitch-vs-PA exposure, pitch-unit stabilization, "
                    "2025 population by prior-PA band. Not a claim-1 re-score.")
    parser.add_argument("--labeled", default="data/processed/pitch_events_labeled.parquet")
    parser.add_argument("--eval-targets", default="data/processed/eval_targets_pa.parquet")
    parser.add_argument("--max-train-season", type=int, default=DEFAULT_MAX_TRAIN_SEASON)
    parser.add_argument("--eval-season", type=int, default=DEFAULT_EVAL_SEASON)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    labeled = pd.read_parquet(args.labeled, columns=["batter", "season", "p_throws", "swing"])
    pa_df = pd.read_parquet(args.eval_targets)

    r, ratio, n_matched = pitch_pa_correlation(labeled, pa_df, args.max_train_season)
    print(f"pitches-vs-PA (LHP, train<={args.max_train_season}): r={r:.4f}, "
          f"ratio={ratio:.3f} pitches/PA, n={n_matched} hitters")

    point, lo, hi, n_hitters = pitch_unit_stabilization(labeled, args.max_train_season, args.seed)
    predicted_pitches = evaluation.STABILIZATION_N_STAR * ratio
    agree = np.isfinite(point) and lo <= predicted_pitches <= hi
    print(f"pitch-unit n* (swing rate vs LHP, r=0.5): {point:.0f} pitches, "
          f"95% CI [{lo:.0f}, {hi:.0f}], n={n_hitters} hitters")
    print(f"  vs STABILIZATION_N_STAR({evaluation.STABILIZATION_N_STAR} PA) x ratio = "
          f"{predicted_pitches:.0f} pitches -- "
          + ("within CI" if agree else "outside CI (different metrics -- not expected to coincide)"))

    population = population_table(pa_df, args.eval_season)
    print(f"\n{args.eval_season} platoon frame (prior PA vs LHP, train-only exposure): "
          f"{population['n_hitters']} hitters, {population['n_above_low_cut']} at/above the "
          f"{evaluation.STRATUM_BOUNDARIES[0]}-PA low-stratum cut")
    print(f"  by stratum: {population['by_stratum']}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "max_train_season": args.max_train_season,
        "eval_season": args.eval_season,
        "pearson_r_pitches_vs_pa": r,
        "pitches_per_pa_ratio": ratio,
        "n_matched_hitters": n_matched,
        "n_star_pa": evaluation.STABILIZATION_N_STAR,
        "n_star_pitches_vc": point,
        "n_star_pitches_vc_ci_low": lo,
        "n_star_pitches_vc_ci_high": hi,
        "n_star_pitches_predicted_from_pa_ratio": predicted_pitches,
        "population": population,
    }
    (out_dir / "exposure_cut_report.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame([{
        "pearson_r_pitches_vs_pa": r,
        "pitches_per_pa_ratio": ratio,
        "n_matched_hitters": n_matched,
        "n_star_pa": evaluation.STABILIZATION_N_STAR,
        "n_star_pitches_vc": point,
        "n_star_pitches_vc_ci_low": lo,
        "n_star_pitches_vc_ci_high": hi,
        "n_star_pitches_predicted_from_pa_ratio": predicted_pitches,
        "eval_season": args.eval_season,
        "n_hitters_eval": population["n_hitters"],
        "n_above_low_cut": population["n_above_low_cut"],
        "n_low": population["by_stratum"]["low"],
        "n_medium": population["by_stratum"]["medium"],
        "n_high": population["by_stratum"]["high"],
    }]).to_csv(out_dir / "exposure_cut_report.csv", index=False)
    print(f"\nwrote 2 files to {out_dir}")


if __name__ == "__main__":
    main()
