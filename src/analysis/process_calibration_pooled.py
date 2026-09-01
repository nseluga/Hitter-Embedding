"""
Phase F.5 -- the claim-1 metric with handedness pooled away.

Every scored row in this project is a (batter, season, pitcher hand) triple, because the
claim is about the platoon split. Nothing in the repo said how well the model predicts a
hitter's overall wOBA. This adds that row.

It is a re-aggregation, not a new scorer: the frozen referee in `claim1_eval` runs
unchanged, on a frame whose hand column has been collapsed to a single value. That is
deliberate -- a second scoring path would be a second thing to keep in step with the first.

TWO THINGS TO READ CAREFULLY.

  The pooling weight is PRIOR side-specific exposure, not eval-season exposure. Weighting a
  forecast by how much the hitter went on to play against each hand conditions the
  prediction on the answer key's own sample, which is the leak `claim1_eval` avoids by
  stratifying on prior exposure. Hitters with no prior record against a hand fall back to
  equal weights.

  The strata do not mean what they mean elsewhere. `STRATUM_BOUNDARIES` was calibrated on
  SIDE-SPECIFIC prior PA, and pooled exposure is roughly twice that, so the same cut points
  select a different population. The rows are emitted with `stratum_basis` set to
  `pooled_prior_pa` and are not comparable to the side-specific strata in any other artifact.

COVERAGE SHORTCUT, stated where the result is stated: only the rungs whose predictions are
cheap to rebuild are pooled -- the model, both C.1 variants, and the no-information
reference. C.2 needs its fitted prior parameters and C.3 needs a GBM refit, so neither is
here. This table therefore does not contain the two named gate opponents and CANNOT be read
as a gate of any kind.

Nothing here trains.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import claim1_eval, provenance
from src.analysis.baseline_ladder_trailing import predict as trailing_predict
from src.analysis.baseline_ladder_report import NO_INFO_BUCKETS

DEFAULT_OUT_DIR = "results/process_calibration"
POOLED_HAND = "B"


def pooling_weights(pa_df, eval_season):
    """
    Prior side-specific denominator PA per (batter, hand), normalized within batter.
    Prior, not eval-season: the weight must be information a forecaster already has.
    """
    prior = claim1_eval.prior_exposure(pa_df, eval_season)
    total = prior.groupby("batter")["prior_pa"].transform("sum")
    # a hitter with no prior record against either hand gets an even split rather than a
    # division by zero; the fallback is stated because it changes what is being averaged
    prior["weight"] = np.where(total > 0, prior["prior_pa"] / total.replace(0, np.nan), 0.5)
    prior["weight"] = prior["weight"].fillna(0.5)
    return prior[["batter", "p_throws", "weight"]]


def pool_predictions(predictions, weights, eval_season):
    """One pred_woba per batter, the prior-exposure-weighted mean of the two side forecasts."""
    merged = predictions.merge(weights, on=["batter", "p_throws"], how="left")
    merged["weight"] = merged["weight"].fillna(0.5)
    merged["weighted"] = merged["pred_woba"] * merged["weight"]
    grouped = merged.groupby("batter", as_index=False).agg(
        weighted=("weighted", "sum"), total=("weight", "sum"))
    grouped["pred_woba"] = grouped["weighted"] / grouped["total"]
    grouped["season"] = eval_season
    grouped["p_throws"] = POOLED_HAND
    return grouped[claim1_eval.KEY + ["pred_woba"]]


def main():
    parser = argparse.ArgumentParser(
        description="Phase F.5 -- claim-1 metric with handedness pooled away.")
    parser.add_argument("--arm", default="rebuild_baseline")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--eval-season", type=int, default=2024)
    parser.add_argument("--final-run", action="store_true")
    parser.add_argument("--data-dir", default=provenance.CANONICAL_DATA_DIR)
    parser.add_argument("--eval-targets", default="data/processed/eval_targets_pa.parquet")
    parser.add_argument("--model-predictions",
                        default="results/model_v1/model_v1_predictions_rebuild_baseline.csv")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    claim1_eval.assert_not_test_season(args.eval_season, final_run=args.final_run)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pa_df = pd.read_parquet(args.eval_targets)
    weights = pooling_weights(pa_df, args.eval_season)

    model_predictions = pd.read_csv(args.model_predictions)
    model_predictions = model_predictions[model_predictions["season"] == args.eval_season]
    rungs = {
        "model_v1": model_predictions,
        "trailing_raw": trailing_predict(pa_df, args.eval_season, variant="raw"),
        "trailing_bucketed": trailing_predict(pa_df, args.eval_season, variant="bucketed"),
        "no_info_league_average": trailing_predict(pa_df, args.eval_season, variant="bucketed",
                                             buckets=NO_INFO_BUCKETS),
    }

    pooled_pa = pa_df.copy()
    pooled_pa["p_throws"] = POOLED_HAND

    rows, reference_rmse = [], {}
    for name, predictions in rungs.items():
        pooled = pool_predictions(predictions, weights, args.eval_season)
        scored, coverage = claim1_eval.evaluate(pooled_pa, pooled, args.eval_season)
        scored.insert(0, "model", name)
        rows.append(scored)
        if name == "no_info_league_average":
            reference_rmse = dict(zip(scored["stratum"], scored["pa_weighted_rmse"]))

    table = pd.concat(rows, ignore_index=True)
    table["stratum_basis"] = "pooled_prior_pa"
    table["skill_vs_no_info"] = [
        claim1_eval.skill_score(row["pa_weighted_rmse"], reference_rmse.get(row["stratum"]))
        for _, row in table.iterrows()]
    table.to_csv(out_dir / "pooled_scores.csv", index=False)

    summary = {
        "provenance": provenance.stamp(args.data_dir, arm=args.arm, seeds=args.seeds,
                                       eval_season=args.eval_season),
        "pooled_hand_label": POOLED_HAND,
        "pooling_weight": "prior side-specific denominator PA, normalized within batter",
        "rungs_pooled": sorted(rungs),
        "rungs_omitted": ["eb_bivariate", "eb_book_rho_reference", "gbm_outcome",
                          "gbm_full"],
        "omission_reason": ("C.2 needs its fitted prior parameters and C.3 needs a GBM "
                            "refit; both gate opponents are therefore absent and this "
                            "table is not a gate"),
        "stratum_caveat": ("STRATUM_BOUNDARIES was calibrated on side-specific prior PA; "
                           "pooled exposure is roughly double, so these strata are not "
                           "comparable to the side-specific strata elsewhere"),
        "coverage": coverage,
    }
    (out_dir / "pooled_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    print("-- claim-1 metric, handedness pooled --")
    print(table[["model", "stratum", "n_hitters", "pa_weighted_rmse", "noise_floor",
                 "rank_corr_weighted", "skill_vs_no_info"]].round(5).to_string(index=False))
    print(f"\nwrote {out_dir / 'pooled_scores.csv'}")


if __name__ == "__main__":
    main()
