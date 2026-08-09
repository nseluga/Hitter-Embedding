"""
Phase D.5 — put the model's wOBA predictions on the Phase C ladder and score them.

This is the first point in the project where a Phase D number is comparable to anything.
Everything before it — 39 training runs, an ablation table, a promote-only screen — is
measured in held-out log loss per scored row, which says which model predicts PITCHES
better and nothing at all about whether it projects HITTERS better than empirical Bayes.

Two rules this module exists to enforce, both of which fail silently otherwise:

1. THE LADDER IS RE-SCORED, NOT REMEMBERED. Every Phase C rung is recomputed on the same
   eval frame in the same process, so Phase D is never compared against numbers produced
   under a different filter, weighting, or metric vintage. `c_report` owns those baselines
   and is imported rather than reimplemented.

2. THE PAIRED COMPARISON IS THE CLAIM, NOT THE TWO ABSOLUTE NUMBERS. Both models are
   scored against the same noisy answer key, so only the paired difference with batter
   clustering can resolve them (2026-07-29). `_paired_setup` additionally refuses to run
   at all unless both models scored exactly the same groups, which is what makes the
   coverage contract in the D.5 spec load-bearing rather than tidy.

The gate the numbers are read against (2026-07-30, 2026-08-01): RMSE against C.3-full,
ordering against BOTH C.2 and C.3-full on the denominator-weighted rank correlation, each
by a paired bootstrap whose 95% interval excludes zero. A low-stratum ordering null is the
expected outcome on this frame and is not evidence against Phase D.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.analysis import c_report, claim1_eval as evaluation, c3_gbm as c3

OUT_DIR = Path("results/phase_d")
DEFAULT_PREDICTIONS = OUT_DIR / "d5_predictions_d8_baseline.csv"
MODEL_NAME = "phase_d_v1"
# the two rungs the pre-registered gate names; everything else on the ladder is context
GATE_OPPONENTS = ("c3_gbm_full", "c2_bivariate")


def load_predictions(path):
    """The D.5 prediction table, with the claim-1 contract checked before it is scored."""
    frame = pd.read_csv(path)
    missing = sorted(set(evaluation.KEY + ["pred_woba"]) - set(frame.columns))
    assert not missing, f"D.5 predictions are missing {missing}"
    assert not frame.duplicated(evaluation.KEY).any(), "duplicate (batter, season, hand)"
    return frame[evaluation.KEY + ["pred_woba"]]


def compare(frames, name=MODEL_NAME, seed=0):
    """
    Paired RMSE and paired weighted-rank differences against each gate opponent.
    Sign conventions differ between the two and are recorded per row rather than assumed:
    NEGATIVE rmse_difference favours `name`, POSITIVE rank_difference favours `name`.
    """
    tables = []
    for opponent in GATE_OPPONENTS:
        # both helpers already return one row per stratum and do their own slicing, so
        # this must not slice as well — passing a pre-sliced frame would resample inside
        # a stratum and silently narrow every interval
        rmse = evaluation.paired_rmse_difference(frames[name], frames[opponent], seed=seed)
        rank = evaluation.paired_rank_difference(frames[name], frames[opponent], seed=seed)
        merged = rmse.merge(rank, on=["stratum", "n_hitters", "n_batters"],
                            suffixes=("_rmse", "_rank"))
        merged.insert(0, "opponent", opponent)
        # NEGATIVE rmse_difference favours Phase D; POSITIVE rank_difference does. The
        # gate is the INTERVAL excluding zero, not the point estimate's sign.
        merged["rmse_favours_phase_d"] = merged["ci_high_rmse"] < 0
        merged["rank_favours_phase_d"] = merged["ci_low_rank"] > 0
        tables.append(merged)
    return pd.concat(tables, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Score Phase D.5 against the Phase C ladder.")
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--label", default=MODEL_NAME)
    parser.add_argument("--eval-targets", default="data/processed/eval_targets_pa.parquet")
    parser.add_argument("--pitch-events", default="data/processed/pitch_events_labeled.parquet")
    parser.add_argument("--eval-season", type=int, default=2024)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--final-run", action="store_true",
                        help="the ONLY way the frozen test season is ever scored")
    args = parser.parse_args()

    evaluation.assert_not_test_season(args.eval_season, final_run=args.final_run)
    pa_df = c_report.load_targets(args.eval_targets)
    params, _, _ = c_report.fit_and_describe(pa_df, args.eval_season, args.seed)
    process_seasons = c3.season_process(
        pd.read_parquet(args.pitch_events, columns=c_report.PITCH_COLUMNS))

    predictions, _ = c_report.build_predictions(pa_df, args.eval_season, params,
                                                process_seasons, seed=args.seed)
    predictions[args.label] = load_predictions(args.predictions)
    frames, scored, _ = c_report.score_all(pa_df, predictions, args.eval_season)

    comparisons = compare(frames, name=args.label, seed=args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_dir / f"d5_claim1_scores_{args.label}.csv", index=False)
    comparisons.to_csv(out_dir / f"d5_claim1_paired_{args.label}.csv", index=False)

    print("claim-1 scores (all models, one eval frame)")
    print(scored.to_string(index=False, float_format="%.4f"))
    print(f"\npaired comparisons for {args.label}")
    print("  rmse_difference NEGATIVE favours Phase D; rank_difference POSITIVE favours it")
    print(comparisons.to_string(index=False, float_format="%.4f"))

    gate = comparisons[comparisons["stratum"] == "all"]
    rmse_pass = bool(gate[gate["opponent"] == "c3_gbm_full"]["rmse_favours_phase_d"].all())
    rank_pass = bool(gate["rank_favours_phase_d"].all())
    print(f"\nRMSE gate vs C.3-full: {'PASS' if rmse_pass else 'not met'}")
    print(f"ordering gate vs BOTH C.2 and C.3-full: {'PASS' if rank_pass else 'not met'}")
    (out_dir / f"d5_claim1_verdict_{args.label}.json").write_text(json.dumps(
        {"rmse_gate_vs_c3_full": rmse_pass, "ordering_gate_vs_both": rank_pass,
         "eval_season": args.eval_season, "label": args.label}, indent=2))


if __name__ == "__main__":
    main()
