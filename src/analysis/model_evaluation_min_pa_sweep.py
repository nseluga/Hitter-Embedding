"""
E.11b — the committed MIN_EVAL_PA sweep, applied to the Phase D/E headline.

Architecture plan §5 item 2 requires the eval-season censoring threshold to be varied and the
result committed, "so the claim is shown to hold across a 3x swing in censoring rather than
resting on the chosen frame". That sweep was built for the PHASE C headline
(`baseline_ladder_report.min_eval_pa_sensitivity`, committed at `results/baseline_ladder/baseline_ladder_min_eval_pa_sensitivity.csv`)
and hard-codes C.3-full against C.2. The headline is now E.11's gate verdict, and it had no sweep.
This module supplies it.

Why the filter is not neutral, restated because it is the whole point (claim1_eval.py:115-139):
MIN_EVAL_PA censors on eval-season playing time, which is decided AFTER the projection is made
and partly BY how the hitter performed. It bites hardest in the low stratum, which is the
stratum the thesis is graded on -- 18.3% of it at a threshold of 10, 36.6% at 25. It therefore
trims the headline stratum's worst performers and flatters every model's level. No threshold
removes the problem; varying it makes the dependence visible.

Cheap by construction: predictions do not depend on the threshold, so nothing is refit and no
model is re-run. Only the eval frame is rebuilt.

READ-ONLY with respect to the model, and read-only with respect to every prior lane: this module
imports `model_v1_ablation_report.compare` rather than reimplementing the paired comparison, so the sweep and
the headline cannot drift apart in their sign conventions or their clustering.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.analysis import baseline_ladder_gbm as gbm
from src.analysis import baseline_ladder_report, claim1_eval as evaluation, model_v1_ablation_report

OUT_DIR = Path("results/model_evaluation")


def sweep(pa_df, predictions, eval_season, label, seed=0,
          thresholds=evaluation.MIN_EVAL_PA_SENSITIVITY):
    """
    The §189 gate re-read at each threshold. Returns (per-threshold table, verdict dict).

    The gate verdict is recomputed from the intervals at each threshold rather than carried
    over, because the whole question is whether the verdict is a property of the model or of
    the frame.
    """
    rows, verdicts = [], {}
    for threshold in thresholds:
        frames = {name: evaluation.build_eval_frame(pa_df, prediction, eval_season,
                                                    min_eval_pa=threshold)[0]
                  for name, prediction in predictions.items()}
        assert label in frames, f"the scored arm {label!r} is missing from the eval frames"
        comparisons = model_v1_ablation_report.compare(frames, name=label, seed=seed)
        comparisons.insert(0, "min_eval_pa", threshold)
        rows.append(comparisons)

        per_stratum = {}
        for stratum in sorted(comparisons["stratum"].unique()):
            gate = comparisons[comparisons["stratum"] == stratum]
            against_gbm = gate[gate["opponent"] == "gbm_full"]["rmse_favours_model_v1"]
            per_stratum[stratum] = {
                "rmse_gate_vs_gbm_full": bool(against_gbm.all()) if len(against_gbm) else None,
                "ordering_gate_vs_both": bool(gate["rank_favours_model_v1"].all()),
                "n_hitters": int(gate["n_hitters"].iloc[0]),
                "n_batters": int(gate["n_batters"].iloc[0]),
                # the low-stratum ordering margin is the number the thesis turns on, so it is
                # surfaced per threshold rather than left inside the table
                "rank_difference_vs_gbm_full": float(
                    gate[gate["opponent"] == "gbm_full"]["rank_difference"].iloc[0]),
                "rank_ci_low_vs_gbm_full": float(
                    gate[gate["opponent"] == "gbm_full"]["ci_low_rank"].iloc[0]),
                "rank_ci_high_vs_gbm_full": float(
                    gate[gate["opponent"] == "gbm_full"]["ci_high_rank"].iloc[0]),
            }
        verdicts[str(threshold)] = per_stratum

    table = pd.concat(rows, ignore_index=True)
    # the sweep passes only if the DECISIVE stratum's verdict is the same at every threshold.
    # A verdict that flips is not a weaker claim, it is a claim about the filter.
    decisive = [verdicts[str(t)]["low"] for t in thresholds]
    stable = {
        "rmse_gate_stable": len({v["rmse_gate_vs_gbm_full"] for v in decisive}) == 1,
        "ordering_gate_stable": len({v["ordering_gate_vs_both"] for v in decisive}) == 1,
        "low_stratum_n_hitters": [v["n_hitters"] for v in decisive],
        "low_stratum_rank_difference": [v["rank_difference_vs_gbm_full"] for v in decisive],
    }
    return table, {"thresholds": list(thresholds), "decisive_stratum": "low",
                   "by_threshold": verdicts, "stability": stable}


def main():
    parser = argparse.ArgumentParser(description="E.11b — MIN_EVAL_PA sweep on the D/E headline.")
    parser.add_argument("--predictions",
                        default="results/model_v1/model_v1_predictions_rebuild_baseline.csv")
    parser.add_argument("--label", default="min_pa_sweep_rebuild_baseline")
    parser.add_argument("--eval-targets", default="data/processed/eval_targets_pa.parquet")
    parser.add_argument("--pitch-events", default="data/processed/pitch_events_labeled.parquet")
    parser.add_argument("--eval-season", type=int, default=2024)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--final-run", action="store_true")
    args = parser.parse_args()

    evaluation.assert_not_test_season(args.eval_season, final_run=args.final_run)
    pa_df = baseline_ladder_report.load_targets(args.eval_targets)
    params, _, _ = baseline_ladder_report.fit_and_describe(pa_df, args.eval_season, args.seed)
    process_seasons = gbm.season_process(
        pd.read_parquet(args.pitch_events, columns=baseline_ladder_report.PITCH_COLUMNS))
    predictions, _ = baseline_ladder_report.build_predictions(pa_df, args.eval_season, params,
                                                process_seasons, seed=args.seed)
    predictions[args.label] = model_v1_ablation_report.load_predictions(args.predictions)

    table, verdict = sweep(pa_df, predictions, args.eval_season, args.label, seed=args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "min_pa_sweep_b_min_pa_sweep.csv", index=False)
    (out_dir / "min_pa_sweep_b_min_pa_sweep.json").write_text(json.dumps(verdict, indent=2))
    print(table[["min_eval_pa", "opponent", "stratum", "n_hitters", "rmse_difference",
                 "ci_low_rmse", "ci_high_rmse", "rank_difference", "ci_low_rank",
                 "ci_high_rank", "rmse_favours_model_v1", "rank_favours_model_v1"]]
          .to_string(index=False, float_format="%.4f"))
    print()
    print(json.dumps(verdict["stability"], indent=2))
    print(f"\nwrote {out_dir / 'min_pa_sweep_b_min_pa_sweep.json'}")


if __name__ == "__main__":
    main()
