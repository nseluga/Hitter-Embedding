"""Draft-3 exploratory row: 2025 test restricted to hitters in the final model's vocabulary.
Unfiltered pass must reproduce Table 1 first. Run: .venv/bin/python -m src.analysis.paper_seen_only_rescore"""
import json, time
from pathlib import Path
import pandas as pd
from src.analysis import baseline_ladder_gbm as gbm
from src.analysis import baseline_ladder_report, claim1_eval as evaluation, model_v1_ablation_report, model_evaluation_eval

# Scores the 2025 test season (final-run, trained 2015-2024 model) on all hitters vs. only hitters the model has an embedding for.
t0 = time.time()
SEASON, LABEL = 2025, "clean_dim64_final"
evaluation.assert_not_test_season(SEASON, final_run=True)
# Rebuild the baseline-ladder predictions for 2025 so the model can be compared against them on identical rows.
pa_df = baseline_ladder_report.load_targets("data/processed/eval_targets_pa.parquet")
params, _, _ = baseline_ladder_report.fit_and_describe(pa_df, SEASON, 0)
proc = gbm.season_process(pd.read_parquet("data/processed/pitch_events_labeled.parquet",
                                          columns=baseline_ladder_report.PITCH_COLUMNS))
preds, _ = baseline_ladder_report.build_predictions(pa_df, SEASON, params, proc, seed=0)
# Add the final model's cached point predictions next to the baseline predictions.
preds[LABEL] = model_v1_ablation_report.load_predictions(
    "results/model_v1/model_v1_predictions_clean_dim64_final.csv")
# Vocabulary = batter ids that have a trained embedding row; the "seen_only" rows keep only these hitters.
vocab = {int(b) for b in json.load(open("data/processed/phase_d5_clean_final/manifest.json"))["vocabulary"]}
out, calib = [], []
# For each minimum-PA floor, score once on all hitters and once on seen-only hitters (batter in vocab).
for floor in (10, 50):  # the two floors the strata figure draws
    # One eval frame per predictor at this PA floor.
    frames = {k: evaluation.build_eval_frame(pa_df, p, SEASON, min_eval_pa=floor, seen_only=False)[0] for k, p in preds.items()}
    # Same frames restricted to hitters in the model vocabulary.
    seen = {k: f[f["batter"].isin(vocab)] for k, f in frames.items()}
    for tag, fr in (("all", frames), ("seen_only", seen)):
        c = model_v1_ablation_report.compare(fr, name=LABEL, seed=0)
        c.insert(0, "population", tag)
        c.insert(1, "min_eval_pa", floor)
        out.append(c)
        if floor == 10:
            # Calibration slopes by stratum, only at floor 10 (the one used for the Table 1 cross-check below).
            k = model_evaluation_eval.calibration(fr[LABEL])[0]
            k.insert(0, "population", tag)
            calib.append(k)
table = pd.concat(out, ignore_index=True)
calib = pd.concat(calib, ignore_index=True)
# "all" at floor 10 must match Table 1 and calibration.csv, or the seen-only rows mean nothing
ref = pd.read_csv("results/model_evaluation_final_clean/calibration.csv").set_index("stratum").slope
got = calib[calib.population == "all"].set_index("stratum").slope
assert (got - ref).abs().max() < 1e-9, (got, ref)
Path("results/paper_draft3").mkdir(exist_ok=True)
table.to_csv("results/paper_draft3/seen_only_rescore.csv", index=False)
calib.to_csv("results/paper_draft3/seen_only_calibration.csv", index=False)
print(table.to_string(index=False, float_format="%.4f"))
print(calib.to_string(index=False, float_format="%.4f"))
print(f"\nelapsed {time.time() - t0:.0f}s")
