#!/bin/zsh
# Regenerates the 2024 descriptive exhibits of clean_clean_dim64 on SEEN hitters only
# (decision log 2026-11-17). SCORING ONLY: no training, no 2025. The per-seed 2024
# predictions and the platoon gradient are reused from night 5/6; only the population
# changes, through the filter in claim1_eval.build_eval_frame and
# model_evaluation_eval.platoon_frame. Overwrites the tracked night-5/6 outputs in place;
# `git diff results/` shows what moved.
set -eu
cd ~/hitter-embedding
export OMP_NUM_THREADS=4 PYTHONPATH=.
OUT=results/v_chain_clean
MC=results/measurement_ceiling_clean
BUILD=data/processed/phase_d5_clean
ARM=clean_clean_dim64
PY=.venv/bin/python
PRED=results/model_v1/model_v1_predictions_$ARM.csv
FRAME=$OUT/model_evaluation/platoon_frame.csv

set -x
$PY -m src.analysis.model_evaluation_eval --arm $ARM --data-dir $BUILD --predictions $PRED \
  --out-dir $OUT/model_evaluation
$PY -m src.analysis.platoon_sign_agreement --frame $FRAME --out-dir $OUT/paper_figures
$PY -m src.analysis.model_evaluation_platoon_ceiling --frame $FRAME --eval-season 2024 \
  --out-dir $OUT/model_evaluation
$PY -m src.analysis.process_calibration_pooled --arm $ARM --seeds 0 1 2 3 4 --eval-season 2024 \
  --data-dir $BUILD --model-predictions $PRED --out-dir $OUT/process_calibration
$PY -m src.analysis.measurement_ceiling_report --eval-season 2024 --platoon-frame $FRAME \
  --model-predictions $PRED \
  --seed-predictions "results/model_v1/model_v1_predictions_${ARM}_s{seed}.csv" \
  --seeds 0 1 2 3 4 --manifest $BUILD/manifest.json \
  --pooled-scores $OUT/process_calibration/pooled_scores.csv \
  --ceiling-json $OUT/model_evaluation/ceiling.json \
  --gbm-full-cache results/measurement_ceiling/differential_gbm_full_predictions.csv \
  --out-dir $MC
$PY -m src.analysis.model_visualization_platoon --data-dir $BUILD --arm $ARM \
  --platoon-frame $FRAME --hitter-stats $OUT/hitter_stats.csv --n-dims 64 \
  analyse --raw $OUT/platoon/platoon_gradient_raw.csv --out-dir $OUT/platoon
$PY -m src.analysis.model_visualization_disagreement --platoon-frame $FRAME \
  --names data/processed/hitter_names_clean.csv --out-dir $OUT
