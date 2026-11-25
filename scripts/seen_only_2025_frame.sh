#!/bin/zsh
# Re-scores the 2025 platoon comparison on seen hitters only (decision log 2026-11-17).
# READS THE SEALED 2025 SEASON (--final-run). Scoring only: saved predictions of
# clean_dim64_final, no training. Overwrites the eval outputs in
# results/model_evaluation_final_clean and results/case_studies_2025; `git diff results/`
# shows what moved.
set -eu
cd ~/hitter-embedding
export OMP_NUM_THREADS=1 PYTHONPATH=.
P=.venv/bin/python

$P -m src.analysis.model_evaluation_eval --arm clean_dim64_final \
    --data-dir data/processed/phase_d5_clean_final \
    --eval-season 2025 --final-run --out-dir results/model_evaluation_final_clean
$P -m src.analysis.case_studies_2025
