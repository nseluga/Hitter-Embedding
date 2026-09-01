# Results

Model outputs, checkpoints, evaluation artifacts, and findings.

## Structure (TBD)

- `checkpoints/` — Trained model weights (v1 ensemble seeds, v2 upgrades)
- `feature_screening/` — Feature-value analysis results (reliability plots, SHAP, bat-tracking ablations)
- `baseline_ladder/` — Baseline incumbent model performance (bucketed, empirical Bayes, XGBoost)
- `model_v1/` — v1 model training logs (loss curves, validation metrics)
- `model_evaluation/` — Evaluation results
  - `probe_checkpoint/` — Linear probe recovery signal
  - `claim_1/` — Hold-out season performance (RMSE, rank correlation by exposure stratum)
  - `calibration/` — Ensemble uncertainty calibration (reliability diagrams)
  - `composition_validation/` — Markov-composed run values vs. actual
  - `deployment_bias/` — Observed matchup patterns, natural experiment analysis
  - `dual_sampler/` — Hierarchical vs. pooled comparison
- `process_calibration/` — v2 upgrade results (if gates fire)
- `paper/` — Findings for SSAC27 paper

## Reporting Standards

- All results stratified by prior exposure (low-exposure strata are the headline)
- Negative/null results kept and reported; no cherry-picking
- Ablations documented with their claim-1 metric impact
- Dual comparisons side-by-side when applicable
