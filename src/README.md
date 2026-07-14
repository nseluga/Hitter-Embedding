# Source Code

Model implementation, training pipeline, and evaluation code.

## Structure (TBD)

Organize by phase as the project develops:

- `phase_a/` — Data foundation (Statcast loading, feature engineering, split config)
- `phase_b/` — Feature-value analysis (reliability, GBM screening, bat-tracking ablations)
- `phase_c/` — Baseline models (bucketed averages, empirical Bayes, XGBoost)
- `phase_d/` — v1 model (conditional-query architecture, training, ensembles)
- `phase_e/` — Evaluation (probe checkpoint, claim-1 metrics, calibration)
- `phase_f/` — v2 upgrades (history encoder, pitcher-ID residual)
- `phase_g/` — Query library (Monte Carlo, Markov composition)

- `config/` — Training config, split definitions, frozen decision paths
- `utils/` — Data loading, feature standardization, run-value mapping, unit tests

## Key Principles

- **Config-driven runs:** all decisions frozen in config before model comparison (no hardcoding)
- **Unit tests for data pipeline:** leakage checks as tests, label integrity verification
- **Seeded reproducibility:** all runs reproducible from config + seed
- **No outcome-luck features:** all inputs checked against non-negotiable no-luck rule
