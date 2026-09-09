# Leakage / Overfitting Provenance Audit

Read-only audit. Traced every number below to the code path and config that produced
it; nothing here retrained a model or touched `src/model/`. Verified against:
`src/config/split_config.json`, `src/config/split_config_final_run.json`,
`scripts/overnight.sh`, `src/analysis/claim1_eval.py`, and the per-module source
cited in each row.

## Two trained arms

| Arm | Config | Train seasons | Val season | Test season | Guard |
|---|---|---|---|---|---|
| `embedding_sgd_sgd_lr1` (pre-registration) | `split_config.json` | 2015–2023 | 2024 | 2025 (never spent by this arm) | `claim1_eval.assert_not_test_season` refuses `eval_season=2025` unless `--final-run` |
| `embedding_sgd_sgd_lr1_final` (refit) | `split_config_final_run.json` | 2015–2024 | none | 2025 | requires `--final-run`; `assert_not_test_season` is a no-op with the flag set |

Default `--eval-season` in every analysis module is **2024** (`model_evaluation_eval.py:535`,
`query.py:891`). Anything run without `--eval-season 2025 --final-run` is scoring the
pre-reg arm against 2024, which that arm never trained on.

## Headline-number table

| # | Claim / number | Source file | Arm | Train seasons | Target season(s) | Verdict | Note |
|---|---|---|---|---|---|---|---|
| 1 | Claim-1 rank/RMSE gates vs `gbm_full` (min-PA sweep + 2025 strata figure) | `results/model_evaluation_final/min_pa_sweep_b_min_pa_sweep.csv`, `model_v1_claim1_verdict_min_pa_sweep_embedding_sgd_sgd_lr1_final.json`, fig via `src/analysis/paper_figures.py:fig_claim1_2025_strata` (`MIN_PA_SWEEP_CSV = results/model_evaluation_final/...`) | `embedding_sgd_sgd_lr1_final` | 2015–2024 | 2025 | **held-out** | Verdict json itself shows the gates mostly fail (`rmse_gate_vs_gbm_full: false` in low/high, `true` only in medium) — this is a real, unfavorable, held-out result, not a cherry-picked pass. |
| 2 | Calibration 2025 (slope/intercept per stratum, fig) | `results/model_evaluation_final/calibration.csv`, `calibration_reliability.csv`; `src/analysis/paper_figures.py:fig_calibration_2025` (`CALIBRATION_CSV`/`CALIBRATION_RELIABILITY_CSV` both point at `model_evaluation_final/`) | `embedding_sgd_sgd_lr1_final` | 2015–2024 | 2025 | **held-out** | `model_evaluation.py --arm $FINAL_ARM ... --eval-season 2025 --final-run` (overnight.sh stage 6) is the only caller of `model_evaluation_eval` that writes to this dir. |
| 3 | "The 2025 predictions" (`model_v1_predictions_embedding_sgd_sgd_lr1_final.csv`) | `results/model_v1/model_v1_predictions_embedding_sgd_sgd_lr1_final.csv` | `embedding_sgd_sgd_lr1_final` | 2015–2024 | 2025 | **held-out** | Produced by stage 5 of `overnight.sh`: `src.model.query --arm $FINAL_ARM --data-dir $FINAL_DATA --eval-season 2025 --final-run`. |
| 4 | Level query, pooled `r_partial` — in-sample 0.3743, high 0.8241 | `results/model_visualization/level_query.json` | `embedding_sgd_sgd_lr1` | 2015–2023 | correlates against `woba_level`, computed on the **same 2015–2023 pitches the embedding trained on** | **in-sample** | Module's own docstring (`src/analysis/level_query_heldout.py`, top) calls this "an in-sample comparison that cannot be defended in review." |
| 4b | Level query, held-out — pooled `r_partial` 0.3815, high 0.5164 | `results/paper_figures/level_query_heldout.json` | `embedding_sgd_sgd_lr1` | 2015–2023 | 2024 (`HELDOUT_SEASON = 2024` in `src/analysis/level_query_heldout.py`) | **held-out** | Same per-hitter `level_query` values, re-correlated against genuine 2024 wOBA. Note the high-stratum correlation *drops* from 0.824 → 0.516 once scored on truly unseen data — the in-sample number in row 4 materially overstated it. |
| 5 | Platoon sign agreement (`delta_obs` in `platoon_frame.csv`) | `results/paper_figures/platoon_sign_agreement.json`, reads `results/model_evaluation/platoon_frame.csv` (`DEFAULT_FRAME_PATH` in `src/analysis/platoon_sign_agreement.py:71`) | `embedding_sgd_sgd_lr1` | 2015–2023 | `delta_obs` = `actual[actual["season"]==eval_season]` in `model_evaluation_eval.py:platoon_frame`, called with `eval_season=2024` (default) | **held-out** | `delta_obs` is the realized 2024 platoon split, not career/training-window data; Route A's league-differential baseline is separately fit on `pa_df[pa_df["season"] < eval_season]` (train-only), so no leakage there either. |
| 6 | Arm agreement (`results/paper_figures/arm_agreement.json`) | `results/paper_figures/arm_agreement.json`, `src/analysis/model_visualization_arm_agreement.py` | compares embeddings of both arms directly (`reference_arm: embedding_sgd_sgd_lr1`, `comparison_arm: embedding_sgd_sgd_lr1_final`) | n/a — structural comparison, no eval season | n/a | **n/a (not a held-out-vs-in-sample claim)** | This measures cosine similarity between two independently-trained embedding tables (median 0.967 vs a permutation-null median 0.086); it is a stability check, not a target-leakage-prone metric. No overfitting verdict applies. |
| 7a | Reliability stabilization point, wOBA vs LHP, n\* ≈ 226.04 | `results/feature_screening/stabilization_panel.csv`, produced by `src/analysis/stabilization.py --max-train-season 2023` (default) | n/a — pre-model Phase B feature-screening stat, not a model output | data restricted to `season <= 2023` (`train = lambda d: d[d["season"] <= args.max_train_season]`) | none (describes raw-stat volatility, not scored against a target season) | **in-sample by design, not a leak** | This is a descriptive statistic about how many PA it takes for the raw observed statistic itself to stabilize (KR-21 style), computed only on train seasons, before any model exists. It characterizes the target's own noise, not model skill, so "in-sample" here doesn't mean contamination — but it does mean n\* was never checked against 2024/2025 data, so its generalization to the held-out seasons is unverified. |
| 7b | Measurement ceiling, `ceiling_rank_corr` = 0.7420 | `results/measurement_ceiling/level_ceiling_level_ceiling.json` (`intersection.ceiling_rank_corr`), produced by `src/analysis/measurement_ceiling_report.py` (`EVAL_SEASON` hard-set to 2024; module docstring: "2025 is never read... the only season this module scores is 2024") | `embedding_sgd_sgd_lr1` | 2015–2023 | 2024 | **held-out** | Ceiling is estimated from 2024 between/within-hitter variance decomposition on the M.6 intersection population (n=545); 2025 is explicitly excluded from this module per its own docstring, and the report is labeled "post-selection descriptive," not a pre-registered hypothesis test. |
| 8 | Neighbour purity + silhouette (`results/embedding_structure/`) | `neighbour_purity.csv`, `embedding_structure.json`, `src/analysis/embedding_structure.py` | `embedding_sgd_sgd_lr1` embeddings, attributes from `results/model_visualization/hitter_stats.csv` (`HITTER_STATS_PATH` default) | attributes computed on **2015–2023 only** (`TRAIN_SEASONS = (2015, 2023)` in `src/analysis/model_visualization_stats.py`, whose own docstring states "Everything is computed on TRAINING seasons only ... never on 2024/2025") | none — this is a structural/clustering check on the training-window embedding, not scored against a held-out target | **in-sample by construction, not a target leak — but see pitcher-batter contamination below** | The attributes (handedness, `ev_p90` power tercile, `contact_rate`) being clustered are training-window descriptive stats of the same hitters the embedding trained on. This is appropriate for "what did the embedding learn" (a representation-quality check, not a predictive-accuracy check), so it is not leakage in the claim-1 sense. It IS however contaminated by pitcher-batters — see below — since `hitter_stats.csv` is the direct input. |

## "How would we know if this overfit?" — checks available without retraining

1. **Held-out-season scoring already exists for the headline claim.** `claim1_eval.assert_not_test_season` (`src/analysis/claim1_eval.py:655`) hard-asserts that the pre-reg arm cannot be scored on 2025 outside `--final-run`. The claim-1 gates, calibration, and predictions in the table above (rows 1–3) are scored on 2025 by a model that never trained on 2025 (`split_config_final_run.json`: train `2015–2024`, test `[2025]`). This is the strongest available check and it has already run — read `results/model_evaluation_final/*`.
2. **A pre-registered replay gate exists and is checked before the refit is allowed to spend the test season.** `scripts/overnight.sh` stage 3 reproduces the frozen-split arm from a fixed step budget (no early-stopping/validation signal) and refuses stage 4 (the refit) unless the replay reference lands within `GATE_CENTER ± GATE_TOL` (`1.02386 ± 0.00018`) of the five original seeds' mean. This is a check that the refit's training dynamics reproduce the pre-registered run rather than a re-tuned one — read `results/model_v1/replay_schedule.json` and the `03_replay.log` / `gate_passed` sentinel referenced in the script.
3. **The level-query in-sample-vs-held-out comparison (row 4 vs 4b above) is itself a direct overfitting probe** — `src/analysis/level_query_heldout.py` exists specifically to re-score the same per-hitter values against genuine 2024 data instead of the training-window target, and it shows the correlation *does* drop materially in the high stratum (0.824 → 0.516). That drop is evidence of some in-sample inflation in the original number, already caught and reported.
4. **Arm agreement (row 6) is a stability-not-leakage check**: it asks whether the pre-reg arm and the refit arm (trained on an overlapping-but-different window, different seeds implicitly) land in the same embedding geometry (median cosine 0.967 vs a null floor of ~0.085). High agreement between two independently-trained arms is evidence the learned structure isn't an artifact of one particular training run, though it does not by itself rule out both arms sharing the same systematic overfit.
5. **Not yet run, and would strengthen the case:** there is no check in this repo that re-scores the pre-reg arm's *training-season* fit (e.g., in-sample RMSE on 2015–2023) against its 2024/2025 held-out RMSE side by side in one artifact — the classic train-vs-test gap plot. The pieces exist to build one (predictions + `eval_targets_pa.parquet` cover training seasons too) but no module currently does it.
6. **The min-PA / stratum sensitivity sweep** (`min_pa_sweep_b_min_pa_sweep.csv`, `MIN_EVAL_PA_SENSITIVITY = (10, 25, 50)` in `claim1_eval.py`) checks that the claim-1 result isn't an artifact of one arbitrary PA floor — already run, both for the 2024 pre-reg arm and the 2025 final arm.

**Bottom line:** the paper's headline claim-1/calibration numbers (rows 1–3) are genuinely scored on data the reporting arm never trained on, and the replay gate is a real pre-registration-style check that already ran and is logged. The clearest evidence of *any* in-sample inflation found in this repo is row 4 (level query), and it is already flagged and superseded by row 4b in the paper's own artifacts.

## Pitcher-batter contamination

`drop_pitcher_batters` (`src/data/eval_targets.py`) is applied wherever a quantity
describes **hitter talent** (claim-1 eval frames, platoon frames, `hitter_stats.csv`'s
`woba_level`/`prior_pa`/`stratum`/`obs_platoon_diff` columns) — but it is defined by
`primarily_pitchers`, which has a loophole:

```
PITCHER_MIN_BATTERS_FACED = 50   # faced >= 50 batters that season => "real pitcher"
TWO_WAY_MIN_PA = 50              # batted >= 50 PA that season => carved out as "two-way"
```

A batter is excluded for a season only if he's a real pitcher **and** batted fewer
than 50 PA that season. Full-time NL starters before the 2022 universal DH routinely
batted 50–90 PA/season just from their own starts — enough to trip the "two-way"
carve-out meant for players like Ohtani, so **they are not excluded** in those
seasons. Verified directly:

| Pitcher | MLBAM id | Seasons with PA ≥ 50 (survives exclusion) | Seasons with PA < 50 (correctly excluded) |
|---|---|---|---|
| Bartolo Colón | 112526 | 2015 (64), 2016 (65) | 2017 (20), 2018 (4) |
| Madison Bumgarner | 518516 | 2015 (81), 2016 (97), 2019 (76), 2021 (52) | 2017 (36), 2018 (46) |
| Zack Greinke | 425844 | 2015 (77), 2016 (60), 2017 (70), 2018 (71), 2019 (56) | 2021 (2) |

**Count and distribution:** `results/model_visualization/hitter_stats.csv` has 1762
rows; **163 of them (9.2%)** are pitcher-batters that `primarily_pitchers` failed to
exclude in at least one train-window season. **All 163 land in the `low` exposure
stratum** — the stratum the thesis is graded on (per the manifest's own framing) — and
162/163 carry a non-null `woba_level` (typically 0.10–0.26, well below any genuine
low-exposure hitter's plausible range).

**Which downstream numbers include them, and which don't:**

- **Claim-1 RMSE/rank, calibration, min-PA sweep, platoon sign agreement (rows 1, 2, 5 above):** **NOT contaminated at scoring time.** These are all scored against `eval_season` 2024 or 2025, and MLB has had the universal DH since 2022, so no pitcher ever bats in the scored seasons regardless of the exclusion-rule loophole. The contamination is confined to the 2015–2023 training/description window.
- **Model training tensors / embedding vocabulary:** **contaminated.** `src/data/model_dataset.py:drop_pitcher_at_bats` uses the same `primarily_pitchers` set, so the same ~163-batter, several-hundred-PA leak lets real pitchers' batting appear in the training data as if they were low-exposure hitters. This means the trained embedding actually has learned representations for these pitcher-seasons sitting in "hitter" vocabulary slots.
- **`results/embedding_structure/` (neighbour purity, silhouette) — row 8 above:** **contaminated.** `hitter_stats.csv` is the direct attribute source; the pitch-level stats (`contact_rate`, `ev_p90`, `swing_rate`, etc., built by `per_batter_pitch_stats` with no pitcher filter at all — only `per_batter_exposure_and_platoon`'s wOBA columns apply `drop_pitcher_batters`) are populated for all 163. Pitchers are extremely uniform on power/contact, so a cluster of "low stratum, low contact, low power" hitters could be partly or wholly an artifact of ~163 real pitchers sitting in that cluster, not learned hitter-talent structure.
- **Cold-start prior baseline (`src/analysis/cold_start_prior_eval.py`, `cold_start_prior_diagnostic.py`):** **contaminated.** Both explicitly use `hitter_stats.csv`'s low-stratum mean `woba_level` (and the low-stratum mean embedding vector) as the debut-prior baseline, per `cold_start_prior_eval.py`'s own docstring. Since all 163 leaked pitcher-batters sit in the low stratum with wOBA ~0.10–0.26, this baseline's low-stratum mean is pulled down by real pitchers, not just genuinely inexperienced hitters — this could bias any comparison that credits the model for beating (or matching) this prior.

**Undetermined:** I did not re-run the prior/embedding-structure numbers with the loophole patched, so I cannot quantify how much the ~163-pitcher contamination moves `neighbour_purity`/`silhouette` or the cold-start prior's low-stratum mean — only that the mechanism exists and the affected files are identified above. That would require a code change (tightening `TWO_WAY_MIN_PA` or making the carve-out season-count-based rather than single-season-PA-based) and is out of scope for a read-only audit.
