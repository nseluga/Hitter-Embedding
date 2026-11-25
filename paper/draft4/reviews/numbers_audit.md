# Numbers audit of paper/draft4/main.tex

Read-only audit. Nothing was edited, trained or re-run; only existing result files, docs and source comments were read.
Line numbers refer to /Users/nateseluga/hitter-embedding/paper/draft4/main.tex. "Point" = 0.001 wOBA.
Scale: about 250 individual figures were checked, grouped below into the claim rows (one row can cover a whole table or a run of figures that come from one file).

Totals: 2 MISMATCH, 2 NOT FOUND (plus 1 partial), 7 provenance or wording caveats, everything else MATCH.

## 1. MISMATCH

| Line | Claim | Source file and field | Source value | Verdict |
|---|---|---|---|---|
| 245 (footnote) | Five seeds give vectors with "a median cosine similarity of 0.83" | results/v_chain_clean/seed_stability_corrected_null.json, `real_mean` (per-seed medians 0.8316 to 0.8410; the paper's own source comment says 0.837) | 0.8366, which rounds to 0.84 | MISMATCH (rounding; paper should say 0.84, or "about 0.84"). The null, 0.12, matches (0.1229). |
| 137 | "Unmeasured balls in play, about 7%" | results/model_v1/model_v1_diagnostics_clean_clean_dim64_s0.json, `measured_share` (identical in s1) | 0.92353, so unmeasured = 7.65% | MISMATCH (minor): rounds to 8%, not 7%. "About 7.6%" or "about 8%" fits the file. |

## 2. NOT FOUND

| Line | Claim | Where I looked | Verdict |
|---|---|---|---|
| 158 (comment at 156) | n* = 226 PA with 95% interval "198 to 265" | The cited src/analysis/claim1_eval.py:95-108 gives n* = 226 (MATCH) but its comment says "Hitters-only CI [199, 264]", not [198, 265]. No results/*.json or *.csv carries a PA-scale n* interval (results/exposure_cut/exposure_cut_report.json has n_star_pa 226 and a pitch-scale CI of 96.5 to 119.0 only). docs/decision-log.md and docs/lab-notebook.md have no 198/265 pair. The only places "198 to 265" appears are paper/draft3, draft4 and reviews/round1_structure.md. | NOT FOUND, and the one cited source contradicts it by 1 PA at each end. Either the paper uses an unlogged re-bootstrap or it is a typo for 199 to 264. Needs the author. |
| 74 | "A plate appearance lasts about four pitches" | No Source comment. results/exposure_cut/exposure_cut_report.json has `pitches_per_pa_ratio` 3.893 (2025 vs-LHP frame), which supports "about four". | NOT FOUND as cited; supported by 3.893, so low risk. Add a Source comment. |
| 76 / Intro (Tango) | Right-handed hitter needs about 2,200 PA vs LHP, left-handed about 1,000 (The Book p.157) | External. docs/decision-log.md (around lines 1178-1188) records the constants 2200 (RHB) and 1000 (LHB) and the p.157 quotation. | MATCH to the decision log; the book itself was not checked. |
| 78 / Intro (Alcorn) | Nine-dimension embedding, Retrosheet 2013 through 2015 | External paper, not in results. | NOT FOUND in the repo (external citation). Not verifiable here; check against the Alcorn paper. |
| 129 | "each copy trains on a CPU in about 25 minutes" | results/model_v1/sweep_log.csv has no clean_dim64_final rows and the final diagnostics JSONs have no timing field. The selection builds clean_clean_dim64 seeds 0-4 took 1444 to 1669 s (24.1 to 27.8 min, mean 25.6). | NOT FOUND for the final model; supported by the selection model's timings. |

## 3. Provenance and wording caveats (the number matches its cited source, but the claim around it is shakier)

| Line | Issue |
|---|---|
| 129 (footnote on cold-start prior) | "lowered their RMSE by 0.0008 and left rank correlation unchanged" is -0.00077 [-0.00119, -0.00020], rank difference 0 (decision log 2026-09-04). It was measured on the OLD embedding_sgd_sgd_lr1 build (207,469 parameters) whose cold-start prior was later found to be built from 20.4% pitcher-batters (decision log 2026-09-09). It was not re-measured on clean_dim64_final, yet the paper says it was used for all results. The 2026-09-09 entry shows the corrected prior moves low-stratum RMSE by only +0.000001 (floor 10) to -0.000197 (floor 50), so the conclusion probably stands, but the 0.0008 is from the old build. |
| 129 (warmup footnote) | "mean log-loss change 0.00005" (under 0.0001) matches sweep_log.csv means; the seed-0 change alone is 0.0002. Fine as a mean, but "under 0.0001" is not true per seed. |
| Figure 2 caption (around line 190) | "changes no row's outcome" holds against GBM; for seen-only low versus EB the verdict flips at the 50 PA floor (seen_only_rescore.csv). |
| 291 (Table 3 caption) | "three on whom the model and EB disagreed most in 2024": in disagreement_top.csv Ozuna is 2nd (gap 310), Altuve 9th (258) and Judge 10th (257); Robert, Farmer, McCann, Diaz and others rank higher. The table values are right; the caption is not. |
| 322 | "Bohm's surface is nearly flat, about 30 to 40 points above average everywhere, with no cell above about 70": from surfaces_location.csv (quantity q) Bohm's medians are 39 (LHP) and 31 (RHP), the maximum is 71 and 66, but the minimum is 16 (LHP) and 1 (RHP). "Everywhere" overstates it; "no cell above about 70" is correct. |
| 322 | Figure 5 caption "average of 100 hitters": matches `n_league=100` in src/analysis/model_visualization_anchors.py (a code default, not a result file). |
| 243 | n = 1,493 in the LDA comment versus 1,564 hitters in hitter_stats.csv: the 1,493 is the embedding_structure.json LDA sample after dropping hitters without enough data, and 1,672 (final model) is a different build. The three counts are consistent with their builds but a reader could confuse them. |

## 4. MATCH (grouped by section; source and what was compared)

| Lines | Claims | Source | Verdict |
|---|---|---|---|
| 63 abstract, 92-94 | 7,245,923 pitches 2015-2025; 6,540,616 pitches through 2024 (abstract "6.5 million"); 1,672 hitters; 2,182 pitchers; 64-dim | data/processed/phase_d5_clean_final/manifest.json (`n_pitches`, `n_hitters`); season.npy (sum through 2024 = 6,540,616) | MATCH |
| 63 abstract and 163-229 | Pooled RMSE 0.0452 / 0.0453 / 0.0460; pooled rank correlation 0.47 / 0.42 / 0.40; "113 to 451 PA" medium stratum is the only stratum where the model beats both on both measures | results/model_evaluation_final_clean/model_v1_claim1_scores_min_pa_sweep_clean_dim64_final.csv; verdict JSON | MATCH |
| 167-190 Table 1 | Every row (strata low/medium/high/all, unit counts such as 384 low, RMSE and rank columns for model/GBM/EB) | same scores CSV | MATCH |
| 167-190 | Paired RMSE and rank-difference intervals; verdict gates | model_v1_claim1_paired_..._final.csv; verdict JSON | MATCH |
| 156-158 | Cuts 113 and 452 (n*/2 and 2n*), floor 10 PA, 226 | src/analysis/claim1_eval.py:97-108 (`STRATUM_BOUNDARIES` = (113, 452), `MIN_EVAL_PA` = 10) | MATCH (CI is the exception, see section 2) |
| 193-199 | Counts of batters in strata under the power analysis (377, 5,158, 246, 236, factors x1.5978 and x20.97) | verdict JSON `rank_power` | MATCH |
| 203 | Calibration slopes 0.98 medium, 0.93 high, 0.59 low, 0.81 all; about 9 points high-stratum and 7 points low-stratum bias; 87% interval coverage | results/paper_draft3/seen_only_calibration.csv; v_chain_clean/paper_figures calibration JSON | MATCH |
| 200-229 | Fractions of the measurement ceiling: 60% rank-order, 64% level, 0.2965 differential, ceilings | results/measurement_ceiling_clean/{measurement_ceiling_summary.json, differential_fraction_of_ceiling.csv, level_ceiling_level_ceiling.json, differential_exhibit.csv} | MATCH |
| 125-129 | 266,317 parameters; dim64 1.02327 vs dim32 1.02378; lr1e3 1.02584/1.02585 vs warmup 1.02564/1.02596; AdamW lr 1e-3, batch 8,192, weight decay 1e-2, trunk dropout 0.1, SGD lr 1.0; RPS screen 1.19715 vs 1.07582 | sweep_log.csv; clean_dim64_final_s0.pt state dict; src/model/v1.py, train.py; decision-log.md line 370 | MATCH |
| 107 | 6 heads, 24 quality bins | src/model/v1.py; manifest `n_quality_bins` 24 | MATCH |
| 134-140 | Chain structure, 2,000 reference pitches | src/model/query.py; model_visualization_heads.py (`N_REFERENCE_PITCHES` = 2000) | MATCH |
| 242-250 | LDA 5-fold accuracies power 0.791, contact 0.754, zone swing 0.746, chase 0.734, chance 0.33; observed wOBA rises with power and contact | results/v_chain_clean/embedding_structure/embedding_structure.json, embedding_alt_views.json | MATCH |
| 254-276 | Miss-damage figures, head partial loadings, platoon sign agreement 310 of 545 (57%) vs null 0.498 (paper: 50%), direction cosine 0.14 (0.08 to 0.20) | embedding_alt_views.json; head_partial_loadings.csv; paper_figures/platoon_sign_agreement.json; platoon/platoon_projection.json | MATCH |
| 281-285 | Neighbour lists for Pederson, Schwarber, Soto, Bohm, Trout; Trout's ten-statistic neighbours; "7 of 40" overlap (recomputed: Bohm 0, Pederson 2, Soto 1, Schwarber 3, Trout 1 = 7); 0.57 vs 0.33 purity; silhouette at most 0.054 | results/paper_draft3/neighbours_seed_mean_high.csv; embedding_structure/neighbour_purity.csv | MATCH |
| 290-315 Table 3, text | 539 / 296 / 243 units, wmae 0.0448 vs 0.0462 (difference 0.0014, CI 0.0002 to 0.0026), 51% (interval 0.5065 to 0.5918, paper 51% to 59%); Grossman, Ozuna, Judge, Altuve ranks; rank correlations 0.77 and 0.69; all table rows | results/case_studies_2025/case_studies_2025.{json,csv}; all_hitters_bootstrap.json (untracked, see section 5); v_chain_clean/disagreement_summary.json, disagreement_top.csv, disagreement.csv | MATCH |
| 320-330 | Surface description and figure | results/v_chain_clean/surfaces_location.csv (model clean_clean_dim64, seed 0) | MATCH apart from the "everywhere" wording above |
| 331-360 Table 4 and text | Bohm 0.3317 vs staff and 0.3567 vs LHP, ranks 1/1; Marsh 0.2514 / 0.2865, ranks 13/12; all 13 Phillies rows; staff of Sale (25.08 BF) plus 16 relievers (14.889 BF per game); weight dropped 0.0018; "7 to 35 points" (Marchan 0.0072 to Marsh 0.0350); three adjacent pairs swap (ranks 2/3, 10/11, 12/13); Kepler in place of Sosa in the top nine | results/paper_draft3/single_game_staff.{json,csv}, single_game_staff_weights.csv | MATCH |
| 362-369 | Future-directions numbers | results/model_evaluation_final_clean verdict JSON | MATCH |

## 5. Modified and untracked files: do any paper numbers depend on them?

git status shows:

* results/process_calibration_final_clean/{heads_summary,pooled_summary,process_summary}.json (modified). The only difference between HEAD and the working copy is `git_revision` (3d3a636... to dbca3f2...). No numeric value differs. No paper number is affected.
* results/model_v1/league_reference_rates.json (modified). The working copy was regenerated: training seasons now include 2024 (n_pa 1,526,308 to 1,707,696; bb 0.08042 to 0.08032; k 0.22338 to 0.22378; bip 0.68571 to 0.68534), `eval_season` 2024 to 2025 (eval_n_pa 181,388 to 181,724), and the per-split wOBA constants changed in the fourth decimal (e.g. LHB_vs_RHP 0.33441 to 0.33587). This is the file from which the paper's "unmeasured balls in play" and league-average values might be taken, but the paper's cited numbers (7%, `measured_share` 0.9235) come from the model diagnostics JSON, not from this file. I found no paper number that traces to it. It does describe the 2025 final build (2015 to 2024 training), so the working copy is the one consistent with the paper's final model; the committed version describes the older 2024-eval build.
* results/case_studies_2025/all_hitters_bootstrap.json (UNTRACKED). The paper's case-study interval (share of hitters where the model is closer, 51% with interval 0.5065 to 0.5918, and the wmae difference 0.0014 with CI 0.0002 to 0.0026) comes from this file. The values agree with the paper, but the file is not committed; commit it before submission or the number cannot be reproduced from the repository. The counts 539/296/243 and wmae 0.0448/0.0462 come from the tracked case_studies_2025.json.
* results/pitcher_type_query_clean/ (UNTRACKED). The paper says only the handedness query was graded, and no number in the text or tables traces to this directory (the make_figs3.py hard-thrower figure reads from it, but main.tex gives no number for it that I could link). No dependency found.
* paper/main.tex is deleted in the working tree and paper/draft4/ is untracked (the draft under audit is itself uncommitted).

## 6. Numbers without a Source comment covering them

* Line 63 (abstract): no Source comment of its own (the comment at line 58 points to paper/abstract.tex). Its numbers were checked against the body: 6.5 million (6,540,616), 1,672, 64, 6 heads, 0.0452/0.0453/0.0460, 0.47/0.42/0.40, 113 to 451, 60%, 64%, 0.79, 0.75, 0.33 all agree with the body and the sources. No abstract/body disagreement found.
* Line 74: "about four pitches" per PA (see section 2).
* Line 129: "about 25 minutes" per copy (the line 125 comment cites sweep_log.csv, which has no row for the final model).
* Line 129: batch size 8,192 and the optimizer settings are covered only by the generic source comment at line 125 (they match src/model/train.py).
* Lines 76-78 external citations (Tango p.157, Alcorn): the Tango one has a Source comment (line 76); Alcorn's nine dimensions and Retrosheet 2013 to 2015 have none.
* Line 322: the "100 hitters" caption figure and the "30 to 40 / 70 points" sentence are covered by the line 320 comment, but 100 comes from code, not a result file.
* The Table 3 caption (line 291) and Figure 2 caption wording have no source for their qualitative claims.
