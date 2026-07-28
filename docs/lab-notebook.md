# Lab Notebook — Hitter Embedding

One entry per working session. Format fixed by
`~/os/knowledge/frameworks/research-standards.md` §5 — `/research-partner`
appends to it every session.

Write entries for a stranger reading them: honest about
mistakes and negative results, but in a semi-formal register,
not raw stream-of-consciousness.

---

## <YYYY-MM-DD> — <session focus>
- **Did:** what was built/run, and where it lives
- **Why:** the reasoning, at explain-it-in-an-interview depth
- **Found:** results, and mistakes worth keeping, from this session
- **Learned:** new concepts introduced this session (these mark the concept as "explained" — see teaching rule)
- **Next:** the concrete next step

---

## 2026-07-15 — Phase A data foundation: pull, profile, clean
- **Did:** Pulled 2015–2025 Statcast to a versioned parquet snapshot of 7.80M pitches (`src/data/pull_statcast.py`). Verified batted-ball spin is unavailable, so contact-quality is EV, LA, and spray. Built and ran the profiling notebook. Wrote and tested `src/data/clean.py` (9 tests), producing the 7.35M-pitch modeling table.
- **Why:** The snapshot is versioned because Statcast revises retroactively. Profiling before coding caught a DH-era confound in position-player detection — batting-PA alone misfires since NL pitchers batted before 2022, while batters-faced is era-robust. The core-vs-optional missingness split avoids imputing context the conditional query depends on. The two-table principle keeps model filters from biasing evaluation targets.
- **Learned:** batters-faced as an era-robust role discriminator; spin-vs-movement redundancy via active Magnus spin; the two-table modeling/target separation.
- **Next:** Freeze the walk-forward split config, then build the label and feature-derivation module.

---

## 2026-07-17 — Phase A completion: labels and frozen splits
- **Did:** Built `src/data/labels.py` (swing/contact/quality labels + spray angle, 16 tests), producing `pitch_events_labeled.parquet` with a reconciliation report. Froze the walk-forward split in `src/config/split_config.json` with a validating loader (`src/config/splits.py`, 8 tests). Both verified on the real 7.35M table. Phase A steps 3–4 complete.
- **Why:** Labels follow the §1.2 factorization (swing → contact | swing → quality | contact); contact-quality is in-play only because fouls carry EV but no batted-ball outcome, and masked columns stop foul EV from poisoning the quality head. Spray uses a three-source-corroborated formula with a real-data guard since MLB doesn't publish the coordinate origin. The split is contiguous walk-forward, frozen before any comparison, because random CV would leak a hitter's future into his own ID embedding.
- **Learned:** the process-head factorization and its nesting invariants; in-play-only quality masking; the spray-angle derivation and pull-mirroring; walk-forward-as-forecasting and why random splits contaminate the small-sample claim.
- **Next:** Run `/research-review` on Phase A, then start Phase B (feature-value stage: stabilization, GBM screening, bat-tracking placement, outcome-dimension ablations).

---

## 2026-07-20 — Phase B step 1: eval targets and stabilization
- **Did:** Started Phase B (bat-tracking excluded from v1, Nate's call). Cleaned tracking-artifact pitch_type codes (UN/AB/PO, 36 rows) at source in `clean.py` and regenerated both Phase A parquets. Built the context vectorizer (`src/features/context_features.py`, 48-dim, train-only fit). Built the wOBA eval-target aggregation (`src/data/eval_targets.py` + `src/config/woba_weights.json`) from the complete raw source. Built the split-half stabilization estimator (`src/analysis/stabilization.py`) and ran the process-vs-outcome panel. 64 tests pass.
- **Why:** Eval targets come from the complete source (two-table principle) so side-specific wOBA is unbiased ground truth; the FanGraphs wOBA weights were sourced, formula-verified, and validated by reproducing published league wOBA to ±0.0005 rather than recalled from memory. Stabilization (split-half + Spearman-Brown) quantifies signal-per-PA — the small-sample currency — and B.1 tests the process-beats-outcome premise before building anything.
- **Found:** Process stabilizes an order of magnitude faster than outcome: swing ~97 pitches, whiff ~34 swings, EV ~34 BBIP, LA ~21 BBIP, spray ~77 BBIP vs side-specific wOBA ~407 PA. Spray is the slowest process signal (location-driven noise); its near-plate artifact (clipping deferred) may depress it further. The exposure asymmetry is the bite: at 800 PA only 143 hitters qualify vs LHP against 476 vs RHP.
- **Learned:** feature-selection taxonomy (filter/wrapper/embedded plus permutation-importance and SHAP); split-half reliability + Spearman-Brown + the stabilization point; the wOBA linear-weights formula and complete-source eval-target build; train-only fit as the standardization leakage boundary; the closed-form synthetic gate for verifying a reliability estimator.
- **Next:** Scaffold `notebooks/02_feature_value.ipynb` (B.1 writeup), then B.2 GBM screening. Phase A `/research-review` is still deferred; run it before any Phase B result is promoted.

---

## 2026-07-21 — Phase B.1 methodology hardening and results
- **Did:** Hardened the stabilization methodology and generated the B.1 artifacts. Added a variance-components (one-way random-effects) estimator with bootstrap CIs and a sequential-split mode to `stabilization.py` (6 new tests, 70 pass); added `src/analysis/b1_report.py`, writing the panel, common-PA-axis ranking, spray-clipping check, and wOBA survivorship decomposition to `results/phase_b/` (5 CSVs + 3 figures). Logged three decisions. Decided B.1 doesn't warrant its own notebook — the logic lives in the tested module and results/ + this entry are the record; a thin reporting notebook is deferred to paper time.
- **Why:** Two weaknesses in the 07-20 numbers: (1) the single r=0.5 threshold was unsourced, so we now report both r=0.5 and the literature's r=0.7 "reliable" convention; (2) the pooled-process-vs-side-specific-outcome comparison was asymmetric, and split-half at large n only uses hitters who reach that n, biasing the outcome number. The variance-components estimator decomposes signal and noise over all hitters, giving an analytic reliability(n), CIs, and the shrinkage constant; the sequential split gives the across-circumstance number the projection task actually faces.
- **Found:** Survivorship bias was real and material — split-half put wOBA vs LHP at ~435 PA, variance-components on all 2142 hitters put it at ~190. The two estimators agree on a fixed population; restricting to durable regulars halves the between-hitter signal variance (0.00137 → 0.00073) and doubles n*. On a common PA axis the honest process-beats-outcome gap is several-fold, not an order of magnitude: whiff ~28 PA-equiv (~7x), swing ~31, LA ~62, EV ~63 (~3x), spray ~122 (~1.6x) vs wOBA ~190–198. Matched side-specific process stays fast (whiff vs LHP 45, vs RHP 50, ~pooled 51). Spray-clipping barely helps (n* 82 → 73, VC) — spray remains the slowest process signal regardless.
- **Learned:** variance-components / one-way-random-effects reliability and its equivalence to Cronbach's alpha and KR-21; survivorship bias in split-half-at-each-n and how to detect it; the r=0.5 vs r=0.7 threshold distinction and the shrinkage reading of n*; random vs sequential (across-circumstance) split-half.
- **Next:** Run the deferred Phase A `/research-review` before promoting B.1, then start B.2 GBM feature screening (XGBoost + SHAP on the 48 context features, through the frozen split, tuned on val never test). The spray label decision (drop `|spray|>90` vs keep) now has evidence — the gain is small and would require a `labels.py` change plus a parquet regen, so it's Nate's call, not yet made.

---

## 2026-07-27 — Phase C opens: the claim-1 metric, the C.1 baselines, and a contaminated hitter population
- **Did:** Built the claim-1 evaluation harness (PA-weighted RMSE plus rank correlation, stratified by exposure, with noise-floor deconvolution) and the C.1 trailing-average baselines in two variants, raw and bucketed. Found and fixed a contaminated hitter population — pitchers taking their own at-bats were leaking into every hitter-talent quantity — and excluded them project-wide. Started C.2 but reverted it (see Next). Three decision-log entries.
- **Why:** The harness needed to exist before any baseline could be scored. C.1's two variants decompose the value of shrinkage from the value of doing it properly. The pitcher contamination was found via a variance diagnostic, not the test gates.
- **Found:** Trailing averages underperform ignoring the hitter entirely below high exposure, turning positive only for veterans. Raw RMSE is mostly target noise; the deconvolution cross-checks cleanly against B.1's independent estimate. Removing pitcher-batters materially shifted stabilization points and widened the process-vs-outcome gap. The reverted C.2 attempt also used the wrong within-PA variance assumption — caught by questioning, not the gates.
- **Learned:** noise-floor deconvolution and why RMSE against a noisy target is floor-dominated; that an unbiased estimator can still be unusable in small samples; the Diebold-Mariano paired-loss idea for model comparison (deferred).
- **Next:** Restart C.2 with cell-specific variance and gates covering real sample sizes. Deferred Phase A/B `/research-review` still gates promoting B.1, B.2, and now C.1.

---

## 2026-07-27 — C.2 rebuilt: the incumbent now beats ignoring the hitter
- **Did:** Built the bivariate empirical-Bayes C.2 baseline and a shared `c_report.py` so every Phase C number comes from one seeded command.
- **Why:** Shrinking the two platoon sides jointly rather than the split directly is the same model in rotated coordinates, but identifiable where the split is not — PA vs LHP/RHP are disjoint, so the cross-side covariance needs no noise subtraction.
- **Found:** C.2 is the first Phase C baseline to beat ignoring the hitter in the low-exposure stratum, and rank correlation improves substantially. Platoon skill exists but its magnitude isn't tightly resolved; switch hitters remain effectively unidentified. The Book's published constants look larger than this data supports, though not conclusively. Every baseline over-predicts low-exposure hitters, mostly from lineup management rather than league drift. My initial design rationale (zero-weak-side borrowing) turned out to be a minor benefit — the real win is continuous shrinkage replacing C.1's step function. Two of my own estimator bugs (an unbounded-correlation failure, a coupled-seed gate) were caught before reaching a result.
- **Learned:** bivariate empirical Bayes and the BLUP form of the posterior mean; that a parameterization can preserve information yet ruin identifiability; the paired bootstrap for comparing two models against a noisy shared target.
- **Next:** Build C.3, then run the Phase C review before promoting any result. Exposure-conditional prior mean flagged but not built (scope call).

---

## 2026-07-28 — C.3 closes Phase C, and most of its win turns out to be a level correction
- **Did:** Built C.3, the last Phase C baseline — XGBoost over the hitter's own prior-window context rates, in two feature sets (`outcome`, `full`). Added a label-shuffle null, a per-stratum bias table, and an oracle level-correction diagnostic to the shared report. Two decision-log entries.
- **Why:** The plan named the model but not its row unit; the claim-1 metric settled it. The two feature sets separate "does a learned model beat empirical Bayes on the same inputs" from "does process signal actually project."
- **Found:** C.3-full beats C.2 on RMSE in every stratum but loses to it on low-exposure rank correlation — the two frozen metrics disagree for the first time. Most of C.3's RMSE margin turns out to be a level correction (it has exposure as a feature, C.2 structurally can't) rather than real ordering skill; on identical inputs a GBM doesn't clearly beat C.2. Two of my own bugs — a broken label-shuffle null and a switch-hitter labeling error — were caught before reaching a claim.
- **Learned:** gradient boosting as a projection baseline; what a valid label-shuffle null requires; oracle recentering as a way to separate a model's level error from its ordering skill.
- **Next:** Run the Phases A–C review, then consolidate results into reporting notebooks. Phase D's gate is now two-dimensional: RMSE bar from C.3, low-exposure rank bar from C.2.
