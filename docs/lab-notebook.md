# Lab Notebook — Hitter Embedding

One entry per working session. Format fixed by
`~/os/knowledge/frameworks/research-standards.md` §5 — `/research-partner`
appends to it every session.

---

## <YYYY-MM-DD> — <session focus>
- **Did:** what was built/run, and where it lives
- **Why:** the reasoning, at explain-it-in-an-interview depth
- **Found:** results that changed a decision, and what they changed
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
- **Did:** Built `src/data/labels.py` (swing/contact/quality labels + spray angle, 16 tests), producing `pitch_events_labeled.parquet` with a reconciliation report. Froze the walk-forward split in `src/config/split_config.json` with a validating loader (`src/config/splits.py`, 8 tests). Both verified on the real 7.35M table.
- **Why:** Labels follow the §1.2 factorization (swing → contact | swing → quality | contact); contact-quality is in-play only because fouls carry EV but no batted-ball outcome, and masked columns stop foul EV from poisoning the quality head. Spray uses a three-source-corroborated formula with a real-data guard since MLB doesn't publish the coordinate origin. The split is contiguous walk-forward, frozen before any comparison, because random CV would leak a hitter's future into his own ID embedding.
- **Learned:** the process-head factorization and its nesting invariants; in-play-only quality masking; the spray-angle derivation and pull-mirroring; walk-forward-as-forecasting and why random splits contaminate the small-sample claim.
- **Next:** Run `/research-review` on Phase A, then start Phase B.

---

## 2026-07-20 — Phase B step 1: eval targets and stabilization
- **Did:** Started Phase B (bat-tracking excluded from v1, Nate's call). Built the context vectorizer (`src/features/context_features.py`, 48-dim, train-only fit), the wOBA eval-target aggregation (`src/data/eval_targets.py` + `src/config/woba_weights.json`) from the complete raw source, and the split-half stabilization estimator (`src/analysis/stabilization.py`). Ran the process-vs-outcome panel. 64 tests pass.
- **Why:** Eval targets come from the complete source (two-table principle) so side-specific wOBA is unbiased ground truth; the FanGraphs wOBA weights were sourced, formula-verified, and validated by reproducing published league wOBA to ±0.0005 rather than recalled from memory. Stabilization quantifies signal-per-PA — the small-sample currency — and B.1 tests the process-beats-outcome premise before building anything.
- **Found:** Process stabilizes an order of magnitude faster than outcome: swing ~97 pitches, whiff ~34 swings, EV ~34 BBIP, LA ~21 BBIP, spray ~77 BBIP vs side-specific wOBA ~407 PA. Spray is the slowest process signal. The exposure asymmetry is the bite: at 800 PA only 143 hitters qualify vs LHP against 476 vs RHP.
- **Learned:** feature-selection taxonomy (filter/wrapper/embedded, permutation importance, SHAP); split-half reliability + Spearman-Brown + the stabilization point; the wOBA linear-weights formula and complete-source eval-target build; train-only fit as the standardization leakage boundary; the closed-form synthetic gate for verifying a reliability estimator.
- **Next:** Scaffold `notebooks/02_feature_value.ipynb`, then B.2 GBM screening.

---

## 2026-07-21 — Phase B.1 methodology hardening and results
- **Did:** Added a variance-components (one-way random-effects) estimator with bootstrap CIs and a sequential-split mode to `stabilization.py`, and `src/analysis/b1_report.py`, writing the panel, common-PA-axis ranking, spray-clipping check, and wOBA survivorship decomposition to `results/phase_b/`. 70 tests pass. Three decisions logged.
- **Why:** Two weaknesses in the 07-20 numbers: the single r=0.5 threshold was unsourced, so both r=0.5 and the literature's r=0.7 convention are now reported; and the pooled-process-vs-side-specific-outcome comparison was asymmetric, while split-half at large n only uses hitters who reach that n. The variance-components estimator decomposes signal and noise over all hitters, giving an analytic reliability(n), CIs, and the shrinkage constant; the sequential split gives the across-circumstance number the projection task actually faces.
- **Found:** Survivorship bias was real and material — split-half put wOBA vs LHP at ~435 PA, variance-components on all 2142 hitters put it at ~190. On a common PA axis the honest process-beats-outcome gap is several-fold rather than an order of magnitude: whiff ~28 PA-equiv, swing ~31, LA ~62, EV ~63, spray ~122, against wOBA ~190–198.
- **Learned:** variance-components reliability and its equivalence to Cronbach's alpha; survivorship bias in split-half-at-each-n; the r=0.5 vs r=0.7 distinction and the shrinkage reading of n*; random vs sequential split-half.
- **Next:** B.2 GBM feature screening on the 48 context features, through the frozen split, tuned on val never test.

---

## 2026-07-27 — Phase C opens: the claim-1 metric, the C.1 baselines, and a contaminated hitter population
- **Did:** Built the claim-1 evaluation harness (PA-weighted RMSE plus rank correlation, stratified by exposure, with noise-floor deconvolution) and the C.1 trailing-average baselines in two variants. Excluded pitchers taking their own at-bats from every hitter-talent quantity. Three decisions logged.
- **Why:** The harness needed to exist before any baseline could be scored. C.1's two variants decompose the value of shrinkage from the value of doing it properly.
- **Found:** Trailing averages underperform ignoring the hitter entirely below high exposure, turning positive only for veterans. Raw RMSE is mostly target noise, and removing pitcher-batters materially shifted stabilization points and widened the process-vs-outcome gap.
- **Learned:** noise-floor deconvolution and why RMSE against a noisy target is floor-dominated; that an unbiased estimator can still be unusable in small samples; the Diebold-Mariano paired-loss idea for model comparison.
- **Next:** Build C.2 with cell-specific variance and gates covering real sample sizes.

---

## 2026-07-27 — C.2 rebuilt: the incumbent now beats ignoring the hitter
- **Did:** Built the bivariate empirical-Bayes C.2 baseline and a shared `c_report.py` so every Phase C number comes from one seeded command.
- **Why:** Shrinking the two platoon sides jointly rather than the split directly is the same model in rotated coordinates, but identifiable where the split is not — PA vs LHP/RHP are disjoint, so the cross-side covariance needs no noise subtraction.
- **Found:** C.2 is the first Phase C baseline to beat ignoring the hitter in the low-exposure stratum, though platoon skill's magnitude is not tightly resolved and switch hitters remain effectively unidentified. Every baseline over-predicts low-exposure hitters, mostly from lineup management rather than league drift, and the real win turned out to be continuous shrinkage replacing C.1's step function rather than the cross-side borrowing the design was built for.
- **Learned:** bivariate empirical Bayes and the BLUP form of the posterior mean; that a parameterization can preserve information yet ruin identifiability; the paired bootstrap for comparing two models against a noisy shared target.
- **Next:** Build C.3, then run the Phase C review before promoting any result.

---

## 2026-07-28 — C.3 closes Phase C, and most of its win turns out to be a level correction
- **Did:** Built C.3, the last Phase C baseline — XGBoost over the hitter's own prior-window context rates, in two feature sets (`outcome`, `full`). Added a label-shuffle null, a per-stratum bias table, and an oracle level-correction diagnostic to the shared report. Two decisions logged.
- **Why:** The plan named the model but not its row unit; the claim-1 metric settled it. The two feature sets separate "does a learned model beat empirical Bayes on the same inputs" from "does process signal actually project."
- **Found:** C.3-full beats C.2 on RMSE in every stratum but loses to it on low-exposure rank correlation — the two frozen metrics appear to disagree for the first time. Most of C.3's RMSE margin is a level correction (it has exposure as a feature, C.2 structurally can't) rather than real ordering skill; on identical inputs a GBM doesn't clearly beat C.2.
- **Learned:** gradient boosting as a projection baseline; what a valid label-shuffle null requires; oracle recentering as a way to separate a model's level error from its ordering skill.
- **Next:** Run the Phases A–C review. Phase D's gate is now two-dimensional: RMSE bar from C.3, low-exposure rank bar from C.2.

---

## 2026-07-29 — Phase A-C review: two headline claims did not survive it
- **Did:** Ran `/research-review` over Phases A-C and worked the findings — added a paired bootstrap for ordering claims, moved both paired comparisons onto batter clusters, broke the eval-filter drop out per stratum and swept it, lowered the cut 25 → 10, and put every scoring weight on the wOBA denominator. Six decisions logged; 179 tests pass and all Phase B/C results regenerated.
- **Why:** The project applied a paired bootstrap to every RMSE claim and nothing to the rank claims, which is backwards — rank correlation is unweighted, so it is the noisier metric and needed the interval more. The eval filter needed the same scrutiny because it censors on eval-season playing time, decided after the projection and partly by the hitter's own performance.
- **Found:** The two frozen metrics do not actually disagree — the low-stratum ordering gap does not clear zero, so no model on the ladder demonstrably orders low-exposure hitters better, while the RMSE headline survived both batter clustering and a 3x swing in censoring. C.3 emits a single constant for the 43% of the low stratum with no prior history, which is the mechanism behind its apparent level advantage. The C.1 step also decomposes better than we had been telling it: raw trailing wOBA orders low-exposure hitters as well as C.2 while being badly miscalibrated, and bucketing fixes the level but destroys the order, so only continuous shrinkage gets both.
- **Learned:** the cluster bootstrap and why the resampling unit must be the batter when a hitter contributes two rows; selection on a post-treatment variable, and why an eval filter conditioning on playing time is the same hazard as stratifying on held-out PA; permutation-importance deflation under collinearity; and the distinction between role-matched baselines and information-matched ones, the second belonging inside Phase D.
- **Next:** Consolidate results into reporting notebooks, then Phase D. Owed before the context tower is built: B.2's six flagged features and their dissolved deferral target, and a decision-log entry for the bat-tracking exclusion in Nate's words.

---

## 2026-07-30 — Phase D planning: build order, gates, and cold-start handling
- **Did:** Planned Phase D as D.0 decisions through D.8 ablations, and logged two decisions: the ordering gate and cold-start handling for hitters absent from the training seasons. No code.
- **Why:** Phase C closed with the RMSE bar settable and the rank bar not, so Phase D's criteria had to be fixed before any number exists; the gate is a rule rather than a threshold because C.2's low-stratum figure was retracted and its sign reverses across the censoring sweep. Cold start follows from the hitter tower being a lookup table — a row only receives a gradient on batches containing that hitter, so a hitter with no training data has no row and no defined behaviour, and that population is 43% of the low stratum.
- **Found:** Two gaps in the architecture plan. It specifies no behaviour for a hitter with no training data while §1.4's mechanism presumes he has some, and it files the query machinery under Phase G though §5.2's claim-1 evaluation cannot run without it, since the model emits per-pitch conditionals and the metric wants side-specific wOBA. The query machinery moves into Phase D, composing terminal states to wOBA points so the output lands on the frozen metric's scale.
- **Learned:** the embedding table as a sparse-gradient lookup, so training a row means only the rows present in a batch move; UNK substitution and frequency-inverse word dropout; zero initialization as the choice that makes an untrained row and the generic hitter the same point; and that a technique can be sound in itself and still be wrong here, if it moves the same quantity the claim is about.
- **Next:** Settle the inner validation season — Phase D needs 2023 carved out of train for early stopping and hyperparameter selection, since C.3 is held to that standard on the same eval frame. Then D.1: pitch tensors, hitter index with the reserved zero row, and train-only bin edges for EV, LA and spray. B.2's six flagged context features enter as a block ablation at D.8.

---

## 2026-08-01 — D.0 closed, D.1 built, D.2 approved: Phase D is specified
- **Did:** Settled the Phase D selection frame, ran the D.0 measurements (`src/analysis/d0_checks.py`), added a denominator-weighted rank correlation to `claim1_eval` and regenerated Phase C on 2024 under it, built the pitch tensors (`src/data/model_dataset.py`), and wrote `docs/phase-d-spec.md` for approval. Seven decisions logged; 206 tests pass.
- **Why:** The frozen split already allocated 2024 as a validation season and 2025 as the final test, but Phase C never needed a selection budget — its baselines are pre-registered — so Phase D is the first phase where that allocation binds. Selecting on 2024 and reporting on 2025 puts the headline on data no comparison has read, at the cost of spending the test season on v1, which is acceptable because Phase F is judged unlikely. Ordering claims moved to a precision-weighted statistic before any Phase D number existed, so the choice could not be made to suit a result.
- **Found:** Weighting the ordering metric by the wOBA denominator changes one logged Phase C conclusion and leaves the other standing — C.2 now orders low-exposure hitters better than raw trailing wOBA with an interval excluding zero, while the retracted C.3-vs-C.2 low-stratum gap stays retracted. Of B.2's six flagged context features only `effective_speed` is redundant against the kept ten, and the three release features carry the most independent information of the six, inverting the prior. Two defects surfaced and were fixed: reporting on 2025 without refitting through 2024 would leave 75% of the low stratum on the untrained embedding row against 42.7% with the refit, and the hitter vocabulary was 35.7% pitchers taking their own at-bats while they held 1.60% of the pitches.
- **Learned:** the optimism of the maximum, and why a selection set and a report set cannot be the same season; weighted rank correlation as weighted Pearson over weighted ranks, and that "ranks carry no PA" describes the ranks rather than the coefficient; teacher forcing in an autoregressive head; AdamW's decoupled decay and why coupling it to gradient magnitude would tie shrinkage to exposure; and that batch size sets the decay-to-gradient ratio, so it modulates shrinkage rather than only speed.
- **Next:** D.3, building the model to `docs/phase-d-spec.md`, starting with a one-epoch benchmark before committing to any sweep. Owed: the Phase C ladder must be re-scored on 2025 with `--final-run` in the same pass as Phase D's headline, never before; the bat-tracking exclusion still needs a decision-log entry in Nate's words.

---

## 2026-08-03 — D.3 built and D.4 passed: the model exists and its decisions are executable
- **Did:** Built the v1 network and factorised loss (`src/model/v1.py`), the tensor loader (`src/model/loader.py`), the training loop (`src/model/train.py`), the D.4 verification gates (`tests/test_model_v1.py`), and an overnight run driver (`src/model/sweep.py`). Benchmarked one epoch on CPU and MPS. Two decisions logged; 226 tests pass.
- **Why:** D.2 fixed the equations, shapes, loss, and optimizer in advance so nothing would be settled during implementation, which made D.3 a transcription rather than a design step. The gates come before any real run because every failure mode here — a broadcast head, a leaked split, a masked row entering the loss, a frozen row that moves — produces a trained model with a plausible loss curve rather than an error.
- **Found:** Compute is not a constraint and the device does not matter: ~70 s/epoch with ±15% run-to-run variation, CPU and MPS agreeing on loss to five decimals, and only 2.0 s of that epoch spent moving data. The interaction term in its full form costs 1,048,832 parameters — five times the rest of the network — while the low-rank form recovers 98.1% of planted interaction variance for 13,312, and all 32 of its directions stay live after training. Two properties of the tooling also surfaced: torch's intra-op pool oversubscribes against the BLAS threads in the C.2 tests, stalling a suite whose every file passes alone; and cross-entropy's invariance to a constant shift across a row's logits means a masking gate that perturbs a whole row cannot distinguish a masked model from an unmasked one.
- **Learned:** reduced-rank random slopes as the structure the bilinear term expresses, and why the low-rank form is the honest version of "each hitter has context-dependent strengths"; `padding_idx` as the native mechanism for a permanently frozen embedding row; the locality of the log score and why a strictly proper rule is what licenses training on single observed outcomes; thread oversubscription between independent numeric libraries; and that partial state restoration is a worse failure than redoing work, because it breaks the link between a seed and its run silently.
- **Next:** Train the RPS screen arms and D.6's five baseline seeds overnight, which resolves the epoch count — the last unmeasured term in the phase's schedule. The screen still needs its scoring pass: both arms must be scored on held-out 2024 log-likelihood, since each arm's recorded loss is in its own units and the two are not comparable. Owed: four D.8 arms need code that is not written (B.2's flagged five, spray, per-head mean weighting, inverse-frequency in the contact head); the bat-tracking exclusion still needs a decision-log entry in Nate's words; C.2 inheriting frozen rule #1's empirical-Bayes role is unlogged; Gneiting & Raftery and the DRPS paper are not in the library; D.5's enumerate-vs-sample decision for the 24³ quality chain is open.

---

## 2026-08-04 — Phase D trained: every pre-registered null held, and no arm helped
- **Did:** Ran all 39 Phase D training runs — the RPS screen's two arms, D.6's five baseline seeds, and D.8's six arms at five seeds each — after building the two loss arms §7 still owed, per-head mean weighting and inverse frequency inside the contact head. Results in `results/phase_d/sweep_log.csv`.
- **Why:** The D.8 arms exist to say which pre-registered design choices are load-bearing. Three of them train against something other than the likelihood, so every run also records a canonical yardstick — unweighted log loss per scored row — because an arm's own objective is not comparable to another arm's.
- **Found:** Baseline seed-to-seed spread is 0.00096, which sets the bar every arm must clear, and none clears it favourably: `d = 64` and the bilinear term are null, `d = 16` is marginal, per-head mean weighting costs 0.0015, and inverse-frequency weighting costs 0.0295 — about thirty times the noise, the decalibration §4's argument predicted. Hitter identity is worth 0.0289 per scored row against the same model queried with every hitter mapped to the generic row, concentrated in exit velocity (0.0683) rather than swing or contact (0.024, 0.023).
- **Learned:** that departing from a proper scoring rule shows up as decalibration against the likelihood rather than as a wash, which is what makes a pre-registered null worth running; that an inference-time ablation bounds a component's contribution from above rather than estimating it, since the trained model has already allocated capacity around that component; and that comparing arm means calls for the standard error of the mean, not the spread of the seeds behind it.
- **Next:** Score the RPS screen — both arms on 2024 under the canonical objective, with reliability and resolution reported beside it, promote-only. Then spec and build D.5's query machinery: no claim-1 number exists until the per-pitch conditionals compose to side-specific wOBA, so nothing in the ablation table is yet a verdict, and beating a generic-hitter model is not the same as beating Phase C. Still owed: the B.2 block and spray arms need code, and the bat-tracking exclusion still needs a decision-log entry in Nate's words.

---

## 2026-08-09 — D.5 built and claim-1 scored: the gate fails on a level bias, not on ordering
- **Did:** Specified and built D.5's query machinery (`src/model/query_tables.py`, `src/model/query.py`, `src/analysis/d5_report.py`), added the three-class contact-split head and retrained all eight ablation arms as ledger stage `d9`, and produced the project's first claim-1 numbers for three Phase D variants. Thirteen decisions logged; results in `results/phase_d/`.
- **Why:** No claim-1 number could exist until the per-pitch conditionals composed to side-specific wOBA, so thirty-nine training runs and an ablation table were still measured in a unit that says nothing about projecting hitters. The fourth factor was added because a binary in-play/not-in-play split folds caught foul tips into the two-strike self-loop, and a foul tip is a strikeout there while a foul is not.
- **Found:** All three variants fail both pre-registered gates, but the failure is one-dimensional — Phase D orders hitters better than anything on the ladder (weighted rank 0.4608 against C.3-full's 0.4417) while losing on calibration, and no ordering interval excludes zero. The calibration deficit is almost entirely a uniform level bias: removing it leaves a residual sd of 0.04590 against C.3-full's 0.04634. The block arm fires at 1.8x the noise floor, so B.2's five flagged features earn their place, and the take surface turns out to condition on location but not count, which reproduces the composition residual exactly.
- **Learned:** absorbing Markov chains solved by backward induction, and the two-strike foul self-loop as a geometric series closing in one division; deep ensembles combined by averaging conditionals rather than parameters or predictions; that a factorised expectation over a 24³ joint collapses to one matmul when the conditioning enters a linear head, so enumeration replaces sampling; that a lookup surface can be unbiased marginally while carrying a large structured error along a dimension it does not condition on; and that dividing by an estimated quantity converts sampling noise into upward bias.
- **Next:** Run `/research-review` over Phase D before building on these numbers, then work the plan in `docs/phase-d5-review.md` — diagnose the level bias without reading claim-1, discharge the `nospray` arm and the two owed diagnostics, and re-score reporting both ways. Still owed: the bat-tracking exclusion needs a decision-log entry in Nate's words.
