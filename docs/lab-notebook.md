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
- **Did:** Built D.5's query machinery (`src/model/query_tables.py`, `src/model/query.py`, `src/analysis/d5_report.py`), added the three-class contact-split head, retrained all eight ablation arms as ledger stage `d9`, and produced the project's first claim-1 numbers. Thirteen decisions logged.
- **Why:** No claim-1 number could exist until per-pitch conditionals composed to side-specific wOBA. The fourth factor was added because a binary in-play split folds caught foul tips (a strikeout at two strikes) into fouls.
- **Found:** All three variants fail both gates, but the failure is one-dimensional — Phase D orders hitters better than anything on the ladder (weighted rank 0.4608 vs C.3-full's 0.4417) while losing on calibration, and it's almost entirely a uniform level bias (residual sd 0.04590 vs 0.04634 once removed). The take surface conditions on location but not count, reproducing the composition residual exactly.
- **Learned:** absorbing Markov chains solved by backward induction; deep ensembles combined by averaging conditionals; a factorised expectation over a joint grid collapsing to one matmul when conditioning enters a linear head; that a lookup surface can be unbiased marginally while carrying structured error along an unconditioned dimension.
- **Next:** Run `/research-review` over Phase D, then work `docs/phase-d5-review.md`'s plan — diagnose the level bias without reading claim-1, discharge the `nospray` arm.

---

## 2026-08-15 — D.5 remediation: the null survives every fix and is now attributable
- **Did:** Worked the seven-step D5-R14..R18 remediation plan: fidelity check rebuild, cleaning pass, xwOBA in three roles, the four level contributors, bin contest plus a 40-run retrain, claim-1 for all eight arms, gradient test (b), reporting fixes.
- **Why:** No fix could flip a gate, so the work is justified by making the null attributable. The sanctioned validation channel had never been able to detect a level error of this size, and zero of eight arms carried a claim-1 number.
- **Found:** Level bias now two-thirds accounted for; remainder is one rate (walks fail; under the credit rule, none of the improvement is credited — every move is smaller than its own between-seed spread). Equal-mass binning won its own contest outright. Claim-1 and held-out log loss have rank correlation exactly 0.000 across seven comparable arms; four arms beat baseline by margins inside one arm's own seed range. Gradient (b) reversed the phase's expectation: low-exposure rows are displaced, not shrunk — decay removes only ~8% of embedding length per run, ~12x too weak to bind (predicted sd 0.0374 low-exposure vs 0.0281 regulars).
- **Learned:** the exact 1-D DP for supervised binning and why per-dimension objectives are wrong for a joint-cell prediction; that a metric can validly compare models yet carry no information about the claim being settled; that a credit rule comparing a paired delta to a level's dispersion credits nothing, however real the fix.
- **Next:** Phase E. Walk gap points to take frequency (swing head), not ball-given-take. Adopting `block` needs the ladder re-scored on the no-block build.

---

## 2026-08-18 — Phase E opening window: validate the scorer, then evaluate the claim
- **Did:** Pre-registered the window in `docs/phase-e-spec.md`, then ran ten steps: E.1-E.5 (population-matched fidelity, contamination, level-bias closure, calibration, platoon differential), E.6 (swing-head calibration on real pitches), E.7-E.10 (the four steps E.6's fork opened). `query.py` untouched, so the scorer gate holds trivially. Nothing trained; 2025 final run moved to after Phase O.
- **Why:** Validate the scorer first (E.1-E.3), then evaluate the model (E.4-E.5), only then chase the residual. Two traps designed around in advance: the D.5 walk gap compared a modelled side over trained hitters against an observed side over every PA — selection on exposure — so E.1 rebuilt the observed side over the scored population, closing 54% of the gap. The contamination test regresses `modelled-observed` on `observed`, which is mechanically compressive from target sampling noise alone; corrected with the binomial noise variance `p(1-p)/n`.
- **Found:** Only 25.5% of the composed-metric level bias is a rate effect (wrong absorbing-state frequencies); the rest is a value effect (modelled E[wOBA|BIP] 0.37864 vs observed 0.36353). E.10 shows a count chain built from real counted frequencies (no model) still over-predicts walks +1.44% relative — a property of treating a PA as independent count draws, not of anything trained, so no retrain fixes it. The resampler distorts `P(ball|take)` (+0.00093 walks) and `p_swing` (−0.00096) in opposite directions that nearly cancel.
- **Learned:** errors-in-variables bias in a residual-on-target regression; Route A (overall skill + league-average split) as the platoon null a real signal must beat; Oaxaca-style rate/value decomposition of a composed-metric bias; that a Markov chain fed perfect real transitions can still be structurally wrong.
- **Next:** Strikeout side is the open thread — passes unmatched (−1.67%) but fails matched (−4.78%), and E.10's structural bias on strikeouts is the wrong sign to explain it. ~22% of the walk gap is also unowned. Both point to the contact-split table (how often a two-strike PA survives on a foul) as the next diagnostic.

---

## 2026-08-19 — Phase E closing window: both gates fail, and the failure is partly located
- **Did:** Re-scored claim-1 on the D.10 arm, ran the committed MIN_EVAL_PA sweep against the D/E headline, and added four analyses — the ablation-table level confound, the BIP value-gap decomposition, an embedding probe with ensemble interval coverage, and the platoon measurement ceiling (`src/analysis/e_{min_pa_sweep,level_confound,bip_value,probe_coverage,platoon_ceiling}.py`, artefacts in `results/phase_e/`). E.11-E.12 and E.13's era-drift check are fork-opened, generated and tested on the same data; E.13 Check A, E.14 and E.15 were pre-registered in `phase-e-spec.md` §12.
- **Why:** The window's job was to make the null attributable rather than merely recorded, which meant auditing the standing commitments — architecture §5's MIN_EVAL_PA sweep existed only for Phase C, and its ensemble-coverage item had been owed since 2026-08-08 — before chasing any residual.
- **Found:** Both gates fail in every stratum, and the low-stratum ordering result is underpowered rather than absent: D.10 leads C.3-full on weighted rank 0.2592 to 0.1665, a paired gap of +0.0927 [-0.0343, 0.2194] needing 914 batters against 239 available. The verdict survives the censoring sweep at 10/25/50, where the low-stratum gap in fact grows to 0.1099 and one isolated cell flips. Three results locate the failure with unequal strength: the ablation table's level/RMSE confound is decisive on the all-stratum column (r=0.888, p=0.0076) but only marginal in the decisive `low` stratum (r=0.756, p=0.049; Spearman p=0.215), so it constrains Phase O without settling it; the BIP value gap is era drift in the frozen value table (+0.028836, 191%) rather than the measurement seam (-0.016154, wrong sign), leaving a 16.1% residual that absorbs the un-orthogonalised interaction; and a ridge probe decodes the LHB platoon split above its shuffled null (+0.223 [0.094, 0.351]) while RHB does not (+0.048 [-0.054, 0.142] against null +0.075 [-0.019, 0.175]) — overlapping intervals with no test of the difference, on a linear read-out against a shrunk C.2 target, so a bound on linearly decodable information and not a demonstrated representation failure. Three pre-registrations failed: E.13's Check A came in at the wrong sign and larger than the whole gap, E.15's corrected between-stand share landed at 0.814 [0.42, 3.50] against a predicted 50-60%, and E.15's asymmetry check could not be evaluated at all because the noise-corrected LHB within-stand variance came back negative. That correction is the window's one reversal: 86.6% of the *total* platoon-differential variance is sampling noise, and removing it from the within-stand term alone lifts the observed between-stand share from 0.109 onto the model's own 81.7%. Ensemble intervals under-cover globally (0.391/0.690/0.856 against 0.50/0.80/0.95), and cold-start rows inside the low stratum are indistinguishable from trained ones at n=162/218 — too few to resolve an effect of the size at issue, so shared shrinkage is unrefuted rather than excluded.
- **Learned:** RMSE = bias² + variance as the reason a level offset can dominate an ablation table while carrying no ranking information, and the scale check — between-arm spread against within-arm seed spread — that converts that suspicion into a verdict; reliability as true-talent variance over observed variance, with √reliability the attenuation ceiling on an achievable rank correlation and Spearman-Brown the split-half route to it; that errors-in-variables attenuation recurs in verbal framing as well as in a regression, the 0.109 and the 0.814 being the same trap at two scales; that a shuffled-target null for an out-of-fold ridge is not centred on zero, since out-of-fold predictions of noise are anti-correlated with the target, which is why the test is "beats null" and not "beats zero"; interval coverage as an honesty property independent of the slope property E.4 measured; and that a pre-registered test can be refuted structurally before it runs, Jensen's inequality being unable to operate on a composition linear in the bin distribution.
- **Next:** Phase O inherits three constraints: it must not select arms on the RMSE column alone, it owns the unowned strikeout shortfall and ~22% of the walk gap from E.10, and any ensemble-width change re-opens E.14's coverage. Two builds sit on disk with different `quality_bin_edges` and five Phase E modules still default to the stale one — harmless today, a mis-binning trap for any future module reading the quality bins. Still owed: the `block` adoption needs the C-ladder re-scored on the no-block build; the bat-tracking exclusion and C.2's empirical-Bayes role need decision-log entries in Nate's words; and The Book p.157 is decision-bearing but unverified, which blocks the write-up.

## 2026-08-20 — grading the process model instead of only its scalar summary
- **Did:** assembled `notebooks/04_model.ipynb` and `notebooks/05_evaluation.ipynb` from committed artefacts only, then built a small Phase F lane to fill the hole those notebooks exposed. New modules: `src/analysis/provenance.py` (tensor-build stamping and a quality-bin assertion), `f3_heads.py` (per-head calibration on 705,344 held-out pitches), `f4_process.py` (held-out NLL against a cold-start and a frequency reference), `f5_pooled.py` (the claim-1 metric with handedness pooled away). Artefacts in `results/phase_f/`. Backfilled 27 unit tests across the five Phase E audit modules that had none, plus 7 for the Phase F helpers; suite 344 to 378. Closed two latent bugs in `e_resample` and `e_take_mass` after confirming neither moves a committed number. Five decision-log entries.
- **Why:** the question that started it was "did we evaluate whether the model measures what we wanted, or how accurate it actually is?" The honest answer was the first only. Every gate in this project scores `expected_woba`, which is not a model output at all — it is composed downstream by an exact Markov count-chain solve over the six heads and a frozen value table. So we had built a process model and graded only its scalar summary. F.3 and F.4 grade the process; F.5 answers the accuracy question the platoon framing had crowded out. The design constraint throughout was that no new referee gets written: F.5 re-runs `claim1_eval` unchanged on a re-aggregated frame, and F.3 imports `v1.factor_masks` rather than reimplementing the head nesting, because a second copy of either is a second thing to keep in step.
- **Learned:**
  - **Proper scoring rules for head diagnostics.** Accuracy or AUC would have thrown away the calibration that the composition actually integrates over. NLL is proper — it is minimized by reporting your true belief — so it measures the thing the count-chain consumes.
  - **Ablating identity by reference row, not by shuffle.** Replacing the hitter embedding with the reserved zero row removes identity. Shuffling it would inject a *different* hitter's signal, so the contrast would measure confusion rather than absence.
  - **Prior-exposure weighting as a leakage boundary.** Pooling the two hand-specific forecasts by eval-season PA would have conditioned the forecast on the answer key's own sample. Prior exposure is information a forecaster actually holds, and it is the same reason `claim1_eval` stratifies on prior rather than realized PA.
  - **A sign check can exonerate a suspect faster than an interval can convict one.** The contact head is biased, but biased in the direction that makes strikeouts *more* frequent, while the residual is too *few* strikeouts. No confidence interval was needed to rule it out.
- **Open:**
  - The strikeout residual is still unowned; the contact head is now excluded, leaving the early-count swing rate and E.10's count-chain independence assumption.
  - F.3's effect sizes are unclustered. Batter-clustered intervals were not computed and the argument rests on sign consistency instead.
  - Four known bugs left unfixed in the Phase E audit modules, all latent on today's data, all recorded in the log.
  - Still owed from D.5: decision-log entries in Nate's words for the bat-tracking exclusion and for C.2 inheriting frozen rule #1's empirical-Bayes role.
  - **The Book p.157 remains UNVERIFIED and decision-bearing** (`src/analysis/c_report.py:45` records that we have no copy). This blocks the write-up, not the code.
- **Next:** re-execute notebooks 04 and 05 against `results/phase_f`, then Phase O. Notebook 04 gets refreshed after the O refit regardless.

## 2026-08-20 — the project stops trying to win and starts measuring the ceiling
- **Did:** restructured the remaining arc in the architecture plan — Phase O narrowed from arm selection to hyperparameter tuning, a new Phase M added as the measurement phase, order fixed at O → M → V → 2025 run → write-up. Built Phase O: `--lr` and `--warmup-steps` on `train.py` (both were module constants), a `LinearWarmup` that stands down permanently once done and a `warmup_for` guard that refuses a warmup still running when `ReduceLROnPlateau` can first fire, an `o1` sweep stage (3 learning rates × {no warmup, one epoch}, pinned to the build D.10 trained on), and `src/analysis/o1_select.py` whose promotion rule is fixed in code before any `o1` run exists. `sweep_log.csv` gains `lr` and `warmup_steps`, backfilled. 24 new tests, suite 378 → 399 → 402. One two-epoch smoke run confirms warmup end to end. `docs/phase-o-spec.md`, five decision-log entries, and a gate-outcome note in the manifest.
- **Why:** the session opened on whether the platoon goal is feasible at all and whether the project should become embedding-building with platoon left to a frontend. It should not: that converts a pre-registered null into an unstated one and leaves nothing falsifiable. The gate failed against a target where noise is 86.6% of observed platoon-differential variance and the maximum achievable rank correlation is 0.356 — so the honest headline is the ceiling, with the failed gate reported prominently as its setup. Phase O exists because a null on an untuned model is a weaker null, and "the learning rate was 1e-3 for all 119 runs and was never varied" does not survive review.
- **Found:**
  - The displacement artifact is a block inside `d5_level_attribution.json`, not its own file. Every embedding row initialises at norm 0.057 and travels **outward**; the rarest quintile ends at mean norm 0.74–1.00 against the most-exposed quintile's 0.52–0.61, while its projection on the wOBA-raising axis runs −0.16 to −0.22 against +0.06 to +0.07. Consistent in all five seeds. Nothing is being pulled toward the origin.
  - That **rules out weight decay and batch size as the cause.** At the 10th percentile of exposure the decay-to-gradient ratio is 23.9:1, so a binding decay would crush the rarest rows *into* the origin. They are the furthest out, so decay is being overwhelmed.
  - `o1` had to pin `provenance.CANONICAL_DATA_DIR` explicitly. The sweep's `--data-dir` default is `phase_d` but D.10 trained on `phase_d5`, and the two builds have different quality-bin edges and different manifest shas — so inheriting the default would have put `o1` and its own incumbent in different units with every column still lining up.
- **Learned:**
  - **A metric can be barred for ranking and legitimate for detection.** Held-out likelihood correlates 0.000 with claim 1 across seven converged arms, which says it cannot *rank* good models. It can still detect an *undertrained* one, because an undertrained run is worse on its own objective. Conflating the two nearly cost the project its tuning phase.
  - **A selection axis and a tuning knob are different things,** and the guard against the first is not a reason to skip the second.
  - **Weight decay's strength is not the same question as whether it binds.** The decay-to-gradient ratio said the rarest rows should be the most shrunk; the artifact says they are the least. The ratio was computed correctly and still pointed the wrong way, because it measures decay's opportunity, not its outcome against AdamW's per-coordinate normalisation.
  - **A screen that selects among many challengers against one incumbent has to say so in its own artifact.** The verdict carries `requires_confirmation` rather than leaving the five-seed re-run to a line in the spec.
- **Open:**
  - The strikeout residual still has two unowned suspects — the early-count swing rate and E.10's count-chain independence assumption.
  - No low-exposure platoon ceiling exists: E.15 stratified by stand, not by exposure, and the low stratum is the one the project is about.
  - C.2 and C.3 have never been scored on the platoon-differential cut, so the ceiling table has no incumbent in it. This is a prerequisite to the reframe, not an item inside it.
  - E.15 Part 3's noise-corrected LHB within-stand variance is negative (−3.75e-05), so the L/R asymmetry cannot be presented as a finding and the by-stand fractions stay descriptive.
  - **The Book p.157 remains UNVERIFIED and decision-bearing.** Still blocks the write-up.
- **Next:** run `o1` (twelve runs, ~4.5 h, one overnight), then `o1_select`, then Phase M — C.2/C.3 differential gap first, since the reframe is not final without it.

## 2026-08-20 (later) — the review found three real things, one of which the review itself was wrong about

**Did.** Implemented the Phase O steps, then ran a review over them and worked its
findings. Three Critical, six Important, five Minor. Fixed twelve, disagreed with one,
converted one into a documentation obligation rather than a code change.

**Why.** "Implement, test, and review" was the instruction, and a review whose findings you
read and then don't act on is a review you did for the feeling of having done one.

**Found.**

*The tensor build.* The single most expensive finding, and it turned out fine — but only
because it was checked. The `o1` stage pins `phase_d5`, on the strength of a source comment
saying d10 ran there. `sweep.py`'s own `--data-dir` default is `phase_d`, and d10's stage
tuples pin nothing, so the comment was the only evidence. Two builds, different quality-bin
edges, and `reference` is log loss over those bins — wrong answer means the guard fires
after a full night of compute and the sweep returns `guard_failed`. Settled it by
reproduction: `phase_d5` reproduces d10 baseline seed 0's epoch-0 losses to the digit
(1.05990 / 1.04681), and `phase_d` can't even run the command — `KeyError('split')`,
because that build predates the three-class contact split. Every d10 arm used `--split`.
So the pin was right, and it is now a recorded fact in a `data_dir` ledger column rather
than a comment.

*The firewall test was theatre.* `test_selector_cannot_see_claim1` grepped the module's
source text, and the reviewer showed it still passed with `from src.analysis.claim1_eval
import evaluate` inserted at the top — one clause was already satisfied by the docstring's
own mention of `claim1_eval`, and the other only matched a spelling nobody writes. Replaced
it with a behavioural check: import the module in a clean interpreter, look at
`sys.modules`. It failed immediately — and not on the reviewer's hypothetical, but on a
real leak *I* had introduced four edits earlier, when I imported `sweep.STAGES` to get the
expected arm list and dragged `claim1_eval` in transitively. Fixed by declaring the grid in
`o1_select` itself, with a test that keeps the two copies in step.

*Two spellings of one number.* `knobs()` fell back to `str(LEARNING_RATE)` — `'0.001'` —
into a column whose 119 rows said `'1e-3'`. Not cosmetic: the column is read as text and
grouped on. One `canonical_lr`, re-backfilled.

*Where I disagreed.* The reviewer's third Critical was that Phase O tunes on 2024 while
claim 1 is scored on 2024, so the headline is post-selection. The premise is right and the
framing isn't: 2024 is the *frozen selection season* and 2025 is the test season, so every
Phase D ablation already selected on 2024 and Phase O adds no new leak. What it does add is
an obligation — Phase M's 2024 numbers, computed on a build tuned on 2024, are descriptive
and must be labelled so. That went into the decision log and the spec rather than into a
change to Phase O. The reviewer was right, though, that my `r = 0.000` argument was bad:
seven points give it a ±0.75 interval, so it establishes nothing and I had leaned on it as
if it did. The licence is the detection-vs-ranking distinction, which never needed it.

**Learned.** A test that asserts on source text is asserting on a proxy, and the proxy and
the property drift apart silently — the behavioural version caught a live regression on its
first run, which the text version had been quietly failing to catch. Also: a comment
explaining which artifact a run used is not provenance. It took one six-minute reproduction
to convert it into evidence, against a downside of one wasted night.

**Open.**
- The warmup grid is 719 steps; Ma & Yarats' default is ≈2/(1−β₂) = 2,000 at β₂ = 0.999.
  This grid is on the short side of the literature default. Logged as a limitation.
- `data_dir` is blank for `screen`, `d6` and `d8`. Recoverable the same way if any of those
  numbers is ever read down a column with a d9/d10 one.

**Next.** Run the o1 sweep (~4.5 h, 12 runs at two seeds), then `o1_select`. Then Phase M,
starting with the C.2/C.3 differential gap, which is the prerequisite for everything else
in that phase.

## 2026-08-25 — the tuning phase returns a null, and the measurement phase inherits an unchanged build

**Did.** Ran the o1 sweep — the 3×2 learning-rate-by-warmup factorial at two seeds,
twelve runs, about three hours on `data/processed/phase_d5` — then the pre-registered
selector. Verdict `incumbent_stands`. Wrote the β₂ and Phase O outcome entries into the
decision log.

**Why.** Phase O exists to remove one specific reviewer answer: that the learning rate was
held at 1e-3 for all 119 runs in the ledger and so is evidence for nothing. A null on a
tuned model is a stronger null than the same null on an untuned one, and that is the only
thing Phase O was ever buying.

**Found.** The untuned setting was already best. `lr1e3` wins at margin 0.0; the best
challenger `lr1e3_warm` reaches +0.8 SE against a 2 SE bar, and the other four arms come
in at −1.5, −2.7, −10.3 and −10.5 SE. No arm in the swept 10× range beat the incumbent. Warmup helped directionally at both higher learning rates and never cleared the bar,
and `lr1e3_warm`'s own two seeds spread 1.27e-4 — wider than the 8.5e-5 margin they were
supposed to support, which is the thin basis the rule correctly refused.

That answers O.2 without new computation. The pre-registered prediction was that the
rarest exposure quintile's embedding norm would fall toward the most-exposed quintile's
on the tuned build. The tuned build is the D.10 build, `gradient_b`'s five per-seed
measurements read the `d10_baseline` checkpoints, so the exposure–norm gradient is intact
and O.2 is measured separately rather than argued from provenance. The plan says a null
there is reported, not retuned around, so it is reported. It also means the 2026-08-20
quarantine entry's revisit condition has now fired, pointing at batch size and weight
decay under the terms that entry already fixed.

One trap worth naming before Phase M touches anything: the `o1` `lr1e3` arm is a fresh
two-seed re-run, not the D.10 checkpoints. The frozen uncertainty decision is a five-seed
deep ensemble, so Phase M's build is `d10_baseline_s0..s4` and the two o1 checkpoints are
a screening artifact that must never become the build.

**Learned.** A promotion rule fixed in code before the first run is what makes a +0.8 SE
result readable as a null instead of as a near-miss worth one more grid. And a noise floor
taken from the quietest arm understates run-to-run spread by roughly 4× here, so the SE
column reads generously — it happens not to matter at these margins.

**Next.** `/research-review` on Phase O — the sweep, the selector run, and the reading of
the verdict, none of which the earlier implementation review covered. Then Phase M item 1,
the C.2/C.3 differential gap, which is the declared prerequisite for the ceiling table.
Three `unverified` references are still open and one of them, The Book p.157, is
decision-bearing and blocks the write-up.

## 2026-08-25 (later) — warmup rescales the embedding space without touching the exposure gradient

**Did.** Ran O.2 as the spec wrote it: two composition passes over the `o1_lr1e3_warm`
checkpoints (seeds 0 and 1, ~1.7 h each, concurrent), then `gradient_b` against the
five-seed `d10_baseline` reference. Corrected `shrinkage_in_woba_direction` in
`src/analysis/d5_level.py` and re-emitted `d5_level_attribution.json`.

**Why.** The Phase O outcome entry had closed O.2 on the argument that the tuned build is
the measured build. That is a statement about the file, not about the behaviour, and it
would license skipping any diagnostic whose inputs had not changed.

**Found.** The pre-registered expectation is contradicted. Seeds are matched pairs —
`torch.manual_seed` fixes both initialisation and batch order — so the comparison is
paired, and paired it shows warmup shrinking every quintile by a near-uniform factor
(0.915 to 0.927 across the five quintiles in seed 0). The q1/q5 norm ratio moves −1.3% and
−0.04%; the exposure-normalised slope moves −3.8% and −2.2%, away from pooling rather than
toward it. Global rescaling, not differential shrinkage. Separately, the corrected wOBA
flag reads `false` in all five d10 seeds at −0.0050 to −0.0074 per 1,000 pitches with the
interval excluding zero — anti-shrinkage holds along the wOBA axis, not only in norm.

**Learned.** A slope test is a distance test only when the quantity has a fixed sign; on a
sign-crossing quantity it reports the opposite of what it names. And an unpaired range
comparison throws away the seed variance that matched seeds were there to remove — the
paired differences here are 3–8× smaller than the across-seed spread and would have been
invisible.

**Next.** Phase M item 1, the C.2/C.3 differential gap, which is the prerequisite for the
ceiling table. Two obligations open: the verification gate on the warmup scheduler
(`train.py:124-155` is training code with no recorded gate run) and the stray
`results/checkpoints/o1_warmup_evidence_s0.pt` with no ledger row. Three `unverified`
references remain, The Book p.157 still decision-bearing.
