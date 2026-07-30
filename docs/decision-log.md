# Decision Log — Hitter Embedding

Append-only. Format fixed by `~/os/knowledge/frameworks/research-standards.md`
§4 — `/research-partner` and `/research-review` both parse against it.

---

## <YYYY-MM-DD> — <decision title>
- **Decision:** what was chosen
- **Alternatives:** what was considered and rejected
- **Rationale:** why, in terms a skeptical reviewer would accept
- **Reference:** external source backing the decision (optional)
- **Revisit if:** the condition under which this should be reopened

---

## 2026-07-14 — Statcast raw-snapshot storage design
- **Decision:** Raw Statcast frozen to `data/raw/statcast/snapshot_<date>/season=<YYYY>.parquet` — per-season, all columns, immutable; a committed `manifest.json` holds pull metadata. Data gitignored; script + manifest committed.
- **Alternatives:** Monolithic parquet (forces full-memory loads); column pruning at pull time (unrecoverable without re-pull).
- **Rationale:** Date-stamped snapshots handle Statcast's retroactive revisions; immutable raw lets the pitch-event table re-derive without re-fetching.
- **Revisit if:** snapshot outgrows local disk.

---

## 2026-07-15 — Statcast cleaning spec for the modeling pitch table
- **Decision:** Modeling table is regular season only, dropping position-player pitches, pitchouts, automatic balls/strikes, and bunts (intentional balls kept). Rows missing or physically impossible on core context (type, velocity, location, movement) are dropped; optional spin context is kept with missingness indicators. Drops 8 deprecated columns, dedupes on pitch key, sorts by game_date then pitch key. Filters apply only to the modeling table, never to evaluation targets.
- **Alternatives:** A minimum-PA hitter floor (rejected: deletes the low-exposure population the thesis targets). A velocity-based position-player rule (rejected: misclassifies hard-armed position players). Dropping spin columns (deferred to a Phase B ablation).
- **Rationale:** Every filter is backed by the profiling notebook; the modeling-vs-target split keeps sharpening filters from biasing ground truth.
- **Revisit if:** Phase B feature screening changes the retained context set, or target construction needs a filtered field.

---

## 2026-07-17 — Contact-quality label domain
- **Decision:** The contact-quality head (EV, LA, spray) is labeled on balls in play only (`hit_into_play`); `ev`/`la`/`spray` are null elsewhere.
- **Alternatives:** Labeling all contact with EV (rejected: fouls carry EV/LA ~70% of the time but no spray and no batted-ball outcome, giving ragged masking and no run-value mapping).
- **Rationale:** Fouls are contact for the whiff head but non-terminal count transitions in the Markov composition; only in-play balls carry the run-value outcome the quality head feeds.
- **Revisit if:** the outcome space or run-value mapping (§1.5) changes to need foul-ball measurements.

---

## 2026-07-17 — Spray angle derivation
- **Decision:** Spray = `atan((hc_x − 125.42)/(198.27 − hc_y))` in degrees, mirrored so positive = pull for both hands (flip sign for RHB).
- **Alternatives:** Empirically calibrating the home-plate origin (dropped once constants were sourced); raw field-side angle without mirroring (rejected: not batter-intrinsic).
- **Rationale:** Formula and constants corroborated by three sources (abdwr3e App. C, BGSU, Weise). MLB doesn't publish the origin, so a real-data regression guard (field mean ≈ 0, pull-mean > 0) confirms the export matches scale.
- **Revisit if:** the near-plate artifact (`|spray| > 90°`, ~1% of in-play) needs clipping, decided in Phase B.

---

## 2026-07-17 — Walk-forward split frozen
- **Decision:** Contiguous walk-forward, train 2015-2023 / val 2024 / test 2025, single fold, frozen in `src/config/split_config.json` and validated on load.
- **Alternatives:** Random k-fold (rejected: leaks a hitter's future PAs into his own ID embedding, manufacturing a positive result). Rolling multi-fold (deferred: multiplies compute against the <$200 budget with no gain on the axis we grade on). Val/test gap season (rejected: wastes data, breaks Phase B's same-season window trade).
- **Rationale:** Projection is a forecasting task, so eval must mirror deployment — train on the past, test on a strictly-later season. Freezing before any model comparison pre-registers the held-out season; contiguity minimizes distribution shift so the metric measures projection skill, not regime drift. Robustness comes from exposure stratification + dual-sampler, not temporal folds.
- **Revisit if:** never for this project (frozen rule); a new fold requires a new entry naming this one.

---

## 2026-07-21 — Stabilization reported at two thresholds (r=0.5 and r=0.7)
- **Decision:** Every stabilization point is reported at both r=0.5 and r=0.7. r=0.5 (signal variance = noise variance) is the equal-weight-with-prior point — the small-sample projection currency the thesis trades in. r=0.7 is the stricter "reliable measurement" convention.
- **Alternatives:** Single r=0.5 (rejected: invites an unanswered "why 0.5?" and hides that half the variance is still noise). Single r=0.7 (rejected: not the quantity the shrinkage/projection argument uses).
- **Rationale:** The two thresholds answer different questions; reporting both preempts the threshold objection and lets us cross-check against Carleton's published r=0.7 numbers.
- **Reference:** Carleton, "Reliably Stable (You Keep Using That Word)" (Baseball Prospectus); FanGraphs "A Long-Needed Update on Reliability."
- **Revisit if:** the paper's reviewers want a different reliability convention.

---

## 2026-07-21 — Variance-components estimator added alongside split-half
- **Decision:** Added a one-way random-effects (variance-components / Cronbach-alpha) estimator to `src/analysis/stabilization.py`: one signal/noise decomposition over all hitters yields an analytic reliability(n), stabilization point, and bootstrap CI. Split-half is kept as an independent cross-check; the two agree on the closed-form synthetic. The variance-components point is the headline where they diverge, since split-half at large n only keeps hitters with ≥n observations (survivorship).
- **Alternatives:** Split-half only (rejected: survivorship-biased at large n, exactly where the wOBA outcome lives, and gives no CI). Mixed-model REML (rejected for now: heavier, and method-of-moments ANOVA matches Cronbach's alpha at a fraction of the code).
- **Rationale:** On the real table the two estimators agree on the fast process metrics but diverge ~2x on wOBA (VC 190 PA vs split-half 435 PA vs LHP) — a survivorship artifact the variance-components method removes by using all 2142 hitters, not just the durable ones.
- **Reference:** FanGraphs "A New Way to Look at Sample Size (Math Supplement)" — Cronbach-alpha signal/noise decomposition; KR-21 for the binary heads.
- **Revisit if:** the VC/split-half divergence on wOBA turns out to be heteroscedasticity in the VC assumptions rather than survivorship in split-half (disentangled in notebooks/02).

---

## 2026-07-21 — Matched-slice and across-time reporting for the process-vs-outcome comparison
- **Decision:** B.1 additionally reports (a) process metrics sliced by pitcher hand (whiff/EV vs LHP and vs RHP), matching the side-specific outcome slice; and (b) a sequential (chronological early-half vs late-half) split alongside the random split, so absolute points reflect across-circumstance reliability, not just within-sample consistency.
- **Alternatives:** Pooled process vs side-specific outcome only (rejected: apples-to-oranges — some of the gap is the split, not the process/outcome distinction). Random split only (rejected: measures within-sample consistency, which flatters the projection-relevant number).
- **Rationale:** Matched slicing kills the comparison-asymmetry objection — process stays fast even split by hand (whiff ~45–50 swings per side vs pooled 51). The process-beats-outcome gap survives every slicing and split choice; only the absolute points move.
- **Reference:** Carleton, "Reliably Stable" — sequential/different-circumstance splits drop reliability vs same-circumstance splits.
- **Revisit if:** the sequential split shows a large systematic across-time degradation on the headline metrics (modest in the 2026-07-21 run).

---

## 2026-07-27 — C.1 trailing-average design: 3-season window, both shrinkage variants
- **Decision:** Phase C.1 baseline uses a 3-season trailing window, built in two reported variants: `raw` (unshrunk trailing side-specific wOBA) and `bucketed` (shrunk toward league average by PA bucket). `src/analysis/c1_trailing.py`.
- **Alternatives:** All-prior-seasons window (rejected on measurement — barely helps low-exposure hitters, costs veterans data). A single variant, shrunk or unshrunk (rejected either way: collapses the decomposition the raw/bucketed/C.2 sequence is meant to show).
- **Rationale:** Window chosen by measurement, not preference. The two variants together isolate the value of shrinkage itself before C.2 tests doing it properly.
- **Revisit if:** the raw variant's low-exposure advantage turns out to be an artifact of eval-season target noise rather than real ordering signal.

---

## 2026-07-27 — Noise-floor deconvolution added as a companion to the claim-1 metric
- **Decision:** `src/analysis/claim1_eval.py` now reports a noise-floor and deconvolved model-RMSE alongside the frozen §5.2 PA-weighted RMSE, plus a skill-score helper. Additive only — the frozen metric is unchanged.
- **Alternatives:** Raw RMSE alone as §5.2 specifies (rejected: see Rationale). Estimating the floor by simulation (rejected: the analytic form is exact and free).
- **Rationale:** The held-out target is itself a small-sample measurement, so errors add in quadrature; on the eval frame, target noise dominates RMSE enough to compress real model differences into what reads as rounding. Independently validated against B.1's separately-estimated between-hitter signal variance.
- **Revisit if:** a model's raw RMSE lands materially below its estimated floor in any stratum.

---

## 2026-07-27 — Pitchers' own at-bats excluded from every hitter-talent quantity
- **Decision:** `src/data/eval_targets.py` gains a filter dropping batters who are primarily pitchers per season (by batters-faced vs PA, so two-way players stay hitters). Applied to every hitter-talent quantity — B.1, C.1, claim1_eval — but not to the evaluation target table itself, which stays built from the complete source per the two-table principle (2026-07-15).
- **Alternatives:** Filtering in `clean.py` (rejected: wrong table, and the affected quantities bypass it by design). A career-level or PA-only rule (rejected: misses role changes / drops genuine call-ups).
- **Rationale:** The eval-target table contained pre-2022 NL pitchers batting, a low-wOBA population that inflates between-hitter signal variance in any prior or league average. Found via a variance diagnostic, not the test gates. Correcting it moved B.1's stabilization points and widened the process-vs-outcome gap; C.1 stratum boundaries also shifted.
- **Revisit if:** a future season reintroduces pitchers batting in volume, or the two-way threshold misclassifies a genuine two-way player.

---

## 2026-07-27 — C.2 estimand: shrink the two sides jointly, not the platoon split
- **Decision:** C.2 is bivariate empirical Bayes over (talent vs LHP, talent vs RHP) per batter type L/R/switch, with cell-specific variance and the cross-side covariance estimated only on durable hitters. `src/analysis/c2_bivariate_eb.py`.
- **Alternatives:** Split-level (The Book) rejected as the estimator, kept as a scored reference — its variance requires subtracting an unstable noise term. Rate-level (ρ=0) rejected as indefensible, kept as the nesting gate. Unrestricted covariance rejected on measurement — unstable below ~50 PA.
- **Rationale:** The two parameterizations are the same model in rotated coordinates, so the choice is purely identifiability — PA vs LHP and vs RHP are disjoint, so the joint form needs no noise subtraction where the split form does.
- **Reference:** Efron & Morris (1972); the multivariate Fay–Herriot EBLUP; The Book (constants sourced, functional form not independently verifiable).
- **Revisit if:** the ρ interval tightens enough to separate our estimate from The Book's implied value, or the durable-hitter cut needs revisiting.

---

## 2026-07-28 — C.3 design: hitter x context aggregates, two feature sets, inner-val season
- **Decision:** C.3 is an XGBoost model at the unit claim-1 scores — one row per (batter, target season, pitcher hand) — with features built from the hitter's own prior-window rates sliced by context. Two feature sets are fitted and reported: `outcome` (trailing wOBA/PA only) and `full` (adds process signals). Same 3-season window as C.1/C.2; pre-registered hyperparameters; early-stopped on 2023 only, then refit on all seasons at that round count. `src/analysis/c3_gbm.py`.
- **Alternatives:** The 48-dim context vector as features (rejected: carries no hitter identity — B.2's own finding). Early stopping or hyperparameter search on the eval frame (rejected: both leak into the frame every Phase C result is scored on). A single feature set (rejected: can't isolate where any skill comes from).
- **Rationale:** The plan specifies the model but not the row unit; the metric settles it. The outcome/full pair is the first hitter-level test of whether process signal actually projects, not just stabilizes.
- **Revisit if:** Phase D needs a tuned-GBM upper bound rather than a pre-registered one.

---

## 2026-07-28 — C.2's prior mean stays exchangeable; the confound is reported, not corrected
- **Decision:** C.2 is left exactly as logged 2026-07-27 — a single exchangeable prior mean per (batter type, pitcher hand). No exposure-conditional prior is built; the resulting confound in the C.2/C.3 comparison is documented instead of corrected.
- **Alternatives:** Adding an exposure-conditional C.2 variant (rejected as scope — Phase C's job is to produce the incumbent bar, not iterate on it).
- **Rationale:** Prior exposure correlates with talent, so C.2 systematically over-predicts low-exposure hitters where C.3 (which has exposure as a feature) does not. An oracle-recentering check shows most of C.3's apparent low-exposure RMSE advantage is this level effect rather than real ordering skill — reported as a limitation, not fixed.
- **Revisit if:** Phase D's margin over C.3 turns out to be the same level effect, requiring an exposure-conditional prior for a fair comparison.

---

## 2026-07-29 — Ordering claims get a paired bootstrap, and the resampling unit is the batter

- **Decision:** `claim1_eval.paired_rank_difference` is added as the rank counterpart of `paired_rmse_difference`, and both now resample BATTERS rather than (batter, hand) rows (`batter_clusters` / `_resample`, 9 new gates). No model-vs-model ordering claim is made from two bare rank correlations. `c_report.py` emits `results/phase_c/c_rank_paired.csv` covering the same three head-to-heads the RMSE tables cover.
- **Alternatives:** Leaving ordering claims unquantified (rejected: it held the noisier of the two frozen metrics to a lower evidentiary bar than the calibration metric, and the Phase D gate was about to be set on an unquantified difference). A permutation test on the rank difference (rejected: the paired bootstrap is already the project's idiom for exactly this comparison and reuses the same eval frames). Keeping row resampling and only fixing the docstring (rejected: a batter's vs-LHP and vs-RHP rows share his talent, health, and park, so they are not two independent draws — the assumption should be correct rather than merely disclosed).
- **Rationale:** The argument `paired_rmse_difference` was written on — two absolute numbers scored against the same noisy answer key do not resolve a difference, only the paired difference does — is metric-agnostic, and rank correlation is the *more* noise-exposed of the two because ranks carry no PA weight. Applying the argument to the rank metric overturns a headline: the low-stratum C.3-vs-C.2 ordering gap is -0.068 with CI [-0.176, +0.050], so **neither model demonstrably orders low-exposure hitters better** and the reported 0.169-vs-0.102 "disagreement between the two frozen metrics" is not established. Two claims survive the same treatment and are now quantified for the first time: C.2 beats C.1-bucketed on ordering in every stratum (low +0.147, CI [+0.021, +0.276], 99.4%), and C.3's process features improve ordering across the frame (+0.025, CI [-0.001, +0.053], 96.9% — directional). The clustering correction moves the RMSE intervals only slightly (C.3-vs-C.2 low CI [-0.00484, -0.00051] -> [-0.00493, -0.00043], still excluding zero), so no RMSE conclusion changes; the change was made for correctness, and its small size is itself the reportable result.
- **Reference:** none — internal consistency with the 2026-07-27 `paired_rmse_difference` entry and its Diebold-Mariano framing. Cluster/block bootstrap for grouped observations is standard; no project-library source was consulted.
- **Tier:** 1 (evaluation-validity: a claim reported without the interval the project's own standard requires).
- **Revisit if:** a future eval frame has a materially different rows-per-batter structure (both hands scored for nearly every hitter, or nearly none), which would change how much the clustering matters.

---

## 2026-07-29 — MIN_EVAL_PA logged, broken out per stratum, and reported as a sensitivity

- **Decision:** The 25-PA eval-season floor (`claim1_eval.MIN_EVAL_PA`, in place since 2026-07-27 but never logged) is kept as the headline threshold. It gains two obligations it did not have: `build_eval_frame` now reports the drop **per stratum** with the dropped hitters' observed wOBA (`stratum_coverage`, in `c_coverage.json`), and `c_report.py` re-scores the headline C.3-vs-C.2 comparison across `MIN_EVAL_PA_SENSITIVITY = (10, 25, 50)` into `results/phase_c/c_min_eval_pa_sensitivity.csv`. Strata are attached before the filter, on prior exposure, so dropped groups are attributable without leakage.
- **Alternatives:** No floor at all (rejected: a 5-PA observed wOBA is essentially a coin flip and scoring against it measures the answer key, not the model). Inverse-probability weighting on P(eval PA >= 25 | prior exposure) instead of a hard cut (rejected as disproportionate: the sensitivity sweep answers the same question for a fraction of the machinery, and can be revisited if the sweep ever showed instability — it does not). Reporting only the aggregate drop count (rejected: it reads as neutral hygiene and hides a filter that is anything but uniform).
- **Rationale:** The filter censors on eval-season playing time, which is decided AFTER the projection and partly BY the hitter's performance — the same deployment/selection bias the module already cites as its reason not to stratify on held-out PA. Measured, it removes **36.6% of the low stratum** (n=170, mean wOBA .224 against the .278 of those kept) versus 6.2% of the high, so it trims the headline stratum's worst performers. Three consequences are now measured rather than assumed: (1) **the RMSE headline is not a filter artifact** — C.3-full's low-stratum margin is -0.0031 / -0.0026 / -0.0025 at thresholds 10 / 25 / 50, sign and magnitude stable, CI excluding zero at 10 and 25 (the 50 CI grazes zero on n=195, which is power, not instability); (2) **the ordering comparison is unstable in sign** — the low-stratum rank difference runs +0.015 / -0.068 / -0.003 across the same thresholds, reversing direction, which independently confirms the 2026-07-29 rank-bootstrap finding that no ordering claim is supported there; (3) **the level bias is understated by the filter, as predicted** — C.2's low-stratum over-prediction is +0.0188 / +0.0169 / +0.0121 across the thresholds, shrinking as the cut gets stricter and removes more weak hitters. The "81% of C.3's margin is a level correction" figure is likewise a point on a curve spanning 57–81%; the qualitative claim holds at every threshold, the precise number should not be quoted as stable.
- **Reference:** none — judgment call, with the sensitivity sweep standing in for a source.
- **Tier:** 1 (evaluation validity: an undocumented filter conditioning on a post-projection outcome, upstream of every Phase C number).
- **Revisit if:** a future eval frame's low-stratum drop share moves materially away from ~37%, or the sweep ever shows the RMSE margin changing sign across the range.

---

## 2026-07-29 — MIN_EVAL_PA frozen at 25 for the remainder of the project

- **Decision:** `MIN_EVAL_PA = 25` is frozen as the scored eval frame for every remaining claim-1 number, Phase C and Phase D, including the final 2025 test-season run. The sweep over (10, 25, 50) stays as a committed robustness report; it is never the place a headline threshold is chosen from. Extends the same-day entry above, which logged the filter's properties.
- **Alternatives:** Moving the headline to 10 (rejected — see Rationale; the principled case for it is real but it is now unusable). Moving to 50 (rejected: censors >58% of the low stratum, and n=195 costs the power the low-exposure claim depends on). Metric-specific thresholds, a looser frame for PA-weighted RMSE and a stricter one for unweighted rank (rejected: the two metrics would no longer score the same groups, which breaks the paired machinery's identical-group requirement and makes the two headline numbers non-comparable).
- **Rationale:** 25 entered in commit `dc3f2de` (2026-07-24) alongside `claim1_eval.py` itself — three days before C.2 existed and four before C.3. It is therefore pre-registered with respect to every comparison it now governs, which is the property that makes it defensible; it is not claimed to be optimal. The case against moving is decisive: the sweep shows min_eval_pa=10 gives C.3 its LARGEST low-stratum margin (-0.0031 at 99.55%, against -0.0026 at 99.35%), so adopting it after seeing that table would be selecting the evaluation frame that most flatters the model under test. A principled story for 10 is available (it retains 29% more of the low-exposure population the thesis targets, censors less on a post-projection outcome, and PA-weighting already discounts 10-24 PA rows), and that is exactly the problem — the story would be constructed after the fact. Nothing is lost by staying: the margin's sign is stable across the full range, so no conclusion depends on the cut.
- **Consequences carried forward:** (1) the 36.6% low-stratum drop is a standing limitation reported wherever a low-stratum result is reported, not a defect to be fixed; (2) level-error figures are lower bounds (C.2's low-stratum bias is +0.0169 on this frame, ~+0.019 at the least-censored one); (3) the oracle level share is quoted as the 57-81% range, never as 81%.
- **Reference:** none — judgment call, resting on the commit-order evidence above.
- **Tier:** 1 (evaluation validity / frozen-rule discipline: post-hoc frame selection).
- **Revisit if:** never for this project. The rank metric's instability under the cut is a METRIC problem (unweighted ranks give a 26-PA coin flip the same vote as a 400-PA hitter), not a threshold problem, and any fix belongs to Phase D as a §5.2 amendment with its own entry — raising the cut to mask it is explicitly not the remedy.

---

## 2026-07-29 — MIN_EVAL_PA moved to 10; supersedes the freeze at 25 logged the same day

- **Decision:** `MIN_EVAL_PA = 10`. This reopens and replaces the "frozen at 25" entry above, which stands as the record of the reasoning it overrides. 25 and 50 remain in `MIN_EVAL_PA_SENSITIVITY`, so every headline is still reported across a 3x swing in censoring.
- **Alternatives:** Holding 25 on pre-registration grounds (the position argued in the superseded entry — rejected below). Reporting 10 and 25 as co-primary with claims required to hold at both (rejected as unnecessary once 10 is primary and the sweep is committed; it is the same evidence with more machinery). Dropping the filter entirely (rejected: a 3-PA observed wOBA is a coin flip and scoring against it measures the answer key).
- **Rationale (Nate's, recorded in his terms):** the cut is being lowered "because we want to include the guys who have less games played since that's our target audience." This is a construct-validity argument, not a results argument: at 25 the filter removed 36.6% of the low-exposure stratum, and low-exposure hitters are the object of study, so the scored frame was measuring a materially different population from the one claim 1 is about. At 10 that drop is 18.3%. The population recovered is not marginal — 85 low-stratum groups with a median 16 PA over 9 games, mean wOBA .249. In side-specific terms 10 PA is a median ~8 games of exposure against that hand, not 2-3, because a hitter takes only 1.6 (vs LHP) to 2.0 (vs RHP) PA per game against a given hand; 25 PA was a median 12-16 games, which is over three weeks of a call-up and a high bar for the population the thesis serves.
- **Known hazard, stated rather than left to inference:** C.3-full's low-stratum margin over C.2 is LARGER at 10 than at 25 (-0.0031 vs -0.0026, 99.6% vs 99.4%), and this was measured and reported BEFORE the cut was changed (`c_min_eval_pa_sensitivity.csv`, committed under the superseded entry). Adopting a threshold that flatters the model under test is a real hazard regardless of motive, and the mitigation is that 25 and 50 stay in the committed sweep: the RMSE claim is shown to hold at every threshold in the range, so it does not rest on this choice. The ordering claim fails at every threshold and reverses sign between them, which the change does not affect.
- **Reference:** none — judgment call, with the sensitivity sweep and the per-hand PA/game measurements as evidence.
- **Tier:** 3 (scope/population definition — the user's call, taken after the Tier 1 selection hazard was raised in writing and answered on substance).
- **Revisit if:** never for a better score. Reopen only if the low-stratum drop share at 10 moves materially away from ~18% on a future eval frame, or if the RMSE margin's sign ever becomes threshold-dependent in the committed sweep.

---

## 2026-07-29 — Near-plate spray artifact nulled at label time (logs a decision taken 2026-07-22)

- **Decision:** `|spray| > 90` is nulled in `labels.py` at label time (`SPRAY_ABS_MAX = 90.0`), so the spray label carries no survivors. This records a decision actually taken on 2026-07-22 and left in a source comment; it discharges the "Revisit if" clause of the 2026-07-17 spray-angle entry ("the near-plate artifact needs clipping, decided in Phase B"), which until now was open in the log while closed in the code.
- **Alternatives:** Clipping to the limit rather than nulling (rejected: a clipped value is a fabricated measurement at exactly the boundary, and the artifact is a coordinate failure, not a real extreme pull — n* 81.7 vs 77.4 also favours nulling). Keeping the raw angle (rejected: physically impossible values, n* 88.4). Deciding it downstream per-analysis (rejected: guarantees drift between analyses).
- **Rationale:** A fair ball lies inside the ~90-degree foul-line wedge, so `|spray| > 90` cannot be a real batted-ball direction — it is the angle formula blowing up when hit coordinates sit near the plate origin. The physical argument is the reason; the reliability gain is corroboration, not justification. Measured on the 2026-07-29 run (`results/phase_b/spray_clipping.csv`), the artifact is **9,575 of 1,004,409 in-play balls (0.95%)**, confirming the "~1%" asserted in July but never previously verified against a committed number, and VC n* at r=0.5 runs 88.4 unclipped -> 81.7 clipped -> 77.4 nulled. The intervals overlap heavily ([78.6, 98.9] vs [68.6, 87.0]), so the gain is directional, not established. EV and LA come from launch tracking rather than hit coordinates and stay valid on these rows, so only spray is nulled.
- **Correction carried by this entry:** the figure previously cited for this decision, "n* 82 -> 73", was honestly measured on 2026-07-21 while the clip still lived downstream of the check. Moving the clip into `labels.py` on 2026-07-22 silently turned `b1_report.spray_clipping` into a no-op — it began comparing three copies of the already-nulled data — so the 2026-07-27 regeneration overwrote the artifact with three identical rows (n* 77.4278, n_bbip 994,834 in all three, including the treatment whose job is to delete rows) while the citation still pointed at it. The check now rebuilds the pre-clip angle from hit coordinates and asserts it can still see artifacts (`b1_report.unclipped_spray`, `tests/test_b1_report.py`, 5 gates). The level moved 82->88 and 73->77 because the 2026-07-27 pitcher-batter exclusion changed the population every n* is estimated on.
- **Downstream:** nothing changes. The B.1 panel and the PA-equivalent ranking read the `spray` LABEL, which is the nulled treatment, so the committed n* 77.4 / 114.5 PA-equiv were already correct. Spray remains the slowest process signal under every treatment (130.6 PA-equiv unclipped vs 114.5 nulled, against wOBA's 226).
- **Reference:** abdwr3e App. C, BGSU, Weise for the coordinate formula and origin constants (2026-07-17 entry).
- **Tier:** 1 (a frozen decision was reopened and resettled in a source comment rather than the log).
- **Revisit if:** a future Statcast revision changes the hit-coordinate origin, which would move the artifact population.

---

## 2026-07-29 — B.2 context screen: protocol logged, and the six flagged features recorded as UNDECIDED

- **Decision:** Log B.2's screening protocol and result, which produced eight committed artifacts on 2026-07-22 and never received a decision-log or lab-notebook entry. The screen fits one XGBoost head per process outcome (swing / whiff / EV / LA / spray) on the exact 48-dim context vector the DL tower will consume, trains on TRAIN, early-stops on VAL, and measures **out-of-sample permutation importance on VAL**. A feature is KEEP if it clears 1% of a head's baseline metric for at least one head or is frozen-in (count, stand, p_throws); otherwise FLAG. **Nothing is auto-dropped** — B.2's own design defers the feature decision to the B.3 DL common-window ablation, on the ground that a GBM null does not prove a different model class cannot use a feature.
- **Result:** 10 KEEP (plate_x .77, plate_z .68, count .31, pfx_x .14, pfx_z .10, stand .06, pitch_type .04, release_speed .03, release_pos_x .01, p_throws .01) and **6 FLAG** — effective_speed (.0081), release_pos_z (.0075), spin_axis (.0054), release_spin_rate (.0019), release_pos_y (.00064), release_extension (.00064).
- **STATUS OF THE SIX FLAGS: undecided, orphaned, and blocking.** `b2_screen.py` defers them to "the DL common-window ablation (B.3, §4)". That deferral does not resolve, for two independent reasons. First, **it names the wrong step**: plan §3 Phase B step 3 (the common-window ablation) is v1 *with and without BAT-TRACKING* on an identical mid-2023->2025 window — it was never about the 48 context features. Second, **that step no longer exists**: Phase B steps 3, 4 and 5 are all bat-tracking placement, and all three were dissolved when bat-tracking was excluded from v1 (Nate's call, lab notebook 2026-07-20). Skipping them was correct once that call was made; the side effect nobody noticed is that B.2's deferral target went with them. §4 still binds the six — it names "context features" explicitly as ablation-decided — but **no phase step currently owns that ablation**, so it has to become a Phase D context-tower ablation or it will not happen. The six sit at `frozen_in = False` with no disposition and Phase D would inherit them silently. Frozen rule #2 ("flag unclear ablation status before entering the model") is not satisfied.
- **Two reasons the flag list must not be acted on as-is:**
  1. **The fits were budget-capped, not converged.** `n_estimators=600` with `early_stopping_rounds=30`, and `b2_screen_summary.json` reports `best_iteration = 599` for four of the five heads (spray, at 522, is the only one where early stopping actually fired). The models were still improving when the budget ran out, so importances were read off under-trained fits — which understates features that only pay off in later, finer splits, plausibly the marginal ones here.
  2. **Permutation importance is deflated by collinearity, and the flag list is exactly the collinear set.** effective_speed is a deterministic function of release_speed (KEEP) and release_extension; release_extension is its other component; spin_axis and release_spin_rate are redundant with pfx_x / pfx_z (both KEEP), the movement they produce — the spin-vs-movement redundancy already noted in the 2026-07-15 lab entry. Permuting one member of a correlated pair lets the model recover the signal from its partner, so a genuinely informative feature can score near zero. Flagged as **unverified — parametric**: this is a standard property of permutation importance but was not checked against a library source.
- **Cheap test that settles it (Tier 2, proposed not run):** re-run the screen with converged fits (raise the cap until early stopping fires on every head) and permute correlated features **jointly as blocks** — {release_speed, effective_speed, release_extension} and {release_spin_rate, spin_axis, pfx_x, pfx_z} — alongside the existing per-feature permutation. A block that matters while no member does individually is redundancy; a block that does not matter is genuine irrelevance and the drop is defensible. Reuses the existing harness.
- **Alternatives:** Dropping the six now on the B.2 evidence (rejected: both defects above, plus B.2's own stated position that a GBM null is not a DL verdict). Keeping all 48 silently (rejected: that is the current de-facto state and is exactly what this entry refuses to let happen unremarked).
- **Reference:** none for the flag list. The screen's own protocol notes (b2_screen.py docstring) are the source for the discipline described here.
- **Tier:** 2 (empirically resolvable — the test above settles it).
- **Revisit if:** resolved by the block-permutation re-run, or superseded if Phase D runs a context-tower ablation on the claim-1 metric directly, in which case that ablation is the verdict and this entry closes against it.
- **Related gap found while writing this:** the bat-tracking exclusion that dissolved Phase B steps 3-5 is recorded only as a parenthetical in the 2026-07-20 lab-notebook entry and has **no decision-log entry of its own**, despite being the architectural call that cancelled three planned measurement steps and converted "feature value vs. history depth is empirical" (manifest) into a judgment call. Needs its own entry in Nate's words.

---

## 2026-07-29 — One PA unit for scoring: the wOBA denominator, not total plate appearances

- **Decision:** Every quantity in `claim1_eval` that expresses "how much this group's observation is worth" now uses the wOBA **denominator** (PA minus IBB, SH, INT) rather than total PA: the `min_eval_pa` filter, the `pa_weighted_rmse` weight, the noise-floor weight, and the paired bootstrap. `prediction_bias` and `oracle_debias` in `c_report` follow, and `c1_trailing.trailing_pa` (which sets the shrinkage bucket) switches too. `score()` now reports **both** `pa` and `scoring_pa` so the gap is visible rather than inferred. 3 new gates.
- **Alternatives:** Standardising on total PA instead (rejected: `pa` does not govern the precision of the thing being weighted). Leaving the mixture and documenting it (rejected: a units boundary inside the referee is the kind of drift that produced the spray-clipping no-op).
- **Rationale:** wOBA is numerator/denominator, so `Var(observed wOBA) = within-group variance / denominator`. The denominator is therefore the count that sets an observation's precision, and it was already what `sampling_noise`, `prior_exposure` and the stratum boundaries used (the latter because B.1's n* was estimated in denominator units). The filter and the RMSE weight used total PA, which is ~1% larger. Small, but **systematic rather than random**: intentional walks accrue to the best hitters and sac bunts to the weakest, so total-PA weighting slightly over-weighted the tails relative to how precisely they were measured. It also removed a units boundary between the referee and the models it grades — C.1's `trailing_woba`, C.2 (whose `_fitting_window` filters `in_denominator` before observations are formed) and C.3 (`season_outcome`, `_weights`) were all already denominator-based.
- **Effect: negligible, as expected and as predicted before running.** C.3-full vs C.2 in the low stratum moves -0.0030548 -> -0.0030790 (CI [-0.00520, -0.00102], 99.6%); frame-wide -0.0010827 -> -0.0010930. No conclusion, CI, or coverage count changes — the same 130 groups are dropped at the 10-PA cut, since a group needs a non-denominator PA among its first ten to flip. The fix is for correctness and consistency, not because a number was wrong.
- **Correction this entry carries:** an earlier reading of this session held that C.2 was also affected — that `side_observations` computed numerator/TOTAL PA and so estimated a quantity ~0.0017 below the wOBA it is scored against, rising to ~0.015 for the most intentionally-walked hitters. **That was wrong.** `_fitting_window` filters `in_denominator` before `side_observations` runs, so C.2's `x` equals true wOBA and its `n` equals the denominator exactly (verified: max absolute difference 0.00e+00 and 0 across all 1,960 window groups). The erroneous figure came from recomputing the statistic on the unfiltered PA table, which is not what C.2 does. No C.2 change was made and none was needed.
- **Reference:** none — the wOBA denominator convention is the standard one already sourced in the 2026-07-20 eval-target work (FanGraphs weights, league-wOBA reconciliation to +/-0.0005).
- **Tier:** 1 (metric definition inside the frozen scoring function).
- **Revisit if:** never — but any new weight, threshold, or filter added to `claim1_eval` must use `denominator`, and the gates in `tests/test_claim1_eval.py` enforce it.

---

## 2026-07-29 — The Phase C baseline ladder: why each baseline exists, and why its features are what they are

Consolidating entry. The individual baselines are logged above (C.1 2026-07-27, C.2 2026-07-27, C.3 2026-07-28); this records the design of the SET, which was never written down in one place and is the thing a reviewer asks about first.

- **Decision:** Phase C is built as a **decomposition, not a horse race**. Every adjacent pair on the ladder differs in exactly one respect, so the value of each ingredient is priced separately and Phase D's margin can be attributed rather than merely observed.

  | rung | what it is | what the step from below isolates |
  |---|---|---|
  | `no_info` | side-specific league average | reference; its deconvolved error IS the between-hitter talent spread, so skill scores read as share of talent variance captured |
  | C.1-raw | trailing 3-season side-specific wOBA, unshrunk | the hitter's own record, unregularized |
  | C.1-bucketed | same, blended to league average by a step function of PA | the value of shrinkage AT ALL |
  | C.2 | bivariate EB: continuous n/(n+n*) shrinkage + cross-side borrowing via rho | the value of doing shrinkage PROPERLY |
  | C.2 (Book rho) | same machinery at The Book's published constants | the literal incumbent frozen rule #1 names |
  | C.3-outcome | GBM on exactly the features C.2 saw | the value of a flexible FUNCTIONAL FORM, information held fixed |
  | C.3-full | + process features and context slices | the value of extra INFORMATION, model class held fixed |
  | Phase D | conditional-query DL: hitter embedding x per-pitch context | cross-hitter parameter sharing conditioned on pitch context |

- **Why the features are what they are, per baseline:**
  - **C.1 — 3-season window, two variants.** The window was settled by measurement, not preference: on the eval frame it costs the low stratum almost nothing (median prior PA 10.5 -> 10.0, since those hitters' whole careers sit inside it) and only trims veterans. Two variants because the pair decomposes what one hides — and the 2026-07-29 numbers show why this mattered: raw orders low-exposure hitters as well as C.2 does (rank .114 vs .114) while being catastrophically miscalibrated (skill -5.18), and bucketing fixes the level but DESTROYS the order (rank .114 -> .016) because a step function ties everyone inside a bucket. Only continuous shrinkage gets both. A single C.1 variant would have made that invisible.
  - **C.2 — shrink the two sides jointly, not the split.** Same model in rotated coordinates, but identifiable where the split is not: a hitter's PA vs LHP and vs RHP are disjoint events, so Cov(x_L, x_R) estimates the talent covariance with no noise term to subtract, where the direct route must cancel a term ~12x the quantity sought. sigma^2 is cell-specific (LHB vs LHP is .2374, not the pooled .261); rho is estimated only on hitters with >=50 PA both sides, because restricting a correlation to the durable subpopulation is a milder assumption than restricting a variance.
  - **C.3 — hitter-level rows, two feature sets.** The plan names "XGBoost with context-interaction features" without a row unit; the metric's unit settles it, because the 48-dim context vector carries no hitter identity by design (B.2's own finding) and so cannot emit the quantity claim 1 grades. Features are therefore the hitter's OWN prior-window rates sliced by context: pitcher hand, pitch group, two-strike count. `outcome` sees exactly what C.2 saw; `full` adds swing/whiff/EV/LA/spray and their slices. The pair is the point — without it, a GBM win cannot be attributed to the model or to the features.
- **What the decomposition bought (2026-07-29 numbers, low stratum, paired and batter-clustered):** C.1-bucketed -> C.2 is **established** (-0.00422, CI [-0.00649, -0.00202]). C.2 -> C.3-outcome is a **null** — a gradient-boosted model given exactly what empirical Bayes has does not beat it. C.3-outcome -> C.3-full is directional only (-0.00075, 86%). C.2 -> C.3-full is established (-0.00308, CI [-0.00520, -0.00102]) but **76% of it is a level correction** the oracle bound removes. Reading: shrinkage matters enormously, functional form does not pay, extra information pays a little, and most of the headline margin is exposure-conditional level that C.2's exchangeable prior is structurally forbidden from expressing. The C.2 -> C.3-outcome null is the most load-bearing result in the phase — it forecloses "your GBM won because it is a machine-learning model" without us having to argue.
- **Alternatives (set-level):** A single strong baseline instead of a ladder (rejected: a margin over one baseline is uninterpretable — you cannot say what produced it). Dropping C.1-raw as obviously bad (rejected: it is the rung that proves shrinkage fixes level rather than order). An information-matched per-pitch GBM with hitter identity, aggregated through the query machinery, as the C.3 baseline (rejected for Phase C, **logged as a planned Phase D ablation**: its aggregation layer IS Phase D's §1.3 query machinery, so building it as a baseline means front-loading Phase D infrastructure before any DL training — and the question it answers, "does the architecture earn its complexity on identical inputs," is better answered by ablating Phase D against itself than by comparing across model classes, where class and inputs both vary).
- **Rationale for the split of duties:** frozen rule #1 is written in **role-matched** terms — trailing averages, The Book, XGBoost are what a competent analyst does today, and the bar is practical. §4's feature-decision rule is **information-matched** — it isolates components. Both are needed and neither substitutes for the other; Phase C owns the first, Phase D owns the second.
- **Reference:** architecture plan §3 (build order), §4 (feature-decision rule), manifest frozen rules #1-#2. Per-baseline citations in their own entries.
- **Tier:** 1 (the evaluation design the whole thesis is graded through).
- **Revisit if:** Phase D's margin turns out to be the same exposure-conditional level effect the oracle bound isolates here, in which case an exposure-conditional prior mean must be built for C.2 and for the Phase D comparison before any claim is made.
- **Known open at time of writing:** B.2's six flagged context features are still undecided and their deferral target no longer exists (2026-07-29 B.2 entry); C.3's hand-interaction rationale is contradicted by its own importance table and its zero-history collapse is unlogged (item 5b); the enrichment of C.3's context slices to velocity band / location zone / hand x pitch group is proposed and unbuilt (item 5c); no paired test has been run against the Book-rho reference, so frozen rule #1's named EB incumbent has not been formally cleared (item 5a, deliberately deferred).
