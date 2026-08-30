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
- **Decision:** Modeling table is regular season only, dropping position-player pitches, pitchouts, automatic balls/strikes, and bunts. Rows missing or physically impossible on core context are dropped; optional spin context is kept with missingness indicators. Filters apply only to the modeling table, never to evaluation targets.
- **Alternatives:** A minimum-PA hitter floor (rejected: deletes the low-exposure population the thesis targets). A velocity-based position-player rule (rejected: misclassifies hard-armed position players). Dropping spin columns (deferred to a Phase B ablation).
- **Rationale:** Every filter is backed by the profiling notebook; the modeling-vs-target split keeps sharpening filters from biasing ground truth.
- **Revisit if:** Phase B feature screening changes the retained context set, or target construction needs a filtered field.

---

## 2026-07-17 — Contact-quality label domain
- **Decision:** The contact-quality head (EV, LA, spray) is labeled on balls in play only; `ev`/`la`/`spray` are null elsewhere.
- **Alternatives:** Labeling all contact with EV (rejected: fouls carry EV/LA but no spray and no batted-ball outcome, giving ragged masking and no run-value mapping).
- **Rationale:** Fouls are contact for the whiff head but non-terminal count transitions in the Markov composition; only in-play balls carry the run-value outcome the quality head feeds.
- **Revisit if:** the outcome space or run-value mapping (§1.5) changes to need foul-ball measurements.

---

## 2026-07-17 — Spray angle derivation
- **Decision:** Spray = `atan((hc_x − 125.42)/(198.27 − hc_y))` in degrees, mirrored so positive = pull for both hands.
- **Alternatives:** Empirically calibrating the home-plate origin (dropped once constants were sourced); raw field-side angle without mirroring (rejected: not batter-intrinsic).
- **Rationale:** MLB doesn't publish the coordinate origin, so the sourced constants carry a real-data regression guard — field mean ≈ 0 and pull-mean > 0 confirm the export matches scale.
- **Reference:** abdwr3e App. C, BGSU, Weise — three independent corroborations of formula and constants.
- **Revisit if:** the near-plate artifact needs clipping — decided 2026-07-29 below.

---

## 2026-07-17 — Walk-forward split frozen
- **Decision:** Contiguous walk-forward, train 2015-2023 / val 2024 / test 2025, single fold, frozen in `src/config/split_config.json` and validated on load.
- **Alternatives:** Random k-fold (rejected: leaks a hitter's future PAs into his own ID embedding, manufacturing a positive result). Rolling multi-fold (deferred: multiplies compute against the <$200 budget with no gain on the axis we grade on). Val/test gap season (rejected: wastes data, breaks Phase B's same-season window trade).
- **Rationale:** Projection is a forecasting task, so eval must mirror deployment. Freezing before any model comparison pre-registers the held-out season, and contiguity minimizes distribution shift so the metric measures projection skill rather than regime drift.
- **Revisit if:** never for this project (frozen rule); a new fold requires a new entry naming this one.

---

## 2026-07-21 — Stabilization reported at two thresholds (r=0.5 and r=0.7)
- **Decision:** Every stabilization point is reported at both r=0.5, the equal-weight-with-prior point the thesis trades in, and r=0.7, the stricter "reliable measurement" convention.
- **Alternatives:** Single r=0.5 (rejected: invites an unanswered "why 0.5?" and hides that half the variance is still noise). Single r=0.7 (rejected: not the quantity the shrinkage argument uses).
- **Rationale:** The two thresholds answer different questions; reporting both preempts the threshold objection and allows a cross-check against Carleton's published r=0.7 numbers.
- **Reference:** Carleton, "Reliably Stable (You Keep Using That Word)"; FanGraphs, "A Long-Needed Update on Reliability."
- **Revisit if:** the paper's reviewers want a different reliability convention.

---

## 2026-07-21 — Variance-components estimator added alongside split-half
- **Decision:** A one-way random-effects estimator in `stabilization.py` yields an analytic reliability(n), stabilization point, and bootstrap CI over all hitters. Split-half is kept as an independent cross-check; the variance-components point is the headline where they diverge.
- **Alternatives:** Split-half only (rejected: survivorship-biased at large n, exactly where the wOBA outcome lives, and gives no CI). Mixed-model REML (rejected for now: heavier, and method-of-moments ANOVA matches Cronbach's alpha at a fraction of the code).
- **Rationale:** Split-half at large n only keeps hitters who reach that n, so restricting to durable regulars halves the between-hitter signal variance and doubles n*. The variance-components method removes that artifact by using every hitter.
- **Reference:** FanGraphs, "A New Way to Look at Sample Size (Math Supplement)" — Cronbach-alpha signal/noise decomposition.
- **Revisit if:** the divergence on wOBA turns out to be heteroscedasticity in the VC assumptions rather than survivorship in split-half.

---

## 2026-07-21 — Matched-slice and across-time reporting for the process-vs-outcome comparison
- **Decision:** B.1 additionally reports process metrics sliced by pitcher hand, matching the side-specific outcome slice, and a sequential split alongside the random one.
- **Alternatives:** Pooled process vs side-specific outcome only (rejected: apples-to-oranges — some of the gap is the split, not the process/outcome distinction). Random split only (rejected: measures within-sample consistency, which flatters the projection-relevant number).
- **Rationale:** Matched slicing kills the comparison-asymmetry objection, since process stays fast even split by hand. The gap survives every slicing and split choice; only the absolute points move.
- **Reference:** Carleton, "Reliably Stable" — sequential splits drop reliability relative to same-circumstance splits.
- **Revisit if:** the sequential split shows a large systematic across-time degradation on the headline metrics.

---

## 2026-07-27 — C.1 trailing-average design: 3-season window, both shrinkage variants
- **Decision:** Phase C.1 uses a 3-season trailing window in two reported variants: `raw` (unshrunk trailing side-specific wOBA) and `bucketed` (shrunk toward league average by PA bucket).
- **Alternatives:** All-prior-seasons window (rejected on measurement — barely helps low-exposure hitters, costs veterans data). A single variant (rejected: collapses the decomposition the raw/bucketed/C.2 sequence exists to show).
- **Rationale:** Window chosen by measurement, not preference. The two variants together isolate the value of shrinkage itself before C.2 tests doing it properly.
- **Revisit if:** the raw variant's low-exposure advantage turns out to be an artifact of eval-season target noise rather than real ordering signal.

---

## 2026-07-27 — Noise-floor deconvolution added as a companion to the claim-1 metric
- **Decision:** `claim1_eval.py` reports a noise floor and deconvolved model RMSE alongside the frozen §5.2 PA-weighted RMSE, plus a skill-score helper. Additive only — the frozen metric is unchanged.
- **Alternatives:** Raw RMSE alone as §5.2 specifies (rejected: see Rationale). Estimating the floor by simulation (rejected: the analytic form is exact and free).
- **Rationale:** The held-out target is itself a small-sample measurement, so errors add in quadrature and target noise dominates RMSE enough to compress real model differences into what reads as rounding. The floor was independently validated against B.1's separately-estimated signal variance.
- **Revisit if:** a model's raw RMSE lands materially below its estimated floor in any stratum.

---

## 2026-07-27 — Pitchers' own at-bats excluded from every hitter-talent quantity
- **Decision:** `eval_targets.py` drops batters who are primarily pitchers per season, by batters-faced vs PA so two-way players stay hitters. Applied to every hitter-talent quantity but not to the evaluation target table, which stays built from the complete source.
- **Alternatives:** Filtering in `clean.py` (rejected: wrong table, and the affected quantities bypass it by design). A career-level or PA-only rule (rejected: misses role changes and drops genuine call-ups).
- **Rationale:** Pre-2022 NL pitchers batting are a low-wOBA population that inflates between-hitter signal variance in any prior or league average. Removing them moved B.1's stabilization points and shifted C.1's stratum boundaries.
- **Revisit if:** a future season reintroduces pitchers batting in volume, or the two-way threshold misclassifies a genuine two-way player.

---

## 2026-07-27 — C.2 estimand: shrink the two sides jointly, not the platoon split
- **Decision:** C.2 is bivariate empirical Bayes over (talent vs LHP, talent vs RHP) per batter type, with cell-specific variance and the cross-side covariance estimated only on durable hitters.
- **Alternatives:** Split-level (The Book) rejected as the estimator, kept as a scored reference — its variance requires subtracting an unstable noise term. Rate-level (ρ=0) rejected as indefensible, kept as the nesting gate. Unrestricted covariance rejected on measurement — unstable below ~50 PA.
- **Rationale:** The two parameterizations are the same model in rotated coordinates, so the choice is purely identifiability: PA vs LHP and vs RHP are disjoint, so the joint form needs no noise subtraction where the split form must cancel a term far larger than the quantity sought.
- **Reference:** Efron & Morris (1972); the multivariate Fay–Herriot EBLUP.
- **Revisit if:** the ρ interval tightens enough to separate our estimate from The Book's implied value.

---

## 2026-07-28 — C.3 design: hitter x context aggregates, two feature sets, inner-val season
- **Decision:** C.3 is an XGBoost model at the unit claim-1 scores — one row per (batter, target season, pitcher hand) — over the hitter's own prior-window rates sliced by context. Two feature sets are reported: `outcome` (trailing wOBA/PA only) and `full` (adds process signals). Pre-registered hyperparameters, early-stopped on 2023 only, then refit on all seasons at that round count.
- **Alternatives:** The 48-dim context vector as features (rejected: carries no hitter identity, per B.2's own finding). Early stopping or hyperparameter search on the eval frame (rejected: both leak into the frame every Phase C result is scored on). A single feature set (rejected: can't isolate where any skill comes from).
- **Rationale:** The plan specifies the model but not the row unit; the metric settles it. The outcome/full pair is the first hitter-level test of whether process signal projects, not just stabilizes.
- **Revisit if:** Phase D needs a tuned-GBM upper bound rather than a pre-registered one.

---

## 2026-07-28 — C.2's prior mean stays exchangeable; the confound is reported, not corrected
- **Decision:** C.2 keeps a single exchangeable prior mean per (batter type, pitcher hand). No exposure-conditional prior is built; the resulting confound in the C.2/C.3 comparison is documented instead.
- **Alternatives:** Adding an exposure-conditional C.2 variant (rejected as scope — Phase C's job is to produce the incumbent bar, not iterate on it).
- **Rationale:** Prior exposure correlates with talent, so C.2 systematically over-predicts low-exposure hitters where C.3, which has exposure as a feature, does not. An oracle-recentering check shows most of C.3's apparent low-exposure advantage is this level effect rather than ordering skill.
- **Revisit if:** Phase D's margin over C.3 turns out to be the same level effect, requiring an exposure-conditional prior for a fair comparison.

---

## 2026-07-29 — Ordering claims get a paired bootstrap, and the resampling unit is the batter
- **Decision:** `paired_rank_difference` is added as the rank counterpart of `paired_rmse_difference`, and both resample batters rather than (batter, hand) rows. No ordering claim is made from two bare rank correlations.
- **Alternatives:** Leaving ordering claims unquantified (rejected: held the noisier metric to a lower bar than the calibration metric). A permutation test (rejected: the paired bootstrap is already the project's idiom). Row resampling (rejected: a batter's two rows share his talent, health and park).
- **Rationale:** Two absolute numbers scored against the same noisy answer key cannot resolve a difference; only the paired difference can, and rank correlation needs the interval more because ranks carry no PA weight. Applied, it retracts the low-stratum ordering gap between C.3 and C.2, so neither model demonstrably orders low-exposure hitters better.
- **Revisit if:** a future eval frame has a materially different rows-per-batter structure.

---

## 2026-07-29 — MIN_EVAL_PA logged, broken out per stratum, and reported as a sensitivity
- **Decision:** The 25-PA eval-season floor, in place since 2026-07-27 but never logged, now reports its drop per stratum and is swept across (10, 25, 50). Strata are assigned before the filter so dropped groups are attributable without leakage.
- **Alternatives:** No floor (rejected: scoring against a 5-PA wOBA measures the answer key, not the model). Inverse-probability weighting (rejected as disproportionate — the sweep answers the same question). Reporting only an aggregate drop count (rejected: it hides a filter that is anything but uniform).
- **Rationale:** The filter censors on eval-season playing time, decided after the projection and partly by the hitter's performance — the same deployment-bias hazard the module guards against elsewhere, and at 25 it removes 36.6% of the low stratum against 6.2% of the high. Measured, the RMSE headline holds at every cut while the ordering comparison reverses sign across them.
- **Revisit if:** the low-stratum drop share moves materially, or the RMSE margin changes sign across the sweep.

---

## 2026-07-29 — MIN_EVAL_PA frozen at 25 — SUPERSEDED same day, see below
- **Decision:** `MIN_EVAL_PA = 25` frozen for all remaining claim-1 numbers. Superseded by the entry below; retained as the record of the reasoning that was overridden.
- **Alternatives:** Moving to 10 (rejected here, adopted below). Moving to 50 (rejected: censors over half the low stratum and costs the power the claim depends on).
- **Rationale:** 25 entered alongside `claim1_eval.py` itself, days before C.2 and C.3 existed, so it is pre-registered with respect to every comparison it governs. Moving to 10 after seeing that 10 gives C.3 its largest low-stratum margin would be selecting the frame that most flatters the model under test.
- **Revisit if:** superseded below.

---

## 2026-07-29 — MIN_EVAL_PA moved to 10; supersedes the freeze at 25 above
- **Decision:** `MIN_EVAL_PA = 10`. 25 and 50 stay in the committed sweep, so every headline is still reported across a 3x swing in censoring.
- **Alternatives:** Holding 25 on pre-registration grounds (the superseded entry above). Co-primary 10 and 25 (rejected: same evidence, more machinery). Dropping the filter (rejected: a 3-PA wOBA is a coin flip).
- **Rationale (Nate's):** lower the cut "because we want to include the guys who have less games played since that's our target audience" — at 25 the filter removed 36.6% of the low-exposure stratum, so the scored frame measured a materially different population from the one claim 1 is about, against 18.3% at 10. The hazard that C.3's margin is larger at 10 is real and was known before the change; the mitigation is the committed sweep, which shows the RMSE claim holding at every threshold.
- **Revisit if:** never for a better score. Reopen only if the drop share moves materially from ~18%, or the RMSE margin's sign becomes threshold-dependent.

---

## 2026-07-29 — Near-plate spray artifact nulled at label time
- **Decision:** `|spray| > 90` is nulled in `labels.py`, so the spray label carries no survivors. Discharges the 2026-07-17 spray entry's revisit clause.
- **Alternatives:** Clipping to the limit (rejected: a clipped value is a fabricated measurement at exactly the boundary). Keeping the raw angle (rejected: physically impossible values). Deciding it per-analysis downstream (rejected: guarantees drift).
- **Rationale:** A fair ball lies inside the ~90-degree foul-line wedge, so `|spray| > 90` is the angle formula blowing up near the plate origin, not a real direction. It is 0.95% of in-play balls, and since EV and LA come from launch tracking rather than hit coordinates, only spray is affected.
- **Reference:** abdwr3e App. C, BGSU, Weise (2026-07-17 entry).
- **Revisit if:** a future Statcast revision changes the hit-coordinate origin.

---

## 2026-07-29 — B.2's six flagged context features recorded as UNDECIDED
- **Decision:** B.2 screens the 48 context features with one XGBoost head per process outcome, trained on TRAIN, early-stopped on VAL, scored by out-of-sample permutation importance; a feature is kept if it clears 1% of a head's baseline metric or is frozen-in. Ten are kept and six — effective_speed, release_pos_z, spin_axis, release_spin_rate, release_pos_y, release_extension — are flagged and left UNDECIDED. Nothing is auto-dropped. Full importances in `results/phase_b/`.
- **Alternatives:** Dropping the six on B.2's evidence (rejected: the fits were budget-capped rather than converged, and permutation importance is deflated by collinearity, which is exactly what the flagged six are). Keeping all 48 silently (rejected: that is the de-facto state this entry refuses to leave unremarked).
- **Rationale:** A GBM null does not prove a different model class cannot use a feature, so B.2 deferred the six to the DL common-window ablation — but Phase B steps 3-5 were all bat-tracking placement and dissolved when bat-tracking left v1, taking the deferral target with them. §4 still binds the six, so the ablation must become a Phase D context-tower ablation or frozen rule #2 is not satisfied.
- **Revisit if:** settled by a re-run with converged fits and block permutation of the correlated groups, or superseded by a Phase D context-tower ablation. The bat-tracking exclusion that dissolved Phase B steps 3-5 also still needs its own entry in Nate's words.

---

## 2026-07-29 — One PA unit for scoring: the wOBA denominator, not total plate appearances
- **Decision:** Every quantity in `claim1_eval` expressing an observation's weight uses the wOBA denominator rather than total PA — the eval floor, the RMSE weight, the noise-floor weight, and the paired bootstrap. `score()` reports both so the gap stays visible.
- **Alternatives:** Standardising on total PA (rejected: it does not govern the precision of the thing being weighted). Leaving the mixture documented (rejected: a units boundary inside the referee is exactly where drift hides).
- **Rationale:** `Var(observed wOBA) = within-group variance / denominator`, so the denominator sets an observation's precision. Total-PA weighting was only ~1% larger but systematically so, since intentional walks accrue to the best hitters and sac bunts to the weakest; the correction changed no conclusion or interval.
- **Revisit if:** never — any new weight or threshold in `claim1_eval` must use the denominator, and the gates enforce it.

---

## 2026-07-29 — The Phase C baseline ladder is a decomposition, not a horse race
- **Decision:** Every adjacent pair on the ladder differs in exactly one respect, so each ingredient is priced separately and Phase D's margin can be attributed rather than merely observed.

  | rung | what it is | what the step from below isolates |
  |---|---|---|
  | `no_info` | side-specific league average | reference; its deconvolved error IS the between-hitter talent spread |
  | C.1-raw | trailing 3-season side-specific wOBA, unshrunk | the hitter's own record, unregularized |
  | C.1-bucketed | same, blended to league average by a step function of PA | the value of shrinkage AT ALL |
  | C.2 | bivariate EB: continuous n/(n+n*) shrinkage + cross-side borrowing | the value of doing shrinkage PROPERLY |
  | C.2 (Book rho) | same machinery at The Book's published constants | the literal incumbent frozen rule #1 names |
  | C.3-outcome | GBM on exactly the features C.2 saw | the value of a flexible FUNCTIONAL FORM, information fixed |
  | C.3-full | + process features and context slices | the value of extra INFORMATION, model class fixed |
  | Phase D | conditional-query DL: hitter embedding x per-pitch context | cross-hitter parameter sharing conditioned on pitch context |

- **Alternatives:** A single strong baseline (rejected: a margin over one baseline cannot be attributed). Dropping C.1-raw as obviously bad (rejected: it is the rung proving shrinkage fixes level rather than order). An information-matched per-pitch GBM with hitter identity (deferred to Phase D: its aggregation layer is Phase D's own query machinery).
- **Rationale:** Frozen rule #1 is role-matched — what a competent analyst does today — while §4's rule is information-matched and isolates components; Phase C owns the first and Phase D the second. The decomposition's payoff is the C.2 → C.3-outcome null: a gradient-boosted model given exactly what empirical Bayes has does not beat it, which forecloses "your GBM won because it is a machine-learning model" without argument.
- **Reference:** architecture plan §3, §4; manifest frozen rules #1-#2.
- **Revisit if:** Phase D's margin turns out to be the same exposure-conditional level effect, in which case an exposure-conditional prior mean must be built for C.2 first.

---

## 2026-07-30 — Phase D ordering gate: beat both baselines, or claim nothing
- **Decision:** No numeric rank threshold is pre-registered. An ordering claim holds only if Phase D beats both C.2 and C.3-full on rank correlation in the stratum claimed, by `paired_rank_difference` with batter clustering and a 95% interval excluding zero. The RMSE gate is unchanged and set against C.3-full.
- **Alternatives:** Gating at C.2's low-stratum 0.169 (rejected: retracted 2026-07-29, and its sign reverses across the censoring sweep). A non-inferiority gate against C.2 (rejected: a not-worse finding is not the claim the paper makes). No ordering criterion (rejected: leaves the metric open to post-hoc framing).
- **Rationale:** "if the final model significantly outperforms both then that is sufficient." The rule is fully specified in advance even though no threshold is, so it cannot be loosened after results, and it is stricter than the RMSE gate. Rank correlation is unweighted, so a low-stratum null is the expected outcome on this frame and is not evidence against Phase D.
- **Revisit if:** never for a looser bar; transfers unchanged if §5.2 adopts a PA-weighted ordering statistic.

---

## 2026-07-30 — Cold start: unseen hitters get an untrained zero row
- **Decision:** A batter absent from the train vocabulary routes to a reserved embedding index, zero-initialized and never trained. No dropout or unknown-row training enters v1; shrinkage comes from zero init plus weight decay on the embedding table (§2.3), with the §2.1 dimension sweep as the capacity lever.
- **Alternatives:** Frequency-inverse hitter dropout with a trained unknown row (rejected — see Rationale). A freely-learned unknown row (same objection, plus a parameter live at inference that training never touches). Leaving the destination unspecified (rejected: an untouched random row gives unseen hitters a random personality and fails silently).
- **Rationale:** the mechanism "may make the model generalize better but will mute its findings somewhat." It pulls low-exposure hitters toward generic, which is the population and direction claim 1 is about, so a Phase D margin would split between representation sharing and tuned shrinkage — the attribution failure the ladder exists to prevent. This also records a gap in the plan: §2.1 and §2.2 specify no behaviour for a hitter with no training data while §1.4 presumes he has some, and that population is 43% of the low stratum.
- **Reference:** Iyyer et al. 2015 for the rejected mechanism; no source found for cold start in player-embedding work.
- **Revisit if:** the D.7 `‖e_h‖`-vs-`n_h` diagnostic shows low-exposure rows failing to shrink and v1's low-stratum RMSE exceeds C.2's, in which case hitter dropout reopens as a pre-registered ablation with a do-nothing control.

---

## 2026-08-01 — Phase D selection frame: select on 2024, refit through 2024, report on 2025
- **Decision:** Phase D trains on pitches 2015-2023 and decides early stopping and every §4 ablation on the 2024 claim-1 frame, then refits the winning configuration from scratch on 2015-2024 and reports its headline on 2025. The Phase C ladder is re-scored on 2025 for the comparison; only the configuration, including the epoch count, carries across the refit.
- **Alternatives:** An inner claim-1 frame carved from train at 2023, reporting on 2024 and reserving 2025 for Phase F (rejected: Phase F judged unlikely). Reporting a model trained only through 2023, which needs no refit (rejected: it leaves 75% of the 2025 low stratum on the untrained embedding row against 42.7% with the refit, so three quarters of the headline stratum would receive one shared constant). Deciding the ablations on the reported frame itself (rejected: the reported margin would carry the maximum over the configuration search while every Phase C rung carries a single try).
- **Rationale:** Phase F is "unlikely." §2.2 already allocates 2024 as the validation season and 2025 as the final test, and Phase D is the first phase to need a selection budget at all, so the allocation binds here for the first time. Selecting on a season never trained on puts the headline on data no comparison has read, and refitting through that season leaves only report-season debutants without an embedding row — a population no model on the ladder can know, and the same 42.6% every Phase C number was produced under.
- **Reference:** Layer1_Architecture_Plan_v2.md §2.2 (split roles), §2.3 (Phase F gates).
- **Revisit if:** Phase F's gate fires despite being judged unlikely; a new entry must then name this one and fix its evaluation frame.

---

## 2026-08-01 — effective_speed dropped on measured redundancy; five features carry to the D.8 ablation
- **Decision:** `effective_speed` leaves the context tower. The remaining five of B.2's flagged six — `release_extension`, `release_pos_y`, `release_pos_z`, `release_spin_rate`, `spin_axis` — enter D.8 as a single pre-registered block ablation, decomposed by mechanism only if the block fires.
- **Alternatives:** Dropping all six on B.2's permutation nulls (rejected: five are not determined by the kept features, and B.2's own entry flagged those nulls as collinearity-deflated). One ablation per feature (rejected: the group is mutually collinear, so effect sizes add and a single block is the higher-powered test at this frame's resolution).
- **Rationale:** Regressed on the ten features B.2 kept, `effective_speed` reaches R² = 0.993 once `release_extension` is among them, since perceived velocity is velocity adjusted for extension. The other five sit between 0.05 and 0.83, so none is recoverable from what is kept. The test is one-directional: OLS understates what a nonlinear trunk can reconstruct, so a low R² licenses no admission on its own.
- **Reference:** `results/phase_d/d0_redundancy_r2.csv`; discharges the 2026-07-29 B.2 deferral for this feature only.
- **Revisit if:** `release_extension` ever leaves the context tower, since it is what makes `effective_speed` redundant.

---

## 2026-08-01 — Ordering claims move to a denominator-weighted rank correlation
- **Decision:** `claim1_eval` gains a denominator-weighted Spearman coefficient and the Phase D ordering gate reads it. The unweighted §5.2 statistic is retained and reported beside it; `paired_rank_difference` defaults to the weighted form.
- **Alternatives:** Keeping the unweighted statistic alone (rejected: at a 10-PA floor its low stratum is dominated by groups whose observed wOBA is near a coin flip). Replacing §5.2's statistic outright (rejected: it would restate every Phase C rank number under a metric they were not reported with). Deferring until Phase D has a result (rejected: choosing an ordering statistic after seeing the ordering result is the post-hoc framing the 2026-07-30 gate forecloses).
- **Rationale:** Ranks carry no PA, but the correlation computed over them takes weights like any other statistic, so the unweighted form was an assumption rather than a property of the metric. Weighting by the wOBA denominator puts ordering on the same precision weighting RMSE already uses, and makes a rank cumulative plate appearances rather than a position in a roster list.
- **Reference:** Bailey, Emad, Zhang & Xie, "wCorr Formulas" (2023) — the only documented weighted-Spearman implementation and the source of the estimator; it states no commonly accepted coefficient exists. Its weights are inverse-probability sampling weights rather than precision weights, so its consistency results do not transfer.
- **Revisit if:** the two statistics disagree on a Phase C conclusion by more than their paired intervals, which would make the choice of statistic itself a reportable result.

---

## 2026-08-01 — Pitchers' own at-bats excluded from the Phase D training table
- **Decision:** `model_dataset` drops pitches thrown to a pitcher taking his own turn at bat, per season, reusing `eval_targets.primarily_pitchers` so Phase C and Phase D share one definition. Vocabulary falls from 2,486 to 1,762 hitters and the table from 7,347,953 to 7,293,321 pitches.
- **Alternatives:** Keeping them as part of "all MLB hitters" per §2.2 (rejected: that clause exists so stars anchor the quality scale, and pitcher-batters anchor a bottom the eval frame does not contain). A second pitcher-batter definition local to Phase D (rejected: two definitions of the same population drift).
- **Rationale:** `clean.py` filters position players on the pitching side only, so before this they held 35.7% of the embedding table's rows against 1.60% of its pitches — parameters spent on a population `claim1_eval` drops before scoring, and therefore one the query machinery can never ask about. The 2026-07-27 entry already requires their exclusion from every hitter-talent quantity, and a learned table of hitter representations is one.
- **Revisit if:** a future season reintroduces pitchers batting in volume, which the 2026-07-27 entry already names.

---

## 2026-08-01 — v1 loss is the plain likelihood; re-weighting becomes an ablation arm
- **Decision:** The training loss sums raw per-row factor losses with no per-head or inverse-frequency weighting. Per-head means and inverse-frequency weighting within the contact head enter D.8 as two ablation arms.
- **Alternatives:** Per-head means as the default (rejected: a departure from the likelihood, adopted to make the quality heads matter more than their sample supports). Explicit per-head weights (rejected: four more hyperparameters chosen on the selection frame).
- **Rationale (Nate's):** the model should translate baseball rather than be told what to care about. Summing raw losses maximizes the joint likelihood under the §1.2 factorization, so each head's influence tracks the evidence available to it. It is also what D.5 needs, since the conditionals feed a Markov composition that §5.4 requires to reproduce league run scoring, and any re-weighting decalibrates them against the real pitch distribution; B.1's stabilization ranking points the same way, the high-count heads also being the fastest-stabilizing.
- **Revisit if:** a D.8 arm shows re-weighting materially improves low-stratum projection, which would put calibration and representation quality in tension and require §5.4 to be re-checked under the winner.

---

## 2026-08-01 — Trunk and context widths pre-registered, not swept
- **Decision:** Context tower 2x128, trunk 2x256, fixed before any run. Only the embedding dimension is swept, over §2.1's {16, 32, 64}.
- **Alternatives:** Sweeping width and depth at D.8 (rejected: §2.1 names the embedding sweep specifically and gives only ranges elsewhere, and every extra configuration is another draw on a selection frame that cannot resolve small differences).
- **Rationale:** Both sit inside §2.1's stated ranges and follow the convention that a first hidden layer is at least as wide as its input, which the trunk's 160-wide input satisfies at 256. The resulting model is ~207k parameters against §2.1's "well under 1M". Fixing them in advance makes this a stated choice rather than an unremarked one.
- **Revisit if:** the D.6 first run underfits, with training and validation loss plateauing together, making capacity rather than selection the constraint.

---

## 2026-08-01 — Optimizer pre-registered: AdamW, plateau schedule, and batch size tied to weight decay
- **Decision:** AdamW at lr 1e-3, weight decay 1e-2, batch 8,192, `ReduceLROnPlateau` (factor 0.3, patience 1) with early stopping at patience 3, both reading 2024 validation loss. Batch size and weight decay are one setting and neither is swept.
- **Alternatives:** Adam (rejected: its coupled decay scales with gradient magnitude, which varies with a hitter's exposure, so shrinkage strength would depend on exposure uncontrolled). Cosine annealing (rejected: it needs a total step count that early stopping makes unknowable in advance). Sweeping weight decay or batch size (rejected: both are the shrinkage lever, and tuning shrinkage splits a Phase D margin between the §1.4 hypothesis and the tuning — the failure the 2026-07-30 cold-start entry refused).
- **Rationale:** AdamW decays every parameter every step while an embedding row receives a gradient only in batches containing that hitter, so shrinkage is the ratio of the two — about 1:1 for the median hitter and 24:1 at the 10th percentile of exposure. Batch size sets steps per epoch, so it scales that ratio directly and cannot be varied independently of decay.
- **Reference:** Loshchilov & Hutter, "Decoupled Weight Decay Regularization", ICLR 2019 (New Orleans), arXiv:1711.05101; originally circulated as "Fixing Weight Decay Regularization in Adam". **Verified 2026-08-19** against the ICLR proceedings record and the authors' reference implementation. The paper establishes exactly the property this entry's mechanism rests on: decay is applied directly to the weights and is decoupled from the adaptive, loss-based update, so a parameter decays on every step whether or not it received a gradient. That is what makes the shrinkage ratio argument above hold for sparse embedding rows. Not yet in the project library as a note. Settings and shrinkage table recorded in `docs/phase-d-spec.md` §5.
- **Revisit if:** D.7's `‖e_h‖`-vs-`n_h` diagnostic shows low-exposure rows failing to shrink, which reopens the cold-start entry rather than this one.

---

## 2026-08-02 — Phase D tensors stored row-major, and the hitter vocabulary is kept
- **Decision:** `model_dataset.save` writes every array C-contiguous, and `build` records the batter-to-embedding-row map in the manifest. The rebuild under both changes is bit-identical across all eight arrays, so D.1 is reproducible from the labeled parquet.
- **Alternatives:** Converting to row-major inside the Phase D loader (rejected: the 2015-2024 refit rebuilds through `save`, so the layout would return for the run producing the reported headline). Re-deriving the vocabulary downstream (rejected: two definitions of the embedding rows, the drift objection the 2026-08-01 pitcher-exclusion entry sustained).
- **Rationale:** `context` is assembled by column selection and `np.save` preserves memory order, so one pitch's 46 features sat a column apart; gathering a batch of 8,192 rows measured 8.74 s memmapped against 0.033 s row-major. The vocabulary is what joins a trained embedding table back to hitters, which the §5.1 probe, the query machinery, and the D.7 shrinkage diagnostic each require.
- **Reference:** timings measured on the built table (M2, 8.6 GB RAM); both properties gated in `tests/test_model_dataset.py`.
- **Revisit if:** the context tower moves to categorical index columns, which changes what is stored and therefore what the layout costs.

---

## 2026-08-02 — Run value stays out of the training loss
- **Decision:** The loss does not weight errors by the run consequence of the outcome they concern. Run value enters once, in §1.5's mapping from batted-ball characteristics to runs that D.5 consumes.
- **Alternatives:** Weighting every factor's error by run value (rejected on the grounds below). Weighting only the quality heads (rejected: the same objection, and it breaks the chain-rule decomposition that makes the raw sum non-arbitrary).
- **Rationale:** Run-value weighting is not a proper scoring rule, so its optimum is a tilted distribution rather than the true conditional and the model overstates hard contact precisely because hard contact scores more. It also applies run value twice, since D.5 multiplies these conditionals by a separately fit mapping. It is the mechanism the 2026-08-01 entry already refused: the model should translate baseball rather than be told what to care about.
- **Reference:** Gneiting & Raftery (2007), JASA 102:359-378, on strictly proper scoring rules and the uniqueness of the honest report at the optimum; not yet in the project library.
- **Revisit if:** §5.4's composition validation fails in a way traced to the loss rather than to the run-value mapping or the Markov composition, which would put calibration and run-scoring fidelity in tension and require a new entry naming this one.

---

## 2026-08-02 — Ordinal-aware scoring built behind a flag and gated by a promote-only screen
- **Decision:** v1 keeps the log-likelihood loss. The ranked probability score over all five factors, which reduces to the Brier score on swing and contact, is built behind a `rule` flag and screened before D.6 on held-out per-pitch log-likelihood, with reliability and resolution reported beside it. The screen can only promote RPS to a claim-1 ablation, never adopt it. Expected outcome, recorded in advance: null.
- **Alternatives:** Adopting RPS for v1 (rejected: no evidence exists, and reopening a pre-registered objective before the first run is the drift pre-registration prevents). Mixing log loss on the binary factors with RPS on the quality factors (rejected: different units, and normalising RPS by K-1 moves the quality heads' influence by a factor of 23). A five-seed D.8 arm without a screen (rejected: every arm is another draw on the selection frame). Total variation distance as the referee (rejected: not computable from one outcome per pitch).
- **Rationale:** Both rules are strictly proper, so each is uniquely minimised by the true conditional and they differ only in ranking imperfect answers. Free bin probabilities reproduced a bimodal target identically under both; under an imposed capacity restriction each won its own metric while total variation to the truth tied at 0.429, so promotion must require RPS to win on the likelihood's own metric. Frozen rule #2 reserves adoption for claim-1 regardless.
- **Reference:** Gneiting & Raftery (2007), JASA 102:359-378, on strict properness and the locality of the log score; free-head and restricted-head fits measured directly, the free-head case agreeing to 1.5e-03.
- **Revisit if:** the screen shows RPS-trained quality conditionals beating log-trained ones on held-out log-likelihood, or matching them at better reliability and equal resolution, which promotes it to a D.8 arm and puts §5.4's composition validation in scope.

---

## 2026-08-02 — LA and spray are scored only where their conditioning bins were observed
- **Decision:** The launch-angle factor is scored only where both the EV and LA bins are present, and spray only where all three are. 273 rows carrying a valid LA against a masked EV therefore leave those factors, out of 1,266,309 and 1,239,195.
- **Alternatives:** Keeping those rows with an all-zero conditioning vector (rejected: the factor would stop being the conditional it is named after). Conditioning on an imputed EV bin (rejected: silent imputation, which the Phase B missingness rule forbids).
- **Rationale:** The chain factorises as p(ev) · p(la | ev) · p(spray | ev, la), so with EV unobserved the later factors have nothing to condition on and are not part of that pitch's probability. Strict nesting is what keeps the loss the plain likelihood, at a cost of 0.02% of balls in play.
- **Reference:** Layer1_Architecture_Plan_v2.md §1.5 (autoregressive factorisation); counts measured on the built table.
- **Revisit if:** a future outcome dimension carries materially more conditioning-only missingness, at which point dropping rows stops being negligible and an explicit unobserved category earns its own comparison.

---

## 2026-08-02 — ReLU and dropout 0.1 pre-registered from the architecture plan, not swept
- **Decision:** ReLU activations throughout the context tower and trunk, dropout 0.1 on the trunk output only, both taken from §2.1. Neither is swept.
- **Alternatives:** Saturating activations (rejected: their derivative approaches zero away from the origin, so gradient decays through depth). GELU or SiLU (rejected: gains reported on far larger models, an unregistered deviation with no measurement behind it). Dropout at 0.5 (rejected: that value comes from heavily overparameterised networks, and this model carries about 35 training rows per parameter). Dropout on the context tower (rejected per spec §3.2: that vector is observed for most pitches and missingness carries explicit flags).
- **Rationale:** A nonlinearity is what makes the interaction representable at all, since a linear function of the concatenated hitter and context vectors is exactly a hitter main effect plus a context main effect, the §1.4 failure mode the architecture exists to test. Dropout strength is also a regularisation lever, and the 2026-07-30 and 2026-08-01 entries refused to tune regularisation because a margin produced by tuning cannot be credited to the representation-sharing hypothesis.
- **Reference:** Layer1_Architecture_Plan_v2.md §2.1 ("ReLU, dropout ~0.1"); the saturation and dead-unit background is standard and unverified, not in the project library.
- **Revisit if:** the D.6 first run underfits with training and validation loss plateauing together, which the 2026-08-01 widths entry already names as the condition reopening capacity rather than selection.

---

## 2026-08-03 — The bilinear interaction term is built low-rank, and stays a D.8 arm
- **Decision:** The §3.3 interaction term is `W_b (P_e e_h ⊙ P_z z_c)` with rank 32 and no biases: 13,312 parameters, added to the trunk output. It is built now, defaults off, and is measured as a D.8 arm rather than turned on for the first run.
- **Alternatives:** The full bilinear form `e_hᵀ W_b z_c` (rejected: 1,048,832 parameters, which takes the model from 207k to 1.26M and breaks §2.1's "well under 1M" — the interaction term alone would outweigh the rest of the network five to one). Turning it on by default (rejected: frozen rules #1 and #2 reserve every architecture choice for an ablation on the claim-1 metric, and a first run carrying an unmeasured term cannot attribute its margin).
- **Rationale:** The full form learns all 4,096 hitter-by-context pairings independently; the low-rank form spends its budget on 32 shared interaction directions instead, which is the reduced-rank random-slope structure the term is meant to express — each hitter gets context-dependent strengths drawn from a small common set rather than 4,096 free ones. On synthetic data with a planted interaction the rank-32 form recovered 98.1% of interaction variance against the additive model's 92.5% (weak) and 99.7% against 94.9% (strong), and both forms are identical at zero interaction, so the arm cannot cost anything when the failure mode it targets is absent.
- **Reference:** `docs/phase-d-spec.md` §3.3 and §8's interaction-learning risk; parameter counts and recovery fractions measured directly; the reduced-rank / factor-analytic random-slope analogue is standard mixed-model practice and is unverified, not in the project library.
- **Revisit if:** the D.8 arm fires — the interaction term improving the claim-1 metric — which makes rank itself a quantity worth measuring and requires a new entry naming this one.

---

## 2026-08-03 — Phase D runs locally on CPU, one device per comparison set
- **Decision:** Every Phase D training run executes on this machine's CPU, and a comparison set — the five seeds of an arm, and all arms compared against one another — stays on one device and one thread setting. Runs are queued sequentially, never in parallel.
- **Alternatives:** MPS (rejected: 66-81 s/epoch against the CPU's 63-74, a difference inside run-to-run noise). Rented GPU (rejected: the sweep fits in roughly three overnight sessions at no cost, leaving the <$200 budget untouched). Two concurrent runs (rejected: thread oversubscription on 8 cores, which already stalls the test suite the same way).
- **Rationale:** One epoch over 5.88M pitches costs about 70 s with ±15% run-to-run variation, so the device changes nothing measurable. CPU and MPS agree on loss to five decimals after a full epoch but are not bit-identical, since different kernels accumulate floats in a different order — mixing them inside a comparison set would put backend noise into the seed-to-seed spread the ensemble reports as variance.
- **Reference:** nine benchmark runs in `results/phase_d/d3_benchmark.csv`; peak RSS 1.2-1.6 GB of 8.6 GB.
- **Revisit if:** the epoch count or the arm count takes the sweep past what overnight sessions absorb, at which point rented compute is priced against the budget.

---

## 2026-08-03 — Interrupted runs are redone, never resumed
- **Decision:** The overnight driver records only completed runs and redoes anything interrupted. No optimizer, scheduler, or RNG state is ever restored mid-run.
- **Alternatives:** Mid-run checkpoint and resume (rejected: restoring weights without AdamW's moment estimates and both RNG streams produces a run that trains normally and is no longer the run its seed names).
- **Rationale:** At roughly 25 minutes per run, redoing an interrupted one costs less than the reproducibility it would put at risk. The five-seed spread is only interpretable as seed variance if each seed determines its run completely.
- **Reference:** `src/model/sweep.py`; the ledger is `results/phase_d/sweep_log.csv`.
- **Revisit if:** a single run grows long enough that redoing it is expensive, which the refit on 2015-2024 is the first candidate for.

---

## 2026-08-08 — RPS screen returns a decisive null; log loss stays v1's objective
- **Decision:** RPS not promoted to a claim-1 ablation. Scored on 2024: RPS 1.19715 vs log 1.07582 — 126x the 0.00096 seed noise floor. `rule` flag stays in code, defaulting to log.
- **Alternatives:** Comparing arms' own recorded losses (rejected: different units, not comparable).
- **Rationale:** Log score is local (scores only the outcome bin); RPS is distance-sensitive and rewards less-confident probabilities that log score then penalizes. Confirms the pre-registered asymmetry rather than ranking the rules.
- **Reference:** `results/phase_d/screen_scores.csv`; discharges the 2026-08-02 screen.
- **Revisit if:** unchanged.

---

## 2026-08-08 — Reliability/resolution not computed for the RPS screen
- **Decision:** Screen closes on its first promotion clause; no calibration/refinement decomposition built.
- **Rationale:** The second promotion clause (match log loss at better reliability, equal resolution) is unreachable given the 126x margin. That machinery belongs at §5.3's ensemble calibration check instead.
- **Revisit if:** §5.3 is built.

---

## 2026-08-08 — D.5 repertoire and called-strike surface keyed on batter handedness
- **Decision:** Resample real pitch rows grouped by `(pitcher, stand, balls, strikes)`; called-strike model fit as one surface per batter-hand x pitcher-hand cell, not pooled.
- **Alternatives:** Overwrite the `stand` one-hot on a pooled group (rejected: submits pitch rows that never existed). Single pooled surface (rejected, same evidence).
- **Rationale:** RHP pitch mix differs sharply by batter hand (offspeed 20.8% vs LHB, 8.0% vs RHB); pooling would misrepresent both repertoire and take-location by hand.
- **Reference:** Clemens 2025 FanGraphs (usage splits, unreviewed); Deshpande & Wyner 2017 JQAS (precedent for four separate surfaces).
- **Revisit if:** a per-hand cell is too sparse for smoothing to preserve the pitcher's own mix.

---

## 2026-08-08 — Count chain composed by exact solve, not simulation
- **Decision:** 12 count states solved by backward induction over repertoire-averaged transitions; two-strike foul self-loop closed in closed form, `W(b,2) = A / (1 - E[P_foul])`. No simulation, no cap.
- **Alternatives:** Monte Carlo with a cap (rejected: adds noise on top of the 0.00096 floor). Truncated foul loop (rejected: closed form is exact and cheaper).
- **Rationale:** Pitch draw is Markov in count given repertoire averaging, so a linear solve applies with no matrix inversion needed.
- **Reference:** Yonushonis 2011 SABR (foul-loop geometric series); Tenneal 2015 FanGraphs (12-state chain precedent, unreviewed).
- **Revisit if:** within-PA pitch sequencing enters the repertoire.

---

## 2026-08-08 — 24³ quality chain enumerated exactly, not sampled
- **Decision:** `E[wOBA|BIP,h,x]` computed as the exact enumerated sum over all 13,824 bin combinations, via one trunk forward pass + broadcast.
- **Alternatives:** Sampling (rejected: enumeration is cheap and exact).
- **Rationale:** Quality heads are linear over `[trunk; onehot]`, so the joint logits are an outer sum; reconstructed logits agree with real forward calls to 9.5e-7 (~1 float32 ULP).
- **Revisit if:** a head stops conditioning via concatenation into a linear layer.

---

## 2026-08-08 — Fourth factor splits contact three ways; v1 retrained to carry it
- **Decision:** Add a three-class {foul, foul_tip, in_play} head conditioned on contact; retrain v1. Measured first against a league-average baseline table so the head's effect is isolated.
- **Alternatives:** Binary in_play head (rejected: folds foul tips into fouls, wrongly inflating two-strike survival by ~68k of 1.37M events). Probe on the frozen trunk (rejected: trunk was never trained to preserve this distinction).
- **Rationale:** Contact is really three states, and foul vs foul-tip only diverge at two strikes — exactly the chain's missing distinction.
- **Reference:** Clemens 2025, Baumann 2024, Tenneal 2015 (unreviewed, descriptive only — see 2026-08-12 entry for the real justification).
- **Revisit if:** the retrained arm's claim-1 doesn't separate from the league-table baseline by more than seed noise.

---

## 2026-08-08 — Pitcher population is prior seasons only, weighted by batters faced
- **Decision:** Simulator draws pitchers from 2015-2023 only, weighted by batters faced; no within-window reweighting.
- **Alternatives:** Include eval-season pitchers (rejected: reads the season being predicted). Recency weighting (deferred). Resample a fresh pitcher per pitch (rejected: breaks one-pitcher-per-PA reality).
- **Rationale:** Matches every Phase C rung's information set, keeping the comparison fair.
- **Revisit if:** composition validation shows drift traceable to pool season composition.

---

## 2026-08-08 — Ensemble seeds combined by averaging conditionals
- **Decision:** Average the five seeds' per-pitch conditionals, run one composition on the average.
- **Alternatives:** Average five separate wOBA compositions (rejected as headline: averages a nonlinear functional, a Jensen error; kept as the source of between-seed spread).
- **Rationale:** A deep ensemble's prediction is the mixture of members' predictive distributions — averaging probabilities is the correct mixture.
- **Reference:** Lakshminarayanan et al. 2017 NeurIPS (mixture form; composing through a downstream nonlinearity is this project's extension).
- **Revisit if:** §5.3 needs the functional's dispersion as headline uncertainty.

---

## 2026-08-08 — Called-strike model uses raw plate coordinates, no batter-height normalization
- **Decision:** Fit `p(ball/called-strike/HBP | take)` on `plate_x`/`plate_z`; `sz_top`/`sz_bot` excluded.
- **Alternatives:** Normalize via `sz_top`/`sz_bot` (rejected: those are Statcast-derived from past umpire calls — circular). Statcast `zone` (rejected: coarser function of the same two columns).
- **Rationale:** Normalizing by an umpire-derived zone would regress umpire behavior on itself.
- **Reference:** Deshpande & Wyner 2017 (raw-coordinate precedent); Freiman 2018 FanGraphs (batter height explains R²=0.23 of the low strike vs 0.05 high).
- **Revisit if:** real batter-height data becomes available.

---

## 2026-08-08 — C.2 discharges frozen rule #1's empirical-Bayes baseline
- **Decision:** C.2 (bivariate EB) is the empirical-Bayes incumbent frozen rule #1 requires; the Book-rho rung is the literal published-constants comparator.
- **Alternatives:** Separate split-level Book estimator (rejected 2026-07-27: unstable variance).
- **Rationale:** The two parameterizations are the same model in rotated coordinates, so this satisfies the requirement on the estimand.
- **Revisit if:** the rho interval tightens enough to separate this estimate from The Book's implied value.

---

## 2026-08-08 — D.5's own knobs validated on composition fidelity only, never on claim-1
- **Decision:** Pitcher-pool size, pitches-per-cell, and smoothing strength are pre-registered and validated only against composition fidelity, never tuned on claim-1.
- **Alternatives:** Treat as ordinary §4 ablation knobs (rejected: D.5 produces the claim-1 metric itself, so tuning on it flatters the measuring instrument).
- **Rationale:** Composition fidelity scores against observed league run-scoring, independent of the model's own margin.
- **Revisit if:** a knob changes the Phase D vs C ranking without changing composition fidelity.

---

## 2026-08-08 — Fourth factor retrains all arms together, as ledger stage d9
- **Decision:** All eight D.8 arms rerun with the three-class split head as new ledger stage `d9`; the 30 completed `d8` runs stand unchanged as the pre-split record.
- **Alternatives:** Run only the two never-run arms on the old architecture (rejected: answers a question about a model that doesn't ship). Mix six old + two new (rejected: mixes units in one column).
- **Rationale:** Claim-1 now flows through the split-head model, so the whole table must share one architecture; rerunning all eight costs one overnight session since the retrain rebuilds every arm anyway.
- **Revisit if:** never for comparability — a future factor needs the same treatment.

---

## 2026-08-08 — D.5 scores against the whole pitcher population, not a sampled panel
- **Decision:** Query every 2015-2023 pitcher (weighted by batters faced), 6 pitch rows per cell — full population as default.
- **Alternatives:** 60-pitcher sampled panel (rejected: measured 0.0048 wOBA level shift between two draws vs a 0.033 between-hitter spread).
- **Rationale:** PA-weighted RMSE charges for exactly that panel-level shift; the full pass removes the source instead of shrinking it.
- **Revisit if:** pool size stops fitting a session.

---

## 2026-08-09 — One pitcher per simulated PA, pitches drawn independently per count
- **Decision:** Count chain solved per pitcher from his own 12 cells, results averaged by batters faced afterward; 6 real pitch rows per cell. Pitch draw depends only on count.
- **Alternatives:** Average transition probabilities across pitchers first, solve once (rejected: division by `1-P_foul` makes the orders diverge; no hitter faces the composite). Condition repertoire on prior pitches in the PA (rejected: breaks the Markov property, too few cells).
- **Rationale:** Independence-given-count is what licenses the closed-form solve; the cost is unmodeled pitch sequencing.
- **Reference:** 6 draws inflate the two-strike multiplier by +0.0098 vs ~1.40 baseline, propagating to ~0.001-0.002 wOBA.
- **Revisit if:** level bias survives other diagnoses, making `n_pitches` the next lever.

---

## 2026-08-12 — Composition validation splits into an unscored probe and a scored per-PA check
- **Decision:** Keep the zero-row probe unscored; add a scored check on the four per-PA absorbing rates (BB/K/HBP/BIP) against true handedness-weighted train-window observed rates, each within 2% relative (HBP 20%), credited only if a fix moves a rate beyond between-seed spread.
- **Alternatives:** Rebuild the old check in place (rejected: the zero row isn't a hitter, so it has no meaningful pass condition). Score against eval-season hitters (rejected for now: doubles a run). Keep flat 25% handedness weights (rejected: −0.00271 wOBA error; both weightings now reported).
- **Rationale:** The old check read per-pitch masses at the 0-0 count and couldn't see a walk/strikeout tradeoff; the four absorbing rates are what wOBA actually terminates in.
- **Reference:** observed train-window (n=1,526,308 PA): BB 0.08042, K 0.22338, HBP 0.01049, BIP 0.68571; true-share wOBA 0.31639 vs flat-25% 0.31368.
- **Revisit if:** a fix passes all four rates while composition fidelity and claim-1 disagree.

---

## 2026-08-12 — Third contact class rests on a rule of baseball, not the cited posts
- **Decision:** Restates the 2026-08-08 fourth-factor entry — support is the playing rule (caught foul tip at 2 strikes = strikeout), not the cited blog posts, which speak only to learnability of a hitter-specific foul-tip rate.
- **Alternatives:** Cite a peer-reviewed foul-tip study (rejected: the load-bearing premise is a rule, not an empirical estimate). Amend the original entry (rejected: log is append-only).
- **Rationale:** Separates the mechanical argument (strong) from the posts (weak, different question) so the retrain's justification doesn't rest on the posts being right.
- **Revisit if:** a per-hitter foul-tip rate becomes a feature or claim.

---

## 2026-08-12 — xwOBA enters as a rung and a second answer key, never the primary target
- **Decision:** xwOBA enters three ways only: a C.1 ladder rung, a second answer key scored beside realized wOBA, and an approximate achievable-error floor (RMSE between the two keys). Never the primary target or a gate input.
- **Alternatives:** xwOBA as primary target (rejected: changes the claim, makes ground truth another model's output). Build the rung from `V` marginalized over spray (rejected: inherits `V`'s own defects).
- **Rationale:** The gap between the two keys separates "wrong about batted-ball quality" (fixable) from "unknowable" (fielding, sequencing). Floor is 0.02923 vs Phase D's 0.0492 all-stratum RMSE — ~40% of error is answer-key noise.
- **Reference:** floor by stratum 0.0364/0.0331/0.0252/0.0292; ordering advantage grows under xwOBA (0.5763 vs 0.5389) vs realized (0.4608 vs 0.4417).
- **Revisit if:** a future model gains a fielding-alignment channel.

---

## 2026-08-12 — D.5 level excess: two fixes land, cell-size exonerated, exposure fails talent control
- **Decision:** Count-specific take-surface offsets (12 per surface) land; in-play mass split 96.37/3.63 with the unmeasured share valued at 0.29973. `n_pitches` stays 6; no escalation to 48 count-keyed surfaces. HBP needs no separate fix.
- **Alternatives:** 48 surfaces (declined: pooled surface's marginal error is only +0.0002 over 3.89M takes; offsets already recover the per-count component). Raise `n_pitches` (declined: monotone in the wrong direction — 0.308/0.309/0.309 at M=6/12/24). Reweight `fit_outcome_table`'s sample (rejected: distorts individual cells to fix an aggregate).
- **Rationale:** The unmeasured-category split passes a clean two-sided test (league wOBA −0.00171 vs predicted ~−0.0015, absorbing rates untouched). Offsets close the strikeout gap but not the walk gap. Talent-controlled gradient rules out exposure as the driver; a real but small trunk interaction exists (~1/10 the level bias) — the excess is a level, not that gradient.
- **Reference:** offset effect: BB −0.0017, K −0.0045, BIP +0.0060 (walk gap needs 0.0053 removed, gets 0.0017; K needs 0.0011, gets 0.0045 — overshoots, see 08-18 credit-rule fix).
- **Revisit if:** walk gap survives the Step 4 rebuild — points to take frequency (swing head), not ball-given-take.

---

## 2026-08-12 — Equal-mass binning wins its own contest; the top-bin defect is not a binning problem
- **Decision:** Quality bins stay equal-mass, 24 per dimension, refit once with Statcast placeholder pairs dropped from the edge fit and masked from all three quality targets. Edges frozen in the build manifest.
- **Alternatives:** `top_decile_split` (rejected: narrows top-EV bin 15.10→12.80mph but raises joint within-cell variance 0.1431→0.1451). `variance_min` via exact 1-D DP (rejected, worst performer at 0.1487 — optimizes per-dimension, the wrong objective for a joint-cell prediction). `n_bins=32` (deferred: 2.4x cell blowup, separate decision). Mask placeholders instead of dropping (rejected: still pollutes quantile computation).
- **Rationale:** Pre-registered objective was within-cell realized wOBA variance over the joint 24³ grid. Equal mass wins outright — the top-bin defect can't be fixed by re-binning at fixed `n_bins`.
- **Reference:** 954,223 scored BIP; variance 0.1431/0.1451/0.1487; placeholder drop 3.61% of BIP.
- **Revisit if:** another candidate scheme is proposed, or the high-stratum discrimination loss survives rebuild.

---

## 2026-08-12 — D.5 report gets per-stratum verdicts, a power factor, and a split spread diagnostic
- **Decision:** Report emits a gate verdict for all four strata (not just low); every rank-gap null restated as a power statement (SE + batter multiplier needed); predicted-spread diagnostic splits on training-vocabulary membership, excluding cold-start rows.
- **Alternatives:** Swap which stratum is hard-coded (rejected: reading only one is the defect). Leave low-stratum verdict as flat "null" (rejected: conflates "measured absent" with "unmeasurable on this frame"). Pool cold-start into the spread diagnostic (rejected: their spread reflects only context variation).
- **Rationale:** Each defect misstated confidence, not model behavior — fixes are re-reads of existing files. Splitting the spread diagnostic reverses its direction: cold-start pooling was masking an anti-shrinkage finding.
- **Reference:** shipped-arm low-stratum rank gap +0.0908 (SE 0.0687, z=1.32) needs ~4.5x more batters (~1,074) for 80% power. Trained-row-only spread 0.0374 (low) vs 0.0281 (regulars), 33% wider — vs a misleadingly narrow pooled 0.0305.
- **Revisit if:** an eval frame spans enough seasons for ~1,080 low-stratum batters.

---

## 2026-08-14 — Gradient test (b): the low-exposure embedding is displaced, not shrunk; exposure-conditional prior declines
- **Decision:** Exposure-conditional C.2 prior not built. Gradient (b) run on d10 baseline, 5 seeds, reserved row excluded; reports both embedding norm and its projection onto the wOBA-raising direction, since the two disagree in sign.
- **Alternatives:** Build the prior anyway on the univariate exposure gradient (rejected: exactly the case pre-registration guarded against). Substitute observed 2024 wOBA as the talent proxy (rejected: manufactures the correlation it reports).
- **Rationale:** Both required-fail conditions triggered independently — talent-controlled gradient (a) already ruled out exposure, gradient (c) prices a real but small hitter interaction. Gradient (b) supplies the mechanism: low-exposure rows sit farther from origin (not shrunk in norm) but their wOBA-direction component IS shrunk/displaced toward low predicted wOBA — capacity exists, oriented orthogonally to the scored axis.
- **Reference:** norm slope negative in all 5 seeds (e.g. −0.0186 [−0.0207,−0.0167] per 1k train pitches); projection slope positive in all 5 (+0.0218 [+0.0205,+0.0233]). By exposure quintile, mean norm falls 1.00→0.61 then flattens while projection rises −0.22→+0.07, flipping sign only in the top quintile.
- **Revisit if:** a future arm adds exposure-dependent regularization, or claim-1 shows the low-stratum loss is direction-driven.

---

## 2026-08-15 — claim-1 for all eight d10 arms: four beat baseline, margins small, ledger retired as architecture instrument
- **Decision:** All eight arms scored on claim-1 (paired bootstrap, batter-clustered, decisive stratum `all`). `block`, `bilinear`, `meanweight`, `dim64` beat baseline; `nospray`, `dim16` are nulls. Shipped arm unchanged — `block`'s margin sits inside baseline's own between-seed range, so adoption is deferred to Phase E with a re-scored ladder.
- **Alternatives:** Keep reading architecture verdicts from held-out log loss (rejected: Spearman correlation between log-loss rank and claim-1 rank across 7 comparable arms is exactly 0.000). Pre-register `low` as decisive (rejected 08-12: noisiest stratum). Score `nospray` against uniform spray (rejected: no ball follows uniform). Adopt `block` now (declined: out of phase scope, flagged since the rule does select it).
- **Rationale:** Log loss and claim-1 measure different things — baseline is #1 on log loss, #6 on claim-1; `block` is the reverse. `block` (drops B.2's five release/spin features) wins claim-1 while losing log loss: those features help next-pitch prediction, not hitter projection.
- **Reference:** margins (negative favors arm): `block` −0.00078 [−0.00107,−0.00048], `bilinear` −0.00063, `meanweight` −0.00044, `dim64` −0.00028, `nospray` −0.00010 (n.s.), `dim16` −0.00003 (n.s.), `invfreq` +0.01274 (worse). Baseline seed spread in `all` stratum: 0.00186 range — comparable to the whole best-to-baseline spread (0.00078).
- **Revisit if:** Phase E re-scores on the no-block build; or an arm's margin exceeds a single arm's own seed range.

---

## 2026-08-18 — The 2025 final run moves after Phase O; Phase E evaluates on the 2024 frame only
- **Decision:** Phase E computes every validation/effectiveness number on 2024. The 2015-2024 refit and single 2025 report move to the end of Phase O, on whatever configuration Phase O settles. `assert_not_test_season` keeps blocking 2025 outside `--final-run`.
- **Alternatives:** Follow the architecture plan's literal order, report 2025 in Phase E then optimize (rejected: leaves Phase O nothing untouched to prove a gain on). Score 2025 now without a refit (rejected: 75% of the 2025 low stratum would sit on the untrained zero row vs 42.7% with refit — not comparable to Phase C). Refit twice (rejected: second report is test-set reuse regardless of labeling).
- **Rationale:** The plan's own lines conflict — it places the 2025 table before optimization but also requires only the winning configuration cross the refit, which doesn't exist yet at that point. This keeps one refit, one 2025 report, the 42.7% cold-start share, and only moves which phase boundary they sit behind. The manifest makes the decision log authoritative where it conflicts with the architecture plan.
- **Reference:** architecture plan lines 119-120, 167, 169, 175.
- **Revisit if:** Phase O returns no adopted change; or the deadline moves such that a 2025 number is needed before Phase O completes.

---

## 2026-08-18 — The D.5 credit rule is repaired: a paired delta is graded against the spread of the paired delta
- **Decision:** A composition fix is credited when its mean paired difference across seeds exceeds the between-seed spread of that paired difference (not of the raw level, as the old rule compared). Credit verdict and 2%-band result always reported together. Applies Phase E forward; does not retroactively re-credit D.5's published verdicts.
- **Alternatives:** Leave the old rule (rejected: seed effect cancels inside a paired difference but not inside a level, so the old denominator was the wrong order of magnitude — no real fix could ever clear it). Re-credit D.5's offsets now (declined: needs its own entry). Drop the credit rule, keep only the band (rejected: band says nothing about whether a change was real).
- **Rationale:** Both sides of a paired comparison share the seed, so seed variance subtracts out of the difference — grading against an undifferenced level's spread asks the effect to beat noise already removed by pairing. Costs nothing new: the corrected denominator is a re-read of already-persisted per-seed files.
- **Revisit if:** a future arm has fewer than three seeds; or D.5's count offsets are re-graded under this rule (explicitly not done here).

---

## 2026-08-18 — The walk gap is 54% population, 24% composition structure, 0% resampler
- **Decision:** The shipped +6.17% walk-rate fidelity failure decomposes into three parts. E.1's population-matched control accounts for 54% (matched observed 0.08312 vs shipped unmatched 0.08042). E.10 accounts for 51.6% of the remainder (+0.00117) as a model-free property of the independent-pitch count chain. E.7-E.9 (resampler channels) net to ~zero. ~22% remains unowned.
- **Alternatives:** Carry the headline unattributed into Phase O (what D.5 shipped). Stop after E.1. Attribute the residual to the swing head (the handoff's suspect — refuted by E.6, wrong sign).
- **Rationale:** Standing risk gates any ablation reading on attributed composition fidelity. E.10 is load-bearing because it's model-free — a counted-frequency property no retrain can fix.
- **Revisit if:** composition is changed to condition on within-PA history; or the 22% is closed by a later diagnostic.

---

## 2026-08-18 — The swing head is exonerated as owner of the walk gap
- **Decision:** Swing head recorded as calibrated, removed from the suspect list. On 705,344 real held-out 2024 pitches (resampler excluded): predicted swing rate 0.479256 vs observed 0.477975 (+0.27%); every handedness cell within 0.5%.
- **Alternatives:** Treat the handoff's framing as settled, open a Phase O retrain item.
- **Rationale:** A head-owned gap would show a swing shortfall in three-ball counts; measured result is the opposite sign at 3-0 (+8.7%), 3-1 (+0.9%), 3-2 (+2.1%) — too many swings, can't manufacture extra walks.
- **Revisit if:** the head is retrained or the eval season changes.

---

## 2026-08-18 — The walk excess is a spread problem, not only a level problem
- **Decision:** Walk gap treated as claim-1 relevant, not just fidelity. Noise-corrected compression coefficient b=0.642 for walks (vs 0.773 K, 0.736 BIP, 0.095 HBP) — the model expresses only ~64% of true between-hitter spread in walk rate.
- **Alternatives:** Read the naive regression slope of (modelled−observed) on observed (rejected: mechanically compressive even at zero true compression, since target sampling noise sits in the regressor).
- **Rationale:** A pure level bias would leave ranking intact; this is compression of the exact quantity claim-1 measures. Correction uses the known binomial noise variance p(1-p)/n.
- **Revisit if:** the walk gap is closed.

---

## 2026-08-18 — The platoon adoption rule is not met on 2024
- **Decision:** Model's platoon differential not adopted over Route A (overall skill + league-average split). No stratum's paired 95% rank interval excludes zero. RMSE favors the model only pooled (−0.00115 [−0.00216,−0.00017]).
- **Alternatives:** Call pooled RMSE alone "adoption" (rejected: rests the claim on one metric-stratum pair of eight). Defer reading until walk gap fixed (rejected: leaves Phase O with no claim evaluation).
- **Rationale:** 81.7% of the model's predicted differential variance is just the batter-stand main effect (vs 10.9% observed) — the model has moved about a fifth of the way from "apply the league split to everyone" toward a real differential. Small, real, not enough per frozen rule #1.
- **Revisit if:** walk gap closes and C-ladder is re-scored under d10; or the refit changes the low-exposure stratum (currently 42.7% cold-start).

---

## 2026-08-18 — Model confidence tracks exposure, reported unscored
- **Decision:** E.4 calibration recorded descriptively, not as a scored gate. Regression slope of observed-on-modelled: 0.529 low-exposure (z=−4.90 vs 1), 0.664 medium, 0.998 high, 0.841 pooled.
- **Alternatives:** Treat the low-exposure slope as a failure, open remediation.
- **Rationale:** Matches architecture prediction — embedding under-dispersed where hitters lack history, calibrated where they have it. Left unscored: scoring it on a claim-1-adjacent metric would be circular (the scorer is the model).
- **Revisit if:** the declined exposure-conditional prior is ever adopted.

## 2026-08-19 — Platoon variance framing was errors-in-variables biased; the 2026-08-18 entry is superseded
- **Decision:** the "81.7% vs 10.9%" variance-share framing in the 2026-08-18 platoon-adoption entry is withdrawn — it compared a noise-free predicted variance to a noise-dominated observed one, the same errors-in-variables trap E.2 guards against. Corrected share (binomial noise removed) is expected near 50-60%, computed in E.15. No share figure is quotable until then.
- **Alternatives:** leave the figure with a caveat (rejected: leaves a wrong number standing). Withdraw the whole 08-18 entry (rejected: its adoption verdict doesn't depend on the share).
- **Rationale:** the adoption null stands on the paired interval, which was computed correctly — only the "sharpest statement" framing was biased.
- **Reference:** none — standard errors-in-variables attenuation, already used in E.2.
- **Revisit if:** E.15's corrected share lands outside 40-70%, which would also require restating E.5's "real but small departure" reading.

## 2026-08-19 — Platoon skill is a separable talent; the split constant we disagree with is second-hand
- **Decision:** platoon skill is a separable talent, per C.2's model-free estimate: rho=0.652 LHB [0.384,0.902], 0.719 RHB [0.479,0.987], 0.713 SHB [0.093,0.999] (correlation between a hitter's true talent vs LHP and vs RHP). rho<1 is the whole claim.
- **Alternatives:** read the model's shrinkage toward league average as evidence skill is near-absent (rejected: circular — that reads the imposed EB prior back out of its own posterior).
- **Rationale:** separates "platoon skill is real" from "our model measures it well" — evidence says yes to the first, largely no to the second. Recovered within-stand spread is 54% of true for LHB vs 18% RHB, a 3x asymmetry a uniform prior cannot produce.
- **Reference:** UNVERIFIED — our rho disagrees with The Book's implied 0.887 LHB/0.949 RHB, whose constants reach us second-hand via Tango 2009, not the primary source. Must be checked against the primary source before this disagreement is asserted in any write-up.
- **Revisit if:** the primary source is obtained; or the 2015-2024 refit changes the posterior.

## 2026-08-19 — The §5.1 probe checkpoint is demoted from early gate to retrospective diagnostic
- **Decision:** architecture §5 item 1's probe, specified as the early detector for the §1.4 failure mode, never ran as a gate. Run instead in E.14 as a retrospective diagnostic — it can explain, not stop, anything now.
- **Alternatives:** run it as originally specified and treat its verdict as a gate (rejected: the runs it would gate are already complete). Skip it (rejected: it's the likeliest explanation for E.5's result).
- **Rationale:** the §1.4 failure mode appears to have fired, and the detector meant to catch it early sat unrun through Phase D — found in evaluation after 39 training runs instead. A checkpoint with no step that forces its execution is not a checkpoint.
- **Revisit if:** a future phase adds training runs, at which point the probe returns as a gate with an explicit forcing step.

## 2026-08-19 — The calibration/refinement decomposition is retired; ensemble interval coverage is not
- **Decision:** the full calibration/refinement decomposition of a proper score, owed since 2026-08-08, is retired unbuilt. The ensemble interval coverage check it was bundled with is kept, discharged in E.14.
- **Alternatives:** build it on claim-1 to close the debt as written (rejected below).
- **Rationale:** the decomposition fits a probabilistic categorical forecast; claim-1 is RMSE on a continuous target, so the machinery doesn't fit the surface. Its reliability half is discharged by the coverage check, resolution half already answered by E.4's slope and the rank correlations. Coverage is kept because it's genuinely untested — architecture §2.1 states calibration on very-low-exposure hitters "remains assumed until §5.3 checks it."
- **Reference:** Gneiting & Raftery 2007 (calibration and sharpness as separate properties) — verified, held locally.
- **Revisit if:** the project ever scores a categorical probabilistic forecast as a headline claim.

## 2026-08-19 — Phase E scope closes at E.15; two items are deferred by name
- **Decision:** Phase E is E.1-E.15. (a) Non-handedness conditional queries (former E.16) move to the frontend phase. (b) The deployment-bias audit (architecture §5 item 5) moves to write-up, method recorded in `phase-e-spec.md` §12.6.
- **Alternatives:** run both here (rejected: (a) architecture §1.3 says pitcher-typed queries are the same machinery with no new code path, so it tests nothing handedness hasn't; (b) needs a cohort build and the Oct 1 deadline doesn't allow it).
- **Rationale:** deferred, not dropped — the audit is the direct answer to the regression-to-archetype objection a reviewer will raise, hardest on the low stratum where 42.7% of hitters share the cold-start row.
- **Revisit if:** a reviewer objection turns on either; the audit should be promoted ahead of the frontend if the low-exposure claim survives E.11.

## 2026-08-19 — Phase Q added to the build order; the only architecture-plan change this window
- **Decision:** a query-dashboard phase (Phase Q) is added to the architecture plan's §3 build order after Phase V, outside the paper's claims — non-handedness conditional queries live there. The only architecture-plan edit made during Phase E.
- **Alternatives:** amend the plan wherever a Phase E finding contradicts it (rejected: Phase E is an evaluation window, not a design window — rewriting the spec during evaluation erases the record of what was designed against).
- **Rationale:** the plan's own §7 already makes the decision log authority over it, so the log running ahead is designed behavior; Phase Q is a genuine addition to the build order, not a correction, which is why it clears the bar.
- **Revisit if:** the dashboard needs a scorer code path after all, contradicting §1.3 and making it a model change rather than a surface.

## 2026-08-19 — E.11: claim-1 re-scored on the D.10 arm; both gates fail, and the ordering null is underpowered
- **Decision:** the claim-1 gate is re-scored on D.10 and adopted as the project's verdict, superseding `results/phase_d/d5_claim1_verdict_phase_d_baseline.json`, which was written against the D.8 arm under a four-key schema since expanded to twelve. Both gates fail in every stratum, including the decisive low stratum.
- **Alternatives:** re-run under the original label (rejected: the frozen rule keeps prior-lane artifacts absent a confirmed bug, and both labels on disk side by side serve a later D.8/D.10 comparison).
- **Rationale:** D.10 is the best low-exposure ranker on the board by point estimate — weighted rank 0.2592 against C.3-full's 0.1665, a paired gap of +0.0927 [-0.0343, 0.2194] — but 914 batters would be needed against the 239 available, so the ordering result is a null with a power factor and not a measured absence. All-stratum RMSE is significantly worse and the high stratum loses on both metrics; removing the level bias does not rescue the claim.
- **Reference:** none — the gate is this project's own, pre-registered 2026-07-30 and 2026-08-01.
- **Revisit if:** the 2015-2024 refit changes the low-exposure population (42.6% cold start), or the deferred deployment-bias audit shows the low-stratum answer key is biased rather than merely noisy.

## 2026-08-19 — E.12: the D.10 ablation table is confounded with each arm's level, and cannot select alone
- **Decision:** the ablation table's PA-weighted RMSE column is confounded with each arm's mean predicted wOBA and is not a ranking of ranking skill; Phase O must not select arms on it alone.
- **Alternatives:** select on weighted rank correlation with a paired interval instead (not adopted here: the decisive-stratum rank spread across arms is 0.0119, which may itself be inside noise, so the replacement needs its own test when Phase O opens).
- **Rationale:** across the seven non-degenerate arms, mean predicted wOBA tracks all-stratum RMSE at r=0.888 (p=0.0076) and decisive-stratum RMSE at r=0.756 (p=0.049, Spearman p=0.215), while tracking rank correlation not at all; between-arm level spread is only 8% larger than the baseline's own five-seed spread. The confound is decisive on the all-stratum column and marginal in the decisive `low` stratum, which is enough to bar selection but not to settle the replacement. Fork-opened: the hypothesis was generated and confirmed on the same data, and the level offset is never subtracted.
- **Revisit if:** a future arm set is scored, testing the confound out of sample.

## 2026-08-19 — E.11b: the claim-1 verdict does not depend on the eval-season playing-time filter
- **Decision:** architecture §5's committed MIN_EVAL_PA sweep is discharged for the D/E headline; both gate verdicts are stable across thresholds 10/25/50.
- **Alternatives:** continue quoting the Phase C sweep (rejected: it hard-codes c3-vs-c2 and had never been run against this headline).
- **Rationale:** the filter conditions on playing time decided after the projection and partly by how the hitter performed, so its neutrality had to be measured rather than assumed; tightening it removes half the low stratum while the low-stratum rank gap grows to 0.1099 without ever becoming significant. One isolated cell flips at threshold 50 and is not the gate.
- **Revisit if:** the 2015-2024 refit changes the low-exposure population.

## 2026-08-19 — E.13: the BIP value gap is era drift in the frozen value table, not the measurement seam
- **Decision:** the +0.015116 BIP value gap is attributed to era drift in V (+0.028836) rather than the D5-R15 measurement seam (-0.016154); the 16.1% residual is below the one-third stop rule, so the gap is treated as explained.
- **Alternatives:** none — this is a measurement.
- **Rationale:** identically-struck balls were worth less in 2024 than under the train-season table, and the seam moves the modelled value the opposite way, so the pre-registered expectation that the seam explained a minority of the gap failed on sign as well as size. The split is not orthogonalised, so the residual absorbs any interaction. The era-drift check is fork-opened, substituted after the pre-registered Jensen test was refuted structurally: V is a lookup table and the composition is linear in the bin distribution, so convexity cannot operate.
- **Revisit if:** the 2015-2024 refit rebuilds V, which should shrink the drift term directly.

## 2026-08-19 — the two tensor builds differ only in the quality bins, and no Phase E result is affected
- **Decision:** `data/processed/phase_d` and `phase_d5` differ only in `quality_bin_edges` and the `ev`/`la`/`spray` index arrays; all other fields are byte-identical, and the differing fields have one Phase E consumer, which already reads `phase_d5`.
- **Alternatives:** flip the stale `--data-dir` defaults in the five modules that still name `phase_d` (rejected: the frozen rule bars prior-lane edits absent a confirmed bug, those five read only byte-identical fields, and `phase_d` lacks `split.npy`).
- **Rationale:** D.10 trained on `phase_d5`, so a module feeding it `phase_d` bin indices would mis-bin model inputs; the exposure was therefore audited field by field rather than inferred from module names. The stale defaults remain a trap for any future module that reads the quality bins.
- **Revisit if:** a new module reads `ev`/`la`/`spray` from `--data-dir` — it must default to `phase_d5` or assert against the shipped constants.

## 2026-08-19 — E.14: the platoon signal is linearly decodable for LHB and not for RHB; ensemble intervals under-cover
- **Decision:** architecture §5's probe is discharged as a retrospective diagnostic and its ensemble-coverage item as measured; the model's LHB platoon split decodes above null (+0.223 [0.094, 0.351]) and the RHB split does not (+0.048 [-0.054, 0.142] against null +0.075).
- **Alternatives:** read the RHB result as a representation failure (rejected: the intervals overlap, no test of the difference was run, the probe is linear, and the C.2 target is shrunk harder for RHB — attenuation alone predicts a weaker decode).
- **Rationale:** the shuffled null is not centred on zero because out-of-fold ridge predictions of noise are anti-correlated with the target, which is why the test is "beats null"; the pooled row is confounded by stand and the per-stand rows are the headline. Fair intervals cover 0.391/0.690/0.856 against nominal 0.50/0.80/0.95, every Wilson interval excluding its nominal level in every stratum, so the pre-registered low-stratum pathology is real but global. Cold-start and trained rows inside the low stratum are indistinguishable at n=162/218, which is too few to resolve the effect at issue.
- **Reference:** Gneiting & Raftery 2007 for calibration subject to sharpness; the full reliability/resolution decomposition was retired 2026-08-19 as native to categorical forecasts.
- **Revisit if:** Phase O changes the ensemble width, or the refit changes the cold-start share.

## 2026-08-19 — E.15: the corrected between-stand share is 0.814, and the LHB reliability routes disagree
- **Decision:** the errors-in-variables debt from 2026-08-18 is discharged — the noise-corrected observed between-stand share is 0.814 [0.42, 3.50], superseding the withdrawn 0.109, and the interval is quoted with it. No single reliability figure is adopted; any reading of E.5 carries both ceilings.
- **Alternatives:** adopt the C.2-derived reliability alone (rejected: the split-half route contradicts it for LHB with an interval excluding the C.2 value, so a single figure would hide a real disagreement).
- **Rationale:** 86.6% of the total platoon-differential variance is sampling noise, and removing it from the within-stand term lifts the share onto the model's own 81.7% — reversing the concern that the model over-weighted stand, though the interval is loose because the denominator is a small difference of large numbers. The pre-registered 50-60% expectation failed, and the asymmetry check could not be evaluated because the noise-corrected LHB within-stand variance came back negative. For LHB, C.2 gives reliability 0.157 while split-half gives -0.366 [-0.84, 0.008]: two of three routes say LHB platoon differential has no measurable true spread in 2024, while the LHB signal is the one that decodes.
- **Reference:** The Book p.157 remains UNVERIFIED and decision-bearing — we disagree with its rho against our 0.652/0.719, and `src/analysis/c_report.py:45` records we lack a copy.
- **Revisit if:** the LHB contradiction needs adjudicating — a bootstrap CI on the C.2-derived reliability propagating rho's [0.384, 0.902] interval would move tau²_split by ~3x.

## 2026-08-20 — every Phase F artifact carries a tensor-build stamp, and reads assert the quality bins
- **Decision:** `src/analysis/provenance.py` stamps each Phase F summary with `data_dir`, the sha256 of that build's `manifest.json`, `train_seasons`, which quality-bin arrays are present, the arm, the seeds, the eval season, and the git revision. Any module reading `ev`/`la`/`spray` calls `assert_quality_bins`, which refuses to run against a build whose `quality_bin_edges` differ from `data/processed/phase_d5`.
- **Alternatives:** flip the stale `--data-dir` defaults in the five prior-lane modules (rejected again, for the same reason as on 2026-08-19: the frozen rule bars prior-lane edits absent a confirmed bug); do nothing and rely on the field-by-field audit already in the log (rejected: that audit covers the modules that existed when it was written, and by its own terms does not cover new ones).
- **Rationale:** the 2026-08-19 entry closed the exposure question and left one condition open — "a new module reads `ev`/`la`/`spray` from `--data-dir`; it must default to `phase_d5` or assert against the shipped constants." F.3, F.4, and F.5 are exactly that module. This discharges the condition mechanically rather than by reviewer memory, and the stamp means a future reader can tell which tensors produced a number without re-deriving it from module defaults.
- **Reference:** none — judgment call. This is a hygiene mechanism, not a technique adoption or a precedent-answerable modeling choice.
- **Revisit if:** the Phase O refit produces a new canonical build; `CANONICAL_DATA_DIR` must move with it, and every stamped Phase F number is then stale by construction.

## 2026-08-20 — the six heads are scored directly, and the contact head is cleared as the owner of the strikeout shortfall
- **Decision:** `src/analysis/f3_heads.py` scores each of the six factor heads on the 705,344 held-out 2024 pitches, by count and by handedness, against the observed rate in the same cell. Conditioned heads are fed the OBSERVED `ev`/`la` bins, exactly as training does, so the number is a head diagnostic and not a compounded-decode diagnostic.
- **Alternatives:** keep scoring only the composed wOBA scalar (rejected: we built a process model and graded only its scalar summary, so no artifact said whether any individual head was well calibrated); free-run the autoregressive chain and score the decoded joint (rejected: that measures decode drift, which is a different question and would blur head-level attribution).
- **Rationale:** the open residual from Phase E was a strikeout rate that fails population-matched at -4.78%, with the contact-split table as the named suspect. F.3 measures the contact head at **-1.40% overall, negative in all twelve counts** (worst 3-0 at -2.11%, 0-2 at -2.05%). Under-predicting contact means over-predicting whiffs, which pushes strikeouts *up*. The bias therefore runs against the observed shortfall and cannot own it — correcting the contact head would make the shortfall worse. The swing head splits into -0.41% non-two-strike and +1.37% two-strike, internal structure that cancels into E.6's aggregate +0.27%; the early-count under-swing is directionally consistent with the walk excess through a path E.6 never measured, because E.6 looked only at terminal three-ball counts. Expected bin index runs -2.6% on `la`, -2.3% on `spray`, +0.7% on `ev`.
- **Coverage shortcut, stated here because it qualifies the effect sizes:** the cell gaps are roughly 8-14 unclustered standard errors, and **clustered intervals were not computed**. Batter clustering would widen these; the signs and the all-twelve-count consistency are what the argument rests on, not the nominal significance.
- **Reference:** none — judgment call on diagnostic scope. No new technique is adopted; the scorer reuses `v1.factor_masks` rather than reimplementing the head nesting.
- **Revisit if:** the strikeout residual is re-attacked. The contact head is excluded as the owner; the early-count swing rate and the count-chain independence assumption from E.10 are the remaining candidates.

## 2026-08-20 — identity is worth a real but minority share of what the model knows about process
- **Decision:** `src/analysis/f4_process.py` reports held-out negative log-likelihood per head against two references: the same ensemble with the hitter embedding replaced by the reserved zero row (`identity_gain`), and a Laplace-smoothed empirical frequency table conditioned on count and both handednesses, fit on train seasons only (`gain_vs_reference`).
- **Alternatives:** report accuracy or AUC per head (rejected: the heads are calibrated probability distributions and the composition integrates over them, so a proper scoring rule is the honest measure); use a shuffled-identity reference instead of the cold-start row (rejected: shuffling injects another hitter's signal rather than removing identity, so the contrast would not isolate what identity buys).
- **Rationale:** every gate in this project scores a composed scalar, which cannot say whether the model learned process at all. It did: model NLL beats both references on all six heads. Identity's share of the total gain over the frequency reference is swing 10.6%, contact 17.9%, split 22.0%, **ev 34.8%**, la 12.1%, spray 13.6%. So the bulk of what the model knows is context, and identity is a real but minority contributor — largest on exit velocity, which is the dimension where hitters most plausibly differ. That is consistent with claim 1 failing its adoption gate while the model still ranks hitters better than any ladder rung: the identity signal is present and is small.
- **Coverage shortcut:** 6.46% of eval pitches belong to hitters with no trained embedding row, so for those the model and cold-start references coincide and the identity gain is diluted downward by that fraction.
- **Reference:** none — judgment call. Cold-start ablation against a reserved index is the project's own existing mechanism (`RESERVED_HITTER_INDEX`), not an imported technique.
- **Revisit if:** Phase O changes embedding dimension or regularization. Identity share is the natural thing to re-measure, since it is the quantity an embedding change is supposed to move.

## 2026-08-20 — a handedness-pooled score exists, and it is a description, not a gate
- **Decision:** `src/analysis/f5_pooled.py` collapses the pitcher-hand column to a single value and re-runs the unchanged `claim1_eval` referee, answering how well the model predicts a hitter's overall wOBA rather than the platoon split. Predictions are pooled by **prior** side-specific denominator PA, normalized within batter, with an even split for hitters carrying no prior record.
- **Alternatives:** weight the pooling by eval-season PA against each hand (rejected as leakage: that conditions the forecast on the held-out sample's own exposure, which is the leak `claim1_eval` avoids by stratifying on prior exposure); write a second scoring path (rejected: a second referee is a second thing to keep in step with the first).
- **Rationale:** the project's entire evaluation surface is (batter, season, pitcher hand) triples because the claim is about the split. Nothing said how accurate the model is overall, which is the first question any reader asks. Pooled, the model scores PA-weighted RMSE 0.03904 against a 0.02863 noise floor, weighted rank correlation 0.486, and a skill score of +0.161 over the no-information reference — positive in every stratum, including low exposure at +0.143 where `c1_raw` is -1.49 and `c1_bucketed` is -0.070.
- **Coverage shortcuts, both stated in the artifact itself:** only the model, both C.1 variants, and the no-information reference are pooled. C.2 needs its fitted prior parameters and C.3 needs a GBM refit, so **neither named gate opponent is in the table** and it cannot be read as a gate of any kind. Separately, `STRATUM_BOUNDARIES` was calibrated on side-specific prior PA and pooled exposure is roughly double, so these strata select a different population and are not comparable to the side-specific strata anywhere else; the rows carry `stratum_basis = pooled_prior_pa` to say so.
- **Reference:** none — judgment call on reporting scope.
- **Revisit if:** the pooled comparison is ever wanted as a real ladder. That requires regenerating C.2 and C.3 predictions, and the stratum boundaries would have to be recalibrated on pooled exposure first.

## 2026-08-20 — two latent traps in the Phase E audit modules are closed, and no committed number moves
- **Decision:** removed the `.fillna(0)` applied to `exact_ball` alone in `e_resample.main`, and added the missing `out_dir.mkdir` in `e_take_mass.main`.
- **Alternatives:** leave both and log them (rejected: the `fillna` one silently biases a reference downward, which is the kind of thing that surfaces as a wrong exoneration months later).
- **Rationale:** `sampled_ball` and `league_ball` were averaged raw while `exact_ball` had empty cells coerced to P(ball)=0, which would inflate the reported draw gap — and it contradicted `_take_means`, which deliberately returns NaN for that case. Checked before changing: **all 48 audit cells are non-NaN in the committed artifact**, so the E.7 resampler exoneration quoted in notebook 05 is unaffected and the fix is a no-op on today's numbers. Four further findings from the test backfill are left unfixed and recorded here as known: a dead `hasattr` branch in `resampler_audit`, a `relative_gap` that can divide by zero (0 of 44 committed cells are non-finite), a wrongly-shaped empty return from `paired_rows`, and a `KeyError: 'low'` in `e_min_pa_sweep.sweep` when a threshold censors the decisive stratum.
- **Reference:** none — judgment call, bug hygiene.
- **Revisit if:** any Phase E audit module is re-run on a different season or a smaller sample, where empty cells and censored strata become reachable rather than hypothetical.

## 2026-08-20 — the pooled skill column is barred as a Phase O selection axis
- **Decision:** No Phase O arm may be selected, ranked, or promoted on `f5_pooled`'s skill score, RMSE, or rank correlation. Selection stays on the frozen claim-1 metric over (batter, season, pitcher hand) rows, stratified by side-specific prior exposure; the pooled table is reported as description only.
- **Alternatives:** Treat the pooled column as a secondary tiebreak when the claim-1 strata disagree (rejected: a tiebreak is a selection axis with extra steps). Promote it to a real ladder by regenerating C.2 and C.3 pooled and recalibrating the strata (rejected as scope, and it would not remove the objection below).
- **Rationale:** Pooling collapses the hand column that carries the platoon claim, so optimizing the pooled score optimizes general hitter sorting — the easier, already-solved problem — against the split the project exists to measure. It is also the one column in the repo where the model is positive in every stratum, which is exactly the condition under which an unguarded metric captures selection. Frozen rule #2 already names the claim-1 metric as the arbiter of every ablation; this entry closes the gap rather than adding a rule. The pooled table additionally omits both named gate opponents, so it could not adjudicate an arm even if it were permitted to.
- **Reference:** none — judgment call on selection scope, resting on the manifest's frozen rule #2 and the coverage shortcut recorded in the 2026-08-20 pooled-score entry.
- **Revisit if:** the project's headline claim is ever restated as overall hitter projection rather than the platoon split, which would require reopening the manifest, not this entry.

## 2026-08-20 — the headline becomes the measurement ceiling, and frozen rule #1's gate is reported as failed
- **Decision:** The paper's headline claim is restated from *the model beats the incumbent on held-out side-specific wOBA* to *held-out side-specific wOBA has a reliability ceiling far below 1, and here is what fraction of it each method reaches*. Frozen rule #1's pre-registered gate is reported as failed, in the results and not the limitations. No rescue is attempted: the population is not re-cut to left-handed hitters, no post-hoc stratum is introduced, and no second gate is written.
- **Alternatives:** Drop the platoon claim and reframe the project as embedding-building with platoon left to a frontend query layer (rejected: it converts a pre-registered null into an unstated one and leaves the project with no falsifiable claim). Restrict the claim to left-handed hitters, where the model reaches 0.569 of the ceiling against right-handers' 0.122 (rejected: E.15 Part 3's noise-corrected LHB within-stand variance is **negative** at −3.75e-05, so the asymmetry's own supporting statistic is broken, and the L/R difference was never tested).
- **Rationale:** `e5_platoon_paired.csv` has `rank_favours_model` false in every stratum with all rank intervals crossing zero, and E.15 Part 2 puts noise at 86.6% of observed platoon-differential variance — the gate did not fail because the model is uninformative, it failed against a target that admits a maximum rank correlation of 0.356. A failed pre-registered gate reported alongside the ceiling that explains it is a stronger result than an unfalsifiable reframe, and it is the one framing that does not require re-opening a frozen rule. The by-stand ceiling fractions stay in the ceiling table as description, carrying the negative-variance caveat.
- **Reference:** Franks, D'Amour, Cervone & Bornn (2016), "Meta-analytics: tools for understanding the statistical properties of sports metrics," *JQAS* 12(4):151–165, DOI 10.1515/jqas-2016-0098 — the primary citation. Their *discrimination* \(D_{sm} = 1 - E[V[X]]/V[X]\) is algebraically identical to the reliability used here by the law of total variance, and the paper's whole argument is that a sports metric should be reported alongside how much of its spread is signal. Spearman (1904), DOI 10.2307/1412159, for the attenuation bound that turns reliability into a ceiling on rank correlation (\(\sqrt{\text{reliability}}\)); the same quantity is Lord & Novick's (1968) "index of reliability," for which Revelle, *Psychometrics with R*, ch. 7, is the linkable modern statement — the specific Lord & Novick page number could not be verified and is not cited. Brown (2008), *Annals of Applied Statistics* 2(1):113–152, DOI 10.1214/07-AOAS138, is the peer-reviewed baseball precedent for treating the ceiling as the estimand rather than the model. **Gap, stated rather than papered over:** no located source publishes a numeric ceiling for a hitter projection, so the number Phase M reports is ours and is presented as a measurement, not as agreement with a literature value. *The Book* p.157 remains UNVERIFIED and is not relied on here.
- **Revisit if:** closing the C.2/C.3 differential gap shows an incumbent at a materially higher fraction of the same ceiling, which would make the ceiling a story about this model rather than about the target.

## 2026-08-20 — Phase O is narrowed from arm selection to hyperparameter tuning
- **Decision:** Phase O tunes learning rate, warmup, and the epoch budget on the frozen D.10 architecture, arm, loss, features, and tensor build. Selection is on the ledger's `reference` column — held-out 2024 unweighted log loss per scored row — and on nothing else; **claim 1 is not read during Phase O**, as a tiebreak or otherwise. Budget is two overnight sessions, hard cap. The original Phase O scope (surviving D.8 arms, embedding dimension, the flagged-five block, spray, the bilinear term) is withdrawn, as is the optional pooled sampler.
- **Alternatives:** Keep Phase O as arm selection on claim 1 (rejected: claim 1 is the headline, so selecting on it makes the headline a selected number). Run no training at all and make Phase O purely a measurement phase (rejected: it conflates arm selection with tuning — the 0.000 correlation between claim 1 and held-out likelihood was measured across seven already-converged arms, which says likelihood cannot *rank* good models, not that it cannot *detect* an undertrained one). Select arms on F.5's pooled skill column (already barred by the entry above).
- **Rationale:** Learning rate, warmup, and the epoch budget were held at one value each for all 119 runs in the ledger, so none of them is evidence for anything, and a null reported on an untuned model is a weaker null than the same null on a tuned one. Likelihood is the right instrument for the only question Phase O asks — is this run undertrained — and the wrong one for choosing between arms, which is why the same statistic is legitimate here and barred one entry above. `src/analysis/o1_select.py` fixes the promotion rule in code before any o1 run exists so it cannot be adjusted after seeing which arm it promotes, and does not import `claim1_eval`.
- **Reference:** Warmup as a mechanism: Goyal et al. (2017), arXiv:1706.02677 (SGD, the original large-batch statement); Liu et al. (2020), "On the Variance of the Adaptive Learning Rate and Beyond," arXiv:1908.03265, for Adam specifically — early adaptive step sizes have high variance because the second-moment estimate is built from few samples; Ma & Yarats (2021), arXiv:1910.04209 (AAAI), which *refutes* Liu et al.'s variance explanation, re-grounds the effect in early update magnitude, and still recommends linear warmup, with a default length of about \(2/(1-\beta_2)\) steps — 2,000 at \(\beta_2=0.999\), against the 719 used here, so this grid is on the short side of the literature default and that is a limitation, not a tuned choice. **Synthesis, not sourced:** the specific claim that warmup should reduce the D.5 gradient-(b) displacement of rarely-updated embedding rows is *mine*, and no located source states it. What is sourced is the mechanism it rests on: Duchi et al. (2011), *JMLR* 12:2121, note that per-coordinate normalisation deliberately gives infrequent features large steps — i.e. the displacement is AdamW behaving as designed, not a bug; Kunstner et al. (2024), arXiv:2402.19449 (NeurIPS), and Qiu et al. (2025), arXiv:2505.05605, on frequency-adaptive rates for embedding tables. O.2 is therefore pre-registered as a **diagnostic**, and a null result there is a finding, not a failure.
- **Revisit if:** the o1 grid promotes an arm whose margin comes from a single seed, in which case the confirmation runs decide and the two-seed screen is not reported as the result.

## 2026-08-20 — batch size and weight decay are quarantined through Phase O
- **Decision:** Batch size stays at 8,192 and weight decay at 1e-2 for every Phase O run, and neither is exposed as a training argument. If Phase M has time, they enter there as a pre-registered arm whose strength is fixed before any claim-1 number is read, checked by re-measuring F.4's identity share.
- **Alternatives:** Sweep them in Phase O alongside learning rate (rejected for the contamination reason below). Drop them permanently (rejected: the decay-to-gradient ratio is a real and unmeasured lever, and refusing to ever look at it is not a finding).
- **Rationale:** The two are one setting — AdamW decays every parameter every step while an embedding row receives a gradient only in batches containing that hitter, so their ratio *is* the shrinkage applied to low-exposure hitters, which is the quantity C.2 estimates and the quantity claim 1 is about. Tuning them toward the low stratum would implement C.2's shrinkage inside the network and then score the result against C.2. The mechanism the D.5 gradient (b) artifact actually implicates is not decay in any case: rows at the 10th percentile of exposure see a 23.9:1 decay-to-gradient ratio yet finish *furthest* from the origin (mean norm 0.74–1.00 against the most-exposed rows' 0.52–0.61, consistent across all five seeds), which is the opposite of a binding decay and points at AdamW's per-coordinate second-moment normalisation instead.
- **Reference:** `results/phase_d/d5_level_attribution.json` `gradient_b`; the decay-to-gradient table in `docs/phase-d-spec.md`.
- **Revisit if:** Phase O's tuned build leaves the gradient (b) exposure–norm gradient intact, which would mean learning rate and warmup are not the lever and the ratio is worth a pre-registered arm sooner than Phase M's spare time allows.

## 2026-08-20 — Phase M is added as the measurement phase, and the 2025 run is spent after it
- **Decision:** A new Phase M sits between Phase O and Phase V and carries the paper: close the C.2/C.3 platoon-differential gap, build the per-exposure-stratum platoon ceiling, build the level-side ceiling (O.1), fix E.14's interval coverage by adding target sampling noise, re-run gradient (b) against Phase O's pre-registered prediction, and reconcile the E.5 and F.5 populations onto their intersection. Phase order is strict: O → M → V → the 2025 final run → write-up. The low-exposure stratum becomes the illustration of the ceiling argument rather than the gate it failed.
- **Alternatives:** Fold measurement into the write-up (rejected: items 1 and 2 are new computation, not new prose). Spend the 2025 run before measurement so the paper reports test-season numbers throughout (rejected: it is spent once, and running it before the build is final makes it a number the build was selected on).
- **Rationale:** Closing the C.2/C.3 gap is a prerequisite rather than an item — `e5_platoon_scores.csv` carries only `delta_pred` and `delta_route_a`, so the ceiling table currently has no incumbent in it and the reframe is undefensible without one. The per-stratum ceiling is likewise not a re-read: E.15 stratified by stand, not by exposure, so the stratum the project is about has no ceiling figure at all. E.14's interval reports seed spread alone and therefore cannot cover a target that is itself a small-sample measurement, which is a stated precondition of the query layer displaying platoon numbers.
- **Reference:** `results/phase_e/e15_ceiling.json`; `results/phase_e/e5_platoon_scores.csv`.
- **Revisit if:** the per-stratum ceiling cannot be estimated in the low stratum at acceptable precision, in which case the illustration moves to the stratum where it can and says so.

## 2026-08-20 — the run ledger gains lr and warmup columns, and the pre-Phase-O history is backfilled
- **Decision:** `results/phase_d/sweep_log.csv` gains `lr` and `warmup_steps`. All 119 pre-Phase-O rows are backfilled with `0.001` and `0` rather than left blank. *(Amended 2026-08-20 after review: a third column, `data_dir`, was added at the same time, the learning rate is written in one canonical spelling, and the `.bak` was dropped in favour of git history — see the entry below.)*
- **Alternatives:** Leave the columns blank for historical rows (rejected: blank reads as *unknown*, and the value is not unknown). Start a separate Phase O ledger (rejected: the ledger is the project's run record and splitting it would put two stages' `reference` columns in two files that look independently authoritative).
- **Rationale:** Every one of those runs used the module constants, so the backfill records a fact rather than an assumption — and the fact that one learning rate covers 119 rows is itself Phase O's justification. `append_ledger` writes a header only when the file is new, so a schema change had to be applied to the existing file or every subsequent row would have been written under the old header in the new field order.
- **Reference:** none — provenance hygiene.
- **Revisit if:** a pre-Phase-O run is ever discovered to have been launched with an overriding `--train-args`, which would make one of the backfilled cells false.

## 2026-08-20 — d10's tensor build is established by reproduction, and the ledger records the build from now on
- **Decision:** Every ledger row carries a `data_dir`. `d10` is backfilled to `data/processed/phase_d5` (its `block` arm to `phase_d5_noblock`, `d9` to `phase_d_split`), and Phase O's `--data-dir` pin is confirmed correct rather than assumed correct. Learning rates are written through a single `canonical_lr` so the column is groupable, and `append_ledger` now refuses to write when the header on disk does not match `LEDGER_FIELDS`. The `sweep_log.pre_o.bak` file is deleted: git already holds the pre-migration version and a second copy is one more thing that can be read as authoritative.
- **Alternatives:** Trust the source comment that said d10 ran on the D5-R17 rebuild (rejected: it was a comment, not evidence, and the whole point of the Phase O guard is that a moved build is invisible in every column). Leave `data_dir` blank for history (rejected here, unlike the `lr` backfill, because for d9/d10 the value *is* recoverable and therefore not unknown; it stays blank for `screen`, `d6` and `d8`, where it genuinely is).
- **Rationale:** Two independent lines of evidence agree. Re-running d10 baseline seed 0 on `phase_d5` reproduces its logged epoch-0 train and validation loss exactly (1.05990 / 1.04681), and the same command on `phase_d` raises `KeyError('split')` because that build predates the three-class contact split — so no `--split` run, which is every d10 arm, could ever have used it. Had the answer gone the other way, the o1 sweep would have spent a night and returned `guard_failed`, since `reference` is log loss over quality bins and the two builds have different bin edges (`e_bip_value.py:232` records 0.96371 against 0.92313 across them, roughly 400 noise-floor sds apart). The two spellings `1e-3` and `0.001` were a live defect, not a tidiness point: `knobs()` would have written `0.001` into a column whose 119 existing rows said `1e-3`.
- **Reference:** `results/phase_o/provenance_probe.log`; `results/phase_d/logs/d10_baseline_s0.log`; `src/analysis/e_bip_value.py:232`.
- **Revisit if:** a `d8`-or-earlier number is ever read down the same column as a `d9`/`d10` one, at which point the blank cells have to be established the same way rather than assumed.

## 2026-08-20 — Phase O selects on the 2024 season, and Phase M must label its 2024 numbers accordingly
- **Decision:** Phase O's selection metric stays `reference` — held-out log loss on the 2024 validation split — and the consequence is disclosed rather than mitigated: any claim-1 number computed on 2024 from the tuned build is **post-selection and descriptive**. The confirmatory number is the 2025 run, which stays sealed until Phase M finishes. The `o1_select` docstring carries this, so the artifact cannot be read without it.
- **Alternatives:** Split 2024 into a tuning half and a measurement half (rejected for now — it halves the exposure in exactly the low-exposure stratum the ceiling argument is about, which is the wrong thing to spend precision on, and it changes a frozen split). Tune on 2023 (rejected: 2023 is a training season, so its loss is not held out at all).
- **Rationale:** This is the frozen walk-forward protocol working as designed — 2024 is the selection season, 2025 is the test season — and every Phase D ablation already selected on 2024. Phase O adds no new leak. What it does add is a build whose 2024 numbers have been optimised, and Phases E and F report claim 1 on 2024; carrying those forward unlabelled after tuning would quietly convert them into selected numbers. The fix is a labelling obligation on Phase M, not a change to Phase O.
- **Reference:** raised by the 2026-08-20 review of the Phase O implementation; architecture plan §5, selection frame.
- **Revisit if:** Phase M's ceiling estimate turns out to be sensitive to the tuned build at all — if the ceiling moves between the incumbent and the tuned arm, the ceiling is not the property of the data it is claimed to be, and that is a bigger finding than Phase O.

---

## 2026-08-25 — β₂ and ε stay at the AdamW defaults, and are not swept in Phase M
- **Decision:** β₂ = 0.999 and ε = 1e-8 for every remaining run. Neither enters Phase M as an arm, pre-registered or otherwise.
- **Alternatives:** Sweep β₂ over {0.98, 0.999} in Phase O alongside learning rate (rejected at the time as untestable before the learning rate was settled — an objection the o1 result now discharges, since the learning rate *is* settled). Lower β₂ specifically to shorten the second-moment averaging window on rarely-updated embedding rows (rejected on the power argument below).
- **Rationale:** A 10× learning-rate range moved held-out loss by at most 0.8 SE against a 2 SE bar, with the two extreme arms 10 SE the wrong way. Learning rate scales every step directly; β₂ only reshapes how quickly the denominator those steps are divided by adapts. A knob with strictly weaker leverage than one that failed to clear the bar will not clear it either, and spending an overnight session establishing that costs Phase M time it has better uses for. The 719-step warmup grid remains short against the ≈2/(1−β₂) = 2,000-step default, which is the same limitation logged at the grid's design and is not reopened by this entry.
- **Reference:** Ma & Yarats (2021), arXiv:1910.04209 (AAAI), for the β₂-to-warmup-length relation and the default; Liu et al. (2020), arXiv:1908.03265, for the second-moment-variance account it refutes. The decisive argument here is the measured o1 margin table, not a source.
- **Revisit if:** the quarantined-knob arm runs in Phase M and moves `reference` by more than 2 SE, which would establish that optimiser settings have leverage on this build after all and make the whole second-moment family worth one grid.

---

## 2026-08-25 — Phase O returns `incumbent_stands`, and Phase M runs on the D.10 baseline ensemble
- **Decision:** The o1 3×2 factorial at two seeds returns `incumbent_stands`: `lr1e3` wins at margin 0.0, the best challenger `lr1e3_warm` reaches +0.8 SE against a 2 SE bar, and the remaining four arms are negative, at −1.5, −2.7, −10.3 and −10.5 SE. Grid complete, no underpowered arms, build guard passed at 4.5e-5 drift against a 4.2e-4 tolerance. Phase M therefore runs on the D.10 build unchanged — specifically the existing five-seed `d10_baseline_s0..s4` checkpoints, not the two fresh `o1` `lr1e3` seeds, which are a screening artifact and never a build.
- **Alternatives:** Promote `lr1e3_warm` on its +0.8 SE and confirm at five seeds (rejected: the promotion rule was fixed in code before any run existed, and its two seeds were 1.02567 and 1.02585 — a 1.27e-4 spread, wider than the margin itself). Extend the grid to a wider learning-rate range or a longer warmup (rejected: no arm in the swept 10× range beat the incumbent, and monotone failure across a range is not evidence of an interior optimum).
- **Rationale:** `incumbent_stands` is a real Phase O result and the one the spec named as such — the answer to "the learning rate was fixed at 1e-3 across all 119 runs and never varied" is now that it was varied over a 10× range and 1e-3 was already best. O.2 is not settled by this entry; it is measured separately below.
- **Reference:** `results/phase_o/o1_selection.json`; `results/phase_d/d5_level_attribution.json` `gradient_b`, whose five per-seed measurements read the `d10_baseline` checkpoints.
- **Revisit if:** Phase M's schedule leaves room for the quarantined-knob arm — the 2026-08-20 quarantine entry's revisit condition is now met, since neither learning rate nor warmup moved gradient (b), and that arm's strength must be fixed before any claim-1 number is read and checked by re-measuring F.4's identity share.

---

## 2026-08-25 — The promotion bar is the 95% convention, and the noise floor understates run-to-run spread
- **Decision:** `MARGIN_SES = 2.0` stands as the two-sided 95% convention; no derivation exists and none is retrofitted. The o1 standard-error column is read with the caveat that its `sd` is the quietest single arm's across-seed spread (1.04e-4) against a six-arm pooled spread of 4.08e-4.
- **Alternatives:** Recompute the rule on the pooled sd (rejected: the estimator was fixed in code before any run, and changing it after seeing the table is what pre-registration forbids). Derive the bar from a power calculation (rejected: no effect size was ever named to power against).
- **Rationale:** `incumbent_stands` holds under either estimator and at any threshold from 1 SE upward, so neither point is a rescue and neither changes the verdict.
- **Reference:** `src/analysis/o1_select.py:67`; `results/phase_o/o1_selection.json`.
- **Revisit if:** an arm ever lands between 1 and 2 SE, at which point the bar is doing real work and needs a real derivation.

## 2026-08-25 — `shrinkage_in_woba_direction` tests |projection|, not the signed projection
- **Decision:** `d5_level.gradient_b` regresses **|projection|** on exposure; a third reported series `abs_projection` carries it, and the signed `projection` series is retained unchanged for the quintile table. Self-checks now cover this flag in both the planted-shrinkage and planted-anti-shrinkage directions.
- **Alternatives:** Re-centre the projection on the population mean (rejected: the wOBA axis has a meaningful zero — the league-average hitter — so distance from zero is the quantity "closer to the origin" names).
- **Rationale:** The signed projection crosses zero, so its slope is positive whenever low-exposure rows sit on the negative side — which is the anti-shrinkage case, not shrinkage. Under the corrected test the flag is `false` in all five d10 seeds, with the abs-projection slope at −0.0050 to −0.0074 per 1,000 train pitches and the bootstrap interval excluding zero in every seed. D.5's anti-shrinkage holds along the wOBA axis specifically, not only in total norm.
- **Reference:** `src/analysis/d5_level.py:331-361, 416-433`; `results/phase_d/d5_level_attribution.json`.
- **Revisit if:** any other flag in the module is derived from a slope on a sign-crossing quantity.

## 2026-08-25 — O.2 returns a null: warmup rescales the embedding space and leaves the exposure gradient intact
- **Decision:** The pre-registered O.2 expectation — that on the warm build the rarest quintile's norm falls toward the most-exposed quintile's — is **contradicted**. Phase O closes on `incumbent_stands` and Phase M runs on the `d10_baseline` five-seed ensemble unchanged. Warmup is ruled out as the lever on anti-shrinkage.
- **Alternatives:** Widen the warmup grid toward the ≈2/(1−β₂) = 2,000-step default (rejected, and logged as the same limitation recorded at the grid's design: a diagnostic is not retuned around). Run the remaining three warm seeds (rejected: seeds are matched pairs and agree on every measure).
- **Rationale:** `torch.manual_seed` fixes initialisation and batch order, so `d10_baseline_s{n}` and `o1_lr1e3_warm_s{n}` are matched pairs and the comparison is paired. Warmup shrinks every quintile by a near-uniform factor — 0.915 to 0.927 across the five quintiles in seed 0 — so the q1/q5 norm ratio moves −1.3% and −0.04%, and the exposure-normalised slope moves the wrong way. The effect is a global rescaling, not the differential pooling O.2 predicted. **n = 2, descriptive only, never a promotion criterion.**
- **Reference:** `results/phase_o/o2_gradient_b.json`; `results/phase_d/d5_level_attribution.json` gradient_b for the paired d10 reference.
- **Revisit if:** the quarantined-knob arm runs in Phase M — with warmup measured and ruled out, batch size and weight decay are the only optimiser-side explanation of anti-shrinkage left.

## 2026-08-25 — The anti-shrinkage mechanism is AdamW's first step, not a short random walk
- **Decision:** The mechanism sentence in the 2026-08-20 quarantine entry is narrowed to its first-step clause. Rare rows are neither lightly updated nor displaced in an arbitrary direction.
- **Alternatives:** none — this is an arithmetic correction to a recorded claim.
- **Rationale:** Every pitch is seen every epoch, so a 30-pitch row takes ≈510 updates against ≈12,200 for the most-exposed, and pure diffusion over 510 full steps reaches ≈0.13 against an observed 0.74–1.00. The mean projection is −0.16 to −0.22 in all five seeds, which is a consistent direction. What survives is that AdamW's bias correction makes m̂/√v̂ exactly sign(g) on step one, so a row's first update is a full learning-rate step in every coordinate. This is estimate arithmetic, not measurement.
- **Reference:** none — judgment call on how an existing measurement is described; the underlying numbers are `results/phase_d/d5_level_attribution.json` gradient_b.
- **Revisit if:** the write-up needs a quantitative mechanism claim, which would require measuring displacement across epochs rather than bounding it.

## 2026-08-25 — The build guard verifies by exact reproduction, not by drift within a tolerance
- **Decision:** `o1_select.build_check` compares the incumbent and the noise-floor arm **seed by seed**. Where they share seeds the guard passes only on bit-identical `reference`, and reports `drift_is_informative: false`. Where they share none, it falls back to the 4 sd drift tolerance, which is then the only test available. The grid check is also made symmetric: an arm in the ledger but not in `EXPECTED_ARMS` returns `incomplete_grid`.
- **Alternatives:** Keep the drift tolerance alone (rejected: on the o1 ledger the incumbent rows *are* two of the five floor rows, so the drift was an algebraic function of its own reference set and could not detect anything). Tighten the tolerance (rejected: it does not fix a statistic computed against itself, and the tolerance at 4.17e-4 was already twice the promotion bar).
- **Rationale:** The guard exists as a units check — `phase_d` and `phase_d5` carry different quality-bin edges, so a wrong-build run lands on a different scale (0.96371 against 0.92313) while every ledger column still lines up. Bit-identical reproduction of a same-seed run is a direct test of that and is strictly stronger than any tolerance. The o1 incumbent reproduces `d10/baseline` seeds 0 and 1 exactly at 1.02584 and 1.02585, `best_epoch` 17 in both, so the build is verified. The verdict is unchanged: `incumbent_stands`, winner `lr1e3`.
- **Reference:** `src/analysis/o1_select.py` `build_check`; `results/phase_o/o1_selection.json` `guard`; `tests/test_o1_select.py` — four tests covering both regimes and the undeclared-arm path.
- **Revisit if:** a stage runs the incumbent on fresh seeds only, which puts the guard back on the tolerance and makes the tolerance's width load-bearing.

## 2026-08-25 — The warmup scheduler's verification gate is discharged
- **Decision:** `train.LinearWarmup` is cleared to run. Two of the seven ml-engineer gates are re-run on the warm path — overfit-one-batch with the schedule in the loop, and bit-identical losses across two same-seed runs. The other five (shape assertions, loss scale at init, split boundary, eval-mode hygiene, decode one batch) are **not** re-run, and this entry records why.
- **Alternatives:** Re-run all seven (rejected: five of them are properties of the model and the build, which the scheduler cannot reach — it multiplies a learning rate and touches no tensor, no split, and no decode). Skip the gate on the strength of the no-op proof alone (rejected: the no-op proof covers `warmup_steps=0`, which is the path where the scheduler does not exist).
- **Rationale:** A learning-rate schedule has exactly two failure modes reachable by these gates: a wrong scale stops the run learning, and a scheduler carrying state makes two same-seed runs diverge. Both are now tested. Separately, the no-op path is proved from the ledger rather than argued: `o1/lr1e3` (`warmup_steps=0`) reproduces `d10/baseline` seeds 0–1 bit-identically at 1.02584 and 1.02585, so introducing the scheduler changed nothing on runs that do not use it. The schedule's shape, its permanent stand-down, and `warmup_for`'s refusal of a warmup overlapping the first plateau cut were already covered.
- **Reference:** `src/model/train.py:118-168`; `tests/test_o1_select.py` — `test_the_warm_path_still_overfits_one_batch`, `test_two_warm_runs_at_the_same_seed_are_bit_identical`; `results/phase_d/sweep_log.csv` rows `d10/baseline` and `o1/lr1e3` seeds 0–1.
- **Revisit if:** the scheduler is changed to touch anything but `param_group["lr"]`, or a schedule is added that runs past epoch 2.

## 2026-08-25 — Phase O's selection metric carries a winner's curse; the decisive comparison is draw-balanced
- **Decision:** Recorded as a limitation of Phase O, not corrected. `reference` equals `best_val_loss` on all twelve o1 rows — it is a minimum over `best_epoch + patience + 1` draws of the same held-out split it is then scored on, and arms draw unequally (≈12 for `lr3e3`, ≈21 for `lr1e3`, ≈31 for `lr3e4`), so the low-learning-rate arms get an optimistic bias the fast arms do not.
- **Alternatives:** Score on a third split held out from early stopping (rejected: it would re-run the whole grid to fix a bias that does not change the verdict, and 2025 is sealed). Penalise by draw count (rejected: no defensible penalty exists at these sample sizes).
- **Rationale:** The bias inflates every arm in the same direction, so it cannot manufacture the null. It could in principle have hidden a real winner among the fast arms — but the only challenger with a positive margin is `lr1e3_warm`, at `best_epoch` 17 and 18 against the incumbent's 17 and 17. The decisive comparison is between two arms with the same number of draws, so the curse does not touch it. `corr(best_epoch, reference)` is +0.35 across the twelve rows, the direction the bias predicts.
- **Reference:** `results/phase_d/sweep_log.csv`, o1 stage; `results/phase_o/o1_selection.json`.
- **Revisit if:** any future stage selects across arms whose `best_epoch` differs by more than about 2×, where the bias stops being common-mode.

## 2026-08-25 — Phase M's promotion rule, pre-registered before any Phase M run exists
- **Decision:** Any Phase M comparison that promotes one build over another uses: (1) the **pooled** across-seed sd from every arm in the comparison, not the quietest arm's; (2) a **Bonferroni-corrected** threshold over the number of challengers, ≈2.9 SE for five challengers at the normal approximation and ≈3.5 SE on t at 4 degrees of freedom; (3) a confirmation step that **discards the screening seeds** and re-tests on fresh ones. Phase O's rule is **not** amended — it was fixed before its runs and stands as it ran.
- **Alternatives:** Carry Phase O's rule forward unchanged (rejected: its `sd` was the quietest arm's, roughly 4× too small and anti-conservative; 2 SE at 4 dof is p≈0.116, not 0.046; the family-wise false-promotion rate across five challengers was ≈46%; and its confirmation step was a seed *count*, reusing the screening seeds and shrinking the standard error, so passing confirmation was easier than passing the screen). Retrofit the correction onto Phase O (rejected: choosing an estimator after seeing the table is exactly what pre-registration forbids, and `incumbent_stands` holds under either estimator anyway).
- **Rationale:** Phase O's verdict is a null and every defect above is anti-conservative, so each one made promotion *easier* and none of them produced the null. That is why the fix is prospective and not a re-run. Phase M is where a positive result would be claimed, and a positive result under an anti-conservative rule is the failure mode that matters.
- **Reference:** `~/os/knowledge/frameworks/research-standards.md` §6; `results/phase_o/o1_selection.json` for the o1 spread the pooled estimate is drawn from.
- **Revisit if:** Phase M's comparison set is a single challenger against a single incumbent, where the Bonferroni term is 1 and only the pooled sd and the fresh-seed confirmation apply.

## 2026-08-30 — The τ² route rule: B′ primary, A sensitivity, B provenance
- **Decision:** Phase M estimates within-stand τ² by three reported routes. **Route B′** — `c2_bivariate_eb.fit` on the 2016–2024 window restricted to the M.6 intersection population — is the pre-registered primary. **Route A** (2024 observed minus modeled noise) is always reported beside it as the same-season sensitivity, carrying a ±3% noise-model fragility band. **Route B** (the unrestricted nine-season fit, 0.00059034) is one provenance row; the B→B′ delta is the population diagnostic. The rule is fixed before B′ is computed.
- **Alternatives:** Bracket A and B with no primary (rejected as headline: the population mismatch between them is fixable, and B′ fixes it — the bracket survives as the §10 fallback if B′ degenerates). Route B primary (rejected: fitted on hitters and seasons the claim is not about). Route A primary (rejected: τ²_A = 0.00419563 − 0.00407848 is a near-total cancellation — a 3% noise-model error swings it ~100% — and 88%-of-ceiling flatters the model, so a post-hoc pick would read as selection).
- **Rationale:** B′ makes the nine-season route directly comparable to A on population, leaving window and estimator stability as the only axes of disagreement. A's fragility is structural and known before any number is read, so naming B′ primary today is not selection. Restricting past seasons to 2024 eval hitters conditions on survival to 2024 — correct for this claim, and labeled.
- **Reference:** `results/phase_e/e15_ceiling.json` (corrected provenance, 2026-08-29 handoff); `docs/phase-m-spec.md` §M.0.
- **Revisit if:** B′'s τ² is negative or the restricted fit degenerates — the pre-decided fallback (spec §10.3) then reverts the headline to the A/B bracket with B′ reported as degenerate.

## 2026-08-30 — Route C gets one bounded diagnostic, and its worst case demotes Route A
- **Decision:** The split-half route's −0.366 gets one day, hard cap: a code audit of the split, and a null simulation of the split-half estimator under τ²=0 and τ²=B′ using each hitter's real PA count and E.15's noise model. Verdicts are pre-decided: inside the τ²=0 range → reported as consistent with small τ² (evidence line, never an estimator); bug found → fixed and re-emitted descriptively; outside both ranges with no bug → the shared noise model is suspect, the headline stands on B′ alone, Route A is demoted to "reported, unvalidated-noise-model caveat," and Route C is reported unexplained, flagged at the top of `m0_routes.json`.
- **Alternatives:** Drop it as broken (rejected: forfeits a third vote on the contested parameter, and a reviewer would run the check we skipped). Rebuild it as a full third route (rejected: at ~35 PA per half the estimate is likely irreducible, and days would be spent proving that).
- **Rationale:** Route A *is* the noise model — a subtraction whose survival depends on the model being right to within ~3%. If the world can produce −0.366 at no skill level our simulation allows, the noise model mis-describes the data and A's near-cancellation cannot be trusted; B′ does not lean on the 2024 noise model the same way. Pre-deciding the demotion keeps the choice out of the hands of whichever answer flatters the model.
- **Reference:** `docs/phase-m-spec.md` §M.0 (Route C diagnostic), §10.2.
- **Revisit if:** the simulation itself cannot be validated by the §9 planted-recovery checks, in which case its verdict carries no weight and the diagnostic is reported as inconclusive.

## 2026-08-30 — M.1 scores C.2 and C.3-full on the differential cut, and no other rung
- **Decision:** The ceiling table's opponents are C.2 (the incumbent) and C.3-full (the ML control). `no_info` and the C.1 rungs are omitted from the differential cut, and the paper states the omission. If C.3-full cannot emit side-specific predictions from existing artifacts, the table reports C.2 only (spec §10.1) — no differential head is improvised.
- **Alternatives:** All non-degenerate rungs (rejected: the ladder's decomposition logic was designed for the level claim; the middle rungs answer no live question here and add anchor points). C.2 only (rejected: leaves "ordinary ML would reach more of the ceiling" open on the exact cut the paper lives on).
- **Rationale:** Two questions are live on this cut — does the incumbent beat the model's fraction of ceiling, and would a GBM have done better. Two rows answer them; C.2's `predict()` already emits side-specific wOBA, so both are scoring passes, not new modeling.
- **Reference:** `results/phase_e/e5_platoon_scores.csv`; `docs/phase-m-spec.md` §M.1.
- **Revisit if:** a reviewer requires the full ladder on the differential cut, at which point the omitted rungs are scoring passes and can be added without re-opening anything.

## 2026-08-30 — The knob arm runs after Phase M, with its strength pre-registered now
- **Decision:** The quarantined-knob arm (2026-08-20 quarantine; revisit condition fired by O.2's warmup null) runs **after** all Phase M measurement items, not before and not inside them. Its arms are fixed before any M number is read: `wd3e1` (weight decay 1e-2 → 3e-1) and `bs2048` (batch 8,192 → 2,048), two seeds each, matched pairs against `d10_baseline_s0,s1`. Judged on gradient (b) movement and on `reference` under the 2026-08-25 promotion rule (k=2) — with promotion to the paper's build barred regardless of outcome, F.4's identity share re-measured on any arm that moves gradient (b), and no M.0–M.3 number ever recomputed on a knob build.
- **Alternatives:** Run it before Phase M (rejected: ordering only matters if the arm could change the build M measures, and it cannot — the quarantine's circularity argument bars promotion, and the ceiling is a property of the data, not the model; running first would also delay the Oct 1 abstract's numbers). Inside Phase M (rejected: an overnight spent on a mechanism paragraph while headline work waits). Never (rejected: the fired revisit condition would be silently dropped).
- **Rationale:** The trigger obligates a decision, not an immediate run. Pre-registering the strengths now removes the contamination running-after would otherwise invite: the knob values cannot be chosen in light of M's numbers. Strength rationale: the 10th-percentile decay-to-gradient ratio is 23.9:1 at wd=1e-2, so decay needs ≥24× to bind — 30× binds with margin; a 4× smaller batch separates update-count effects from decay effects.
- **Reference:** `docs/phase-m-spec.md` §8; 2026-08-20 quarantine entry; `results/phase_o/o2_gradient_b.json`.
- **Revisit if:** Phase M's schedule collapses entirely, in which case the arm moves to future work and the paper says the mechanism is unresolved.

## 2026-08-30 — Phase M's contingencies are pre-decided, and the spec is the build authority
- **Decision:** `docs/phase-m-spec.md` is adopted as the authority for the Phase M build window. Its §10 converts five former blocking questions into pre-decided fallback rules (C.3 differential unavailability, Route C outside both nulls, B′ degeneracy, Route B reproduction failure, low-stratum CI precision — the last now numeric: a 95% bootstrap CI including zero moves the illustration to the narrowest-relative-CI stratum). Hard stops remain for frozen-split config, stratum redefinition, reading 2025, and anything off-spec. The build agent applies fired rules, flags them prominently, and continues; the knob arm is out of the build window's scope.
- **Alternatives:** Leave the contingencies as stop-and-ask (rejected: each stop stalls an unattended window, and every one of the five had a decidable rule that could be fixed before any number is read). Delegate the hard stops too (rejected: frozen rules are never delegated).
- **Rationale:** Pre-deciding contingencies before their trigger data exists is the same discipline as pre-registering the route rule — it removes every remaining point where a choice could be steered by results. Adjustments of this kind bind *how* the plan is executed, never the claims, gates, metrics, or splits, all of which stand unchanged; the dated log entry is the audit trail against forking-paths concerns.
- **Reference:** `docs/phase-m-spec.md` §10; `docs/phase-m-opus-prompt.md`.
- **Revisit if:** a fallback rule fires in a way its wording did not anticipate, which is spec §10.7 territory: stop and report rather than improvise.
