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
- **Reference:** Loshchilov & Hutter, "Decoupled Weight Decay Regularization" — unverified, not yet in the project library. Settings and shrinkage table recorded in `docs/phase-d-spec.md` §5.
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

## 2026-08-08 — The RPS screen returns a decisive null; log loss stays v1's objective
- **Decision:** RPS is not promoted to a claim-1 ablation. Scored on 2024 under the canonical objective the RPS arms reach 1.19715 against the log arms' 1.07582, which is 126x the 0.00096 seed noise floor. The `rule` flag stays in the code, defaulting to log.
- **Alternatives:** Reading the arms' own recorded losses (rejected: 0.934 and 1.076 are in different units, so their ordering carries no information). Re-running the screen at more seeds (rejected: the gap is two orders of magnitude past anything seed variance produces).
- **Rationale:** The log score is local, so only the mass on the bin that occurred is scored, while RPS is distance-sensitive and spends mass on neighbouring bins — exactly what a local rule charges for. On the binary factors RPS reduces to Brier, whose gradient penalises confident-wrong outcomes less, yielding less extreme probabilities that the log score then charges for again. Each rule wins on its own metric, as the 2026-08-02 restricted-capacity test measured, so this confirms the screen's deliberate asymmetry rather than ranking the two rules.
- **Reference:** `results/phase_d/screen_scores.csv`; discharges the promote-only screen registered 2026-08-02.
- **Revisit if:** a future factorisation makes bin adjacency load-bearing for the run-value mapping, which changes what the quality heads are asked to get right.

---

## 2026-08-08 — Reliability and resolution are not computed for the RPS screen
- **Decision:** The screen closes on its first promotion clause alone, and no calibration/refinement decomposition is built at Phase D.
- **Alternatives:** Building it to honour the 2026-08-02 wording (rejected: no value of either quantity is reachable that changes the outcome). Holding the screen's verdict until it exists (rejected: the verdict does not depend on it).
- **Rationale:** The second clause promotes RPS only if it matches log loss at better reliability and equal resolution, and at 126x the noise floor it does not match, so the clause is unreachable rather than unmeasured. The machinery is owed exactly once, to §5.3's ensemble calibration check in the low-exposure strata, and belongs there where it is decision-bearing.
- **Revisit if:** §5.3 is built, at which point the decomposition enters with that check rather than this one.

---

## 2026-08-08 — D.5's pitcher repertoire and called-strike surface are keyed on batter handedness
- **Decision:** D.5 resamples whole real pitch rows grouped by `(pitcher, stand, balls, strikes)`, and the empirical called-strike model is fit as a separate surface per batter-hand × pitcher-hand combination rather than one pooled surface.
- **Alternatives:** Grouping on `(pitcher, balls, strikes)` and overwriting the `stand` one-hot to match the queried hitter (rejected: context columns 37-38 are that one-hot, so the overwrite submits a pitch row that never existed, and the repertoire would be averaged over whatever batter-hand mix the pitcher happened to face rather than the one the query asks about). A single pooled called-strike surface (rejected on the same evidence).
- **Rationale:** League pitch mix for right-handed pitchers differs sharply by batter hand — offspeed 20.8% against LHB against 8.0% against RHB, breaking 25.9% against 36.2%, sinker 9.7% against 19.5% — so a pooled repertoire would sample offspeed to same-handed batters at 2.6x its real rate. Taken-pitch location differs across the four handedness cells by the same mechanism, which is why the peer-reviewed precedent fits one surface per cell instead of pooling. The cost is sparser cells, absorbed by the smoothing toward pitch-type pools.
- **Reference:** Clemens, "Let's Take a Peek at Some Early 2025 Pitch Usage Trends," FanGraphs 2025-05-01 — usage splits, unreviewed blog post, cited for descriptive tables only. Deshpande & Wyner (2017), JQAS 13(3):95-112, Fig. 3 and the four separate called-strike surfaces its handedness differences motivate.
- **Revisit if:** a per-hand cell is too sparse for the smoothing to leave the pitcher's own mix visible, which would make the split cost more precision than the pooling bias it removes.

---

## 2026-08-08 — D.5 composes the count chain by exact solve, not Monte Carlo simulation
- **Decision:** The 12 non-terminal ball-strike states are solved by backward induction over repertoire-averaged transition probabilities, with the two-strike foul self-loop divided out in closed form as `W(b,2) = A / (1 - E[P_foul])`. No plate appearances are simulated and no iteration cap exists.
- **Alternatives:** Monte Carlo simulation with an explicit cap and reported truncation, which `docs/phase-d-spec.md` §6 pre-registered (rejected: it adds sampling noise to a frame whose measured seed noise floor is 0.00096, and the cap is itself a coverage shortcut standards §6 would require reporting). Truncating the foul loop at fixed depth (rejected: the closed form is exact and cheaper).
- **Rationale:** The pitch draw is independent given the count, so the repertoire-averaged transition matrix is exactly Markov in count and the chain admits a linear solve. Transitions only ever raise balls or strikes apart from the single self-loop, so backward induction over decreasing `b+s` needs no matrix inversion. Every terminal state sits inside the wOBA denominator, so `pred_woba = W(0,0)` with no renormalization.
- **Reference:** Yonushonis (2011), SABR BRJ — closes the two-strike foul with an infinite geometric series rather than truncating; he never frames it as a Markov chain, so the identification with `(I - Q)^-1` is this project's inference. Tenneal (2015), FanGraphs Community — the 12-count absorbing chain solved exactly by the limit of `P^n`; unreviewed and in-sample only.
- **Revisit if:** within-plate-appearance pitch sequencing enters the repertoire, which would make the pitch draw depend on the pitches already thrown and break the conditional independence the solve assumes.

---

## 2026-08-08 — The 24³ quality chain is enumerated exactly, not sampled
- **Decision:** `E[wOBA points | in play, h, x]` is the exact sum `Σ_e p(e) Σ_l p(l|e) Σ_s p(s|e,l) · V[e,l,s]` over all 13,824 bin combinations, computed from one trunk forward pass per (hitter, pitch) by broadcast.
- **Alternatives:** Sampling the chain once per simulated batted ball, which `docs/phase-d-spec.md` §6 pre-registered on the grounds that enumeration was prohibitive (rejected: the premise is false, and sampling adds variance to a quantity available in closed form).
- **Rationale:** `head_la` and `head_spray` are plain linear layers over `[trunk ; onehot]`, so conditioning bins enter as added weight columns with no trunk interaction and the joint logits are an outer sum. Reconstructed logits agree with real forward calls to 9.5e-07 maximum absolute error, which is one float32 ULP at the observed logit scale of 8.21.
- **Reference:** `src/model/v1.py:113-114`; agreement measured directly over all 576 conditioning pairs on 64 rows of `d6_baseline_s0`.
- **Revisit if:** any quality head conditions on the bins through something other than concatenation into a linear layer, which would destroy the outer-sum structure.

---

## 2026-08-08 — A fourth factor splits contact three ways, and v1 is retrained to carry it
- **Decision:** A three-class head over {foul, foul_tip, in_play} conditioned on contact is added to v1 and the model retrained. D.5 is built first on a league-average table for that split, and the retrained number is reported against it so the head's effect is measured rather than assumed.
- **Alternatives:** A binary `p(in_play | contact)` head (rejected: a foul tip is a caught tip and therefore a strikeout at two strikes, so folding it into the foul class inflates two-strike survival by 68,244 of 1,373,659 non-in-play contact events). A probe on the frozen trunk (rejected: the trunk was trained on a loss with no reason to preserve foul-versus-in-play information, so weak recovery could not distinguish a missing representation from a weak probe, and D.5's output is the headline). The league table alone (rejected: it compresses the split to league average and risks a false null on the hitter effect).
- **Rationale:** Architecture plan §1.3 treats contact as a terminal state, but `contact` is {foul, foul_tip, in_play}, so nothing in v1 predicts whether a plate appearance ends on contact and the chain cannot terminate. Foul and foul tip transition identically below two strikes and diverge only at two strikes, so the third class is exactly the distinction the chain needs and costs one logit.
- **Reference:** Clemens (2025), FanGraphs — a count-conditional foul score repeats year over year at r ≈ 0.3, though "early count" is never defined so the construction is not reproducible from the text. Baumann (2024), FanGraphs — raw foul rate per swing relates weakly to hitter quality (max |r| 0.18) while fouls per whiff separates player types, and that ratio is recovered jointly by this head and the existing contact head. Tenneal (2015) measures in-zone contact fouling at 47% against out-of-zone at 55%, roughly four times his count effect, and his own null is on expected K%, a metric insensitive to batted-ball reallocation.
- **Revisit if:** the retrained arm's claim-1 number does not separate from the league-table baseline by more than the seed noise floor, which would make the league table the honest reported form.

---

## 2026-08-08 — D.5's pitcher population is prior seasons only, weighted by batters faced
- **Decision:** The simulator draws pitchers from the training seasons only (2015-2023), weighted by batters faced, and holds one pitcher for a whole simulated plate appearance. Pitches are not reweighted by season inside that window.
- **Alternatives:** Including eval-season pitchers (rejected: a projection that reads the eval season's pitcher population is reading the season it predicts). Recency-weighting the pool to track within-window usage drift (deferred as a possible later improvement, not built for v1). Resampling a fresh pitcher per pitch (rejected: a plate appearance is against one pitcher, and pooling pitches across pitchers is the Jensen error §1.3 already refuses).
- **Rationale:** Every Phase C rung is built from prior seasons only, so the Phase D query must be too or the comparison is not information-matched. Batters-faced weighting makes the simulated opponent distribution the one hitters actually face rather than one weighted by roster count.
- **Reference:** `src/config/split_config.json` (frozen 2026-07-17); the batters-faced idiom at `src/data/eval_targets.py:156`. Clemens (2025) documents the drift the deferred recency weighting would address — right-handed pitchers' sinker usage against left-handed batters fell from 21.0% to 10.2% across 2015-2023 while their usage against right-handed batters stayed flat.
- **Revisit if:** §5.4's composition validation shows league run scoring off in a direction traceable to the pool's season composition, which is what recency weighting would correct.

---

## 2026-08-08 — The five ensemble seeds are combined by averaging conditionals
- **Decision:** The five seeds' per-pitch conditional probabilities are averaged and one composition is run on the average. The reported `pred_woba` is that single number.
- **Alternatives:** Running five compositions and averaging the resulting wOBA (rejected for the headline: it averages a nonlinear functional rather than the predictive distribution, the same Jensen objection §1.3 raises one level down; retained as the source of the between-seed spread §5.3's calibration check needs).
- **Rationale:** A deep ensemble's prediction is the uniformly-weighted mixture of its members' predictive distributions, which for classification is exactly averaging the predicted probabilities. Averaging first is also one composition rather than five.
- **Reference:** Lakshminarayanan, Pritzel & Blundell (2017), NeurIPS 30:6402-6413, for the mixture form and the classification case. The paper never composes its mixture through a downstream nonlinearity, so it prescribes the combination order without validating it in this setting, and the Jensen argument is this project's.
- **Revisit if:** §5.3 needs the functional's dispersion as the headline uncertainty rather than as a companion, which would make the five separate compositions the primary object.

---

## 2026-08-08 — The called-strike model uses raw plate coordinates, without batter-height normalization
- **Decision:** `p(ball, called strike, hit by pitch | take)` is fit on `plate_x` and `plate_z` with no rescaling to a per-batter zone. `sz_top` and `sz_bot` stay out of `clean.RETAIN_COLUMNS`.
- **Alternatives:** Re-pulling `sz_top`/`sz_bot` and normalizing the vertical axis (rejected on the grounds below, and it would modify the frozen data pipeline, which trips the CLAUDE.md verification gate). Keying on Statcast `zone` (rejected: it is a coarse deterministic function of the same two retained columns and discards resolution).
- **Rationale:** Statcast derives each hitter's zone top and bottom from previous major-league umpire calls, so a called-strike model normalized by those fields would regress umpire behavior on a rescaling of umpire behavior. The fields are also a human annotation set per plate appearance rather than a measurement.
- **Reference:** Deshpande & Wyner (2017), JQAS 13(3):95-112 — had the PITCHf/x zone boundaries available and deliberately collapsed them to a league average used only for filtering and figures, fitting the surface itself in raw coordinates. Baseball Prospectus #37347 (2018-01-29) for the annotation provenance and the circularity. Freiman (2018), FanGraphs, prices what is given up: batter height explains R² = 0.23 of the low called strike, computed after dropping three outliers, against 0.05 of the high one.
- **Revisit if:** batter height from roster data becomes available, which would recover the low-zone effect without inheriting the umpire-call circularity.

---

## 2026-08-08 — C.2 discharges frozen rule #1's empirical-Bayes baseline
- **Decision:** C.2's bivariate empirical Bayes is the estimator satisfying frozen rule #1's "empirical-Bayes platoon regression (The Book)" requirement, and the C.2 rung fitted at The Book's published constants is the literal incumbent scored beside it.
- **Alternatives:** Building the split-level Book estimator as the primary (rejected 2026-07-27: its variance requires subtracting an unstable noise term far larger than the quantity sought). Treating frozen rule #1 as undischarged until a split-level estimator exists (rejected: the two parameterizations are the same model in rotated coordinates, so the requirement is about the estimand, not the algebra).
- **Rationale:** Frozen rule #1 names three role-matched incumbents and C.2 is the third; without this entry its discharge is implied by the ladder rather than recorded. The Book-rho rung exists precisely so the published constants are scored, not only this project's refit of them.
- **Reference:** manifest frozen rule #1; decision log 2026-07-27 (C.2 estimand) and 2026-07-29 (the ladder as a decomposition).
- **Revisit if:** the rho interval tightens enough to separate this estimate from The Book's implied value, which the 2026-07-27 entry already names.

---

## 2026-08-08 — D.5's own knobs are validated on composition fidelity, never on claim-1
- **Decision:** The pitcher-pool size, pitches per `(pitcher, stand, count)` cell, and repertoire smoothing strength are pre-registered before any claim-1 number exists and validated only against §5.4's composition check. No D.5 knob is ever set by its effect on the claim-1 metric.
- **Alternatives:** Treating D.5's knobs as ordinary §4 feature decisions settled by ablation on claim-1 (rejected: the claim-1 metric is produced by D.5, so tuning D.5 on it selects the measuring instrument to flatter the measurement).
- **Rationale:** Frozen rule #2 sends unclear choices to a claim-1 ablation, but it presumes the choice sits upstream of the metric, and D.5 sits inside it. Composition fidelity is an independent criterion because it scores the simulator against observed league run scoring rather than against the model's own margin over Phase C.
- **Reference:** manifest frozen rule #2; architecture plan §4 (feature-decision rule) and §5.4 (composition validation).
- **Revisit if:** a D.5 knob changes the Phase D versus Phase C ranking without changing composition fidelity, which would mean the two criteria have come apart and the knob needs its own pre-registered treatment.

---

## 2026-08-08 — The fourth factor retrains every arm together, as ledger stage d9
- **Decision:** All eight D.8 arms are re-run carrying the three-class split head, as a new ledger stage `d9` on a dataset rebuilt to carry the split label. The 30 completed `d8` runs stand as the record of the pre-split architecture and are neither deleted nor extended.
- **Alternatives:** Running the two never-run arms (`block`, `nospray`) on the pre-split architecture first (rejected: the split head changes the loss and the shipped model, so an ablation measured without it answers a question about a model that does not ship). Carrying the six existing arms forward and running only the two new ones (rejected: every arm gains a loss term, so one `reference` column would mix two units).
- **Rationale:** Frozen rule #2 settles architecture choices by ablation on the claim-1 metric, and claim-1 now flows through a model carrying the fourth factor, so the whole table has to sit on one architecture. Re-running all eight costs one overnight session rather than two, because the retrain rebuilds every arm regardless.
- **Reference:** `docs/phase-d-spec.md:253-254` registers the two arms that had never run; `results/phase_d/sweep_log.csv` holds the six that had. A `d8` reference and a `d9` reference are in different units and must never be read down the same column.
- **Revisit if:** never for comparability — a further factor would require the same treatment and its own entry naming this one.

---

## 2026-08-08 — D.5 scores against the whole pitcher population, not a sampled panel
- **Decision:** The simulator queries every pitcher in the 2015-2023 pool, weighted by batters faced, with 6 pitch rows per `(pitcher, stand, balls, strikes)` cell. Taking the full population is the default rather than an option.
- **Alternatives:** A sampled panel of 60 pitchers at 12 rows per cell (rejected on the measurement below). Fewer pitchers with more rows each (rejected: within-cell noise averages away across 12 counts and 2,027 pitchers, while panel noise does not).
- **Rationale:** A sampled panel leaves a common level shift across every hitter at once — 0.0048 wOBA between two draws at 60 pitchers, against a 0.033 between-hitter spread — and PA-weighted RMSE charges for exactly that level. Taking the whole population removes the source instead of shrinking it, and makes the number independent of the draw.
- **Reference:** measured across two panel draws at 60 pitchers — per-hitter correlation 0.988, mean level shift 0.0048, residual per-hitter noise 0.0020 once that shift is removed. Settled on estimator stability, never on claim-1, per the 2026-08-08 knob entry.
- **Revisit if:** the pool grows enough that a full pass stops fitting a session, at which point a panel returns and its level noise is reported beside every number built on it.

---

## 2026-08-09 — A simulated plate appearance faces one pitcher, whose pitches are drawn independently per count
- **Decision:** The count chain is solved separately for each pitcher from his own twelve `(pitcher, stand, balls, strikes)` cells and the results averaged by batters faced afterwards, with six real pitch rows drawn per cell. Within a plate appearance the pitch draw depends on the count and on nothing else that has happened in it.
- **Alternatives:** Averaging every pitcher's transition probabilities into one composite and solving the chain once (rejected: the chain divides by `1 - P_foul`, so the two orders give different answers, and no hitter ever faces the composite). Conditioning the repertoire on the pitches already thrown in the plate appearance (rejected: it multiplies the cell count past what a 5.9M-row pool supports, and it destroys the Markov property the exact solve depends on).
- **Rationale:** Independence given the count is what makes the repertoire-averaged transition matrix exactly Markov, which is what licenses a closed-form solve rather than simulation with a truncation to report. The cost is pitch sequencing, which is real and is not represented.
- **Reference:** `src/model/query.py` (`_group_woba` solves per pitcher, then weights); `src/model/query_tables.py` (`Repertoire.sample`, three-level backoff, 83.3% of cells exact). Six draws inflate the two-strike multiplier by +0.0098 against ~1.40 across the ten busiest 3-2 cells, propagating to order 0.001-0.002 wOBA.
- **Revisit if:** the level bias survives the take-surface and reserved-row diagnoses, at which point `n_pitches` is the remaining knob and is raised under the composition check rather than against claim-1.

---

## 2026-08-12 — D.5 composition validation splits into an unscored probe and a scored per-PA check
- **Decision:** The §5.4 zero-row composition probe is retained but never scored, and a second check carries the pass condition: spec §8's four per-plate-appearance absorbing rates (BB, K, HBP, BIP), computed over a plate-appearance-weighted population of trained hitters with the four (stand, p_throws) cells weighted by their true train-window shares. Each rate must fall within 2% relative of the observed train-window rate and HBP within 20%, and a fix is credited only when it moves a rate by more than the between-seed spread, which the five spec §6 per-seed compositions are now persisted to supply.
- **Alternatives:** Rebuilding the single existing check in place (rejected: the five arms already diagnosed against the zero row lose their common reference, and the zero row is not a hitter, so no pass condition applied to it means anything). Scoring the four rates on train-window hitters rather than the eval season's (rejected for this phase: it needs a second full population pass, roughly doubling a run, and the population mismatch is instead reported by carrying both windows' observed rates). Keeping the flat 25% handedness weights for continuity (rejected: they are a −0.00271 wOBA error before the model contributes anything, and both weightings are now reported so continuity is preserved without keeping the error).
- **Rationale:** The 2026-08-08 knob entry makes composition fidelity the sole validator of D.5's own knobs, and the check named there could not detect the level error the phase is carrying: it read five per-pitch masses at the 0-0 count rather than the four per-PA rates the chain terminates in, so a walk-for-strikeout distortion was invisible. wBB is 0.689 and wHBP 0.720, so trading terminal states barely moves a wOBA level, and a strikeout and a batted-ball out are both worth zero — one aggregate number cannot separate four failure modes. The rates are obtained by substituting indicator payoffs into `solve_chain` rather than by a second recursion, so they cannot drift from the wOBA number they explain, and their sum-to-one is asserted as a mass-conservation check.
- **Reference:** `src/model/query.py` (`absorbing_rates`, `observed_absorbing_rates`, `league_fidelity`, `handedness_shares`); observed train-window rates over 1,526,308 plate appearances in the wOBA denominator — BB 0.08042, K 0.22338, HBP 0.01049, BIP 0.68571; observed league wOBA 0.31639 under true handedness shares against 0.31368 under flat 25%. Tolerances pre-registered in the 2026-08-12 remediation plan before the per-PA observed rates were computed.
- **Revisit if:** a fix passes all four rates while composition fidelity and claim-1 disagree about it, at which point the population mismatch named above becomes the first suspect and the train-window-hitter pass is the test.

---

## 2026-08-12 — The third contact class rests on a rule of baseball, not on the cited posts
- **Decision:** The 2026-08-08 fourth-factor entry stands, and its support is restated here per the append-only rule: the three-class contact split {foul, foul_tip, in_play} is carried because a caught foul tip at two strikes is a strikeout, so the two-strike branch of the count chain cannot close without the distinction. The three blog posts cited there speak only to whether the hitter-specific foul-tip rate is learnable, and no verdict rests on them.
- **Alternatives:** Sourcing the split to a peer-reviewed foul-tip study (rejected: the load-bearing premise is a playing rule, and a citation for it would imply the retrain was justified by an empirical estimate it was not). Amending the 2026-08-08 entry in place (rejected: the log is append-only, and the entry's decision has not changed).
- **Rationale:** A 40-run retrain supported only by unreviewed posts reads as an empirical bet, which misstates how well supported it is in both directions — the mechanical argument is stronger than the posts, and the posts are weaker than a reader would assume from their position in the entry. Separating the two makes the retrain's justification survive the posts being wrong.
- **Reference:** decision log 2026-08-08 (fourth factor as ledger stage `d9`); `docs/phase-d5-review.md` D5-R18(5).
- **Revisit if:** the learnability question becomes load-bearing — a per-hitter foul-tip rate used as a feature or a claim would need the empirical support the posts do not supply.
