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

---

## 2026-08-12 — xwOBA enters as a rung and a second answer key, never as the primary target
- **Decision:** `estimated_woba_using_speedangle` takes three separate roles and no fourth: a C.1-xwOBA ladder rung built from Statcast's own field rather than from `V` marginalised over spray, a second answer key every rung and arm is scored against beside realized wOBA, and an approximate achievable floor read as RMSE between the two answer keys. Realized wOBA remains the primary claim-1 target, and no gate verdict anywhere reads an xwOBA row.
- **Alternatives:** xwOBA as the primary target (rejected: it changes the claim from runs to a latent quality measure, makes the answer key another model's output rather than ground truth, and would re-score every frozen Phase C number). Building the rung from `V` marginalised over spray (rejected: a rung sharing `V`'s defects is no longer an external incumbent, which is the only thing a baseline is for). Subtracting the floor from the model's RMSE (rejected under the 2026-08-08 knob entry, and the floor is approximate in both directions since xwOBA uses actual K and BB and carries its own (EV, LA) mapping error).
- **Rationale:** The gap between the two keys decomposes claim-1 error into "wrong about batted-ball quality," which is the model's to fix, and "could not have known" — fielder placement and sequencing, which the composition has no channel to express. The measurement answers the question it was built for: the floor does not explain the null, since it sits at 0.02923 against Phase D's 0.0492 all-stratum RMSE, so roughly 40% of the error is answer-key noise and the remainder is the model's.
- **Reference:** `src/analysis/c1_trailing.py` (`measure="xwoba"`), `src/analysis/c_report.py` (`c1_xwoba` rung), `src/analysis/d5_report.py` (`second_target_table`, `achievable_floor`), `results/phase_d/d5_both_targets_*.csv`. Floor by stratum 0.03640 / 0.03309 / 0.02521 / 0.02923; half the calibration gap against C.3-full survives the change of target (0.0029 realized against 0.0014 xwOBA) while the ordering advantage grows (weighted rank 0.5763 against 0.5389 on xwOBA, 0.4608 against 0.4417 on realized); C.1-xwOBA lands between C.1-bucketed and C.2 on RMSE at 0.0484 and beats C.2 on weighted rank at 0.4248 while being weak at low exposure, where batted balls are scarce.
- **Revisit if:** a Phase E model gains a channel for the "could not have known" component — batted-ball direction against a fielding alignment — at which point the floor stops being a floor for that model and the two keys stop bounding the same quantity.

---

## 2026-08-12 — D.5's level excess: two composition fixes land, the cell-size term is exonerated, and exposure does not survive a talent control
- **Decision:** Two of the four D5-R15 contributors become code and two do not. Twelve count-specific logit offsets are raked onto the four existing take surfaces with the location shape held fixed, and the in-play mass is split 96.37 / 3.63 with the unmeasured share valued at the scalar 0.29973. The six-pitch cell size stays at `n_pitches = 6`, and the take surface does not escalate to 48 count-keyed surfaces, so neighbour smoothing stays latent. HBP needs no fix of its own: its 43%-high 0-0 pricing is entirely the count-blind surface, which the offsets now correct.
- **Alternatives:** Escalating to 48 surfaces (declined: the pooled surface's marginal ball error is +0.0002 over 3.89M takes, so it is right on average and structured only per count, and raking recovers the whole of that per-count component — the residual walk gap therefore cannot live in ball-given-take, and 48 surfaces would buy count-specific location shape at 3.65× the cells plus a mandatory neighbour-smoothing component for an error term just shown to be nearly exhausted). Raising `n_pitches` (declined on the measurement below, which shows the term has the wrong sign). Subtracting the estimated Jensen term or any other computed bias off the level (rejected standing, 2026-08-08). Reweighting `fit_outcome_table`'s retained sample instead of splitting the mass (rejected: it distorts every cell's conditional to fix an aggregate, which the Phase B missingness rule forbids).
- **Rationale:** The unmeasured-category split passes its two-sided condition exactly — league wOBA falls 0.00171 against a predicted ≈0.0015 while all four absorbing rates stay bit-identical to seventeen digits, which is the signature of a change wired into valuation rather than transitions. The offsets move every failing rate toward observed and close the strikeout gap outright but not the walk gap. The cell-size term is the cleanest negative result in the step: predicted league wOBA runs 0.30801 / 0.30870 / 0.30928 at M = 6 / 12 / 24, so it is monotone increasing in M and refining the estimator makes the +0.01771 over-prediction worse, and a 1/M extrapolation puts the limit near 0.3094–0.3099, leaving M = 6 about 0.0015–0.0020 below its own limit — an order of magnitude under the level bias and pointing the other way. Gradient (a) removes exposure as the explanation once talent is controlled, and gradient (c) shows the trunk is mildly multiplicative but prices that interaction at roughly a tenth of the level bias, so the two results are compatible: the excess is a level, and the mechanism that would have made it a gradient exists but is too small to be it.
- **Reference:** `src/model/query_tables.py` (`fit_take_count_offsets`), `src/model/query.py` (`--take-count-offsets`, unmeasured branch), `src/analysis/d5_level.py` (`gradient_a`, `gradient_c`), `results/phase_d/d5_take_count_offsets.json`, `d5_step3_knobs.json`, `d5_level_attribution.json` (full population) and `d5_level_attribution_subset.json` (the 50-hitter pair gradient (c) is fitted on). Offsets span ball 0.6745 at 3-0 to 1.1200 at 0-1 and called strike 0.4338 at 0-2 to 1.3175 at 2-0, at shrink alpha 0.3643. Offset effect, agreeing across the row-0 probe and a 50-hitter subset: BB −0.00172 / −0.00186, K −0.00446 / −0.00478, BIP +0.00598 / +0.00644, HBP +0.00020 / +0.00020, wOBA +0.00107 / +0.00114, against Step 0's scored gaps BB +0.00692 (+8.60%, fail), K +0.00557 (+2.49%, fail), HBP +0.00023 (+2.21%, pass), BIP −0.01272 (−1.85%, pass) — K needs 0.00110 removed to reach tolerance and gets 0.00446, BB needs 0.00531 and gets 0.00172. HBP observed per take by count 0.00307 at 0-0, 0.00854 at 0-2, 0.00104 at 3-0 against a 0.00506 marginal; both pre-registered HBP branches are exonerated, shrinkage at population inflation −0.00004 and clipping at net mispricing −0.000008 with in-range edge cells at 0.02945 running above out-of-grid takes at 0.02837. Gradient (a) prior PA +0.000280 [−0.000017, +0.000576] and C.2 posterior −0.144369 [−0.304534, +0.012262], both containing zero against an intercept of +0.059076 [+0.010015, +0.108605] that does not; gradient (c) slope +0.00003979 [+0.00001114, +0.00007595] per 100 prior PA on a mean delta of +0.002112.
- **Revisit if:** the walk gap survives the Step 4 rebuild's re-read, at which point take frequency rather than ball-given-take is the named suspect and the swing head is the place to look, not the take surface; or the pre-registered credit rule is applied to a future fix, since its reference quantity is the between-seed spread of a level (BB 0.00587, K 0.00794, HBP 0.00020, BIP 0.01401) while every fix here is measured as a paired same-ensemble delta, so the rule as written compares a delta to the dispersion of a level and credits nothing in this step.

---

## 2026-08-12 — the shipped equal-mass binning wins its own pre-registered contest, so D5-R7's top-bin defect is not a binning problem
- **Decision:** The quality bins stay **equal-mass at 24 per dimension**, refit once with the Statcast placeholder pairs dropped, and the resulting edges are frozen in the build manifest and never recomputed per arm or per seed. The placeholder rows are dropped from the edge fit and masked out of all three quality targets rather than `la`'s alone, since the pair fabricates the exit velocity too. Two supervised candidates were measured and both lost.
- **Alternatives:** `top_decile_split`, which puts 10% of the mass in the tail across 6 bins (rejected on the measurement: it narrows the top exit-velocity bin from 15.10 to 12.80 mph but raises joint within-cell wOBA variance from 0.143116 to 0.145093). `variance_min`, generated by the exact 1-D dynamic program of Fisher–Jenks / `Ckmeans.1d.dp` (rejected, and it loses worst at 0.148714 — the DP minimises within-bin variance one dimension at a time, which is precisely the marginal-versus-joint error D5-R7 warns about, so optimising the criterion the model does not use costs the criterion it does). Raising `n_bins` to 32, which reaches a 14.3 mph top bin (deferred: it is a 2.4× chain enumeration at 13,824 → 32,768 cells and it changes the output space, so it is a separate decision). Masking the placeholders rather than dropping them (rejected: masking leaves them in the quantile computation, which is where they did the damage).
- **Rationale:** The objective was pre-registered as within-cell **realized** wOBA variance over the joint 24³ grid on train seasons only — joint because the model predicts into a joint cell, and realized because `estimated_woba_using_speedangle` is what `V` estimates, so scoring against it would be circular. The result contradicts the expectation the step was built on: equal mass wins outright, so **the top-bin defect cannot be fixed by re-binning at fixed `n_bins`** — narrowing the tail necessarily widens something else, and the something else costs more than the tail saves. Equal mass also leaves the softmax class frequencies uniform by construction, so it does not push the lever `invfreq` (+0.02524) and `meanweight` (+0.00131) already probed and lost on, which an unequal scheme would have.
- **Reference:** `src/data/bin_design.py`, `src/data/model_dataset.py` (`BIN_SCHEMES`, `_variance_min_edges`, `_snap_for_dp`, `joint_cell_variance`, the `encode_labels` target mask), `results/phase_d/d5_bin_design.json`. Measured over 954,223 scored train-season balls in play: within-cell variance 0.143116 / 0.145093 / 0.148714, variance explained 0.5783 / 0.5725 / 0.5619, top exit-velocity bin span 15.10 / 12.80 / 11.80 mph, occupied cells 13,726 / 13,049 / 11,344 and median occupied cell 58 / 57 / 22. Placeholder drop 45,932 rows, 3.6085% of balls in play. The DP is O(`n_bins`·m²) in distinct feature values, which `ev` (1,074) and `la` (181) satisfy but `spray` (877,932 — an arctangent of hit coordinates rather than a reported quantity) does not, so values above `DP_MAX_DISTINCT` = 2048 are snapped to a uniform grid, 0.088° for spray.
- **Revisit if:** the pre-launch blocker is ever evaluated on a candidate other than these three, in which case note that `min_cell_count` cannot carry it — the smallest occupied joint cell is 1 under the shipped equal-mass edges too, so an absolute floor fires on the status quo and discriminates nothing. It was restated as the share of in-play mass in cells below `BACKOFF_MIN_CELL` measured relative to the shipped scheme: 4.2481% for equal mass against 5.4263% and 6.0496%, so the winner is the reference and `fit_outcome_table`'s (ev, la) backoff needs no re-fit. Also revisit if D5-R7's high-stratum discrimination loss survives the rebuild, since the remaining levers are more bins or a within-bin conditional mean, and both are Phase E.

---

## 2026-08-12 — the D.5 report gets a verdict per stratum, a power factor beside every null, and its spread diagnostic split on training membership
- **Decision:** Four reporting defects are fixed in code rather than in prose. `d5_report.main` now emits a gate verdict for **all four strata** with `low` pre-registered as decisive, restates each rank-gap null as a power statement carrying the standard error and the batter multiplier it would take to resolve, and splits the predicted-spread diagnostic on training-vocabulary membership instead of pooling cold-start rows into it. The ordering figures are read from the scored arm's own files, and the debiased comparison is reported with its interval as a diagnostic the 2026-08-08 knob entry forbids acting on.
- **Alternatives:** Swapping which single stratum `d5_report.py:113` hard-codes (rejected: reading exactly one stratum is the defect, and a report that gates on `low` alone still cannot say that the RMSE loss is real in `high` and absent everywhere else). Leaving the low-stratum verdict as "null" (rejected: "measured and absent" and "not measurable on this frame" are different claims and only the second is supported). Pooling cold-start rows into the spread diagnostic (rejected: they share the reserved embedding, so their spread measures context variation and nothing about shrinkage). The multi-season pooled low-stratum estimand that would actually resolve the null (declined: it touches the frozen split).
- **Rationale:** Each of the four misstates how well supported a verdict is rather than what the model does, so each is a restatement from files that already exist and none changes a shipped number. The power restatement takes its standard error from the bootstrap interval rather than a formula, so it inherits the batter clustering; a formula-based SE would drop it and understate the factor. The spread diagnostic reverses direction once cold-start rows come out, which turns a shrinkage story into an anti-shrinkage one — the sharper reading, and the one that actually indicts `phase-d-spec.md` §5's decay-ratio argument.
- **Reference:** `src/analysis/d5_report.py` (`power_restatement`, `trained_row_spread`, per-stratum verdicts, `debiased_diagnostic`), `tests/test_d5_report.py`, `tests/test_platoon.py`, `results/phase_d/d5_claim1_paired_phase_d_retrained_head.csv`. Shipped-arm low stratum against C.3-full: rank gap +0.0908 on a bootstrap SE of 0.0687, z = 1.32, so 80% power needs 4.49× the 239 batters — about 1,074, or four to five eval seasons. The RMSE loss is significant in `high` alone (+0.00408 [+0.00229, +0.00592]) and the ordering claim fails there (−0.0364 [−0.0957, +0.0207]), so the shipped arm's low-stratum weighted rank is 0.2573 and the "leads the whole ladder" reading is false. Spread on trained rows only: 0.0374 at low exposure against 0.0281 for regulars, 33% wider, where the pooled read gives 0.0305 against 0.0281 because 42.6% of the scored low stratum is cold start at sd 0.0128 across four distinct values.
- **Revisit if:** an eval frame ever spans enough seasons to give the low stratum ~1,080 batters, at which point the power restatement stops being the finding and the rank gap becomes a measurement; or the reserved row stops being a single frozen embedding, which would make the cold-start split stop separating what it currently separates.

---

## 2026-08-14 — gradient test (b): the low-exposure embedding is displaced, not shrunk, and the exposure-conditional prior declines
- **Decision:** The pre-registered exposure-conditional C.2 prior is **not built**, and the declination is recorded here rather than left silent. Gradient test (b) runs post-rebuild on the `d10` baseline's five seeds, excludes the reserved row per D5-R18(1), and reports both the embedding norm and its projection onto a numerically estimated wOBA-raising direction, because on this trunk the two disagree in sign and only the projection answers the question the diagnostic was built for. This is the D.7 shrinkage diagnostic, and it returns a negative result on the mechanism `phase-d-spec.md` §5 predicted.
- **Alternatives:** Building the prior anyway on the strength of the univariate exposure gradient (rejected: the conditional was pre-registered precisely so that a gradient with a mechanical explanation would not license a comparator change). Substituting observed 2024 wOBA for the C.2 prior-seasons posterior as gradient (a)'s talent proxy, which would have let (a) pass (rejected standing: regressing a residual on its own target manufactures the correlation it then reports). Reporting the gradient and saying nothing about the prior (not available: the 2026-07-28 C.2 prior clause and the 2026-07-29 ladder clause have both fired, so the declination is owed an entry either way). Tuning C.2 to share the model's error rather than to be harder to beat (rejected: that selects the instrument to flatter the measurement).
- **Rationale:** The conditional required the exposure coefficient to survive test (a) with the talent proxy included **and** neither (b) nor (c) to account for the gradient; both clauses fail independently, so the antecedent is false twice over. Test (a) already removed exposure once talent was controlled (2026-08-12), test (c) prices a real hitter-dependent interaction whose interval excludes zero, and test (b) now supplies the mechanism that makes (a)'s univariate gradient need no talent story at all. The finding itself is the opposite of the one the phase expected: low-exposure rows sit **farther** from the origin, not nearer, so AdamW decay cannot be pulling them toward zero, and what is actually shrunk is their component along the direction that moves wOBA — the rows have capacity, and it is oriented orthogonally to the axis the claim is scored on and displaced toward low predicted wOBA. The init confound is ruled out on its own terms: the first quintile's mean norm is 18× the `EMBEDDING_INIT_STD` reference, so these rows have moved a long way from initialization rather than sitting near it.
- **Reference:** `src/analysis/d5_level.py` (`gradient_b`), `results/phase_d/d5_level_attribution.json`, run as `--gradient-b-arm d10_baseline --seeds 0 1 2 3 4 --skip-hbp`. 1,762 trained rows, 541 scored, median 1,244 train pitches, `init_norm_reference` 0.0566, wOBA-direction fit R² 0.9130 / 0.9259 / 0.9310 / 0.9223 / 0.9218. Norm slope per 1,000 train pitches is negative and excludes zero in every seed — −0.018580 [−0.020656, −0.016686], −0.013381, −0.015315, −0.016349, −0.011028, Spearman −0.632 to −0.520 — while the projection slope is positive and excludes zero in every seed: +0.021810 [+0.020480, +0.023341], +0.018940, +0.021338, +0.019766, +0.018154, Spearman +0.586 to +0.629. So all five seeds report shrinkage in norm **false** and shrinkage in the wOBA direction **true**. By exposure quintile (seed 0), median train pitches 30 / 348 / 1,244 / 3,701 / 10,544 give mean norm 1.0005 / 0.7930 / 0.6465 / 0.6034 / 0.6115 and mean projection −0.2181 / −0.2063 / −0.0776 / −0.0120 / +0.0697: norm falls then flattens, projection rises monotonically and only changes sign in the top quintile. Bootstrap resamples hitters rather than (hitter, hand) rows, 2,000 draws.
- **Revisit if:** a future arm gives the embedding an exposure-dependent regularizer or a per-row prior, which would make the norm and the projection move together again and put the decay-ratio argument back in play; or the claim-1 table from Step 5 shows the low stratum's loss is driven by direction rather than by level, at which point the orthogonal-capacity finding stops being a diagnostic and becomes the thing to fix.

---

## 2026-08-15 — claim-1 for all eight `d10` arms: four beat baseline, every margin sits inside one arm's own seed range, and the ledger is retired as an architecture instrument
- **Decision:** Every `d10` arm now carries a claim-1 number, and frozen rule #2's verdict is read from it rather than from held-out per-pitch log loss. The pre-registered decisive stratum is `all`, the test is a paired bootstrap clustered on batter at 2,000 draws whose interval must exclude zero, and per-seed spread is reported as context only. Under that rule `block`, `bilinear`, `meanweight` and `dim64` beat baseline and `nospray` and `dim16` resolve as nulls. **The shipped arm does not change in this phase:** D.5 is attribution, `block` trains and scores on a different build (35 context columns against 46), and its margin is smaller than the baseline's own between-seed range, so adoption is a Phase E action taken with a re-scored ladder rather than a table row. The `nospray` arm is scored by collapsing `V` over the empirical train-window P(spray | ev, la), backing off to the global spray marginal in cells no train ball reached, with those cell masses returned **unshrunk**.
- **Alternatives:** Continuing to read architecture verdicts off `best_val_loss` or `reference` (rejected on the measurement: over the seven arms whose objectives are comparable, the Spearman correlation between reference-log-loss rank and claim-1 rank is exactly 0.000, so the ledger carries no ordering information about the claim it was being used to settle). Pre-registering the `low` stratum as decisive for arm selection (rejected 2026-08-12: one number is needed to pick an architecture and the aggregate is the least noisy, with a low-stratum win against an aggregate loss recorded as an explicit finding instead). Per-seed runs for all eight arms rather than baseline alone (rejected: ~40 further composition runs, so every other arm's interval carries the stated assumption that its seed noise resembles baseline's). Scoring `nospray` against a uniform spray distribution (rejected: spray depends strongly on launch angle, so uniform prices the ablation against a distribution no ball follows). Shrinking the spray cell masses the way `table` is shrunk (rejected: `table` is a conditional over outcome classes and the masses are not, so pooling them would move balls into spray bins no ball reached). Adopting `block` as the shipped arm on this table (declined per the phase's no-new-architecture scope, recorded rather than silent because the pre-registered rule does select it).
- **Rationale:** The gap the step closes is that every architecture verdict in the phase rested on a metric its own report says "says which model predicts PITCHES better and nothing at all about whether it projects HITTERS better than empirical Bayes," and the zero rank correlation makes that concrete rather than rhetorical — `baseline` is first on log loss and sixth on claim 1, `block` is sixth and first. The size of the result is what governs its reading: the whole spread from best arm to baseline is 0.00078 wOBA while baseline's five seeds span 0.00186 in the same stratum, so architecture is resolved but is not what limits claim 1, which makes the phase's null attributable rather than rescued. `block` — B.2's five flagged pitcher release and spin features removed — winning claim 1 while ranking sixth on log loss is the coherent reading of both numbers at once: those features help predict the next pitch and do not transfer to next-season hitter projection. `nospray`'s null says the spray head carries no measurable claim-1 signal, and its scored fidelity says it is not free: strikeouts land 2.7% low against the 2% band where baseline is 1.7% low and passes.
- **Reference:** `src/analysis/d5_arms.py`, `src/model/query.py` (`league_spray_ratio`, `spray_kernels`, `_conditionals`, `expected_woba_naive`), `src/model/query_tables.py` (`fit_outcome_table`'s fourth return), `tests/test_query.py`, `results/phase_d/d5_arms_d10.csv`, `d5_arms_paired_d10.csv`, `d5_arms_verdict_d10.json`, `d5_arms_baseline_per_seed_d10.csv`. Decisive stratum `all`, 1,149 hitter-side rows and 604 distinct batters, negative favours the arm: `block` −0.000780 [−0.001066, −0.000482], `bilinear` −0.000625 [−0.001048, −0.000156], `meanweight` −0.000441 [−0.000766, −0.000106], `dim64` −0.000284 [−0.000481, −0.000082], `nospray` −0.000102 [−0.000474, +0.000269], `dim16` −0.000029 [−0.000395, +0.000341], `invfreq` +0.012738 [+0.009496, +0.015781]; baseline RMSE 0.047931. Baseline per-seed spread, context only: low 0.05944 sd 0.00044 range 0.00094, medium 0.04945 / 0.00075 / 0.00200, high 0.04380 / 0.00098 / 0.00245, all 0.04815 / 0.00076 / 0.00186. `block` by stratum −0.000057 [−0.000666, +0.000568] low, −0.000946 [−0.001648, −0.000243] medium, −0.001022 [−0.001344, −0.000676] high, so it wins where hitters have history and is null where they do not. No arm wins `low` while losing the decisive stratum. Mean reference log loss over five seeds: baseline 1.025800, bilinear 1.026030, dim64 1.026068, dim16 1.026534, meanweight 1.027236, block 1.027548, invfreq 1.051688; `nospray`'s 0.814064 is incomparable by construction, since its objective sums fewer log-loss terms. Verification for the spray-less path: the factored form matches the literal (M, 24, 24, 24) joint, P(spray | ev, la) sums to 1 in every cell including unobserved ones, and passing the mass table to a spray-headed arm leaves its predictions bit-identical, so the seven arms scored before the fix stay comparable to the eighth. 316 tests pass.
- **Revisit if:** Phase E re-scores the ladder on the no-block build, at which point `block`'s selection stops being a recorded table row and becomes an architecture decision with the ladder numbers to support it; or a future arm's claim-1 margin exceeds the between-seed range of a single arm, which is the first margin this instrument could call large rather than merely resolved; or `nospray` is revisited for cost rather than signal, since removing the head is a real reduction in the output space and its only measured price is league fidelity.

---

## 2026-08-18 — the 2025 final run moves after Phase O; Phase E evaluates on the 2024 frame only
- **Decision:** Phase E computes every validation and every effectiveness number on the **2024** frame. The refit on 2015–2024 and the single 2025 report are moved to the end of Phase O and happen once, on whatever configuration Phase O settles. The refit itself is unchanged and remains mandatory before any 2025 number is quoted. `assert_not_test_season` keeps refusing 2025 outside `--final-run` at all six call sites, and nothing in Phase E passes that flag.
- **Alternatives:** Following the architecture plan's literal order — report 2025 in Phase E, then optimize in Phase O (rejected: Phase O would then have no untouched season to prove a gain on, so either its result is never tested out-of-sample or 2025 is scored twice and stops being a test season; both void the protocol the phase order exists to protect). Scoring 2025 now from the 2015–2023 checkpoints without a refit (rejected on the plan's own number: 75% of the 2025 low-exposure stratum would sit on the reserved zero embedding row against 42.7% with the refit, so the headline stratum would be measuring the context tower and would not be comparable to any Phase C baseline, all of which were produced at 42.7%). Refitting twice, once for Phase E and once after Phase O (rejected: the second report is test-set reuse regardless of how the first is labelled, and a discarded first report is not recoverable as unspent). Dropping the refit requirement (not available: it is the architecture plan's §5 protocol and its justification is numerical, not stylistic).
- **Rationale:** Three lines of the architecture plan cannot all hold in the order written. Line 167 places the 2025 claim-1 table in Phase E, line 169 places optimization after it, and line 120 requires that only the **winning** configuration be carried across the refit. Under the written order the winning configuration does not yet exist when the refit fires. The resolution keeps every substantive requirement — one refit, one 2025 report, the 42.7% cold-start share — and changes only which phase boundary they sit behind, so the test season is spent last, on the final configuration, which is what the one-shot rule was for. The Oct 1 deadline does not force the earlier order: line 175 asks the abstract for "a **preliminary** claim-1 table", and the 2024 table is that. The manifest makes `docs/decision-log.md` the authority where it and the architecture plan disagree, so this entry is the mechanism the plan itself names for the change.
- **Reference:** `~/os/knowledge/library/baseball-research/Layer1_Architecture_Plan_v2.md` lines 119–120 (frozen splits and the selection frame, quoted: "then refit the winning configuration from scratch on **2015–2024** and report on **2025**. Only the configuration — including the epoch count — carries across the refit"), 167 (Phase E contents), 169 (Phase O contents), 175 (timeline anchor); `docs/phase-e-spec.md` §2; `src/analysis/claim1_eval.py:655` and its six call sites (`d5_level.py:460`, `d5_report.py:200`, `c_report.py:349`, `d5_arms.py:168`, `query.py:864`, `train.py:319`). Not a §3a search trigger: this is a phase-ordering call on this project's own protocol, not a technique adoption, and no external precedent settles it. The argument is internal and is stated in full above.
- **Revisit if:** Phase O returns no adopted change, in which case the configuration Phase E evaluated is already final and the refit could fire immediately with nothing lost; or the abstract deadline is moved forward such that a 2025 number is required before Phase O can complete, in which case the choice becomes an explicit trade of protocol for deadline and is recorded as one rather than absorbed silently.

---

## 2026-08-18 — the D.5 credit rule is repaired: a paired delta is graded against the spread of the paired delta, not the spread of a level
- **Decision:** A composition fix is credited when the mean paired difference across seeds exceeds the between-seed spread **of that paired difference**, measured on identical seeds with the knob off and on. The rule this replaces graded a paired delta against the between-seed spread of a **level** (0.00587 for the walk rate). The credit verdict and the 2% band verdict are always reported together. This amends the 2026-08-12 level-bias entry, governs Phase E forward, and does **not** retroactively re-credit D.5's published verdicts.
- **Alternatives:** Leaving the rule as written (rejected: it is not conservative, it is inoperative — the seed effect cancels inside a paired difference and does not cancel inside a level, so the comparison denominator is the wrong order of magnitude and essentially no real fix can clear it). Widening or narrowing the level spread by a fudge factor (rejected: the defect is that the two sides measure different quantities, and no scalar makes a level spread into a paired-delta spread). Re-crediting D.5's count offsets under the corrected rule inside this window (declined: that reopens a closed phase's published verdicts and is owed its own entry with its own numbers, not a footnote to a repair). Dropping the credit rule and reporting only the 2% band (rejected: the band says whether a rate is acceptable and says nothing about whether a change was real, and D.5's whole difficulty was moves smaller than their own noise).
- **Rationale:** Both sides of a paired comparison share the seed, so the seed's contribution to the level enters both terms and subtracts out. What remains in the difference is the knob's effect plus the seed-by-knob interaction, which is small. Grading that residual against the spread of the un-differenced level asks the effect to exceed noise that the pairing already removed, which is why every D.5 move came back uncredited regardless of size. The repair costs nothing new to compute: the per-seed compositions D.5 already persists carry both terms, so the corrected denominator is a re-read of committed files rather than a re-run. Reporting the credit verdict and the band verdict together is not decoration — they answer different questions, a knob can move a rate by a real and measurable amount and still leave it outside tolerance, and D.5's summary language collapsed the two.
- **Reference:** `docs/decision-log.md` 2026-08-12 (the entry this amends, which records the defect without repairing it); `src/model/query.py:89` `FIDELITY_TOLERANCE` and the per-seed composition block in `main`; `results/phase_d/d5_diagnostics_d10_baseline_s0.json` through `_s4.json` (the per-seed files the corrected denominator is computed from); `docs/phase-e-spec.md` §9. Not a §3a search trigger: this is internal statistical bookkeeping on a paired-versus-unpaired variance comparison, not a technique adoption, and the argument is the standard one that a paired difference removes the shared component's variance. Verified against this project's own per-seed files rather than asserted.
- **Revisit if:** a future arm is scored with fewer than three seeds, at which point the paired-difference spread has too few draws to be a denominator and the rule needs a stated fallback; or D.5's count offsets are re-graded under this rule, which is the entry this one explicitly does not write.

## 2026-08-18 — the walk gap is 54% population, 24% composition structure, and 0% resampler

- **Decision:** the shipped `+6.17%` walk-rate fidelity failure is decomposed and reported as
  three parts rather than carried as one number. E.1's population-matched control accounts for
  54% of it (matched observed walk rate 0.08312 against the shipped unmatched 0.08042, leaving
  a residual excess of +2.71%). E.10 accounts for 51.6% of that residual (+0.001166) as a
  structural property of the independent-pitch count chain, measured model-free. E.7-E.9 price
  both channels of the pitch resampler and find them net zero (-0.00003). Roughly 22% of the
  original gap has no named owner and is recorded as open.
- **Alternatives:** carry the +6.17% headline unattributed into Phase O, which is what D.5
  shipped; or stop after E.1's population control, which halves the number without explaining
  the rest; or attribute the residual to the swing head, which was the handoff's stated
  suspect and which E.6 refutes on 705,344 real held-out pitches (+0.27% overall, and
  over-predicting swings in every three-ball count, the wrong sign to make walks).
- **Rationale:** standing risk §8 makes composition fidelity a gate on reading any ablation
  result, so an unattributed failure blocks the phase. A gap with three measured parts and one
  named open residual is a gate a reviewer can evaluate; a single number is not. E.10 is the
  load-bearing piece because it is model-free: every transition in it is a counted frequency
  from real pitches, so the +1.44% relative walk bias it returns cannot be blamed on anything
  Phase D trained, and no retrain can fix it.
- **Reference:** none — internal measurement against this project's own data. No §3a trigger
  fires: no named technique is adopted, and the decomposition is arithmetic on the project's
  own chain, not a modeling choice with a literature answer.
- **Revisit if:** the composition is ever changed to condition on within-plate-appearance
  history (a sequence model, or a count chain conditioned on the pitch that preceded it), which
  would move the E.10 term directly; or if the unnamed 22% is closed by a later diagnostic.

## 2026-08-18 — the swing head is exonerated as the owner of the walk gap

- **Decision:** the swing head is recorded as calibrated and removed from the suspect list.
  Scored on 705,344 real held-out 2024 pitches with the resampler removed from the
  measurement entirely, the 5-seed ensemble predicts a swing rate of 0.479256 against an
  observed 0.477975, a relative gap of +0.27%; trained hitters +0.14%; every handedness cell
  within +0.5%.
- **Alternatives:** treat the handoff's framing as settled and open a Phase O retrain item on
  the swing head, which is what the D.5 handoff implied.
- **Rationale:** §8 pre-registered the signature a head-owned gap would leave — a swing
  SHORTFALL concentrated in three-ball counts, where an extra take converts directly into a
  walk. The measured result is the opposite sign in all three: 3-0 +8.7% (n=6,807), 3-1 +0.9%,
  3-2 +2.1%. A head that swings too often cannot manufacture an excess of walks. Recording
  this closes a retrain item that would otherwise have been opened on a suspicion.
- **Reference:** none — internal measurement. Deep-ensemble averaging of probabilities rather
  than logits follows the project's own frozen architecture decision #5 and `query.expected_woba`.
- **Revisit if:** the head is retrained, or the eval season changes, since calibration is a
  property of a fitted model and not of the architecture.

## 2026-08-18 — the walk excess is a spread problem, not only a level problem

- **Decision:** the walk gap is treated as claim-1 relevant rather than as a fidelity-only
  issue. E.2's noise-corrected compression coefficient is `b = 0.642` for walks (k 0.773,
  bip 0.736, hbp 0.095): the model expresses about 64% of the true between-hitter spread in
  walk rate, so the error is not a uniform level shift that a constant would absorb.
- **Alternatives:** read the naive regression of `modelled - observed` on `observed`, which
  returns a compressive slope even at zero true compression, because target sampling noise in
  the regressor produces a mechanical slope of `-var(e)/var(o)`. Reporting that alone would
  have manufactured the finding.
- **Rationale:** a pure level bias would leave hitter ranking intact and would be a
  presentation problem. Compression of the between-hitter spread is exactly the quantity
  claim 1 is about, so it belongs in the claim's evidence and not in a fidelity appendix. The
  errors-in-variables correction uses the binomial noise variance `p(1-p)/n`, which is
  available per hitter and needs no extra assumption.
- **Reference:** none — the errors-in-variables correction is standard regression algebra
  applied to this project's own counts; no technique is adopted from outside.
- **Revisit if:** the walk gap is closed, since compression measured against a biased level
  is not the same quantity as compression measured against a calibrated one.

## 2026-08-18 — the platoon adoption rule is not met on 2024

- **Decision:** the model's platoon differential is **not** adopted over Route A (overall
  skill plus the league-average side split). No stratum's paired 95% interval excludes zero on
  rank: low +0.0436 [-0.0396, +0.1185], medium -0.0313, high -0.0076, all -0.0029. RMSE favours
  the model only pooled, -0.001148 [-0.002160, -0.000166]. The result is kept and reported.
- **Alternatives:** read the pooled RMSE interval alone and call it adoption, which would rest
  the claim on the one metric-stratum pair out of eight that clears; or defer the reading until
  after the walk gap is fixed, which would mean entering Phase O with no evaluation of the
  claim the project exists to make.
- **Rationale:** the sharpest statement is the decomposition, not the interval: 81.7% of the
  model's predicted differential variance is the batter-stand effect alone, against 10.9% of
  the observed differential's. Route A is 100% by construction, so the model has moved about
  a fifth of the way from "the league split applied to everyone" toward a hitter-specific
  differential. That is a real but small departure, and frozen rule #1 makes beating baselines
  only in aggregate a null. Honest reporting of non-results is manifest working style #7.
- **Reference:** none — the adoption rule is this project's own, pre-registered in
  `docs/phase-e-spec.md` §7 before any platoon number was read.
- **Revisit if:** the walk gap is closed and the C-ladder is re-scored under `d10`, since both
  change the level the differential is read off; or if the eventual 2015-2024 refit changes the
  low-exposure stratum, where 42.7% of hitters currently sit on the reserved zero row.

## 2026-08-18 — model confidence tracks exposure, and is reported unscored

- **Decision:** E.4's calibration is recorded as a described property, not as a scored gate:
  regression slope of observed on modelled is 0.529 in the low-exposure stratum (z = -4.90
  against 1), 0.664 medium, 0.998 high, 0.841 pooled.
- **Alternatives:** treat the low-exposure slope as a failure and open a remediation item.
- **Rationale:** the pattern is exactly what the architecture predicts — the embedding is
  under-dispersed where hitters lack history and calibrated where they have it — so it
  describes the shrinkage working, not a defect. It stays unscored because scoring a
  query-machinery property on a claim-1-adjacent metric is the circularity `phase-d5-spec.md`
  §9 forbids: the scorer is the model.
- **Reference:** none — judgment call on reporting status, not a technique adoption.
- **Revisit if:** the exposure-conditional prior declined in D.5 is ever adopted, since it
  targets precisely the low-exposure under-dispersion this measures.
