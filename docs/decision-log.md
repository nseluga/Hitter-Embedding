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
