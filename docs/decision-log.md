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
