# Decision Log — Hitter Embedding

Append-only. Format fixed by `~/os/knowledge/frameworks/research-standards.md`
§4 — `/research-partner` and `/research-review` both parse against it.

Entries are written in Simplified Technical English and keep the project's own
domain vocabulary: one idea per sentence, active voice, jargon intact. One to
three sentences per field.

An entry whose `Decision:` states an outcome rather than a choice carries
`Finding: ` after the em dash in its title. Grep `— Finding:` for the results
record, and the rest for the design record.

---

## <YYYY-MM-DD> — <decision title>
- **Decision:** what was chosen
- **Alternatives:** what was considered and rejected
- **Rationale:** why, in terms a skeptical reviewer would accept
- **Reference:** external source backing the decision (optional)
- **Revisit if:** the condition under which this should be reopened

---

## 2026-07-14 — Statcast raw-snapshot storage design
- **Decision:** Raw Statcast is frozen to `data/raw/statcast/snapshot_<date>/season=<YYYY>.parquet` — per-season, all columns, immutable. A committed `manifest.json` holds pull metadata. Data is gitignored; script and manifest are committed.
- **Alternatives:** A monolithic parquet file (forces full-memory loads). Column pruning at pull time (unrecoverable without re-pull).
- **Rationale:** Date-stamped snapshots handle Statcast's retroactive revisions. Immutable raw data lets the pitch-event table re-derive without re-fetching.
- **Revisit if:** the snapshot outgrows local disk.

---

## 2026-07-15 — Statcast cleaning spec for the modeling pitch table
- **Decision:** The modeling table covers regular season only. It drops position-player pitches, pitchouts, automatic balls/strikes, and bunts. Rows missing or physically impossible on core context are dropped. Optional spin context stays, with missingness indicators. Filters apply only to the modeling table, never to evaluation targets.
- **Alternatives:** A minimum-PA hitter floor (rejected: deletes the low-exposure population the thesis targets). A velocity-based position-player rule (rejected: misclassifies hard-armed position players). Dropping spin columns (deferred to a Phase B ablation).
- **Rationale:** Every filter is backed by the profiling notebook. The modeling-vs-target split keeps sharpening filters from biasing ground truth.
- **Revisit if:** Phase B feature screening changes the retained context set, or target construction needs a filtered field.

---

## 2026-07-17 — Contact-quality label domain
- **Decision:** The contact-quality head (EV, LA, spray) is labeled on balls in play only. `ev`/`la`/`spray` are null elsewhere.
- **Alternatives:** Labeling all contact with EV (rejected: fouls carry EV/LA but no spray and no batted-ball outcome, giving ragged masking and no run-value mapping).
- **Rationale:** Fouls are contact for the whiff head, but they are non-terminal count transitions in the Markov composition. Only in-play balls carry the run-value outcome the quality head feeds.
- **Revisit if:** the outcome space or run-value mapping (§1.5) changes to need foul-ball measurements.

---

## 2026-07-17 — Spray angle derivation
- **Decision:** Spray = `atan((hc_x − 125.42)/(198.27 − hc_y))`, in degrees, mirrored so positive equals pull for both hands.
- **Alternatives:** Empirically calibrating the home-plate origin (dropped once constants were sourced). Raw field-side angle without mirroring (rejected: not batter-intrinsic).
- **Rationale:** MLB does not publish the coordinate origin. The sourced constants carry a real-data regression guard: field mean ≈ 0 and pull-mean > 0 confirm the export matches scale.
- **Reference:** abdwr3e App. C, BGSU, Weise — three independent corroborations of formula and constants.
- **Revisit if:** the near-plate artifact needs clipping — decided 2026-07-29 below.

---

## 2026-07-17 — Walk-forward split frozen
- **Decision:** The split is contiguous walk-forward: train 2015-2023, val 2024, test 2025, single fold. It is frozen in `src/config/split_config.json` and validated on load.
- **Alternatives:** Random k-fold (rejected: leaks a hitter's future PAs into his own ID embedding, manufacturing a positive result). Rolling multi-fold (deferred: multiplies compute against the <$200 budget with no gain on the axis we grade on). Val/test gap season (rejected: wastes data, breaks Phase B's same-season window trade).
- **Rationale:** Projection is a forecasting task, so eval must mirror deployment. Freezing the split before any model comparison pre-registers the held-out season. Contiguity minimizes distribution shift, so the metric measures projection skill, not regime drift.
- **Revisit if:** never, for this project (frozen rule). A new fold requires a new entry naming this one.

---

## 2026-07-21 — Stabilization reported at two thresholds (r=0.5 and r=0.7)
- **Decision:** Every stabilization point is reported at two thresholds: r=0.5, the equal-weight-with-prior point the thesis trades in, and r=0.7, the stricter "reliable measurement" convention.
- **Alternatives:** Single r=0.5 (rejected: invites an unanswered "why 0.5?" and hides that half the variance is still noise). Single r=0.7 (rejected: not the quantity the shrinkage argument uses).
- **Rationale:** The two thresholds answer different questions. Reporting both preempts the threshold objection and allows a cross-check against Carleton's published r=0.7 numbers.
- **Reference:** Carleton, "Reliably Stable (You Keep Using That Word)"; FanGraphs, "A Long-Needed Update on Reliability."
- **Revisit if:** the paper's reviewers want a different reliability convention.

---

## 2026-07-21 — Variance-components estimator added alongside split-half
- **Decision:** `stabilization.py` adds a one-way random-effects estimator: analytic reliability(n), stabilization point, bootstrap CI, over all hitters. Split-half is a cross-check; variance-components is the headline on divergence.
- **Alternatives:** Split-half only (rejected: survivorship-biased at large n, where wOBA lives, and gives no CI). Mixed-model REML (rejected for now: heavier; method-of-moments ANOVA matches Cronbach's alpha at a fraction of the code).
- **Rationale:** Split-half at large n keeps only hitters reaching n, halving between-hitter signal variance and doubling n*. Variance-components uses every hitter, removing that artifact.
- **Reference:** FanGraphs, "A New Way to Look at Sample Size (Math Supplement)" — Cronbach-alpha signal/noise decomposition.
- **Revisit if:** wOBA divergence is heteroscedasticity in VC assumptions, not survivorship in split-half.

---

## 2026-07-21 — Matched-slice and across-time reporting for the process-vs-outcome comparison
- **Decision:** B.1 also reports process metrics sliced by pitcher hand, matching the side-specific outcome slice. It adds a sequential split alongside the random one.
- **Alternatives:** Pooled process vs. side-specific outcome only (rejected: apples-to-oranges — some of the gap is the split, not the process/outcome distinction). Random split only (rejected: measures within-sample consistency, which flatters the projection-relevant number).
- **Rationale:** Matched slicing kills the comparison-asymmetry objection, since process stays fast even split by hand. The gap survives every slicing and split choice; only the absolute points move.
- **Reference:** Carleton, "Reliably Stable" — sequential splits drop reliability relative to same-circumstance splits.
- **Revisit if:** the sequential split shows a large systematic across-time degradation on the headline metrics.

---

## 2026-07-27 — C.1 trailing-average design: 3-season window, both shrinkage variants
- **Decision:** Phase C.1 uses a 3-season trailing window, in two reported variants: `raw` (unshrunk trailing side-specific wOBA) and `bucketed` (shrunk toward league average by PA bucket).
- **Alternatives:** An all-prior-seasons window (rejected on measurement: it barely helps low-exposure hitters and costs veterans data). A single variant (rejected: it collapses the decomposition the raw/bucketed/C.2 sequence exists to show).
- **Rationale:** The window is chosen by measurement, not preference. The two variants together isolate the value of shrinkage itself, before C.2 tests doing it properly.
- **Revisit if:** the raw variant's low-exposure advantage is an artifact of eval-season target noise, not real ordering signal.

---

## 2026-07-27 — Noise-floor deconvolution added as a companion to the claim-1 metric
- **Decision:** `claim1_eval.py` now reports a noise floor and deconvolved model RMSE, alongside the frozen §5.2 PA-weighted RMSE, plus a skill-score helper. This is additive only; the frozen metric is unchanged.
- **Alternatives:** Raw RMSE alone, as §5.2 specifies (rejected: see Rationale). Estimating the floor by simulation (rejected: the analytic form is exact and free).
- **Rationale:** The held-out target is itself a small-sample measurement. Errors add in quadrature, and target noise dominates RMSE enough to compress real model differences into what reads as rounding. The floor was independently validated against B.1's separately-estimated signal variance.
- **Revisit if:** a model's raw RMSE lands materially below its estimated floor, in any stratum.

---

## 2026-07-27 — Pitchers' own at-bats excluded from every hitter-talent quantity
- **Decision:** `eval_targets.py` drops batters who are primarily pitchers per season, by batters-faced vs. PA, so two-way players stay hitters. This applies to every hitter-talent quantity, but not to the evaluation target table, which stays built from the complete source.
- **Alternatives:** Filtering in `clean.py` (rejected: wrong table, and the affected quantities bypass it by design). A career-level or PA-only rule (rejected: misses role changes and drops genuine call-ups).
- **Rationale:** Pre-2022 NL pitchers batting are a low-wOBA population that inflates between-hitter signal variance in any prior or league average. Removing them moved B.1's stabilization points and shifted C.1's stratum boundaries.
- **Revisit if:** a future season reintroduces pitchers batting in volume, or the two-way threshold misclassifies a genuine two-way player.

---

## 2026-07-27 — C.2 estimand: shrink the two sides jointly, not the platoon split
- **Decision:** C.2: bivariate empirical Bayes over (talent vs LHP, talent vs RHP), per batter type, with cell-specific variance; cross-side covariance is estimated only on durable hitters.
- **Alternatives:** Split-level (The Book): rejected as estimator, kept as scored reference — variance needs subtracting an unstable noise term. Rate-level (ρ=0): rejected as indefensible, kept as nesting gate. Unrestricted covariance: rejected on measurement, unstable below ~50 PA.
- **Rationale:** The two parameterizations are the same model, rotated: the choice is identifiability. PA vs LHP/RHP are disjoint, so the joint form needs no noise subtraction; the split form must cancel a term far larger than the quantity sought.
- **Reference:** Efron & Morris (1972); the multivariate Fay–Herriot EBLUP.
- **Revisit if:** the ρ interval tightens enough to separate our estimate from The Book's implied value.

---

## 2026-07-28 — C.3 design: hitter x context aggregates, two feature sets, inner-val season
- **Decision:** C.3: an XGBoost model on unit claim-1 scores — one row per (batter, season, pitcher hand) — over the hitter's own prior-window rates by context. Feature sets: `outcome` (trailing wOBA/PA), `full` (adds process). Hyperparameters pre-registered, early-stopped on 2023 only, refit on all seasons at that round count.
- **Alternatives:** The 48-dim context vector as features (rejected: no hitter identity, per B.2's finding). Early stopping or hyperparameter search on the eval frame (rejected: leaks into the Phase C scoring frame). A single feature set (rejected: can't isolate where skill comes from).
- **Rationale:** The plan specifies the model, not the row unit; the metric settles it. The outcome/full pair is the first hitter-level test of whether process signal projects, not just stabilizes.
- **Revisit if:** Phase D needs a tuned-GBM upper bound, not a pre-registered one.

---

## 2026-07-28 — C.2's prior mean stays exchangeable; the confound is reported, not corrected
- **Decision:** C.2 keeps a single exchangeable prior mean per (batter type, pitcher hand). No exposure-conditional prior is built. The resulting confound in the C.2/C.3 comparison is documented instead.
- **Alternatives:** Adding an exposure-conditional C.2 variant (rejected as scope — Phase C's job is to produce the incumbent bar, not iterate on it).
- **Rationale:** Prior exposure correlates with talent, so C.2 systematically over-predicts low-exposure hitters. C.3 has exposure as a feature and does not. An oracle-recentering check shows most of C.3's apparent low-exposure advantage is this level effect, not ordering skill.
- **Revisit if:** Phase D's margin over C.3 turns out to be the same level effect, requiring an exposure-conditional prior for a fair comparison.

---

## 2026-07-29 — Ordering claims get a paired bootstrap, and the resampling unit is the batter
- **Decision:** `paired_rank_difference` adds the rank counterpart of `paired_rmse_difference`; both resample batters, not (batter, hand) rows. No ordering claim from two bare rank correlations.
- **Alternatives:** Leaving ordering claims unquantified (rejected: held the noisier metric to a lower bar than the calibration metric). A permutation test (rejected: the paired bootstrap is the project's idiom). Row resampling (rejected: a batter's two rows share his talent, health, and park).
- **Rationale:** Two absolute numbers scored against the same noisy answer key cannot resolve a difference — only the paired difference can. Rank correlation needs the interval more; ranks carry no PA weight. Applied, it retracts the C.3/C.2 low-stratum ordering gap: neither model demonstrably orders low-exposure hitters better.
- **Revisit if:** a future eval frame has a materially different rows-per-batter structure.

---

## 2026-07-29 — MIN_EVAL_PA logged, broken out per stratum, and reported as a sensitivity
- **Decision:** The 25-PA eval-season floor, in place since 2026-07-27 but never logged, now reports its drop per stratum, swept across (10, 25, 50). Strata are assigned before the filter, so dropped groups stay attributable without leakage.
- **Alternatives:** No floor (rejected: scoring against a 5-PA wOBA measures the answer key, not the model). Inverse-probability weighting (rejected as disproportionate; the sweep answers the same question). An aggregate drop count alone (rejected: hides a filter that is anything but uniform).
- **Rationale:** The filter censors on eval-season playing time, decided after the projection, partly by hitter performance — the deployment-bias hazard the module guards against elsewhere. At 25 PA it removes 36.6% of the low stratum, against 6.2% of the high. The RMSE headline holds at every cut; ordering reverses sign across them.
- **Revisit if:** the low-stratum drop share moves materially, or the RMSE margin changes sign across the sweep.

---

## 2026-07-29 — MIN_EVAL_PA frozen at 25 — SUPERSEDED same day, see below
- **Decision:** `MIN_EVAL_PA = 25` is frozen for all remaining claim-1 numbers. The entry below supersedes it. This entry stays as the record of the overridden reasoning.
- **Alternatives:** Moving to 10 (rejected here, adopted below). Moving to 50 (rejected: it censors over half the low stratum and costs the power the claim depends on).
- **Rationale:** 25 entered alongside `claim1_eval.py` itself, days before C.2 and C.3 existed. It is pre-registered with respect to every comparison it governs. Moving to 10 after seeing that 10 gives C.3 its largest low-stratum margin would select the frame that most flatters the model under test.
- **Revisit if:** superseded below.

---

## 2026-07-29 — MIN_EVAL_PA moved to 10; supersedes the freeze at 25 above
- **Decision:** `MIN_EVAL_PA = 10`. 25 and 50 stay in the committed sweep; every headline is still reported across a 3x censoring swing.
- **Alternatives:** Holding 25 on pre-registration grounds (above). Co-primary 10 and 25 (rejected: same evidence, more machinery). Dropping the filter (rejected: a 3-PA wOBA is a coin flip).
- **Rationale (Nate's):** lower the cut "because we want to include the guys who have less games played since that's our target audience" — at 25 the filter removed 36.6% of the low-exposure stratum, against 18.3% at 10, measuring a different population than claim 1's target. The margin-inflation hazard at 10 was known beforehand; the committed sweep shows the RMSE claim holds at every threshold.
- **Revisit if:** never for a better score. Reopen only if the drop share moves materially from ~18%, or the RMSE margin's sign becomes threshold-dependent.

---

## 2026-07-29 — Near-plate spray artifact nulled at label time
- **Decision:** `|spray| > 90` is nulled in `labels.py`. The spray label carries no survivors past that value. This discharges the 2026-07-17 spray entry's revisit clause.
- **Alternatives:** Clipping to the limit (rejected: a clipped value is a fabricated measurement at exactly the boundary). Keeping the raw angle (rejected: physically impossible values). Deciding it per-analysis downstream (rejected: guarantees drift).
- **Rationale:** A fair ball lies inside the ~90-degree foul-line wedge. So `|spray| > 90` is the angle formula blowing up near the plate origin, not a real direction. It affects 0.95% of in-play balls. EV and LA come from launch tracking rather than hit coordinates, so only spray is affected.
- **Reference:** abdwr3e App. C, BGSU, Weise (2026-07-17 entry).
- **Revisit if:** a future Statcast revision changes the hit-coordinate origin.

---

## 2026-07-29 — B.2's six flagged context features recorded as UNDECIDED
- **Decision:** B.2 screens the 48 context features with one XGBoost head per process outcome, trained on TRAIN, early-stopped on VAL, scored by out-of-sample permutation importance. A feature is kept if it clears 1% of a head's baseline metric, or is frozen-in. Ten are kept; six — effective_speed, release_pos_z, spin_axis, release_spin_rate, release_pos_y, release_extension — are flagged and left UNDECIDED, not auto-dropped. Full importances: `results/phase_b/`.
- **Alternatives:** Dropping the six on B.2's evidence (rejected: the fits were budget-capped, not converged, and permutation importance is deflated by collinearity — exactly what the flagged six are). Keeping all 48 silently (rejected: the de-facto state this entry refuses to leave unremarked).
- **Rationale:** A GBM null does not prove another model class cannot use a feature. B.2 deferred the six to the DL common-window ablation, but Phase B steps 3-5 (bat-tracking placement) dissolved when bat-tracking left v1. §4 still binds the six: the ablation must become a Phase D context-tower ablation, or frozen rule #2 fails.
- **Revisit if:** settled by a re-run with converged fits and block permutation of the correlated groups, or superseded by a Phase D context-tower ablation. The bat-tracking exclusion that dissolved steps 3-5 still needs its own entry, in Nate's words.

---

## 2026-07-29 — One PA unit for scoring: the wOBA denominator, not total plate appearances
- **Decision:** Every quantity in `claim1_eval` weighting an observation uses the wOBA denominator, not total PA — the eval floor, RMSE weight, noise-floor weight, and paired bootstrap. `score()` reports both, so the gap stays visible.
- **Alternatives:** Standardising on total PA (rejected: it does not govern the weighted thing's precision). Leaving the mixture documented (rejected: a units boundary inside the referee is where drift hides).
- **Rationale:** `Var(observed wOBA) = within-group variance / denominator`, so the denominator sets precision. Total-PA weighting was only ~1% larger, but systematically: intentional walks accrue to the best hitters, sac bunts to the weakest. The correction changed no conclusion or interval.
- **Revisit if:** never — any new weight or threshold in `claim1_eval` must use the denominator, and the gates enforce it.

---

## 2026-07-29 — The Phase C baseline ladder is a decomposition, not a horse race
- **Decision:** Every adjacent pair on the ladder differs in exactly one respect, so each ingredient prices separately and Phase D's margin can be attributed, not just observed.

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

- **Alternatives:** A single strong baseline (rejected: an unattributable margin). Dropping C.1-raw as obviously bad (rejected: it proves shrinkage fixes level, not order). An information-matched per-pitch GBM with hitter identity (deferred to Phase D: its aggregation layer is Phase D's own query machinery).
- **Rationale:** Frozen rule #1 is role-matched — what a competent analyst does today; §4's rule is information-matched and isolates components. Phase C owns the first, Phase D the second. The payoff is the C.2 → C.3-outcome null: a GBM given exactly what empirical Bayes has does not beat it, foreclosing "your GBM won because it is a machine-learning model" without argument.
- **Reference:** architecture plan §3, §4; manifest frozen rules #1-#2.
- **Revisit if:** Phase D's margin turns out to be the same exposure-conditional level effect; then C.2 needs an exposure-conditional prior mean built first.

---

## 2026-07-30 — Phase D ordering gate: beat both baselines, or claim nothing
- **Decision:** No numeric rank threshold is pre-registered. An ordering claim holds only if Phase D beats both C.2 and C.3-full on rank correlation in the claimed stratum, by `paired_rank_difference` with batter clustering and a 95% interval excluding zero. The RMSE gate stays unchanged, set against C.3-full.
- **Alternatives:** Gating at C.2's low-stratum 0.169 (rejected: retracted 2026-07-29, sign reverses across the censoring sweep). A non-inferiority gate against C.2 (rejected: not-worse isn't the paper's claim). No ordering criterion (rejected: opens the metric to post-hoc framing).
- **Rationale:** "if the final model significantly outperforms both then that is sufficient." The rule is fully specified in advance, even with no threshold set, so it can't loosen after results, and is stricter than the RMSE gate. Rank correlation is unweighted, so a low-stratum null is expected here and is not evidence against Phase D.
- **Revisit if:** never for a looser bar; transfers unchanged if §5.2 adopts a PA-weighted ordering statistic.

---

## 2026-07-30 — Cold start: unseen hitters get an untrained zero row
- **Decision:** A batter absent from train routes to a reserved embedding index, zero-initialized and never trained. No dropout or unknown-row training enters v1. Shrinkage comes from zero init plus weight decay on the embedding table (§2.3), with the §2.1 dimension sweep as the capacity lever.
- **Alternatives:** Frequency-inverse hitter dropout with a trained unknown row (rejected — see Rationale). A freely-learned unknown row (same objection, plus a parameter live at inference training never touches). Leaving the destination unspecified (rejected: an untouched random row gives unseen hitters a random personality and fails silently).
- **Rationale:** the mechanism "may make the model generalize better but will mute its findings somewhat." It pulls low-exposure hitters toward generic — the population claim 1 is about — so a Phase D margin would split between representation sharing and tuned shrinkage, the attribution failure the ladder prevents. This also flags a plan gap: §2.1 and §2.2 specify no behavior for a hitter with no training data, while §1.4 presumes some. That population is 43% of the low stratum.
- **Reference:** Iyyer et al. 2015 for the rejected mechanism; no source found for cold start in player-embedding work.
- **Revisit if:** the D.7 `‖e_h‖`-vs-`n_h` diagnostic shows low-exposure rows failing to shrink and v1's low-stratum RMSE exceeds C.2's; hitter dropout then reopens as a pre-registered ablation with a do-nothing control.

---

## 2026-08-01 — Phase D selection frame: select on 2024, refit through 2024, report on 2025
- **Decision:** Phase D trains on 2015-2023 pitches. It decides early stopping and every §4 ablation on the 2024 claim-1 frame. The winning configuration then refits from scratch on 2015-2024 and reports its headline on 2025. The Phase C ladder is re-scored on 2025 for the comparison. Only the configuration, including epoch count, carries across the refit.
- **Alternatives:** An inner claim-1 frame carved from train at 2023, reporting on 2024 and reserving 2025 for Phase F (rejected: Phase F judged unlikely). A model trained only through 2023, needing no refit (rejected: it leaves 75% of the 2025 low stratum on the untrained embedding row, against 42.7% with the refit). Deciding the ablations on the reported frame (rejected: the margin would carry the maximum over the configuration search, while every Phase C rung carries a single try).
- **Rationale:** §2.2 allocates 2024 as validation and 2025 as final test. Phase D is the first phase to need a selection budget, so the allocation binds here. Selecting on a season never trained on puts the headline on data no comparison has read. Refitting through 2024 leaves only report-season debutants without an embedding row — a population no model on the ladder can know, and the same 42.6% every Phase C number was produced under.
- **Reference:** Layer1_Architecture_Plan_v2.md §2.2 (split roles), §2.3 (Phase F gates).
- **Revisit if:** Phase F's gate fires despite being judged unlikely; a new entry must then name this one and fix its evaluation frame.

---

## 2026-08-01 — effective_speed dropped on measured redundancy; five features carry to the D.8 ablation
- **Decision:** `effective_speed` leaves the context tower. The remaining five of B.2's flagged six — `release_extension`, `release_pos_y`, `release_pos_z`, `release_spin_rate`, `spin_axis` — enter D.8 as one pre-registered block ablation, decomposed by mechanism only if it fires.
- **Alternatives:** Dropping all six on B.2's permutation nulls (rejected: five aren't determined by the kept features, and B.2's entry flagged those nulls as collinearity-deflated). One ablation per feature (rejected: the group is mutually collinear, so effect sizes add — a single block is the higher-powered test here).
- **Rationale:** Regressed on B.2's ten kept features, `effective_speed` reaches R² = 0.993 once `release_extension` is included, since perceived velocity is velocity adjusted for extension. The other five sit between 0.05 and 0.83 — none recoverable from what's kept. The test is one-directional: OLS understates what a nonlinear trunk can reconstruct, so a low R² licenses no admission alone.
- **Reference:** `results/phase_d/d0_redundancy_r2.csv`; discharges the 2026-07-29 B.2 deferral for this feature only.
- **Revisit if:** `release_extension` ever leaves the context tower, since it is what makes `effective_speed` redundant.

---

## 2026-08-01 — Ordering claims move to a denominator-weighted rank correlation
- **Decision:** `claim1_eval` gains a denominator-weighted Spearman coefficient; the Phase D ordering gate reads it. The unweighted §5.2 statistic stays, reported beside it. `paired_rank_difference` defaults to the weighted form.
- **Alternatives:** Keeping the unweighted statistic alone (rejected: at a 10-PA floor its low stratum is dominated by groups whose observed wOBA is near a coin flip). Replacing §5.2's statistic outright (rejected: it would restate every Phase C rank number under a metric they weren't reported with). Deferring until Phase D has a result (rejected: choosing a statistic after seeing the result is the post-hoc framing the 2026-07-30 gate forecloses).
- **Rationale:** Ranks carry no PA, but the correlation over them takes weights like any statistic — the unweighted form was an assumption, not a metric property. Weighting by the wOBA denominator puts ordering on the same precision weighting RMSE already uses, making a rank cumulative plate appearances rather than a roster position.
- **Reference:** Bailey, Emad, Zhang & Xie, "wCorr Formulas" (2023) — the only documented weighted-Spearman implementation and source of the estimator; it states no commonly accepted coefficient exists. Its weights are inverse-probability sampling weights, not precision weights, so its consistency results don't transfer.
- **Revisit if:** the two statistics disagree on a Phase C conclusion by more than their paired intervals — making the statistic choice itself a reportable result.

---

## 2026-08-01 — Pitchers' own at-bats excluded from the Phase D training table
- **Decision:** `model_dataset` drops pitches thrown to a pitcher taking his own at-bat, per season, reusing `eval_targets.primarily_pitchers` so Phase C and Phase D share one definition. Vocabulary falls from 2,486 to 1,762 hitters; the table from 7,347,953 to 7,293,321 pitches.
- **Alternatives:** Keeping them under "all MLB hitters" per §2.2 (rejected: that clause anchors the quality scale with stars, not a bottom the eval frame lacks). A second pitcher-batter definition local to Phase D (rejected: two definitions of one population drift).
- **Rationale:** `clean.py` filters position players on the pitching side only. Before this they held 35.7% of the embedding table's rows against 1.60% of its pitches — parameters spent on a population `claim1_eval` drops before scoring, one the query machinery can never ask about. The 2026-07-27 entry already requires their exclusion from every hitter-talent quantity; a learned table of hitter representations is one.
- **Revisit if:** a future season reintroduces pitchers batting in volume, which the 2026-07-27 entry already names.

---

## 2026-08-01 — v1 loss is the plain likelihood; re-weighting becomes an ablation arm
- **Decision:** The training loss sums raw per-row factor losses, with no per-head or inverse-frequency weighting. Per-head means and inverse-frequency weighting within the contact head enter D.8 as two ablation arms.
- **Alternatives:** Per-head means as default (rejected: a departure from the likelihood, adopted to make quality heads matter more than their sample supports). Explicit per-head weights (rejected: four more hyperparameters chosen on the selection frame).
- **Rationale (Nate's):** the model should translate baseball rather than be told what to care about. Summing raw losses maximizes the joint likelihood under §1.2's factorization, so each head's influence tracks its available evidence. It's also what D.5 needs: the conditionals feed a Markov composition that §5.4 requires to reproduce league run scoring, and re-weighting would decalibrate them against the real pitch distribution. B.1's stabilization ranking agrees — the high-count heads are also fastest-stabilizing.
- **Revisit if:** a D.8 arm shows re-weighting materially improves low-stratum projection, putting calibration and representation quality in tension and requiring §5.4 to be re-checked under the winner.

---

## 2026-08-01 — Trunk and context widths pre-registered, not swept
- **Decision:** Context tower is 2x128, trunk is 2x256, both fixed before any run. Only the embedding dimension is swept, over §2.1's {16, 32, 64}.
- **Alternatives:** Sweeping width and depth at D.8 (rejected: §2.1 names the embedding sweep specifically and gives only ranges elsewhere. Every extra configuration is another draw on a selection frame that cannot resolve small differences).
- **Rationale:** Both sit inside §2.1's stated ranges. They follow the convention that a first hidden layer is at least as wide as its input, which the trunk's 160-wide input satisfies at 256. The resulting model is ~207k parameters against §2.1's "well under 1M." Fixing them in advance makes this a stated choice, not an unremarked one.
- **Revisit if:** the D.6 first run underfits, with training and validation loss plateauing together, making capacity rather than selection the constraint.

---

## 2026-08-01 — Optimizer pre-registered: AdamW, plateau schedule, and batch size tied to weight decay
- **Decision:** AdamW at lr 1e-3, weight decay 1e-2, batch 8,192, `ReduceLROnPlateau` (factor 0.3, patience 1), early stopping at patience 3 — both reading 2024 validation loss. Batch size and weight decay are one setting; neither is swept.
- **Alternatives:** Adam (rejected: coupled decay scales with gradient magnitude, which varies with exposure, so shrinkage strength would go uncontrolled). Cosine annealing (rejected: needs a total step count early stopping makes unknowable in advance). Sweeping weight decay or batch size (rejected: both are the shrinkage lever, and tuning shrinkage splits a Phase D margin between the §1.4 hypothesis and the tuning — the failure the 2026-07-30 cold-start entry refused).
- **Rationale:** AdamW decays every parameter every step; an embedding row gets a gradient only in batches containing that hitter. Shrinkage is the ratio of the two — about 1:1 for the median hitter, 24:1 at the 10th exposure percentile. Batch size sets steps per epoch, scaling that ratio directly, so it can't vary independently of decay.
- **Reference:** Loshchilov & Hutter, "Decoupled Weight Decay Regularization", ICLR 2019 (New Orleans), arXiv:1711.05101; originally circulated as "Fixing Weight Decay Regularization in Adam." **Verified 2026-08-19** against the ICLR proceedings record and the authors' reference implementation. It establishes the property this entry rests on: decay applies directly to the weights, decoupled from the adaptive update, so a parameter decays every step whether or not it got a gradient — what makes the shrinkage ratio argument hold for sparse embedding rows. Not yet in the project library. Settings and shrinkage table: `docs/phase-d-spec.md` §5.
- **Revisit if:** D.7's `‖e_h‖`-vs-`n_h` diagnostic shows low-exposure rows failing to shrink, reopening the cold-start entry rather than this one.

---

## 2026-08-02 — Phase D tensors stored row-major, and the hitter vocabulary is kept
- **Decision:** `model_dataset.save` writes every array C-contiguous. `build` records the batter-to-embedding-row map in the manifest. The rebuild under both changes is bit-identical across all eight arrays, so D.1 is reproducible from the labeled parquet.
- **Alternatives:** Converting to row-major inside the Phase D loader (rejected: the 2015-2024 refit rebuilds through `save`, so the layout would return for the run producing the reported headline). Re-deriving the vocabulary downstream (rejected: two definitions of the embedding rows — the drift objection the 2026-08-01 pitcher-exclusion entry sustained).
- **Rationale:** `context` is assembled by column selection, and `np.save` preserves memory order, so one pitch's 46 features sat a column apart. Gathering a batch of 8,192 rows measured 8.74 s memmapped against 0.033 s row-major. The vocabulary joins a trained embedding table back to hitters, required by the §5.1 probe, the query machinery, and the D.7 shrinkage diagnostic.
- **Reference:** timings measured on the built table (M2, 8.6 GB RAM); both properties gated in `tests/test_model_dataset.py`.
- **Revisit if:** the context tower moves to categorical index columns, changing what is stored and therefore what the layout costs.

---

## 2026-08-02 — Run value stays out of the training loss
- **Decision:** The loss does not weight errors by run consequence. Run value enters once, in §1.5's mapping from batted-ball characteristics to runs that D.5 consumes.
- **Alternatives:** Weighting every factor's error by run value (rejected on the grounds below). Weighting only the quality heads (rejected: same objection; it also breaks the chain-rule decomposition that makes the raw sum non-arbitrary).
- **Rationale:** Run-value weighting is not a proper scoring rule: its optimum is a tilted distribution, not the true conditional, so the model overstates hard contact because hard contact scores more. It also double-counts run value, since D.5 already multiplies conditionals by a separately fit mapping. This is the mechanism the 2026-08-01 entry already refused.
- **Reference:** Gneiting & Raftery (2007), JASA 102:359-378, on strictly proper scoring rules and the uniqueness of the honest report at the optimum; not yet in the project library.
- **Revisit if:** §5.4's composition validation fails in a way traced to the loss, not to the run-value mapping or the Markov composition. That puts calibration and run-scoring fidelity in tension and requires a new entry naming this one.

---

## 2026-08-02 — Ordinal-aware scoring built behind a flag and gated by a promote-only screen
- **Decision:** v1 keeps log-likelihood loss. RPS (ranked probability score) over all five factors — reducing to Brier score on swing and contact — sits behind a `rule` flag, screened before D.6 on held-out per-pitch log-likelihood, reliability and resolution reported alongside. The screen can only promote RPS to a claim-1 ablation, never adopt it. Expected outcome, recorded in advance: null.
- **Alternatives:** Adopting RPS for v1 (rejected: no evidence exists; it reopens a pre-registered objective before the first run, the drift pre-registration prevents). Mixing log loss on binary factors with RPS on quality factors (rejected: different units; normalising RPS by K-1 moves quality-head influence by a factor of 23). Five-seed D.8 arm without a screen (rejected: another draw on the selection frame). Total variation distance as referee (rejected: not computable from one outcome per pitch).
- **Rationale:** Both rules are strictly proper — each minimised only by the true conditional, differing only in ranking imperfect answers. Free bin probabilities reproduced a bimodal target identically under both. Under a capacity restriction each won its own metric, while total variation to truth tied at 0.429, so promotion requires RPS to win on the likelihood's own metric. Frozen rule #2 reserves adoption for claim-1 regardless.
- **Reference:** Gneiting & Raftery (2007), JASA 102:359-378, on strict properness and the log score's locality; free-head and restricted-head fits measured directly, free-head agreeing to 1.5e-03.
- **Revisit if:** the screen shows RPS-trained quality conditionals beating log-trained ones on held-out log-likelihood, or matching at better reliability and equal resolution — promoting RPS to a D.8 arm, putting §5.4's composition validation in scope.

---

## 2026-08-02 — LA and spray are scored only where their conditioning bins were observed
- **Decision:** The launch-angle (LA) factor is scored only where both EV and LA bins are present; spray only where all three are. 273 rows with a valid LA but masked EV drop those factors, out of 1,266,309 and 1,239,195.
- **Alternatives:** Keeping those rows with an all-zero conditioning vector (rejected: the factor stops being the conditional it is named after). Conditioning on an imputed EV bin (rejected: silent imputation, forbidden by the Phase B missingness rule).
- **Rationale:** The chain factorises as p(ev) · p(la | ev) · p(spray | ev, la). With EV unobserved, the later factors have nothing to condition on and are not part of that pitch's probability. Strict nesting keeps the loss the plain likelihood, at a cost of 0.02% of balls in play.
- **Reference:** Layer1_Architecture_Plan_v2.md §1.5 (autoregressive factorisation); counts measured on the built table.
- **Revisit if:** a future outcome dimension carries materially more conditioning-only missingness. Then dropping rows stops being negligible, and an explicit unobserved category earns its own comparison.

---

## 2026-08-02 — ReLU and dropout 0.1 pre-registered from the architecture plan, not swept
- **Decision:** ReLU activations run throughout the context tower and trunk. Dropout 0.1 applies to the trunk output only. Both come from §2.1. Neither is swept.
- **Alternatives:** Saturating activations (rejected: derivative approaches zero away from the origin, so gradient decays through depth). GELU or SiLU (rejected: gains reported on far larger models — an unregistered deviation with no measurement behind it). Dropout at 0.5 (rejected: that value comes from heavily overparameterised networks; this model carries about 35 training rows per parameter). Dropout on the context tower (rejected per spec §3.2: that vector is observed for most pitches, missingness carrying explicit flags).
- **Rationale:** A nonlinearity makes the interaction representable: a linear function of the concatenated hitter and context vectors is exactly a hitter main effect plus a context main effect — the §1.4 failure mode the architecture exists to test. Dropout strength is also a regularisation lever, and the 2026-07-30 and 2026-08-01 entries already refused to tune regularisation, since a margin produced by tuning cannot be credited to the representation-sharing hypothesis.
- **Reference:** Layer1_Architecture_Plan_v2.md §2.1 ("ReLU, dropout ~0.1"); the saturation and dead-unit background is standard and unverified, not in the project library.
- **Revisit if:** the D.6 first run underfits, with training and validation loss plateauing together. The 2026-08-01 widths entry already names this as the condition reopening capacity rather than selection.

---

## 2026-08-03 — The bilinear interaction term is built low-rank, and stays a D.8 arm
- **Decision:** The §3.3 interaction term is `W_b (P_e e_h ⊙ P_z z_c)`, rank 32, no biases: 13,312 parameters added to the trunk output. It is built now, defaults off, and measured as a D.8 arm rather than turned on for the first run.
- **Alternatives:** The full bilinear form `e_hᵀ W_b z_c` (rejected: 1,048,832 parameters, taking the model from 207k to 1.26M, breaking §2.1's "well under 1M"). Turning it on by default (rejected: frozen rules #1 and #2 reserve every architecture choice for an ablation on the claim-1 metric; a first run carrying an unmeasured term cannot attribute its margin).
- **Rationale:** The full form learns all 4,096 hitter-by-context pairings independently; the low-rank form spends its budget on 32 shared interaction directions instead — the reduced-rank random-slope structure the term is meant to express. On synthetic data with a planted interaction, rank-32 recovered 98.1% of interaction variance against the additive model's 92.5% (weak) and 99.7% against 94.9% (strong). Both forms are identical at zero interaction, so the arm cannot cost anything when the failure mode it targets is absent.
- **Reference:** `docs/phase-d-spec.md` §3.3 and §8's interaction-learning risk; parameter counts and recovery fractions measured directly; the reduced-rank / factor-analytic random-slope analogue is standard mixed-model practice and is unverified, not in the project library.
- **Revisit if:** the D.8 arm fires — the interaction term improving the claim-1 metric. That makes rank itself a quantity worth measuring and requires a new entry naming this one.

---

## 2026-08-03 — Phase D runs locally on CPU, one device per comparison set
- **Decision:** Every Phase D training run executes on this machine's CPU. A comparison set — the five seeds of an arm, and all arms compared against one another — stays on one device and one thread setting. Runs queue sequentially, never in parallel.
- **Alternatives:** MPS (rejected: 66-81 s/epoch against the CPU's 63-74, a difference inside run-to-run noise). Rented GPU (rejected: the sweep fits in roughly three overnight sessions at no cost, leaving the <$200 budget untouched). Two concurrent runs (rejected: thread oversubscription on 8 cores, which already stalls the test suite).
- **Rationale:** One epoch over 5.88M pitches costs about 70 s, with ±15% run-to-run variation, so the device changes nothing measurable. CPU and MPS agree on loss to five decimals after a full epoch but are not bit-identical, since different kernels accumulate floats in a different order — mixing them inside a comparison set would put backend noise into the seed-to-seed spread the ensemble reports as variance.
- **Reference:** nine benchmark runs in `results/phase_d/d3_benchmark.csv`; peak RSS 1.2-1.6 GB of 8.6 GB.
- **Revisit if:** the epoch count or arm count takes the sweep past what overnight sessions absorb. Then rented compute is priced against the budget.

---

## 2026-08-03 — Interrupted runs are redone, never resumed
- **Decision:** The overnight driver records only completed runs. It redoes anything interrupted. No optimizer, scheduler, or RNG state is ever restored mid-run.
- **Alternatives:** Mid-run checkpoint and resume (rejected: restoring weights without AdamW's moment estimates and both RNG streams produces a run that trains normally but is no longer the run its seed names).
- **Rationale:** At roughly 25 minutes per run, redoing an interrupted one costs less than the reproducibility it would risk. The five-seed spread is only interpretable as seed variance if each seed determines its run completely.
- **Reference:** `src/model/sweep.py`; the ledger is `results/phase_d/sweep_log.csv`.
- **Revisit if:** a single run grows long enough that redoing it is expensive. The refit on 2015-2024 is the first candidate for this.

---

## 2026-08-08 — Finding: RPS screen returns a decisive null; log loss stays v1's objective
- **Decision:** RPS is not promoted to a claim-1 ablation. Scored on 2024, RPS is 1.19715 vs log 1.07582 — 126x the 0.00096 seed noise floor. The `rule` flag stays in code, defaulting to log.
- **Alternatives:** Comparing arms' own recorded losses (rejected: different units, not comparable).
- **Rationale:** Log score is local; it scores only the outcome bin. RPS is distance-sensitive and rewards less-confident probabilities that log score then penalizes. This confirms the pre-registered asymmetry rather than ranking the rules.
- **Reference:** `results/phase_d/screen_scores.csv`; discharges the 2026-08-02 screen.
- **Revisit if:** unchanged.

---

## 2026-08-08 — Reliability/resolution not computed for the RPS screen
- **Decision:** The screen closes on its first promotion clause. No calibration/refinement decomposition is built.
- **Rationale:** The second promotion clause (match log loss at better reliability, equal resolution) is unreachable given the 126x margin. That machinery belongs at §5.3's ensemble calibration check instead.
- **Revisit if:** §5.3 is built.

---

## 2026-08-08 — D.5 repertoire and called-strike surface keyed on batter handedness
- **Decision:** Resample real pitch rows, grouped by `(pitcher, stand, balls, strikes)`. Fit the called-strike model as one surface per batter-hand x pitcher-hand cell, not pooled.
- **Alternatives:** Overwrite the `stand` one-hot on a pooled group (rejected: submits pitch rows that never existed). Single pooled surface (rejected, same evidence).
- **Rationale:** RHP pitch mix differs sharply by batter hand (offspeed 20.8% vs LHB, 8.0% vs RHB). Pooling would misrepresent both repertoire and take-location by hand.
- **Reference:** Clemens 2025 FanGraphs (usage splits, unreviewed); Deshpande & Wyner 2017 JQAS (precedent for four separate surfaces).
- **Revisit if:** a per-hand cell is too sparse for smoothing to preserve the pitcher's own mix.

---

## 2026-08-08 — Count chain composed by exact solve, not simulation
- **Decision:** The 12 count states are solved by backward induction over repertoire-averaged transitions. The two-strike foul self-loop closes in closed form, `W(b,2) = A / (1 - E[P_foul])`. No simulation, no cap.
- **Alternatives:** Monte Carlo with a cap (rejected: adds noise on top of the 0.00096 floor). Truncated foul loop (rejected: closed form is exact and cheaper).
- **Rationale:** Pitch draw is Markov in count, given repertoire averaging. A linear solve applies, with no matrix inversion needed.
- **Reference:** Yonushonis 2011 SABR (foul-loop geometric series); Tenneal 2015 FanGraphs (12-state chain precedent, unreviewed).
- **Revisit if:** within-PA pitch sequencing enters the repertoire.

---

## 2026-08-08 — 24³ quality chain enumerated exactly, not sampled
- **Decision:** `E[wOBA|BIP,h,x]` is computed as the exact enumerated sum over all 13,824 bin combinations, via one trunk forward pass plus broadcast.
- **Alternatives:** Sampling (rejected: enumeration is cheap and exact).
- **Rationale:** Quality heads are linear over `[trunk; onehot]`, so the joint logits are an outer sum. Reconstructed logits agree with real forward calls to 9.5e-7 (~1 float32 ULP).
- **Revisit if:** a head stops conditioning via concatenation into a linear layer.

---

## 2026-08-08 — Fourth factor splits contact three ways; v1 retrained to carry it
- **Decision:** Add a three-class {foul, foul_tip, in_play} head, conditioned on contact. Retrain v1. Measure first against a league-average baseline table, so the head's effect is isolated.
- **Alternatives:** Binary in_play head (rejected: folds foul tips into fouls, wrongly inflating two-strike survival by ~68k of 1.37M events). Probe on the frozen trunk (rejected: the trunk was never trained to preserve this distinction).
- **Rationale:** Contact is really three states. Foul and foul-tip only diverge at two strikes — exactly the chain's missing distinction.
- **Reference:** Clemens 2025, Baumann 2024, Tenneal 2015 (unreviewed, descriptive only — see 2026-08-12 entry for the real justification).
- **Revisit if:** the retrained arm's claim-1 doesn't separate from the league-table baseline by more than seed noise.

---

## 2026-08-08 — Pitcher population is prior seasons only, weighted by batters faced
- **Decision:** The simulator draws pitchers from 2015-2023 only, weighted by batters faced. No within-window reweighting.
- **Alternatives:** Include eval-season pitchers (rejected: reads the season being predicted). Recency weighting (deferred). Resample a fresh pitcher per pitch (rejected: breaks one-pitcher-per-PA reality).
- **Rationale:** This matches every Phase C rung's information set, keeping the comparison fair.
- **Revisit if:** composition validation shows drift traceable to pool season composition.

---

## 2026-08-08 — Ensemble seeds combined by averaging conditionals
- **Decision:** Average the five seeds' per-pitch conditionals. Run one composition on the average.
- **Alternatives:** Average five separate wOBA compositions (rejected as headline: this averages a nonlinear functional, a Jensen error; kept as the source of between-seed spread).
- **Rationale:** A deep ensemble's prediction is the mixture of members' predictive distributions. Averaging probabilities is the correct mixture.
- **Reference:** Lakshminarayanan et al. 2017 NeurIPS (mixture form; composing through a downstream nonlinearity is this project's extension).
- **Revisit if:** §5.3 needs the functional's dispersion as headline uncertainty.

---

## 2026-08-08 — Called-strike model uses raw plate coordinates, no batter-height normalization
- **Decision:** Fit `p(ball/called-strike/HBP | take)` on `plate_x`/`plate_z`. Exclude `sz_top`/`sz_bot`.
- **Alternatives:** Normalize via `sz_top`/`sz_bot` (rejected: those are Statcast-derived from past umpire calls — circular). Statcast `zone` (rejected: a coarser function of the same two columns).
- **Rationale:** Normalizing by an umpire-derived zone would regress umpire behavior on itself.
- **Reference:** Deshpande & Wyner 2017 (raw-coordinate precedent); Freiman 2018 FanGraphs (batter height explains R²=0.23 of the low strike vs 0.05 high).
- **Revisit if:** real batter-height data becomes available.

---

## 2026-08-08 — Finding: C.2 discharges frozen rule #1's empirical-Bayes baseline
- **Decision:** C.2 (bivariate EB) is the empirical-Bayes incumbent that frozen rule #1 requires. The Book-rho rung is the literal published-constants comparator.
- **Alternatives:** A separate split-level Book estimator (rejected 2026-07-27: unstable variance).
- **Rationale:** The two parameterizations are the same model in rotated coordinates. This satisfies the requirement on the estimand.
- **Revisit if:** The rho interval tightens enough to separate this estimate from The Book's implied value.

---

## 2026-08-08 — D.5's own knobs validated on composition fidelity only, never on claim-1
- **Decision:** Pitcher-pool size, pitches-per-cell, and smoothing strength are pre-registered. They are validated only against composition fidelity, never tuned on claim-1.
- **Alternatives:** Treat them as ordinary §4 ablation knobs (rejected: D.5 produces the claim-1 metric itself, so tuning on it flatters the measuring instrument).
- **Rationale:** Composition fidelity scores against observed league run-scoring, independent of the model's own margin.
- **Revisit if:** A knob changes the Phase D vs C ranking without changing composition fidelity.

---

## 2026-08-08 — Fourth factor retrains all arms together, as ledger stage d9
- **Decision:** All eight D.8 arms rerun with the three-class split head, as new ledger stage `d9`. The 30 completed `d8` runs stand unchanged as the pre-split record.
- **Alternatives:** Run only the two never-run arms on the old architecture (rejected: answers a question about a model that doesn't ship). Mix six old runs with two new ones (rejected: mixes units in one column).
- **Rationale:** Claim-1 now flows through the split-head model, so the whole table must share one architecture. Rerunning all eight costs one overnight session, since the retrain rebuilds every arm anyway.
- **Revisit if:** Never, for comparability — a future factor needs the same treatment.

---

## 2026-08-08 — D.5 scores against the whole pitcher population, not a sampled panel
- **Decision:** Query every 2015-2023 pitcher, weighted by batters faced, with 6 pitch rows per cell. Full population is the default.
- **Alternatives:** A 60-pitcher sampled panel (rejected: measured a 0.0048 wOBA level shift between two draws, vs a 0.033 between-hitter spread).
- **Rationale:** PA-weighted RMSE charges for exactly that panel-level shift. The full pass removes the source instead of shrinking it.
- **Revisit if:** Pool size stops fitting a session.

---

## 2026-08-09 — One pitcher per simulated PA, pitches drawn independently per count
- **Decision:** The count chain is solved per pitcher, from his own 12 cells, with 6 real pitch rows per cell. Results are averaged by batters faced afterward. Pitch draw depends only on count.
- **Alternatives:** Average transition probabilities across pitchers first, then solve once (rejected: division by `1-P_foul` makes the orders diverge; no hitter faces the composite). Condition repertoire on prior pitches in the PA (rejected: breaks the Markov property, too few cells).
- **Rationale:** Independence-given-count licenses the closed-form solve. The cost is unmodeled pitch sequencing.
- **Reference:** 6 draws inflate the two-strike multiplier by +0.0098 vs a ~1.40 baseline, propagating to ~0.001-0.002 wOBA.
- **Revisit if:** Level bias survives other diagnoses, making `n_pitches` the next lever.

---

## 2026-08-12 — Composition validation splits into an unscored probe and a scored per-PA check
- **Decision:** Zero-row probe stays unscored. New scored check: four per-PA absorbing rates (BB/K/HBP/BIP) vs true handedness-weighted train-window rates, each within 2% relative (HBP 20%). Credit needs a fix to move a rate past between-seed spread.
- **Alternatives:** Rebuild the old check (rejected: zero row isn't a hitter, no pass condition). Score against eval-season hitters (rejected for now: doubles a run). Flat 25% handedness weights (rejected: −0.00271 wOBA error; both weightings now reported).
- **Rationale:** The old check read per-pitch masses at 0-0 and missed the walk/strikeout tradeoff. The four absorbing rates are what wOBA terminates in.
- **Reference:** train-window (n=1,526,308 PA): BB 0.08042, K 0.22338, HBP 0.01049, BIP 0.68571; true-share wOBA 0.31639 vs flat-25% 0.31368.
- **Revisit if:** A fix passes all four rates while composition fidelity and claim-1 disagree.

---

## 2026-08-12 — Finding: Third contact class rests on a rule of baseball, not the cited posts
- **Decision:** This restates the 2026-08-08 fourth-factor entry. The support is the playing rule — a caught foul tip at 2 strikes is a strikeout — not the cited blog posts, which speak only to the learnability of a hitter-specific foul-tip rate.
- **Alternatives:** Cite a peer-reviewed foul-tip study (rejected: the load-bearing premise is a rule, not an empirical estimate). Amend the original entry (rejected: the log is append-only).
- **Rationale:** This separates the mechanical argument (strong) from the posts (weak, a different question), so the retrain's justification doesn't rest on the posts being right.
- **Revisit if:** A per-hitter foul-tip rate becomes a feature or claim.

---

## 2026-08-12 — xwOBA enters as a rung and a second answer key, never the primary target
- **Decision:** xwOBA enters three ways only: a C.1 ladder rung, a second answer key scored beside realized wOBA, and an approximate achievable-error floor (RMSE between the two keys). Never the primary target or a gate input.
- **Alternatives:** xwOBA as primary target (rejected: changes the claim, makes ground truth another model's output). Build the rung from `V` marginalized over spray (rejected: inherits `V`'s own defects).
- **Rationale:** The gap between the two keys separates "wrong about batted-ball quality" (fixable) from "unknowable" (fielding, sequencing). Floor is 0.02923 vs Phase D's 0.0492 all-stratum RMSE — ~40% of error is answer-key noise.
- **Reference:** floor by stratum 0.0364/0.0331/0.0252/0.0292; ordering advantage grows under xwOBA (0.5763 vs 0.5389) vs realized (0.4608 vs 0.4417).
- **Revisit if:** A future model gains a fielding-alignment channel.

---

## 2026-08-12 — Finding: D.5 level excess: two fixes land, cell-size exonerated, exposure fails talent control
- **Decision:** Count-specific take-surface offsets (12/surface) land. In-play mass splits 96.37/3.63; unmeasured share valued 0.29973. `n_pitches` stays 6, no escalation to 48 surfaces. HBP needs no fix.
- **Alternatives:** 48 surfaces (declined: pooled surface's marginal error is +0.0002 over 3.89M takes; offsets already recover the per-count component). Raise `n_pitches` (declined: wrong direction — 0.308/0.309/0.309 at M=6/12/24). Reweight `fit_outcome_table`'s sample (rejected: distorts cells to fix an aggregate).
- **Rationale:** Unmeasured-category split passes a clean two-sided test (league wOBA −0.00171 vs predicted ~−0.0015, absorbing rates untouched). Offsets close the strikeout gap, not the walk gap. Talent-controlled gradient rules out exposure; a real but small trunk interaction exists (~1/10 the level bias) — excess is a level, not a gradient.
- **Reference:** offset effect: BB −0.0017, K −0.0045, BIP +0.0060 (walk gap needs 0.0053, gets 0.0017; K needs 0.0011, gets 0.0045 — overshoots, see 08-18 credit-rule fix).
- **Revisit if:** Walk gap survives the Step 4 rebuild, pointing to take frequency (swing head), not ball-given-take.

---

## 2026-08-12 — Finding: Equal-mass binning wins its own contest; the top-bin defect is not a binning problem
- **Decision:** Quality bins stay equal-mass, 24/dimension, refit once with Statcast placeholder pairs dropped from the edge fit and masked from all three quality targets. Edges frozen in the build manifest.
- **Alternatives:** `top_decile_split` (rejected: narrows top-EV bin 15.10→12.80mph, raises joint variance 0.1431→0.1451). `variance_min` via exact 1-D DP (rejected, worst at 0.1487 — wrong objective for a joint-cell prediction). `n_bins=32` (deferred: 2.4x cell blowup, separate decision). Mask placeholders instead of dropping (rejected: still pollutes quantile computation).
- **Rationale:** Pre-registered objective was within-cell realized wOBA variance over the joint 24³ grid. Equal mass wins outright — re-binning at fixed `n_bins` can't fix the top-bin defect.
- **Reference:** 954,223 scored BIP; variance 0.1431/0.1451/0.1487; placeholder drop 3.61% of BIP.
- **Revisit if:** Another scheme is proposed, or the high-stratum discrimination loss survives rebuild.

---

## 2026-08-12 — D.5 report gets per-stratum verdicts, a power factor, and a split spread diagnostic
- **Decision:** The report emits a gate verdict for all four strata, not just low. Every rank-gap null restates as a power statement (SE plus batters needed). The spread diagnostic splits on training-vocabulary membership, excluding cold-start rows.
- **Alternatives:** Swap the hard-coded stratum (rejected: reading only one is the defect). Leave the low-stratum verdict as flat "null" (rejected: conflates "measured absent" with "unmeasurable"). Pool cold-start into the spread diagnostic (rejected: their spread reflects only context variation).
- **Rationale:** Each defect misstated confidence, not model behavior — fixes are re-reads of existing files. Splitting the diagnostic reverses its direction: cold-start pooling masked an anti-shrinkage finding.
- **Reference:** shipped-arm low-stratum rank gap +0.0908 (SE 0.0687, z=1.32) needs ~4.5x more batters (~1,074) for 80% power. Trained-row-only spread 0.0374 (low) vs 0.0281 (regulars), 33% wider than a misleadingly narrow pooled 0.0305.
- **Revisit if:** An eval frame spans enough seasons for ~1,080 low-stratum batters.

---

## 2026-08-14 — Finding: Gradient test (b): the low-exposure embedding is displaced, not shrunk; exposure-conditional prior declines
- **Decision:** Exposure-conditional C.2 prior not built. Gradient (b) runs on d10 baseline, 5 seeds, reserved row excluded; reports embedding norm and its projection onto the wOBA-raising direction, since the two disagree in sign.
- **Alternatives:** Build the prior on the univariate exposure gradient (rejected: the case pre-registration guarded against). Substitute observed 2024 wOBA as the talent proxy (rejected: manufactures the correlation it reports).
- **Rationale:** Both required-fail conditions triggered independently — gradient (a) ruled out exposure, gradient (c) prices a real but small hitter interaction. Gradient (b) supplies the mechanism: low-exposure rows sit farther from origin, not shrunk in norm, but their wOBA-direction component is shrunk/displaced toward low predicted wOBA — capacity exists, oriented orthogonally to the scored axis.
- **Reference:** norm slope negative in all 5 seeds (e.g. −0.0186 [−0.0207,−0.0167] per 1k train pitches); projection slope positive in all 5 (+0.0218 [+0.0205,+0.0233]). By exposure quintile, mean norm falls 1.00→0.61 then flattens, projection rises −0.22→+0.07, flipping sign only in the top quintile.
- **Revisit if:** A future arm adds exposure-dependent regularization, or claim-1 shows the low-stratum loss is direction-driven.

---

## 2026-08-15 — Finding: claim-1 for all eight d10 arms: four beat baseline, margins small, ledger retired as architecture instrument
- **Decision:** All eight arms score on claim-1 (paired bootstrap, batter-clustered, decisive stratum `all`). `block`, `bilinear`, `meanweight`, `dim64` beat baseline; `nospray`, `dim16` are nulls. Shipped arm unchanged: `block`'s margin sits inside baseline's own seed range, so adoption defers to Phase E with a re-scored ladder.
- **Alternatives:** Read verdicts from held-out log loss (rejected: Spearman correlation between log-loss and claim-1 rank, across 7 arms, is exactly 0.000). Pre-register `low` as decisive (rejected 08-12: noisiest stratum). Score `nospray` against uniform spray (rejected: no ball follows uniform). Adopt `block` now (declined: out of phase scope, though the rule selects it).
- **Rationale:** Log loss and claim-1 measure different things — baseline is #1 on log loss, #6 on claim-1; `block` is the reverse. `block` (drops B.2's five release/spin features) wins claim-1 while losing log loss: those features help next-pitch prediction, not hitter projection.
- **Reference:** margins (negative favors arm): `block` −0.00078 [−0.00107,−0.00048], `bilinear` −0.00063, `meanweight` −0.00044, `dim64` −0.00028, `nospray` −0.00010 (n.s.), `dim16` −0.00003 (n.s.), `invfreq` +0.01274 (worse). Baseline seed spread in `all` stratum: 0.00186 range, comparable to the best-to-baseline spread (0.00078).
- **Revisit if:** Phase E re-scores on the no-block build, or an arm's margin exceeds a single arm's own seed range.

---

## 2026-08-18 — The 2025 final run moves after Phase O; Phase E evaluates on the 2024 frame only
- **Decision:** Phase E computes every validation and effectiveness number on 2024. The 2015-2024 refit and single 2025 report move to Phase O's end, on whatever configuration it settles. `assert_not_test_season` still blocks 2025 outside `--final-run`.
- **Alternatives:** Follow the plan's literal order — report 2025 in Phase E, then optimize (rejected: leaves Phase O nothing untouched to prove a gain on). Score 2025 now without a refit (rejected: 75% of the 2025 low stratum would sit on the untrained zero row vs 42.7% with refit — not comparable to Phase C). Refit twice (rejected: the second report is test-set reuse regardless of labeling).
- **Rationale:** The plan's own lines conflict: it places the 2025 table before optimization but requires only the winning configuration cross the refit, which doesn't exist yet. This keeps one refit, one 2025 report, and the 42.7% cold-start share, only moving which phase boundary they sit behind. The manifest makes the decision log authoritative where it conflicts with the plan.
- **Reference:** architecture plan lines 119-120, 167, 169, 175.
- **Revisit if:** Phase O returns no adopted change, or the deadline moves such that a 2025 number is needed before Phase O completes.

---

## 2026-08-18 — The D.5 credit rule is repaired: a paired delta is graded against the spread of the paired delta
- **Decision:** A composition fix is credited when its mean paired difference across seeds exceeds the between-seed spread of that paired difference, not of the raw level as before. Credit verdict and 2%-band result are always reported together. Applies Phase E forward; does not retroactively re-credit D.5's published verdicts.
- **Alternatives:** Leave the old rule (rejected: seed effect cancels inside a paired difference but not inside a level, so the old denominator was the wrong order of magnitude — no real fix could ever clear it). Re-credit D.5's offsets now (declined: needs its own entry). Drop the credit rule, keep only the band (rejected: the band says nothing about whether a change was real).
- **Rationale:** Both sides of a paired comparison share the seed, so seed variance subtracts out of the difference. Grading against an undifferenced level's spread asks the effect to beat noise pairing already removed. Costs nothing new: the corrected denominator is a re-read of already-persisted per-seed files.
- **Revisit if:** A future arm has fewer than three seeds, or D.5's count offsets are re-graded under this rule (explicitly not done here).

---

## 2026-08-18 — Finding: The walk gap is 54% population, 24% composition structure, 0% resampler
- **Decision:** The shipped +6.17% walk-rate fidelity failure decomposes into three parts. E.1's population-matched control accounts for 54% (matched observed 0.08312 vs shipped unmatched 0.08042). E.10 accounts for 51.6% of the remainder (+0.00117), a model-free property of the independent-pitch count chain. E.7-E.9 (resampler channels) net to ~zero. About 22% remains unowned.
- **Alternatives:** Carry the headline unattributed into Phase O (what D.5 shipped). Stop after E.1. Attribute the residual to the swing head (the handoff's suspect — refuted by E.6, wrong sign).
- **Rationale:** Standing risk gates any ablation reading on attributed composition fidelity. E.10 is load-bearing because it's model-free — a counted-frequency property no retrain can fix.
- **Revisit if:** Composition is changed to condition on within-PA history, or the 22% is closed by a later diagnostic.

---

## 2026-08-18 — Finding: The swing head is exonerated as owner of the walk gap
- **Decision:** The swing head is calibrated and is removed from the suspect list. On 705,344 real held-out 2024 pitches (resampler excluded), predicted swing rate is 0.479256 vs observed 0.477975 (+0.27%). Every handedness cell is within 0.5%.
- **Alternatives:** Treat the handoff's framing as settled. Open a Phase O retrain item.
- **Rationale:** A head-owned gap would show a swing shortfall in three-ball counts. The measured result has the opposite sign at 3-0 (+8.7%), 3-1 (+0.9%), and 3-2 (+2.1%). The head produces too many swings; it cannot manufacture extra walks.
- **Revisit if:** the head is retrained, or the eval season changes.

---

## 2026-08-18 — Finding: The walk excess is a spread problem, not only a level problem
- **Decision:** The walk gap is claim-1 relevant, not only a fidelity issue. The noise-corrected compression coefficient for walks is b=0.642 (vs 0.773 K, 0.736 BIP, 0.095 HBP). The model expresses only ~64% of true between-hitter spread in walk rate.
- **Alternatives:** Read the naive regression slope of (modelled−observed) on observed (rejected: this is mechanically compressive even at zero true compression, because target sampling noise sits in the regressor).
- **Rationale:** A pure level bias would leave ranking intact. This is compression of the exact quantity claim-1 measures. The correction uses the known binomial noise variance p(1-p)/n.
- **Revisit if:** the walk gap closes.

---

## 2026-08-18 — Finding: The platoon adoption rule is not met on 2024
- **Decision:** The model's platoon differential is not adopted over Route A (overall skill plus league-average split). No stratum's paired 95% rank interval excludes zero. RMSE favors the model only pooled (−0.00115 [−0.00216,−0.00017]).
- **Alternatives:** Call pooled RMSE alone "adoption" (rejected: it rests the claim on one metric-stratum pair of eight). Defer reading until the walk gap is fixed (rejected: it leaves Phase O with no claim evaluation).
- **Rationale:** 81.7% of the model's predicted differential variance is just the batter-stand main effect, vs 10.9% observed. The model has moved about a fifth of the way from "apply the league split to everyone" toward a real differential. Small and real, but not enough per frozen rule #1.
- **Revisit if:** the walk gap closes and the C-ladder is re-scored under d10; or the refit changes the low-exposure stratum (currently 42.7% cold-start).

---

## 2026-08-18 — Finding: Model confidence tracks exposure, reported unscored
- **Decision:** E.4 calibration is recorded descriptively, not as a scored gate. Regression slope, observed-on-modelled: 0.529 low-exposure (z=−4.90 vs 1), 0.664 medium, 0.998 high, 0.841 pooled.
- **Alternatives:** Treat the low-exposure slope as a failure. Open remediation.
- **Rationale:** This matches the architecture prediction: the embedding is under-dispersed where hitters lack history, calibrated where they have it. Left unscored — scoring it on a claim-1-adjacent metric would be circular (the scorer is the model).
- **Revisit if:** the declined exposure-conditional prior is ever adopted.

---

## 2026-08-19 — Finding: Platoon variance framing was errors-in-variables biased; the 2026-08-18 entry is superseded
- **Decision:** The "81.7% vs 10.9%" variance-share framing in the 2026-08-18 platoon-adoption entry is withdrawn. It compared a noise-free predicted variance to a noise-dominated observed one — the same errors-in-variables trap E.2 guards against. The corrected share (binomial noise removed) is expected near 50-60%, to be computed in E.15. No share figure is quotable until then.
- **Alternatives:** Leave the figure with a caveat (rejected: this leaves a wrong number standing). Withdraw the whole 08-18 entry (rejected: its adoption verdict does not depend on the share).
- **Rationale:** The adoption null stands on the paired interval, which was computed correctly. Only the "sharpest statement" framing was biased.
- **Reference:** none — standard errors-in-variables attenuation, already used in E.2.
- **Revisit if:** E.15's corrected share lands outside 40-70%. This would also require restating E.5's "real but small departure" reading.

---

## 2026-08-19 — Finding: Platoon skill is a separable talent; the split constant we disagree with is second-hand
- **Decision:** Platoon skill is a separable talent (C.2 model-free estimate): rho=0.652 LHB [0.384,0.902], 0.719 RHB [0.479,0.987], 0.713 SHB [0.093,0.999] — true talent vs LHP/RHP correlation. rho<1 is the whole claim.
- **Alternatives:** Read the model's shrinkage toward league average as near-absent skill (rejected: circular — reads the imposed EB prior back out of its own posterior).
- **Rationale:** Separates "platoon skill is real" from "our model measures it well": evidence supports the first, largely not the second. Recovered within-stand spread: 54% LHB, 18% RHB — a 3x asymmetry no uniform prior produces.
- **Reference:** UNVERIFIED — our rho disagrees with The Book's implied 0.887 LHB/0.949 RHB, reaching us second-hand via Tango 2009, not the primary source. Check the primary source before asserting this disagreement.
- **Revisit if:** the primary source is obtained, or the 2015-2024 refit changes the posterior.

---

## 2026-08-19 — The §5.1 probe checkpoint is demoted from early gate to retrospective diagnostic
- **Decision:** Architecture §5's item-1 probe, the early §1.4-failure detector, never ran as a gate. It runs in E.14 instead — a retrospective diagnostic that explains but doesn't stop results.
- **Alternatives:** Run it as specified, as a gate (rejected: the gated runs are already complete). Skip it (rejected: likeliest explanation for E.5's result).
- **Rationale:** The §1.4 failure mode appears to have fired. The detector sat unrun through Phase D, found only after 39 training runs — a checkpoint with no forcing step is not a checkpoint.
- **Revisit if:** a future phase adds training runs; the probe returns as a gate with an explicit forcing step.

---

## 2026-08-19 — The calibration/refinement decomposition is retired; ensemble interval coverage is not
- **Decision:** The calibration/refinement decomposition (owed since 2026-08-08) is retired unbuilt. Its bundled ensemble interval coverage check is kept, discharged in E.14.
- **Alternatives:** Build it on claim-1 to close the debt as written (rejected below).
- **Rationale:** The decomposition fits a probabilistic categorical forecast; claim-1 is RMSE on a continuous target — the machinery doesn't fit the surface. The coverage check discharges the reliability half; E.4's slope and rank correlations answer the resolution half. Coverage stays as untested: §2.1 says calibration on very-low-exposure hitters "remains assumed until §5.3 checks it."
- **Reference:** Gneiting & Raftery 2007 (calibration and sharpness as separate properties) — verified, held locally.
- **Revisit if:** the project ever scores a categorical probabilistic forecast as a headline claim.

---

## 2026-08-19 — Phase E scope closes at E.15; two items are deferred by name
- **Decision:** Phase E covers E.1-E.15. (a) Non-handedness conditional queries (former E.16) move to the frontend phase. (b) The deployment-bias audit (architecture §5 item 5) moves to write-up; its method is recorded in `phase-e-spec.md` §12.6.
- **Alternatives:** Run both here (rejected: (a) architecture §1.3 says pitcher-typed queries use the same machinery with no new code path, so it tests nothing handedness has not; (b) it needs a cohort build, and the Oct 1 deadline does not allow it).
- **Rationale:** Both items are deferred, not dropped. The audit is the direct answer to the regression-to-archetype objection a reviewer will raise, hardest on the low stratum, where 42.7% of hitters share the cold-start row.
- **Revisit if:** a reviewer objection turns on either item; the audit should then be promoted ahead of the frontend if the low-exposure claim survives E.11.

---

## 2026-08-19 — Phase Q added to the build order; the only architecture-plan change this window
- **Decision:** A query-dashboard phase (Phase Q) is added to the plan's §3 build order, after Phase V, outside the paper's claims — non-handedness conditional queries live there. The only architecture-plan edit made during Phase E.
- **Alternatives:** Amend the plan wherever a Phase E finding contradicts it (rejected: Phase E is an evaluation window, not a design window; rewriting the spec mid-evaluation erases the record of what was designed against).
- **Rationale:** §7 already makes the decision log authority over the plan, so the log running ahead is designed behavior. Phase Q is a genuine addition, not a correction, which clears the bar.
- **Revisit if:** the dashboard needs a scorer code path after all, contradicting §1.3, making it a model change, not a surface.

---

## 2026-08-19 — Finding: E.11: claim-1 re-scored on the D.10 arm; both gates fail, and the ordering null is underpowered
- **Decision:** The claim-1 gate is re-scored on D.10, adopted as the project's verdict — superseding `results/phase_d/d5_claim1_verdict_phase_d_baseline.json` (written against D.8, four-key schema since expanded to twelve). Both gates fail in every stratum, including the decisive low stratum.
- **Alternatives:** Re-run under the original label (rejected: the frozen rule keeps prior-lane artifacts absent a confirmed bug; both labels on disk serve a later D.8/D.10 comparison).
- **Rationale:** D.10 is the best low-exposure ranker by point estimate — weighted rank 0.2592 vs C.3-full's 0.1665, a paired gap of +0.0927 [-0.0343, 0.2194]. But 914 batters are needed against 239 available, so the ordering result is a power-limited null, not a measured absence. All-stratum RMSE is significantly worse, the high stratum loses on both metrics; removing the level bias doesn't rescue the claim.
- **Reference:** none — pre-registered 2026-07-30 and 2026-08-01.
- **Revisit if:** the 2015-2024 refit changes the low-exposure population (42.6% cold start), or the deferred deployment-bias audit shows the low-stratum answer key is biased, not merely noisy.

---

## 2026-08-19 — Finding: E.12: the D.10 ablation table is confounded with each arm's level, and cannot select alone
- **Decision:** The ablation table's PA-weighted RMSE column is confounded with each arm's mean predicted wOBA — not a ranking of ranking skill. Phase O must not select arms on it alone.
- **Alternatives:** Select on weighted rank correlation with a paired interval instead (not adopted: the decisive-stratum rank spread across arms is 0.0119, itself possibly inside noise, so the replacement needs its own test when Phase O opens).
- **Rationale:** Across seven non-degenerate arms, mean predicted wOBA tracks all-stratum RMSE at r=0.888 (p=0.0076) and decisive-stratum RMSE at r=0.756 (p=0.049, Spearman p=0.215), but not rank correlation. Between-arm level spread is only 8% larger than the baseline's five-seed spread. The confound is decisive all-stratum, marginal in the decisive `low` stratum — enough to bar selection, not settle the replacement. Fork-opened: the hypothesis was generated and confirmed on the same data; the level offset is never subtracted.
- **Revisit if:** a future arm set is scored, testing the confound out of sample.

---

## 2026-08-19 — Finding: E.11b: the claim-1 verdict does not depend on the eval-season playing-time filter
- **Decision:** Architecture §5's committed MIN_EVAL_PA sweep is discharged for the D/E headline. Both gate verdicts are stable across thresholds 10/25/50.
- **Alternatives:** Continue quoting the Phase C sweep (rejected: it hard-codes c3-vs-c2 and had never been run against this headline).
- **Rationale:** The filter conditions on playing time decided after the projection, and partly by how the hitter performed. Its neutrality had to be measured, not assumed. Tightening it removes half the low stratum, and the low-stratum rank gap grows to 0.1099 without ever becoming significant. One isolated cell flips at threshold 50 and is not the gate.
- **Revisit if:** the 2015-2024 refit changes the low-exposure population.

---

## 2026-08-19 — Finding: E.13: the BIP value gap is era drift in the frozen value table, not the measurement seam
- **Decision:** The +0.015116 BIP value gap is attributed to era drift in V (+0.028836), not the D5-R15 measurement seam (-0.016154). The 16.1% residual is below the one-third stop rule, so the gap counts as explained.
- **Alternatives:** none — this is a measurement.
- **Rationale:** Identically-struck balls were worth less in 2024 than under the train-season table; the seam moves the modelled value the opposite way. The pre-registered expectation — that the seam explained a minority of the gap — failed on sign and size. The split isn't orthogonalised, so the residual absorbs any interaction. The era-drift check is fork-opened, substituted after the Jensen test was refuted structurally: V is a lookup table, linear in the bin distribution — convexity cannot operate.
- **Revisit if:** the 2015-2024 refit rebuilds V, shrinking the drift term directly.

---

## 2026-08-19 — Finding: the two tensor builds differ only in the quality bins, and no Phase E result is affected
- **Decision:** `data/processed/phase_d` and `phase_d5` differ only in `quality_bin_edges` and the `ev`/`la`/`spray` index arrays — all other fields are byte-identical. The differing fields have one Phase E consumer, already reading `phase_d5`.
- **Alternatives:** Flip the stale `--data-dir` defaults in the five modules still naming `phase_d` (rejected: the frozen rule bars prior-lane edits absent a confirmed bug; those five read only byte-identical fields; `phase_d` lacks `split.npy`).
- **Rationale:** D.10 trained on `phase_d5`, so a module feeding it `phase_d` bin indices would mis-bin model inputs — the exposure was audited field by field, not inferred from module names. Stale defaults remain a trap for any future module reading the quality bins.
- **Revisit if:** a new module reads `ev`/`la`/`spray` from `--data-dir`; it must default to `phase_d5` or assert against the shipped constants.

---

## 2026-08-19 — Finding: E.14: the platoon signal is linearly decodable for LHB and not for RHB; ensemble intervals under-cover
- **Decision:** Architecture §5's probe is discharged as a retrospective diagnostic, its ensemble-coverage item as measured. LHB platoon split decodes above null (+0.223 [0.094, 0.351]); RHB does not (+0.048 [-0.054, 0.142] against null +0.075).
- **Alternatives:** Read the RHB result as a representation failure (rejected: intervals overlap, no difference test was run, the probe is linear, and the C.2 target is shrunk harder for RHB — attenuation alone predicts a weaker decode).
- **Rationale:** The shuffled null isn't centred on zero: out-of-fold ridge predictions of noise are anti-correlated with the target, hence the "beats null" test. The pooled row is confounded by stand; per-stand rows are the headline. Fair intervals cover 0.391/0.690/0.856 against nominal 0.50/0.80/0.95 — every Wilson interval excludes its nominal level in every stratum, so the pre-registered low-stratum pathology is real but global. Cold-start and trained rows in the low stratum are indistinguishable at n=162/218, too few to resolve the effect.
- **Reference:** Gneiting & Raftery 2007, for calibration subject to sharpness; the full reliability/resolution decomposition was retired 2026-08-19 as native to categorical forecasts.
- **Revisit if:** Phase O changes the ensemble width, or the refit changes the cold-start share.

---

## 2026-08-19 — Finding: E.15: the corrected between-stand share is 0.814, and the LHB reliability routes disagree
- **Decision:** The errors-in-variables debt from 2026-08-18 is discharged. The noise-corrected between-stand share is 0.814 [0.42, 3.50], superseding the withdrawn 0.109. No single reliability figure is adopted; any reading of E.5 carries both ceilings.
- **Alternatives:** Adopt the C.2-derived reliability alone (rejected: the split-half route contradicts it for LHB, with an interval excluding the C.2 value).
- **Rationale:** 86.6% of platoon-differential variance is sampling noise. Removing it lifts the share to 81.7% (loose: denominator is a small difference of large numbers), reversing the over-weighting concern. The pre-registered 50-60% expectation failed. The asymmetry check couldn't run: noise-corrected LHB within-stand variance came back negative. C.2 gives LHB reliability 0.157; split-half gives -0.366 [-0.84, 0.008] — two of three routes find no measurable LHB spread in 2024, yet LHB is the signal that decodes.
- **Reference:** The Book p.157 stays UNVERIFIED: its rho disagrees with our 0.652/0.719; `src/analysis/c_report.py:45` records we lack a copy.
- **Revisit if:** The LHB contradiction needs resolving — a bootstrap CI on the C.2-derived reliability, propagating rho's [0.384, 0.902] interval, would move tau²_split by ~3x.

---

## 2026-08-20 — every Phase F artifact carries a tensor-build stamp, and reads assert the quality bins
- **Decision:** `src/analysis/provenance.py` stamps each Phase F summary with `data_dir`, the sha256 of that build's `manifest.json`, `train_seasons`, present quality-bin arrays, arm, seeds, eval season, and git revision. Any `ev`/`la`/`spray` reader calls `assert_quality_bins`, refusing builds whose `quality_bin_edges` differ from `data/processed/phase_d5`.
- **Alternatives:** Flip the stale `--data-dir` defaults in the five prior-lane modules (rejected again: the frozen rule bars prior-lane edits absent a confirmed bug). Rely on the existing field-by-field audit (rejected: it covers only modules that existed when written).
- **Rationale:** The 2026-08-19 entry left one condition open: a new `ev`/`la`/`spray` reader must default to `phase_d5` or assert the shipped constants; F.3, F.4, F.5 are that module. This closes it mechanically — the stamp lets a reader trace which tensors produced a number without checking module defaults.
- **Reference:** none — judgment call; hygiene, not a technique adoption.
- **Revisit if:** A Phase O refit produces a new canonical build — `CANONICAL_DATA_DIR` must move with it, and every stamped number is then stale by construction.

---

## 2026-08-20 — the six heads are scored directly, and the contact head is cleared as the owner of the strikeout shortfall
- **Decision:** `src/analysis/f3_heads.py` scores each of the six factor heads on 705,344 held-out 2024 pitches, by count and handedness, against the observed rate per cell. Conditioned heads get the OBSERVED `ev`/`la` bins, so the number is a head diagnostic, not a compounded-decode one.
- **Alternatives:** Score only the composed wOBA scalar (rejected: it wouldn't say whether any head was calibrated). Free-run the autoregressive chain and score the decoded joint (rejected: measures decode drift, a different question).
- **Rationale:** The open Phase E residual was a strikeout rate failing population-matched at -4.78%, with the contact-split table suspect. F.3 measures the contact head at **-1.40% overall, negative in all twelve counts** (worst 3-0 at -2.11%, 0-2 at -2.05%). Under-predicting contact over-predicts whiffs, pushing strikeouts up — opposite the shortfall, so the contact head can't own it, and fixing it would worsen the gap. The swing head splits -0.41% non-two-strike and +1.37% two-strike, cancelling into E.6's aggregate +0.27%; early-count under-swing tracks the walk excess through a path E.6 never measured (E.6 covered only terminal three-ball counts). Expected bin index runs -2.6% on `la`, -2.3% on `spray`, +0.7% on `ev`.
- **Coverage shortcut:** cell gaps run roughly 8-14 unclustered standard errors; **clustered intervals weren't computed**. Batter clustering would widen these — the signs and all-twelve-count consistency carry the argument, not nominal significance.
- **Reference:** none — judgment call on diagnostic scope; the scorer reuses `v1.factor_masks` rather than reimplementing head nesting.
- **Revisit if:** The strikeout residual is re-attacked. The contact head is excluded; the early-count swing rate and E.10's count-chain independence assumption remain candidates.

---

## 2026-08-20 — identity is worth a real but minority share of what the model knows about process
- **Decision:** `src/analysis/f4_process.py` reports held-out NLL per head against two references: the ensemble with hitter embedding replaced by the reserved zero row (`identity_gain`), and a Laplace-smoothed empirical frequency table conditioned on count and handedness, fit on train seasons only (`gain_vs_reference`).
- **Alternatives:** Report accuracy or AUC per head (rejected: the heads are calibrated distributions, so a proper scoring rule is honest). Use a shuffled-identity reference instead of cold-start (rejected: shuffling injects another hitter's signal rather than removing identity).
- **Rationale:** Every project gate scores a composed scalar, which can't say whether the model learned process. It did: model NLL beats both references on all six heads. Identity's share of the gain over the frequency reference: swing 10.6%, contact 17.9%, split 22.0%, **ev 34.8%**, la 12.1%, spray 13.6% — real but minority, largest on exit velocity, where hitters most plausibly differ. This fits claim 1 failing its gate while the model still outranks any ladder rung.
- **Coverage shortcut:** 6.46% of eval pitches belong to hitters with no trained embedding row; there the model and cold-start references coincide, diluting the identity gain.
- **Reference:** none — judgment call; cold-start ablation against a reserved index (`RESERVED_HITTER_INDEX`) is the project's own mechanism, not imported.
- **Revisit if:** Phase O changes embedding dimension or regularization — identity share is the natural re-measure, since it is what an embedding change should move.

---

## 2026-08-20 — a handedness-pooled score exists, and it is a description, not a gate
- **Decision:** `src/analysis/f5_pooled.py` collapses the pitcher-hand column to one value and reruns the unchanged `claim1_eval` referee, answering how well the model predicts overall wOBA rather than the platoon split. Predictions pool by **prior** side-specific PA, normalized within batter, with an even split for hitters with no prior record.
- **Alternatives:** Weight pooling by eval-season PA per hand (rejected as leakage: conditions the forecast on the held-out sample's own exposure). Write a second scoring path (rejected: one more thing to keep in step).
- **Rationale:** The evaluation surface is (batter, season, pitcher hand) triples, since the claim is about the split — nothing measured overall accuracy. Pooled, it scores PA-weighted RMSE 0.03904 against a 0.02863 noise floor, weighted rank correlation 0.486, and skill +0.161 over no-information — positive in every stratum, including low exposure at +0.143, where `c1_raw` is -1.49 and `c1_bucketed` -0.070.
- **Coverage shortcuts:** only the model, both C.1 variants, and no-information are pooled — C.2 needs fitted prior parameters, C.3 a GBM refit, so **neither named gate opponent is in the table**. `STRATUM_BOUNDARIES` was calibrated on side-specific PA, and pooled exposure runs roughly double, so these strata aren't comparable elsewhere; rows carry `stratum_basis = pooled_prior_pa`.
- **Reference:** none — judgment call on reporting scope.
- **Revisit if:** The pooled comparison is wanted as a real ladder — that needs regenerating C.2/C.3 predictions and recalibrating strata on pooled exposure.

---

## 2026-08-20 — two latent traps in the Phase E audit modules are closed, and no committed number moves
- **Decision:** Removed the `.fillna(0)` applied to `exact_ball` alone in `e_resample.main`, and added the missing `out_dir.mkdir` in `e_take_mass.main`.
- **Alternatives:** Leave both and log them (rejected: the `fillna` silently biases a reference downward, the kind of thing that surfaces as a wrong exoneration months later).
- **Rationale:** `sampled_ball`/`league_ball` were averaged raw while `exact_ball`'s empty cells were coerced to P(ball)=0, inflating the reported draw gap and contradicting `_take_means`, which returns NaN there by design. Checked first: **all 48 audit cells are non-NaN in the committed artifact**, so notebook 05's E.7 exoneration is unaffected — the fix is a no-op on today's numbers. Four further findings from the test backfill stay unfixed, recorded as known: a dead `hasattr` branch in `resampler_audit`, a `relative_gap` divide-by-zero risk (0 of 44 committed cells non-finite), a wrongly-shaped empty return from `paired_rows`, and a `KeyError: 'low'` in `e_min_pa_sweep.sweep` when a threshold censors the decisive stratum.
- **Reference:** none — judgment call, bug hygiene.
- **Revisit if:** Any Phase E audit module runs on a different season or smaller sample, where empty cells and censored strata become reachable.

---

## 2026-08-20 — the pooled skill column is barred as a Phase O selection axis
- **Decision:** No Phase O arm may be selected, ranked, or promoted on `f5_pooled`'s skill score, RMSE, or rank correlation. Selection stays on the frozen claim-1 metric over (batter, season, pitcher hand) rows, stratified by side-specific prior exposure; the pooled table is description only.
- **Alternatives:** Treat the pooled column as a secondary tiebreak (rejected: a tiebreak is a selection axis with extra steps). Promote it to a real ladder by regenerating C.2/C.3 pooled and recalibrating strata (rejected as scope, and it wouldn't remove the objection below).
- **Rationale:** Pooling collapses the hand column carrying the platoon claim, so optimizing it optimizes general hitter sorting — the easier, solved problem — against the split the project measures. It's also the one column where the model is positive in every stratum, exactly where an unguarded metric captures selection. Frozen rule #2 already names claim-1 as arbiter; this closes the gap rather than adding a rule. The pooled table also omits both named gate opponents, so it couldn't adjudicate an arm even if permitted.
- **Reference:** none — judgment call, resting on frozen rule #2 and the coverage shortcut in the 2026-08-20 pooled-score entry.
- **Revisit if:** The headline claim is ever restated as overall hitter projection rather than the platoon split — that requires reopening the manifest, not this entry.

---

## 2026-08-20 — the headline becomes the measurement ceiling, and frozen rule #1's gate is reported as failed
- **Decision:** The headline claim moves from *the model beats the incumbent on held-out side-specific wOBA* to *side-specific wOBA has a reliability ceiling far below 1, and here is what fraction each method reaches*. Frozen rule #1's gate is reported failed, in the results, not the limitations. No rescue: no re-cut to left-handed hitters, no post-hoc stratum, no second gate.
- **Alternatives:** Drop the platoon claim, reframe as embedding-building with platoon left to a frontend query layer (rejected: converts a pre-registered null into an unstated one, leaving no falsifiable claim). Restrict the claim to left-handed hitters, where the model reaches 0.569 of ceiling against right-handers' 0.122 (rejected: E.15 Part 3's noise-corrected LHB within-stand variance is **negative** at −3.75e-05, so the asymmetry's own supporting statistic is broken, and the L/R difference was never tested).
- **Rationale:** `e5_platoon_paired.csv` has `rank_favours_model` false in every stratum, all rank intervals crossing zero; E.15 Part 2 puts noise at 86.6% of platoon-differential variance — the gate failed not because the model is uninformative, but against a target admitting a maximum rank correlation of 0.356. A failed pre-registered gate reported beside the ceiling that explains it beats an unfalsifiable reframe, needing no frozen rule reopened. By-stand ceiling fractions stay in the table as description, with the negative-variance caveat.
- **Reference:** Franks, D'Amour, Cervone & Bornn (2016), *JQAS* 12(4):151–165, DOI 10.1515/jqas-2016-0098 — primary citation; their *discrimination* \(D_{sm} = 1 - E[V[X]]/V[X]\) equals this reliability by the law of total variance. Spearman (1904), DOI 10.2307/1412159, for the attenuation bound (\(\sqrt{\text{reliability}}\)) turning reliability into a rank-correlation ceiling; same quantity as Lord & Novick's (1968) index of reliability — Revelle, *Psychometrics with R* ch. 7, is the modern statement; the Lord & Novick page is unverified. Brown (2008), *Annals of Applied Statistics* 2(1):113–152, DOI 10.1214/07-AOAS138, is the peer-reviewed baseball precedent for the ceiling as estimand. **Gap:** no source publishes a numeric hitter-projection ceiling; Phase M's number is ours, a measurement, not literature agreement. *The Book* p.157 remains UNVERIFIED, unrelied on.
- **Revisit if:** Closing the C.2/C.3 differential gap shows an incumbent at a materially higher ceiling fraction, making the ceiling a story about this model rather than the target.

---

## 2026-08-20 — Phase O is narrowed from arm selection to hyperparameter tuning
- **Decision:** Phase O tunes learning rate, warmup, and epoch budget on the frozen D.10 architecture, arm, loss, features, and tensor build. Selection runs on the ledger's `reference` column — held-out 2024 unweighted log loss per scored row — and nothing else; **claim 1 is not read during Phase O**, tiebreak or otherwise. Budget: two overnight sessions, hard cap. The original scope (surviving D.8 arms, embedding dimension, the flagged-five block, spray, the bilinear term) is withdrawn, as is the optional pooled sampler.
- **Alternatives:** Keep Phase O as arm selection on claim 1 (rejected: makes the headline a selected number). Run no training; make Phase O purely measurement (rejected: conflates arm selection with tuning — the 0.000 correlation between claim 1 and held-out likelihood, measured across seven already-converged arms, shows likelihood can't *rank* good models, not that it can't *detect* an undertrained one). Select arms on F.5's pooled skill column (barred above).
- **Rationale:** Learning rate, warmup, and epoch budget were held at one value each for all 119 ledger runs, so none is evidence for anything — a null on an untuned model is weaker than the same null tuned. Likelihood answers Phase O's only question — is this run undertrained — and is wrong for choosing between arms, so it's legitimate here and barred above. `src/analysis/o1_select.py` fixes the promotion rule in code before any o1 run exists, so it can't change after seeing which arm it promotes, and doesn't import `claim1_eval`.
- **Reference:** Warmup: Goyal et al. (2017), arXiv:1706.02677, first states it for large-batch SGD. Liu et al. (2020), arXiv:1908.03265, attributes it to high-variance early Adam second-moment estimates. Ma & Yarats (2021), arXiv:1910.04209 (AAAI), refute this, ground the effect in early update magnitude, and recommend linear warmup at ≈\(2/(1-\beta_2)\) steps — 2,000 at β2=0.999 against the 719 used here, a limitation not a choice. **Unsourced synthesis:** warmup reducing D.5's gradient-(b) displacement of rarely-updated rows is mine. Sourced: Duchi et al. (2011), *JMLR* 12:2121, on per-coordinate normalisation giving infrequent features large steps — displacement is AdamW as designed, not a bug; Kunstner et al. (2024), arXiv:2402.19449 (NeurIPS), and Qiu et al. (2025), arXiv:2505.05605, on frequency-adaptive embedding-table rates. O.2 is pre-registered as a diagnostic; a null there is a finding, not a failure.
- **Revisit if:** The o1 grid promotes an arm whose margin comes from a single seed — confirmation runs then decide, and the two-seed screen is not reported as the result.

---

## 2026-08-20 — batch size and weight decay are quarantined through Phase O
- **Decision:** Batch size stays 8,192 and weight decay 1e-2 for every Phase O run, neither exposed as a training argument. If Phase M has time, they enter as a pre-registered arm fixed before any claim-1 number is read, checked by re-measuring F.4's identity share.
- **Alternatives:** Sweep them in Phase O alongside learning rate (rejected: contamination, per rationale below). Drop them permanently (rejected: the decay-to-gradient ratio is a real, unmeasured lever — never looking is not a finding).
- **Rationale:** The two are one setting: AdamW decays every parameter every step, while an embedding row gets a gradient only in batches containing that hitter — their ratio *is* the shrinkage applied to low-exposure hitters, the quantity C.2 estimates and claim 1 is about. Tuning toward the low stratum would implement C.2's shrinkage inside the network, then score against C.2. The D.5 gradient (b) artifact doesn't implicate decay regardless: 10th-percentile-exposure rows see a 23.9:1 decay-to-gradient ratio yet finish *furthest* from the origin (mean norm 0.74–1.00 vs. most-exposed rows' 0.52–0.61, across all five seeds) — opposite of binding decay, pointing at AdamW's per-coordinate second-moment normalisation instead.
- **Reference:** `results/phase_d/d5_level_attribution.json` `gradient_b`; the decay-to-gradient table in `docs/phase-d-spec.md`.
- **Revisit if:** Phase O's tuned build leaves gradient (b) intact — learning rate and warmup aren't the lever, and the ratio deserves a pre-registered arm sooner than Phase M's spare time allows.

---

## 2026-08-20 — Phase M is added as the measurement phase, and the 2025 run is spent after it
- **Decision:** A new Phase M sits between Phase O and Phase V and carries the paper: close the C.2/C.3 platoon-differential gap, build the per-exposure-stratum platoon ceiling, build the level-side ceiling (O.1), fix E.14's interval coverage by adding target sampling noise, re-run gradient (b) against Phase O's pre-registered prediction, and reconcile E.5 and F.5 populations onto their intersection. Order is strict: O → M → V → the 2025 final run → write-up. The low-exposure stratum becomes the ceiling argument's illustration rather than the gate it failed.
- **Alternatives:** Fold measurement into the write-up (rejected: items 1-2 are new computation, not prose). Spend the 2025 run before measurement so the paper reports test-season numbers throughout (rejected: it's spent once, and running it before the build is final makes it a number the build was selected on).
- **Rationale:** Closing the C.2/C.3 gap is a prerequisite: `e5_platoon_scores.csv` carries only `delta_pred` and `delta_route_a`, so the ceiling table has no incumbent, and the reframe is undefensible without one. The per-stratum ceiling isn't a re-read either — E.15 stratified by stand, not exposure, so the stratum the project is about has no ceiling figure. E.14's interval reports seed spread alone and can't cover a target that is itself a small-sample measurement, a stated precondition for the query layer displaying platoon numbers.
- **Reference:** `results/phase_e/e15_ceiling.json`; `results/phase_e/e5_platoon_scores.csv`.
- **Revisit if:** The per-stratum ceiling can't be estimated in the low stratum at acceptable precision — the illustration moves to a stratum where it can, and says so.

---

## 2026-08-20 — the run ledger gains lr and warmup columns, and the pre-Phase-O history is backfilled
- **Decision:** `results/phase_d/sweep_log.csv` gains `lr` and `warmup_steps`. All 119 pre-Phase-O rows are backfilled with `0.001` and `0` rather than left blank. *(Amended 2026-08-20: a third column, `data_dir`, was added at the same time, learning rate written in one canonical spelling, and the `.bak` dropped for git history — see the entry below.)*
- **Alternatives:** Leave the columns blank for historical rows (rejected: blank reads as *unknown*, and it isn't). Start a separate Phase O ledger (rejected: splitting would put two stages' `reference` columns in files that look independently authoritative).
- **Rationale:** Every one of those runs used the module constants, so the backfill records a fact, not an assumption — one learning rate covering 119 rows is itself Phase O's justification. `append_ledger` writes a header only when the file is new, so the schema change had to apply to the existing file or later rows would use the old header in the new field order.
- **Reference:** none — provenance hygiene.
- **Revisit if:** A pre-Phase-O run is ever found launched with an overriding `--train-args`, making a backfilled cell false.

---

## 2026-08-20 — d10's tensor build is established by reproduction, and the ledger records the build from now on
- **Decision:** Every ledger row carries a `data_dir`. `d10` backfills to `data/processed/phase_d5` (its `block` arm to `phase_d5_noblock`, `d9` to `phase_d_split`), and Phase O's `--data-dir` pin is confirmed, not assumed correct. Learning rates go through a single `canonical_lr` for groupability; `append_ledger` now refuses to write when the on-disk header doesn't match `LEDGER_FIELDS`. `sweep_log.pre_o.bak` is deleted: git already holds the pre-migration version, and a second copy is one more thing readable as authoritative.
- **Alternatives:** Trust the source comment that d10 ran on the D5-R17 rebuild (rejected: a comment isn't evidence, and the Phase O guard exists because a moved build is invisible in every column). Leave `data_dir` blank for history (rejected here, unlike the `lr` backfill, since d9/d10's value *is* recoverable; it stays blank for `screen`, `d6`, `d8`, where it genuinely isn't).
- **Rationale:** Two lines of evidence agree. Re-running d10 baseline seed 0 on `phase_d5` reproduces its logged epoch-0 train/validation loss exactly (1.05990 / 1.04681); the same command on `phase_d` raises `KeyError('split')`, since that build predates the three-class contact split — no `--split` run (every d10 arm) could have used it. Had it gone the other way, the o1 sweep would have returned `guard_failed`, since `reference` is log loss over quality bins, and the builds have different bin edges (`e_bip_value.py:232`: 0.96371 vs. 0.92313, ~400 noise-floor sds apart). The `1e-3`/`0.001` spellings were a live defect: `knobs()` would have written `0.001` into a column whose 119 rows said `1e-3`.
- **Reference:** `results/phase_o/provenance_probe.log`; `results/phase_d/logs/d10_baseline_s0.log`; `src/analysis/e_bip_value.py:232`.
- **Revisit if:** A `d8`-or-earlier number is read down the same column as a `d9`/`d10` one — the blank cells then need the same establishment, not assumption.

---

## 2026-08-20 — Phase O selects on the 2024 season, and Phase M must label its 2024 numbers accordingly
- **Decision:** Phase O's selection metric stays `reference` — held-out log loss on the 2024 validation split. The consequence is disclosed, not mitigated: any claim-1 number computed on 2024 from the tuned build is **post-selection and descriptive**. The confirmatory number is the 2025 run, sealed until Phase M finishes. The `o1_select` docstring carries this.
- **Alternatives:** Split 2024 into tuning and measurement halves (rejected for now: it halves exposure in exactly the low-exposure stratum the ceiling argument is about, and changes a frozen split). Tune on 2023 (rejected: a training season, so not held out at all).
- **Rationale:** This is the frozen walk-forward protocol as designed — 2024 selection, 2025 test — and every Phase D ablation already selected on 2024, so Phase O adds no new leak. It does add a build whose 2024 numbers are optimised; Phases E/F report claim 1 on 2024, and carrying those forward unlabelled would quietly convert them into selected numbers. The fix is a labelling obligation on Phase M, not a change to Phase O.
- **Reference:** Raised by the 2026-08-20 review of the Phase O implementation; architecture plan §5, selection frame.
- **Revisit if:** Phase M's ceiling estimate proves sensitive to the tuned build — if it moves between incumbent and tuned arm, the ceiling isn't the data property claimed, a bigger finding than Phase O.

---

## 2026-08-25 — β₂ and ε stay at the AdamW defaults, and are not swept in Phase M
- **Decision:** β₂ = 0.999 and ε = 1e-8 for every remaining run. Neither enters Phase M as an arm, pre-registered or otherwise.
- **Alternatives:** Sweep β₂ over {0.98, 0.999} in Phase O alongside learning rate (rejected then as untestable before learning rate was settled — now discharged, since it is settled). Lower β₂ to shorten the second-moment window on rarely-updated rows (rejected on the power argument below).
- **Rationale:** A 10× learning-rate range moved held-out loss at most 0.8 SE against a 2 SE bar, with the two extreme arms 10 SE the wrong way. Learning rate scales every step directly; β₂ only reshapes how fast the denominator adapts — a knob with strictly weaker leverage than one that failed the bar won't clear it either, and an overnight session establishing that costs Phase M time better spent elsewhere. The 719-step warmup grid stays short against the ≈2/(1−β₂)=2,000-step default, the same limitation logged at the grid's design, not reopened here.
- **Reference:** Ma & Yarats (2021), arXiv:1910.04209 (AAAI), for the β₂-to-warmup relation and default; Liu et al. (2020), arXiv:1908.03265, for the second-moment-variance account it refutes. The decisive argument is the measured o1 margin table, not a source.
- **Revisit if:** The quarantined-knob arm runs in Phase M and moves `reference` by more than 2 SE, showing optimiser settings have leverage after all and making the second-moment family worth one grid.

---

## 2026-08-25 — Finding: Phase O returns `incumbent_stands`, and Phase M runs on the D.10 baseline ensemble
- **Decision:** The o1 3×2 factorial at two seeds returns `incumbent_stands`: `lr1e3` wins at margin 0.0; best challenger `lr1e3_warm` reaches +0.8 SE against a 2 SE bar; the remaining four arms run negative at −1.5, −2.7, −10.3, −10.5 SE. Grid complete, no underpowered arms, build guard passed at 4.5e-5 drift against 4.2e-4 tolerance. Phase M runs on the D.10 build unchanged — the existing five-seed `d10_baseline_s0..s4` checkpoints, not the two fresh `o1` `lr1e3` seeds, a screening artifact and never a build.
- **Alternatives:** Promote `lr1e3_warm` on its +0.8 SE and confirm at five seeds (rejected: the promotion rule was fixed in code before any run existed, and its two seeds were 1.02567 and 1.02585 — a 1.27e-4 spread, wider than the margin itself). Extend the grid wider or longer (rejected: no arm in the swept 10× range beat the incumbent, and monotone failure isn't evidence of an interior optimum).
- **Rationale:** `incumbent_stands` is a real Phase O result, the one the spec named as such — the earlier note that learning rate was fixed at 1e-3 across all 119 runs and never varied is now answered: it was varied over a 10× range, and 1e-3 was already best. O.2 isn't settled here; it's measured separately below.
- **Reference:** `results/phase_o/o1_selection.json`; `results/phase_d/d5_level_attribution.json` `gradient_b`, whose five per-seed measurements read the `d10_baseline` checkpoints.
- **Revisit if:** Phase M's schedule leaves room for the quarantined-knob arm — the 2026-08-20 entry's revisit condition is now met, since neither learning rate nor warmup moved gradient (b), and that arm's strength must be fixed before any claim-1 number is read, checked by re-measuring F.4's identity share.

---

## 2026-08-25 — Finding: The promotion bar is the 95% convention, and the noise floor understates run-to-run spread
- **Decision:** `MARGIN_SES = 2.0` stands as the two-sided 95% convention; no derivation exists or is retrofitted. The o1 SE column's `sd` is the quietest arm's across-seed spread (1.04e-4), against a six-arm pooled spread of 4.08e-4.
- **Alternatives:** Recompute on the pooled sd (rejected: the estimator was fixed before any run; changing it after seeing the table is what pre-registration forbids). Derive the bar from a power calculation (rejected: no effect size was ever named).
- **Rationale:** `incumbent_stands` holds under either estimator, at any threshold from 1 SE up — neither point rescues the result.
- **Reference:** `src/analysis/o1_select.py:67`; `results/phase_o/o1_selection.json`.
- **Revisit if:** an arm lands between 1 and 2 SE, where the bar starts doing real work and needs a real derivation.

---

## 2026-08-25 — `shrinkage_in_woba_direction` tests |projection|, not the signed projection
- **Decision:** `d5_level.gradient_b` now regresses **|projection|** on exposure, carried in a third series `abs_projection`; the signed `projection` series stays unchanged in the quintile table. Self-checks cover this flag in both the planted-shrinkage and planted-anti-shrinkage directions.
- **Alternatives:** Re-centre the projection on the population mean (rejected: the wOBA axis has a meaningful zero — the league-average hitter — so distance from zero is what "closer to the origin" names).
- **Rationale:** The signed projection crosses zero: slope turns positive when low-exposure rows sit negative — the anti-shrinkage case, not shrinkage. Under the corrected test the flag is `false` in all five d10 seeds; the abs-projection slope runs −0.0050 to −0.0074 per 1,000 train pitches, and the bootstrap interval excludes zero in every seed. D.5's anti-shrinkage holds along the wOBA axis specifically, not just total norm.
- **Reference:** `src/analysis/d5_level.py:331-361, 416-433`; `results/phase_d/d5_level_attribution.json`.
- **Revisit if:** any other flag in the module derives from a slope on a sign-crossing quantity.

---

## 2026-08-25 — Finding: O.2 returns a null: warmup rescales the embedding space and leaves the exposure gradient intact
- **Decision:** The pre-registered O.2 expectation — that the warm build's rarest quintile norm falls toward the most-exposed quintile's — is **contradicted**. Phase O closes on `incumbent_stands`; Phase M runs on `d10_baseline`'s five-seed ensemble unchanged. Warmup is ruled out as the anti-shrinkage lever.
- **Alternatives:** Widen the warmup grid toward the ≈2/(1−β₂) = 2,000-step default (rejected — the same limitation logged at the grid's design: a diagnostic is not retuned around). Run the remaining three warm seeds (rejected: seeds are matched pairs, agreeing on every measure).
- **Rationale:** `torch.manual_seed` fixes initialisation and batch order, so `d10_baseline_s{n}` and `o1_lr1e3_warm_s{n}` form matched pairs. Warmup shrinks every quintile by a near-uniform factor (0.915–0.927 across the five quintiles, seed 0), so the q1/q5 norm ratio moves −1.3% and −0.04%, and the exposure-normalised slope moves the wrong way — a global rescaling, not the differential pooling O.2 predicted. **n = 2, descriptive only, never a promotion criterion.**
- **Reference:** `results/phase_o/o2_gradient_b.json`; `results/phase_d/d5_level_attribution.json` gradient_b, paired d10 reference.
- **Revisit if:** the quarantined-knob arm runs in Phase M — with warmup ruled out, batch size and weight decay are the only optimiser-side explanation of anti-shrinkage left.

---

## 2026-08-25 — Finding: The anti-shrinkage mechanism is AdamW's first step, not a short random walk
- **Decision:** The mechanism sentence in the 2026-08-20 quarantine entry narrows to its first-step clause. Rare rows are neither lightly updated nor displaced in an arbitrary direction.
- **Alternatives:** none — this is an arithmetic correction to a recorded claim.
- **Rationale:** Every pitch is seen every epoch, so a 30-pitch row takes ≈510 updates against ≈12,200 for the most-exposed; pure diffusion over 510 steps reaches ≈0.13 against an observed 0.74–1.00. The mean projection is −0.16 to −0.22 in all five seeds — a consistent direction. What survives: AdamW's bias correction makes m̂/√v̂ exactly sign(g) on step one, so a row's first update is a full learning-rate step in every coordinate. This is estimate arithmetic, not measurement.
- **Reference:** none — judgment call on how an existing measurement is described; underlying numbers in `results/phase_d/d5_level_attribution.json` gradient_b.
- **Revisit if:** the write-up needs a quantitative mechanism claim — requiring displacement measured across epochs, not bounded.

---

## 2026-08-25 — The build guard verifies by exact reproduction, not by drift within a tolerance
- **Decision:** `o1_select.build_check` compares incumbent and noise-floor arm **seed by seed**: shared seeds pass only on bit-identical `reference` (`drift_is_informative: false`); no shared seeds falls back to the 4 sd drift tolerance, the only test then available. The grid check is now symmetric: a ledger arm missing from `EXPECTED_ARMS` returns `incomplete_grid`.
- **Alternatives:** Keep the drift tolerance alone (rejected: on the o1 ledger the incumbent rows *are* two of the five floor rows, so drift was algebraic on its own reference set, detecting nothing). Tighten the tolerance (rejected: doesn't fix a statistic computed against itself; 4.17e-4 was twice the promotion bar).
- **Rationale:** The guard is a units check — `phase_d`/`phase_d5` carry different quality-bin edges, so a wrong-build run lands on a different scale (0.96371 vs 0.92313) while every ledger column lines up. Bit-identical reproduction of a same-seed run is a direct, stronger test than any tolerance. The o1 incumbent reproduces `d10/baseline` seeds 0–1 exactly at 1.02584/1.02585, `best_epoch` 17 both — build verified. Verdict unchanged: `incumbent_stands`, winner `lr1e3`.
- **Reference:** `src/analysis/o1_select.py` `build_check`; `results/phase_o/o1_selection.json` `guard`; `tests/test_o1_select.py` — four tests, both regimes plus the undeclared-arm path.
- **Revisit if:** a stage runs the incumbent on fresh seeds only — the guard reverts to the tolerance, its width now load-bearing.

---

## 2026-08-25 — The warmup scheduler's verification gate is discharged
- **Decision:** `train.LinearWarmup` is cleared to run. Two of the seven ml-engineer gates re-run on the warm path — overfit-one-batch with the schedule in the loop, and bit-identical losses across two same-seed runs. The other five (shape assertions, loss scale at init, split boundary, eval-mode hygiene, decode one batch) are **not** re-run; this entry records why.
- **Alternatives:** Re-run all seven (rejected: five are properties of the model and build the scheduler cannot reach — it multiplies a learning rate, touching no tensor, split, or decode). Skip the gate on the no-op proof alone (rejected: that proof covers `warmup_steps=0`, the path where the scheduler doesn't exist).
- **Rationale:** A learning-rate schedule has two failure modes these gates can reach: a wrong scale stops the run learning, and a state-carrying scheduler makes two same-seed runs diverge. Both are tested. The no-op path is proved from the ledger, not argued: `o1/lr1e3` (`warmup_steps=0`) reproduces `d10/baseline` seeds 0–1 bit-identically at 1.02584 and 1.02585 — the scheduler changed nothing on runs that don't use it. The schedule's shape, its permanent stand-down, and `warmup_for`'s refusal of a warmup overlapping the first plateau cut were covered.
- **Reference:** `src/model/train.py:118-168`; `tests/test_o1_select.py` — `test_the_warm_path_still_overfits_one_batch`, `test_two_warm_runs_at_the_same_seed_are_bit_identical`; `results/phase_d/sweep_log.csv` rows `d10/baseline`, `o1/lr1e3` seeds 0–1.
- **Revisit if:** the scheduler touches anything but `param_group["lr"]`, or a schedule runs past epoch 2.

---

## 2026-08-25 — Finding: Phase O's selection metric carries a winner's curse; the decisive comparison is draw-balanced
- **Decision:** Recorded as a Phase O limitation, not corrected. `reference` equals `best_val_loss` on all twelve o1 rows — a minimum over `best_epoch + patience + 1` draws of the same held-out split it's scored on. Arms draw unequally (≈12 for `lr3e3`, ≈21 for `lr1e3`, ≈31 for `lr3e4`), so low-learning-rate arms get an optimistic bias the fast arms don't.
- **Alternatives:** Score on a third split held out from early stopping (rejected: re-runs the whole grid to fix a bias that doesn't change the verdict, and 2025 is sealed). Penalise by draw count (rejected: no defensible penalty exists at these sample sizes).
- **Rationale:** The bias inflates every arm the same direction, so it can't manufacture the null. It could in principle hide a real winner among the fast arms — but the only challenger with a positive margin is `lr1e3_warm`, at `best_epoch` 17 and 18 against the incumbent's 17 and 17. The decisive comparison is between two arms with the same draw count, so the curse doesn't touch it. `corr(best_epoch, reference)` is +0.35 across the twelve rows — the direction the bias predicts.
- **Reference:** `results/phase_d/sweep_log.csv`, o1 stage; `results/phase_o/o1_selection.json`.
- **Revisit if:** a future stage selects across arms whose `best_epoch` differs by more than ≈2×, where the bias stops being common-mode.

---

## 2026-08-25 — Phase M's promotion rule, pre-registered before any Phase M run exists
- **Decision:** Any Phase M promotion comparison uses: (1) the **pooled** across-seed sd from every arm, not the quietest arm's; (2) a **Bonferroni-corrected** threshold over the number of challengers — ≈2.9 SE for five challengers at the normal approximation, ≈3.5 SE on t at 4 dof; (3) a confirmation step that **discards the screening seeds** and re-tests on fresh ones. Phase O's rule is **not** amended — it was fixed before its runs and stands as it ran.
- **Alternatives:** Carry Phase O's rule forward unchanged (rejected: its `sd` was the quietest arm's, ≈4× too small and anti-conservative; 2 SE at 4 dof is p≈0.116, not 0.046; the family-wise false-promotion rate across five challengers was ≈46%; its confirmation step was a seed *count* reusing the screening seeds and shrinking the SE, so confirmation was easier to pass than the screen). Retrofit the correction onto Phase O (rejected: choosing an estimator after seeing the table is what pre-registration forbids, and `incumbent_stands` holds under either estimator anyway).
- **Rationale:** Phase O's verdict is a null, and every defect above is anti-conservative — each made promotion *easier*, and none produced the null. That's why the fix is prospective, not a re-run. Phase M is where a positive result would be claimed, and a positive result under an anti-conservative rule is the failure mode that matters.
- **Reference:** `~/os/knowledge/frameworks/research-standards.md` §6; `results/phase_o/o1_selection.json`, source of the o1 spread the pooled estimate draws from.
- **Revisit if:** Phase M's comparison set is a single challenger against a single incumbent — the Bonferroni term is 1, and only the pooled sd and fresh-seed confirmation apply.

---

## 2026-08-30 — The τ² route rule: B′ primary, A sensitivity, B provenance
- **Decision:** Phase M estimates within-stand τ² by three reported routes. **Route B′** — `c2_bivariate_eb.fit` on the 2016–2024 window restricted to the M.6 intersection population — is the pre-registered primary. **Route A** (2024 observed minus modeled noise) always reports beside it as the same-season sensitivity, carrying a ±3% noise-model fragility band. **Route B** (the unrestricted nine-season fit, 0.00059034) is one provenance row; the B→B′ delta is the population diagnostic. The rule is fixed before B′ is computed.
- **Alternatives:** Bracket A and B with no primary (rejected as headline: the population mismatch is fixable and B′ fixes it — the bracket survives as the §10 fallback if B′ degenerates). Route B primary (rejected: fitted on hitters and seasons the claim isn't about). Route A primary (rejected: τ²_A = 0.00419563 − 0.00407848 is a near-total cancellation — a 3% noise-model error swings it ~100% — and 88%-of-ceiling flatters the model, so a post-hoc pick would read as selection).
- **Rationale:** B′ makes the nine-season route directly comparable to A on population, leaving window and estimator stability as the only axes of disagreement. A's fragility is structural, known before any number is read, so naming B′ primary today isn't selection. Restricting past seasons to 2024 eval hitters conditions on survival to 2024 — correct for this claim, and labeled.
- **Reference:** `results/phase_e/e15_ceiling.json` (corrected provenance, 2026-08-29 handoff); `docs/phase-m-spec.md` §M.0.
- **Revisit if:** B′'s τ² is negative or the restricted fit degenerates — the pre-decided fallback (spec §10.3) reverts the headline to the A/B bracket, B′ reported as degenerate.

---

## 2026-08-30 — Route C gets one bounded diagnostic, and its worst case demotes Route A
- **Decision:** The split-half route's −0.366 gets one day, hard cap: a code audit of the split, plus a null simulation of the split-half estimator under τ²=0 and τ²=B′, using each hitter's real PA count and E.15's noise model. Verdicts are pre-decided: inside the τ²=0 range → reported as consistent with small τ² (evidence line, never an estimator); bug found → fixed and re-emitted descriptively; outside both ranges with no bug → the shared noise model is suspect, the headline stands on B′ alone, Route A demotes to "reported, unvalidated-noise-model caveat," and Route C is reported unexplained, flagged at the top of `m0_routes.json`.
- **Alternatives:** Drop it as broken (rejected: forfeits a third vote on the contested parameter, and a reviewer would run the check we skipped). Rebuild it as a full third route (rejected: at ~35 PA per half the estimate is likely irreducible, and days would be spent proving that).
- **Rationale:** Route A *is* the noise model — a subtraction whose survival depends on the model being right to within ~3%. If the world can produce −0.366 at no skill level our simulation allows, the noise model mis-describes the data and A's near-cancellation can't be trusted; B′ doesn't lean on the 2024 noise model the same way. Pre-deciding the demotion keeps the choice out of whichever answer flatters the model.
- **Reference:** `docs/phase-m-spec.md` §M.0 (Route C diagnostic), §10.2.
- **Revisit if:** the simulation can't be validated by the §9 planted-recovery checks — its verdict carries no weight, and the diagnostic is reported as inconclusive.

---

## 2026-08-30 — M.1 scores C.2 and C.3-full on the differential cut, and no other rung
- **Decision:** The ceiling table's opponents are C.2 (the incumbent) and C.3-full (the ML control). `no_info` and the C.1 rungs are omitted from the differential cut, and the paper states the omission. If C.3-full can't emit side-specific predictions from existing artifacts, the table reports C.2 only (spec §10.1) — no differential head is improvised.
- **Alternatives:** All non-degenerate rungs (rejected: the ladder's decomposition logic was designed for the level claim; middle rungs answer no live question here and only add anchor points). C.2 only (rejected: leaves "ordinary ML would reach more of the ceiling" open on the exact cut the paper lives on).
- **Rationale:** Two questions are live on this cut — does the incumbent beat the model's fraction of ceiling, and would a GBM have done better. Two rows answer them; C.2's `predict()` emits side-specific wOBA, so both are scoring passes, not new modeling.
- **Reference:** `results/phase_e/e5_platoon_scores.csv`; `docs/phase-m-spec.md` §M.1.
- **Revisit if:** a reviewer requires the full ladder on the differential cut — the omitted rungs are then scoring passes, addable without re-opening anything.

---

## 2026-08-30 — The knob arm runs after Phase M, with its strength pre-registered now
- **Decision:** The quarantined-knob arm (2026-08-20 quarantine; revisit condition fired by O.2's warmup null) runs **after** all Phase M measurement items — not before, not inside. Its arms are fixed before any M number is read: `wd3e1` (weight decay 1e-2 → 3e-1) and `bs2048` (batch 8,192 → 2,048), two seeds each, matched pairs against `d10_baseline_s0,s1`. Judged on gradient (b) movement and on `reference` under the 2026-08-25 promotion rule (k=2) — promotion to the paper's build barred regardless of outcome, F.4's identity share re-measured on any arm moving gradient (b), and no M.0–M.3 number ever recomputed on a knob build.
- **Alternatives:** Run it before Phase M (rejected: ordering only matters if the arm could change the build M measures, and it can't — the quarantine's circularity argument bars promotion, and the ceiling is a property of the data, not the model; running first would delay the Oct 1 abstract's numbers). Inside Phase M (rejected: an overnight spent on a mechanism paragraph while headline work waits). Never (rejected: the fired revisit condition would be silently dropped).
- **Rationale:** The trigger obligates a decision, not an immediate run. Pre-registering the strengths now removes the contamination running-after would invite: knob values can't be chosen in light of M's numbers. Strength rationale: the 10th-percentile decay-to-gradient ratio is 23.9:1 at wd=1e-2, so decay needs ≥24× to bind — 30× binds with margin; a 4× smaller batch separates update-count effects from decay effects.
- **Reference:** `docs/phase-m-spec.md` §8; 2026-08-20 quarantine entry; `results/phase_o/o2_gradient_b.json`.
- **Revisit if:** Phase M's schedule collapses entirely — the arm moves to future work, and the paper says the mechanism is unresolved.

---

## 2026-08-30 — Phase M's contingencies are pre-decided, and the spec is the build authority
- **Decision:** `docs/phase-m-spec.md` is adopted as the authority for the Phase M build window. Its §10 converts five former blocking questions into pre-decided fallback rules (C.3 differential unavailability, Route C outside both nulls, B′ degeneracy, Route B reproduction failure, low-stratum CI precision — the last now numeric: a 95% bootstrap CI including zero moves the illustration to the narrowest-relative-CI stratum). Hard stops remain for frozen-split config, stratum redefinition, reading 2025, and anything off-spec. The build agent applies fired rules, flags them prominently, and continues; the knob arm stays out of the build window's scope.
- **Alternatives:** Leave the contingencies as stop-and-ask (rejected: each stop stalls an unattended window, and every one of the five had a decidable rule fixable before any number is read). Delegate the hard stops too (rejected: frozen rules are never delegated).
- **Rationale:** Pre-deciding contingencies before their trigger data exists is the same discipline as pre-registering the route rule — it removes every remaining point where a choice could be steered by results. These adjustments bind *how* the plan executes, never the claims, gates, metrics, or splits, which stand unchanged; the dated log entry is the audit trail against forking-paths concerns.
- **Reference:** `docs/phase-m-spec.md` §10; `docs/phase-m-opus-prompt.md`.
- **Revisit if:** a fallback rule fires in a way its wording didn't anticipate — spec §10.7 territory: stop and report, don't improvise.

---

## 2026-08-30 — Finding: Route B reproduces exactly; the spec's fitting-window prose does not describe the code
- **Decision:** §10.4 does not fire. A live `c2_bivariate_eb.fit` reproduces every committed `tau2_split_derived` to twelve decimals (L 0.0007322758887773689, R 0.0004917731167240981, S 0.000588220547130464); the per-hitter weight-average over M.6 lands on the spec's pooled 0.00059034. The spec calls Route B a "nine-season, 2016–2024 fit" — wrong prose: `c1_trailing.TRAILING_SEASONS = 3` makes the committed fit a 2021–2023 trailing window. The value stands as committed; the window is recorded correctly in `m0_routes.json`, and the spec text isn't retro-edited.
- **Alternatives:** Treat the mismatch as a reproduction failure and fire §10.4 (rejected: rule 4 covers a committed *value* failing to reproduce, and the value reproduces exactly — the defect is in a sentence describing the code, not the code or number). Refit Route B on an actual nine-season window to match the prose (rejected: replaces the committed incumbent with a new estimate after the route rule was frozen — exactly the substitution pre-registration forbids).
- **Rationale:** Route B exists in the rule as a provenance row, served by reproducing what was committed, not by matching its description. The window matters to one downstream reading: B and B′ must differ in *population alone* for the B→B′ delta to be a population diagnostic — and they do, since B′ uses the same three-season window with the hitter set restricted.
- **Reference:** `tests/test_m_report.py` — `test_unrestricted_c2_refit_reproduces_the_committed_route_b_tau2`, `test_committed_prior_parameter_file_matches_the_refit`, `test_pooled_route_b_lands_on_the_spec_value`; `results/phase_m/m0_routes.json` `reproduction`.
- **Revisit if:** a future stage refits C.2 on a genuinely different window — B and B′ would differ on two axes, and the population diagnostic would no longer isolate one.

---

## 2026-08-30 — §10 fallback rule 1 fires: M.1 reports C.2 only
- **Decision:** C.3-full is omitted from the M.1 differential table. `results/` holds no persisted fitted C.3 artifact, and `c3_gbm.predict` calls `c3_gbm.fit` inside the call, so side-specific C.3 predictions need retraining the GBM on the 341MB labeled pitch table — the condition rule 1 names. The omission is flagged at the top level of `m0_routes.json` and `m_summary.json` with the reason attached; no differential head is improvised.
- **Alternatives:** Retrain C.3 to fill the row (rejected: the spec pre-decided this is out of the window's scope, and a rule fires on its stated condition, not on whether the work is convenient). Report the row as unavailable without a reason (rejected: an absent control reads as a control that lost).
- **Rationale:** Recorded with a caveat that matters for how the gap is read: the refit is a *supported existing code path* (`src/analysis/c_report.py:364`) needing no new feature code — a scope boundary the spec pre-set, not a capability limit of C.3. The live question the row would have answered — whether ordinary ML reaches more of the ceiling on this cut — stays open; C.2's row shows the model doesn't beat the incumbent on it either way.
- **Reference:** `src/analysis/m_report.py` `C3_FULL_AVAILABILITY`; `tests/test_m_report.py::test_c3_full_omission_is_recorded_as_a_fired_fallback_not_a_silent_gap`; `docs/phase-m-spec.md` §10.1.
- **Revisit if:** C.3-full is refit for any other reason — the row is a scoring pass, addable without re-opening the route rule.

---

## 2026-08-30 — Finding: sqrt(reliability) is not a conservative bound on the rank ceiling at these exposures
- **Decision:** Every fraction-of-ceiling figure in Phase M is labeled as running a few percent relatively high; the Monte-Carlo rank ceiling is emitted beside the analytic one in `m0_routes.json`, not the analytic value alone. The analytic map is retained as the reported ceiling — it is not replaced by the simulated one.
- **Alternatives:** Report the simulated rank ceiling as the headline (rejected: a 300-draw estimate with sd 0.042 at the primary route, so substituting it trades a small known bias for a larger sampling error, and breaks comparability with E.15's committed table). Assert the textbook direction and move on (rejected: the §9.1 check measured the opposite, so the assertion would have been false).
- **Rationale:** `sqrt(reliability)` is derived under joint normality and bounds the Pearson correlation; the project scores with a rank correlation. Under a homoscedastic simulation the rank ceiling sits 3.6% below the analytic value, as the textbook predicts. Under the real 2024 exposure profile — a ~36× spread in per-hitter sampling variance — it comes in 3.9% *above* it: rank correlation is robust to the heavy-tailed observations low-exposure hitters contribute, and that robustness buys back more than joint normality costs. The Pearson arm of the same simulation matches the analytic value, so the reliability→ceiling map is verified exactly, and only the rank-vs-Pearson step is approximate.
- **Reference:** `src/analysis/m_ceiling.py` module docstring and `monte_carlo_ceiling`; `tests/test_m_ceiling.py` — `test_ceiling_formula_matches_monte_carlo_best_possible_correlation`, `test_rank_ceiling_gap_changes_sign_with_the_exposure_skew`; `results/phase_m/m0_routes.json` `monte_carlo_rank_ceiling`.
- **Revisit if:** a ceiling is quoted on a population with a materially flatter exposure profile — the sign of the gap reverts to the textbook direction.

---

## 2026-08-30 — Finding: Route C is retired: the negative reliability is an artifact, not a finding
- **Decision:** Route C's verdict is `uninformative_inside_both_nulls`: reported as evidence in neither direction. §10.2 does not fire — the value is inside both nulls, so the noise model stands, and Route A keeps its status as reported sensitivity.
- **Alternatives:** Read the LHB value as support for small τ² and Route A (rejected: also inside the τ²=B′ null, so it separates nothing). Report it unexplained (rejected: the simulation explains it, and an unexplained −0.366 invites an unsupported conclusion).
- **Rationale:** The audit found no bug in `split_half_reliability`: game-parity split, per-hitter pairing, exposure floor on both halves, pooled row centred by stand. A null simulation using E.15's noise model puts the raw LHB half correlation (−0.155) at the 18th percentile of τ²=0 and 6th of τ²=B′ — ordinary under both. The alarming −0.366 is Spearman-Brown (2r/(1+r)) on a negative input, which only magnifies it. At a median 43 PA vs LHP per half, sampling variance is too large to discriminate — the bounded outcome the spec allowed.
- **Reference:** `results/phase_m/m0_route_c_diagnostic.json`; `src/analysis/m_ceiling.py` `simulate_split_half`, `locate_in_simulation`; `tests/test_m_ceiling.py` — `test_split_half_estimator_recovers_positive_reliability_at_large_exposure`, `test_split_half_null_at_zero_tau2_straddles_zero`.
- **Revisit if:** a split-half estimate is wanted as more than a diagnostic, needing per-half exposures several times larger than 2024 provides.

---

## 2026-08-30 — The headline is a bracket: selection closes 42% of the A-to-B gap, and Route A breaks under its own band
- **Decision:** Phase M reports the ceiling and achieved fraction under B′ and A side by side, B′ first, as a bracket, not a point. Under B′ the within-stand ceiling is 0.2965 and the model reaches 49.3%; under A it is 0.1671 and reaches 87.5%. Route B (0.3556, 41.1%) is the provenance row.
- **Alternatives:** Quote B′ alone, now non-degenerate (rejected: the route rule requires A beside it, and A's band matters to the reader). Quote A's 88% (rejected: the flattering end of a bracket whose other end is 49%, and A is the fragile end).
- **Rationale:** Two results make the bracket the only honest form. The B→B′ diagnostic — restricting the C.2 fit to hitters reaching the 2024 eval population — drops τ² 33% and closes 41.7% of the B-to-A gap: part of the disagreement is population, the rest window or estimator. Route A's ±3% fragility band breaks rather than widens the estimate: at ×0.97 the ceiling rises to 0.2389, at ×1.03 τ² goes negative and A returns no ceiling. B′ conditions on survival to 2024, labeled as such — right for a claim about 2024 eval hitters, not a general-population τ².
- **Reference:** `results/phase_m/m0_routes.json` `b_to_b_prime_diagnostic`, `fragility_band`; `results/phase_m/m1_fraction_of_ceiling.csv`; `tests/test_m_report.py` — `test_route_b_prime_is_not_degenerate_and_sits_between_a_and_b`, `test_the_fragility_band_is_wide_enough_to_matter`.
- **Revisit if:** a τ² estimate becomes available on a population not conditioned on survival to 2024, letting the bracket close from the B′ end.

---

## 2026-08-30 — Per-stratum fractions compare within-stand quantities, and the M.2 precision clause is conditional
- **Decision:** M.2's achieved correlation is the within-stand rank correlation, via E.15's residualisation; fraction of ceiling is formed from it. The raw correlation is emitted alongside, labeled not comparable to a within-stand ceiling. The precision clause applies as written: the narrowest-relative-CI tie-break fires only if the default stratum's own 95% CI includes zero. No stratum's CI includes zero on the real table, so the illustration stays on `low`.
- **Alternatives:** Compare against the raw per-stratum correlation (rejected: τ² is a within-stand variance, so the raw figure — mostly the between-stand main effect per E.5's decomposition — produces fractions above 1, e.g. 1.57 in `low`, the apples-to-oranges error that decomposition prevents). Take the narrowest relative CI unconditionally (rejected: that reads the clause as a standing preference for precision, moving the illustration off the graded stratum for a 1.6-point difference in relative CI width — the clause handles an unusable interval, not a ranking of usable ones).
- **Rationale:** τ² is not refit per stratum — a new, unauthorised estimator — so the common B′ fit runs against each stratum's own sampling-variance profile, as E.15 applies a common fit across stands. The ceiling rises monotonically with exposure (low 0.2531, medium 0.3113, high 0.3248) while the fraction reached falls (75%, 46%, 34%): closest to the ceiling where the ceiling is lowest. Route A per stratum is degenerate in `medium` outright and in 7.5–20% of bootstrap draws elsewhere, so its intervals are reported conditional on non-degeneracy, degenerate share beside them, no draw clipped to zero.
- **Reference:** `results/phase_m/m2_stratum_ceiling.csv`; `src/analysis/m_report.py` `precision_clause`; `tests/test_m_report.py` — `test_the_precision_clause_moves_off_a_stratum_whose_ci_includes_zero`, `test_the_precision_clause_stays_on_low_when_low_is_usable`.
- **Revisit if:** a re-run's bootstrap puts the low stratum's CI across zero; the clause fires and the illustration moves to the narrowest-relative-CI stratum, the move stated in the artifact.

---

## 2026-08-30 — The stabilization threshold is reported under both conventions
- **Decision:** PA* is emitted twice: `pa_star_weak_side` (PAs vs LHP only) and `pa_star_both_sides` (both sides growing together). Weak-side confirms the handoff's unverified figures — 433 PA under Route B ("~430"), 2,184 under A (">2000"), B′ 651. Both-sides figures run roughly double throughout (877 / 4,419 / 1,317).
- **Alternatives:** Report weak-side alone (rejected: it is the convention `implied_split_constant` uses and the one the handoff quoted, but a reader comparing PA* to a season assumes both-side accumulation and would read the threshold as half its value). Report both-sides alone (rejected: breaks continuity with `n_star_split_implied` in the committed C.2 artifacts).
- **Rationale:** The two conventions differ by about 2×, large enough to change what "not enough exposure" means. Against a 2024 median of 86 PA vs LHP, every route says the differential is unmeasurable at realistic exposures either way — 5× the median at best, 25× at worst.
- **Reference:** `results/phase_m/m0_routes.json` `stabilization`; `src/analysis/m_report.py` `stabilization_table`; `results/phase_c/c2_prior_parameters.csv` `n_star_split_implied`.
- **Revisit if:** a claim is made about exposure needed in practice, needing the both-sides convention and joint growth assumption stated explicitly.

---

## 2026-08-31 — Batch size and weight decay stay quarantined; the knob arm does not run
- **Decision:** Phase O is not reopened; the §8 knob arm is not run at Phase M. Both knobs stay frozen at 8,192 and 1e-2 for the rest of the project.
- **Alternatives:** Run the arm at Phase M under plan §3's conditional admission. Spend a third overnight session on Phase O against the two-session cap.
- **Rationale:** The knobs set one shrinkage: the ratio of decay applied every step to a gradient received only when a hitter is sampled — 24:1 at the tenth exposure percentile. Tuning them implements C.2's shrinkage inside the network, then scores the result against C.2. Plan §3 admits the arm only with strength fixed before any claim-1 number is read; M.1 and M.3 have now read claim 1. D.5's gradient (b) independently shows decay overwhelmed, not binding, so expected gain is near zero either way.
- **Reference:** plan §2.2 and §3 (Phase O); D.5 gradient (b); `results/phase_m/m1_fraction_of_ceiling.csv`, `results/phase_m/m3_level_ceiling.json`.
- **Revisit if:** a later phase needs the arm against a target claim 1 does not score, where the pre-registration condition is not already spent.

---

## 2026-08-31 — Phase O and Phase M are reviewed before the 2025 refit
- **Decision:** `/research-review` runs on Phase O and on Phase M first. The 2015–2024 refit and 2025 claim-1 table follow the review, not the other way round.
- **Alternatives:** Refit first and review the whole arc in one pass, saving a review cycle against the Oct 1 abstract date.
- **Rationale:** 2025 is sealed and readable once, so a defect found after the refit cannot be corrected without spending the season again. Neither phase has been reviewed — Phase O's sweep and selector run were never covered, and Phase M is new — and both carry the write-up.
- **Revisit if:** both reviews return no findings, after which later phases can review at the write-up boundary instead of at each phase close.

---

## 2026-08-31 — Ceiling fractions are reported on RMSE and on rank, each against its own denominator
- **Decision:** Phase M reports the fraction of ceiling twice: PA-weighted RMSE against the `claim1_eval` noise floor, and rank correlation against the Monte Carlo Spearman ceiling. The analytic Pearson bound stays the reported closed form but no longer divides a rank numerator.
- **Alternatives:** Keep rank only (rejected: manifest rule 2 defines claim 1 as RMSE and rank; rank-only was never argued). Keep the analytic denominator (rejected: exact for Pearson, but the simulated Spearman ceiling is 0.30931 against 0.29645).
- **Rationale:** The differential claim is a prediction claim, so error is primary and ordering secondary; `noise_floor` and `deconvolve` already compute the RMSE bound, and the Phase M spec derives τ² from them. A Spearman numerator over a Pearson denominator overstates the fraction by about 4%.
- **Revisit if:** the claim narrows to relative valuation, where ordering is the estimand and RMSE secondary.

---

## 2026-08-31 — The differential tables carry every Phase C opponent and an interval in every cell
- **Decision:** C.3-full is refit and scored on the M.1 frame; M.2 gains C.2 and C.3-full columns; every achieved value in M.1 and M.2 carries a paired hitter-level bootstrap interval. One table replaces `m1_differential_scores.csv` and `m2_stratum_ceiling.csv`.
- **Alternatives:** Let §10 fallback rule 1 stand (rejected: it covers baselines unreachable without retraining, and `src/analysis/c_report.py:364` is a supported path). Report the stratum table without an opponent (rejected: frozen rule 1 grades the low stratum against baselines, not a ceiling alone).
- **Rationale:** A bare model number beside a ceiling is the defect M.1 fixed at the pooled level; M.2 carries the fix at the stratum level, where the thesis is graded. C.3-full beats the model on the level side, so omitting it from the only differential table reads as curation. The pooled fraction's interval is wide enough that a point estimate misstates the result.
- **Revisit if:** the C.3-full refit cannot reproduce its committed Phase C parameters, making its absence a capability limit where rule 1 applies as written.

---

## 2026-08-31 — Warmup is measured at the literature default before the 2025 refit
- **Decision:** One arm runs at lr 1e-3, warmup 2,000 steps, seeds 0–4, judged on `reference` under the k=2 promotion rule. The 2026-08-25 rejection of a wider warmup grid is reopened.
- **Alternatives:** Close the limitation in prose (rejected: the grid was `{0, 719}`, and 719 was one epoch, never a value chosen against β₂). Run two seeds under the existing arm size (rejected: the existing two-seed spread of 1.27e-4 already exceeds the margin it measures).
- **Rationale:** 719 steps is 0.7 of the second-moment memory length, so full learning rate arrives before the AdamW denominator converges; the default is two memory lengths. O.2's null rules warmup out as the lever on the exposure gradient — a different question from whether the build is undertrained, the question Phase O scores. Selection is on `reference`, and `o1_select` does not import `claim1_eval`, so the pre-registered procedure stays intact.
- **Reference:** Ma & Yarats (2021), arXiv:1910.04209 (AAAI) — linear warmup over 2/(1−β₂) iterations, 2,000 at β₂ = 0.999.
- **Revisit if:** the arm clears 2 SE, promoting it and requiring the Phase M measurement items to rerun on the promoted build.

---

## 2026-08-31 — The 2024 eval population is unchanged and the PA floor is reported as a sensitivity
- **Decision:** The M.6 intersection stays the primary population and `min_eval_pa` stays at 10. `min_eval_pa_sensitivity` runs at 5, 10, and 25, reported alongside the fractions.
- **Alternatives:** Score the differential on the F.5 population (rejected: its 72 extra hitters lack enough PA against one hand for a split to be defined, not merely noisy). Lower the floor to reach the part-time tail (rejected: below ten PA the observed value is nearly all sampling noise, entering the estimate at near-zero weight).
- **Rationale:** M.3 already reports both populations and they agree to 0.2%, so the choice is immaterial where available at all. The floor removes 32 hitters whose mean wOBA is far below the retained, and every other filter cuts the same direction, so the exclusion is stated as a measured sensitivity, not left as a default.
- **Revisit if:** the sensitivity moves a reported fraction by more than its bootstrap interval.
</content>

---
