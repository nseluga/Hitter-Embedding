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
