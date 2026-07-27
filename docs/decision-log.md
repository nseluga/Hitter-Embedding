# Decision Log — Hitter Embedding

Append-only. Format fixed by `~/os/knowledge/frameworks/research-standards.md`
§4 — `/research-partner` and `/research-review` both parse against it.

---

## <YYYY-MM-DD> — <decision title>
- **Decision:** what was chosen
- **Alternatives:** what was considered and rejected
- **Rationale:** why, in terms a skeptical reviewer would accept
- **Revisit if:** the condition under which this should be reopened

---

## 2026-07-14 — Statcast raw-snapshot storage design
- **Decision:** Raw Statcast frozen to `data/raw/statcast/snapshot_<date>/season=<YYYY>.parquet` — per-season, all columns, immutable; committed `manifest.json` holds pull metadata. Data gitignored; script + manifest committed.
- **Alternatives:** Monolithic parquet (forces full-memory loads); column pruning at pull time (unrecoverable without re-pull).
- **Rationale:** Date-stamped snapshots handle Statcast's retroactive revisions; immutable raw lets the pitch-event table re-derive without re-fetching.
- **Revisit if:** snapshot outgrows local disk.

---

## 2026-07-15 — Statcast cleaning spec for the modeling pitch table
- **Decision:** Modeling table is regular season only. It removes position-player pitches, pitchouts, automatic balls and strikes, and bunts, and keeps intentional balls. It drops rows missing or physically impossible on core context of type, velocity, location, and movement, and keeps optional spin context with missingness indicators. It drops the 8 deprecated columns, dedupes on pitch key, and sorts by game_date then pitch key. Filters apply only to the modeling table, never to evaluation targets.
- **Alternatives:** A minimum-PA hitter floor was rejected because it deletes the low-exposure population the thesis targets. A velocity-based position-player rule was rejected because it misclassifies hard-armed position players. Dropping spin columns was deferred to a Phase B ablation.
- **Rationale:** Every filter is backed by the profiling notebook, and the modeling-versus-target split keeps sharpening filters from biasing ground truth.
- **Revisit if:** Phase B feature screening changes the retained context set, or target construction needs a filtered field.

---

## 2026-07-17 — Contact-quality label domain
- **Decision:** The contact-quality head (EV, LA, spray) is labeled on balls in play only (`hit_into_play`); `ev`/`la`/`spray` are null on all other pitches.
- **Alternatives:** Labeling all contact-with-EV was rejected because fouls carry EV/LA ~70% of the time but have no spray and no batted-ball outcome, giving ragged masking and no run-value mapping.
- **Rationale:** Fouls are contact for the whiff head but are non-terminal count transitions in the Markov composition; only in-play balls have the run-value-bearing outcome the quality head feeds.
- **Revisit if:** the outcome space or run-value mapping (§1.5) changes to need foul-ball measurements.

---

## 2026-07-17 — Spray angle derivation
- **Decision:** Spray = `atan((hc_x − 125.42)/(198.27 − hc_y))` in degrees, then mirrored so positive = pull for both hands (flip sign for RHB).
- **Alternatives:** Empirically calibrating the home-plate origin was dropped once the constants were sourced; raw field-side angle (no mirroring) was rejected as not batter-intrinsic.
- **Rationale:** Formula and constants corroborated by three sources (abdwr3e App. C, BGSU, Weise); MLB does not publish the origin, so a real-data regression-guard (field mean ≈ 0, pull-mean > 0) confirms our export matches the scale.
- **Revisit if:** the near-plate artifact (`|spray| > 90°`, ~1% of in-play) needs clipping, decided in Phase B.

---

## 2026-07-17 — Walk-forward split frozen
- **Decision:** Contiguous walk-forward, train 2015-2023 / val 2024 / test 2025, single fold, frozen in `src/config/split_config.json` and validated on load.
- **Alternatives:** Random k-fold rejected (leaks a hitter's future PAs into his own ID embedding, memorizing the outcome we claim to project and manufacturing a positive result). Rolling multi-fold deferred (multiplies compute against the <$200 budget with no gain on the exposure axis we grade on). A val/test gap season rejected (wastes data, breaks Phase B's same-season window trade).
- **Rationale:** Projection is a forecasting task, so eval must mirror deployment: train on the past, test on a strictly-later season. Freezing before any model comparison keeps the held-out season from becoming a shopped hyperparameter (pre-registration). Contiguity minimizes distribution shift so the metric measures projection skill, not regime drift. Robustness comes from exposure stratification + dual-sampler, not temporal folds.
- **Revisit if:** never for this project (frozen rule); a new fold requires a new entry naming this one.

---

## 2026-07-21 — Stabilization reported at two thresholds (r=0.5 and r=0.7)
- **Decision:** Every stabilization point is reported at both r=0.5 and r=0.7, not a single threshold. r=0.5 (signal variance equals noise variance) is framed as the equal-weight-with-prior point — the regression-to-the-mean ballast, and the small-sample projection currency the thesis trades in. r=0.7 is the stricter "reliable measurement" convention.
- **Alternatives:** Single r=0.5 point (rejected: invites an unanswered "why 0.5?" and hides that 0.5 still means half the variance is noise). Single r=0.7 (rejected: not the quantity the shrinkage/projection argument uses).
- **Rationale:** The two thresholds answer different questions; reporting both preempts the threshold objection and lets us cross-check against Carleton's published r=0.7 numbers.
- **Reference:** Carleton, "Reliably Stable (You Keep Using That Word)" (Baseball Prospectus); FanGraphs "A Long-Needed Update on Reliability."
- **Tier:** 2.
- **Revisit if:** the paper's reviewers want a different reliability convention.

---

## 2026-07-21 — Variance-components estimator added alongside split-half
- **Decision:** Added a one-way random-effects (variance-components / Cronbach-alpha) estimator to src/analysis/stabilization.py: one signal/noise decomposition over all hitters yields an analytic reliability(n), an analytic stabilization point, and a bootstrap CI. Split-half is kept as an independent cross-check; the two must agree on the closed-form synthetic (they do). The variance-components point is the headline where the two diverge, because split-half at large n only uses hitters with >= n observations (survivorship).
- **Alternatives:** Split-half only (rejected: survivorship-biased at large n — precisely where the wOBA outcome lives — and gives no confidence interval). Mixed-model REML (rejected for now: heavier, and the method-of-moments ANOVA matches Cronbach's alpha at a fraction of the code).
- **Rationale:** On the real table the two estimators agree on the fast process metrics but diverge ~2x on wOBA (VC 190 PA vs split-half 435 PA vs LHP) — a survivorship artifact the variance-components method removes by using all 2142 hitters, not just the durable ones.
- **Reference:** FanGraphs "A New Way to Look at Sample Size (Math Supplement)" — Cronbach-alpha signal/noise decomposition; KR-21 for the binary heads.
- **Tier:** 2. Changed a tested module → re-ran the closed-form synthetic gate (test_stabilization.py, 14 tests) before trusting the numbers, per the ml-engineer gate.
- **Revisit if:** the VC/split-half divergence on wOBA turns out to be heteroscedasticity in the VC assumptions rather than survivorship in split-half (to be disentangled in notebooks/02).

---

## 2026-07-21 — Matched-slice and across-time reporting for the process-vs-outcome comparison
- **Decision:** The B.1 comparison additionally reports (a) process metrics sliced by pitcher hand (whiff/EV vs LHP and vs RHP), so process is measured on the same side-specific slice as the outcome; and (b) a sequential (chronological early-half vs late-half) split alongside the random split, so absolute points reflect across-circumstance reliability, not just within-sample consistency.
- **Alternatives:** Pooled process vs side-specific outcome only (rejected: apples-to-oranges — some of the gap is the split, not the process/outcome distinction). Random split only (rejected: measures within-sample consistency, which flatters the projection-relevant number).
- **Rationale:** Matched slicing kills the comparison-asymmetry objection — process stays fast even split by hand (whiff ~45–50 swings per side vs pooled 51). The process-beats-outcome gap survives every slicing and split choice; only the absolute points move, so the B.1 headline is robust.
- **Reference:** Carleton, "Reliably Stable" — sequential/different-circumstance splits drop reliability vs same-circumstance splits.
- **Tier:** 2.
- **Revisit if:** the sequential split shows a large systematic across-time degradation on the headline metrics (modest in the 2026-07-21 run).

---

## 2026-07-27 — C.1 trailing-average design: 3-season window, both shrinkage variants
- **Decision:** The Phase C.1 baseline uses a **3-season trailing window** and is built in **two variants, both reported**: `raw` (predict the hitter's observed trailing side-specific wOBA unshrunk; side-specific league average only where he has zero prior PA vs that hand) and `bucketed` (blend his trailing wOBA toward the side-specific league average by a fixed weight per PA bucket: <50 PA → 0.0, 50–199 → 0.5, 200+ → 1.0). Implemented in `src/analysis/c1_trailing.py` (10 gates in `tests/test_c1_trailing.py`).
- **Alternatives:** All-prior-seasons window (rejected on measurement, below). Last-3 only as convention (accepted, but on evidence rather than convention). A single shrunk variant (rejected: it collapses C.1 into a crude duplicate of C.2 and destroys the decomposition). A single unshrunk variant (rejected alone: catastrophic in low exposure, and beating it proves little — a reviewer calls it a strawman).
- **Rationale:** *Window* — decided by measurement, not preference. On the 2024 eval frame a 3-season window costs the low-exposure stratum essentially nothing (median prior PA 10.5 → 10.0; the same 117 groups sit at zero either way), because low-exposure hitters are young or newly arrived and their whole career already falls inside the window. It only trims veterans, who have ample sample regardless (median 963 → 590 PA). So the recency/aging benefit is free in exactly the stratum the thesis is graded on. *Both variants* — the pair decomposes a quantity a single variant hides: raw→bucketed measures the value of shrinkage at all; bucketed→C.2 measures the value of doing shrinkage properly; C.2→Phase D measures the value of cross-hitter structure, which is the thesis. The marginal cost is one parameter and one scoring run.
- **Reference:** none — judgment call on the decomposition; the window is settled by our own measurement on the 2024 eval frame, not by literature.
- **Tier:** 3 (user decision, both points confirmed by Nate).
- **Revisit if:** the raw variant's low-exposure rank-correlation advantage over bucketed turns out to be an artifact of the eval-season target noise rather than real ordering signal (see the noise-floor decomposition, same date).

---

## 2026-07-27 — Noise-floor deconvolution added as a companion to the claim-1 metric
- **Decision:** `src/analysis/claim1_eval.py` now reports, alongside the frozen §5.2 PA-weighted RMSE, two companion columns per stratum: `noise_floor` (the irreducible RMSE contributed by sampling noise in the held-out target) and `model_rmse` (`sqrt(RMSE² − floor²)`, clamped at zero). A `skill_score(model, reference)` helper expresses a model as `1 − (model/reference)²` — the share of between-hitter talent variance it captures, where the no-information baseline is 0 and perfect is 1. **The frozen metric itself is unchanged**; these are additive columns for legibility, not a replacement.
- **Alternatives:** Reporting raw PA-weighted RMSE alone as §5.2 specifies (rejected: see rationale). Replacing RMSE with a skill score (rejected: §5.2 is frozen, and the raw number must stay comparable to anything already computed). Estimating the floor by simulation (rejected: the analytic within-group variance is exact under the same independence assumption and costs nothing).
- **Rationale:** The held-out "actual" side-specific wOBA is itself a small-sample measurement, so errors add in quadrature: `observed RMSE² = model error² + target noise²`. Measured on the 2024 eval frame, that noise is **60–70% of the mean squared error in every stratum** (low: floor 0.0472 of an observed 0.0600). The consequence is that raw RMSE compresses real differences into what reads as rounding — a model cutting true error in half moves RMSE only 0.0600 → 0.0507, and a genuinely strong result would look like a null. It also charges the model for luck it could not have predicted: batter `677956` was predicted at league average (.3131) and "actually" hit .5737 in 31 PA, which is noise, not a miss. Independent validation: the deconvolved error of the no-information baseline came out at 0.0368, and B.1's variance-components estimator independently put the between-hitter signal SD for side-specific wOBA at √0.00137 = 0.0370 — different code, different quantity, agreement to three digits, exactly as theory requires (predicting league average for everyone must yield an error equal to the full talent spread).
- **Reference:** the same signal/noise decomposition as B.1's variance-components estimator (`src/analysis/stabilization.py`, decision log 2026-07-21), applied to the evaluation target rather than the predictor.
- **Tier:** 2. Assumption stated: model error and target sampling noise are independent, which holds because no model reads the held-out season. The floor is an estimate and is labeled as one wherever reported. It is per stratum, not global (low 0.0472 at ~103 mean eval PA; high 0.0332 at ~232).
- **Revisit if:** a model's raw RMSE lands materially below its estimated floor in any stratum — that would indicate the floor is biased high (e.g. wOBA points are not conditionally independent within a hitter-season) rather than that the model is better than perfect.

---

## 2026-07-27 — Pitchers' own at-bats excluded from every hitter-talent quantity
- **Decision:** `src/data/eval_targets.py` gains `primarily_pitchers` / `drop_pitcher_batters`: a batter is excluded, per season, when he faced >= 50 batters AND took < 50 PA. Both conditions are required so two-way players (Ohtani) remain hitters. Applied to every quantity describing HITTER talent — B.1 stabilization, the C.1 league average, and `claim1_eval` scoring. **Deliberately NOT applied to the evaluation target itself**, which stays built from the complete unfiltered source per the two-table principle (2026-07-15).
- **Alternatives:** Filtering in `clean.py` (rejected: `clean.py` produces the modeling table, and the affected quantities are all built from the eval-target table, which by design bypasses it). A career-level rather than per-season rule (rejected: role changes season to season). A PA-only threshold (rejected: would drop genuine September call-ups, who ARE the low-exposure population the thesis targets).
- **Rationale:** The eval-target table is built from the complete source on purpose, so it contains NL pitchers batting before the 2022 universal DH — **1,223 of 2,487 distinct train-season batters, median 12 PA, mean wOBA .153**. That is correct for ground truth but wrong for any prior, stabilization point, or league average, where a population of ~.15-wOBA non-hitters inflates between-hitter signal variance. The bug was found by a bin-wise diagnostic in which observed variance fell *below* its own claimed sampling floor — impossible unless the noise model was inflated, which it was: pooled within-PA variance is 0.261, but a pitcher's is 0.137. Filter validated two ways: it tracks actual DH rule history (2015–19 and 2021 drop 267–328 batters, 2020 drops 3, 2022+ drop 2–6) and retains Ohtani in every season including his two-way years.
- **Supersedes (numbers only, methods unchanged):**
  - **2026-07-21 "Variance-components estimator added alongside split-half"** and the 2026-07-21 lab-notebook entry: side-specific wOBA n* was reported as **190 PA vs LHP / 198 vs RHP**. Hitters-only values are **226 [198, 265] and 254 [223, 287]**. The process metrics barely move (whiff 29.7 PA-equiv, swing 31.6, EV 66.8, LA 68.5, spray 114.5), so the process-beats-outcome gap WIDENS: whiff ~7.6x and spray ~2.0x, against the previously reported ~7x and ~1.6x. The B.1 headline is strengthened, not weakened, by the correction.
  - **2026-07-27 "C.1 trailing-average design"**: strata boundaries move from (95, 380) to **(113, 452)** with `STABILIZATION_N_STAR` 190 -> 226. C.1's conclusions are unchanged; the high-exposure skill score moves 36% -> 39%.
  - **2026-07-27 "Noise-floor deconvolution"**: low-stratum floor 0.0472 -> **0.0465**, no-info reference RMSE 0.0600 -> **0.0591**.
- **Reference:** mirrors the position-player-pitcher rule in `src/data/clean.py` (2026-07-15 cleaning spec), in the batting direction, with the same thresholds.
- **Tier:** 1 — a contaminated prior population is an evaluation-validity defect, not a preference. 4 gates added in `tests/test_eval_targets.py`.
- **Revisit if:** a future season reintroduces pitchers batting in volume, or the 50-PA two-way threshold misclassifies a genuine two-way player.

---

## 2026-07-27 — C.2 estimand: shrink the two sides jointly, not the platoon split
- **Decision:** C.2 is bivariate empirical Bayes over (talent vs LHP, talent vs RHP) per batter type L/R/switch: `θ̂ = μ + Σ(Σ+V)⁻¹(x−μ)`. μ, τ², σ² come from a one-way random-effects ANOVA per (type, hand), with σ² **cell-specific and never pooled**; the off-diagonal comes from the cross-side sample covariance. The covariance **alone** is estimated on hitters with ≥50 PA vs both hands (`RHO_MIN_PA`); the diagonal keeps the full population. `src/analysis/c2_bivariate_eb.py`, 21 gates.
- **Alternatives:** Split-level (The Book) rejected as the *estimator*, retained as a scored reference row — Var(split) requires subtracting a noise term ~12x its own size, and a 9% error there is what moved the reverted attempt 16,419 → 4,390. Rate-level (ρ=0) rejected as indefensible (it asserts 600 PA vs RHP say nothing about lefty talent), retained as the nesting gate. Unrestricted covariance rejected on measurement: ρ = 4.74 with no threshold, 1.09 at 25 PA, stable 0.85–1.10 from 50 up.
- **Rationale:** The two parameterizations are the same model in rotated coordinates, so the choice is purely identifiability. A hitter's PA vs LHP and vs RHP are disjoint events, so `Cov(x_L, x_R) = Cov(θ_L, θ_R)` exactly — no noise term to subtract — and the split variance is *derived* from three separately identified estimates instead of one catastrophic cancellation. Restricting a correlation to the durable subpopulation is a far milder assumption than restricting a variance, which is why only the off-diagonal is cut (B.1's survivorship finding governs the diagonal).
- **Reference:** Efron & Morris (1972), *Biometrika* 59(2):335–347 (vector-observation EB); the intercept-only case of the multivariate Fay–Herriot EBLUP (Permatasari & Ubaidillah 2021, *R Journal* §2.1). Brown (2008), *AoAS* 2(1):113–152 for heteroscedastic EB and remark 3b (exposure–talent correlation, measured here at +0.24 / +0.26). The Book p. 157 constants via Tango (insidethebook.com, 2009) — **constants sourced, functional form NOT verifiable online**; every public implementation is a disagreeing reconstruction. Standing gap, unchanged.
- **Tier:** 3 (Nate's call: bivariate only). The `RHO_MIN_PA` split is Tier 2, settled by the sensitivity measurement above.
- **Revisit if:** the ρ interval tightens enough to separate our estimate from The Book's implied value, or a future window makes the ≥50 PA cut drop a materially different population.
