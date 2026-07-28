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
- **Decision:** Phase C.1 baseline uses a 3-season trailing window, built in two variants, both reported: `raw` (observed trailing side-specific wOBA unshrunk, league average only where the hitter has zero prior PA vs that hand) and `bucketed` (blend trailing wOBA toward league average by a fixed weight per PA bucket: <50 PA → 0.0, 50–199 → 0.5, 200+ → 1.0). Implemented in `src/analysis/c1_trailing.py` (10 gates).
- **Alternatives:** All-prior-seasons window (rejected on measurement — see Rationale). A single shrunk variant (rejected: collapses C.1 into a crude duplicate of C.2, destroying the decomposition). A single unshrunk variant (rejected alone: catastrophic in low exposure, and beating it proves little).
- **Rationale:** Window chosen by measurement, not preference — on the 2024 eval frame a 3-season window costs the low-exposure stratum almost nothing (median prior PA 10.5 → 10.0), because low-exposure hitters' whole careers already sit inside the window; it only trims veterans (median 963 → 590 PA), who have ample sample regardless. Both variants together decompose a quantity a single variant hides: raw→bucketed measures the value of shrinkage at all, bucketed→C.2 the value of doing it properly, C.2→Phase D the value of cross-hitter structure — the thesis.
- **Revisit if:** the raw variant's low-exposure rank-correlation advantage turns out to be an artifact of eval-season target noise rather than real ordering signal (see the noise-floor decomposition, same date).

---

## 2026-07-27 — Noise-floor deconvolution added as a companion to the claim-1 metric
- **Decision:** `src/analysis/claim1_eval.py` now reports, alongside the frozen §5.2 PA-weighted RMSE, two companion columns per stratum: `noise_floor` (irreducible RMSE from sampling noise in the held-out target) and `model_rmse` (`sqrt(RMSE² − floor²)`, clamped at zero). A `skill_score(model, reference)` helper expresses a model as `1 − (model/reference)²`, the share of between-hitter talent variance captured. The frozen §5.2 metric itself is unchanged; these are additive columns.
- **Alternatives:** Raw RMSE alone as §5.2 specifies (rejected: see Rationale). Replacing RMSE with a skill score (rejected: §5.2 is frozen, and the raw number must stay comparable to prior work). Estimating the floor by simulation (rejected: the analytic within-group variance is exact under the same independence assumption and costs nothing).
- **Rationale:** The held-out target is itself a small-sample measurement, so errors add in quadrature: `observed RMSE² = model error² + target noise²`. Measured on the 2024 eval frame, that noise is 60–70% of MSE in every stratum, so raw RMSE compresses real differences into what reads as rounding — halving true error only moves RMSE 0.0600 → 0.0507. It also charges the model for unpredictable luck (a 31-PA hitter predicted at league average who "actually" hit .5737). Independent validation: the deconvolved no-info-baseline error (0.0368) matches B.1's variance-components between-hitter signal SD (√0.00137 = 0.0370) to three digits, as theory requires.
- **Revisit if:** a model's raw RMSE lands materially below its estimated floor in any stratum — that would indicate the floor is biased high rather than the model beating perfect.

---

## 2026-07-27 — Pitchers' own at-bats excluded from every hitter-talent quantity
- **Decision:** `src/data/eval_targets.py` gains `primarily_pitchers` / `drop_pitcher_batters`: a batter is excluded, per season, when he faced ≥50 batters AND took <50 PA (both conditions required so two-way players like Ohtani remain hitters). Applied to every quantity describing hitter talent — B.1 stabilization, the C.1 league average, and `claim1_eval` scoring. Deliberately NOT applied to the evaluation target itself, which stays built from the complete unfiltered source per the two-table principle (2026-07-15).
- **Alternatives:** Filtering in `clean.py` (rejected: that produces the modeling table, and the affected quantities are built from the eval-target table, which bypasses it by design). A career-level rule (rejected: role changes season to season). A PA-only threshold (rejected: would drop genuine September call-ups, the low-exposure population the thesis targets).
- **Rationale:** The eval-target table is built from the complete source on purpose, so it contained NL pitchers batting before the 2022 universal DH — 1,223 of 2,487 distinct train-season batters, median 12 PA, mean wOBA .153 — correct for ground truth but wrong for any prior, stabilization point, or league average, where a ~.15-wOBA non-hitter population inflates between-hitter signal variance. Found via a bin-wise diagnostic where observed variance fell below its own claimed sampling floor, which is impossible unless the noise model was inflated (pooled within-PA variance 0.261 vs a pitcher's 0.137). Validated against actual DH rule history (2015–19/2021 drop 267–328 batters, 2020 drops 3, 2022+ drop 2–6) and retains Ohtani in every season including his two-way years. This correction supersedes prior numbers, methods unchanged: side-specific wOBA n* moves 190/198 → 226/254 (widening the B.1 process-beats-outcome gap to ~7.6x whiff / ~2.0x spray, from ~7x/~1.6x); C.1 strata boundaries move (95, 380) → (113, 452); noise-floor low-stratum values move 0.0472 → 0.0465 and 0.0600 → 0.0591.
- **Revisit if:** a future season reintroduces pitchers batting in volume, or the 50-PA two-way threshold misclassifies a genuine two-way player.

---

## 2026-07-27 — C.2 estimand: shrink the two sides jointly, not the platoon split
- **Decision:** C.2 is bivariate empirical Bayes over (talent vs LHP, talent vs RHP) per batter type L/R/switch: `θ̂ = μ + Σ(Σ+V)⁻¹(x−μ)`. μ, τ², σ² come from a one-way random-effects ANOVA per (type, hand), with σ² cell-specific and never pooled; the off-diagonal comes from the cross-side sample covariance, estimated only on hitters with ≥50 PA vs both hands (`RHO_MIN_PA`); the diagonal keeps the full population. `src/analysis/c2_bivariate_eb.py`, 21 gates.
- **Alternatives:** Split-level (The Book) rejected as the estimator, retained as a scored reference row — Var(split) requires subtracting a noise term ~12x its own size, and a 9% error there is what moved the reverted attempt 16,419 → 4,390. Rate-level (ρ=0) rejected as indefensible, retained as the nesting gate. Unrestricted covariance rejected on measurement: ρ = 4.74 with no threshold, 1.09 at 25 PA, stable 0.85–1.10 from 50 up.
- **Rationale:** The two parameterizations are the same model in rotated coordinates, so the choice is purely identifiability. A hitter's PA vs LHP and vs RHP are disjoint events, so `Cov(x_L, x_R) = Cov(θ_L, θ_R)` exactly — no noise term to subtract — and split variance is derived from three separately identified estimates instead of one catastrophic cancellation. Restricting a correlation to the durable subpopulation is a milder assumption than restricting a variance, so only the off-diagonal is cut.
- **Reference:** Efron & Morris (1972), *Biometrika* 59(2):335–347 (vector-observation EB); the intercept-only case of the multivariate Fay–Herriot EBLUP (Permatasari & Ubaidillah 2021, *R Journal* §2.1). Brown (2008), *AoAS* 2(1):113–152 for heteroscedastic EB and exposure–talent correlation (measured here at +0.24/+0.26). The Book p. 157 constants via Tango (insidethebook.com, 2009) — constants sourced, functional form not verifiable online; standing gap, unchanged.
- **Revisit if:** the ρ interval tightens enough to separate our estimate from The Book's implied value, or a future window makes the ≥50 PA cut drop a materially different population.
