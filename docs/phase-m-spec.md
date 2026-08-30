# Phase M Spec — Measurement

Written 2026-08-30 for an Opus build agent. Authority: `Layer1_Architecture_Plan_v2.md` §3
(Phase M), the 2026-08-25 decision-log entries (Phase O close, pre-registered promotion
rule), and the four decisions settled 2026-08-30 (τ² rule, Route C, M.1 scope, knob-arm
timing). Everything marked **pre-registered** in this spec is fixed before any Phase M
number is read; do not revise it after seeing results.

## §0 Ground rules (binding)

1. **Build:** the `d10_baseline_s0..s4` five-seed ensemble, unchanged. Phase O closed
   `incumbent_stands`; the o1 seeds are a screening artifact, never a build.
2. **2025 is sealed.** Nothing in Phase M reads, scores, or refits on 2025. The 2025 run
   happens after Phase M per the plan's strict order.
3. **Labeling obligation** (2026-08-20 entry): every claim-1-style number computed on 2024
   is **post-selection and descriptive**, and every emitted artifact carries that label in
   a top-level field or column.
4. **Promotion rule** (2026-08-25 entry, pre-registered): any comparison that promotes one
   build over another uses the pooled across-seed sd, a Bonferroni-corrected threshold
   over the number of challengers, and a fresh-seed confirmation that discards screening
   seeds. Applies to §8 only; §§1–7 are measurement, not selection.
5. **Frozen rules** in `docs/research-manifest.md` bind. Any decision not covered here or
   in the plan: **stop and ask** (CLAUDE.md blocking-questions rule). §10 lists the
   anticipated blockers.
6. **Order of work:** §9 self-checks → M.6 population → M.0 routes → M.1 → M.2 → M.3 →
   M.4/M.5 (write-up items) → §8 knob arm (after all measurement items).

## M.0 The τ² apparatus and the pre-registered route rule

The ceiling hangs on one contested parameter: within-stand true-talent variance τ².
Committed inputs (`results/phase_e/e15_ceiling.json`, corrected 2026-08-29 provenance):
within-stand observed variance **0.00419563**, mean sampling variance **0.00407848**.

### Routes

- **Route A (2024 empirical):** τ² = observed − sampling = **0.00011715** → reliability
  0.028 → ceiling 0.167 → model at 88%. Recompute from source, do not transcribe.
- **Route B (nine-season C.2 fit, provenance only):** τ² = **0.00059034** → ceiling 0.356
  → model at 41%. Reported as one provenance row; never primary. Before any new fit,
  **reproduce this committed number** from `c2_bivariate_eb.fit` on the unrestricted
  fitting window — a failed reproduction is a §10 blocker, not something to work around.
- **Route B′ (new):** the same `c2_bivariate_eb.fit` call on the same 2016–2024 fitting
  window, with `pa_df` **restricted to the M.6 intersection population** before fitting.
  Extract the *identical field* E.15's Route B read (match the quantity exactly — the
  within-stand τ², not the derived split variance; follow `e_platoon_ceiling.py`'s
  provenance for which field that is, and record the field name in the output). Label:
  restricting past seasons to 2024 eval hitters conditions on survival to 2024 — correct
  for this claim, stated in the artifact.
- **Route C (split-half + Spearman-Brown):** currently returns **−0.366** for LHB —
  impossible for a reliability. Handled by the diagnostic below; never primary.

### Pre-registered route rule

1. **Primary: B′.** Chosen now because Route A's fragility is structural — a near-total
   cancellation where a 3% noise-model error swings τ² ~100% — and that is known before
   any number is read.
2. **Route A is always reported** as the same-season sensitivity, carrying its
   **fragility band**: recompute the A ceiling with the sampling-variance term scaled by
   ×0.97 and ×1.03 and report the resulting ceiling range. This demonstrates the
   fragility; it never picks the route.
3. **Route B is one provenance row.** The B→B′ delta is reported as the population
   diagnostic: a large drop toward A means selection into the 2024 eval population
   explains the gap; no material drop means the remaining A/B′ gap is window/estimator
   and is labeled as such.
4. **The headline table carries the ceiling and fraction-of-ceiling under B′ and A
   side by side**, B′ first. No single-route headline is emitted without the other
   beside it.
5. **Degenerate outcome:** if B′'s τ² comes back negative or the fit degenerates on the
   restricted population, stop — §10 blocker.

### Stabilization threshold (verification item from the 2026-08-29 handoff)

Express each route's τ² as a stabilization threshold: PA\* = per-PA noise variance / τ²
(derive per-PA noise from the same noise model E.15 used, not from a new one). The
handoff's rough numbers — ~430 PA under B, >2000 under A — are **unverified**; confirm or
correct them, compute the B′ value, and report all against the observed exposure
(median ~70 PA vs LHP, recompute from the frame). This is τ² re-expressed in the unit
baseball readers know, not a new estimator.

### Route C diagnostic (bounded: one day, hard cap)

Two candidate causes; the diagnostic decides.

1. **Code audit:** read the split-half implementation once (split definition, pairing,
   no shared PAs across halves, Spearman-Brown applied to the raw r). If a bug is found:
   fix it, re-emit the corrected Route C value as a descriptive third estimate, log the
   bug.
2. **Null simulation:** using each LHB eval hitter's real 2024 PA-vs-LHP count, simulate
   per-half wOBA under (a) τ² = 0 and (b) τ² = B′, with the E.15 noise model. Emit the
   simulated distribution of the split-half estimate under each and locate −0.366 in
   both.
   - Inside the τ²=0 range and plausible under small τ²: report Route C as
     *consistent with small τ²* — an evidence line for the M.0 table (leans Route A),
     never an estimator.
   - Outside both simulated ranges and no bug found: **stop — §10 blocker** (the noise
     model itself is then suspect, which contaminates Route A too).

Output: `results/phase_m/m0_routes.json` (all routes, rule verdict, fragility band,
stabilization thresholds, Route C verdict, field-name provenance, labels).

## M.1 Close the incumbent gap (prerequisite)

`results/phase_e/e5_platoon_scores.csv` carries only `delta_pred` and `delta_route_a` —
the ceiling table has no opponent in it. Add, on the same frame, weights (harmonic-mean
`weight` per E.5), and M.6 population:

- **`delta_c2`:** C.2's platoon differential from `predict()`'s side-specific wOBA
  (already emitted — this is scoring, not new modeling).
- **`delta_c3full`:** C.3-full's platoon differential. If C.3-full cannot emit
  side-specific predictions from its existing fitted artifacts without retraining or new
  feature code, **stop — §10 blocker** (do not improvise a differential head).

Then recompute the paired comparisons and fraction-of-ceiling for model, C.2, and
C.3-full under both B′ and A ceilings. Scope decision (2026-08-30): these two rungs only;
`no_info` is degenerate on a differential (constant prediction), and the C.1 rungs answer
no live question on this cut — the paper states the omission.

Output: `results/phase_m/m1_differential_scores.csv`,
`results/phase_m/m1_fraction_of_ceiling.csv`.

## M.2 Per-exposure-stratum platoon ceiling

E.15 stratified by stand, not exposure — the stratum the project is about has no ceiling.
Apply the M.0 apparatus within each claim-1 exposure stratum (use the existing frozen
stratum definitions from the claim-1 machinery; do not redefine strata):

- Per-stratum observed variance, per-stratum mean sampling variance, per-stratum τ² under
  B′ (refit restricted to the stratum's hitters? **No** — τ² under B′ is fitted once on
  the full intersection; per-stratum ceilings use the stratum's own sampling-variance
  profile against the common τ². A per-stratum refit is a new estimator and is not
  authorized).
- Bootstrap CI (resample hitters within stratum) on each stratum ceiling.
- **Precision clause** (from the 2026-08-20 revisit condition, made numeric 2026-08-30):
  if a stratum's 95% bootstrap CI on the ceiling includes zero, that stratum cannot carry
  the illustration; the illustration moves to the stratum with the narrowest CI relative
  to its point estimate, and the artifact says so explicitly.
- Route A per stratum is reported beside B′ with the same fragility caveat.

Output: `results/phase_m/m2_stratum_ceiling.csv`.

## M.3 The level-side ceiling (plan item O.1)

So "X% of the platoon ceiling" has a comparable level figure beside it. Per the plan:
both reliability terms are committed in `f5_pooled_scores.csv` — `noise_floor` is
E[Var[noise]], and the no-information rung's `model_rmse` is the observed between-hitter
spread (a constant predictor's RMSE is that spread), so the deconvolved true-talent
variance follows by subtraction. Two cross-check routes, both emitted:

1. C.2-derived variance composition (the level-side analogue of Route B′ — restrict to
   the M.6 population).
2. Split-half on `game_pk` parity.

Emit the ceiling (refit-invariant) and the achieved fraction (not refit-invariant) as
**separate fields**, per the plan's wording. The by-stand L/R ceiling fractions
(0.569 / 0.122) are reported descriptively only, carrying the plan's caveat that the
asymmetry's supporting statistic is broken and the difference was never tested.

Output: `results/phase_m/m3_level_ceiling.json`.

## M.4 Coverage labeling (fallback already fired)

E.14's fix — adding target sampling noise to the ensemble interval — is already computed
and still lands at **85.6% vs nominal 95%**. Per the plan's own fallback, the remaining
work is labeling, not iteration: every displayed interval in notebook 08 (and anything
Phase Q later inherits) carries its **measured coverage**, not the nominal level. One
verification pass that the 85.6% is computed on the M.6 population; no further attempts
to close the gap.

Output: coverage label fields added to the M artifacts; no new estimator.

## M.5 Gradient (b) — record the null

The plan's item as written ("re-run gradient (b) on the tuned build") is unsatisfiable:
Phase O closed `incumbent_stands`, so the tuned build it names never existed. The
pre-registered prediction was already tested in O.2 on the paired warm arm and
**contradicted** (global rescaling, exposure gradient intact, n=2 descriptive). M.5
closes by recording exactly that in notebook 08 with references to
`results/phase_o/o2_gradient_b.json` — a reported null, no new computation. The knob arm
(§8) is the live continuation of this question, after Phase M.

## M.6 Population reconciliation (run first)

E.5 scores 545 hitters, F.5 scores 617. Compute the intersection once, emit it as
`results/phase_m/population.csv` (batter id, in_e5, in_f5, in_intersection, PA counts by
side). Every M.0–M.3 artifact computes on the intersection; any figure that does not is
labeled with its own n in the artifact itself. The headline runs on the intersection.

## §8 Knob arm — after Phase M, pre-registered now

Timing decision (2026-08-30): all measurement items complete first; the arm then runs
under the conditions below. This is a mechanism probe, never the paper's build — the
quarantine circularity (tuning batch/decay builds C.2's shrinkage into the network, then
scores it against C.2) bars promotion regardless of outcome.

**Pre-registered arms** (values fixed here, before any M number is read):

1. `wd3e1`: weight decay 1e-2 → **3e-1**, all else d10_baseline. Rationale: the
   10th-percentile decay-to-gradient ratio is 23.9:1 at 1e-2, so decay needs ≥24× to
   bind; 30× makes it bind with margin.
2. `bs2048`: batch 8,192 → **2,048**, all else d10_baseline. Rationale: 4× more updates
   at fixed lr separates update-count effects from decay effects.

Two seeds each (0, 1), matched pairs against `d10_baseline_s0,s1` via `torch.manual_seed`
as in O.2. Judged on: (a) gradient (b) — does the rarest quintile's norm fall toward the
most-exposed quintile's, and does the exposure-normalized slope move; (b) `reference`
under the §0.4 promotion rule with k=2 challengers — for the record only, since promotion
is barred. **F.4's identity share is re-measured on any arm that moves gradient (b)**
(the quarantine entry's check). Ledger rows carry `data_dir` and canonical lr per the
2026-08-20 provenance entries. Config-only change per CLAUDE.md — no gate re-run — but
the §9 reproduction check applies (each arm's seed-0 epoch-0 losses logged and compared
against d10_baseline's to confirm the build).

**Prohibitions:** no M.0–M.3 number is recomputed on a knob build; the arm never enters
the ceiling or fraction tables; a "win" on `reference` changes nothing about the paper's
model and is reported as mechanism evidence only.

## §9 Verification (blocking, before any committed number)

New analysis code gets planted-recovery self-checks, in `tests/`:

1. **Planted τ²:** simulate hitters with known τ² and the real per-hitter PA
   distribution; assert Route A's subtraction recovers τ² within tolerance, and the
   ceiling formula matches the Monte-Carlo best-possible rank correlation of true skill
   against simulated observed wOBA (this validates the reliability→ceiling map, the load-
   bearing step).
2. **Planted-zero:** under τ²=0 the pipeline must report reliability ≈ 0 and must not
   emit a negative τ² silently — negative estimates are emitted as negative with a
   `degenerate` flag, never clipped.
3. **Reproduction gates:** (a) unrestricted B refit reproduces 0.00059034; (b) the E.15
   pooled numbers reproduce from source before any module is edited; (c) M.1's
   `delta_pred` column reproduces E.5's committed values on the intersection.
4. **Route C simulation** (M.0's diagnostic) doubles as its self-check: the estimator run
   on simulated data with known τ² must recover a positive reliability at large PA.

Standard repo hygiene: config-driven code in `src/analysis/`, notebooks read results and
never recompute (notebooks/README.md rule), pytest for every new module.

## §10 Contingencies — pre-decided fallback rules and hard stops

**Fallback rules (decided 2026-08-30; apply them, flag their firing prominently in the
report, do not stop):**

1. **C.3-full cannot emit side-specific predictions** from existing fitted artifacts
   without retraining or new feature code (M.1) → the differential table reports C.2
   only; the omission and its reason are stated in the artifact and notebook. Never
   improvise a differential head.
2. **Route C's −0.366 falls outside both simulation nulls with no bug found** (M.0) →
   the shared noise model is suspect, and Route A uses it too. The headline stands on B′
   alone; Route A is demoted from sensitivity to "reported, unvalidated-noise-model
   caveat"; Route C is reported as unexplained. Flag this outcome at the top of
   `m0_routes.json` and the report.
3. **B′'s τ² is negative or the restricted fit degenerates** (M.0) → the headline falls
   back to the B-vs-A bracket (ceiling 0.356 vs 0.167 shape); B′ is reported as
   degenerate and read as evidence for small τ², parallel to Route C's reading.
4. **The committed Route B number fails to reproduce** (M.0) → the value from current
   code becomes B; both numbers are logged prominently with the git-history cause if
   findable; the build continues.
5. **Low-stratum CI** (M.2) → governed by the numeric precision clause in M.2; no
   judgment call remains.

**Hard stops (never delegated — stop and report, regardless of anything above):**

6. Any step that would touch the frozen split config, redefine strata, or read 2025.
7. Anything not covered by this spec, the plan, or the decision log.
8. The §8 knob arm is out of scope for the build window — it runs after Nate reviews the
   measurement results.

## Deliverables

- `results/phase_m/`: `population.csv`, `m0_routes.json`, `m1_differential_scores.csv`,
  `m1_fraction_of_ceiling.csv`, `m2_stratum_ceiling.csv`, `m3_level_ceiling.json`,
  Route C diagnostic JSON, knob-arm results (post-M).
- `notebooks/08_measurement` per §3.1's notebook table (reads results, recomputes
  nothing): the ceiling apparatus, the fraction-of-ceiling table under B′ and A, the
  stratum table, the level-side figure, the M.4 coverage labels, the M.5 null.
- Decision-log entries for anything that settles during the build, in the standards §4
  format; lab-notebook entry at session end.
- Every 2024 number labeled post-selection descriptive (§0.3).
- Notebook 08 carries a short methods note for the write-up: the ceiling apparatus, the
  route rule, and the §10 fallback rules were all fixed after the pre-registered gate
  failed and before their numbers were read, with the dated decision-log entries as the
  timeline. Adjustments bound execution, never the claims, gates, metrics, or splits.
