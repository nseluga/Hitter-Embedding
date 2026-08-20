# Phase E — Evaluation Specification

What Phase E measures, in what order, and against what pre-registered expectation.
Phase E has two halves that must not be run together: first **validate the instrument**
(does the scorer report what it claims to report), then **evaluate effectiveness** (does
the model do a good job of what it set out to do). A number from the second half is
uninterpretable while the first half fails.

Every choice below is either fixed by the architecture plan, fixed by a decision-log entry
named by date, pre-registered here, or scheduled as an ablation — nothing is left to be
settled during implementation. Amendments after approval go to `docs/decision-log.md`,
which remains the authority.

**Pre-registration clause.** Each numbered step below carries a **pre-registered
expectation**, written before the number it describes was computed. A step whose result
contradicts its expectation is reported as a contradiction, never reframed. Negative and
null results are kept and reported (frozen rule #2, research standards §6).

---

## 1. What this window computes, and what it does not

Phase E's opening window is **read-only with respect to the model**. It scores no new
season, trains nothing, and changes no architecture, loss, or data-pipeline code. It
consumes checkpoints and prediction tables that Phase D.5 already produced.

| step | what it establishes | half |
|---|---|---|
| E.1 | whether the walk gap is a population artefact | validate |
| E.2 | whether the walk bias is uniform or hitter-differential | validate |
| E.3 | whether the absorbing-rate errors account for the wOBA level bias | validate |
| E.4 | whether `pred_woba` is calibrated against realized wOBA | validate |
| E.5 | whether the model carries real platoon knowledge (Route A vs Route B) | evaluate |
| E.6 | whether the swing head or the pitch resampler owns the walk excess | validate |

**Deliberately out of this window, each with its reason:**

| item | reason |
|---|---|
| the 2025 final run | moved after Phase O — see §2 |
| per-count attribution of the walk excess | requires a full re-solve over the pitcher pool (hours); E.6 answers the ownership question more cheaply |
| the Phase C ladder re-scored under `d10` | an overnight sweep; its own window |
| the `block` adoption decision | depends on the ladder re-score |
| any fix to the walk gap | diagnosis only; a fix is a Phase O item |
| retroactive re-crediting of D.5's count offsets under the corrected credit rule | reopens a closed phase; its own decision-log entry |

---

## 2. The 2025 final run moves after Phase O

`Layer1_Architecture_Plan_v2.md:167` places the 2025 claim-1 table in Phase E. Line 169
places optimization in Phase O, **after** it. Line 120 requires that the winning
configuration be refit from scratch on 2015–2024 before 2025 is reported, and that only
the configuration carries across the refit.

Those three lines cannot all hold in the order written. Under it the test season is spent
on a configuration that Phase O then changes, leaving two options: Phase O's gain is never
tested out-of-sample, or 2025 is scored twice and stops being a test season. Neither is
acceptable.

**Pre-registered resolution.** Phase E evaluates on the **2024** frame only. The refit on
2015–2024 and the single 2025 report happen once, after Phase O has settled the
configuration. `Layer1_Architecture_Plan_v2.md:175` does not block this: the Oct 1 abstract
requires "a **preliminary** claim-1 table", and the 2024 table is that.

The refit itself is **not** optional and is not being dropped. Its justification is
numerical and stands: without it 75% of the 2025 low-exposure stratum — the headline
stratum — sits on the reserved zero embedding row, against 42.7% with it, and 42.7% is the
share every Phase C baseline was produced under. Scoring 2025 from 2015–2023 checkpoints
would produce a number that is neither meaningful nor comparable to the ladder.

The guard stays as written: `assert_not_test_season` refuses 2025 outside `--final-run` in
all six call sites. Nothing in this window passes that flag.

---

## 3. E.1 — Matched-population fidelity control

The §8 league-fidelity check compares two differently-selected populations. The modelled
side averages the ~1,074 hitters holding a trained embedding row, weighted by their 2024
plate appearances. The observed side averages **all** plate appearances in the train
window, including every hitter with too little history to earn a row. `query.py:375-407`
states the caveat in its own docstring and does not fix it.

Selection into "has a trained row" is selection on exposure, and exposure correlates with
plate discipline. E.1 recomputes the observed rates over exactly the population the
modelled side scores, using the identical trained-hitter set and the identical PA weights.

Reported as a 2×2 of {train window, 2024 window} × {all hitters, trained hitters only} for
all four absorbing rates, so the population effect and the window effect are separable.

**Pre-registered expectation.** The matched observed walk rate is **higher** than the
unmatched 0.08042, because trained rows select on exposure and exposure correlates with
walk rate. Direction is predicted; magnitude is not.

**Pre-registered decision fork.**
- Matched relative error inside 2% → the walk gap was substantially a population artefact.
  Log it, correct the pass target, and E.6 becomes confirmatory rather than diagnostic.
- Matched relative error outside 2% → the residual is the real target and E.6 owns it.
- Either way the finding is kept and reported.

---

## 4. E.2 — Contamination test

E.1 measures the size of the walk bias. E.2 measures whether it can corrupt claim-1 at all.

Claim-1's headline is a **rank** correlation and a PA-weighted RMSE. A bias that is the
same for every hitter shifts a level and leaves every rank untouched. A bias that scales
with the hitter's own walk rate compresses or stretches the spread and does corrupt ranks.
These are different failures and the §8 check cannot tell them apart, because it reports
one league aggregate.

Per hitter-hand group, define `err_bb = modelled rate_bb − observed 2024 rate_bb`, and
regress `err_bb` on the group's own observed walk rate, PA-weighted.

**Pre-registered expectation.** The slope is **negative** — the composition shrinks toward
a league-average take profile, so it over-predicts walks for low-walk hitters and
under-predicts for high-walk hitters. A slope statistically indistinguishable from zero
would mean the walk gap is a pure level artefact and cannot touch the rank claim.

**Pre-registered reading.** The reported quantities are the slope, its PA-weighted
standard error, and the implied compression ratio `1 + slope`. A compression ratio near 1
clears the walk gap of any effect on the rank claim; a ratio far from 1 makes the walk gap
a claim-1 problem and not only a composition-fidelity problem.

---

## 5. E.3 — Level-bias closure

The composition carries a `+0.01771` wOBA level bias (2026-08-12). The absorbing rates are
wrong by known amounts. E.3 asks whether the second explains the first.

The chain is solved exactly, so the level contribution of an absorbing-rate error is exact
arithmetic: an excess of a state's rate multiplied by the difference between that state's
wOBA value and the value of the mass it displaced. Walks are paid `wBB`, hit-by-pitch
`wHBP`, strikeouts zero, and balls in play their composed expected points.

**Pre-registered expectation.** The four absorbing-rate errors account for **less than
half** of the `+0.01771` level bias. The remainder sits in `V`, the contact-quality
valuation, which is where an over-valued ball in play would show up and which no absorbing
rate can express.

---

## 6. E.4 — Calibration check

A named Phase E deliverable (`Layer1_Architecture_Plan_v2.md:167`). **Unscored readout** —
it sets no knob and adopts nothing, because the scorer is the model and a knob tuned on a
claim-1-adjacent number is circular (`phase-d5-spec.md` §9).

Regress realized wOBA on `pred_woba`, PA-weighted, overall and per exposure stratum.
Report the intercept, the slope, and a decile reliability table.

**Pre-registered expectation.** The slope is **below 1** in every stratum and **lowest in
the low-exposure stratum**, because the composition shrinks a cold or thin embedding row
toward the league profile and shrinkage flattens the predicted spread. A slope above 1
would mean the model over-disperses, which would contradict the shrinkage account and is
reported as a contradiction if it occurs.

---

## 7. E.5 — Platoon differential, scored

This is the effectiveness half, and it is the test claim-1 cannot perform.

A model can score well on side-specific wOBA while holding **zero** platoon knowledge: it
learns each hitter's overall quality and applies the league-average platoon split to
everyone (call this Route A). Route B is real, hitter-specific platoon skill. Claim-1's
side-specific RMSE cannot separate them, because most of a side-specific wOBA's variance is
overall quality.

The separating quantity is the **difference**, per hitter, between the two sides:

```
delta_pred = pred_woba(h, L) − pred_woba(h, R)
delta_obs  = woba(h, L)      − woba(h, R)
```

Scored over hitters with both sides in the eval frame, weighted by the harmonic
denominator of the two sides (the precision of a difference is set by its scarcer side).
Metrics are the same two claim-1 uses: PA-weighted RMSE and weighted rank correlation,
stratified by prior exposure, low stratum the headline. Stratum is assigned on the
**minimum** of the two sides' prior exposure, for the same reason the weight is harmonic: a
differential is only as well-informed as its scarcer side, and assigning on the vs-RHP
side alone would call almost every hitter well-exposed.

**Reference model, pre-registered.** Route A explicitly instantiated: every hitter is
assigned the league-average platoon differential for his stand, computed on train seasons
only. It has, by construction, zero hitter-specific platoon knowledge, and it is the
opponent — beating realized noise is not the claim, beating Route A is.

**Adoption rule, pre-registered.** The model is credited with platoon knowledge only if it
beats Route A on weighted rank correlation in the stratum claimed, by paired difference
with batter clustering, with a 95% interval excluding zero. This is the same form as the
Phase D gate (`Layer1_Architecture_Plan_v2.md:189`) and no numeric threshold is
pre-registered.

**Pre-registered expectation.** The null is expected to survive: `delta_obs` is a
difference of two noisy means and therefore carries roughly twice the variance of either
side, while the true between-hitter spread in platoon skill is small. The power arithmetic
is reported alongside the result, so a null is distinguishable from an absence.

---

## 8. E.6 — Swing-head calibration on real pitches

Two suspects own the walk excess and D.5 ruled out neither.

1. The **swing head** predicts too few swings, so too much mass reaches ball four.
2. The **pitch resampler** draws a more out-of-zone pitch mix than real pitching, so the
   swing head is correct and the pitches it is asked about are wrong. `_sample_grid`
   (`query.py:502`) draws six real rows per (pitcher, count) cell and D.5 never checked
   that draw against the real distribution.

They separate cleanly by removing the resampler from the measurement. Score **real
held-out 2024 pitch rows** directly through the ensemble, no repertoire draw, and compare
the mean predicted `p_swing` against the observed swing rate on the same rows, broken down
by the 12 non-terminal counts, the four (stand, p_throws) cells, and in-zone vs out-of-zone
(`zone <= 9`, `query.py:441`).

The ensemble convention must not drift: swing probabilities are averaged across seeds, not
logits (`query.py:231, 238-245`). Runs under `model.eval()` and `torch.no_grad()`.

**Pre-registered expectation.** If the head owns the gap, the shortfall **concentrates in
three-ball counts**, because those are where one extra take converts directly into a walk.
A shortfall that is flat across all 12 counts points at the resampler instead.

**Pre-registered reading.**
- Head matches observed swing rate on real pitches → the resampler owns the excess. That is
  a query-machinery finding under the D.5 gate table, and no retrain is implied.
- Head predicts too few swings → the head owns it. That is a Phase O item and is **not**
  fixed in this window.

---

## 8b. E.7-E.10 — the fork E.6 opened

**These four steps were not pre-registered.** §8 pre-registered a two-way fork and bound
this window to whichever branch the data selected; E.6 selected the resampler branch, and
E.7-E.10 are the diagnostics that branch requires. They are recorded here as
**fork-opened**, not as pre-registered predictions, and no result below is credited under
the §9 rule. The distinction is the whole reason §8 wrote the fork down in advance: it
constrains which question gets asked next, and it does not license a free hand afterwards.

Each of the four is stated with the expectation written before it was run, since the
sequence is genuinely sequential and each result determined the next module.

### E.7 — is the resampled pitch mix the real pitch mix?

`_sample_grid` draws six real pitch rows per (pitcher, stand, count) cell, backing off to
the pitcher's rows at that strike count and then to all his rows against that stand. D.5's
diagnostics record backoff firing on 16.7% of cells. E.7 evaluates the SAME take surface on
the drawn rows and on the pitchers' own exact cells, weighted identically, so the only thing
that differs is the draw.

**Expectation before running:** the draw is more out-of-zone than the exact cell in
three-ball counts, because a 3-0 cell that backs off to all-zero-strike rows imports pitches
thrown when the pitcher was under no obligation to find the zone.

### E.8 — what the draw gap is worth in walks

A gap in `P(ball | take)` at 1-0 is not worth what one at 3-1 is worth, so E.7's table is
not yet an answer. E.8 builds a league chain from **counted frequencies of real pitch
descriptions**, shifts each count's take mass by E.7's measured gap, and re-solves
`absorbing_rates`. No model is loaded. The chain is solved by backward induction, so the
attribution carries no sampling noise.

**Expectation before running:** positive and material, since E.7's gaps skew toward more
balls in the counts that feed ball four.

### E.9 — the draw's second channel

Ball mass is `(1 - p_swing) * P(ball | take, location)`. E.8 priced the second factor while
holding the first at its league value, which leaves a channel unmeasured: the swing head is
calibrated on real pitches (E.6) but is asked in composition about **resampled** ones, and a
correctly calibrated head returns a different `p_swing` on a different pitch mix. E.9
measures that paired — same hitters, same pitchers, same counts, drawn rows against rows
from the pitcher's own cell with backoff disabled — and prices it through E.8's chain.

**Coverage shortcut, stated where the result is stated (standards §6):** a sampled panel of
200 pitchers per handedness cell and 4 hitters spanning the exposure range, against 24
reference draws per cell. The estimand is a paired difference in which the hitter effect
cancels, which is what makes a small panel adequate; it is still a sample and the summary
JSON records it.

**Expectation before running:** same sign as E.8, adding to it rather than offsetting it.

### E.10 — is the chain itself biased toward walks?

If neither named suspect owns the residual, the composition's own structure is the next
candidate. The chain treats a plate appearance as independent draws from a (pitcher, count)
cell; real plate appearances are sequences, and how a pitcher attacks depends on the hitter
in a way a marginal cell rate averages away. E.10 tests this **model-free**: build the chain
from perfectly observed transition frequencies and compare its absorbing rates against the
observed plate-appearance outcomes on exactly the same plate appearances, joined on
`(game_pk, at_bat_number)` so no population difference can leak in. Whatever gap survives
belongs to the composition and to nothing Phase D trained.

Two aggregations are reported, because the composition uses the second: a single pooled
chain, and a chain per pitcher with `W(0,0)` averaged under the composition's own weights.
Their difference is the Jensen term the composition already carries.

**Expectation before running:** a positive walk bias, because conditioning on the count
alone discards the within-plate-appearance dependence that ends plate appearances early.

---

## 9. The corrected credit rule

The D.5 credit rule compares a **paired delta** (same seeds, knob off vs on) against the
between-seed spread of a **level** (0.00587 for walks). Those are different quantities. The
seed effect cancels inside a paired difference and does not cancel inside a level, so the
paired delta is far quieter than the level's spread and the comparison can essentially
never credit anything. The 2026-08-12 entry records the defect; this is the repair.

> A fix is credited when the mean paired difference across seeds exceeds the between-seed
> spread **of that paired difference**, not the between-seed spread of the level. The paired
> difference is measured on identical seeds with the knob off and on. The credit verdict and
> the 2% band verdict are always reported together; a knob may earn credit and still leave
> the rate outside the band, and reporting either alone misstates the result.

This rule governs Phase E and forward. D.5's already-published verdicts are **not**
retroactively re-credited in this window.

---

## 10. Constraints binding this window

- 2025 is never scored outside `--final-run`. Nothing here touches 2025.
- The `+0.01771` level bias is never subtracted from any reported score.
- D.5-style knobs validate on composition fidelity, never on claim-1
  (`phase-d5-spec.md` §9, the circularity break).
- `query.py` is the scorer. Any edit to it here must be **additive and default-off**, such
  that re-running the existing fidelity path reproduces
  `results/phase_d/d5_diagnostics_d10_baseline.json` field-for-field.
- The CLAUDE.md ML verification gate blocks architecture, loss, or pipeline changes before a
  real run. **This window trips none of them.** It adds analysis modules and reads existing
  checkpoints. The gate's training-run items — overfit one batch, loss-scale sanity,
  determinism across runs, split-boundary — do not apply because nothing trains; the
  applicable items are shape assertions, eval-mode hygiene, decode-one-batch, and the full
  suite.

---

## 11. Decision status

| item | status |
|---|---|
| 2025 final run moves after Phase O | pre-registered here, decision log 2026-08-18 |
| Phase E evaluates on the 2024 frame only | pre-registered here |
| the 2015–2024 refit remains required before any 2025 report | architecture plan §5, line 120 |
| corrected credit rule (paired-difference spread) | pre-registered here, decision log 2026-08-18 |
| matched-population fidelity as the pass target | pre-registered here, §3 fork |
| contamination slope as the claim-1 relevance test | pre-registered here |
| platoon differential scored against an explicit Route A | pre-registered here |
| platoon adoption rule, paired, batter-clustered, 95% excluding zero | pre-registered here |
| calibration reported unscored | 2026-08-08 (§9 circularity break) |
| per-count attribution of the walk excess | deferred, §1 |
| C-ladder re-score under `d10` and `block` adoption | deferred, §1 |
| retroactive re-crediting of D.5 count offsets | deferred, §1 |
| any fix to the walk gap | Phase O, §1 |
| E.7-E.10 | **fork-opened, not pre-registered** — §8b |
| resampler exonerated as the walk-gap owner | E.9, both channels priced, net -0.00003 |
| composition structure as the named residual owner | E.10, model-free, +0.00117 |
| the strikeout shortfall has no named owner | open, §8b — E.10's structural K bias is the wrong sign |
| E.11-E.12 | **fork-opened, not pre-registered** — §12.0 |
| E.13-E.15 | pre-registered here, §12 |
| E.16 (non-handedness conditional query) | deferred to the frontend phase, §12.6 |
| full calibration/refinement decomposition of a proper score | **retired**, §12.4 |
| ensemble interval coverage in low-exposure strata | owed since 2026-08-08, discharged in E.14 |
| deployment-bias audit (architecture §5 item 5) | deferred to write-up, §12.6 |
| the claim-1 verdict on disk is from the D.8 arm | superseded by E.11 |

---

## 12. Remaining steps (2026-08-19)

### 12.0 Pre-registration disclosure

E.11 and E.12 were proposed **after** an exploratory read of the numbers underlying them
(`d5_arms_verdict_d10.json`, `d5_predictions_d10_*.csv`, `c_claim1_scores.csv`). They carry
the same `fork-opened` label E.7-E.10 carry and are not blind predictions. E.13, E.14 and
E.15 are pre-registered: the expectation recorded under each was written before the step ran.

The corrected credit rule of §9 applies unchanged — a paired difference is compared against
the between-seed spread **of that paired difference**, not against the spread of either side.

### 12.1 E.11 — the §189 claim-1 gate, re-scored on the D.10 arm

**Why this exists.** The gate is not unbuilt; `d5_report.py` implements it in full, and
D5-R18(3) already replaced the single hard-coded stratum with a per-stratum verdict whose
decisive row is pre-registered as `low`. What is wrong is the *vintage*:
`results/phase_d/d5_claim1_verdict_phase_d_baseline.json` is dated 2026-08-09, the D.10
prediction tables are dated 2026-08-14, and the on-disk verdict carries four keys where the
current code writes twelve. Every claim-1 number quoted in this project to date is therefore
a **D.8** number read through a superseded verdict schema.

**Method.** Re-run `d5_report.py` unchanged against `d5_predictions_d10_baseline.csv`, writing
under label `e11_d10_baseline` into `results/phase_e/`. No new code, no edit to the scorer, no
touch to any Phase D artefact. This is the module's own rule 1 — the ladder is re-scored, not
remembered.

**Read against.** §189: RMSE against C.3-full; ordering against **both** C.2 and C.3-full on
the denominator-weighted rank correlation; each by paired bootstrap with batter clustering and
a 95% interval excluding zero; decisive stratum `low`. The power restatement travels with any
null, so a null cannot be read as "measured and absent" (D5-R18(4)).

**Standing caveat, not re-litigated here.** Architecture §243: 42.7% of the reported low
stratum sits on the shared cold-start row. A low-stratum result is substantially a statement
about the context tower, not about the embedding. `d5_trained_spread_*` splits it.

### 12.2 E.12 — is the D.10 ablation table ranking arms by level bias?

**Claim under test.** Across the seven non-degenerate D.10 arms, the decisive RMSE column may
be tracking each arm's mean predicted wOBA rather than its ranking skill. If so, the ablation
table selects arms for having a smaller level offset, and Phase O — which is specified to
select on that table — would inherit the confound.

**Method.** For each arm, mean `pred_woba` from `d5_predictions_d10_<arm>.csv` against that
arm's decisive RMSE and rank correlation from `d5_arms_verdict_d10.json`. Pearson and Spearman
on both, plus the five-seed within-arm spread of the mean as the scale the between-arm spread
must beat. `invfreq` is reported but excluded from the correlation as degenerate (its level
moved -0.050 and its RMSE blew out to 0.0607); excluding it is stated, not silent.

**Why it is not a proposal to debias.** It is not. The `+0.01771` level bias is never
subtracted (§10), and D.5-style knobs validate on composition fidelity, never on claim-1. E.12
is a statement about **what the table measures**, addressed to Phase O's selection criterion.

**Where the finding goes.** Decision log only. Per the 2026-08-19 scope ruling, no Phase E
finding amends the architecture plan; the §3 Phase O selection line is proposed for amendment
when Phase O opens, and the log is the authority in the interval (architecture §7).

### 12.3 E.13 — why is modelled `E[wOBA | BIP]` 0.37864 against 0.36353 observed?

E.3 localizes 74.5% of the total level bias in the **value** channel, not the rate channel,
and the balls-in-play term carries it. Two candidate mechanisms, **hard cap at two** — if
neither convicts, the residual is reported as unexplained rather than pursued.

**Check A — the measured-share seam** (`src/model/query.py:460-500`). `quality` is
`E[points | in play AND Statcast measured it]`; the unmeasured remainder is priced at a fixed
league constant. If the modelled and observed populations differ in measured share, or if the
unmeasured constant is set above the truth, the seam produces a level offset with no
mispredicted pitch behind it. **Expectation before running:** this accounts for a minority of
the gap, order 0.002-0.004 of the 0.0151, because `measured_share` is a league rate applied
uniformly and cannot easily generate a one-sided error of this size.

**Check B — Jensen's inequality on a convex payoff.** `V` maps batted-ball descriptors to
points and is convex over the region that matters (barrels are worth disproportionately more).
The model emits a distribution over descriptors and integrates; if its distribution is
over-dispersed relative to the truth, `E[V(x)] > V(E[x])` inflates the value with correctly
centred inputs. This is structurally identical to what E.10 convicted the count chain of.
**Expectation before running:** this is the larger share, order 0.008-0.012, and the test is
whether re-integrating `V` against the *observed* descriptor distribution over the same
population closes most of the gap.

**Stop rule.** Both checks are read-only against existing prediction tables and the fitted `V`.
Neither edits `query.py`. If the two together leave more than a third of 0.0151 unexplained,
that residual is written down as unowned and Phase E closes on it.

### 12.4 E.14 — the §5.1 probe, and the ensemble coverage owed since 2026-08-08

Two questions, one step, because both read the same frozen checkpoints.

**Probe (architecture §5 item 1).** The plan's §1.4 pre-registered failure mode is that the
model learns hitter main effects plus context main effects and no interaction, in which case
every query returns the league-average context penalty and platoon skill washes out. The probe
checkpoint was specified as the *early detector* and never ran. It is run here as a
**retrospective diagnostic**, which is a demotion and is logged as one: it can no longer stop
anything, only explain. **Method:** linear decode of the C.2-estimated true split
(`c2_prior_parameters.csv`, `tau2_split_derived`) from the frozen hitter embeddings,
cross-validated by batter, reported as decoded-versus-true correlation, separately by stand.
**Expectation before running:** LHB decodes meaningfully and RHB decodes near zero, tracking
E.5's recovered-spread asymmetry (54% LHB against 18% RHB). If instead both decode well, the
failure is in the composition, not the representation, and that is the more interesting result.

**Coverage (architecture §5 item 3, decision log 2026-08-08).** The 2026-08-08 entry closed the
RPS screen on its first promotion clause alone and recorded that the reliability machinery *"is
owed exactly once, to §5.3's ensemble calibration check in the low-exposure strata"*. That debt
is discharged here. **Method:** the five per-seed compositions §1.3 retains are used as the
uncertainty source; for nominal levels 50/80/95% the empirical coverage of the observed wOBA is
computed per exposure stratum, with a reliability diagram. **This is not E.4.** E.4 measured
the *slope* of observed on predicted — whether the spread of point predictions is right, and it
is not (0.529 low). Coverage asks whether the stated intervals are honest, which is a different
property and can fail or pass independently. **Expectation before running:** intervals are too
narrow in the low stratum, because a five-seed spread measures disagreement between seeds and
not the shared shrinkage all five inherit from the same prior.

**Consequence is bounded in advance, and this is why the step is safe to run.** Architecture
§2.1 line 124: *"if calibration fails, that is a reported limitation, not a rebuild."* The
hierarchical-Bayes fallback is out of scope. Failure produces a sentence, not a work item.

**Retired here: the full calibration/refinement decomposition of a proper score.** The
three-way split (uncertainty - resolution + reliability) is a decomposition of a *probabilistic
forecast of a categorical outcome*. Claim-1 is RMSE on a continuous target, so the machinery
does not fit the surface it is owed on, and forcing it there would be machinery for its own
sake. The reliability half is discharged by the coverage check above; the resolution half is
already answered in substance by E.4's slope and by the rank correlations. **Cost, recorded:**
if a reviewer asks how much of the score is resolution against reliability, there is no number.
Accepted 2026-08-19.

### 12.5 E.15 — the measurement ceiling on the observed platoon differential

**Why.** E.5 reports a within-stand rank correlation of 0.146 and this has been read in-session
as "essentially zero". That reading is wrong if the quantity is close to unmeasurable, and
`c2_prior_parameters.csv` says it is: rho = 0.652 LHB / 0.719 RHB, true split sd 0.0271 / 0.0222,
`n_star_split_implied` 320.65 / 562.98 against single-season exposures far below both.

**Method.** Reliability = true-talent variance over observed variance, from the C.2 posterior
variance components and the realized per-hitter denominators. The ceiling on achievable rank
correlation is approximately the square root of reliability. Report E.5's 0.146 as a fraction
of its own ceiling. Split-half with a Spearman-Brown correction is computed as an independent
check on the C.2-derived number, and if the two disagree by more than a factor of 1.5 both are
reported and neither is adopted.

**The errors-in-variables correction this step also discharges.** The "81.7% against 10.9%"
framing used in-session compared a noise-free predicted variance against a noise-dominated
observed one — the exact trap E.2 was written to avoid, committed one level up in the framing
rather than in the code. The corrected share is recomputed here with the binomial noise
variance removed from the observed side, and the decision-log entry of 2026-08-18 that quotes
the uncorrected figures is superseded by name.

**Expectation before running:** single-season reliability lands near 0.13 LHB and 0.06 RHB,
capping achievable rank correlation near 0.36 and 0.25, which would make E.5's 0.146 roughly
40-55% of ceiling. If reliability comes back far higher than that, E.5 is a genuine failure and
should be reported as one.

### 12.6 Explicitly deferred out of this window

**E.16, non-handedness conditional queries** (pitch-type, velocity band, location). Architecture
§1.3 states these are the same machinery with the pitcher population filtered and no new code
path, so nothing is learned about the *model* by running one here. Moved to the frontend phase,
where the query surface is the deliverable. 2026-08-19.

**Deployment-bias audit** (architecture §5 item 5). Observed matchups are non-random because
managers already platoon: a fringe hitter's weak-side sample is selected on the very trait being
predicted, and the few PAs he does get are taken in unrepresentative contexts (pinch-hit,
blowout, injury cover), so the answer key is biased and not merely noisy. The remedy §5 names is
the natural experiment of hitters whose cross-side exposure expanded in the held-out season, and
it is also the direct answer to the regression-to-archetype objection. That cohort has to be
built, and the SSAC abstract is due Oct 1. Deferred to write-up. 2026-08-19.
