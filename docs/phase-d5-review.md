# Phase D.5 — review findings and pending adjustments

Working file for defects, corrections, and optimizations found while reviewing the D.5
machinery and its claim-1 results. Nothing here is implemented. Items graduate out of this
file by being either implemented (with a decision-log entry) or rejected (with the reason
recorded in place).

**What belongs here:** anything found after D.5 shipped that would change a number, a
table, or a conclusion — specification errors, comparability breaks, owed diagnostics,
efficiency work.

**What does not:** new scope. A proposal that adds a modeling capability rather than
correcting one goes to the architecture plan as a v2 item, not here.

**The gate every item must name.** D.5's knobs are validated on composition fidelity and
never on claim-1 (decision log, 2026-08-08). Each item below records which channel exposed
it, because that determines whether it may be acted on now:

| exposed by | may be fixed now | reason |
|---|---|---|
| composition fidelity check | yes | the sanctioned validation channel doing its job |
| a unit or coverage error | yes | the comparison was never valid to begin with |
| the claim-1 number itself | **no** | fixing D.5 to improve claim-1 selects the instrument to flatter the measurement |

Claim-1 numbers produced before an accepted fix are not retracted. Both the pre-fix and
post-fix numbers are reported, per frozen rule #2.

Status values: `open` · `accepted` · `implemented` · `rejected` · `deferred to v2`

**Cost reference.** Rebuilding the D.5 tables and re-scoring one claim-1 variant is ~1 h on
CPU; all three variants ~3 h. A full `d9` retrain is 40 runs at ~25 min ≈ one overnight
session. Rebuilding tensors forces a retrain; rebuilding D.5 tables does not.

---

# Defects

## D5-R1 — the take surface is not conditioned on count

**Status:** open · **Exposed by:** composition fidelity check · **Fixable now:** yes

- **Finding:** `query_tables.fit_take_surfaces` keys the called-strike surface on
  `(stand, p_throws)` and plate location only. Umpires change the zone with the count. The
  surface does not see the count.
- **Evidence:** Shadow-zone takes (`|plate_x|` 0.55–1.05, `plate_z` 1.5–3.5), train seasons:
  **0.7558** called strikes at 3-0, **0.4185** at 0-2. The surface applies a pooled 0.6266 to
  both. Marginal error over 3.89M takes is **+0.0002** — near perfect on average — but
  per-count it is structured: batter-ahead under-called (3-0 −0.0395, 2-0 −0.0322,
  1-0 −0.0187, 0-0 −0.0100), pitcher-ahead over-called (0-1 +0.0253, 0-2 +0.0221,
  1-2 +0.0178).
- **Possible impact:** D.5's only job is count transitions, so the error applies at every
  step and compounds — too many walks in hitter's counts, too many strikeouts in pitcher's.
  Reproduces the composition residual quantitatively: 0-0 takes are 0.710 of 0-0 pitches and
  the 0-0 error is −0.0100, implying a strike-rate error of **−0.00711** against a measured
  residual of −0.0068. League-level, so it should hit calibration far harder than ordering.
- **Causes to explore:** none — the cause is known. The open choice is which fix.
  Light: keep the four location surfaces, add twelve count-specific logit offsets (a constant
  logit shift moves probability most near p ≈ 0.5, which is exactly the shadow zone).
  Heavy: key on `(stand, p_throws, balls, strikes)` — 48 surfaces at ~32 obs per grid cell,
  needing the hierarchical backoff `fit_outcome_table` already uses.
- **Cost to fix:** no retrain. Rebuild D.5 tables + re-score three variants, **~3 h**.
  Accept only if the composition residual shrinks; claim-1 must not be consulted while
  choosing between the two options.
- **Note:** the composition check reported the symptom (ball 0.4022 vs 0.3939, strike 0.3847
  vs 0.3915) and it was read as a pass. The check has no stated pass condition. Write one
  before evaluating any fix, or the fix and its acceptance criterion get chosen together.

---

## D5-R2 — a uniform level bias carries most of the calibration deficit

**Status:** open · **Exposed by:** claim-1 · **Fixable now:** **no** — cause only, never the level

- **Finding:** `phase_d_retrained_head` over-predicts every hitter by a near-constant amount.
  The whole ladder over-predicts 2024; Phase D is the worst.
- **Evidence:** PA-weighted, same eval frame (n=1149):

  | model | RMSE | bias | debiased sd | bias share of MSE |
  |---|---|---|---|---|
  | c3_gbm_full | 0.04634 | +0.00932 | **0.04540** | 4.0% |
  | **phase_d** | 0.04920 | **+0.01771** | **0.04590** | **13.0%** |
  | c2_bivariate | 0.04743 | +0.01068 | 0.04622 | 5.1% |
  | c3_gbm_outcome | 0.04676 | +0.00646 | 0.04631 | 1.9% |
  | no_info | 0.05171 | +0.00462 | 0.05151 | 0.8% |

  Near-uniform across exposure strata: +0.0128 / +0.0157 / +0.0152 / +0.0206.
- **Possible impact:** decides the gate. Debiased, Phase D moves from +0.00286 behind
  C.3-full to **+0.00050** — second, not first. The pre-registered null stands either way,
  but its cause is a level offset rather than a worse model.
- **Causes to explore:** (a) **D5-R1**, leading candidate for the Phase D-specific part;
  (b) **training-window length** — Phase C fits a 3-season window (league wOBA 0.3143),
  Phase D trains on 9 (0.3164), a 0.0021 gap that is ~25% of Phase D's +0.00839 excess over
  C.3-full; (c) anything else inside the D.5 composition. **Not** the reserved row — it
  returns 0.3061, below league, so it pushes predictions down (see D5-R3).
- **Cost to fix:** depends on the cause. D5-R1 route ~3 h, no retrain. Shortening the
  training window is a full `d9` retrain, one overnight session. **Never subtract the bias** —
  fitting a correction to claim-1 is the circularity the 2026-08-08 knob entry forbids.

---

## D5-R3 — the composition check is run at the zero embedding

**Status:** open · **Exposed by:** claim-1 follow-up · **Fixable now:** yes

- **Finding:** `query.league_composition` scores the reserved row — the frozen zero
  embedding — and compares its composition to observed league rates.
- **Evidence:** that run returns **0.3061** wOBA. Actual 2024 league wOBA is **0.3102**. The
  PA-weighted mean of real hitter predictions is **0.3283**. The trained rows' centroid has
  norm 0.1136 against a typical row's 0.7079, so the mass of hitters sits near but not on the
  origin, and the trunk is nonlinear so `f(mean) ≠ mean(f)`.
- **Possible impact:** the composition check is the **only** channel licensed to adjudicate
  every D.5 knob. If its reference point is not the league, every knob it has certified was
  certified at the wrong place. This is a validity problem for the check, **not** a cause of
  D5-R2 — the zero row is low, so it pushes predictions down.
- **Causes to explore:** whether the zero row is intended as a calibrated league reference or
  merely a stable origin for cold start. The 2026-07-30 entry says "an unseen hitter and a
  generic hitter are the same point," which fixes cold-start behaviour but says nothing about
  level. Settle the intent before changing anything.
- **Cost to fix:** no retrain, **~1 h**. Either re-run the check as a PA-weighted average over
  real hitters, or keep the zero row as the reference and document the 0.022 offset so the
  check is read correctly.

---

## D5-R4 — roughly 5% of train batted balls carry imputed launch data

**Status:** open, unverified · **Exposed by:** data audit · **Fixable now:** yes, expensively

- **Finding:** two exact launch-angle values appear far more often than their neighbours, with
  exit velocity collapsing onto two values. Almost certainly Statcast placeholders for
  untracked batted balls.
- **Evidence:** launch angle **−21°** appears 32,374 times against ~5,800 for each neighbour;
  **69°** appears 21,053 times. Exit velocity at those rows concentrates on **82.9** and
  **80.0**. Share of balls in play: 3.6–5.8% and 2.7–3.5% through 2019, dropping to **0.60%**
  and **0.27%** from 2020 — the TrackMan → Hawk-Eye transition.
- **Possible impact:** contaminates the quality-head labels, the outcome table `V`, and the
  quantile bin edges themselves. Concentrated in low-value regions (ground balls, popups), so
  the effect on wOBA level is likely small, but the model is trained to predict a spike that
  is not physical.
- **Causes to explore:** verify against Statcast documentation before acting — this is
  inferred from the distribution, not confirmed. Check whether those rows carry other tells
  (missing `hit_distance_sc`, specific `des` text). Decide whether to drop them, mask their
  quality labels, or keep and document.
- **Cost to fix:** dropping or masking changes the tensors, which forces rebuilt bin edges and
  a **full `d9` retrain** — one overnight session plus ~3 h of re-scoring. Verification alone
  is ~1 h and should happen first.

---

## D5-R5 — the `nospray` ablation arm is unreadable

**Status:** open · **Exposed by:** unit error · **Fixable now:** yes

- **Finding:** `nospray`'s reference score is not comparable to any other `d9` arm. Dropping
  the spray head removes a loss term, so it scores five factors where every other arm scores
  six.
- **Evidence:** `nospray` reference 0.81499 against baseline 1.02666. Training logs record only
  the epoch total (`train 0.79585 val 0.81513 ref 0.81513`) with no per-head decomposition, so
  the comparison cannot be recovered from `results/phase_d/sweep_log.csv`.
- **Possible impact:** the spray-head ablation has no verdict. Frozen rule #2 requires every
  pre-registered arm to be measured and reported, so it is **not discharged** until this is
  resolved.
- **Causes to explore:** none — the cause is known. Same units error as the `d8`-vs-`d9`
  reference columns, which already has a 2026-08-08 entry; the guard it describes was not
  applied within a single stage.
- **Cost to fix:** no retrain, **~1 h**. A scoring pass over the five shared factors for the
  `nospray` and baseline checkpoints, in the manner of `src/model/score_screen.py`.

---

# Open questions

## D5-R6 — predicted spread does not narrow at low exposure

**Status:** open · **Exposed by:** claim-1 follow-up · **Fixable now:** diagnose first

- **Finding:** implicit shrinkage should pull rarely-seen hitters toward the reserved row, so
  their predictions should cluster more tightly than regulars'. They do not.
- **Evidence:** PA-weighted predicted sd by prior side-specific exposure —
  **0.0268 / 0.0263 / 0.0270 / 0.0267** across 0-50, 50-200, 200-700, 700+. Observed sd falls
  over the same strata (0.0598 / 0.0522 / 0.0488 / 0.0480) as sampling noise drops.
- **Possible impact:** the shrinkage story the project has been telling — AdamW decay every
  step against gradients only on batches containing the hitter — may not be producing
  exposure-dependent partial pooling in the output. Bears directly on the low-exposure claim,
  which is the thesis population.
- **Causes to explore:** the **D.7 diagnostic** (`‖e_h‖` against `n_h`) is the pre-registered
  instrument and is still unbuilt. Spread within a stratum is not the same as shrinkage toward
  the mean, so the level gradient (0.3022 → 0.3399 across strata) must be read alongside it —
  it may be real talent rather than pooling.
- **Cost to fix:** diagnosis only, **~2 h**, no retrain. Measured on the embedding, so clean to
  run without touching claim-1. Any actual change to shrinkage is a retrain.

---

## D5-R7 — the top exit-velocity bin is open-ended and holds disproportionate value

**Status:** open · **Exposed by:** design audit · **Fixable now:** yes, expensively
**Needs Nate's read before a fix is chosen.**

- **Finding:** EV bin 23 is the open-ended tail. It spans a range the model cannot resolve
  inside, and that range is where batted-ball value is most sensitive.
- **Evidence:** bin 23 covers **107.2 → 122.4 mph**. It is **4.2%** of batted balls and carries
  **11.2%** of all batted-ball wOBA. Within it, at sweet-spot angles (8–32°), mean wOBA rises
  1.2648 (107–110) → 1.3250 (110–113) → 1.3651 (113–116) → **1.4521** (116+).
- **Possible impact:** a 0.19 spread the model cannot express, concentrated in the highest-value
  region. Mass-weighted the effect is second-order — most of the bin sits at 107–110 — so this
  is a resolution limit rather than a bias.
- **Causes to explore:** 24 equal-mass quantile bins were fixed in the architecture plan §1.5
  before Phase A; the recorded rationale covers equal-mass vs equal-width only, never bin count
  or tail handling. Options: more bins, unequal bin widths in the tails, or an explicit tail
  model. **Nate's call** — he flags 107–122 as likely too wide.
- **Cost to fix:** changing bin count or edges rebuilds the tensors and forces a **full `d9`
  retrain** — one overnight session plus ~3 h re-scoring.

---

## D5-R8 — the Phase C ladder has no trailing-xwOBA rung

**Status:** open · **Exposed by:** design audit · **Fixable now:** yes
**Needs Nate's read — this is a baseball judgement, not a code one.**

- **Finding:** C.1 is trailing wOBA. xwOBA stabilises faster and is arguably the strongest
  cheap projection baseline in the sport. It is absent from the ladder.
- **Evidence:** `estimated_woba_using_speedangle` **is present** in the raw snapshot
  (`data/raw/statcast/snapshot_2026-07-14/`) but is dropped by `clean.RETAIN_COLUMNS`, so it
  never reaches the processed table.
- **Possible impact:** if trailing xwOBA beats trailing wOBA on this frame, the ladder
  currently understates what Phase D has to clear, and any Phase D margin is measured against
  a weaker incumbent than the field would use.
- **Causes to explore:** whether a *trailing* xwOBA rung is the right comparison at all —
  xwOBA is descriptive, uses actual K and BB, and is not a projection. Nate's baseball read
  decides whether this belongs on the ladder or is a category error. Also whether to use
  Statcast's field or compute the equivalent from the project's own `V` table marginalised
  over spray.
- **Cost to fix:** no Phase D retrain. Add one column to `RETAIN_COLUMNS`, re-run `clean.py`
  and `eval_targets.py`, build the rung, re-score the ladder — **~2 h** plus the cleaning pass.

---

# Confirmations

These change a conclusion but need no fix. Each is owed a decision-log entry.

## D5-R9 — ordering leads the entire ladder

- **Finding:** `phase_d_retrained_head` orders hitters better than every Phase C rung.
- **Evidence:** weighted rank **0.4608** vs C.3-full 0.4417, C.2 0.4103; unweighted rank
  0.3952 vs 0.3704. Low stratum **0.2643** vs 0.1665 and 0.1389 — but the paired interval is
  [−0.035, +0.286] and includes zero.
- **Possible impact:** the ordering gate is not met and nothing is demonstrated. The point
  estimates are nonetheless the best on the board, and the low-stratum gap is the thesis claim.
- **Causes to explore:** whether the low stratum can ever resolve at n=380 against a one-season
  target, or whether the claim needs a different estimand.
- **Cost to fix:** none. Report as a null with the point estimates stated.

## D5-R10 — the block arm fires

- **Finding:** removing B.2's five flagged context features hurts. They earn their place.
- **Evidence:** +0.00163 against a seed noise floor of 0.00091 — **1.8×** the floor.
- **Possible impact:** discharges the B.2 deferral open since 2026-07-29.
- **Causes to explore:** none.
- **Cost to fix:** none. Owed a decision-log entry.

## D5-R11 — the bilinear arm is still null with the split head present

- **Finding:** the low-rank hitter×context interaction remains null after the fourth factor
  landed.
- **Evidence:** +0.00028, **0.3×** the noise floor.
- **Possible impact:** closes the handoff item "re-run the bilinear arm if the foul head
  lands." Two-strike plate protection was the likeliest hitter×context interaction and it did
  not appear.
- **Causes to explore:** none. A negative result on the interaction hypothesis.
- **Cost to fix:** none. Owed a decision-log entry.

## D5-R12 — platoon splits reproduce without a platoon parameter

- **Finding:** the model recovers the textbook platoon asymmetry from one embedding per hitter.
- **Evidence:** n=308 with ≥50 PA vs LHP and ≥150 vs RHP. LHB predicted **−0.0310** vs observed
  −0.0283; RHB predicted **+0.0147** vs observed +0.0160. Individual split correlation
  **+0.41**, despite observed splits at these sample sizes being mostly noise.
- **Possible impact:** a passed diagnostic the project never registered. Platoon skill is
  emergent — it comes from one embedding routed through hand-specific pitch distributions,
  with no per-hand parameter anywhere.
- **Causes to explore:** none. Worth registering as a standing diagnostic so a future change
  that breaks it is caught.
- **Cost to fix:** none.

## D5-R13 — the quality chain carries most of what the model knows

- **Finding:** conditioning launch angle on exit velocity, and spray on both, is where nearly
  all the quality-head signal lives. Predicting the three independently would discard it.
- **Evidence:** held-out 2024, five-seed ensemble. Launch angle: trunk alone 0.0282 nats, chain
  adds **0.2083** (88.1%). Spray: trunk 0.0289, chain adds 0.2317 (88.9%). Model-free
  `I(LA;EV) = 0.2044` confirms the LA figure. Independence would over-value batted balls by
  **+0.0117** wOBA.
- **Possible impact:** justifies the autoregressive factorisation on measured grounds rather
  than assumption.
- **Causes to explore:** the chain **order** was never tested. EV → LA → spray is one of six
  exact factorisations, and they are not equally learnable under finite capacity and teacher
  forcing. Not currently scheduled.
- **Cost to fix:** none for the finding. Testing an alternative order is a full `d9` retrain.
