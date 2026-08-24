# Phase O — Tuning Specification

What Phase O tunes, on what metric, and what it is forbidden from touching. Phase O is
**tuning the run, not choosing the model.** The architecture, the arm, the loss, the
features, and the tensor build are frozen exactly as D.10 shipped them.

Every choice below is fixed by the architecture plan (§3, Phase O), fixed by a
decision-log entry dated 2026-08-20, or pre-registered here. Amendments after approval go
to `docs/decision-log.md`, which remains the authority.

**Pre-registration clause.** Each step carries an expectation written before the number it
describes was computed. A step whose result contradicts its expectation is reported as a
contradiction, never reframed.

---

## 1. Why Phase O is not what it was

Phase O was originally "whatever the ablation table says is reachable": the surviving D.8
arms, the embedding dimension, the flagged-five block, spray, the bilinear term. Every one
of those is an **arm selection**, and Phase E left the project with no metric that can
select among converged arms:

- Across the seven D.10 arms, the claim-1 rank statistic and held-out log likelihood
  correlate at **0.000**.
- Claim 1 itself is the paper's headline, so selecting on it makes the headline a selected
  number.
- F.5's pooled skill column is the one column where the model is positive in every
  stratum, and it is **barred** as a selection axis (2026-08-20 entry): pooling collapses
  the hand column the platoon claim lives in, and C.2 and C.3 are absent from that table
  so it is not a ladder in the first place.

Selecting arms on a broken metric is how a null becomes a spurious positive. So Phase O
selects no arms.

## 2. What Phase O tunes

Three knobs, each held at a single value for **all 119 runs in the ledger** and therefore
evidence for nothing:

| knob | pre-O value | Phase O grid |
|---|---|---|
| learning rate | `1e-3`, never varied | {3e-4, 1e-3, 3e-3} |
| warmup | none, no schedule existed | {0, 719 steps = 1 epoch} |
| epoch budget | `MAX_EPOCHS = 50` | unchanged; `best_epoch` has never exceeded 27, so early stopping already governs and this knob is observed, not swept |

`o1` is the 3×2 factorial, two seeds per cell, twelve runs. At the measured 62 s/epoch and
a realized `best_epoch` in the 9–20 band, that is roughly four and a half hours — one
overnight session, against a two-session hard cap.

**Why a null on an untuned model is a weaker null.** "The learning rate was fixed at 1e-3
across all 119 runs and never varied" is not an answer that survives review, and the
project's headline is a null. Phase O's job is to remove that answer.

## 3. The selection metric, and why it is legitimate here

**`reference` — held-out 2024 unweighted log loss per scored row — and nothing else.**

This is the same statistic barred one section above, and the distinction is not a
convenience. Log likelihood cannot be shown to **rank** converged models against each
other. It can detect an **undertrained** one, because an undertrained run is worse on its
own training objective. Detection is the only question Phase O asks.

A caveat on the 0.000 correlation, added after review: it was measured across seven arms,
and seven points carry a 95% interval of roughly ±0.75. It is not evidence that likelihood
and claim 1 are unrelated, and it should never be quoted as if it were. The licence for
using likelihood here is the detection/ranking distinction above, which does not depend on
that number at all.

**The selection frame is 2024, which is also claim 1's frame.** This is the frozen
walk-forward protocol — 2024 selects, 2025 tests — and every Phase D ablation already
selected on 2024, so Phase O introduces no new leak. It does introduce an obligation:
**any claim-1 number computed on 2024 from the tuned build is post-selection and must be
reported as descriptive.** The confirmatory number is the 2025 run, which stays sealed
until Phase M is finished. Phase M carries this labelling requirement.

**Claim 1 is never read during Phase O.** Not as a tiebreak, not as a sanity check.
`src/analysis/o1_select.py` does not import `claim1_eval`, and `tests/test_o1_select.py`
asserts that it does not.

### The promotion rule, fixed in code before any o1 run exists

1. Score each config by the **mean** of `reference` across its seeds. Lower is better.
2. The noise floor is the across-seed standard deviation of `reference` for the D.10
   `baseline` arm — five seeds, same architecture, same build — read from the ledger at
   run time rather than hardcoded. It is currently **1.04e-4**.
3. **`lr1e3` (lr 1e-3, no warmup) is the incumbent** and is in the grid. An arm is
   promoted only if it beats the incumbent by more than **2 standard errors of the
   difference in means** — `sd × sqrt(1/n_arm + 1/n_incumbent)`, with the noise floor
   supplying the per-run `sd`. At two seeds each this equals the noise floor exactly,
   which is why an earlier draft that divided by the raw `sd` looked right; it stops being
   right the moment two arms carry different seed counts, which is exactly what happens
   at the confirmation step. Otherwise the incumbent stands and Phase O reports that the
   untuned setting was already best.
4. **Five challengers are tested against one incumbent at one threshold**, so a promoted
   arm carries `requires_confirmation` until it has **5 seeds**. A two-seed screen
   selects; it never concludes.
5. **A partial grid is refused, not averaged.** `sweep.queue` is config-major, so a night
   that runs short drops whole arms off the end rather than thinning every arm evenly —
   and a mean over the three arms that finished is indistinguishable from a mean over six.
   The six expected arms are pre-registered in `o1_select.EXPECTED_ARMS`, and any absence,
   or any arm with fewer than two seeds, returns `incomplete_grid`.

### The build guard

The incumbent arm re-runs the D.10 baseline recipe at a different stage, so its mean
`reference` must land within **4 sd** of the D.10 baseline's. `reference` is log loss over
the quality bins, and `data/processed/phase_d` and `phase_d5` have different bin edges —
so a run on the wrong build would be in different units while every column still lined up.
The `o1` stage pins `provenance.CANONICAL_DATA_DIR`; the guard catches it anyway, and
refuses the selection rather than reporting it.

That the pin names the right build is established, not assumed. Re-running d10 baseline
seed 0 on `phase_d5` reproduces its logged epoch-0 losses exactly (1.05990 / 1.04681),
and the same command on `phase_d` raises `KeyError('split')` — that build predates the
three-class contact split, so no `--split` run could have used it. Every ledger row now
records its `data_dir` so this never has to be reconstructed again.

## 4. What Phase O is forbidden to touch

**Batch size and weight decay are quarantined**, frozen at 8,192 and 1e-2. `train.py` does
still expose `--batch-size`, and `sweep.py` forwards `--train-args` verbatim, so the
quarantine is enforced in `sweep.main_argv`: either flag inside `--train-args` on an `o`
stage is a hard error. Batch size is not merely a quarantined knob here — it silently
rescales `warmup_for`'s per-epoch cap, so moving it would move a knob Phase O *is* tuning. They are one setting: AdamW decays every parameter every
step while an embedding row receives a gradient only in batches containing that hitter, so
their ratio *is* the shrinkage applied to low-exposure hitters — which is the quantity C.2
estimates and the quantity claim 1 is about. Tuning them toward the low stratum would
implement C.2's shrinkage inside the network and then score the result against C.2.

Also out of scope: a new model class, a new objective, a new representation mechanism (the
cut §2.3 items), the pooled sampler, and any re-opening of a frozen decision.

## 5. O.2 — the mechanism, pre-registered as a diagnostic

D.5's gradient (b) (`results/phase_d/d5_level_attribution.json`) measured, consistently in
all five seeds:

| | rarest quintile (30 train pitches) | most-exposed quintile (10,544) |
|---|---|---|
| mean embedding norm | 0.74 – 1.00 | 0.52 – 0.61 |
| mean projection on the wOBA-raising axis | −0.16 to −0.22 | +0.06 to +0.07 |

Every row initialises at norm **0.057**. So no row is being pulled toward the origin —
every row travels outward, and the *rarest* rows travel furthest while pointing the wrong
way. This is the opposite of what weight decay would produce: at the 10th percentile of
exposure the decay-to-gradient ratio is **23.9:1**, so if decay were binding those rows
would be crushed into the origin, not thrown past everyone else. Decay is being
overwhelmed, not binding — which **rules out the quarantined knobs as the cause** and
points at AdamW's per-coordinate second-moment normalisation: a coordinate is divided by
its own running gradient scale, so a row updated 30 times takes a near-full-size step
every one of those 30 and lands on a short random walk of large norm and arbitrary
direction. Learning rate and warmup are the levers on that; batch size is not.

**Pre-registered expectation:** on the Phase O build, the rarest quintile's mean norm falls
toward the most-exposed quintile's, and its mean projection moves toward zero. This is
measured in **Phase M**, after the tuned build is fixed. A null is reported, not retuned
around, and it is not a promotion criterion — the promotion rule in §3 is the only one.

## 6. Run

```
caffeinate -i -s env PYTHONPATH=. .venv/bin/python -m src.model.sweep --stage o1 --hours 9
PYTHONPATH=. .venv/bin/python -m src.analysis.o1_select
```

The sweep is resumable: the ledger keys on `(stage, config, seed)` and a completed run is
skipped on re-launch. `touch results/phase_d/STOP` stops it after the current run.

Four verdicts. `guard_failed` means the build moved and the selection is refused.
`incumbent_stands` is a real Phase O result — the untuned setting was already best of the
six, and Phase M runs on the D.10 build unchanged. `tuned_pending_confirmation` means an
arm cleared the margin on the two-seed screen; five challengers were each tested against
one incumbent at the same threshold, so the family-wise false-promotion rate is several
times the per-comparison one and a two-seed mean is a thin estimate to spend it on. The
winner is re-run at five seeds (`--seeds 5`, resumable, so only the three new seeds cost
anything) and only then becomes `tuned`. A two-seed screen selects; it never concludes.

## 7. Exit condition

Phase O ends at whichever comes first: a selection written to
`results/phase_o/o1_selection.json`, or the two-session budget. Tuned or not, Phase M
starts on whatever build Phase O leaves behind, and Phase M reports which one that was.
