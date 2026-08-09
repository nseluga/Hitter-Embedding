# Phase D.5 — Query Machinery Specification

Equations, conditioning sets, and tensor shapes at every module boundary, written
before any D.5 code exists. Composes v1's per-pitch conditionals into the side-specific
wOBA the §5.2 claim-1 metric scores.

Every choice below is either fixed by the architecture plan, fixed by a decision-log
entry named by date, pre-registered here, or scheduled as an ablation — nothing is left
to be settled during implementation. Amendments after approval go to
`docs/decision-log.md`, which remains the authority.

---

## 1. What D.5 computes

For hitter `h` and pitcher hand `H ∈ {L, R}`, one number on the wOBA scale:

```
pred_woba(h, H) = Σ_p  w_p · W_p(h, 0, 0)
```

- `p` ranges over pitchers who threw in 2015–2023 with `p_throws = H`
- `w_p ∝` batters faced, normalized over the pool (2026-08-08, pitcher population)
- `W_p(h, b, s)` is the expected wOBA points of a plate appearance between `h` and `p`
  entered at count `(b, s)`

The whole of D.5 is the machinery producing `W_p(h, 0, 0)`.

**What v1 supplies** (`docs/phase-d-spec.md` §1): `p(swing | h,c)`, `p(contact | swing, h,c)`,
and `p(ev, la, spray | contact, h,c)` factorized autoregressively.

**What v1 does not supply**, and §2 fills:

| conditional | why v1 lacks it |
|---|---|
| `p(foul, foul_tip, in_play \| contact, h, c)` | architecture plan §1.3 treats contact as terminal; it is not |
| `p(ball, called strike, hbp \| take, ·)` | take outcomes follow from location, an input and never an output |
| pitcher count-conditional repertoires | `pitcher` is deliberately outside v1's context (`context_features.py:51`) |
| `(ev, la, spray) bin → wOBA` | run value is barred from the loss (2026-08-02) and enters only here |

---

## 2. The four missing conditionals

### 2.1 The contact split — three classes

```
p(foul, foul_tip, in_play | contact, h, c)
```

A three-class head on the trunk, added to v1 and trained with it (2026-08-08, fourth
factor). Below two strikes `foul` and `foul_tip` transition identically; they diverge
only at two strikes, where a caught tip is a strikeout and a foul is not.

**Baseline form, built first.** Until the retrain lands, the split comes from a league
table estimated on train seasons and indexed by `(balls, strikes, in_zone)` — 12 × 2 = 24
cells, `in_zone` from the retained `zone` field. Count and zone are the two effects with
any documented magnitude, and conditioning the baseline on both is what makes the
retrained comparison isolate the *hitter-specific* part rather than rediscovering league
structure. This baseline is a floor and is reported as one.

### 2.2 The take split — three classes, so HBP survives

```
p(ball, called strike, hit by pitch | take, plate_x, plate_z)   fit per (stand, p_throws)
```

Four surfaces, one per handedness combination (2026-08-08, handedness keying). Fit on the
train-season take rows. `TAKE_DESCRIPTIONS` (`labels.py:33`) maps as:

| description | class |
|---|---|
| `ball`, `blocked_ball` | ball |
| `called_strike` | called strike |
| `hit_by_pitch` | hit by pitch |
| `intent_ball` | **excluded from the pool entirely** — see §4.3 |

Three classes rather than ball/strike because HBP is a take, a terminal state, and inside
the wOBA denominator (`eval_targets.py:61`). A two-class model would drop it and
systematically under-predict every hitter.

**Estimator:** a 2-D empirical surface on a fixed `(plate_x, plate_z)` grid, shrunk toward
its own `(stand, p_throws)` marginal by the §5 estimator. No new dependency; the repo has
no GAM and does not need one. Grid resolution and shrinkage strength are pre-registered
knobs under §9.

**Raw coordinates, no height normalization** (2026-08-08, called-strike model).
`sz_top`/`sz_bot` are absent from `clean.RETAIN_COLUMNS` and stay absent.

**The plug-in critique does not transfer.** Deshpande & Wyner warn that a two-stage
estimated-probability covariate can over-leverage pitches whose baseline probability is
near 0 or 1, by inflating the intercept and slopes fit on top of it. D.5 fits **no
coefficient** on top of this surface — it enters a deterministic composition as a
transition probability, so there is no slope to inflate. Stated here so a reviewer does
not have to wonder whether it was noticed.

### 2.3 Pitcher repertoires

Whole real pitch rows, resampled from the labeled parquet, grouped by
`(pitcher, stand, balls, strikes)` (2026-08-08, handedness keying).

The property that makes this work: a resampled row carries its own real context vector,
so pitch physics, missingness flags, the `stand` one-hot, and the `balls`/`strikes`
one-hots are all internally consistent **and already match the cell being queried**.
Nothing is ever overwritten. That is the whole argument for resampling rows instead of
generating them.

**Backoff** when a cell holds fewer than `M` rows, in order:

```
(pitcher, stand, balls, strikes) → (pitcher, stand, strikes) → (pitcher, stand)
                                 → (league, stand, balls, strikes)
```

Blending rather than replacement, by the §5 estimator.

### 2.4 Batted-ball outcome table

```
V[e, l, s] ∈ Δ⁴   over {out, 1B, 2B, 3B, HR}
```

Built by joining in-play pitches to their plate appearance's `woba_category` on
`(game_pk, at_bat_number)`, train seasons only. One plate appearance holds at most one
in-play ball, so the join is 1:1.

**Category probabilities, not wOBA points.** The table stores the outcome distribution;
wOBA points are formed at query time as `Σ_c V[e,l,s,c] · weight_c(eval_season)` from
`src/config/woba_weights.json`. This decouples the table's vintage from the weights',
so a table fit on 2015–2023 is scored under the season the target is measured in.

**No representative bin values are needed.** `V` is indexed by bin, never by physical
value, so the open-ended tails at bins 0 and 23 (`manifest["quality_bin_edges"]` holds 23
*interior* edges) never require a midpoint convention. This removes an open item the plan
carried.

24³ = 13,824 cells over **1,020,993** train-season in-play balls — 73.9 per cell on
average, and far more uneven than that, since the bins are equal-mass marginally and not
jointly. Smoothing is not optional; see §5.

---

## 3. Shapes

| symbol | meaning | value |
|---|---|---|
| `N` | eval rows, one per (batter, p_throws) active in the eval season | 1,281 on 2024, before the pitcher-batter drop |
| `P` | pitchers per hand | **all of them** — 1,483 RHP, 544 LHP (2026-08-08) |
| `M` | pitch rows per repertoire cell | 6, pre-registered §9 |
| `K` | quality bins per dimension | 24, fixed by D.1 |
| `S` | non-terminal count states | 12 |

| boundary | tensor | shape | dtype |
|---|---|---|---|
| repertoire cell | `context` | `(M, 46)` | float32 |
| hitter index | `hitter` | `(M,)` broadcast from one id | int64 |
| context tower output | `z_c` | `(M, 128)` | float32 |
| trunk output | `t` | `(M, 256)` | float32 |
| swing / contact | `p_swing`, `p_contact` | `(M,)` | float32 |
| contact split | `p_split` | `(M, 3)` | float32 |
| quality joint (chunked) | `p_joint` | `(chunk, K, K, K)` | float32 |
| outcome table | `V` | `(K, K, K, 5)` | float32 |
| per-pitch wOBA given in play | `Q` | `(M,)` | float32 |
| per-state aggregates | `A_*`, `G_bip` | `(S,)` per (h, p) | float64 |
| chain solution | `W` | `(S,)` per (h, p) | float64 |
| output | `pred_woba` | `(N,)` | float64 |

`float64` from the aggregation step onward: the chain divides by `1 - A_foul`, and the
output is compared at five decimals against a 0.00096 noise floor.

**Context-tower reuse.** `z_c` depends only on the pitch, not the hitter, so it is computed
once per distinct pitch row (`P × S × M` of them per hand) and reused across all hitters.
The trunk and heads still run per `(hitter, pitch)` pair.

---

## 4. The count chain, solved exactly

12 non-terminal states `(b, s)`, `b ∈ {0..3}`, `s ∈ {0..2}`. Terminal states: walk,
strikeout, hit by pitch, ball in play (2026-08-08, exact solve).

### 4.1 Per-state aggregates

For state `σ`, over the `M` pitches `x` in the `(pitcher, stand, b, s)` cell with
normalized weights `u_x`:

```
A_ball (σ) = Σ_x u_x · (1 − p_swing) · q_ball(x)
A_cs   (σ) = Σ_x u_x · (1 − p_swing) · q_cs(x)
A_hbp  (σ) = Σ_x u_x · (1 − p_swing) · q_hbp(x)
A_whiff(σ) = Σ_x u_x · p_swing · (1 − p_contact)
A_foul (σ) = Σ_x u_x · p_swing · p_contact · p_foul
A_tip  (σ) = Σ_x u_x · p_swing · p_contact · p_tip
A_bip  (σ) = Σ_x u_x · p_swing · p_contact · p_inplay
G_bip  (σ) = Σ_x u_x · p_swing · p_contact · p_inplay · Q(h, x)
```

The seven `A_*` sum to 1 — an assertion, not a comment. `G_bip` carries the wOBA points
and is **not** factorable as `A_bip · mean(Q)`: a pitch that is more likely to be put in
play is not the pitch with average contact quality, and separating them would drop that
covariance.

Foul tips join the strike branch, since below two strikes they are a strike and at two
strikes they are a strikeout:

```
A_strike(σ) = A_cs(σ) + A_whiff(σ) + A_tip(σ)
```

### 4.2 Backward induction

```
W(b,s) = A_ball·V_ball + A_strike·V_strike + A_hbp·wHBP + A_foul·V_foul + G_bip

  V_ball   = wBB              if b = 3     else  W(b+1, s)
  V_strike = 0                if s = 2     else  W(b, s+1)
  V_foul   = W(b, s)          if s = 2     else  W(b, s+1)
```

At `s = 2` the foul term is `A_foul · W(b,2)`, so it divides out in closed form:

```
W(b,2) = [ A_ball·V_ball + A_hbp·wHBP + G_bip ] / (1 − A_foul)
```

**Solve order:** `s` descending 2 → 0, and within each `s`, `b` descending 3 → 0. Every
dependency is on a higher `b` or a higher `s`, so nothing is read before it is written.
`W(3,2)` is the base case and depends on no other state.

No iteration cap, no truncation to report, no RNG. `wBB` and `wHBP` are the eval season's
weights, matching the season the target is measured in.

**`pred_woba = W(0,0)` with no renormalization**, because every terminal state is inside
the wOBA denominator. This is why §4.3 matters.

### 4.3 What the simulator never generates

IBB, sacrifice hits, and interference are outside the wOBA denominator
(`eval_targets.py:61`), so a simulator that produced them would generate plate appearances
the metric does not count. `intent_ball` rows are therefore **excluded from the resampling
pool** — a decision, not an omission. Sacrifice flies *are* in the denominator and score
zero in the numerator, so they live inside `V`'s `out` category and need no special case.

### 4.4 The assumption this rests on

The chain is Markov in count because the pitch draw is independent of the pitches already
thrown, given the count. **Pitch sequencing is therefore ignored**, and that is a modeling
assumption rather than an oversight. It is the condition named in the exact-solve entry's
revisit clause.

---

## 5. Enumerating the quality chain, and smoothing

### 5.1 The exact expectation

```
Q(h,x) = Σ_e p(e | ·) Σ_l p(l | e, ·) Σ_s p(s | e, l, ·) · [ Σ_c V[e,l,s,c] · weight_c ]
```

Computed from **one trunk pass** per `(h, x)` (2026-08-08, exact enumeration). The heads
are plain linear layers over `[trunk ; onehot]`, so conditioning enters as added weight
columns:

```
la_logits   (t, i)    = base_la(t) + E[i]
spray_logits(t, i, j) = base_sp(t) + E_s[i] + L_s[j]

base_la = t @ W_la[:, :256].T + b_la          E     = W_la[:, 256:].T
base_sp = t @ W_sp[:, :256].T + b_sp          E_s   = W_sp[:, 256:280].T
                                              L_s   = W_sp[:, 280:].T
```

Verified against real `forward()` calls over all 576 conditioning pairs: max abs error
**9.5e-07**, one float32 ULP at the observed logit scale of 8.21.

The normalizers are **not** an outer sum — the softmax is taken per conditioning slice.

**The implementation uses a factored form, not this literal one.** Materializing
`(rows, 24, 24, 24)` costs 55 KB per row, and the ensemble needs each seed's joint before
averaging, so the literal form is memory-bound at hours per pass. Writing
`u[m,s] = exp(base_spray[m,s])`, `a[e,s] = exp(E_s[e,s])`, `b[l,s] = exp(L_s[l,s])`:

```
R[m,e,l] = Σ_s u[m,s]·a[e,s]·b[l,s]·V_pts[e,l,s]  ÷  Σ_s u[m,s]·a[e,s]·b[l,s]
```

Both sums are one matmul against a `(576, 24)` matrix built **once**, so the peak tensor is
`(rows, 576)` and the work lands in BLAS instead of an elementwise softmax — about 24×
less memory traffic. Two properties make this exact rather than an approximation:

- Any factor constant across `s` cancels in the ratio, so max-shifting each term keeps it
  numerically stable without changing the value.
- `R` is **linear** in the spray conditional, so averaging the seeds' `R` *is* averaging
  their conditionals. The 4-D joint is never materialized, including for the ensemble.

`query.expected_woba_naive` keeps the literal form as the reference the fast path is
tested against; they agree to **1e-5** on trained and random weights, with and without the
bilinear term (`tests/test_query.py`).

### 5.2 The smoothing estimator

One estimator serves `V`, the repertoire backoff, and the take surface: shrink each cell
toward its own coarser marginal by a method-of-moments weight, the idiom
`c2_bivariate_eb.py` already uses for the bivariate prior and `stabilization.py` uses for
`n* = noise/signal` at `t = 0.5`.

```
p̂_cell = (n_cell · p_cell + α · p_pool) / (n_cell + α)
```

`α` is fit per table by method of moments against the between-cell dispersion, not tuned.

> **Sourcing note.** A search for a published empirical-Bayes treatment of pitch-usage or
> batted-ball-outcome tables found no established peer-reviewed treatment. The form above
> follows this project's own C.2 precedent; it is not borrowed from a source, and that is
> stated rather than implied.

---

## 6. Ensemble

Average the five seeds' per-pitch conditionals, then run **one** composition
(2026-08-08, ensemble combination). Averaging happens at the probability level —
`p_swing`, `p_contact`, `p_split`, and the three quality conditionals — before `Q` is
formed, so the 24³ enumeration runs once rather than five times.

The five separate compositions are still computed and retained as the between-seed spread
that §5.3's calibration check needs. They are **not** the headline; the mixture is.

---

## 7. Output contract

```
DataFrame[batter: int, season: int, p_throws: str, pred_woba: float]
```

Row set derived exactly as every Phase C rung derives it:

```python
drop_pitcher_batters(pa_df)[pa_df.season == eval_season][["batter","p_throws"]].drop_duplicates()
```

matching `c1_trailing.py:130`, `c2_bivariate_eb.py:424`, `c3_gbm.py:430`.

**Coverage must be exact.** `build_eval_frame` merges `inner` and files any shortfall
silently into `coverage["dropped_no_prediction"]`; the failure surfaces later at
`claim1_eval.py:445-446`, where `_paired_setup` asserts both models scored the same
groups. Every paired bootstrap routes through it, so a coverage gap kills every
head-to-head rather than degrading one.

**`stand` per (batter, p_throws)**, following `c3_gbm.py:136` — never keyed on batter
alone, or every switch hitter collapses onto his RHP-facing stand.

**Cold start:** a batter outside the 1,762-hitter vocabulary routes to reserved row 0, so
a prediction always exists (2026-07-30). No new behaviour; D.5 inherits it.

**Integration:** one entry in `c_report.build_predictions` (`c_report.py:101-109`).
Two known snags to handle rather than trip over — `score_all` returns only the last loop
iteration's coverage (`c_report.py:139`), and `min_eval_pa_sensitivity` hardcodes
`c3_gbm_full`/`c2_bivariate` at `:193`.

---

## 8. Composition validation (§5.4)

The simulator is checked against observed baseball, never against its own claim-1 margin.

| quantity | compared against |
|---|---|
| simulated BB, K, HBP, BIP rates from `W(·)`'s absorbing probabilities | train-season observed rates |
| league mean `pred_woba`, denominator-weighted | train-window league wOBA, `woba_weights.json` |

Run with the hitter index fixed at reserved row 0 for the cleanest single number: a
generic hitter against the pooled opponent distribution should reproduce league scoring.

**The pool is 2015–2023, so the comparison is to the train window, not the eval season.**
Comparing to eval-season league wOBA would charge the simulator for league-level drift it
was never given the data to know.

**A tension to name, not to paper over.** Composition validation runs on the modeling
table, which is filtered (regular season, no bunts, no pitchouts, no position-player
pitchers, core-context complete), while eval targets are built from the complete record
and never filtered (`eval_targets.py:6-10`). The two populations differ by construction,
so exact agreement is not the bar and a residual gap is expected. The check is for a
gross composition error, not a calibration certificate.

---

## 9. Pre-registered knobs, and the circularity break

| knob | set by |
|---|---|
| `P`, pitchers per hand | **not a knob** — the whole population is used (2026-08-08) |
| `M`, pitch rows per repertoire cell | 6; estimator stability, then frozen |
| take-surface grid resolution | pre-registered |
| smoothing `α` for `V`, repertoire, take surface | method of moments, not tuned |
| enumeration chunk size | performance only, cannot change a number |

**None of these is ever set by its effect on the claim-1 metric** (2026-08-08, D.5 knobs).
Frozen rule #2 sends unclear choices to a claim-1 ablation, but that presumes the choice
sits upstream of the metric — and D.5 sits inside it, so tuning D.5 on claim-1 would
select the measuring instrument to flatter the measurement. §8's composition fidelity is
the independent criterion, because it scores the simulator against observed league scoring
rather than against the model's margin over Phase C.

`M` is a coverage shortcut, so it is stated wherever a result is stated, per standards §6.
`P` stopped being one: a sampled panel left a **common level shift across every hitter at
once** — 0.0048 wOBA between two draws at 60 pitchers, against a 0.033 between-hitter
spread — and PA-weighted RMSE charges for exactly that. Using the full population removes
the source rather than shrinking it, and makes the number independent of the draw. Note
what set that knob: estimator stability, never claim-1.

**Repertoire backoff is reported, not silent.** The run records how many
`(pitcher, stand, count)` cells were served exactly, how many fell back to the pitcher's
rows at that strike count, and how many to all his rows against that stand.

---

## 10. Compute

Benchmark before committing, mirroring D.3's one-epoch benchmark.

Forward rows are `N × P × S × M`, and the cost driver is the 24³ softmax per row, not the
trunk. Local CPU, one device per comparison set (2026-08-03).

---

## 11. Build order

1. **Baseline.** D.5 on §2.1's league table, over the existing `d6_baseline` /
   `d8_baseline` checkpoints. Produces the first claim-1 number. **A floor, not a headline.**
2. **Retrain.** Add the three-class head, retrain the ensemble, rebuild D.5, and report
   the retrained number *against* the baseline so the head's effect is measured.
   **Pre-registered expectation: small.** The evidence cuts both ways — fouls per whiff
   separates player types, while raw foul rate barely relates to hitter quality and the
   documented count effect is small next to the location effect the trunk already sees.
   Recording the expectation now is what keeps a small result from being reframed later.
3. **Re-run the bilinear arm** if the foul head lands. Its D.8 null predates any foul
   factor, and two-strike plate protection is the likeliest hitter × context interaction.

**Owed in the same retrain, and blocking the claim rather than D.5:** the `block` and
`spray` D.8 arms are pre-registered at `phase-d-spec.md:253-254` and have never run, so
frozen rule #2 is not currently satisfied. The retrain rebuilds every arm regardless, so
folding them in costs one overnight session instead of two.

---

## 12. Implementation notes

Recorded so D.6 does not rediscover them:

- **`model.eval()` is mandatory.** The trunk ends in `Dropout(0.1)` and `forward` does not
  set mode. `score_screen.py` gets it free via `run_epoch`; direct-forward code does not.
- **`torch.load(..., weights_only=False)`** — `d8_invfreq_*` stores a tensor in `args`.
- **Rebuild architecture from `embedding_dim` and `bilinear` only.** Older checkpoints
  (`d6_*`, `screen_*`) lack the newer keys and raise on attribute access.
- **`n_bins` and the vocabulary come from the manifest, not the checkpoint.** The
  vocabulary is `{str(batter_id): row}`; D.5 needs the inverse and must cast the keys back.
- **Pin a known number as a self-check**, per `score_screen.py:42`, so a drifted inference
  path fails loudly rather than quietly.
- **The parquet is 7,347,953 rows; the tensors are 7,293,321.** Pitcher-at-bat pitches are
  dropped in `model_dataset`, not in `clean`, so a pool built from the parquet must apply
  `eval_targets.primarily_pitchers` itself.
- **`assert_not_test_season`** (`claim1_eval.py:589`) guards every entry point. 2025 is
  never scored outside `--final-run`.

---

## 13. Decision status

| item | status |
|---|---|
| exact count-chain solve, foul self-loop in closed form | 2026-08-08 |
| exact 24³ enumeration | 2026-08-08 |
| three-class contact split, retrain, league-table baseline first | 2026-08-08 |
| pitcher pool prior seasons only, batters-faced weighted | 2026-08-08 |
| repertoire and take surface keyed on batter handedness | 2026-08-08 |
| ensemble combines by averaging conditionals | 2026-08-08 |
| raw plate coordinates, no height normalization | 2026-08-08 |
| D.5 knobs validated on composition fidelity, never claim-1 | 2026-08-08 |
| cold start to reserved row 0 | 2026-07-30 |
| run value enters only here, never the loss | 2026-08-02 |
| wOBA denominator as the scoring unit | 2026-07-29 |
| three-class take model so HBP survives | pre-registered here |
| `intent_ball` excluded from the resampling pool | pre-registered here |
| `V` stores categories, weights applied at query time | pre-registered here |
| repertoire backoff order and shrinkage form | pre-registered here |
| `P`, `M`, grid resolution, chunk size | benchmark then frozen, §9 |
| recency-weighting the pitcher pool | deferred, 2026-08-08 pitcher population |
| pitch sequencing inside a plate appearance | out of scope, §4.4 |
| §5.3 ensemble calibration decomposition | deferred to §5.3, 2026-08-08 |
