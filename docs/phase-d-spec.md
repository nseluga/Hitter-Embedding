# Phase D.2 — v1 Model Specification

Equations, conditioning sets, and tensor shapes at every module boundary, written
before any model code exists. Implements architecture plan §1.2, §2.1, §2.2.

Approved 2026-08-01. Every choice below is either fixed by the plan, fixed by D.1,
pre-registered here, or scheduled as a D.8 ablation — nothing is left to be settled
during implementation.

---

## 1. What the model computes

One network evaluating, for hitter `h` facing one pitch with context `c`:

```
p(swing | h, c)
p(contact | swing, h, c)
p(ev, la, spray | contact, h, c)
```

The third factorizes autoregressively (§1.5), so no joint grid is ever materialized:

```
p(ev, la, spray | ·) = p(ev | ·) · p(la | ev, ·) · p(spray | ev, la, ·)
```

Take outcomes (ball / called strike) are not predicted. They follow from plate
location, which is in `c`, and are composed in D.5's Markov chain — not here.

---

## 2. Shapes

`B` = batch size (pitches). Every quantity below is per-pitch.

| boundary | tensor | shape | dtype |
|---|---|---|---|
| **input** hitter index | `hitter` | `(B,)` | int64, range `[0, 1762]` |
| **input** context | `context` | `(B, 46)` | float32 |
| hitter embedding lookup | `e_h` | `(B, d)` | float32 |
| context MLP output | `z_c` | `(B, 128)` | float32 |
| trunk input (concat) | `[e_h ; z_c]` | `(B, d + 128)` | float32 |
| trunk output | `t` | `(B, 256)` | float32 |
| swing head | `logit_swing` | `(B,)` | float32 |
| contact head | `logit_contact` | `(B,)` | float32 |
| EV head | `logits_ev` | `(B, 24)` | float32 |
| LA head | `logits_la` | `(B, 24)` | float32 |
| spray head | `logits_spray` | `(B, 24)` | float32 |

Fixed by D.1: context width 46 (35 with the D.8 block removed), vocabulary 1,762
hitters plus index 0 reserved, 24 quantile bins per quality dimension.

---

## 3. Modules

### 3.1 Hitter tower

```
e_h = E[hitter]                E ∈ ℝ^(1763 × d),  d ∈ {16, 32, 64}, default 32
E[0] = 0  at init, and receives no gradient, ever
```

Row 0 is the reserved cold-start row (2026-07-30). Zero-initialized so an unseen
hitter and a generic hitter are the same point, and frozen so no batch can move it.
Rows 1…1762 are initialized `N(0, 0.01²)` and weight-decayed, which is where v1's
only shrinkage comes from (§5).

`d` is swept at D.8. Default 32 per §2.1.

### 3.2 Context tower

```
z_c = ReLU(W₂ · ReLU(W₁ c + b₁) + b₂)        W₁ ∈ ℝ^(128 × 46), W₂ ∈ ℝ^(128 × 128)
```

Two layers, hidden 128, per §2.1's "2-layer MLP (hidden ~64–128)". No dropout here;
the context vector is fully observed for most pitches and missingness already carries
explicit flags.

### 3.3 Trunk

```
t = Dropout₀.₁(ReLU(W₄ · ReLU(W₃ [e_h ; z_c] + b₃) + b₄))     hidden 256, 2 layers
```

Width and depth are pre-registered, not swept (2026-08-01): both sit inside §2.1's
stated ranges, and the first hidden layer is at least as wide as its 160-wide input.

**Ablation (§2.1, and the §8 mitigation for the interaction-learning risk):** add a
bilinear term `e_hᵀ W_b z_c`, projected to the trunk's width. Toggled at D.8; the
failure mode it targets is the model learning hitter main effects plus context main
effects and no interaction, which would make every platoon query return the league
average context penalty.

### 3.4 Heads

```
logit_swing   = w_s · t + b_s
logit_contact = w_k · t + b_k
logits_ev     = W_e · t + b_e
logits_la     = W_l · [t ; onehot(ev_bin)] + b_l
logits_spray  = W_p · [t ; onehot(ev_bin) ; onehot(la_bin)] + b_p
```

Ordering EV → LA → spray follows §1.5. At training time the conditioning bins are
the **observed** ones (teacher forcing). At inference the chain is sampled, not
enumerated — see §6.

---

## 4. Loss

```
L = Σ_all rows BCE(logit_swing, swing)
  + Σ_swings  BCE(logit_contact, contact)
  + Σ_bip     CE(logits_ev, ev_bin)
  + Σ_bip     CE(logits_la, la_bin)
  + Σ_spray   CE(logits_spray, spray_bin)
```

**Raw sums, no per-head or inverse-frequency weighting.** Masks come from D.1's
`MASKED` sentinel; a masked row contributes nothing to its factor, and `MASKED = -1`
is never a valid class index.

This is the plain likelihood under the §1.2 factorization, so each head's influence
tracks the evidence available to it. Two reasons it is the default rather than a
balanced alternative:

- **Calibration.** These conditionals feed D.5's Markov composition, which §5.4
  requires to reproduce actual league run scoring. Re-weighting the heads
  decalibrates them against the real pitch distribution, so the simulator would
  inherit a bias or need a correction factor concealing one.
- **Evidence.** B.1's stabilization ranking puts the high-count heads (swing ~31
  PA-equivalent, whiff ~28) ahead of the low-count ones (EV ~63, spray ~122), so
  row count and information per observation point the same way.

Row counts on the built table:

| head | rows | class balance |
|---|---|---|
| swing | 7,293,321 | 47.0% / 53.0% |
| contact | swings only | 23.2% whiff / 76.8% contact |
| EV | 1,266,036 | 24 equal-mass bins |
| LA | 1,266,309 | 24 equal-mass bins |
| spray | 1,239,195 | 24 equal-mass bins |

**D.8 ablation arms.** Per-head mean weighting (each factor averaged over its own
valid rows, then summed) is one arm. Inverse-frequency weighting inside the contact
head — the only head with real class imbalance — is the second, and §2.2's
inverse-frequency item reduces to exactly that. Both are expected null: they depart
from the likelihood, which is what the calibration argument above says not to do.

---

## 5. Training

| | |
|---|---|
| examples | one per pitch, seasons 2015–2023 (selection runs) |
| population | pitchers' own at-bats excluded (2026-08-01) |
| batch | **8,192 pitches**, shuffled across seasons and hitters |
| optimizer | **AdamW**, lr 1e-3, weight decay **1e-2** on all weights and the embedding, not biases |
| schedule | **ReduceLROnPlateau**, factor 0.3, patience 1, on 2024 validation loss |
| early stopping | 2024 validation loss, patience 3, keep the best checkpoint |
| ensemble | 5 seeds, identical architecture (§2.2) |
| refit | winning config retrained from scratch on 2015–2024 at the selected epoch count |
| report | 2025, once, via `--final-run` |

**Why AdamW rather than Adam.** Adam folds weight decay into the gradient before its
per-weight adaptive scaling, so weights with large gradient history receive less
decay. Here gradient magnitude varies with a hitter's exposure, so that coupling
would make shrinkage strength depend on exposure in an uncontrolled way. AdamW
decouples the decay, leaving the *number of updates* as the only thing exposure
changes — which is the honest version of "more data, less shrinkage."

**Why plateau rather than cosine.** Cosine annealing needs the total step count in
advance; early stopping means it is not known. Set it too high and the schedule never
anneals before stopping fires; too low and training crawls at the floor while
patience runs out. Plateau reads the same 2024 validation signal early stopping
reads, so the two compose without a guessed horizon. §2.2 permits either.

**Batch size and weight decay are one setting, not two.** AdamW decays every
parameter on every step, while an embedding row receives a gradient only in batches
containing that hitter. Shrinkage strength is the ratio, and batch size sets the step
count per epoch (718 at 8,192), so the two cannot be varied independently:

| exposure | train pitches | gradient steps / epoch | decay steps / epoch | ratio |
|---|---|---|---|---|
| 90th pct | 10,544 | ~718 | 718 | 1.0 : 1 |
| median | 1,244 | ~718 | 718 | 1.0 : 1 |
| 10th pct | 30 | ~30 | 718 | **23.9 : 1** |

That is the partial pooling: a hitter with 30 training pitches is decayed back toward
zero roughly 24× for every gradient that moves him, so his embedding stays near
generic. **Neither setting is swept** — sweeping either is sweeping shrinkage, which
the 2026-07-30 cold-start entry rejected on attribution grounds. D.7's `‖e_h‖`-vs-`n_h`
diagnostic is what tells us whether it worked.

A batch of 8,192 touches ~1,145 of the 1,762 hitters (65%).

**Parameter budget** at `d = 32`, block included:

| module | parameters |
|---|---|
| hitter embedding (1763 × 32) | 56,416 |
| context tower | 22,528 |
| trunk | 107,008 |
| five heads | 20,746 |
| **total** | **≈ 207k** |

At `d = 64`, ≈ 271k. Both comfortably inside §2.1's "well under 1M".

**First action in D.3:** benchmark one epoch before committing to 5 seeds × N
configs. The dataset is 1.6 GB on disk and memmapped, so RAM is not the constraint;
wall-clock is the unknown and §6 budgets minutes-to-hours.

---

## 6. What this spec does NOT cover

- **D.5 query machinery.** Composing per-pitch conditionals into side-specific wOBA
  — hierarchical pitcher sampling, the Markov chain over count states, the run-value
  mapping, and the §5.4 composition validation. Separate spec.
- **Sampling the quality chain at inference.** Enumerating the joint is 24³ = 13,824
  combinations per (hitter, context), prohibitive inside a Monte Carlo loop over
  hundreds of pitchers × tens of PAs. Sampling the chain once per simulated batted
  ball is the intended approach and is a D.5 decision, flagged here so it is not
  discovered late.
- **§5.1 probe checkpoint.** Runs on frozen embeddings after the first training run.

---

## 7. Decision status

| item | status |
|---|---|
| factorization, masking, autoregressive order | fixed by §1.2 / §1.5 |
| context width 46, 24 bins, reserved row 0, population | fixed by D.1 |
| loss = raw sums | pre-registered 2026-08-01 |
| trunk 2×256, context 2×128 | pre-registered 2026-08-01, not swept |
| AdamW, plateau, batch 8,192, decay 1e-2 | pre-registered 2026-08-01, not swept |
| embedding dim `d` | swept {16, 32, 64} at D.8, default 32 |
| bilinear interaction term | ablated at D.8 |
| B.2's flagged five | ablated at D.8 as one block |
| spray as a quality dimension | ablated at D.8 per §1.5 |
| per-head mean loss weighting | ablated at D.8, expected null |
| inverse-frequency within the contact head | ablated at D.8, expected null |
