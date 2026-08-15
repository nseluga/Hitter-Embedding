"""
Phase D.5 — composing per-pitch conditionals into side-specific wOBA (docs/phase-d5-spec.md).

v1 emits conditionals per pitch; the claim-1 metric wants one wOBA number per
(batter, pitcher hand). This module is the machinery between them, and it is exact
throughout: no plate appearances are simulated and no quality chain is sampled.

1. THE COUNT CHAIN IS SOLVED, NOT SIMULATED (2026-08-08). The pitch draw is independent
   of the pitches already thrown given the count, so the repertoire-averaged transition
   matrix is exactly Markov in count. Transitions only ever raise balls or strikes apart
   from the two-strike foul self-loop, which divides out in closed form as
   W(b,2) = A / (1 - P_foul). Backward induction over decreasing (b + s) therefore needs
   no matrix inversion, no iteration cap, and no truncation to report. Pitch SEQUENCING
   is the assumption this rests on and is out of scope (spec §4.4).

2. THE QUALITY CHAIN IS ENUMERATED, NOT SAMPLED (2026-08-08). head_la and head_spray are
   plain linear layers over [trunk ; onehot], so the conditioning bins enter as added
   weight columns and the joint logits are an outer sum.

   The naive form materialises (rows, 24, 24, 24) and softmaxes it. This module uses the
   FACTORED form instead, which is algebraically identical and ~24x cheaper. Writing
   u[m,s] = exp(base_spray[m,s]), a[e,s] = exp(E_s[e,s]), b[l,s] = exp(L_s[l,s]):

       R[m,e,l] = Σ_s u[m,s]·a[e,s]·b[l,s]·points[e,l,s]  /  Σ_s u[m,s]·a[e,s]·b[l,s]

   Both sums are one matmul against a (576, 24) matrix built ONCE, so the peak tensor is
   (rows, 576) rather than (rows, 13824) and the work lands in BLAS instead of in a
   memory-bound elementwise softmax. `expected_woba_naive` keeps the literal form as the
   thing the fast path is tested against; they must agree to float tolerance.

3. THE ENSEMBLE AVERAGES CONDITIONALS, NOT wOBA (2026-08-08). Averaging happens at the
   probability level before the composition, which is the deep-ensemble mixture for
   classification. R is LINEAR in p_spray, so averaging the seeds' R is exactly averaging
   their spray conditionals -- the 4-D joint is never materialised to do it.

4. EVERY TERMINAL STATE IS IN THE wOBA DENOMINATOR, so pred_woba = W(0,0) with no
   renormalisation. That only holds because intentional balls are excluded from the
   resampling pool upstream (query_tables.EXCLUDED_DESCRIPTIONS): IBB sits outside the
   denominator, so generating one would produce a plate appearance the metric never counts.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.analysis import claim1_eval
from src.data import eval_targets
from src.model import loader, query_tables as qt
from src.model.v1 import HitterEmbeddingV1

DEFAULT_DATA_DIR = "data/processed/phase_d"
DEFAULT_CHECKPOINT_DIR = "results/checkpoints"
DEFAULT_OUT_DIR = "results/phase_d"
DEFAULT_ARM = "d8_baseline"

# pre-registered knobs (spec §9). Set by benchmark, then frozen; never by their effect on
# the claim-1 metric, because claim-1 is produced BY this module.
# 0 means EVERY pitcher in the pool, which is the default: sampling a panel leaves a
# common level shift across all hitters (measured at 0.005 wOBA between panel draws at 60
# pitchers), and a level error is exactly what PA-weighted RMSE charges for. Taking the
# whole population removes that source outright rather than shrinking it.
DEFAULT_N_PITCHERS = 0
DEFAULT_N_PITCHES = 6
# pitchers per forward pass. The batch is (hitters x chunk x pitches) rows, so this trades
# memory against Python and per-call torch overhead; it cannot change a number.
DEFAULT_CHUNK_PITCHERS = 64

N_BALLS, N_STRIKES = 4, 3
COUNT_STATES = [(b, s) for b in range(N_BALLS) for s in range(N_STRIKES)]

# the four absorbing states of a plate appearance (spec §8). `bip` is the remainder, so the
# observed side cannot lose a plate appearance to an unrecognised event string.
ABSORBING_KEYS = ("bb", "k", "hbp", "bip")
ABSORBING_EVENTS = {"bb": ("walk",),
                    "k": ("strikeout", "strikeout_double_play"),
                    "hbp": ("hit_by_pitch",)}

# PRE-REGISTERED in the 2026-08-12 remediation plan, before the per-PA observed rates were
# computed (D5-R14): each rate within 2% relative of observed, HBP within 20%. HBP is ~1% of
# plate appearances and already runs 43% high per pitch, so a band tight enough to mean
# something on the other three is pure noise on it. A fix is CREDITED only if it moves a rate
# by more than the between-seed spread, which is why the per-seed compositions are persisted.
FIDELITY_TOLERANCE = {"bb": 0.02, "k": 0.02, "hbp": 0.20, "bip": 0.02}

_STARTED = time.monotonic()


def _progress(message):
    """
    Stderr, unbuffered, with elapsed wall time. `predict` over the full pitcher pool runs
    hours and every print in this module used to land after the last group finished, so a
    crash returned a zero-byte log and no way to tell how far it had got.
    """
    print(f"[{time.monotonic() - _STARTED:7.1f}s] {message}", file=sys.stderr, flush=True)


def load_ensemble(checkpoint_paths, manifest, n_context):
    """
    Rebuild each seed's model from its own checkpoint and put it in eval mode.
    Architecture comes from the checkpoint's recorded args, never from the filename.
    Only embedding_dim and bilinear are read: older checkpoints predate the other keys.
    """
    models = []
    for path in checkpoint_paths:
        # weights_only=False is required -- d8_invfreq stores a tensor inside args
        saved = torch.load(path, map_location="cpu", weights_only=False)
        args = saved["args"]
        # `split` and `spray` postdate the first 39 runs, so they are read with defaults
        # matching the pre-2026-08-08 architecture rather than assumed present
        model = HitterEmbeddingV1(manifest["n_hitters"], n_context,
                                  embedding_dim=args["embedding_dim"],
                                  n_bins=manifest["n_quality_bins"],
                                  bilinear=args.get("bilinear", False),
                                  split=args.get("split", False),
                                  spray=not args.get("no_spray", False))
        model.load_state_dict(saved["model"])
        # the trunk ends in Dropout(0.1) and forward() does not set mode, so a missing
        # eval() here would inject noise into every number this module produces
        model.eval()
        models.append(model)
    assert models, "no checkpoints given"
    return models


def _trunk(model, hitter, context):
    embedded = model.embedding(hitter)
    encoded = model.context_tower(context)
    trunk = model.trunk(torch.cat([embedded, encoded], dim=1))
    if model.interaction is not None:
        trunk = trunk + model.interaction(model.project_hitter(embedded)
                                          * model.project_context(encoded))
    return trunk


def league_spray_ratio(points, spray_mass):
    """
    E[wOBA points | ev, la] under the TRAINING spray distribution, as (K_e, K_l).

    The `nospray` arm predicts no spray conditional, so the (K,K,K) points table has to be
    collapsed with a distribution from somewhere. The only defensible one is the empirical
    P(spray | ev, la) on the same train balls `V` itself is fit on -- a uniform prior would
    be plainly wrong, since spray depends strongly on launch angle. So the arm is scored as
    "baseline minus the spray head, plus a league spray prior", which is the ablation
    question the arm was built to ask.

    spray_mass: (K,K,K) raw train cell counts from `fit_outcome_table`. An (ev, la) cell with
    no observations backs off to the global spray marginal rather than dividing by zero.
    """
    mass = spray_mass.to(points.dtype)
    total = mass.sum(dim=-1, keepdim=True)                             # (K_e, K_l, 1)
    globally = mass.sum(dim=(0, 1))
    globally = globally / globally.sum()
    conditional = torch.where(total > 0, mass / total.clamp(min=1.0), globally)
    assert torch.allclose(conditional.sum(dim=-1), torch.ones(())), \
        "P(spray | ev, la) must be a distribution over the spray axis"
    return (conditional * points).sum(dim=-1)                          # (K_e, K_l)


def spray_kernels(model, points, n_bins, spray_mass=None):
    """
    The (576, 24) matrices the factored spray sum contracts against, built once per model.
    points: (K, K, K) wOBA points per quality bin triple. Returns (weighted, plain), each
    (K*K, K), already max-shifted for stability -- constants across s cancel in R's ratio.

    For a model with no spray head returns (None, R) instead, where R is the hitter-
    INDEPENDENT (K_e, K_l) league ratio. `_conditionals` branches on `weighted is None`.
    """
    if model.head_spray is None:
        assert spray_mass is not None, \
            "an arm with no spray head needs tables['spray_mass'] to marginalise the points"
        return None, league_spray_ratio(points, spray_mass)

    hidden = model.head_spray.weight.shape[1] - 2 * n_bins
    column_ev = model.head_spray.weight[:, hidden:hidden + n_bins].T
    column_la = model.head_spray.weight[:, hidden + n_bins:].T

    logits = column_ev.unsqueeze(1) + column_la.unsqueeze(0)          # (K_e, K_l, K_s)
    logits = logits - logits.amax(dim=-1, keepdim=True)
    kernel = torch.exp(logits).reshape(n_bins * n_bins, n_bins)
    weighted = kernel * points.reshape(n_bins * n_bins, n_bins)
    return weighted, kernel


def _conditionals(model, trunk, weighted, kernel, n_bins):
    """
    One model's contribution to the averaged conditionals.
    Returns (p_ev (M,K), p_la (M,K,K), R (M,K,K)) where R is the spray-marginalised
    expected wOBA points given (ev bin, la bin). R is linear in the spray conditional,
    so averaging R across seeds is exactly averaging that conditional.

    weighted is None for an arm with no spray head; `kernel` is then the league (K_e, K_l)
    ratio already, the same for every row, and there is no spray forward pass to run.
    """
    hidden = trunk.shape[1]
    p_ev = torch.softmax(model.head_ev(trunk), dim=1)

    base_la = trunk @ model.head_la.weight[:, :hidden].T + model.head_la.bias
    column_la = model.head_la.weight[:, hidden:].T                     # (K_e, K_l)
    la_logits = base_la.unsqueeze(1) + column_la.unsqueeze(0)          # (M, K_e, K_l)
    p_la = torch.softmax(la_logits, dim=-1)

    if weighted is None:
        return p_ev, p_la, kernel.expand(p_ev.shape[0], n_bins, n_bins)

    base_spray = (trunk @ model.head_spray.weight[:, :hidden].T + model.head_spray.bias)
    scaled = torch.exp(base_spray - base_spray.amax(dim=1, keepdim=True))   # (M, K_s)
    numerator = scaled @ weighted.T                                    # (M, K_e*K_l)
    denominator = scaled @ kernel.T
    ratio = (numerator / denominator).reshape(-1, n_bins, n_bins)
    return p_ev, p_la, ratio


def expected_woba(models, hitter, context, points, n_bins, kernels=None):
    """
    E[wOBA points | in play, hitter, pitch] plus the swing and contact probabilities.
    Conditionals are averaged across the ensemble BEFORE composition (spec §6).
    Returns (p_swing (M,), p_contact (M,), q (M,)) as float64 numpy arrays.
    """
    kernels = kernels or [spray_kernels(model, points, n_bins) for model in models]
    swing = contact = p_ev = p_la = ratio = split = None
    with torch.no_grad():
        for model, (weighted, kernel) in zip(models, kernels):
            trunk = _trunk(model, hitter, context)
            one_ev, one_la, one_ratio = _conditionals(model, trunk, weighted, kernel, n_bins)
            one_swing = torch.sigmoid(model.head_swing(trunk).squeeze(-1))
            one_contact = torch.sigmoid(model.head_contact(trunk).squeeze(-1))
            if model.head_split is not None:
                one_split = torch.softmax(model.head_split(trunk), dim=1)
                split = one_split if split is None else split + one_split
            if swing is None:
                swing, contact = one_swing, one_contact
                p_ev, p_la, ratio = one_ev, one_la, one_ratio
            else:
                swing, contact = swing + one_swing, contact + one_contact
                p_ev, p_la, ratio = p_ev + one_ev, p_la + one_la, ratio + one_ratio

        n = len(models)
        swing, contact = swing / n, contact / n
        p_ev, p_la, ratio = p_ev / n, p_la / n, ratio / n
        quality = torch.einsum("me,mel,mel->m", p_ev, p_la, ratio)
        split = None if split is None else (split / n).double().numpy()
    return (swing.double().numpy(), contact.double().numpy(),
            quality.double().numpy(), split)


def expected_woba_naive(models, hitter, context, points, n_bins, spray_mass=None):
    """
    The literal spec §5.1 form, materialising the full (M, K, K, K) joint.
    Exists only as the reference `expected_woba` is tested against -- it is the
    algebra the fast path claims to compute, written the obvious way.

    For an arm with no spray head the joint's spray axis is the league conditional
    P(spray | ev, la), broadcast over rows, rather than a predicted softmax.
    """
    with torch.no_grad():
        total = None
        for model in models:
            trunk = _trunk(model, hitter, context)
            hidden = trunk.shape[1]
            p_ev = torch.softmax(model.head_ev(trunk), dim=1)
            base_la = trunk @ model.head_la.weight[:, :hidden].T + model.head_la.bias
            p_la = torch.softmax(base_la.unsqueeze(1)
                                 + model.head_la.weight[:, hidden:].T.unsqueeze(0), dim=-1)
            if model.head_spray is None:
                ratio = league_spray_ratio(points, spray_mass)
                p_spray = None
            else:
                base_sp = (trunk @ model.head_spray.weight[:, :hidden].T
                           + model.head_spray.bias)
                column_ev = model.head_spray.weight[:, hidden:hidden + n_bins].T
                column_la = model.head_spray.weight[:, hidden + n_bins:].T
                joint = (base_sp[:, None, None, :] + column_ev[None, :, None, :]
                         + column_la[None, None, :, :])
                p_spray = torch.softmax(joint, dim=-1)
                ratio = torch.einsum("mels,els->mel", p_spray, points)
            contribution = (p_ev, p_la, ratio.expand(p_ev.shape[0], n_bins, n_bins))
            total = contribution if total is None else tuple(
                a + b for a, b in zip(total, contribution))
        p_ev, p_la, ratio = (t / len(models) for t in total)
        return torch.einsum("me,mel,mel->m", p_ev, p_la, ratio).double().numpy()


def solve_chain(aggregates, w_bb, w_hbp, w_k=0.0):
    """
    Expected wOBA points of a plate appearance, by backward induction over the 12 counts.
    aggregates: dict of (..., 4, 3) arrays -- ball, strike, hbp, foul, and the wOBA-points
    contribution of balls in play. Vectorised over any leading axes, normally hitters.
    Returns W with the same shape; W[..., 0, 0] is the answer.

    Order matters and is not incidental: every dependency is on a higher ball or strike
    count, so descending strikes then descending balls means nothing is read before it is
    written. W(3, 2) is the base case and depends on no other state.

    w_k is the payoff of the absorbing strikeout and defaults to 0, which is what the wOBA
    call wants -- a strikeout scores no points. `absorbing_rates` sets it to 1 to read the
    strikeout PROBABILITY out of the same recursion instead of writing a second one.
    """
    lead = aggregates["ball"].shape[:-2]
    W = np.zeros(lead + (N_BALLS, N_STRIKES), dtype="float64")
    for strikes in reversed(range(N_STRIKES)):
        for balls in reversed(range(N_BALLS)):
            ball_value = w_bb if balls == N_BALLS - 1 else W[..., balls + 1, strikes]
            strike_value = w_k if strikes == N_STRIKES - 1 else W[..., balls, strikes + 1]
            fixed = (aggregates["ball"][..., balls, strikes] * ball_value
                     + aggregates["strike"][..., balls, strikes] * strike_value
                     + aggregates["hbp"][..., balls, strikes] * w_hbp
                     + aggregates["bip_points"][..., balls, strikes])
            foul = aggregates["foul"][..., balls, strikes]
            if strikes == N_STRIKES - 1:
                # the self-loop is a geometric series and divides out exactly
                assert np.all(foul < 1.0), "a two-strike foul probability of 1 never ends"
                W[..., balls, strikes] = fixed / (1.0 - foul)
            else:
                W[..., balls, strikes] = fixed + foul * W[..., balls, strikes + 1]
    return W


def absorbing_rates(aggregates):
    """
    P(the plate appearance ends in each of BB, K, HBP, BIP), shaped like solve_chain's W
    (spec §8). This is the block D5-R14 says the sanctioned check never had: a walk or
    strikeout distortion is invisible in a wOBA level, because wBB and wHBP are close enough
    that trading one for the other barely moves the total.

    Reuses solve_chain with indicator payoffs substituted for wOBA points, so the rates come
    out of the identical transition algebra as the wOBA number rather than from a second
    implementation that could drift from it. The four states are exhaustive by construction,
    so their sum is a free check that no probability mass leaks out of the chain.
    """
    zero = np.zeros_like(aggregates["bip_points"])
    #      key    w_bb  w_hbp  w_k   bip_points
    payoffs = {"bb": (1.0, 0.0, 0.0, zero),
               "k": (0.0, 0.0, 1.0, zero),
               "hbp": (0.0, 1.0, 0.0, zero),
               "bip": (0.0, 0.0, 0.0, aggregates["bip"])}
    rates = {key: solve_chain({**aggregates, "bip_points": bip_points}, w_bb, w_hbp, w_k=w_k)
             for key, (w_bb, w_hbp, w_k, bip_points) in payoffs.items()}
    # 1e-6 matches what the per-count masses are actually guaranteed to in `_group_woba`; a
    # tighter band here would be asserting more precision than the input carries
    assert np.allclose(sum(rates.values()), 1.0, atol=1e-6), \
        "PA absorbing rates do not sum to 1 -- mass is leaking out of the count chain"
    return rates


def observed_absorbing_rates(pa_df, seasons):
    """
    The observed share of plate appearances ending in each absorbing state, over `seasons`
    and restricted to the wOBA denominator -- the population the chain covers. Returns
    (rates, n_pa).

    Intentional walks, sacrifice hits and interference are already outside `in_denominator`,
    which is what lets these four exhaust it; `bip` is taken as the remainder rather than
    enumerated, so a new Statcast event string cannot silently fall out of the total.

    Committed to the repository per standards §6: D5-R1 quoted reference rates that were not
    reproducible from the repo, which is how the check ran for a phase with no pass condition.
    """
    window = pa_df[np.isin(pa_df["season"].to_numpy(), seasons)
                   & pa_df["in_denominator"].to_numpy()]
    assert len(window), f"no plate appearances in the denominator for seasons {seasons}"
    events = window["events"].to_numpy()
    counts = {key: int(np.isin(events, names).sum())
              for key, names in ABSORBING_EVENTS.items()}
    counts["bip"] = len(window) - sum(counts.values())
    assert counts["bip"] > 0, "every plate appearance absorbed without a ball in play"
    return {key: counts[key] / len(window) for key in ABSORBING_KEYS}, int(len(window))


def league_fidelity(predictions, pa_df, manifest, observed, eval_season):
    """
    The SCORED §8 league-fidelity check (D5-R14): PA-weighted absorbing rates over trained
    hitters, against `observed`, with the pre-registered pass condition applied.

    Cold-start hitters are excluded. They all share the reserved zero row, so including them
    would score one embedding a few hundred times over and call it a population -- the same
    mistake that makes `league_composition` unable to detect a trained row's bias.

    Population caveat, stated rather than fixed: the hitters are `eval_season`'s, because
    those are the ones `predict` already scored, while the pitcher pool, the auxiliary tables
    and the embeddings are all train-window. `observed` is therefore reported for both
    windows and only the train window is the pass target, per spec §8's reason for comparing
    against the train window at all.
    """
    trained = {int(batter) for batter, row in manifest["vocabulary"].items() if int(row) != 0}
    counts = (pa_df[(pa_df["season"] == eval_season) & pa_df["in_denominator"]]
              .groupby(["batter", "p_throws"]).size().rename("pa").reset_index())
    frame = predictions.merge(counts, on=["batter", "p_throws"], how="inner")
    frame = frame[frame["batter"].isin(trained)]
    assert len(frame), "no trained hitter survived the plate-appearance join"

    weight = frame["pa"].to_numpy(dtype="float64")
    rates = {}
    for key in ABSORBING_KEYS:
        modelled = float(np.average(frame[f"rate_{key}"].to_numpy(), weights=weight))
        relative = modelled / observed[key] - 1.0
        rates[key] = {"modelled": modelled, "observed": observed[key],
                      "relative_error": relative, "tolerance": FIDELITY_TOLERANCE[key],
                      "pass": bool(abs(relative) <= FIDELITY_TOLERANCE[key])}
    return {"n_hitters": int(len(frame)), "n_pa": int(weight.sum()), "rates": rates,
            "pass": all(rate["pass"] for rate in rates.values())}


def _panel(slots, probability, n_pitchers, generator):
    """
    The pitchers a hand's hitters are scored against, with their batters-faced weights.
    n_pitchers <= 0 or >= the pool takes everyone, which is the default and makes the
    result independent of the draw.
    """
    if n_pitchers <= 0 or n_pitchers >= len(slots):
        return slots, probability
    chosen = generator.choice(len(slots), size=n_pitchers, replace=False, p=probability)
    return slots[chosen], probability[chosen] / probability[chosen].sum()


def _stand_lookup(pa_df, eval_season):
    """
    Which side of the plate each hitter stands on, per (batter, pitcher hand).
    Keyed on the PAIR, following c3_gbm.py:136 -- a per-batter mode would label every
    switch hitter right-handed, since they face right-handed pitching far more often.
    """
    window = pa_df[pa_df["season"] == eval_season]
    modes = (window.groupby(["batter", "p_throws"])["stand"]
             .agg(lambda values: values.mode().iat[0]))
    return {(int(batter), hand): stand for (batter, hand), stand in modes.items()}


def _contact_split(tables, frame, pitch_rows, split_head=None):
    """
    p(foul, foul_tip, in_play | contact) per pitch, (n_pitches, 3).
    The league table indexed by (balls, strikes, in_zone) until split_head is supplied,
    at which point the trained three-class head replaces it (spec §2.1).
    """
    if split_head is not None:
        return split_head(pitch_rows)
    subset = frame.iloc[pitch_rows]
    in_zone = (subset["zone"].to_numpy() <= 9).astype("int64")
    return tables["split"][subset["balls"].to_numpy(), subset["strikes"].to_numpy(),
                           in_zone]


def _cell_aggregates(models, kernels, tensors, frame, tables, points, n_bins,
                     hitter_rows, pitch_rows, split_head, use_league_split=False,
                     measured_share=1.0, unmeasured_value=0.0):
    """
    One (pitcher, stand, count) cell scored against every hitter in the group.
    Returns a dict of (n_hitters,) arrays -- the per-state outcome masses and the
    wOBA-points contribution of balls in play.

    bip_points is accumulated in its own term rather than as mass x mean quality: a pitch
    more likely to be put in play is not the pitch with average contact quality, and
    factoring the two apart would silently discard that covariance.
    """
    n_hitters, n_pitches = len(hitter_rows), len(pitch_rows)
    hitter = torch.from_numpy(np.repeat(hitter_rows, n_pitches))
    context = tensors["context"][torch.from_numpy(np.tile(pitch_rows, n_hitters))]

    p_swing, p_contact, quality, model_split = expected_woba(
        models, hitter, context, points, n_bins, kernels)
    shape = (n_hitters, n_pitches)
    p_swing, p_contact = p_swing.reshape(shape), p_contact.reshape(shape)
    quality = quality.reshape(shape)

    take = qt.take_probabilities(tables["take"], frame, pitch_rows,
                                 tables.get("take_count_offsets"))          # (P, 3)
    # the trained head depends on the HITTER as well as the pitch, so it is (H, P, 3)
    # where the league table is (P, 3); `...` indexing below serves both by broadcasting
    if model_split is not None and not use_league_split:
        split = model_split.reshape(n_hitters, n_pitches, 3)
    else:
        split = _contact_split(tables, frame, pitch_rows, split_head)        # (P, 3)
    take_mass, contact_mass = 1.0 - p_swing, p_swing * p_contact
    in_play = contact_mass * split[..., 2]

    # returned UNREDUCED, one column per pitch: the caller reshapes the pitch axis into
    # (pitcher, pitch) and means over pitches, which is what lets one forward pass cover
    # a whole chunk of pitchers instead of one call each
    # D5-R15: `quality` is E[points | in play AND Statcast measured it], because that is the
    # subset `V` was fit on. The population's expectation splits over the measured share, which
    # is a fixed league rate; `measured_share=1.0` reproduces the pre-fix number exactly.
    expected_points = measured_share * quality + (1.0 - measured_share) * unmeasured_value

    return {
        "ball": take_mass * take[:, 0],
        "strike": (take_mass * take[:, 1] + p_swing * (1.0 - p_contact)
                   + contact_mass * split[..., 1]),
        "hbp": take_mass * take[:, 2],
        "foul": contact_mass * split[..., 0],
        "bip": in_play,
        "bip_points": in_play * expected_points,
    }


AGGREGATE_KEYS = ("ball", "strike", "hbp", "foul", "bip", "bip_points")


def _sample_grid(repertoire, slots, stand_slot, n_pitches, generator):
    """
    Pitch rows for every (pitcher, count) cell at once: (n_pitchers, 12, n_pitches).
    Also returns which pitchers are usable and how often backoff fired. A pitcher who
    never faced this stand at some count cannot finish a plate appearance, so he is
    marked unusable rather than scored with an absorbing hole in his chain.
    """
    grid = np.zeros((len(slots), len(COUNT_STATES), n_pitches), dtype="int64")
    usable = np.ones(len(slots), dtype=bool)
    levels = np.zeros(4, dtype="int64")
    for index, slot in enumerate(slots):
        for position, (balls, strikes) in enumerate(COUNT_STATES):
            picked, level = repertoire.sample(slot, stand_slot, balls, strikes,
                                              n_pitches, generator)
            levels[level] += 1
            if len(picked) == 0:
                usable[index] = False
                break
            grid[index, position] = picked
    return grid, usable, levels


def _group_woba(models, kernels, tensors, frame, tables, points, n_bins, hitter_rows,
                grid, pitcher_weights, split_head, chunk, w_bb, w_hbp,
                use_league_split=False, measured_share=1.0, unmeasured_value=0.0,
                progress=None):
    """
    Batters-faced-weighted mean of W(0,0) over the pitchers in `grid`, per hitter.
    Pitchers are processed in chunks so one forward pass covers many of them: the batch is
    (n_hitters x chunk x n_pitches) rows, which is where the arithmetic actually lives.
    Returns (total, used_weight, absorbing) so a partial panel renormalises rather than
    under-counts. `absorbing` carries the same weighted sums for the four §8 absorbing rates:
    they ride on the forward passes the wOBA number already paid for, so the fidelity check
    costs four extra 12-state backward inductions per chunk and nothing else.
    """
    n_hitters = len(hitter_rows)
    n_pitchers, n_states, n_pitches = grid.shape
    total, used = np.zeros(n_hitters, dtype="float64"), 0.0
    absorbing = {key: np.zeros(n_hitters, dtype="float64") for key in ABSORBING_KEYS}

    for start in range(0, n_pitchers, chunk):
        stop = min(start + chunk, n_pitchers)
        block = grid[start:stop]                                   # (C, 12, X)
        width = stop - start
        aggregates = {key: np.zeros((n_hitters, width, n_states)) for key in AGGREGATE_KEYS}

        for position in range(n_states):
            rows = block[:, position, :].reshape(-1)               # (C * X,)
            cell = _cell_aggregates(models, kernels, tensors, frame, tables, points,
                                    n_bins, hitter_rows, rows, split_head,
                                    use_league_split, measured_share, unmeasured_value)
            for key in AGGREGATE_KEYS:
                aggregates[key][:, :, position] = cell[key].reshape(
                    n_hitters, width, n_pitches).mean(axis=2)

        # COUNT_STATES is ordered balls-major, so position = balls * N_STRIKES + strikes
        shaped = {key: value.reshape(n_hitters, width, N_BALLS, N_STRIKES)
                  for key, value in aggregates.items()}
        mass = sum(shaped[key] for key in ("ball", "strike", "hbp", "foul", "bip"))
        assert np.allclose(mass, 1.0, atol=1e-6), \
            "per-count outcome masses do not sum to 1"

        solved = solve_chain(shaped, w_bb, w_hbp)                   # (H, C, 4, 3)
        weights = pitcher_weights[start:stop]
        total += solved[:, :, 0, 0] @ weights
        for key, rate in absorbing_rates(shaped).items():
            absorbing[key] += rate[:, :, 0, 0] @ weights
        used += float(weights.sum())
        if progress is not None:
            progress(stop, n_pitchers)
    return total, used, absorbing


def build_tables(frame, tensors, manifest, pa_df, take_count_offsets=False):
    """
    The four auxiliary tables, all fit on train seasons only (spec §2).
    Returns a dict consumed by `predict`, plus the coverage numbers standards §6 wants
    reported wherever a result built on them is reported.
    """
    train_seasons = manifest["train_seasons"]
    train_mask = np.isin(frame["season"].to_numpy(), train_seasons)
    bins = {name: tensors[name].numpy() for name in ("ev", "la", "spray")}

    surfaces, take_alpha = qt.fit_take_surfaces(frame, train_mask)
    outcome, n_joined, unmeasured, spray_mass = qt.fit_outcome_table(
        frame, bins, train_mask, pa_df, manifest["n_quality_bins"])
    repertoire = qt.build_repertoire(frame, train_mask)
    scored = int((train_mask & (bins["ev"] != -1) & (bins["la"] != -1)
                  & (bins["spray"] != -1)).sum())
    return {
        "take": surfaces,
        # None reproduces the count-blind surface exactly, so D5-R1's delta stays measurable
        "take_count_offsets": (qt.fit_take_count_offsets(surfaces, frame, train_mask)
                               if take_count_offsets else None),
        "split": qt.fit_contact_split(frame, train_mask),
        "outcome": outcome,
        "unmeasured": unmeasured,
        # raw train cell counts, unshrunk -- only the `nospray` arm reads them
        "spray_mass": torch.from_numpy(spray_mass.astype("float64")),
        "repertoire": repertoire,
        "pitcher_weights": qt.pitcher_weights(pa_df, train_seasons, repertoire),
        "coverage": {"take_alpha": take_alpha, "bip_joined": int(n_joined),
                     "bip_available": int(scored), "pitchers": int(repertoire.n_pitchers),
                     "measured_share": unmeasured["measured_share"],
                     "n_unmeasured_bip": unmeasured["n_unmeasured"]},
    }


def _unmeasured_terms(tables, weights, enabled):
    """
    (measured_share, unmeasured_points) for the composition, or the inert pair when disabled.

    Disabled reproduces the pre-fix number exactly -- a share of 1.0 makes the branch vanish
    algebraically rather than approximately -- which is what makes the D5-R15 delta measurable
    instead of asserted.
    """
    if not enabled or "unmeasured" not in tables:
        return 1.0, 0.0
    return float(tables["unmeasured"]["measured_share"]), unmeasured_points(tables, weights)


def unmeasured_points(tables, weights):
    """
    E[wOBA points | ball in play, Statcast did not measure it], under one season's weights.

    `V` is fit only on measured balls, so `quality` from the heads estimates
    E[points | in play AND measured]. The composition needs E[points | in play], which splits as

        measured_share * quality  +  (1 - measured_share) * this

    -- one scalar and one branch, no new parameters and no change to `V`'s 24^3 shape (D5-R15).
    """
    from src.data.eval_targets import CATEGORY_WEIGHT_KEY

    values = np.array([weights.get(CATEGORY_WEIGHT_KEY[name], 0.0)
                       if name in CATEGORY_WEIGHT_KEY else 0.0 for name in qt.BIP_CLASSES])
    return float(np.asarray(tables["unmeasured"]["categories"]) @ values)


def predict(models, tensors, manifest, frame, tables, pa_df, eval_season,
            n_pitchers=DEFAULT_N_PITCHERS, n_pitches=DEFAULT_N_PITCHES, seed=0,
            split_head=None, chunk=DEFAULT_CHUNK_PITCHERS, use_league_split=False,
            unmeasured_split=True, batters=None):
    """
    One pred_woba per (batter, p_throws) active in eval_season (spec §7), plus the four §8
    per-PA absorbing rates the league-fidelity check aggregates.
    tables: the dict returned by `build_tables`. split_head: an optional callable giving
    the three-class contact split per pitch, replacing the league table once the head is
    trained. Returns (predictions DataFrame, diagnostics dict).

    `batters`: restrict to these batter ids. For the perturb-and-re-solve diagnostics
    (D5-R15 gradient test (c), and pricing D5-R1's light fix), where the quantity wanted is a
    DELTA at a handful of hitters and a full pass costs hours. The pitcher pool is untouched,
    so both sides of a delta are drawn against the same panel -- restricting pitchers instead
    would leave panel noise inside the difference.
    """
    weights = eval_targets.load_weights()[str(eval_season)]
    points = torch.from_numpy(qt.woba_points_table(tables["outcome"], weights)).float()
    share, unmeasured = _unmeasured_terms(tables, weights, unmeasured_split)
    vocabulary = {int(batter): row for batter, row in manifest["vocabulary"].items()}
    n_bins = manifest["n_quality_bins"]
    kernels = [spray_kernels(model, points, n_bins, tables["spray_mass"]) for model in models]

    rows = (eval_targets.drop_pitcher_batters(pa_df)
            .query("season == @eval_season")[["batter", "p_throws"]].drop_duplicates())
    if batters is not None:
        wanted = {int(batter) for batter in batters}
        rows = rows[rows["batter"].isin(wanted)]
        # a silently dropped id would shrink a paired delta's population without saying so,
        # and the pairing is the whole point of the diagnostic
        assert set(rows["batter"].astype(int)) == wanted, \
            f"requested batters absent from {eval_season}: {sorted(wanted - set(rows['batter']))}"
    stand_of = _stand_lookup(pa_df, eval_season)
    generator = np.random.default_rng(seed)
    repertoire, pool = tables["repertoire"], tables["pitcher_weights"]
    panels = {hand: _panel(slots, probability, n_pitchers, generator)
              for hand, (slots, probability) in pool.items()}

    predictions, backoff, dropped = [], np.zeros(4, dtype="int64"), 0
    # the repertoire is keyed on the BATTER's stand, so hitters standing on different
    # sides cannot share a draw; (hand, stand) is therefore the unit of work
    for hand in sorted(rows["p_throws"].unique()):
        slots, panel_weights = panels[hand]
        side = rows[rows["p_throws"] == hand]
        batters = side["batter"].to_numpy()
        stands = np.array([stand_of.get((int(b), hand), "R") for b in batters])
        totals = np.zeros(len(batters), dtype="float64")
        rates = {key: np.zeros(len(batters), dtype="float64") for key in ABSORBING_KEYS}

        for stand in ("L", "R"):
            group = stands == stand
            if not group.any():
                continue
            hitter_rows = np.array([vocabulary.get(int(b), 0) for b in batters[group]],
                                   dtype="int64")
            grid, usable, levels = _sample_grid(repertoire, slots,
                                                1 if stand == "R" else 0, n_pitches,
                                                generator)
            backoff += levels
            dropped += int((~usable).sum())
            assert usable.any(), f"no usable pitchers for {hand}HP against {stand}HB"

            weight_vector = panel_weights[usable]
            label = f"{stand}HB vs {hand}HP"
            _progress(f"{label}: {group.sum()} hitters x {int(usable.sum())} pitchers")
            total, used, absorbing = _group_woba(
                models, kernels, tensors, frame, tables, points, n_bins, hitter_rows,
                grid[usable], weight_vector, split_head, chunk, weights["wBB"],
                weights["wHBP"], use_league_split, share, unmeasured,
                progress=lambda done, n, label=label: _progress(
                    f"  {label}: {done}/{n} pitchers"))
            totals[group] = total / used
            for key in ABSORBING_KEYS:
                rates[key][group] = absorbing[key] / used

        predictions.append(pd.DataFrame(
            {"batter": batters, "season": eval_season, "p_throws": hand,
             "pred_woba": totals,
             **{f"rate_{key}": rates[key] for key in ABSORBING_KEYS}}))

    rate_columns = [f"rate_{key}" for key in ABSORBING_KEYS]
    out = pd.concat(predictions, ignore_index=True)[
        claim1_eval.KEY + ["pred_woba"] + rate_columns]
    assert out["pred_woba"].notna().all(), "a hitter received no prediction"
    assert np.isfinite(out["pred_woba"]).all(), "a prediction is not finite"
    assert np.allclose(out[rate_columns].to_numpy().sum(axis=1), 1.0, atol=1e-6), \
        "a hitter's absorbing rates do not sum to 1"
    diagnostics = {"backoff_levels": backoff.tolist(), "pitcher_cells_dropped": int(dropped),
                   "n_pitchers": int(n_pitchers), "n_pitches": int(n_pitches),
                   "rows": int(len(out)), "coverage": tables["coverage"],
                   "unmeasured_split": bool(unmeasured_split),
                   "batter_subset": None if batters is None else sorted(int(b) for b in batters),
                   "measured_share": share, "unmeasured_points": unmeasured,
                   "contact_split_source": "league_table" if (
                       use_league_split or models[0].head_split is None) else "trained_head"}
    return out, diagnostics


def handedness_shares(pa_df, seasons):
    """
    The (stand, p_throws) share of plate appearances in the wOBA denominator over `seasons`.
    D5-R14: the §5.4 check weighted these four cells at a flat 25% against 7.6 / 33.4 / 19.6 /
    39.3%, a -0.00271 wOBA error before the model contributed anything. Read from the data
    rather than pinned as constants, so a re-pull cannot leave a stale number behind.
    """
    window = pa_df[np.isin(pa_df["season"].to_numpy(), seasons)
                   & pa_df["in_denominator"].to_numpy()]
    counts = window.groupby(["stand", "p_throws"]).size()
    return (counts / counts.sum()).to_dict()


def league_composition(models, tensors, manifest, frame, tables, shares, eval_season,
                       n_pitchers=DEFAULT_N_PITCHERS, n_pitches=DEFAULT_N_PITCHES,
                       seed=0, chunk=DEFAULT_CHUNK_PITCHERS, unmeasured_split=True):
    """
    The §5.4 composition probe: the reserved zero row against the pooled opponent
    distribution. UNSCORED, deliberately (D5-R14). The zero row is not a hitter and sits
    0.0233 from the trained-hitter mean, so the sign of its gap carries no information about
    a trained row's bias -- it is kept only so the five arms already diagnosed against it stay
    comparable, and `league_fidelity` is the check that carries a pass condition.

    Compared against the TRAIN window, not the eval season: the pitcher pool is 2015-2023,
    so charging the simulator for league drift it was never given the data to see would be
    scoring it on the wrong question (spec §8).

    Reports per cell and then twice over: `league_pred_woba` weights the four (stand, hand)
    cells by their true share, and `league_pred_woba_uniform` keeps the flat 25% the five
    recorded arm values were produced under, so those stay checkable against a fresh run.
    """
    weights = eval_targets.load_weights()[str(eval_season)]
    points = torch.from_numpy(qt.woba_points_table(tables["outcome"], weights)).float()
    n_bins = manifest["n_quality_bins"]
    kernels = [spray_kernels(model, points, n_bins, tables["spray_mass"]) for model in models]
    share, unmeasured = _unmeasured_terms(tables, weights, unmeasured_split)
    generator = np.random.default_rng(seed)
    repertoire, pool = tables["repertoire"], tables["pitcher_weights"]
    generic = np.zeros(1, dtype="int64")

    cells = {}
    for hand, (slots, probability) in pool.items():
        panel, panel_weights = _panel(slots, probability, n_pitchers, generator)
        for stand_slot, stand in ((0, "L"), (1, "R")):
            grid, usable, _ = _sample_grid(repertoire, panel, stand_slot, n_pitches,
                                           generator)
            if not usable.any():
                continue
            weight_vector = panel_weights[usable]
            _progress(f"probe {stand}HB vs {hand}HP: {int(usable.sum())} pitchers")
            total, part, absorbing = _group_woba(
                models, kernels, tensors, frame, tables, points, n_bins, generic,
                grid[usable], weight_vector, None, chunk, weights["wBB"], weights["wHBP"],
                False, share, unmeasured)

            block = grid[usable]
            first = block[:, 0, :].reshape(-1)
            cell = _cell_aggregates(models, kernels, tensors, frame, tables, points,
                                    n_bins, generic, first, None, False, share, unmeasured)
            masses = {}
            for key in AGGREGATE_KEYS:
                per_pitcher = cell[key].reshape(1, usable.sum(), n_pitches).mean(axis=2)
                masses[f"mass_{key}_00"] = float(per_pitcher[0] @ weight_vector) / part

            cells[f"{stand}HB_vs_{hand}HP"] = {
                "share": float(shares[(stand, hand)]),
                "pred_woba": float(total[0]) / part,
                # per-PA absorbing rates -- the §8 quantity; `mass_*_00` below are per-PITCH
                # masses at the 0-0 count, which is what this check used to report instead
                **{f"rate_{key}": float(absorbing[key][0]) / part for key in ABSORBING_KEYS},
                **masses,
            }

    fields = (["pred_woba"] + [f"rate_{key}" for key in ABSORBING_KEYS]
              + [f"mass_{key}_00" for key in AGGREGATE_KEYS])
    share_weights = np.array([cell["share"] for cell in cells.values()])
    flat = np.ones(len(cells))

    def _mean(field, cell_weights):
        values = np.array([cell[field] for cell in cells.values()])
        return float(values @ cell_weights / cell_weights.sum())

    return {"league_pred_woba": _mean("pred_woba", share_weights),
            "league_pred_woba_uniform": _mean("pred_woba", flat),
            **{field: _mean(field, share_weights) for field in fields if field != "pred_woba"},
            "cells": cells}


def main():
    parser = argparse.ArgumentParser(description="Phase D.5 — compose conditionals to wOBA.")
    parser.add_argument("--arm", default=DEFAULT_ARM, help="checkpoint stem, seeds appended")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--eval-season", type=int, default=2024)
    parser.add_argument("--final-run", action="store_true",
                        help="the ONLY way 2025 is ever scored")
    parser.add_argument("--n-pitchers", type=int, default=DEFAULT_N_PITCHERS)
    parser.add_argument("--n-pitches", type=int, default=DEFAULT_N_PITCHES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--take-count-offsets", action="store_true",
                        help="D5-R1's light fix: 12 per-count raking factors on the four take "
                             "surfaces. Off by default so the count-blind number stays the "
                             "reference the fix is graded against")
    parser.add_argument("--no-unmeasured-split", action="store_true",
                        help="value every ball in play through V, reproducing the pre-D5-R15 "
                             "number. The two-sided check for that fix: league wOBA should fall "
                             "and NO absorbing rate should move, since it touches valuation "
                             "rather than transitions")
    parser.add_argument("--league-split", action="store_true",
                        help="force the league contact-split table even when the model "
                             "carries the trained head — the both-ways comparison")
    parser.add_argument("--batters", type=int, nargs="+", default=None,
                        help="restrict the composition to these batter ids -- the "
                             "perturb-and-re-solve diagnostics, where a full pass costs hours "
                             "for a delta at ~50 hitters. Suppresses the league fidelity check "
                             "and both probes, which are not defined on a subset")
    parser.add_argument("--label", default=None,
                        help="output stem; defaults to the arm name")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--pitch-events", default="data/processed/pitch_events_labeled.parquet")
    parser.add_argument("--eval-targets", default="data/processed/eval_targets_pa.parquet")
    args = parser.parse_args()

    claim1_eval.assert_not_test_season(args.eval_season, final_run=args.final_run)

    label = args.label or args.arm
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _progress("loading tensors and aligning the pitch frame")
    tensors, manifest = loader.load_tensors(args.data_dir)
    frame = qt.align_pitch_frame(args.pitch_events, args.eval_targets, tensors["season"])
    pa_df = pd.read_parquet(args.eval_targets)
    _progress("fitting the four auxiliary tables")
    tables = build_tables(frame, tensors, manifest, pa_df,
                          take_count_offsets=args.take_count_offsets)

    paths = [Path(args.checkpoint_dir) / f"{args.arm}_s{seed}.pt" for seed in args.seeds]
    models = load_ensemble(paths, manifest, tensors["context"].shape[1])

    shares = handedness_shares(pa_df, manifest["train_seasons"])
    observed, n_observed = observed_absorbing_rates(pa_df, manifest["train_seasons"])
    observed_eval, n_observed_eval = observed_absorbing_rates(pa_df, [args.eval_season])
    reference = {"train_seasons": manifest["train_seasons"], "n_pa": n_observed,
                 "rates": observed, "eval_season": args.eval_season,
                 "eval_n_pa": n_observed_eval, "eval_rates": observed_eval,
                 "handedness_shares": {f"{stand}HB_vs_{hand}HP": float(share)
                                       for (stand, hand), share in shares.items()},
                 "tolerance": FIDELITY_TOLERANCE}
    # written BEFORE `predict`, which runs hours. These are read-only aggregates over the
    # eval-target table and standards §6 wants them in the repository regardless of whether
    # the composition that follows survives to the end.
    (out_dir / "league_reference_rates.json").write_text(json.dumps(reference, indent=2))
    _progress(f"wrote {out_dir / 'league_reference_rates.json'}")

    predictions, diagnostics = predict(models, tensors, manifest, frame, tables, pa_df,
                                       args.eval_season, n_pitchers=args.n_pitchers,
                                       n_pitches=args.n_pitches, seed=args.seed,
                                       use_league_split=args.league_split,
                                       unmeasured_split=not args.no_unmeasured_split,
                                       batters=args.batters)
    # the CSV lands before the probe runs, for the same reason
    predictions.to_csv(out_dir / f"d5_predictions_{label}.csv", index=False)
    _progress(f"wrote {out_dir / f'd5_predictions_{label}.csv'}")

    if args.batters is not None:
        # a fidelity number computed over 50 hitters is not the league's, and a diagnostics file
        # that carries one under the same key as a full run's is how a subset number gets quoted
        # as a league number six weeks later
        (out_dir / f"d5_diagnostics_{label}.json").write_text(
            json.dumps({**diagnostics, "reference": reference, "arm": args.arm,
                        "seeds": args.seeds, "eval_season": args.eval_season}, indent=2))
        _progress(f"batter subset ({len(args.batters)} ids): fidelity and both probes skipped, "
                  f"they are not defined on a subset")
        print(f"arm: {args.arm} seeds: {args.seeds}")
        print(f"rows: {diagnostics['rows']}")
        print(f"pred_woba mean: {predictions['pred_woba'].mean():.4f}")
        return

    fidelity = league_fidelity(predictions, pa_df, manifest, observed, args.eval_season)
    composition = league_composition(models, tensors, manifest, frame, tables, shares,
                                     args.eval_season, n_pitchers=args.n_pitchers,
                                     n_pitches=args.n_pitches, seed=args.seed,
                                     unmeasured_split=not args.no_unmeasured_split)

    # spec §6 wants the per-seed compositions; they were computed and discarded, leaving no
    # between-seed spread on record -- which is the number the pass condition credits a fix
    # against. One hitter per run, so five extra runs are cheap next to `predict`.
    per_seed = {}
    for seed, path in zip(args.seeds, paths):
        _progress(f"per-seed composition, seed {seed}")
        single = league_composition(load_ensemble([path], manifest,
                                                 tensors["context"].shape[1]),
                                   tensors, manifest, frame, tables, shares,
                                   args.eval_season, n_pitchers=args.n_pitchers,
                                   n_pitches=args.n_pitches, seed=args.seed,
                                   unmeasured_split=not args.no_unmeasured_split)
        per_seed[str(seed)] = {key: value for key, value in single.items() if key != "cells"}
    spread = {key: (max(run[key] for run in per_seed.values())
                    - min(run[key] for run in per_seed.values()))
              for key in next(iter(per_seed.values()))} if per_seed else {}

    (out_dir / f"d5_diagnostics_{label}.json").write_text(
        json.dumps({**diagnostics, "composition": composition,
                    "composition_per_seed": per_seed, "composition_spread": spread,
                    "fidelity": fidelity, "reference": reference,
                    "arm": args.arm, "seeds": args.seeds,
                    "eval_season": args.eval_season}, indent=2))

    print(f"arm: {args.arm} seeds: {args.seeds}")
    print(f"rows: {diagnostics['rows']}")
    print(f"pred_woba mean: {predictions['pred_woba'].mean():.4f}")
    print(f"pred_woba sd: {predictions['pred_woba'].std():.4f}")
    print(f"backoff levels (exact, strikes, stand, empty): {diagnostics['backoff_levels']}")
    print(f"pitcher cells dropped: {diagnostics['pitcher_cells_dropped']}")
    print("\n-- unscored zero-row probe (§5.4) --")
    for name, value in composition.items():
        if name != "cells":
            print(f"{name}: {value:.4f}  (seed spread {spread.get(name, float('nan')):.4f})")
    print("\n-- SCORED league fidelity (§8), trained hitters, PA-weighted --")
    print(f"{fidelity['n_hitters']} hitters, {fidelity['n_pa']} plate appearances")
    for key in ABSORBING_KEYS:
        rate = fidelity["rates"][key]
        print(f"{key:>4}: {rate['modelled']:.5f} vs {rate['observed']:.5f} observed  "
              f"({rate['relative_error']:+.1%} of {rate['tolerance']:.0%} allowed)  "
              f"{'PASS' if rate['pass'] else 'FAIL'}")
    print(f"fidelity: {'PASS' if fidelity['pass'] else 'FAIL'}")
    coverage = diagnostics["coverage"]
    print(f"bip joined to a PA outcome: {coverage['bip_joined']} of "
          f"{coverage['bip_available']} "
          f"({coverage['bip_joined'] / coverage['bip_available']:.1%})")
    print(f"contact split from: {diagnostics['contact_split_source']}")
    print(f"wrote {out_dir / f'd5_predictions_{label}.csv'}")


if __name__ == "__main__":
    main()
