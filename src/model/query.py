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


def spray_kernels(model, points, n_bins):
    """
    The (576, 24) matrices the factored spray sum contracts against, built once per model.
    points: (K, K, K) wOBA points per quality bin triple. Returns (weighted, plain), each
    (K*K, K), already max-shifted for stability -- constants across s cancel in R's ratio.
    """
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
    """
    hidden = trunk.shape[1]
    p_ev = torch.softmax(model.head_ev(trunk), dim=1)

    base_la = trunk @ model.head_la.weight[:, :hidden].T + model.head_la.bias
    column_la = model.head_la.weight[:, hidden:].T                     # (K_e, K_l)
    la_logits = base_la.unsqueeze(1) + column_la.unsqueeze(0)          # (M, K_e, K_l)
    p_la = torch.softmax(la_logits, dim=-1)

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


def expected_woba_naive(models, hitter, context, points, n_bins):
    """
    The literal spec §5.1 form, materialising the full (M, K, K, K) joint.
    Exists only as the reference `expected_woba` is tested against -- it is the
    algebra the fast path claims to compute, written the obvious way.
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
            base_sp = (trunk @ model.head_spray.weight[:, :hidden].T
                       + model.head_spray.bias)
            column_ev = model.head_spray.weight[:, hidden:hidden + n_bins].T
            column_la = model.head_spray.weight[:, hidden + n_bins:].T
            joint = (base_sp[:, None, None, :] + column_ev[None, :, None, :]
                     + column_la[None, None, :, :])
            p_spray = torch.softmax(joint, dim=-1)
            contribution = (p_ev, p_la, torch.einsum("mels,els->mel", p_spray, points))
            total = contribution if total is None else tuple(
                a + b for a, b in zip(total, contribution))
        p_ev, p_la, ratio = (t / len(models) for t in total)
        return torch.einsum("me,mel,mel->m", p_ev, p_la, ratio).double().numpy()


def solve_chain(aggregates, w_bb, w_hbp):
    """
    Expected wOBA points of a plate appearance, by backward induction over the 12 counts.
    aggregates: dict of (..., 4, 3) arrays -- ball, strike, hbp, foul, and the wOBA-points
    contribution of balls in play. Vectorised over any leading axes, normally hitters.
    Returns W with the same shape; W[..., 0, 0] is the answer.

    Order matters and is not incidental: every dependency is on a higher ball or strike
    count, so descending strikes then descending balls means nothing is read before it is
    written. W(3, 2) is the base case and depends on no other state.
    """
    lead = aggregates["ball"].shape[:-2]
    W = np.zeros(lead + (N_BALLS, N_STRIKES), dtype="float64")
    for strikes in reversed(range(N_STRIKES)):
        for balls in reversed(range(N_BALLS)):
            ball_value = w_bb if balls == N_BALLS - 1 else W[..., balls + 1, strikes]
            strike_value = 0.0 if strikes == N_STRIKES - 1 else W[..., balls, strikes + 1]
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
                     hitter_rows, pitch_rows, split_head, use_league_split=False):
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

    take = qt.take_probabilities(tables["take"], frame, pitch_rows)          # (P, 3)
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
    return {
        "ball": take_mass * take[:, 0],
        "strike": (take_mass * take[:, 1] + p_swing * (1.0 - p_contact)
                   + contact_mass * split[..., 1]),
        "hbp": take_mass * take[:, 2],
        "foul": contact_mass * split[..., 0],
        "bip": in_play,
        "bip_points": in_play * quality,
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
                use_league_split=False):
    """
    Batters-faced-weighted mean of W(0,0) over the pitchers in `grid`, per hitter.
    Pitchers are processed in chunks so one forward pass covers many of them: the batch is
    (n_hitters x chunk x n_pitches) rows, which is where the arithmetic actually lives.
    Returns (total, used_weight) so a partial panel renormalises rather than under-counts.
    """
    n_hitters = len(hitter_rows)
    n_pitchers, n_states, n_pitches = grid.shape
    total, used = np.zeros(n_hitters, dtype="float64"), 0.0

    for start in range(0, n_pitchers, chunk):
        stop = min(start + chunk, n_pitchers)
        block = grid[start:stop]                                   # (C, 12, X)
        width = stop - start
        aggregates = {key: np.zeros((n_hitters, width, n_states)) for key in AGGREGATE_KEYS}

        for position in range(n_states):
            rows = block[:, position, :].reshape(-1)               # (C * X,)
            cell = _cell_aggregates(models, kernels, tensors, frame, tables, points,
                                    n_bins, hitter_rows, rows, split_head,
                                    use_league_split)
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
        used += float(weights.sum())
    return total, used


def build_tables(frame, tensors, manifest, pa_df):
    """
    The four auxiliary tables, all fit on train seasons only (spec §2).
    Returns a dict consumed by `predict`, plus the coverage numbers standards §6 wants
    reported wherever a result built on them is reported.
    """
    train_seasons = manifest["train_seasons"]
    train_mask = np.isin(frame["season"].to_numpy(), train_seasons)
    bins = {name: tensors[name].numpy() for name in ("ev", "la", "spray")}

    surfaces, take_alpha = qt.fit_take_surfaces(frame, train_mask)
    outcome, n_joined = qt.fit_outcome_table(frame, bins, train_mask, pa_df,
                                             manifest["n_quality_bins"])
    repertoire = qt.build_repertoire(frame, train_mask)
    scored = int((train_mask & (bins["ev"] != -1) & (bins["la"] != -1)
                  & (bins["spray"] != -1)).sum())
    return {
        "take": surfaces,
        "split": qt.fit_contact_split(frame, train_mask),
        "outcome": outcome,
        "repertoire": repertoire,
        "pitcher_weights": qt.pitcher_weights(pa_df, train_seasons, repertoire),
        "coverage": {"take_alpha": take_alpha, "bip_joined": int(n_joined),
                     "bip_available": int(scored), "pitchers": int(repertoire.n_pitchers)},
    }


def predict(models, tensors, manifest, frame, tables, pa_df, eval_season,
            n_pitchers=DEFAULT_N_PITCHERS, n_pitches=DEFAULT_N_PITCHES, seed=0,
            split_head=None, chunk=DEFAULT_CHUNK_PITCHERS, use_league_split=False):
    """
    One pred_woba per (batter, p_throws) active in eval_season (spec §7).
    tables: the dict returned by `build_tables`. split_head: an optional callable giving
    the three-class contact split per pitch, replacing the league table once the head is
    trained. Returns (predictions DataFrame, diagnostics dict).
    """
    weights = eval_targets.load_weights()[str(eval_season)]
    points = torch.from_numpy(qt.woba_points_table(tables["outcome"], weights)).float()
    vocabulary = {int(batter): row for batter, row in manifest["vocabulary"].items()}
    n_bins = manifest["n_quality_bins"]
    kernels = [spray_kernels(model, points, n_bins) for model in models]

    rows = (eval_targets.drop_pitcher_batters(pa_df)
            .query("season == @eval_season")[["batter", "p_throws"]].drop_duplicates())
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
            total, used = _group_woba(models, kernels, tensors, frame, tables, points,
                                      n_bins, hitter_rows, grid[usable], weight_vector,
                                      split_head, chunk, weights["wBB"], weights["wHBP"],
                                      use_league_split)
            totals[group] = total / used

        predictions.append(pd.DataFrame({"batter": batters, "season": eval_season,
                                         "p_throws": hand, "pred_woba": totals}))

    out = pd.concat(predictions, ignore_index=True)[claim1_eval.KEY + ["pred_woba"]]
    assert out["pred_woba"].notna().all(), "a hitter received no prediction"
    assert np.isfinite(out["pred_woba"]).all(), "a prediction is not finite"
    diagnostics = {"backoff_levels": backoff.tolist(), "pitcher_cells_dropped": int(dropped),
                   "n_pitchers": int(n_pitchers), "n_pitches": int(n_pitches),
                   "rows": int(len(out)), "coverage": tables["coverage"],
                   "contact_split_source": "league_table" if (
                       use_league_split or models[0].head_split is None) else "trained_head"}
    return out, diagnostics


def league_composition(models, tensors, manifest, frame, tables, eval_season,
                       n_pitchers=DEFAULT_N_PITCHERS, n_pitches=DEFAULT_N_PITCHES,
                       seed=0, chunk=DEFAULT_CHUNK_PITCHERS):
    """
    The §5.4 composition check: a GENERIC hitter against the pooled opponent distribution.
    Runs the reserved zero row, so the result is the simulator's league baseline rather
    than any particular hitter, and returns terminal-state rates beside the wOBA.

    Compared against the TRAIN window, not the eval season: the pitcher pool is 2015-2023,
    so charging the simulator for league drift it was never given the data to see would be
    scoring it on the wrong question (spec §8).
    """
    weights = eval_targets.load_weights()[str(eval_season)]
    points = torch.from_numpy(qt.woba_points_table(tables["outcome"], weights)).float()
    n_bins = manifest["n_quality_bins"]
    kernels = [spray_kernels(model, points, n_bins) for model in models]
    generator = np.random.default_rng(seed)
    repertoire, pool = tables["repertoire"], tables["pitcher_weights"]
    generic = np.zeros(1, dtype="int64")

    woba, used, rates = 0.0, 0.0, {key: 0.0 for key in AGGREGATE_KEYS}
    for hand, (slots, probability) in pool.items():
        panel, panel_weights = _panel(slots, probability, n_pitchers, generator)
        for stand_slot in (0, 1):
            grid, usable, _ = _sample_grid(repertoire, panel, stand_slot, n_pitches,
                                           generator)
            if not usable.any():
                continue
            # each (hand, stand) cell is a quarter of the league's plate appearances
            share = 0.25
            weight_vector = panel_weights[usable]
            total, part = _group_woba(models, kernels, tensors, frame, tables, points,
                                      n_bins, generic, grid[usable], weight_vector, None,
                                      chunk, weights["wBB"], weights["wHBP"])
            woba += share * float(total[0]) / part
            used += share

            block = grid[usable]
            first = block[:, 0, :].reshape(-1)
            cell = _cell_aggregates(models, kernels, tensors, frame, tables, points,
                                    n_bins, generic, first, None)
            for key in AGGREGATE_KEYS:
                per_pitcher = cell[key].reshape(1, usable.sum(), n_pitches).mean(axis=2)
                rates[key] += share * float(per_pitcher[0] @ weight_vector) / part
    return {"league_pred_woba": woba / used,
            **{f"rate_{key}_00": rates[key] / used for key in AGGREGATE_KEYS}}


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
    parser.add_argument("--league-split", action="store_true",
                        help="force the league contact-split table even when the model "
                             "carries the trained head — the both-ways comparison")
    parser.add_argument("--label", default=None,
                        help="output stem; defaults to the arm name")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--pitch-events", default="data/processed/pitch_events_labeled.parquet")
    parser.add_argument("--eval-targets", default="data/processed/eval_targets_pa.parquet")
    args = parser.parse_args()

    claim1_eval.assert_not_test_season(args.eval_season, final_run=args.final_run)

    tensors, manifest = loader.load_tensors(args.data_dir)
    frame = qt.align_pitch_frame(args.pitch_events, args.eval_targets, tensors["season"])
    pa_df = pd.read_parquet(args.eval_targets)
    tables = build_tables(frame, tensors, manifest, pa_df)

    paths = [Path(args.checkpoint_dir) / f"{args.arm}_s{seed}.pt" for seed in args.seeds]
    models = load_ensemble(paths, manifest, tensors["context"].shape[1])

    predictions, diagnostics = predict(models, tensors, manifest, frame, tables, pa_df,
                                       args.eval_season, n_pitchers=args.n_pitchers,
                                       n_pitches=args.n_pitches, seed=args.seed,
                                       use_league_split=args.league_split)
    composition = league_composition(models, tensors, manifest, frame, tables,
                                     args.eval_season, n_pitchers=args.n_pitchers,
                                     n_pitches=args.n_pitches, seed=args.seed)

    label = args.label or args.arm
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_dir / f"d5_predictions_{label}.csv", index=False)
    (out_dir / f"d5_diagnostics_{label}.json").write_text(
        json.dumps({**diagnostics, "composition": composition,
                    "arm": args.arm, "seeds": args.seeds,
                    "eval_season": args.eval_season}, indent=2))

    print(f"arm: {args.arm} seeds: {args.seeds}")
    print(f"rows: {diagnostics['rows']}")
    print(f"pred_woba mean: {predictions['pred_woba'].mean():.4f}")
    print(f"pred_woba sd: {predictions['pred_woba'].std():.4f}")
    print(f"backoff levels (exact, strikes, stand, empty): {diagnostics['backoff_levels']}")
    print(f"pitcher cells dropped: {diagnostics['pitcher_cells_dropped']}")
    for name, value in composition.items():
        print(f"{name}: {value:.4f}")
    coverage = diagnostics["coverage"]
    print(f"bip joined to a PA outcome: {coverage['bip_joined']} of "
          f"{coverage['bip_available']} "
          f"({coverage['bip_joined'] / coverage['bip_available']:.1%})")
    print(f"contact split from: {diagnostics['contact_split_source']}")
    print(f"wrote {out_dir / f'd5_predictions_{label}.csv'}")


if __name__ == "__main__":
    main()
