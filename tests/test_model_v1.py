"""
The D.4 verification gates for the Phase D model, loader, and loss.

Synthetic fixtures only. Every failure these guard against produces a trained model
with a plausible loss curve rather than an error, so each gate is two-sided: it
proves the correct behaviour AND that a broken version is caught. A one-sided
`assert loss > 0` passes on nearly every bug listed below.

Gates 1-7 are the generic ml-engineer set (shapes, overfit-one-batch, loss scale at
init, determinism, split boundary, eval hygiene, decode one batch). Gates 8-10 are
this project's frozen decisions made executable -- the reserved cold-start row, the
strict mask nesting, and the shrinkage that §5's partial-pooling argument rests on.
A framework can be wired perfectly and still have quietly stopped implementing one of
those three, which is the failure that reaches the paper.

D.1's row-major and vocabulary gates live in `tests/test_model_dataset.py`, at the
function that owns them. The label-shuffle test is scheduled once at D.6, before the
first real run, because it costs a full training run.
"""

import json
import math

import numpy as np
import pytest
import torch
import torch.nn as nn

# torch defaults its intra-op pool to the core count, and once it is imported that pool
# coexists with the BLAS threads scipy spawns in the C.2 tests. On 8 cores the two
# oversubscribe and the full suite stalls, while every file passes alone. These gates
# run on batches of at most 8,192 rows, so threading buys them nothing.
torch.set_num_threads(1)

from src.config.splits import load_splits
from src.data.model_dataset import MASKED, RESERVED_HITTER_INDEX
from src.model import loader
from src.model.v1 import (BILINEAR_RANK, CONTEXT_HIDDEN, DEFAULT_EMBEDDING_DIM,
                          DEFAULT_N_BINS, FACTORS, TRUNK_HIDDEN, HitterEmbeddingV1,
                          factor_masks, factorized_loss, weight_decay_groups)

N_HITTERS = 50
N_CONTEXT = 46


def make_labels(n, rng, n_bins=DEFAULT_N_BINS):
    """
    Synthetic labels carrying the §1.2 nesting: contact is defined on swings, the
    quality dimensions on balls in play, and MASKED everywhere else.
    """
    swing = rng.integers(0, 2, n)
    contact = np.where(swing == 1, rng.integers(0, 2, n), MASKED)
    in_play = (contact == 1) & (rng.random(n) < 0.5)
    quality = {name: np.where(in_play, rng.integers(0, n_bins, n), MASKED)
               for name in ("ev", "la", "spray")}
    return {name: torch.from_numpy(np.asarray(values, dtype=np.int64))
            for name, values in {"swing": swing, "contact": contact, **quality}.items()}


def make_batch(n, seed=0, n_hitters=N_HITTERS, n_context=N_CONTEXT):
    """A synthetic batch: hitter indices, context, and nested labels."""
    rng = np.random.default_rng(seed)
    hitter = torch.from_numpy(rng.integers(0, n_hitters + 1, n).astype(np.int64))
    context = torch.from_numpy(rng.normal(0, 1, (n, n_context)).astype(np.float32))
    return hitter, context, make_labels(n, rng)


def build_model(seed=0, **kwargs):
    torch.manual_seed(seed)
    return HitterEmbeddingV1(N_HITTERS, N_CONTEXT, **kwargs)


def forward(model, hitter, context, labels):
    """Teacher forcing: the quality heads condition on the OBSERVED bins."""
    return model(hitter, context, labels["ev"], labels["la"])


# --------------------------------------------------------------------------- 1


def test_every_module_boundary_has_the_shape_the_spec_states():
    hitter, context, labels = make_batch(256)
    model = build_model().eval()
    outputs = forward(model, hitter, context, labels)

    assert outputs["swing"].shape == (256,)
    assert outputs["contact"].shape == (256,)
    for name in ("ev", "la", "spray"):
        assert outputs[name].shape == (256, DEFAULT_N_BINS)
    assert model.embedding(hitter).shape == (256, DEFAULT_EMBEDDING_DIM)
    assert model.context_tower(context).shape == (256, CONTEXT_HIDDEN)


def test_a_binary_head_that_keeps_its_trailing_axis_is_caught():
    # (B, 1) against a (B,) target broadcasts into an all-pairs mean, which trains
    # perfectly happily; the model's own assertion is what refuses it
    hitter, context, labels = make_batch(64)
    model = build_model().eval()
    model.head_swing = nn.Linear(TRUNK_HIDDEN, 2)  # squeeze(-1) cannot flatten this
    with pytest.raises(AssertionError, match="swing head must be"):
        forward(model, hitter, context, labels)


# --------------------------------------------------------------------------- 2


def test_the_model_can_overfit_a_single_batch():
    hitter, context, labels = make_batch(64)
    model = build_model(dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    rows = sum(int(mask.sum()) for mask in factor_masks(labels).values())

    first = None
    for _ in range(400):
        loss, _ = factorized_loss(forward(model, hitter, context, labels), labels)
        first = first if first is not None else loss.item() / rows
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    final = loss.item() / rows
    assert first > 1.0, "a batch at init should cost about a coin flip plus ln 24"
    assert final < 0.05, f"could not overfit 64 rows: {final:.4f} per row"


# --------------------------------------------------------------------------- 3


def test_loss_at_init_is_the_uniform_prediction():
    hitter, context, labels = make_batch(8192)
    model = build_model().eval()
    with torch.no_grad():
        _, parts = factorized_loss(forward(model, hitter, context, labels), labels)
    masks = factor_masks(labels)

    for name in ("swing", "contact"):
        per_row = parts[name].item() / int(masks[name].sum())
        assert per_row == pytest.approx(math.log(2), abs=0.02), f"{name} {per_row}"
    for name in ("ev", "la", "spray"):
        per_row = parts[name].item() / int(masks[name].sum())
        assert per_row == pytest.approx(math.log(DEFAULT_N_BINS), abs=0.05), f"{name} {per_row}"


def test_the_loss_is_a_raw_sum_and_not_a_mean_over_rows():
    # doubling the batch must double the loss. a per-head mean, or any inverse-frequency
    # weighting, leaves it unchanged -- and both are D.8 arms, not the default (§4)
    hitter, context, labels = make_batch(512)
    model = build_model().eval()
    with torch.no_grad():
        single, _ = factorized_loss(forward(model, hitter, context, labels), labels)
        doubled_labels = {name: torch.cat([value, value]) for name, value in labels.items()}
        doubled, _ = factorized_loss(
            forward(model, torch.cat([hitter, hitter]), torch.cat([context, context]),
                    doubled_labels), doubled_labels)
    assert doubled.item() == pytest.approx(2 * single.item(), rel=1e-5)


# --------------------------------------------------------------------------- 4


def train_briefly(seed, steps=5):
    hitter, context, labels = make_batch(256)
    model = build_model(seed=seed)
    optimizer = torch.optim.AdamW(weight_decay_groups(model, 1e-2), lr=1e-3)
    losses = []
    torch.manual_seed(seed)
    for _ in range(steps):
        loss, _ = factorized_loss(forward(model, hitter, context, labels), labels)
        losses.append(loss.item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return losses


def test_the_same_seed_gives_the_same_losses_and_a_different_seed_does_not():
    assert train_briefly(0) == train_briefly(0)
    assert train_briefly(0) != train_briefly(1)


def test_batch_order_is_reproducible_only_through_a_seeded_generator():
    indices = torch.arange(10_000)
    first = next(loader.batches(indices, 256, generator=torch.Generator().manual_seed(0)))
    same = next(loader.batches(indices, 256, generator=torch.Generator().manual_seed(0)))
    other = next(loader.batches(indices, 256, generator=torch.Generator().manual_seed(1)))
    assert torch.equal(first, same)
    assert not torch.equal(first, other)


# --------------------------------------------------------------------------- 5, 7


@pytest.fixture
def built_dir(tmp_path):
    """A saved dataset spanning every configured season, one planted pitch at row 7."""
    config = load_splits()
    seasons = np.repeat(np.array(config["seasons"], dtype=np.int64), 40)
    n = len(seasons)
    rng = np.random.default_rng(0)

    arrays = {
        "context": rng.normal(0, 1, (n, N_CONTEXT)).astype(np.float32),
        "hitter": rng.integers(0, N_HITTERS + 1, n).astype(np.int64),
        "season": seasons,
        **{name: value.numpy() for name, value in make_labels(n, rng).items()},
    }
    # the planted pitch: values no random draw produces, so a join or alignment bug
    # that keeps every column individually valid still shows up here
    arrays["context"][7] = np.arange(N_CONTEXT, dtype=np.float32) * 100.0
    arrays["hitter"][7] = 42
    for name, value in zip(("swing", "contact", "ev", "la", "spray"), (1, 1, 3, 11, 19)):
        arrays[name][7] = value

    for name, array in arrays.items():
        np.save(tmp_path / f"{name}.npy", np.ascontiguousarray(array))
    (tmp_path / "manifest.json").write_text(json.dumps({
        "n_pitches": n, "n_hitters": N_HITTERS, "n_quality_bins": DEFAULT_N_BINS,
        "train_seasons": config["split"]["train"]}))
    return tmp_path


def test_the_splits_partition_the_table_with_no_row_or_season_in_two(built_dir):
    tensors, manifest = loader.load_tensors(built_dir)
    indices = loader.split_indices(tensors["season"])

    seen = set()
    for name, index in indices.items():
        rows = set(index.tolist())
        assert not rows & seen, f"{name} shares rows with an earlier split"
        seen |= rows
    assert len(seen) == manifest["n_pitches"], "some rows belong to no split"

    seasons = {name: set(tensors["season"][index].tolist()) for name, index in indices.items()}
    assert not seasons["train"] & seasons["val"]
    assert not (seasons["train"] | seasons["val"]) & seasons["test"]
    assert not set(manifest["train_seasons"]) & seasons["test"]


def test_a_season_in_two_splits_or_in_none_is_caught(built_dir):
    # the two ways a partition breaks: a leaked season lands in two splits, and a
    # dropped season lands in none. Both are silent -- the first trains on validation
    # rows, the second quietly shrinks the epoch
    config = load_splits()
    tensors, _ = loader.load_tensors(built_dir)

    leaked = {"seasons": config["seasons"],
              "split": {**config["split"], "val": config["split"]["train"][-1:] +
                        config["split"]["val"]}}
    with pytest.raises(AssertionError, match="do not partition"):
        loader.split_indices(tensors["season"], config=leaked)

    dropped = {"seasons": config["seasons"],
               "split": {**config["split"], "train": config["split"]["train"][1:]}}
    with pytest.raises(AssertionError, match="absent from the frozen split config"):
        loader.split_indices(tensors["season"], config=dropped)


def test_a_dataset_fit_on_the_test_season_is_refused(built_dir):
    manifest = json.loads((built_dir / "manifest.json").read_text())
    manifest["train_seasons"] = manifest["train_seasons"] + load_splits()["split"]["test"]
    (built_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(AssertionError, match="frozen test season"):
        loader.load_tensors(built_dir)


def test_a_planted_pitch_survives_save_load_and_batching_unchanged(built_dir):
    tensors, _ = loader.load_tensors(built_dir)
    index = torch.tensor([3, 7, 11])
    hitter, context, labels = loader.gather(tensors, index)

    assert int(hitter[1]) == 42
    assert torch.equal(context[1], torch.arange(N_CONTEXT, dtype=torch.float32) * 100.0)
    for name, value in zip(("swing", "contact", "ev", "la", "spray"), (1, 1, 3, 11, 19)):
        assert int(labels[name][1]) == value, f"{name} decoded wrong"


# --------------------------------------------------------------------------- 6


def test_eval_mode_is_deterministic_and_train_mode_is_not():
    hitter, context, labels = make_batch(512)
    model = build_model()

    model.eval()
    with torch.no_grad():
        first = forward(model, hitter, context, labels)["ev"]
        second = forward(model, hitter, context, labels)["ev"]
    assert torch.equal(first, second), "dropout is live under eval()"

    model.train()
    with torch.no_grad():
        third = forward(model, hitter, context, labels)["ev"]
        fourth = forward(model, hitter, context, labels)["ev"]
    assert not torch.equal(third, fourth), "dropout does nothing under train()"


# --------------------------------------------------------------------------- 8


def test_the_reserved_row_receives_no_gradient_and_never_moves():
    hitter = torch.tensor([RESERVED_HITTER_INDEX] * 8 + [5] * 8)
    _, context, labels = make_batch(16)
    model = build_model()
    optimizer = torch.optim.AdamW(weight_decay_groups(model, 1e-2), lr=1e-2)

    reserved = lambda: float(model.embedding.weight.detach()[RESERVED_HITTER_INDEX].abs().sum())
    assert reserved() == 0.0
    before = model.embedding.weight[5].detach().clone()

    for _ in range(5):
        loss, _ = factorized_loss(forward(model, hitter, context, labels), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert float(model.embedding.weight.grad[RESERVED_HITTER_INDEX].abs().sum()) == 0.0
        optimizer.step()

    assert reserved() == 0.0
    assert not torch.equal(model.embedding.weight[5], before), \
        "row 5 did not move either, so the test proves nothing about row 0"


# --------------------------------------------------------------------------- 9


def test_a_masked_row_contributes_exactly_nothing_and_an_observed_row_does():
    hitter, context, labels = make_batch(512)
    model = build_model().eval()
    with torch.no_grad():
        outputs = forward(model, hitter, context, labels)
        base, _ = factorized_loss(outputs, labels)

        masks = factor_masks(labels)
        masked_row = int(torch.nonzero(~masks["ev"])[0])
        observed_row = int(torch.nonzero(masks["ev"])[0])

        # ONE logit, not the whole row: cross entropy is invariant to a constant
        # shift across a row's logits, so perturbing all 24 leaves the loss unchanged
        # whether or not the row is masked, and the gate would pass for the wrong reason
        outputs["ev"][masked_row, 0] += 25.0
        assert factorized_loss(outputs, labels)[0].item() - base.item() == 0.0

        outputs["ev"][observed_row, 0] += 25.0
        assert factorized_loss(outputs, labels)[0].item() != base.item()


def test_the_quality_factors_are_scored_only_where_their_conditions_were_observed():
    labels = {name: torch.tensor(value) for name, value in {
        "swing": [1, 1, 1], "contact": [1, 1, 1],
        "ev": [3, MASKED, 3], "la": [5, 5, MASKED], "spray": [7, 7, 7]}.items()}
    masks = factor_masks(labels)
    # row 1 has LA without EV -- p(la | ev) is undefined there (2026-08-02)
    assert masks["ev"].tolist() == [True, False, True]
    assert masks["la"].tolist() == [True, False, False]
    assert masks["spray"].tolist() == [True, False, False]


# --------------------------------------------------------------------------- 10


def test_weight_decay_covers_the_weights_and_the_embedding_but_not_the_biases():
    model = build_model()
    groups = weight_decay_groups(model, 1e-2)
    decayed = {id(p) for p in groups[0]["params"]}
    undecayed = {id(p) for p in groups[1]["params"]}

    assert groups[0]["weight_decay"] == 1e-2 and groups[1]["weight_decay"] == 0.0
    assert not decayed & undecayed
    assert len(decayed) + len(undecayed) == len(list(model.parameters()))
    for name, parameter in model.named_parameters():
        expected = undecayed if name.endswith("bias") else decayed
        assert id(parameter) in expected, f"{name} landed in the wrong group"
    assert id(model.embedding.weight) in decayed, "the embedding is the shrinkage lever"


# ---------------------------------------------------------------- interaction term


def test_the_interaction_term_is_off_by_default_and_low_rank_when_on():
    off, on = build_model(), build_model(bilinear=True)
    assert off.interaction is None, "the D.8 arm must not be on by default"

    added = sum(p.numel() for p in on.parameters()) - sum(p.numel() for p in off.parameters())
    expected = BILINEAR_RANK * (DEFAULT_EMBEDDING_DIM + CONTEXT_HIDDEN + TRUNK_HIDDEN)
    assert added == expected == 13_312
    full_form = DEFAULT_EMBEDDING_DIM * CONTEXT_HIDDEN * TRUNK_HIDDEN + TRUNK_HIDDEN
    assert added < full_form / 50, "this is the form the parameter budget rejected"

    for name, parameter in on.named_parameters():
        if name.startswith(("project_", "interaction")):
            assert parameter.dim() == 2, f"{name} has a bias, which is a main effect"


def test_the_interaction_directions_do_not_die_during_training():
    # a rank that trains to all-zero pairings would make the D.8 arm a no-op that
    # still costs 13,312 parameters, and the loss curve would not say so
    hitter, context, labels = make_batch(512)
    model = build_model(bilinear=True, dropout=0.0)
    optimizer = torch.optim.AdamW(weight_decay_groups(model, 1e-2), lr=1e-2)
    for _ in range(100):
        loss, _ = factorized_loss(forward(model, hitter, context, labels), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        paired = (model.project_hitter(model.embedding(hitter))
                  * model.project_context(model.context_tower(context)))
        strength = paired.abs().mean(dim=0)
    live = int((strength > 0.01 * float(strength.max())).sum())
    assert live >= BILINEAR_RANK // 4, f"only {live} of {BILINEAR_RANK} directions carry signal"


# ------------------------------------------------------- D.8 loss arms (§7, off by default)


def test_the_default_loss_is_unweighted_and_the_arms_change_it():
    hitter, context, labels = make_batch(512)
    model = build_model().eval()
    with torch.no_grad():
        outputs = forward(model, hitter, context, labels)
        base, base_parts = factorized_loss(outputs, labels)
        explicit, _ = factorized_loss(outputs, labels, weighting="sum")
        averaged, mean_parts = factorized_loss(outputs, labels, weighting="mean")
        weighted, weighted_parts = factorized_loss(outputs, labels,
                                                   contact_pos_weight=torch.tensor(3.0))

    assert explicit.item() == base.item(), "'sum' must be the default, not a variant"
    # the arms have to actually do something, or a null result means nothing
    assert averaged.item() != base.item()
    assert weighted.item() != base.item()
    # the per-factor parts stay raw so a weighted run and an unweighted one remain
    # comparable head by head; only the total carries the weighting
    for name in FACTORS:
        assert mean_parts[name].item() == pytest.approx(base_parts[name].item())
    for name in ("swing", "ev", "la", "spray"):
        assert weighted_parts[name].item() == pytest.approx(base_parts[name].item()), \
            f"the contact arm moved {name}"
    assert weighted_parts["contact"].item() > base_parts["contact"].item()


def test_mean_weighting_is_each_factor_over_its_own_valid_rows():
    hitter, context, labels = make_batch(512)
    model = build_model().eval()
    with torch.no_grad():
        outputs = forward(model, hitter, context, labels)
        averaged, parts = factorized_loss(outputs, labels, weighting="mean")
    masks = factor_masks(labels)
    expected = sum(parts[name].item() / int(masks[name].sum()) for name in FACTORS)
    assert averaged.item() == pytest.approx(expected, rel=1e-6)


def test_an_unknown_weighting_is_refused():
    hitter, context, labels = make_batch(64)
    model = build_model().eval()
    outputs = forward(model, hitter, context, labels)
    with pytest.raises(AssertionError, match="unknown weighting"):
        factorized_loss(outputs, labels, weighting="inverse")


def test_the_contact_class_weight_is_counted_on_the_train_split_only(built_dir):
    # counting on the full table would read the validation and test seasons to set a
    # training constant, and no loss curve would show it
    from src.model.train import contact_pos_weight
    tensors, _ = loader.load_tensors(built_dir)
    indices = loader.split_indices(tensors["season"])

    train_only = float(contact_pos_weight(tensors, indices["train"]))
    everything = float(contact_pos_weight(tensors, torch.arange(len(tensors["season"]))))
    assert train_only > 0

    observed = tensors["contact"][indices["train"]]
    observed = observed[observed != MASKED]
    positive = int((observed == 1).sum())
    assert train_only == pytest.approx((len(observed) - positive) / positive, rel=1e-6)
    assert train_only != everything, \
        "train-only and whole-table weights coincide, so this proves nothing"
