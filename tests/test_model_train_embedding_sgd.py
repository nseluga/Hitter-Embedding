"""
Phase V item 1: the hitter embedding on plain SGD in its own optimizer while the trunk
stays on AdamW.

Two failures here are silent and both produce a trained model. The embedding ending up in
BOTH optimizers double-steps the table, and a `param_groups` that copies instead of
exposing the live dicts leaves the warmup writing to nothing while every logged lr still
looks right. The first two tests are aimed at exactly those.
"""

from types import SimpleNamespace

import torch

from src.model import sweep, train
from src.model.v1 import HitterEmbeddingV1, factorized_loss
from tests.test_model_v1 import N_CONTEXT, N_HITTERS, make_batch


def args_for(embedding_optimizer="adamw", lr=1e-3, embedding_lr=None):
    return SimpleNamespace(embedding_optimizer=embedding_optimizer, lr=lr,
                           embedding_lr=lr if embedding_lr is None else embedding_lr)


def test_the_default_path_is_still_one_adamw():
    model = HitterEmbeddingV1(N_HITTERS, N_CONTEXT)
    optimizer = train.build_optimizer(model, args_for())

    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == 2
    assert train.embedding_group(optimizer) is None
    assert train.real_optimizers(optimizer) == [optimizer]


def test_the_sgd_path_puts_the_embedding_in_sgd_and_in_no_adamw_group():
    model = HitterEmbeddingV1(N_HITTERS, N_CONTEXT)
    optimizer = train.build_optimizer(model, args_for("sgd", lr=1e-3, embedding_lr=1e-1))
    trunk, embedding = optimizer.optimizers

    assert isinstance(trunk, torch.optim.AdamW) and isinstance(embedding, torch.optim.SGD)
    trunk_params = [id(p) for g in trunk.param_groups for p in g["params"]]
    assert id(model.embedding.weight) not in trunk_params
    assert [id(p) for p in embedding.param_groups[0]["params"]] == [id(model.embedding.weight)]
    # every other parameter is still trained, exactly once
    assert len(trunk_params) == len(set(trunk_params)) == len(list(model.parameters())) - 1
    # coupled L2 under plain SGD subtracts the same lr*wd*w AdamW's decoupled decay does
    assert embedding.param_groups[0]["weight_decay"] == train.WEIGHT_DECAY
    assert embedding.param_groups[0]["lr"] == 1e-1
    assert train.embedding_group(optimizer) is embedding.param_groups[0]


def test_the_wrapper_exposes_the_live_param_groups():
    model = HitterEmbeddingV1(N_HITTERS, N_CONTEXT)
    optimizer = train.build_optimizer(model, args_for("sgd", lr=1e-3, embedding_lr=1e-1))
    trunk, embedding = optimizer.optimizers

    assert len(optimizer.param_groups) == len(trunk.param_groups) + 1
    for group in optimizer.param_groups:
        group["lr"] = 0.5
    assert [g["lr"] for g in trunk.param_groups] == [0.5] * len(trunk.param_groups)
    assert embedding.param_groups[0]["lr"] == 0.5


def test_warmup_ramps_each_optimizer_to_its_own_base():
    model = HitterEmbeddingV1(N_HITTERS, N_CONTEXT)
    optimizer = train.build_optimizer(model, args_for("sgd", lr=1e-3, embedding_lr=1e-1))
    warmup = train.LinearWarmup(optimizer, steps=4)

    warmup.step()
    assert optimizer.optimizers[0].param_groups[0]["lr"] == 1e-3 / 4
    assert optimizer.optimizers[1].param_groups[0]["lr"] == 1e-1 / 4
    for _ in range(3):
        warmup.step()
    assert warmup.done
    assert optimizer.optimizers[0].param_groups[0]["lr"] == 1e-3
    assert optimizer.optimizers[1].param_groups[0]["lr"] == 1e-1
    warmup.step()  # latched: no further writes
    assert optimizer.optimizers[1].param_groups[0]["lr"] == 1e-1


# ----------------------------------- the ml-engineer gates, on the split-optimizer path

def _run(embedding_optimizer, steps=400, warmup_steps=50, seed=3):
    """Overfit one batch with the real optimizer builder and warmup in the loop.

    Shapes, loss at init, the split boundary and the decode are properties of the model and
    the build, unchanged here and discharged by D.4. What a second optimizer CAN break is
    these two: a wrongly wired embedding can stop the run learning, and either optimizer
    carrying stray state can make two same-seed runs diverge."""
    torch.manual_seed(seed)
    model = HitterEmbeddingV1(N_HITTERS, N_CONTEXT, dropout=0.0)
    hitter, context, labels = make_batch(64, seed=seed)
    optimizer = train.build_optimizer(
        model, args_for(embedding_optimizer, lr=1e-2, embedding_lr=1e-1))
    warmup = train.LinearWarmup(optimizer, steps=warmup_steps)

    losses = []
    for _ in range(steps):
        loss, _ = factorized_loss(model(hitter, context, labels["ev"], labels["la"]), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        warmup.step()
        optimizer.step()
        losses.append(loss.item())
    return losses


def test_the_sgd_path_overfits_one_batch():
    losses = _run("sgd")
    assert losses[-1] < losses[0] * 0.05, \
        f"loss only fell from {losses[0]:.1f} to {losses[-1]:.1f} on the sgd path"


def test_two_sgd_runs_at_the_same_seed_are_bit_identical():
    assert _run("sgd") == _run("sgd")


def test_the_default_path_still_passes_both_gates():
    losses = _run("adamw")
    assert losses[-1] < losses[0] * 0.05
    assert losses == _run("adamw")


# --------------------------------------------------------------------------- the stage

def test_the_embedding_sgd_stage_queues_three_single_seed_runs():
    queued = sweep.queue("embedding_sgd", seeds=sweep.DEFAULT_SEEDS["embedding_sgd"])

    assert [(name, seed) for name, _, seed in queued] == [
        ("sgd_lr1e-2", 0), ("sgd_lr1e-1", 0), ("sgd_lr1", 0)]
    for (_, extra, _), rate in zip(queued, ("1e-2", "1e-1", "1")):
        assert extra == [*sweep.O1_BASE, "--embedding-optimizer", "sgd",
                         "--embedding-lr", rate]
    # the knobs the ledger DOES record are the O1 incumbent's, so `reference` is comparable
    assert sweep.knobs(queued[0][1], "unused") == ("0.001", "0", sweep.O1_DATA_DIR)
