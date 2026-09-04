"""
The final refit: a fixed step budget with the plateau cuts replayed, trained against a split
that has no validation season (2026-09-04 decision log).

Three failures here are silent and every one of them produces a trained model.

1. THE REPLAY IS NOT A NO-OP. If the budgeted path consumed the RNG differently, or stepped
   the schedule at the wrong moment, the refit would not be reproducing the arm it claims to
   reproduce -- and with no validation loss nothing downstream would notice. The bit-identity
   test is the check: before the first cut, budgeted and normal training must agree exactly.

2. THE SEALED SEASON. The final-run config moves 2024 into train. The one thing it must never
   do is drag 2025 along, and `loader.split_indices` is the single place every caller passes
   through, so that is where the guard lives and where it is tested -- under BOTH configs.

3. THE SCHEDULE IS OFF BY AN EPOCH. `best epoch` is 0-indexed and its checkpoint was written
   at the END of that epoch, so the budget is (best + 1) epochs of steps. One epoch of 719
   steps is 4% of the run and would show up as nothing but a slightly different final loss.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.config.splits import load_splits, validate_split_config
from src.model import loader, replay_schedule, sweep, train
from src.model.v1 import HitterEmbeddingV1
from tests.test_model_v1 import N_CONTEXT, N_HITTERS, make_batch

FINAL_RUN_CONFIG = Path("src/config/split_config_final_run.json")
FROZEN = load_splits()


# --------------------------------------------------------------- the final-run split config

def test_the_committed_final_run_config_absorbs_val_and_keeps_the_sealed_season():
    config = load_splits(FINAL_RUN_CONFIG)

    assert config["final_run"] is True
    assert config["split"]["val"] == []
    assert config["split"]["test"] == FROZEN["split"]["test"] == [2025]
    # exactly the frozen train seasons plus the frozen validation season, nothing else
    assert config["split"]["train"] == FROZEN["split"]["train"] + FROZEN["split"]["val"]


def test_an_empty_val_is_rejected_unless_the_config_says_final_run():
    config = json.loads(FINAL_RUN_CONFIG.read_text())
    del config["final_run"]
    with pytest.raises(AssertionError):
        validate_split_config(config)


def test_a_final_run_config_cannot_move_the_test_season():
    config = json.loads(FINAL_RUN_CONFIG.read_text())
    config["seasons"] = list(range(2015, 2027))
    config["split"]["train"] = config["split"]["train"] + [2025]
    config["split"]["test"] = [2026]
    with pytest.raises(AssertionError, match="frozen test season"):
        validate_split_config(config)


def test_the_frozen_config_is_untouched():
    assert FROZEN["frozen"] is True and FROZEN["frozen_on"] == "2026-07-17"
    assert FROZEN["split"] == {"train": list(range(2015, 2024)), "val": [2024], "test": [2025]}


# ------------------------------------------------------------------ the split boundary gate

def _season_tensor():
    seasons = [s for s in FROZEN["seasons"] for _ in range(3)]
    return torch.tensor(seasons, dtype=torch.int64)


@pytest.mark.parametrize("config_path", [None, FINAL_RUN_CONFIG])
def test_no_sealed_season_row_reaches_train_or_val_under_either_config(config_path):
    season = _season_tensor()
    config = load_splits(config_path) if config_path else None
    indices = loader.split_indices(season, config)

    for name in ("train", "val"):
        assert 2025 not in set(season[indices[name]].tolist())
    assert set(season[indices["test"]].tolist()) == {2025}
    assert sum(len(index) for index in indices.values()) == len(season)


def test_the_guard_fires_when_a_config_would_train_on_the_sealed_season():
    leaky = {"seasons": [2024, 2025], "split": {"train": [2024, 2025], "val": [], "test": []}}
    with pytest.raises(AssertionError, match="sealed test season"):
        loader.split_indices(torch.tensor([2024, 2025], dtype=torch.int64), leaky)


# ------------------------------------------------------- the schedule recovered from the logs

def test_the_budget_is_best_epoch_plus_one_and_only_earlier_cuts_are_replayed(tmp_path):
    log = tmp_path / "arm_s0.log"
    log.write_text("\n".join([
        "epoch   0  train 1.1  val 1.1  ref 1.1  lr 1.00e-03  emb_lr 1.00e+00  10s (100 steps)",
        "epoch   1  train 1.0  val 1.0  ref 1.0  lr 1.00e-03  emb_lr 1.00e+00  10s (100 steps)",
        "epoch   2  train 0.9  val 0.9  ref 0.9  lr 3.00e-04  emb_lr 3.00e-01  10s (100 steps)",
        "epoch   3  train 0.9  val 0.9  ref 0.9  lr 9.00e-05  emb_lr 9.00e-02  10s (100 steps)",
        "best val loss 0.9 at epoch 2; reference 0.9; checkpoint x.pt",
    ]) + "\n")
    parsed = replay_schedule.parse_log(log)

    assert parsed["best_epoch"] == 2 and parsed["steps_per_epoch"] == 100
    # the cut visible on epoch 2 fired at the end of epoch 1; the one on epoch 3 fired at the
    # end of epoch 2, which is after the checkpoint was written, so it never applied to it
    assert parsed["cut_epochs"] == [1]

    recovered = replay_schedule.recover([log])
    assert recovered["step_budget"] == 300, "three epochs of steps, not two"
    assert recovered["lr_cut_steps"] == [200]


def test_the_committed_schedule_matches_the_five_canonical_runs():
    schedule = json.loads(Path("results/model_v1/replay_schedule.json").read_text())
    logs = [Path(f"results/model_v1/logs/embedding_sgd_sgd_lr1_s{seed}.log") for seed in range(5)]
    if not all(log.exists() for log in logs):
        pytest.skip("run logs are not in this checkout")

    recovered = replay_schedule.recover(logs)
    assert recovered["step_budget"] == schedule["step_budget"]
    assert recovered["lr_cut_steps"] == schedule["lr_cut_steps"]
    assert all(0 < cut < schedule["step_budget"] for cut in schedule["lr_cut_steps"])


# --------------------------------------------------------------------- the replay itself

def test_the_replay_schedule_cuts_every_group_once_per_cut():
    model = HitterEmbeddingV1(N_HITTERS, N_CONTEXT)
    optimizer = train.build_optimizer(model, _args(embedding_optimizer="sgd", embedding_lr=1.0))
    schedule = train.ReplaySchedule(optimizer, [3, 5])
    base = [group["lr"] for group in optimizer.param_groups]

    for step in range(1, 6):
        schedule.step(step)
        expected = train.PLATEAU_FACTOR ** (step >= 3) * train.PLATEAU_FACTOR ** (step >= 5)
        assert [g["lr"] for g in optimizer.param_groups] == pytest.approx(
            [lr * expected for lr in base]), f"after step {step}"
    assert schedule.fired == 2

    # a step count that jumps past several cuts at once still fires each of them
    other = train.build_optimizer(model, _args(embedding_optimizer="sgd", embedding_lr=1.0))
    jumped = train.ReplaySchedule(other, [3, 5])
    jumped.step(9)
    assert jumped.fired == 2


# ---------------------------------------------------------- the ml-engineer gates, budgeted

def _args(**overrides):
    args = SimpleNamespace(
        embedding_optimizer="adamw", lr=1e-2, embedding_lr=1e-2, embedding_weight_decay=None,
        batch_size=16, device=None, warmup_steps=0, max_epochs=50, step_budget=0,
        lr_cut_steps=[], loss_rule="log", loss_weighting="sum", contact_pos_weight=None,
        canonical=True, eval_split=None, seed=0, run_name="pytest_refit", embedding_dim=8,
        bilinear=False, contact_inverse_frequency=False)
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def _tensors(n=160, seed=11):
    hitter, context, labels = make_batch(n, seed=seed)
    return {"hitter": hitter, "context": context, **labels,
            "season": torch.full((n,), 2015, dtype=torch.int64)}


def _budgeted_losses(steps, cut_steps=(), seed=5):
    """Run the budgeted path for `steps` steps, returning every per-step loss."""
    torch.manual_seed(seed)
    model = HitterEmbeddingV1(N_HITTERS, N_CONTEXT, dropout=0.0)
    tensors = _tensors()
    indices = {"train": torch.arange(len(tensors["hitter"]))}
    args = _args(embedding_optimizer="sgd", embedding_lr=1e-1,
                 step_budget=steps, lr_cut_steps=list(cut_steps))
    optimizer = train.build_optimizer(model, args)
    generator = torch.Generator().manual_seed(seed)

    losses = []
    schedule = train.ReplaySchedule(optimizer, args.lr_cut_steps)
    taken = 0
    while taken < steps:
        def on_step(step, loss_per_row, taken=taken):
            schedule.step(taken + step)
            losses.append(loss_per_row)
        _, made, _ = train.run_epoch(model, tensors, indices["train"], optimizer, generator,
                                     args, on_step=on_step, max_steps=steps - taken)
        taken += made
    return losses, schedule


def _normal_losses(steps, seed=5):
    """The same run with no budget and no schedule: the trajectory being reproduced."""
    torch.manual_seed(seed)
    model = HitterEmbeddingV1(N_HITTERS, N_CONTEXT, dropout=0.0)
    tensors = _tensors()
    index = torch.arange(len(tensors["hitter"]))
    args = _args(embedding_optimizer="sgd", embedding_lr=1e-1)
    optimizer = train.build_optimizer(model, args)
    generator = torch.Generator().manual_seed(seed)

    losses = []
    while len(losses) < steps:
        train.run_epoch(model, tensors, index, optimizer, generator, args,
                        on_step=lambda step, loss: losses.append(loss))
    return losses[:steps]


def test_a_budgeted_run_is_bit_identical_to_a_normal_one_before_the_first_cut():
    steps = 40
    budgeted, schedule = _budgeted_losses(steps, cut_steps=(1_000,))
    assert schedule.fired == 0, "the guard case only holds while no cut has fired"
    assert budgeted == _normal_losses(steps), \
        "the replay path changed the trajectory before it changed any learning rate"


def test_the_budget_stops_on_the_step_and_not_on_an_epoch_boundary():
    # 160 rows at batch 16 is 10 steps an epoch, so 25 ends mid-epoch
    losses, _ = _budgeted_losses(25)
    assert len(losses) == 25


def test_every_cut_inside_the_budget_fires_and_the_run_still_learns():
    losses, schedule = _budgeted_losses(60, cut_steps=(20, 40))
    assert schedule.fired == 2
    assert losses[-1] < losses[0], f"loss went from {losses[0]:.3f} to {losses[-1]:.3f}"


def test_two_budgeted_runs_at_the_same_seed_are_bit_identical():
    assert _budgeted_losses(30, cut_steps=(10,))[0] == _budgeted_losses(30, cut_steps=(10,))[0]


def test_the_evaluation_pass_leaves_the_model_in_eval_mode_and_trains_nothing():
    model = HitterEmbeddingV1(N_HITTERS, N_CONTEXT)
    tensors = _tensors()
    index = torch.arange(len(tensors["hitter"]))
    before = [p.detach().clone() for p in model.parameters()]

    model.train(True)
    train.run_epoch(model, tensors, index, None, None, _args(), objective=train.CANONICAL)

    assert not model.training, "the eval pass left dropout live"
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_a_budgeted_loss_is_on_the_same_scale_as_an_untrained_one():
    losses, _ = _budgeted_losses(5)
    # five factors of a few dozen classes; a per-row loss outside this band means the
    # objective or the row denominator moved, not the model
    assert 0.1 < losses[0] < 50.0, losses[0]


# ------------------------------------------------------------------ the no-decay ablation

def test_the_nodecay_stage_zeroes_the_table_and_nothing_else():
    queued = sweep.queue("embedding_sgd_nodecay",
                         seeds=sweep.DEFAULT_SEEDS["embedding_sgd_nodecay"])
    assert [(name, seed) for name, _, seed in queued] == [("sgd_lr1_nodecay", 0)]
    assert queued[0][1] == [*sweep.O1_BASE, "--embedding-optimizer", "sgd",
                            "--embedding-lr", "1", "--embedding-weight-decay", "0"]
    # the ledger knobs stay the O1 incumbent's, so `reference` is comparable to sgd_lr1
    assert sweep.knobs(queued[0][1], "unused") == ("0.001", "0", sweep.O1_DATA_DIR)

    model = HitterEmbeddingV1(N_HITTERS, N_CONTEXT)
    optimizer = train.build_optimizer(
        model, _args(embedding_optimizer="sgd", embedding_lr=1.0, embedding_weight_decay=0.0))
    trunk, embedding = optimizer.optimizers
    assert embedding.param_groups[0]["weight_decay"] == 0.0
    assert {g["weight_decay"] for g in trunk.param_groups} == {train.WEIGHT_DECAY, 0.0}


def test_the_table_decay_knob_is_refused_on_the_single_adamw_path():
    model = HitterEmbeddingV1(N_HITTERS, N_CONTEXT)
    with pytest.raises(AssertionError, match="embedding-optimizer sgd"):
        train.build_optimizer(model, _args(embedding_weight_decay=0.0))
