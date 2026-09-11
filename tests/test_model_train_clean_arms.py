"""
Clean-stage code arms: --embedding-weight-decay-exposure (per-row table decay ∝ 1/n_h)
and --hitter-dropout (a pitch trained on the reserved row with probability ∝ 1/n_h).
Both are training-code changes, so the checks are mechanism checks: the decay lands on
the gradient in the right amount per row, the dropout swaps the right rows at the right
rate, and neither touches evaluation or the default path.
"""

from types import SimpleNamespace

import torch

from src.data.model_dataset import RESERVED_HITTER_INDEX
from src.model import sweep, train
from src.model.v1 import HitterEmbeddingV1, factorized_loss
from tests.test_model_v1 import N_CONTEXT, N_HITTERS, make_batch


def _tensors(counts):
    """A train set where row i appears counts[i] times (row 0 reserved, unused)."""
    hitter = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(counts)])
    return {"hitter": hitter}, torch.arange(len(hitter))


def test_exposure_scaling_is_median_anchored_and_capped():
    tensors, idx = _tensors([0, 10, 100, 1000, 0])
    counts = train.hitter_exposure(tensors, idx, 5)
    scaled = train.exposure_scaled(counts, 1e-2, train.EXPOSURE_DECAY_CAP)
    assert counts.tolist() == [0, 10, 100, 1000, 0]
    assert abs(scaled[2].item() - 1e-2) < 1e-9                    # median row gets the flag's value
    assert abs(scaled[3].item() - 1e-3) < 1e-9          # ten times the exposure, a tenth
    assert abs(scaled[1].item() - train.EXPOSURE_DECAY_CAP) < 1e-7  # 0.1 would be the value; capped
    assert abs(scaled[4].item() - train.EXPOSURE_DECAY_CAP) < 1e-7  # never seen in train: cap


def test_the_exposure_hook_adds_row_decay_to_the_gradient():
    torch.manual_seed(0)
    model = HitterEmbeddingV1(4, N_CONTEXT, split=True)
    tensors, idx = _tensors([0, 10, 100, 1000, 0])
    args = SimpleNamespace(embedding_weight_decay_exposure=1e-2)
    decay = train.exposure_decay_hook(model, tensors, idx, args)
    weight_before = model.embedding.weight.detach().clone()
    # a loss that touches no row: the gradient is then pure decay on every row
    model.embedding.weight.pow(2).sum().mul(0.0).backward()
    grad = model.embedding.weight.grad
    expected = decay[:, None] * weight_before
    assert torch.allclose(grad, expected)
    assert torch.all(grad[RESERVED_HITTER_INDEX] == 0)   # reserved row: decay forced to 0


def test_the_sgd_path_with_exposure_decay_turns_the_scalar_decay_off():
    model = HitterEmbeddingV1(N_HITTERS, N_CONTEXT, split=True)
    args = SimpleNamespace(embedding_optimizer="sgd", lr=1e-3, embedding_lr=1.0,
                           embedding_weight_decay=None, embedding_weight_decay_exposure=1e-2)
    optimizer = train.build_optimizer(model, args)
    assert optimizer.optimizers[1].param_groups[0]["weight_decay"] == 0.0


def test_hitter_dropout_rates_scale_with_exposure_and_drop_the_right_rows():
    tensors, idx = _tensors([0, 10, 100, 1000])
    args = SimpleNamespace(hitter_dropout=0.1)
    rates = train.hitter_dropout_rates(tensors, idx, 4, args)
    assert rates[RESERVED_HITTER_INDEX] == 0
    assert abs(rates[2].item() - 0.1) < 1e-7
    assert abs(rates[3].item() - 0.01) < 1e-7
    assert abs(rates[1].item() - 1.0) < 1e-7                        # 10 pitches: rate 1.0, capped
    hitter = torch.tensor([1, 2, 3] * 1000)
    dropped = train.drop_hitters(hitter, rates, torch.Generator().manual_seed(0))
    swapped = (dropped == RESERVED_HITTER_INDEX).view(-1, 3).float().mean(0)
    assert swapped[0] == 1.0
    assert 0.07 < swapped[1] < 0.13
    assert swapped[2] < 0.03
    assert train.hitter_dropout_rates(tensors, idx, 4, SimpleNamespace(hitter_dropout=0.0)) is None


def _epoch_losses(hitter_dropout, seed=0, steps=3):
    torch.manual_seed(seed)
    hitter, context, labels = make_batch(256, seed=seed)
    tensors = {"hitter": hitter, "context": context, **labels}
    model = HitterEmbeddingV1(N_HITTERS, N_CONTEXT, dropout=0.0)
    args = SimpleNamespace(embedding_optimizer="sgd", lr=1e-3, embedding_lr=1.0,
                           embedding_weight_decay=None, hitter_dropout=hitter_dropout,
                           batch_size=64, device=None, loss_rule="log",
                           loss_weighting="sum", contact_pos_weight=None)
    optimizer = train.build_optimizer(model, args)
    out = []
    for _ in range(steps):
        loss, _, _ = train.run_epoch(model, tensors, torch.arange(256), optimizer,
                                     torch.Generator().manual_seed(seed), args)
        out.append(loss)
    return out


def test_dropout_changes_training_and_is_seed_deterministic():
    assert _epoch_losses(0.0) == _epoch_losses(0.0)
    assert _epoch_losses(0.3) == _epoch_losses(0.3)
    assert _epoch_losses(0.3) != _epoch_losses(0.0)
    assert all(map(torch.isfinite, map(torch.tensor, _epoch_losses(0.3))))


def test_the_clean_stage_declares_the_two_code_arms_with_the_flags_train_knows():
    configs = dict(sweep.STAGES["clean"])
    assert "--embedding-weight-decay-exposure" in configs["clean_wd_exposure"]
    assert "--hitter-dropout" in configs["clean_dropout"]
    for extra in configs.values():
        assert not any(flag in extra for flag in sweep.QUARANTINED_FLAGS)
