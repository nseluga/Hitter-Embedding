"""
V.5 unit tests — pure logic on synthetic data. No checkpoint, parquet, or query chain:
those are exercised only by the real `gradient`/`analyse` runs.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.analysis import model_visualization_platoon as platoon


def test_forward_difference_recovers_known_linear_gap():
    """gap(dim) = 2.0 + 3.0*eps*dim exactly; the finite-difference slope should recover 3.0."""
    eps = 0.05
    gap_base = 2.0
    gap_perturbed = gap_base + 3.0 * eps
    slope = platoon.finite_difference_gradient(gap_perturbed, gap_base, eps)
    assert np.isclose(slope, 3.0)


def test_perturb_restore_leaves_table_bit_identical():
    torch.manual_seed(0)
    model = nn.Module()
    model.embedding = nn.Embedding(5, 4)
    original = model.embedding.weight.detach().clone()

    saved = platoon.perturb_embedding(model, dim=1, eps=0.1)
    assert not torch.equal(model.embedding.weight, original), "perturbation had no effect"
    platoon.restore_embedding(model, saved)
    assert torch.equal(model.embedding.weight, original)
    # row 0 (cold start) must never move
    assert torch.equal(model.embedding.weight[0], original[0])


def test_resume_skips_completed_passes(tmp_path):
    raw_path = tmp_path / "raw.csv"
    pd.DataFrame({"pass_id": ["base", "base", "dim0", "dim0"], "dim": [-1, -1, 0, 0],
                 "eps": [0.0, 0.0, 0.03, 0.03], "batter": [1, 2, 1, 2],
                 "p_throws": ["L", "R", "L", "R"], "pred_woba": [0.3, 0.31, 0.32, 0.29]}
                ).to_csv(raw_path, index=False)
    todo = platoon.pending_passes(raw_path, ["base", "dim0", "dim1"])
    assert todo == ["dim1"]


def test_partial_correlation_removes_pure_confounder():
    """y and x are both driven only by z with independent noise: raw correlation is high,
    partialling on z should collapse it toward zero."""
    rng = np.random.default_rng(0)
    z = rng.normal(size=500)
    x = 2.0 * z + rng.normal(scale=0.05, size=500)
    y = -1.5 * z + rng.normal(scale=0.05, size=500)

    raw_r = float(np.corrcoef(x, y)[0, 1])
    partial_r = platoon.partial_correlation(x, y, z)
    assert abs(raw_r) > 0.8
    assert abs(partial_r) < 0.15


def test_spread_verdict_logic():
    assert platoon.spread_verdict(sd_low=0.10, sd_high=0.20, ci_low_of_diff=0.02) is True
    assert platoon.spread_verdict(sd_low=0.20, sd_high=0.10, ci_low_of_diff=0.02) is False
    assert platoon.spread_verdict(sd_low=0.10, sd_high=0.20, ci_low_of_diff=-0.01) is False
