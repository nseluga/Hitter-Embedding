"""
Phase V.11 / V.1 / V.7 unit tests — pure logic on synthetic tables. Nothing here
reads a checkpoint; the checkpoint-facing path (`check_provenance`) is guarded
by its own assertions and exercised only when the real checkpoints/logs exist.
"""

import numpy as np
import pandas as pd
from scipy.stats import ortho_group

from src.analysis import model_visualization_embeddings as viz


def test_procrustes_recovers_known_rotation():
    """A pure rotation of seed 0 should align back to near-perfect cosine."""
    rng = np.random.default_rng(0)
    seed0 = rng.normal(size=(200, 8))
    rotation = ortho_group.rvs(dim=8, random_state=1)
    seed1 = (seed0 - seed0.mean(axis=0)) @ rotation

    aligned = viz.procrustes_align(seed1, seed0)
    seed0_centered = seed0 - seed0.mean(axis=0)
    cosine = viz.row_cosine(aligned, seed0_centered)
    assert np.median(cosine) > 0.999


def test_row_shuffled_null_is_near_zero():
    rng = np.random.default_rng(2)
    seed0 = rng.normal(size=(300, 8))
    seed0_centered = seed0 - seed0.mean(axis=0)
    shuffled = seed0_centered[rng.permutation(300)]
    cosine = viz.row_cosine(shuffled, seed0_centered)
    assert abs(np.median(cosine)) < 0.3


def test_effective_rank_isotropic_table():
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(2000, 32))
    _, _, effective_rank, participation_ratio = viz.dimension_usage(matrix)
    assert effective_rank > 28
    assert participation_ratio > 28


def test_effective_rank_rank_one_table():
    rng = np.random.default_rng(4)
    direction = rng.normal(size=32)
    scores = rng.normal(size=2000)
    matrix = np.outer(scores, direction)
    _, _, effective_rank, participation_ratio = viz.dimension_usage(matrix)
    assert effective_rank < 1.5
    assert participation_ratio < 1.5


def test_separability_passes_on_separable_synthetic():
    rng = np.random.default_rng(5)
    n = 200
    stand_l = pd.Series(["L"] * (n // 2))
    stand_r = pd.Series(["R"] * (n // 2))
    stand = pd.concat([stand_l, stand_r], ignore_index=True)
    coords = np.vstack([
        rng.normal(loc=[-5, 0], scale=0.5, size=(n // 2, 2)),
        rng.normal(loc=[5, 0], scale=0.5, size=(n // 2, 2)),
    ])
    result = viz.stand_separability(coords, stand)
    assert result["accuracy"] > 0.9
    assert result["passes"]


def test_separability_fails_on_random_labels():
    rng = np.random.default_rng(6)
    n = 300
    coords = rng.normal(size=(n, 2))
    stand = pd.Series(rng.choice(["L", "R"], size=n))
    result = viz.stand_separability(coords, stand)
    assert not result["passes"]


def test_bootstrap_ci_covers_true_mean():
    rng = np.random.default_rng(7)
    values = rng.normal(loc=1.0, scale=0.1, size=5000)
    point, lo, hi = viz.bootstrap_ci(values, lambda v: float(np.mean(v)))
    assert lo < point < hi
    assert lo < 1.0 < hi
