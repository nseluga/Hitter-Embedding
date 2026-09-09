"""Regression tests for the shuffle-before-fit null in seed_stability_corrected_null."""

import numpy as np

from src.analysis.model_visualization_embeddings import procrustes_align, row_cosine
from src.analysis.seed_stability_corrected_null import corrected_null_cosine


def test_corrected_null_on_independent_matrices_is_not_near_zero():
    """
    The defect being regression-tested: shuffling AFTER fitting the rotation
    reports ~0 because the fitted alignment survives the shuffle. Shuffling
    BEFORE fitting gives the null rotation the same fitting freedom, so on
    two unrelated random matrices it should land well above 0 (~0.05-0.15 for
    n~1700, d=32), not collapse to zero.
    """
    rng = np.random.default_rng(0)
    n, d = 1700, 32
    a = rng.normal(size=(n, d))
    b = rng.normal(size=(n, d))

    null_cos = corrected_null_cosine(a, b, rng)
    median = float(np.median(null_cos))
    assert 0.03 < median < 0.2


def test_corrected_null_recovers_near_one_for_a_rotated_copy():
    """A rotated copy of the same matrix should still align to ~1.0 median cosine."""
    rng = np.random.default_rng(1)
    n, d = 1700, 32
    a = rng.normal(size=(n, d))
    rotation, _ = np.linalg.qr(rng.normal(size=(d, d)))
    rotated = a @ rotation

    real_cos = row_cosine(procrustes_align(rotated, a), a - a.mean(axis=0))
    assert float(np.median(real_cos)) > 0.999
