"""Unit tests for the cross-arm agreement check."""

import numpy as np
import pandas as pd
import pytest

from src.analysis import model_visualization_arm_agreement as agreement


def _write_stats(path, batters, embedding_indices):
    pd.DataFrame({"batter": batters,
                  "embedding_index": embedding_indices,
                  "stand": ["R"] * len(batters)}).to_csv(path, index=False)


def test_shared_batter_rows_matches_on_batter_not_index(tmp_path):
    """A batter present in both builds must map to each build's own row."""
    reference_path = tmp_path / "reference.csv"
    comparison_path = tmp_path / "comparison.csv"
    _write_stats(reference_path, [100, 200, 300], [1, 2, 3])
    _write_stats(comparison_path, [300, 200, 400], [1, 2, 3])

    batters, reference_rows, comparison_rows = agreement.shared_batter_rows(
        reference_path, comparison_path)

    assert batters.tolist() == [200, 300]
    assert reference_rows.tolist() == [1, 2]
    assert comparison_rows.tolist() == [1, 0]


def test_shared_batter_rows_rejects_disjoint_builds(tmp_path):
    reference_path = tmp_path / "reference.csv"
    comparison_path = tmp_path / "comparison.csv"
    _write_stats(reference_path, [1, 2], [1, 2])
    _write_stats(comparison_path, [3, 4], [1, 2])

    with pytest.raises(AssertionError):
        agreement.shared_batter_rows(reference_path, comparison_path)


def test_arm_agreement_recovers_a_rotated_copy():
    """A rotated, rescaled copy of the same space must agree near cosine 1."""
    rng = np.random.default_rng(0)
    reference_matrix = rng.normal(size=(120, 8))
    rotation, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    comparison_matrix = 2.5 * reference_matrix @ rotation

    per_hitter, summary = agreement.arm_agreement(
        reference_matrix, comparison_matrix, np.arange(120), n_boot=50)

    assert summary["median_cosine_real"] > 0.99
    assert abs(summary["median_cosine_null"]) < 0.5
    assert summary["separates_from_null"]
    assert len(per_hitter) == 120


def test_arm_agreement_finds_no_signal_in_unrelated_spaces():
    """Independent spaces must not separate from the shuffled null."""
    rng = np.random.default_rng(1)
    reference_matrix = rng.normal(size=(200, 8))
    comparison_matrix = rng.normal(size=(200, 8))

    _, summary = agreement.arm_agreement(
        reference_matrix, comparison_matrix, np.arange(200), n_boot=200)

    assert abs(summary["median_cosine_real"]) < 0.3
    assert not summary["separates_from_null"]
