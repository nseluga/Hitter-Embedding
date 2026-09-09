"""Unit tests for embedding structure: normalization, clustering, and the
neighbour-purity permutation null."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import embedding_structure as structure

TROUT, JUDGE, OHTANI = 545361, 592450, 660271
_HAS_DATA = Path(structure.EVAL_TARGETS_PA_PATH).exists()


def test_unit_normalize_rows_have_norm_one():
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(50, 32)) * rng.uniform(1, 20, size=(50, 1))
    normalized, _ = structure.unit_normalize(matrix)
    row_norms = np.linalg.norm(normalized, axis=1)
    assert np.allclose(row_norms, 1.0, atol=1e-9)


def test_silhouette_picks_planted_k_on_separated_clusters():
    rng = np.random.default_rng(1)
    centers = rng.normal(scale=20, size=(3, 32))
    points = np.concatenate([
        center + rng.normal(scale=0.5, size=(60, 32)) for center in centers
    ])
    normalized, _ = structure.unit_normalize(points)
    scores, best_k = structure.silhouette_curve(normalized, k_range=range(2, 7))
    assert best_k == 3


def test_neighbour_purity_random_attribute_lands_inside_null_interval():
    rng = np.random.default_rng(2)
    matrix, _ = structure.unit_normalize(rng.normal(size=(200, 32)))
    neighbour_idx = structure.nearest_neighbour_indices(matrix)
    random_attribute = rng.integers(0, 2, size=200)

    result = structure.purity_test(neighbour_idx, random_attribute, n_permutations=200)

    assert result["null_ci95"][0] <= result["observed_purity"] <= result["null_ci95"][1]


def test_clustering_uses_32d_not_2d_projection():
    rng = np.random.default_rng(3)
    centers = rng.normal(scale=15, size=(4, 32))
    points = np.concatenate([
        center + rng.normal(scale=0.5, size=(40, 32)) for center in centers
    ])
    normalized, _ = structure.unit_normalize(points)
    labels = structure.cluster_labels(normalized, k=4)

    # Recomputing the 2-D projection with a different random seed (and
    # discarding it) must not change the cluster labels, because clustering
    # is run on the 32-D vectors and never consumes the 2-D coordinates.
    structure.project(normalized, seed=7)
    structure.project(normalized, seed=999)
    labels_again = structure.cluster_labels(normalized, k=4)
    assert np.array_equal(labels, labels_again)


def test_career_pitcher_batter_ids_count_and_keeps_stars():
    if not _HAS_DATA:
        return  # requires the raw eval_targets_pa parquet, not present in every checkout
    pitcher_ids = structure.career_pitcher_batter_ids()
    assert len(pitcher_ids) == 1927
    assert TROUT not in pitcher_ids
    assert JUDGE not in pitcher_ids
    assert OHTANI not in pitcher_ids

    hitter_stats = pd.read_csv(structure.HITTER_STATS_PATH)
    flagged = hitter_stats[hitter_stats["batter"].isin(pitcher_ids)]
    assert len(flagged) == 198
    assert (flagged["stratum"] == "low").all()


def test_load_hitters_excludes_all_career_pitcher_batters():
    if not (_HAS_DATA and Path(structure.DEFAULT_CHECKPOINT_DIR).exists()):
        return  # requires raw data + trained checkpoints
    pitcher_ids = structure.career_pitcher_batter_ids()
    _, hitters, _ = structure.load_hitters(
        structure.DEFAULT_CHECKPOINT_DIR, structure.DEFAULT_ARM,
        structure.HITTER_STATS_PATH, structure.NAMES_PATH)
    assert not hitters["batter"].isin(pitcher_ids).any()
    assert TROUT in hitters["batter"].values
    assert JUDGE in hitters["batter"].values
    assert OHTANI in hitters["batter"].values
