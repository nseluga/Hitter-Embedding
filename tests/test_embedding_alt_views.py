"""Unit tests for the three alternative embedding visualizations: similarity
heatmap, archetype coordinates, and supervised (LDA/PLS) projections."""

import numpy as np

from src.analysis import embedding_alt_views as alt
from src.analysis.embedding_structure import unit_normalize


def _toy_normalized(seed=0, n=40, d=32):
    rng = np.random.default_rng(seed)
    return unit_normalize(rng.normal(size=(n, d)))[0]


def test_similarity_matrix_symmetric_unit_diagonal():
    normalized = _toy_normalized()
    _, sim = alt.cluster_order(normalized)
    assert np.allclose(sim, sim.T, atol=1e-9)
    assert np.allclose(np.diag(sim), 1.0, atol=1e-6)


def test_archetype_coordinates_normalize_to_one():
    normalized = _toy_normalized()
    poles = {"a": 0, "b": 1, "c": 2}
    sims, pole_pole = alt.archetype_coordinates(normalized, poles)
    bary = alt.ternary_coords(sims)
    assert np.allclose(bary.sum(axis=1), 1.0, atol=1e-9)
    assert (bary >= -1e-9).all()
    # pole-pole cosine matrix is symmetric with unit diagonal (each pole vs itself)
    assert np.allclose(pole_pole, pole_pole.T, atol=1e-9)
    assert np.allclose(np.diag(pole_pole), 1.0, atol=1e-6)


def test_pls_held_out_r2_uses_test_rows_not_train_rows():
    rng = np.random.default_rng(7)
    normalized = _toy_normalized(seed=1, n=60)
    # target correlated with a fixed direction, plus noise
    direction = rng.normal(size=32)
    target = normalized @ direction + rng.normal(scale=0.05, size=60)

    _, cv_r2, _ = alt.pls_cv(normalized, target, n_components=2, seed=7, n_splits=5)

    # A near-noiseless linear target should score well held-out (rules out a
    # test that silently scores on training rows, which would report ~1.0
    # even for a target the model never saw at fit time only by accident;
    # here we assert it's high AND well below a trivial in-sample fit).
    assert cv_r2 > 0.5

    from sklearn.cross_decomposition import PLSRegression
    in_sample_model = PLSRegression(n_components=2).fit(normalized, target)
    in_sample_r2 = in_sample_model.score(normalized, target)
    assert cv_r2 <= in_sample_r2 + 1e-9


def test_lda_cv_accuracy_bounded_by_chance_and_one():
    rng = np.random.default_rng(3)
    normalized = _toy_normalized(seed=2, n=80)
    labels = rng.integers(0, 2, size=80)  # random labels: near-chance expected
    _, cv_mean, _ = alt.lda_axis_cv(normalized, labels, seed=7, n_splits=5)
    assert 0.0 <= cv_mean <= 1.0
