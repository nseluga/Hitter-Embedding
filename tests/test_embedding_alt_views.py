"""Unit tests for the alternative embedding visualizations: the ev_p90 /
whiff_brk_minus_fb supervised (LDA) projection and its density-binned
heatmap."""

import numpy as np
import pandas as pd

from src.analysis import embedding_alt_views as alt
from src.analysis.embedding_structure import tercile_labels, unit_normalize


def _toy_normalized(seed=0, n=90, d=32):
    rng = np.random.default_rng(seed)
    return unit_normalize(rng.normal(size=(n, d)))[0]


def test_lda_cv_accuracy_bounded_by_chance_and_one():
    rng = np.random.default_rng(3)
    normalized = _toy_normalized(seed=2, n=80)
    labels = rng.integers(0, 2, size=80)  # random labels: near-chance expected
    _, cv_mean, _ = alt.lda_axis_cv(normalized, labels, seed=7, n_splits=5)
    assert 0.0 <= cv_mean <= 1.0


def test_lda_axis_cv_returns_model_fit_on_all_rows():
    rng = np.random.default_rng(9)
    normalized = _toy_normalized(seed=3, n=60)
    labels = tercile_labels(pd.Series(rng.normal(size=60))).to_numpy()
    model, cv_mean, cv_std = alt.lda_axis_cv(normalized, labels, seed=7)
    scores = model.transform(normalized)
    assert scores.shape[0] == 60
    assert 0.0 <= cv_mean <= 1.0 and cv_std >= 0.0


def test_supervised_projection_png_renders_nonzero(tmp_path):
    rng = np.random.default_rng(1)
    n = 90
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    ev_tercile = tercile_labels(pd.Series(rng.uniform(90, 110, size=n))).to_numpy()
    whiff_tercile = tercile_labels(pd.Series(rng.normal(size=n))).to_numpy()
    path = tmp_path / "fig_supervised_projection.png"
    alt.fig_supervised_projection(x, y, ev_tercile, whiff_tercile, 0.800, 0.633, path)
    assert path.exists() and path.stat().st_size > 0


def test_similarity_heatmap_png_renders_nonzero(tmp_path):
    rng = np.random.default_rng(2)
    n = 200
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    ev_values = rng.uniform(90, 110, size=n)
    path = tmp_path / "fig_similarity_heatmap.png"
    alt.fig_similarity_heatmap(x, y, ev_values, path)
    assert path.exists() and path.stat().st_size > 0


def test_similarity_heatmap_greys_out_low_count_bins(tmp_path):
    # All points crammed into one bin corner except a couple of outliers ->
    # most bins have < min_per_bin hitters and should be masked, not raise.
    x = np.concatenate([np.zeros(3), [10.0]])
    y = np.concatenate([np.zeros(3), [10.0]])
    ev_values = np.array([95.0, 96.0, 97.0, 110.0])
    path = tmp_path / "fig_similarity_heatmap_sparse.png"
    alt.fig_similarity_heatmap(x, y, ev_values, path, n_bins=4, min_per_bin=5)
    assert path.exists() and path.stat().st_size > 0


def test_whiff_brk_minus_fb_by_batter_still_importable_by_axis_screen():
    # results/embedding_structure/axis_screen.py lazily imports this exact
    # function from this module; keep the name/signature stable.
    from results.embedding_structure import axis_screen
    assert axis_screen.MIN_SWINGS_PER_SLICE == 20
    assert callable(alt.whiff_brk_minus_fb_by_batter)
