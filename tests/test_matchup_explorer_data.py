import numpy as np

from src.analysis.matchup_explorer_data import dequantize, double_center, quantize


def test_quantize_round_trip():
    w = np.random.default_rng(0).uniform(0.03, 0.7, (50, 80))
    assert np.abs(dequantize(quantize(w)) - w).max() <= 1e-5


def test_double_center_removes_hitter_and_pitcher_means():
    rng = np.random.default_rng(1)
    w = rng.uniform(0.2, 0.5, (30, 40))
    bf = rng.uniform(1, 100, 40)
    I = double_center(w, bf)
    assert np.allclose(I @ (bf / bf.sum()), 0)  # each hitter's BF-weighted effect is 0
    assert np.allclose(I.mean(0), I.mean(0)[0])  # pitcher columns share one offset
    # additive matrix has no matchup effect
    add = rng.normal(size=30)[:, None] + rng.normal(size=40)[None, :]
    assert np.allclose(double_center(add, bf), 0)
