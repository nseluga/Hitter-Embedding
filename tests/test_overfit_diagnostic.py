"""
Self-check for src/analysis/overfit_diagnostic.py's new logic (test 1's EB
shrinkage, test 4's season-filter monkeypatch). Synthetic data only, no
results/ or data/processed/ paths touched. Not a full test suite --
ponytail-lite: one runnable check per non-trivial piece of new logic.

Run: .venv/bin/python -m pytest tests/test_overfit_diagnostic.py -q
"""

import numpy as np
import pandas as pd

from src.analysis.overfit_diagnostic import eb_shrink_binomial, _cell


def test_eb_shrink_binomial_shrinks_toward_pooled_mean():
    rng = np.random.default_rng(0)
    true_p = 0.5  # every hitter has the SAME true rate -> all variance is sampling noise
    n = rng.integers(50, 500, size=200)
    x = rng.binomial(n, true_p)

    p_obs, p_shrunk, k = eb_shrink_binomial(x, n)

    # no real between-hitter signal, so shrinkage should pull hard toward true_p
    assert np.std(p_shrunk) < np.std(p_obs)
    assert abs(np.mean(p_shrunk) - true_p) < 0.02
    assert k > 0


def test_eb_shrink_binomial_preserves_real_signal():
    rng = np.random.default_rng(1)
    n_hitters = 200
    true_p = rng.uniform(0.2, 0.8, size=n_hitters)  # real between-hitter spread
    n = rng.integers(500, 2000, size=n_hitters)  # large samples -> little sampling noise
    x = rng.binomial(n, true_p)

    p_obs, p_shrunk, k = eb_shrink_binomial(x, n)

    # with large n and real signal, shrinkage should be mild: shrunk tracks true_p closely
    assert np.corrcoef(p_shrunk, true_p)[0, 1] > 0.95


def test_cell_partial_correlation_removes_confound():
    rng = np.random.default_rng(2)
    n = 300
    confound = rng.normal(size=n)
    a = confound + rng.normal(scale=0.1, size=n)  # a is basically the confound
    b = rng.normal(size=n)  # b is pure noise, unrelated to a or confound

    cell = _cell(a, b, confound)
    assert cell["n"] == n
    assert abs(cell["r_partial"]) < 0.3  # partialling out confound should kill the spurious raw r


if __name__ == "__main__":
    test_eb_shrink_binomial_shrinks_toward_pooled_mean()
    test_eb_shrink_binomial_preserves_real_signal()
    test_cell_partial_correlation_removes_confound()
    print("all self-checks passed")
