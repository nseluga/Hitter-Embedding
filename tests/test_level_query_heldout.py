"""
Verification gates for the held-out (2024 wOBA) level query check.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis import level_query_heldout as lqh


def test_partial_correlation_reproduces_known_value():
    """
    Construct a, b as p*c + k*q and r*c + k*q, where q is built by residualising an
    independent vector on c so it is EXACTLY orthogonal to span(1, c) in this sample.
    Residualising a and b on c therefore recovers k*q and k*q exactly (same vector,
    same positive scale), so the analytic partial correlation is exactly 1.0.
    """
    rng = np.random.default_rng(42)
    n = 200
    c = rng.normal(size=n)
    q = lqh.ols_residualise(rng.normal(size=n), c)
    a = 5.0 * c + 2.0 * q + 10.0
    b = -3.0 * c + 2.0 * q - 4.0
    r_partial, lo, hi, n_out = lqh.bootstrap_ci_partial(a, b, c, n_boot=50, seed=7)
    assert r_partial == pytest.approx(1.0, abs=1e-6)
    assert lo == pytest.approx(1.0, abs=1e-3)
    assert n_out == n


def test_heldout_and_in_sample_hitter_sets_overlap():
    """
    The held-out join must be built from the same level_query.csv the in-sample
    analysis used, so the two hitter populations should overlap substantially even
    though the 2024 PA filter drops some of them.
    """
    in_sample = pd.read_csv(lqh.LEVEL_QUERY_CSV)
    merged, n_before, n_after = lqh.build_heldout_frame()
    assert n_before == len(in_sample)
    overlap = set(in_sample["batter"]) & set(merged["batter"])
    assert len(overlap) > 0
    assert len(overlap) == n_after
    # the join can only lose hitters, never invent new ones
    assert set(merged["batter"]) <= set(in_sample["batter"])


def test_heldout_stats_shape_and_no_lookahead():
    """
    heldout_level_query_stats returns the same pooled/low/medium/high key shape as
    the in-sample level_query.json, and the target column it correlates against is
    2024 wOBA, never the in-sample woba_level column.
    """
    merged, _, _ = lqh.build_heldout_frame()
    assert "woba_2024" in merged.columns
    assert "pa_2024" in merged.columns
    stats = lqh.heldout_level_query_stats(merged)
    for key in ("pooled", "low", "medium", "high"):
        assert key in stats
        cell = stats[key]
        for field in ("n", "r_raw", "r_partial", "ci_low", "ci_high"):
            assert field in cell
        assert cell["n"] > 0
        assert cell["ci_low"] <= cell["r_partial"] <= cell["ci_high"]
