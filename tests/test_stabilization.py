"""
Unit tests for the split-half stabilization estimator. The core gate is a synthetic
signal+noise model whose reliability has a closed form, reliability(n) = S/(S + N/n):
the estimator must recover it, proving it computes true reliability, not a lookalike.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis import stabilization as stab


def gaussian_hitters(n_hitters, n_obs, signal_var, noise_var, seed=0):
    """Each hitter has a true talent ~N(0, signal); each obs ~N(talent, noise)."""
    rng = np.random.default_rng(seed)
    talent = rng.normal(0.0, np.sqrt(signal_var), n_hitters)
    rows = []
    for hitter, theta in enumerate(talent):
        obs = rng.normal(theta, np.sqrt(noise_var), n_obs)
        rows.extend({"batter": hitter, "value": v} for v in obs)
    return pd.DataFrame(rows)


def test_spearman_brown_formula():
    assert stab.spearman_brown(0.5) == pytest.approx(2 / 3)
    assert stab.spearman_brown(0.0) == 0.0


def test_reliability_recovers_closed_form():
    # reliability(n) = S / (S + N/n); with S=1, N=4, n=8 -> 1/(1+0.5) = 0.6667
    signal_var, noise_var, n = 1.0, 4.0, 8
    df = gaussian_hitters(4000, n_obs=n, signal_var=signal_var, noise_var=noise_var)
    curve = stab.stabilization_curve(df, "batter", stab.mean_metric("value"),
                                     n_grid=[n], seed=1, n_resamples=8)
    expected = signal_var / (signal_var + noise_var / n)
    assert curve.loc[0, "reliability"] == pytest.approx(expected, abs=0.03)


def test_reliability_increases_with_sample_size():
    df = gaussian_hitters(3000, n_obs=64, signal_var=1.0, noise_var=6.0)
    curve = stab.stabilization_curve(df, "batter", stab.mean_metric("value"),
                                     n_grid=[8, 16, 32, 64], seed=2, n_resamples=4)
    rel = curve["reliability"].to_numpy()
    assert np.all(np.diff(rel) > 0)


def test_n_groups_reflects_survivorship():
    # 100 hitters with 20 obs, 100 with 4 obs; only the first 100 qualify at n=8
    big = gaussian_hitters(100, 20, 1.0, 3.0, seed=3)
    small = gaussian_hitters(100, 4, 1.0, 3.0, seed=4)
    small["batter"] += 1000
    df = pd.concat([big, small], ignore_index=True)
    curve = stab.stabilization_curve(df, "batter", stab.mean_metric("value"),
                                     n_grid=[8], seed=5)
    assert curve.loc[0, "n_groups"] == 100


def test_ratio_metric_handles_zero_denominator():
    df = pd.DataFrame({"num": [1.0, 0.0], "den": [0, 0]})
    assert np.isnan(stab.ratio_metric("num", "den")(df))
    df2 = pd.DataFrame({"num": [1.0, 2.0], "den": [1, 1]})
    assert stab.ratio_metric("num", "den")(df2) == pytest.approx(1.5)


def test_stabilization_point_interpolates_crossing():
    curve = pd.DataFrame({"n": [10, 20, 30], "reliability": [0.3, 0.45, 0.6], "n_groups": [9, 9, 9]})
    # crosses 0.5 between n=20 (0.45) and n=30 (0.6): 20 + (0.05/0.15)*10 = 23.33
    assert stab.stabilization_point(curve, 0.5) == pytest.approx(23.333, abs=0.01)


def test_stabilization_point_nan_when_never_reached():
    curve = pd.DataFrame({"n": [10, 20], "reliability": [0.2, 0.3], "n_groups": [9, 9]})
    assert np.isnan(stab.stabilization_point(curve, 0.5))


def test_curve_is_deterministic_under_seed():
    df = gaussian_hitters(500, 16, 1.0, 4.0)
    a = stab.stabilization_curve(df, "batter", stab.mean_metric("value"), [8, 16], seed=7, n_resamples=3)
    b = stab.stabilization_curve(df, "batter", stab.mean_metric("value"), [8, 16], seed=7, n_resamples=3)
    pd.testing.assert_frame_equal(a, b)
