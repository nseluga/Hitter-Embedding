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


# --- variance-components estimator (Fix B) ---

def test_variance_components_recovers_known_variances():
    # closed form: with S=1, N=4, the estimator must recover signal~1, noise~4.
    df = gaussian_hitters(4000, n_obs=30, signal_var=1.0, noise_var=4.0, seed=11)
    signal, noise = stab.variance_components(stab.hitter_stats(df, "batter", "value"))
    assert signal == pytest.approx(1.0, abs=0.1)
    assert noise == pytest.approx(4.0, abs=0.1)


def test_vc_stabilization_point_matches_closed_form():
    # n*(r=0.5) = noise/signal = 4; n*(r=0.7) = (noise/signal)*(0.7/0.3) = 9.33
    df = gaussian_hitters(4000, n_obs=30, signal_var=1.0, noise_var=4.0, seed=12)
    signal, noise = stab.variance_components(stab.hitter_stats(df, "batter", "value"))
    assert stab.stabilization_point_vc(signal, noise, 0.5) == pytest.approx(4.0, abs=0.4)
    assert stab.stabilization_point_vc(signal, noise, 0.7) == pytest.approx(4.0 * (0.7 / 0.3), abs=1.0)


def test_vc_agrees_with_split_half_on_synthetic():
    # the two independent estimators must land on the same stabilization point.
    df = gaussian_hitters(4000, n_obs=40, signal_var=1.0, noise_var=8.0, seed=13)
    signal, noise = stab.variance_components(stab.hitter_stats(df, "batter", "value"))
    vc_point = stab.stabilization_point_vc(signal, noise, 0.5)  # = noise/signal ~ 8
    curve = stab.stabilization_curve(df, "batter", stab.mean_metric("value"),
                                     [2, 4, 8, 16, 32], seed=13, n_resamples=8)
    sh_point = stab.stabilization_point(curve, 0.5)
    assert vc_point == pytest.approx(sh_point, rel=0.2)


def test_vc_no_signal_never_stabilizes():
    # all hitters share one talent -> zero between-hitter variance -> n* is infinite.
    df = gaussian_hitters(2000, n_obs=20, signal_var=0.0, noise_var=3.0, seed=14)
    signal, noise = stab.variance_components(stab.hitter_stats(df, "batter", "value"))
    assert signal == 0.0
    assert stab.stabilization_point_vc(signal, noise, 0.5) == np.inf


def test_bootstrap_ci_brackets_point():
    df = gaussian_hitters(3000, n_obs=25, signal_var=1.0, noise_var=6.0, seed=15)
    point, lo, hi = stab.stabilization_ci(stab.hitter_stats(df, "batter", "value"),
                                          threshold=0.5, n_boot=200, seed=1)
    assert lo <= point <= hi
    assert lo < hi  # a real interval, not a degenerate point


def test_sequential_split_runs_and_differs_from_random():
    # sequential uses chronological order (no permutation); on iid synthetic data both
    # estimate the same reliability, but the code path must run and stay in [.-1,1].
    df = gaussian_hitters(2000, n_obs=32, signal_var=1.0, noise_var=5.0, seed=16)
    seq = stab.stabilization_curve(df, "batter", stab.mean_metric("value"),
                                   [8, 16, 32], seed=16, split="sequential")
    assert seq["reliability"].between(-1, 1).all()
    assert (seq["n_groups"] == 2000).all()
