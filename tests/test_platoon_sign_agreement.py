"""Tests for src/analysis/platoon_sign_agreement.py."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.platoon_sign_agreement import (
    DEFAULT_FRAME_PATH,
    compute_sign_agreement,
    handedness_only_agreement,
    load_platoon_frame,
    wilson_interval,
)


def make_skewed_independent_frame(n_per_stand=1000, seed=1):
    """
    Independent, uninformative delta_pred (no within-stand signal) whose
    within-stand marginal is skewed positive, like delta_obs's own marginal
    -- the shape that makes a flat 50% chance line the wrong null. Both
    sides are drawn from independent skewed noise, so any within-stand
    agreement above 0.5 here is purely the marginal-skew artifact this fix
    accounts for, not real model skill.
    """
    rng = np.random.default_rng(seed)
    rows = []
    batter_id = 1
    for stand, group_mean in (("L", -0.05), ("R", 0.05)):
        delta_obs = group_mean + (rng.exponential(1.0, n_per_stand) - 0.3)
        delta_pred = group_mean + (rng.exponential(1.0, n_per_stand) - 0.3)
        for i in range(n_per_stand):
            stratum = ["low", "medium", "high"][i % 3]
            rows.append({
                "batter": batter_id,
                "delta_obs": delta_obs[i],
                "delta_pred": delta_pred[i],
                "denom_L": 100.0,
                "denom_R": 100.0,
                "prior_pa": 200.0,
                "stand": stand,
                "weight": 1.0,
                "stratum": stratum,
                "delta_route_a": group_mean,
            })
            batter_id += 1
    return pd.DataFrame(rows)


def make_frame(n_per_stand=40, stand_gap=0.05, within_stand_signal=True, seed=0):
    """
    Synthetic platoon frame.

    `stand_gap` sets how far apart the L and R group means of delta_obs are
    (the handedness effect). `within_stand_signal` controls whether
    delta_pred also tracks each hitter's within-stand deviation, or is
    completely uninformative about it (constant per stand).
    """
    rng = np.random.default_rng(seed)
    rows = []
    batter_id = 1
    for stand, group_mean in (("L", -stand_gap), ("R", stand_gap)):
        within_stand_deviation = rng.normal(0, 1, n_per_stand)
        delta_obs = group_mean + within_stand_deviation
        if within_stand_signal:
            delta_pred = group_mean + within_stand_deviation
        else:
            # Same handedness effect, but the within-stand part is
            # independent noise -- delta_pred carries zero information
            # about which hitters within a stand are above or below
            # their group's mean.
            delta_pred = group_mean + rng.normal(0, 1, n_per_stand)
        for i in range(n_per_stand):
            stratum = ["low", "medium", "high"][i % 3]
            rows.append({
                "batter": batter_id,
                "delta_obs": delta_obs[i],
                "delta_pred": delta_pred[i],
                "denom_L": 100.0,
                "denom_R": 100.0,
                "prior_pa": 200.0,
                "stand": stand,
                "weight": 1.0,
                "stratum": stratum,
                "delta_route_a": group_mean,
            })
            batter_id += 1
    return pd.DataFrame(rows)


def test_perfect_within_stand_recovery_gives_rate_one():
    frame = make_frame(within_stand_signal=True)
    _, summary = compute_sign_agreement(frame)
    assert summary["model_within_stand"]["overall"]["rate"] == pytest.approx(1.0)
    for stratum_name in ("low", "medium", "high"):
        assert summary["model_within_stand"][stratum_name]["rate"] == pytest.approx(1.0)


def test_handedness_confound_is_removed_by_centering():
    frame = make_frame(stand_gap=5.0, within_stand_signal=False)
    _, summary = compute_sign_agreement(frame)

    within_stand_rate = summary["model_within_stand"]["overall"]["rate"]
    raw_rate = summary["raw_model_agreement"]["overall"]["rate"]

    # No within-stand information in delta_pred -> centered_pred is
    # constant (zero) per stand, so agreement is a coin flip.
    assert within_stand_rate == pytest.approx(0.5, abs=0.15)
    # But the huge handedness gap makes the raw, uncentered comparison look
    # almost perfect, because delta_pred always has the same sign as the
    # group's dominant direction.
    assert raw_rate > 0.9
    assert within_stand_rate < raw_rate


def test_wilson_ci_brackets_point_estimate_and_narrows_with_n():
    small_low, small_high = wilson_interval(35, 50)
    large_low, large_high = wilson_interval(350, 500)

    point_estimate = 35 / 50
    assert small_low <= point_estimate <= small_high
    assert large_low <= point_estimate <= large_high

    assert (small_high - small_low) > (large_high - large_low)


def test_handedness_only_agreement_matches_stand_direction():
    frame = make_frame(stand_gap=5.0, within_stand_signal=False, n_per_stand=20)
    flags = handedness_only_agreement(frame)
    # With such a large gap almost every L hitter sits below the overall
    # mean and almost every R hitter sits above it.
    assert flags.mean() > 0.9


def test_route_a_constant_per_stand_is_reported_as_not_used():
    frame = make_frame()
    _, summary = compute_sign_agreement(frame)
    assert summary["route_a_baseline"]["used"] is False
    assert "reason" in summary["route_a_baseline"]


def test_marginal_skew_alone_clears_50_percent_but_not_its_own_null():
    """
    Regression test for the defect: a predictor with zero real within-stand
    signal but a skewed within-stand marginal produces an observed rate
    clearly above the flat 50% line, yet that rate is unremarkable against
    its own permutation null -- it should land inside null_ci95.
    """
    frame = make_skewed_independent_frame()
    _, summary = compute_sign_agreement(frame)
    overall = summary["model_within_stand"]["overall"]

    assert overall["rate"] > 0.53  # clearly above flat chance
    null = overall["permutation_null"]
    assert null["null_ci95"][0] <= overall["rate"] <= null["null_ci95"][1]


def test_low_untied_clears_low_tied_and_ns_add_up():
    """
    Untied low-stratum rows (real hitter-specific predictions) should score
    above tied rows (cold-start hitters sharing one embedding row, no
    hitter-specific signal) -- the low stratum's clearance is not just the
    tied, constant-predictor rows dragging up the average.
    """
    frame = load_platoon_frame(DEFAULT_FRAME_PATH)
    _, summary = compute_sign_agreement(frame)
    low_untied = summary["model_within_stand"]["low_untied"]
    low_tied = summary["model_within_stand"]["low_tied"]
    low = summary["model_within_stand"]["low"]

    assert low_untied["rate"] > low_tied["rate"]
    assert low_tied["n"] + low_untied["n"] == low["n"]


def test_permutation_null_mean_approximates_analytic_independence_rate():
    frame = make_skewed_independent_frame()
    _, summary = compute_sign_agreement(frame)
    for group in ("overall", "low", "medium", "high"):
        cell = summary["model_within_stand"][group]
        assert cell["permutation_null"]["null_mean"] == pytest.approx(
            cell["independence_rate"], abs=0.02)
