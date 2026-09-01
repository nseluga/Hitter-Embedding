"""
Unit tests for E.15's pure statistics (src/analysis/model_evaluation_platoon_ceiling.py).

These test the ARITHMETIC only -- reliability, Spearman-Brown, noise subtraction, and
the between/within variance decomposition -- against hand-computable cases. Nothing here
loads a data file: the numbers E.15 reports are results, not invariants, and pinning them
in a test would make the test a transcription of the answer rather than a check on it.
"""

import numpy as np
import pytest

from src.analysis import model_evaluation_platoon_ceiling as ceiling


def test_reliability_is_signal_over_total_and_refuses_negative_variance():
    # a signal variance equal to the noise variance is reliability 1/2, by definition
    assert ceiling.reliability_from_variances(0.001, 0.001) == pytest.approx(0.5)
    # E.15's regime: tiny true split variance against large sampling noise
    assert ceiling.reliability_from_variances(0.00073228, 0.004) == pytest.approx(
        0.00073228 / 0.00473228)
    # a negative variance is an upstream failure and must raise, never be clipped
    with pytest.raises(AssertionError):
        ceiling.reliability_from_variances(-1e-6, 0.001)


def test_spearman_brown_steps_a_half_length_correlation_up_to_full_length():
    # doubling length: r_full = 2r / (1 + r)
    assert ceiling.spearman_brown(0.5) == pytest.approx(2 / 3)
    assert ceiling.spearman_brown(0.0) == pytest.approx(0.0)
    assert ceiling.spearman_brown(1.0) == pytest.approx(1.0)
    # lengthening must never lower the reliability of a positive correlation
    assert ceiling.spearman_brown(0.08) > 0.08
    # a negative half-split projects to a negative full-length reliability, which is a
    # finding (no true signal), not something to floor at zero
    assert ceiling.spearman_brown(-0.15) < 0


def test_noise_subtraction_is_not_clipped_at_zero():
    assert ceiling.noise_corrected_variance(0.005, 0.004) == pytest.approx(0.001)
    # observed spread smaller than the sampling noise alone: negative, and returned as
    # such so the caller can report "consistent with pure noise"
    assert ceiling.noise_corrected_variance(0.0039, 0.0042) == pytest.approx(-0.0003)


def test_between_plus_within_stand_closes_on_the_total():
    rng = np.random.default_rng(11)
    stand = np.array(["L"] * 40 + ["R"] * 60)
    values = np.where(stand == "L", 0.02, -0.01) + rng.normal(0, 0.05, 100)
    weight = rng.uniform(20, 500, 100)
    parts = ceiling.between_within_stand(values, weight, stand)
    assert parts["between_stand"] + parts["within_stand"] == pytest.approx(parts["total"])
    assert parts["total"] == pytest.approx(ceiling.weighted_variance(values, weight))
    # a constant-per-stand predictor (Route A's shape) has zero within-stand variance
    route_a = np.where(stand == "L", 0.02, -0.01)
    assert ceiling.between_within_stand(route_a, weight, stand)["within_stand"] == (
        pytest.approx(0.0, abs=1e-18))
