"""E.13's arithmetic, tested where it is data-free (src/analysis/model_evaluation_bip_value.py)."""

import numpy as np
import pytest

from src.analysis.model_evaluation_bip_value import drift_terms, seam_attribution


def test_seam_attribution_is_inert_when_the_constants_are_right():
    """Identical train and 2024 constants must attribute exactly zero, not epsilon."""
    out = seam_attribution(0.38, 0.92, 0.14, 0.92, 0.14)
    assert out["share_term"] == 0.0 and out["unmeasured_term"] == 0.0
    assert out["total"] == 0.0
    assert out["repriced_under_2024_constants"] == pytest.approx(0.38)


def test_seam_attribution_backs_out_quality_and_closes():
    """q must reproduce the modelled value through the composition's own seam identity."""
    s, u, m = 0.9231315908347482, 0.1414959852073254, 0.37864233821410337
    out = seam_attribution(m, s, u, 0.90, 0.12)
    assert s * out["implied_quality"] + (1 - s) * u == pytest.approx(m)
    assert out["share_term"] + out["unmeasured_term"] == pytest.approx(out["total"], abs=1e-12)
    # a lower observed share and a lower observed unmeasured value both push the modelled
    # number above truth, so both terms are positive here
    assert out["share_term"] > 0 and out["unmeasured_term"] > 0


def test_drift_terms_separate_conditional_from_mix():
    """Same conditionals, different mass -> mix only. Same mass, different conditionals -> conditional only."""
    points_a = np.array([0.2, 0.6])
    points_b = np.array([0.3, 0.5])
    mass_a = np.array([0.7, 0.3])
    mass_b = np.array([0.5, 0.5])

    mix_only = drift_terms(points_a, points_a, mass_a, mass_b)
    assert mix_only["conditional_drift"] == pytest.approx(0.0)
    assert mix_only["mix_drift"] == pytest.approx(0.2 * 0.2 + (-0.2) * 0.6)

    cond_only = drift_terms(points_a, points_b, mass_a, mass_a)
    assert cond_only["mix_drift"] == pytest.approx(0.0)
    assert cond_only["conditional_drift"] == pytest.approx(0.7 * -0.1 + 0.3 * 0.1)


def test_drift_terms_rejects_a_mass_that_is_not_a_distribution():
    points = np.array([0.2, 0.6])
    with pytest.raises(AssertionError):
        drift_terms(points, points, np.array([0.7, 0.7]), np.array([0.5, 0.5]))
