"""
Tests for the D.0 measurements. The gate that matters is two-sided: the R^2
estimator must return 1 on a target the predictors determine exactly AND ~0 on a
target they cannot explain, because an estimator that always returns a high number
would silently mark every flagged feature redundant and empty the D.8 ablation.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis import d0_checks


def synthetic_pitches(n=4000, seed=0):
    """Pitch rows with the kept-feature columns present and a known seed."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "season": rng.choice([2015, 2020, 2023], size=n),
        "plate_x": rng.normal(size=n),
        "plate_z": rng.normal(size=n),
        "pfx_x": rng.normal(size=n),
        "pfx_z": rng.normal(size=n),
        "release_speed": rng.normal(93.0, 3.0, size=n),
        "release_pos_x": rng.normal(size=n),
        "pitch_type": rng.choice(["FF", "SL", "CH"], size=n),
        "stand": rng.choice(["L", "R"], size=n),
        "p_throws": rng.choice(["L", "R"], size=n),
        "balls": rng.integers(0, 4, size=n),
        "strikes": rng.integers(0, 3, size=n),
        "release_extension": rng.normal(6.5, 0.4, size=n),
        "release_pos_y": rng.normal(54.0, 1.0, size=n),
        "release_pos_z": rng.normal(5.8, 0.4, size=n),
        "release_spin_rate": rng.normal(2200.0, 300.0, size=n),
        "spin_axis": rng.uniform(0.0, 360.0, size=n),
    })
    # a target the kept features determine exactly, and one they cannot touch
    frame["effective_speed"] = 2.0 * frame["release_speed"] - 0.5 * frame["plate_x"] + 10.0
    return frame


def test_r2_is_one_when_predictors_determine_the_target():
    rng = np.random.default_rng(1)
    predictors = np.hstack([rng.normal(size=(500, 3)), np.ones((500, 1))])
    target = predictors @ np.array([1.5, -2.0, 0.25, 4.0])
    assert d0_checks._r2(predictors, target) == pytest.approx(1.0, abs=1e-10)


def test_r2_is_near_zero_on_an_unexplainable_target():
    rng = np.random.default_rng(2)
    predictors = np.hstack([rng.normal(size=(5000, 3)), np.ones((5000, 1))])
    target = rng.normal(size=5000)
    assert d0_checks._r2(predictors, target) < 0.01


def test_exact_linear_feature_is_marked_redundant_and_noise_is_not():
    frame = synthetic_pitches()
    result = d0_checks.redundancy_r2(frame, train_seasons=[2015, 2020, 2023],
                                     sample_n=len(frame)).set_index("feature")
    assert result.loc["effective_speed", "verdict"] == "redundant"
    assert result.loc["effective_speed", "r2"] == pytest.approx(1.0, abs=1e-8)
    # every other flagged column was drawn independently, so none may be dropped
    for feature in ("release_extension", "release_pos_y", "release_pos_z",
                    "release_spin_rate", "spin_axis_sin", "spin_axis_cos"):
        assert result.loc[feature, "verdict"] == "survives to D.8"


def test_conditioning_on_an_extra_predictor_removes_it_as_a_target():
    frame = synthetic_pitches()
    result = d0_checks.redundancy_r2(frame, train_seasons=[2015, 2020, 2023],
                                     extra_predictors=["release_extension"],
                                     sample_n=len(frame))
    assert "release_extension" not in set(result["feature"])


def test_redundancy_uses_only_train_seasons():
    frame = synthetic_pitches()
    # a season outside the train list carries a target the kept features cannot fit;
    # if it leaked into the fit, effective_speed would stop being exactly determined
    contaminant = frame.head(1000).copy()
    contaminant["season"] = 2024
    contaminant["effective_speed"] = np.random.default_rng(3).normal(size=len(contaminant))
    result = d0_checks.redundancy_r2(pd.concat([frame, contaminant]),
                                     train_seasons=[2015, 2020, 2023],
                                     sample_n=len(frame) + len(contaminant))
    r2 = result.set_index("feature").loc["effective_speed", "r2"]
    assert r2 == pytest.approx(1.0, abs=1e-8)
