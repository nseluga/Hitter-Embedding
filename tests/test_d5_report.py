"""
Gate for the debiased-comparison diagnostic (D5-R18).

The number this function produces is quotable and dangerous: "Phase D and C.3-full are tied"
is only true of a model that had its level error removed, which is not the model that ships.
So the two things worth asserting are that the shift really is the PA-WEIGHTED mean excess
(an unweighted mean would remove the wrong amount and the tie would be an artefact) and that
the diagnostic leaves the caller's frame untouched, since `frames` is scored again downstream.
"""

import numpy as np
import pandas as pd

from src.analysis import claim1_eval as ce, d5_report


def _frame(pred, woba, denominator):
    """A minimal build_eval_frame-shaped table: one row per (batter, season, hand)."""
    n = len(pred)
    return pd.DataFrame({
        "batter": np.arange(n),
        "season": 2024,
        "p_throws": ["L", "R"] * (n // 2),
        "pred_woba": np.asarray(pred, dtype=float),
        "woba": np.asarray(woba, dtype=float),
        "denominator": np.asarray(denominator, dtype=float),
        "stratum": pd.Categorical(["low", "high"] * (n // 2), categories=list(ce.STRATUM_NAMES)),
    })


def test_removed_excess_is_pa_weighted():
    """
    The heavy rows carry the bias. An unweighted mean would remove 0.035 here; the weighted
    one removes 0.049, and the difference is the whole tie-versus-loss reading.
    """
    phase_d = _frame(pred=[0.31, 0.34, 0.33, 0.36], woba=[0.30, 0.30, 0.30, 0.30],
                     denominator=[10, 500, 10, 500])
    frames = {"d": phase_d, "c3_gbm_full": _frame(
        pred=[0.30, 0.30, 0.30, 0.30], woba=[0.30, 0.30, 0.30, 0.30],
        denominator=[10, 500, 10, 500])}

    excess, table = d5_report.debiased_diagnostic(frames, "d")

    # (10*.01 + 500*.04 + 10*.03 + 500*.06) / 1020
    assert np.isclose(excess, 0.0494117647), excess
    assert not np.isclose(excess, 0.035), "an unweighted mean would give 0.035"
    assert set(table["stratum"]) <= set(ce.STRATUM_NAMES) | {"all"}


def test_debiasing_zeroes_the_weighted_bias():
    """After the shift the weighted mean excess is zero -- that is what 'debiased' means."""
    phase_d = _frame(pred=[0.32, 0.35, 0.34, 0.37], woba=[0.30, 0.31, 0.29, 0.33],
                     denominator=[20, 300, 40, 600])
    frames = {"d": phase_d, "c3_gbm_full": phase_d.assign(pred_woba=phase_d["woba"])}

    excess, _ = d5_report.debiased_diagnostic(frames, "d")
    shifted = phase_d["pred_woba"] - excess
    residual = np.average(shifted - phase_d["woba"], weights=phase_d["denominator"])

    assert abs(residual) < 1e-12, residual


def test_caller_frame_is_not_mutated():
    """`frames` is read again by the gate verdicts; a shifted column there would poison them."""
    phase_d = _frame(pred=[0.33, 0.36], woba=[0.30, 0.30], denominator=[50, 400])
    frames = {"d": phase_d, "c3_gbm_full": phase_d.assign(pred_woba=phase_d["woba"])}
    before = phase_d["pred_woba"].copy()

    d5_report.debiased_diagnostic(frames, "d")

    pd.testing.assert_series_equal(frames["d"]["pred_woba"], before)
