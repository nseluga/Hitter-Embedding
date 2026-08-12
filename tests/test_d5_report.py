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


def _frame(pred, woba, denominator, xwoba=None):
    """
    A minimal build_eval_frame-shaped table: one row per (batter, season, hand).

    `xwoba=None` leaves the second target absent for every row, which is what a group Statcast
    never estimated looks like -- it must LEAVE the xwOBA scoring set rather than be imputed in.
    """
    n = len(pred)
    denominator = np.asarray(denominator, dtype=float)
    return pd.DataFrame({
        "batter": np.arange(n),
        "season": 2024,
        "p_throws": ["L", "R"] * (n // 2),
        "pred_woba": np.asarray(pred, dtype=float),
        "woba": np.asarray(woba, dtype=float),
        "denominator": denominator,
        "pa": denominator,
        "noise_var": np.full(n, 0.0004),
        "xwoba": np.full(n, np.nan) if xwoba is None else np.asarray(xwoba, dtype=float),
        "xwoba_denominator": np.zeros(n) if xwoba is None else denominator,
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


def test_second_target_scores_both_keys_and_drops_unestimated_groups():
    """
    Two things at once, because they are the same defect if either breaks: the table must carry a
    row per (model, target) so the GAP is readable, and a group with no xwOBA must leave the
    second-target set rather than be scored against a null or an imputed value.
    """
    scored_frame = _frame(pred=[0.31, 0.34, 0.33, 0.36], woba=[0.30, 0.31, 0.29, 0.33],
                          denominator=[10, 500, 10, 500], xwoba=[0.30, 0.32, 0.28, 0.34])
    unestimated = _frame(pred=[0.31, 0.34], woba=[0.30, 0.31], denominator=[10, 500])

    table = d5_report.second_target_table({"has_xwoba": scored_frame, "none": unestimated})

    assert set(table["target"]) == set(ce.TARGETS), "a target went unscored"
    with_key = table[(table["model"] == "has_xwoba") & (table["target"] == "xwoba")]
    without = table[(table["model"] == "none") & (table["target"] == "xwoba")]
    assert (with_key["n_hitters"] > 0).any(), "the estimated groups did not reach the second target"
    assert (without["n_hitters"] == 0).all(), "an unestimated group was scored against xwOBA"
    # the noise floor describes REALIZED wOBA's sampling variance and says nothing about xwOBA's
    # own error, so it must not be reported against the second target
    assert table[table["target"] == "xwoba"]["noise_floor"].isna().all()


def test_caller_frame_is_not_mutated():
    """`frames` is read again by the gate verdicts; a shifted column there would poison them."""
    phase_d = _frame(pred=[0.33, 0.36], woba=[0.30, 0.30], denominator=[50, 400])
    frames = {"d": phase_d, "c3_gbm_full": phase_d.assign(pred_woba=phase_d["woba"])}
    before = phase_d["pred_woba"].copy()

    d5_report.debiased_diagnostic(frames, "d")

    pd.testing.assert_series_equal(frames["d"]["pred_woba"], before)
