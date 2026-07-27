"""
Verification gates for the claim-1 metric (ml-engineer blocking gates).

The scoring function is the one piece of code whose bugs are invisible: a wrong
weighting or a leaked stratum still produces plausible numbers, and every model
comparison downstream inherits the error. So the gates here are closed-form —
hand-computable answers, not "it runs" — plus the leakage boundary.

Gates: closed-form RMSE and rank correlation, PA weighting actually weights,
weighted reduces to unweighted at equal PA, strata partition exhaustively at the
exact boundary, prior exposure never sees the evaluated season, coverage accounting
adds up, and a perfect predictor scores zero error.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis import claim1_eval as ce


def pa_table(rows):
    """Build a minimal PA-level eval-target table from (batter, season, hand, n_pa, woba) tuples."""
    records = []
    pa_id = 0
    for batter, season, hand, n_pa, woba in rows:
        for _ in range(n_pa):
            pa_id += 1
            records.append({
                "batter": batter, "season": season, "p_throws": hand,
                "woba_points": woba, "in_denominator": True,
                # a real pitcher id (never a batter here) so the pitcher-batter
                # filter has the columns it needs and drops nothing
                "pitcher": 900001, "game_pk": pa_id, "at_bat_number": 1,
            })
    return pd.DataFrame(records)


# ---- gate: PA-weighted RMSE is closed-form correct ----

def test_pa_weighted_rmse_closed_form():
    # errors of 0.1 (weight 100) and 0.2 (weight 300):
    # sqrt((100*0.01 + 300*0.04) / 400) = sqrt(13/400) = 0.180277...
    actual = [0.300, 0.300]
    predicted = [0.400, 0.500]
    pa = [100, 300]
    expected = np.sqrt((100 * 0.1**2 + 300 * 0.2**2) / 400)
    assert ce.pa_weighted_rmse(actual, predicted, pa) == pytest.approx(expected)


def test_weighting_actually_changes_the_answer():
    """If PA weights were ignored this test fails — the classic silent-weighting bug."""
    actual, predicted = [0.300, 0.300], [0.400, 0.500]
    heavy_on_small_error = ce.pa_weighted_rmse(actual, predicted, [1000, 1])
    heavy_on_big_error = ce.pa_weighted_rmse(actual, predicted, [1, 1000])
    assert heavy_on_small_error < heavy_on_big_error
    assert heavy_on_small_error == pytest.approx(0.1, abs=0.01)
    assert heavy_on_big_error == pytest.approx(0.2, abs=0.01)


def test_weighted_reduces_to_unweighted_at_equal_pa():
    actual, predicted = [0.300, 0.320, 0.280], [0.310, 0.300, 0.300]
    unweighted = np.sqrt(np.mean((np.array(predicted) - np.array(actual)) ** 2))
    assert ce.pa_weighted_rmse(actual, predicted, [50, 50, 50]) == pytest.approx(unweighted)


def test_perfect_predictor_scores_zero_error():
    actual = [0.300, 0.350, 0.280]
    assert ce.pa_weighted_rmse(actual, actual, [10, 20, 30]) == pytest.approx(0.0)


def test_rmse_shape_mismatch_fails_loud():
    with pytest.raises(AssertionError):
        ce.pa_weighted_rmse([0.3, 0.3], [0.4], [1, 1])


# ---- gate: rank correlation measures ordering, not level ----

def test_rank_correlation_closed_form():
    actual = [0.250, 0.300, 0.350, 0.400]
    assert ce.rank_correlation(actual, [0.1, 0.2, 0.3, 0.4]) == pytest.approx(1.0)
    assert ce.rank_correlation(actual, [0.4, 0.3, 0.2, 0.1]) == pytest.approx(-1.0)


def test_rank_correlation_ignores_calibration():
    """A badly calibrated but correctly ordered model must still score 1.0."""
    actual = [0.250, 0.300, 0.350]
    wildly_off_but_ordered = [10.0, 20.0, 30.0]
    assert ce.rank_correlation(actual, wildly_off_but_ordered) == pytest.approx(1.0)


def test_rank_correlation_degenerate_returns_nan():
    assert np.isnan(ce.rank_correlation([0.3, 0.3, 0.3], [0.1, 0.2, 0.3]))
    assert np.isnan(ce.rank_correlation([0.3], [0.1]))


# ---- gate: strata partition exhaustively, at the exact boundary ----

def test_strata_boundaries_are_half_open_and_exhaustive():
    low_cut, high_cut = ce.STRATUM_BOUNDARIES
    values = pd.Series([0, low_cut - 1, low_cut, high_cut - 1, high_cut, 10_000])
    strata = ce.assign_stratum(values)
    assert list(strata) == ["low", "low", "medium", "medium", "high", "high"]
    assert strata.notna().all(), "every value must land in a stratum"


def test_strata_boundaries_derive_from_stabilization():
    """The cuts are anchored on the B.1 n*, not on round numbers."""
    assert ce.STRATUM_BOUNDARIES == (ce.STABILIZATION_N_STAR // 2, ce.STABILIZATION_N_STAR * 2)


# ---- gate: the leakage boundary — prior exposure never sees the eval season ----

def test_prior_exposure_excludes_the_evaluated_season():
    pa_df = pa_table([
        (1, 2022, "L", 40, 0.0),
        (1, 2023, "L", 60, 0.0),
        (1, 2024, "L", 500, 0.0),   # the evaluated season — must NOT count
    ])
    prior = ce.prior_exposure(pa_df, eval_season=2024)
    assert prior.loc[prior["batter"] == 1, "prior_pa"].iloc[0] == 100


def test_prior_exposure_is_side_specific():
    pa_df = pa_table([
        (1, 2023, "L", 30, 0.0),
        (1, 2023, "R", 300, 0.0),
        (1, 2024, "L", 50, 0.0),
    ])
    prior = ce.prior_exposure(pa_df, eval_season=2024).set_index("p_throws")["prior_pa"]
    assert prior["L"] == 30 and prior["R"] == 300


def test_prior_exposure_with_no_prior_seasons_fails_loud():
    pa_df = pa_table([(1, 2024, "L", 50, 0.0)])
    with pytest.raises(AssertionError):
        ce.prior_exposure(pa_df, eval_season=2024)


# ---- gate: end-to-end coverage accounting adds up and nothing is silently dropped ----

def build_case():
    pa_df = pa_table([
        (1, 2023, "L", 20, 0.400),    # low prior exposure
        (1, 2024, "L", 100, 0.400),
        (2, 2023, "L", 200, 0.300),   # medium prior exposure
        (2, 2024, "L", 100, 0.300),
        (3, 2023, "L", 500, 0.250),   # high prior exposure
        (3, 2024, "L", 100, 0.250),
        (4, 2023, "L", 100, 0.350),
        (4, 2024, "L", 5, 0.350),     # too few held-out PA -> dropped
    ])
    predictions = pd.DataFrame({
        "batter": [1, 2, 3, 4], "season": 2024, "p_throws": "L",
        "pred_woba": [0.400, 0.300, 0.250, 0.350],
    })
    return pa_df, predictions


def test_end_to_end_coverage_accounting():
    pa_df, predictions = build_case()
    metrics, coverage = ce.evaluate(pa_df, predictions, eval_season=2024)
    assert coverage["actual_groups"] == 4
    assert coverage["dropped_below_min_eval_pa"] == 1
    assert coverage["dropped_no_prediction"] == 0
    assert coverage["scored_groups"] == 3
    total = (coverage["dropped_below_min_eval_pa"] + coverage["dropped_no_prediction"]
             + coverage["scored_groups"])
    assert total == coverage["actual_groups"], "coverage must partition the actual groups"


def test_end_to_end_strata_assignment_and_perfect_score():
    pa_df, predictions = build_case()
    metrics, _ = ce.evaluate(pa_df, predictions, eval_season=2024)
    by_stratum = metrics.set_index("stratum")
    assert by_stratum.loc["low", "n_hitters"] == 1      # batter 1, 20 prior PA
    assert by_stratum.loc["medium", "n_hitters"] == 1   # batter 2, 200 prior PA
    assert by_stratum.loc["high", "n_hitters"] == 1     # batter 3, 500 prior PA
    assert by_stratum.loc["all", "n_hitters"] == 3
    # predictions are exactly right, so error is zero everywhere
    assert by_stratum.loc["all", "pa_weighted_rmse"] == pytest.approx(0.0)


def test_hitter_with_no_prior_exposure_lands_in_low_not_dropped():
    pa_df = pa_table([
        (1, 2023, "L", 100, 0.300),
        (1, 2024, "L", 100, 0.300),
        (9, 2024, "L", 100, 0.300),   # debutant: zero prior PA vs L
    ])
    predictions = pd.DataFrame({
        "batter": [1, 9], "season": 2024, "p_throws": "L", "pred_woba": [0.300, 0.300],
    })
    frame, coverage = ce.build_eval_frame(pa_df, predictions, eval_season=2024)
    assert coverage["scored_groups"] == 2
    assert frame.set_index("batter").loc[9, "stratum"] == "low"
    assert frame.set_index("batter").loc[9, "prior_pa"] == 0.0


# ---- gate: malformed prediction frames fail loud ----

def test_duplicate_predictions_fail_loud():
    pa_df, predictions = build_case()
    dupes = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
    with pytest.raises(AssertionError):
        ce.build_eval_frame(pa_df, dupes, eval_season=2024)


def test_predictions_for_wrong_season_fail_loud():
    pa_df, predictions = build_case()
    predictions = predictions.assign(season=2023)
    with pytest.raises(AssertionError):
        ce.build_eval_frame(pa_df, predictions, eval_season=2024)


def test_non_finite_predictions_fail_loud():
    pa_df, predictions = build_case()
    predictions.loc[0, "pred_woba"] = np.nan
    with pytest.raises(AssertionError):
        ce.build_eval_frame(pa_df, predictions, eval_season=2024)


# ---- gate: noise-floor deconvolution ----

def test_noiseless_target_gives_zero_floor():
    """Every PA identical -> no sampling noise -> model_rmse equals raw RMSE."""
    pa_df = pa_table([
        (1, 2023, "L", 100, 0.300),
        (1, 2024, "L", 100, 0.300),   # every PA the same value: zero within-group variance
        (2, 2023, "L", 100, 0.300),
        (2, 2024, "L", 100, 0.300),
    ])
    preds = pd.DataFrame({"batter": [1, 2], "season": 2024, "p_throws": "L",
                          "pred_woba": [0.350, 0.350]})
    metrics, _ = ce.evaluate(pa_df, preds, eval_season=2024)
    row = metrics.set_index("stratum").loc["all"]
    assert row["noise_floor"] == pytest.approx(0.0)
    assert row["model_rmse"] == pytest.approx(row["pa_weighted_rmse"])
    assert row["pa_weighted_rmse"] == pytest.approx(0.050)


def test_sampling_noise_closed_form():
    """Half the PA at 1.0 and half at 0.0 -> within-var 0.25*n/(n-1), noise_var = that / n."""
    rows = [(1, 2024, "L", 50, 1.0), (1, 2024, "L", 50, 0.0)]
    pa_df = pa_table(rows)
    got = ce.sampling_noise(pa_df, 2024).iloc[0]["noise_var"]
    n = 100
    expected = (0.25 * n / (n - 1)) / n
    assert got == pytest.approx(expected)


def test_deconvolution_is_pythagorean():
    assert ce.deconvolve(0.05, 0.03) == pytest.approx(0.04)
    assert ce.deconvolve(0.0472, 0.0472) == pytest.approx(0.0)


def test_deconvolution_clamps_below_the_floor():
    """A model scoring under its floor clamps to zero rather than going imaginary."""
    assert ce.deconvolve(0.02, 0.05) == 0.0


def test_model_rmse_never_exceeds_raw_rmse():
    pa_df, predictions = build_case()
    metrics, _ = ce.evaluate(pa_df, predictions, eval_season=2024)
    assert (metrics["model_rmse"].fillna(0) <= metrics["pa_weighted_rmse"].fillna(0) + 1e-12).all()


def test_skill_score_endpoints():
    reference = 0.0370
    assert ce.skill_score(reference, reference) == pytest.approx(0.0)   # knows nothing
    assert ce.skill_score(0.0, reference) == pytest.approx(1.0)         # perfect
    assert ce.skill_score(0.0282, reference) == pytest.approx(0.419, abs=0.002)


def test_skill_score_rejects_zero_reference():
    with pytest.raises(AssertionError):
        ce.skill_score(0.01, 0.0)


# ---- gate: the test season is refused ----

def test_test_season_is_guarded():
    with pytest.raises(AssertionError, match="frozen TEST season"):
        ce.assert_not_test_season(2025)
    ce.assert_not_test_season(2024)  # val is fine
