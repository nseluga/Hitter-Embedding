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


# ---- gates: paired model-vs-model comparison ----
#
# These are the gates behind every head-to-head claim in Phase C. Two failure modes
# matter and both are silent: resampling the wrong unit (a batter's two rows are not
# two independent draws, so row resampling reports intervals that are too tight), and
# a sign-convention slip (the RMSE and rank differences point in OPPOSITE directions
# because lower error is better but higher correlation is better, so a copy-paste
# between them inverts a verdict while still producing plausible numbers).

def paired_case(n_batters=12):
    """
    An eval frame where every batter appears vs BOTH hands — the structure that makes
    the resampling unit matter. Model A orders the hitters correctly; model B is
    ordered by batter id, which is unrelated to talent.
    """
    rows, a, b = [], [], []
    for i in range(n_batters):
        talent = 0.250 + 0.010 * i
        for hand, prior in (("L", 30 + 40 * i), ("R", 30 + 40 * i)):
            rows.append((i, 2023, hand, max(1, prior), talent))
            rows.append((i, 2024, hand, 60, talent))
            a.append({"batter": i, "season": 2024, "p_throws": hand, "pred_woba": talent})
            b.append({"batter": i, "season": 2024, "p_throws": hand,
                      "pred_woba": 0.250 + 0.010 * (n_batters - 1 - i)})
    pa_df = pa_table(rows)
    frame_a, _ = ce.build_eval_frame(pa_df, pd.DataFrame(a), 2024)
    frame_b, _ = ce.build_eval_frame(pa_df, pd.DataFrame(b), 2024)
    return frame_a, frame_b


def test_batter_clusters_partition_the_rows_exactly():
    """Every row lands in exactly one cluster, and a cluster holds one batter's rows."""
    frame_a, _ = paired_case()
    clusters = ce.batter_clusters(frame_a)
    assert len(clusters) == frame_a["batter"].nunique()
    covered = np.concatenate(clusters)
    assert sorted(covered.tolist()) == list(range(len(frame_a)))
    for cluster in clusters:
        assert frame_a["batter"].to_numpy()[cluster].std() == 0  # one batter per cluster


def test_resample_never_splits_a_batter():
    """
    The gate the whole clustering change exists for: a replicate contains a batter's
    rows all-or-nothing. With two rows per batter here, every batter's count in a
    replicate must be even — an odd count means rows were drawn independently.
    """
    frame_a, _ = paired_case()
    clusters = ce.batter_clusters(frame_a)
    rng = np.random.default_rng(0)
    for _ in range(50):
        drawn = frame_a["batter"].to_numpy()[ce._resample(clusters, rng)]
        assert len(drawn) == len(frame_a)
        counts = pd.Series(drawn).value_counts()
        assert (counts % 2 == 0).all(), "a batter's two rows were split across the resample"


def test_paired_rmse_sign_and_direction():
    """Negative rmse_difference favours A, and A is the model that is actually right."""
    frame_a, frame_b = paired_case()
    out = ce.paired_rmse_difference(frame_a, frame_b, n_boot=200, seed=0)
    row = out[out["stratum"] == "all"].iloc[0]
    assert row["rmse_a"] < row["rmse_b"]
    assert row["rmse_difference"] < 0
    assert row["favours_a_share"] > 0.95
    assert row["ci_high"] < 0


def test_paired_rank_sign_is_opposite_to_rmse():
    """
    Higher rank correlation is better, so rank_difference is POSITIVE when A wins
    while rmse_difference is NEGATIVE. Both report favours_a_share the same way.
    """
    frame_a, frame_b = paired_case()
    rank = ce.paired_rank_difference(frame_a, frame_b, n_boot=200, seed=0).set_index("stratum")
    rmse = ce.paired_rmse_difference(frame_a, frame_b, n_boot=200, seed=0).set_index("stratum")
    assert rank.loc["all", "rank_difference"] > 0
    assert rmse.loc["all", "rmse_difference"] < 0
    assert rank.loc["all", "favours_a_share"] > 0.95
    assert rank.loc["all", "ci_low"] > 0


def test_paired_rank_is_calibration_invariant():
    """
    Ordering is all the rank metric sees: shifting and scaling A's predictions leaves
    every rank difference untouched, while its RMSE moves. Catches a rank comparison
    that has silently picked up a level term.
    """
    frame_a, frame_b = paired_case()
    shifted = frame_a.copy()
    shifted["pred_woba"] = shifted["pred_woba"] * 2.0 + 0.1
    base = ce.paired_rank_difference(frame_a, frame_b, n_boot=200, seed=0)
    moved = ce.paired_rank_difference(shifted, frame_b, n_boot=200, seed=0)
    assert np.allclose(base["rank_difference"], moved["rank_difference"])
    assert np.allclose(base["ci_low"], moved["ci_low"])


def test_paired_identical_models_centre_on_zero():
    """A model against itself has no difference, and the interval must contain zero."""
    frame_a, _ = paired_case()
    rmse = ce.paired_rmse_difference(frame_a, frame_a.copy(), n_boot=200, seed=0)
    rank = ce.paired_rank_difference(frame_a, frame_a.copy(), n_boot=200, seed=0)
    assert np.allclose(rmse["rmse_difference"], 0.0)
    assert np.allclose(rank["rank_difference"], 0.0)
    assert (rmse["ci_low"] <= 0).all() and (rmse["ci_high"] >= 0).all()
    assert (rank["ci_low"] <= 0).all() and (rank["ci_high"] >= 0).all()


def test_paired_comparisons_are_deterministic():
    """Same seed, same numbers — the reproducibility gate every reported CI relies on."""
    frame_a, frame_b = paired_case()
    for fn in (ce.paired_rmse_difference, ce.paired_rank_difference):
        first = fn(frame_a, frame_b, n_boot=200, seed=7)
        second = fn(frame_a, frame_b, n_boot=200, seed=7)
        pd.testing.assert_frame_equal(first, second)


def test_paired_rejects_misaligned_frames():
    """Both models must score exactly the same groups or the pairing is meaningless."""
    frame_a, frame_b = paired_case()
    with pytest.raises(AssertionError, match="exactly the same groups"):
        ce.paired_rank_difference(frame_a, frame_b.iloc[:-2], n_boot=50, seed=0)


def test_paired_rank_reports_surviving_draw_count():
    """n_draws is reported so a mostly-degenerate stratum cannot look like a tight CI."""
    frame_a, frame_b = paired_case()
    out = ce.paired_rank_difference(frame_a, frame_b, n_boot=200, seed=0)
    assert (out["n_draws"] > 100).all()
    assert (out["n_batters"] <= out["n_hitters"]).all()


# ---- gates: the min-PA filter is accounted for per stratum ----
#
# The filter censors on eval-season playing time, which is decided after the
# projection is made. Its aggregate drop count reads as neutral hygiene; its
# per-stratum profile does not. These gates make sure the accounting closes and
# that the strata used for it are the leakage-safe PRIOR-exposure ones.

def censoring_case():
    """
    Two low-exposure hitters, one of whom washes out of the season after 8 PA, and
    one high-exposure regular. The washout is exactly the row the filter removes.
    """
    pa_df = pa_table([
        (1, 2023, "L", 20, 0.400), (1, 2024, "L", 100, 0.400),   # low, survives
        (2, 2023, "L", 20, 0.150), (2, 2024, "L", 8, 0.150),     # low, censored
        (3, 2023, "L", 500, 0.300), (3, 2024, "L", 100, 0.300),  # high, survives
    ])
    predictions = pd.DataFrame({
        "batter": [1, 2, 3], "season": 2024, "p_throws": "L",
        "pred_woba": [0.400, 0.150, 0.300],
    })
    return pa_df, predictions


def test_coverage_breaks_the_drop_out_by_stratum():
    pa_df, predictions = censoring_case()
    _, coverage = ce.build_eval_frame(pa_df, predictions, 2024)
    by_stratum = coverage["by_stratum"]
    assert by_stratum["low"]["scored"] == 1
    assert by_stratum["low"]["dropped"] == 1
    assert by_stratum["low"]["share_dropped"] == pytest.approx(0.5)
    assert by_stratum["high"]["dropped"] == 0
    assert by_stratum["high"]["share_dropped"] == pytest.approx(0.0)


def test_coverage_stratum_counts_reconcile_with_the_totals():
    """Per-stratum accounting must add up to the aggregate counts, or one of them lies."""
    pa_df, predictions = censoring_case()
    _, coverage = ce.build_eval_frame(pa_df, predictions, 2024)
    by_stratum = coverage["by_stratum"].values()
    assert sum(s["scored"] for s in by_stratum) == coverage["scored_groups"]
    assert sum(s["dropped"] for s in by_stratum) == coverage["dropped_below_min_eval_pa"]
    assert (sum(s["scored"] + s["dropped"] for s in by_stratum)
            == coverage["actual_groups"])


def test_coverage_records_how_the_dropped_hitters_actually_hit():
    """
    The number that shows the filter is not neutral: the censored hitter's observed
    wOBA, which is what makes the drop non-random with respect to the target.
    """
    pa_df, predictions = censoring_case()
    _, coverage = ce.build_eval_frame(pa_df, predictions, 2024)
    low = coverage["by_stratum"]["low"]
    assert low["mean_woba_dropped"] == pytest.approx(0.150)
    assert low["mean_woba_scored"] == pytest.approx(0.400)
    assert low["mean_woba_dropped"] < low["mean_woba_scored"]


def test_dropped_groups_are_stratified_on_prior_not_held_out_exposure():
    """
    The censored hitter has 20 PRIOR PA and only 8 held-out PA. He must be counted in
    the LOW stratum (prior exposure), which is the leakage-safe quantity — stratifying
    the drop report on held-out PA would put every censored group in one bucket by
    construction and make the report circular.
    """
    pa_df, predictions = censoring_case()
    _, coverage = ce.build_eval_frame(pa_df, predictions, 2024)
    assert coverage["by_stratum"]["low"]["dropped"] == 1
    assert all(coverage["by_stratum"][s]["dropped"] == 0 for s in ("medium", "high"))


def test_raising_the_threshold_only_ever_drops_more():
    """Monotonicity: a stricter filter cannot score a group a looser one excluded."""
    pa_df, predictions = censoring_case()
    scored = []
    for threshold in (1, 25, 200):
        frame, coverage = ce.build_eval_frame(pa_df, predictions, 2024, min_eval_pa=threshold)
        scored.append(coverage["scored_groups"])
        assert len(frame) == coverage["scored_groups"]
    assert scored == sorted(scored, reverse=True)


# ---- gates: the two PA counts are not interchangeable ----
#
# `pa` counts every completed plate appearance; `denominator` excludes IBB, SH and
# INT. wOBA is numerator/denominator, so the denominator is what sets an
# observation's precision, and it is what every weight and threshold must use. The
# two differ by only ~1% on real data, which is exactly why a mix-up survives
# unnoticed — these gates make the distinction fail loudly instead.

def ibb_table(rows):
    """
    PA table where each group gets `n_scoring` scoring PA plus `n_ibb` intentional
    walks. IBB carries no wOBA points and is out of the denominator, so it inflates
    `pa` without contributing to either the wOBA or its precision.
    """
    records = []
    pa_id = 0
    for batter, season, hand, n_scoring, n_ibb, woba in rows:
        for index in range(n_scoring + n_ibb):
            pa_id += 1
            scoring = index < n_scoring
            records.append({
                "batter": batter, "season": season, "p_throws": hand,
                "woba_points": woba if scoring else 0.0,
                "in_denominator": scoring,
                "woba_category": "1B" if scoring else "IBB",
                "pitcher": 900001, "game_pk": pa_id, "at_bat_number": 1,
            })
    return pd.DataFrame(records)


def test_min_eval_pa_counts_scoring_pa_not_total_pa():
    """
    A hitter with 8 scoring PA and 6 intentional walks has 14 total PA. His wOBA is
    still a mean over 8, so he must fail a 10-PA scoring floor — passing on the
    strength of walks he was never allowed to swing at is the bug.
    """
    pa_df = ibb_table([(1, 2023, "L", 40, 0, 0.300), (1, 2024, "L", 8, 6, 0.300),
                       (2, 2023, "L", 40, 0, 0.300), (2, 2024, "L", 40, 0, 0.300)])
    predictions = pd.DataFrame({"batter": [1, 2], "season": 2024, "p_throws": "L",
                                "pred_woba": [0.300, 0.300]})
    frame, coverage = ce.build_eval_frame(pa_df, predictions, 2024, min_eval_pa=10)
    assert coverage["scored_groups"] == 1
    assert coverage["dropped_below_min_eval_pa"] == 1
    assert frame["batter"].tolist() == [2]


def test_rmse_weights_by_denominator_not_total_pa():
    """
    Two groups with equal scoring PA must carry equal weight even when one of them
    logged a pile of intentional walks. Weighting by total PA would let the walks
    buy influence the observation's precision does not justify.
    """
    pa_df = ibb_table([(1, 2023, "L", 40, 0, 0.500), (1, 2024, "L", 30, 30, 0.500),
                       (2, 2023, "L", 40, 0, 0.100), (2, 2024, "L", 30, 0, 0.100)])
    # symmetric errors: predict the midpoint, so equal weights give a clean .200
    predictions = pd.DataFrame({"batter": [1, 2], "season": 2024, "p_throws": "L",
                                "pred_woba": [0.300, 0.300]})
    metrics, _ = ce.evaluate(pa_df, predictions, 2024, min_eval_pa=10)
    row = metrics[metrics["stratum"] == "all"].iloc[0]
    assert row["pa_weighted_rmse"] == pytest.approx(0.200)
    # and the reported columns keep both counts distinct
    assert row["pa"] == 90.0           # (30 scoring + 30 IBB) + (30 scoring + 0 IBB)
    assert row["scoring_pa"] == 60.0   # 30 + 30 scoring PA — the equal weights above


def test_score_reports_both_pa_counts():
    """Both are reported so a reader can see the gap rather than infer which was used."""
    pa_df, predictions = build_case()
    metrics, _ = ce.evaluate(pa_df, predictions, eval_season=2024)
    assert {"pa", "scoring_pa"} <= set(metrics.columns)
    assert (metrics["scoring_pa"] <= metrics["pa"]).all()
