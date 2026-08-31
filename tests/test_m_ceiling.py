"""
Phase M §9 verification — BLOCKING. No Phase M number is committed until these pass.

Four checks, in the spec's order:

  1. Planted tau2  — Route A's subtraction recovers it, and the reliability -> ceiling map
                     matches the Monte-Carlo best-possible correlation. This is the
                     load-bearing one: every headline number is sqrt(reliability).
  2. Planted zero  — a true tau2 of zero must surface as reliability ~0 and, when the
                     estimate lands negative, as a NEGATIVE number carrying `degenerate`.
                     A silent clip at zero would turn "unmeasurable" into "small but real".
  3. Reproduction  — the committed E.15 pooled numbers, and Route B's tau2, reproduce from
                     source before any module is edited. (Gate 3c, on M.1's `delta_pred`,
                     lives in test_m_report.py where the M.1 frame is built.)
  4. Route C sim   — the split-half estimator, run on simulated data with a known tau2 and
                     large exposures, recovers a positive reliability. This is what
                     licenses reading the real −0.366 against a simulated null.

The exposure profile for 1, 2 and 4 is the REAL one from `e5_platoon_frame.csv` — a
planted-recovery check on tidy homoscedastic hitters would not exercise the skew that
makes the 2024 differential hard to measure.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis import e_platoon_ceiling as e15
from src.analysis import m_ceiling

REPO_ROOT = Path(__file__).resolve().parents[1]
FRAME_PATH = REPO_ROOT / "results/phase_e/e5_platoon_frame.csv"
E15_JSON = REPO_ROOT / "results/phase_e/e15_ceiling.json"
C2_PATH = REPO_ROOT / "results/phase_c/c2_prior_parameters.csv"
EVAL_TARGETS = REPO_ROOT / "data/processed/eval_targets_pa.parquet"

# The committed numbers this phase builds on (docs/phase-m-spec.md §M.0).
COMMITTED_OBSERVED_WITHIN_STAND = 0.00419563
COMMITTED_MEAN_SAMPLING = 0.00407848
COMMITTED_ROUTE_B_TAU2 = 0.00059034
TOLERANCE = 5e-8  # the spec quotes eight decimals; agreement is asserted at that precision


@pytest.fixture(scope="module")
def sampling_profile():
    """Per-hitter sampling variance and E.5 weights, rebuilt from source, not transcribed."""
    if not (FRAME_PATH.exists() and EVAL_TARGETS.exists()):
        pytest.skip("E.5 frame or PA-level eval targets absent")
    frame = pd.read_csv(FRAME_PATH)
    pa_df = pd.read_parquet(EVAL_TARGETS)
    by_batter, by_league = e15.per_pa_variance_tables(pa_df)
    attached = e15.attach_sampling_variance(frame, by_batter, by_league)
    return attached


# ------------------------------------------------------------------ §9.1 planted tau2

def test_route_a_recovers_a_planted_tau2(sampling_profile):
    """Plant a tau2 the size of Route B's, and check the subtraction returns it unbiased."""
    recovery = m_ceiling.recover_route_a(
        COMMITTED_ROUTE_B_TAU2, sampling_profile["sampling_var"],
        sampling_profile["weight"], n_draws=400, seed=11)
    # unbiased to within a Monte-Carlo standard error of the mean
    se = recovery["recovered_sd"] / np.sqrt(recovery["n_draws"])
    assert abs(recovery["recovered_mean"] - COMMITTED_ROUTE_B_TAU2) < 4 * se, recovery
    # and the estimator's own spread is the pre-registered fragility, stated as a number
    assert recovery["recovered_sd"] > 0


def test_ceiling_formula_matches_monte_carlo_best_possible_correlation(sampling_profile):
    """
    THE load-bearing check: sqrt(reliability) is the ceiling on the achievable correlation.

    The best available predictor is true skill itself, so its correlation with the
    simulated observation IS the ceiling. For a weighted Pearson the analytic value must be
    matched to Monte-Carlo error — that is the map, verified.

    For the weighted SPEARMAN that E.5 actually reports, the map is an approximation whose
    error changes sign with the exposure skew, so the assertion is on the SIZE of the gap
    (within 10% relative) and the two regimes are pinned separately below. Asserting
    'Spearman <= analytic' would encode a conservatism that does not hold here.
    """
    result = m_ceiling.monte_carlo_ceiling(
        COMMITTED_ROUTE_B_TAU2, sampling_profile["sampling_var"],
        sampling_profile["weight"], n_draws=300, seed=7)
    pearson_se = result["mc_pearson_sd"] / np.sqrt(result["n_draws"])
    assert abs(result["mc_pearson_mean"] - result["ceiling_rank_corr"]) < 4 * pearson_se, result
    relative_gap = result["mc_spearman_mean"] / result["ceiling_rank_corr"] - 1.0
    assert abs(relative_gap) < 0.10, result


def test_rank_ceiling_gap_changes_sign_with_the_exposure_skew(sampling_profile):
    """
    Why the check above cannot assert a direction, pinned as a fact.

    Homoscedastic hitters give the textbook result — the rank ceiling sits BELOW
    sqrt(reliability). The real 2024 profile, whose per-hitter sampling variance spans a
    factor of ~36, puts it ABOVE: rank correlation shrugs off the heavy-tailed observations
    that the low-exposure hitters contribute. The consequence is reported, not hidden —
    fraction-of-ceiling under the analytic ceiling runs a few percent relatively high.
    """
    weight = sampling_profile["weight"].to_numpy()
    real = sampling_profile["sampling_var"].to_numpy()
    assert real.max() / real.min() > 20, "the exposure skew this test is about is absent"
    flat = np.full_like(real, float(np.average(real, weights=weight)))

    skewed = m_ceiling.monte_carlo_ceiling(COMMITTED_ROUTE_B_TAU2, real, weight,
                                           n_draws=300, seed=7)
    even = m_ceiling.monte_carlo_ceiling(COMMITTED_ROUTE_B_TAU2, flat, weight,
                                         n_draws=300, seed=7)
    assert even["ceiling_rank_corr"] == pytest.approx(skewed["ceiling_rank_corr"]), \
        "the analytic ceiling depends only on the mean, so the two must agree"
    assert even["mc_spearman_mean"] < even["ceiling_rank_corr"], even
    assert skewed["mc_spearman_mean"] > skewed["ceiling_rank_corr"], skewed


def test_ceiling_map_is_monotone_and_bounded():
    """Reliability is a probability and the ceiling is its root — check the map's shape."""
    previous = -1.0
    for tau2 in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
        row = m_ceiling.ceiling_from_variances(tau2, COMMITTED_MEAN_SAMPLING)
        assert 0.0 < row["reliability"] < 1.0
        assert row["ceiling_rank_corr"] > previous
        assert row["ceiling_rank_corr"] == pytest.approx(np.sqrt(row["reliability"]))
        previous = row["ceiling_rank_corr"]


# ------------------------------------------------------------------ §9.2 planted zero

def test_planted_zero_gives_near_zero_reliability(sampling_profile):
    """With no true skill, Route A's estimate must centre on zero, not on something small."""
    recovery = m_ceiling.recover_route_a(
        0.0, sampling_profile["sampling_var"], sampling_profile["weight"],
        n_draws=400, seed=3)
    se = recovery["recovered_sd"] / np.sqrt(recovery["n_draws"])
    assert abs(recovery["recovered_mean"]) < 4 * se, recovery
    # roughly half the draws land negative — the reason negatives cannot be clipped
    assert 0.3 < recovery["share_negative"] < 0.7, recovery


def test_negative_tau2_is_emitted_negative_and_flagged():
    """A negative estimate is a finding about the noise model. Never floored, never hidden."""
    row = m_ceiling.route_a_tau2(observed_within_stand_var=0.0040, mean_sampling_var=0.0042)
    assert row["tau2"] < 0, row
    assert row["tau2"] == pytest.approx(0.0040 - 0.0042)
    assert row["degenerate"] is True
    assert np.isnan(row["reliability"]) and np.isnan(row["ceiling_rank_corr"])

    downstream = m_ceiling.ceiling_from_variances(row["tau2"], 0.0042)
    assert downstream["degenerate"] is True
    assert downstream["tau2"] < 0, "the flag propagates but the sign must too"


def test_stabilization_threshold_is_infinite_for_a_degenerate_tau2():
    """PA* on a non-positive tau2 is 'never stabilizes', reported as inf rather than clipped."""
    assert m_ceiling.stabilization_pa(0.25, -1e-5) == float("inf")
    assert m_ceiling.stabilization_pa(0.25, 0.0) == float("inf")
    assert m_ceiling.stabilization_pa(0.25, 0.00059034) == pytest.approx(423.5, rel=1e-3)


def test_fragility_band_moves_route_a_by_about_a_factor_of_two():
    """
    The pre-registered claim behind choosing B' as primary, asserted rather than asserted-to.

    Route A is 0.004196 − 0.004078; a 3% error in the sampling term is 0.00012, the size of
    the estimate itself. The band must therefore span at least a factor of two in tau2 —
    if it did not, the structural-fragility argument would be wrong.
    """
    band = m_ceiling.fragility_band(COMMITTED_OBSERVED_WITHIN_STAND, COMMITTED_MEAN_SAMPLING)
    assert band["scales"] == [1.0, 0.97, 1.03]
    low, high = min(band["tau2"]), max(band["tau2"])
    assert low < 0 < high, band  # the 1.03 arm drives tau2 negative outright
    assert band["any_degenerate"] is True, band


# ------------------------------------------------------------------ §9.3 reproduction

def test_e15_pooled_numbers_reproduce_from_source(sampling_profile):
    """
    Gate 3b: the committed E.15 pooled inputs come back from the frame and the C.2 file,
    before any module is edited. A mismatch means the artifact on disk is not what the
    code produces, and no Phase M number may be built on it.
    """
    if not E15_JSON.exists():
        pytest.skip("E.15 artifact absent")
    committed = json.loads(E15_JSON.read_text())

    decomposition = e15.between_within_stand(
        sampling_profile["delta_obs"], sampling_profile["weight"], sampling_profile["stand"])
    assert decomposition["within_stand"] == pytest.approx(
        committed["part1_ceiling"]["decomposition_from_e_eval"]["delta_obs"]["within_stand"],
        abs=1e-12)
    assert decomposition["within_stand"] == pytest.approx(
        COMMITTED_OBSERVED_WITHIN_STAND, abs=TOLERANCE)

    c2 = e15.load_c2_components(C2_PATH)
    table = e15.ceiling_table(sampling_profile, c2)
    pooled = table[table["stand"] == "pooled"].iloc[0]
    committed_pooled = next(row for row in committed["part1_ceiling"]["by_stand"]
                            if row["stand"] == "pooled")
    for field in ("tau2_split_true", "mean_sampling_var_weighted",
                  "reliability_variance_ratio", "ceiling_rank_corr"):
        assert pooled[field] == pytest.approx(committed_pooled[field], abs=1e-12), field
    assert pooled["mean_sampling_var_weighted"] == pytest.approx(
        COMMITTED_MEAN_SAMPLING, abs=TOLERANCE)


def test_route_a_reproduces_the_spec_value(sampling_profile):
    """Route A recomputed from source, not transcribed (spec §M.0 'Recompute from source')."""
    decomposition = e15.between_within_stand(
        sampling_profile["delta_obs"], sampling_profile["weight"], sampling_profile["stand"])
    c2 = e15.load_c2_components(C2_PATH)
    pooled = e15.ceiling_table(sampling_profile, c2)
    mean_sampling = float(pooled[pooled["stand"] == "pooled"]["mean_sampling_var_weighted"].iloc[0])
    route_a = m_ceiling.route_a_tau2(decomposition["within_stand"], mean_sampling)
    assert route_a["tau2"] == pytest.approx(0.00011715, abs=TOLERANCE), route_a
    assert route_a["ceiling_rank_corr"] == pytest.approx(0.167, abs=5e-4), route_a


def test_committed_route_b_tau2_matches_the_c2_file(sampling_profile):
    """
    Gate 3a, first half: Route B's tau2 is the weight-weighted mean of C.2's
    `tau2_split_derived` over the E.5 frame. Establishes WHICH field Route B read, which
    B' must then read identically. The refit half of 3a runs in test_m_report.py, where
    the C.2 fit is executed.
    """
    c2 = e15.load_c2_components(C2_PATH)
    tau2 = np.array([c2[side]["tau2_split"] for side in sampling_profile["stand"]])
    pooled = float(np.average(tau2, weights=sampling_profile["weight"]))
    assert pooled == pytest.approx(COMMITTED_ROUTE_B_TAU2, abs=TOLERANCE)


# ------------------------------------------------------------------ §9.4 Route C

def test_split_half_estimator_recovers_positive_reliability_at_large_exposure(sampling_profile):
    """
    §9.4: on simulated data with a known tau2 and generous per-half exposure, Route C's
    estimator must come back positive and near the truth. If it could not do that, the
    real −0.366 would be an indictment of the estimator and the diagnostic would be moot.

    'Large PA' is simulated by shrinking the per-half sampling variance by 20x — the same
    hitters, given roughly twenty times their real exposure.

    The target is the STEPPED-UP reliability, not the half-length one: Spearman-Brown maps
    r_half = tau2/(tau2 + E[s_half]) onto the reliability of the two halves combined, whose
    sampling variance is E[s_half]/2. Comparing against r_half would be checking the
    estimator against the wrong quantity.
    """
    rng = np.random.default_rng(5)
    half_sampling = sampling_profile["sampling_var"].to_numpy() * 2.0 / 20.0
    tau2 = COMMITTED_ROUTE_B_TAU2
    simulated = m_ceiling.simulate_split_half(tau2, half_sampling, rng, n_draws=200)
    full_length = tau2 / (tau2 + float(np.mean(half_sampling)) / 2.0)
    assert simulated["p2_5"] > 0, simulated
    assert simulated["mean"] == pytest.approx(full_length, abs=0.10), (simulated, full_length)


def test_split_half_null_at_zero_tau2_straddles_zero(sampling_profile):
    """Under no true skill the estimator must centre on zero and reach both signs — the
    null a real −0.366 gets located against."""
    rng = np.random.default_rng(6)
    half_sampling = sampling_profile["sampling_var"].to_numpy() * 2.0
    simulated = m_ceiling.simulate_split_half(0.0, half_sampling, rng, n_draws=300)
    assert simulated["p2_5"] < 0 < simulated["p97_5"], simulated
    assert abs(simulated["mean"]) < 0.05, simulated
    located = m_ceiling.locate_in_simulation(-0.366, simulated)
    assert 0.0 <= located["percentile"] <= 100.0
