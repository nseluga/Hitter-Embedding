"""
Verification gates for the C.2 bivariate empirical-Bayes baseline.

The first C.2 attempt was reverted after two silent failures, and the gates below
exist specifically so neither can recur:

  - it pooled within-PA variance by pitcher hand when the real value is cell-specific
    (LHB vs LHP is .2374, not .261), and that single assumption drove the whole
    estimate -> G3 asserts the cells are distinct and that pooling changes the answer;
  - its gates tested 8,000-20,000 hitters when real groups are 120-927, so a
    regime-specific failure was undetectable -> G1 recovers parameters at the REAL
    sample sizes, and the large-N case is a separate, tighter check.

G9 is the gate that justifies the parameterization at all: the covariance estimator
must be insensitive to the noise level, because unlike the split variance it does no
noise subtraction.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis import c2_bivariate_eb as c2

BATTER_ID_BASE = 100_000
PITCHER_ID_BASE = 900_000


def synthetic_pa_table(n_hitters, mu, tau2, rho, sigma2, pa_range, seed=0,
                       seasons=(2021, 2022, 2023), stand="R"):
    """
    Build a PA-level table whose hitters have known true talent vs each hand.
    Draws (theta_L, theta_R) from N(mu, Sigma), then per-PA woba_points from
    N(theta, sqrt(sigma2)). Returns (pa_frame, true_theta) so an estimator can be
    scored against the parameters that generated the data.
    """
    rng = np.random.default_rng(seed)
    matrix = c2.sigma_matrix(np.asarray(tau2, dtype=float), rho)
    theta = rng.multivariate_normal(np.asarray(mu, dtype=float), matrix, size=n_hitters)

    records = []
    pa_id = 0
    for index in range(n_hitters):
        for side_index, hand in enumerate(c2.SIDES):
            low, high = pa_range[hand]
            n_pa = int(rng.integers(low, high + 1))
            points = rng.normal(theta[index, side_index], np.sqrt(sigma2[side_index]), n_pa)
            for value in points:
                pa_id += 1
                records.append({
                    "batter": BATTER_ID_BASE + index,
                    "pitcher": PITCHER_ID_BASE + (pa_id % 40),
                    "season": int(rng.choice(seasons)),
                    "p_throws": hand,
                    "stand": stand,
                    "woba_points": float(value),
                    "in_denominator": True,
                    "game_pk": pa_id,
                    "at_bat_number": 1,
                })
    return pd.DataFrame(records), theta


# ---------------------------------------------------------------------------
# G1 — parameter recovery at the REAL sample sizes, and (tighter) at large N
# ---------------------------------------------------------------------------

REAL_MU = (0.300, 0.320)
REAL_TAU2 = (0.0013, 0.0013)
REAL_SIGMA2 = (0.2374, 0.2610)
REAL_PA = {"L": (1, 300), "R": (3, 900)}


def test_g1_recovers_parameters_at_real_sample_sizes():
    """400 hitters with 1-300 / 3-900 PA — the regime the project actually has."""
    pa_df, _ = synthetic_pa_table(400, REAL_MU, REAL_TAU2, 0.85, REAL_SIGMA2, REAL_PA, seed=1)
    fitted = c2.fit(pa_df, eval_season=2024, rho_min_pa=1)["R"]

    assert fitted["n_hitters"] == 400
    assert np.allclose(fitted["mu"], REAL_MU, atol=0.01)
    assert np.allclose(fitted["sigma2"], REAL_SIGMA2, rtol=0.05)
    # tau^2 and rho are the hard ones at this N; the band is wide on purpose and is
    # the honest precision of the estimator in this regime, not a passing tolerance
    assert np.allclose(fitted["tau2"], REAL_TAU2, rtol=0.40)
    assert 0.65 <= fitted["rho"] <= 1.0


def test_g1_recovers_parameters_tightly_at_large_n():
    """Same generator, 6,000 hitters: the algebra must be right, not just unbiased-ish."""
    pa_df, _ = synthetic_pa_table(6000, REAL_MU, REAL_TAU2, 0.85, REAL_SIGMA2, REAL_PA, seed=2)
    fitted = c2.fit(pa_df, eval_season=2024, rho_min_pa=1)["R"]

    assert np.allclose(fitted["mu"], REAL_MU, atol=0.003)
    assert np.allclose(fitted["sigma2"], REAL_SIGMA2, rtol=0.01)
    assert np.allclose(fitted["tau2"], REAL_TAU2, rtol=0.12)
    assert abs(fitted["rho"] - 0.85) < 0.10


# ---------------------------------------------------------------------------
# G2 — floor guard: no signal must return zero, never an optimizer's bound
# ---------------------------------------------------------------------------

def test_g2_zero_signal_returns_zero_not_a_convergence_floor():
    """
    With tau^2 = 0 every hitter has identical talent, so the estimate must be 0 and
    the implied split constant must be inf. The reverted attempt used
    minimize_scalar(method="bounded"), which silently returned its own bound and
    printed a confident n* of ~46,878 in exactly this case.
    """
    pa_df, _ = synthetic_pa_table(300, REAL_MU, (1e-12, 1e-12), 0.0, REAL_SIGMA2, REAL_PA, seed=3)
    fitted = c2.fit(pa_df, eval_season=2024, rho_min_pa=1)["R"]

    assert fitted["tau2"][0] < 1e-6 and fitted["tau2"][1] < 1e-6
    assert fitted["tau2"][0] >= 0.0 and fitted["tau2"][1] >= 0.0
    n_star_split, _ = c2.implied_split_constant(fitted)
    assert not np.isfinite(n_star_split), "no signal must yield an infinite constant"


# ---------------------------------------------------------------------------
# G3 — within-PA variance is cell-specific, and pooling it changes the answer
# ---------------------------------------------------------------------------

def test_g3_sigma2_is_estimated_per_cell():
    """The two sides carry genuinely different within-PA variance; the fit must see it."""
    pa_df, _ = synthetic_pa_table(800, REAL_MU, REAL_TAU2, 0.85, (0.15, 0.35), REAL_PA, seed=4)
    fitted = c2.fit(pa_df, eval_season=2024, rho_min_pa=1)["R"]
    assert abs(fitted["sigma2"][0] - 0.15) < 0.01
    assert abs(fitted["sigma2"][1] - 0.35) < 0.02
    assert fitted["sigma2"][1] > 2.0 * fitted["sigma2"][0], "per-cell variance was pooled away"


def test_g3_pooling_sigma2_moves_predictions_materially():
    """
    Regression test on the bug that killed attempt 1: substituting one pooled variance
    for the two cell-specific ones must visibly move the shrunk estimates, so a future
    refactor that quietly pools cannot pass unnoticed.
    """
    x = np.array([[0.400, 0.360]])
    n_pa = np.array([[40.0, 400.0]])
    mu, tau2 = np.array(REAL_MU), np.array(REAL_TAU2)
    matrix = c2.sigma_matrix(tau2, 0.85)

    per_cell = c2.posterior_mean(x, n_pa, mu, matrix, np.array([0.2374, 0.2610]))
    pooled = c2.posterior_mean(x, n_pa, mu, matrix, np.array([0.2610, 0.2610]))
    assert abs(per_cell[0, 0] - pooled[0, 0]) > 1e-4


# ---------------------------------------------------------------------------
# G4 — nesting: rho = 0 must reduce exactly to independent univariate shrinkage
# ---------------------------------------------------------------------------

def test_g4_rho_zero_reduces_to_independent_shrinkage():
    """rho = 0 is the rate model; each side must equal the textbook n/(n+n*) formula."""
    x = np.array([[0.400, 0.360], [0.250, 0.500]])
    n_pa = np.array([[40.0, 400.0], [5.0, 120.0]])
    mu, tau2, sigma2 = np.array(REAL_MU), np.array(REAL_TAU2), np.array(REAL_SIGMA2)

    shrunk = c2.posterior_mean(x, n_pa, mu, c2.sigma_matrix(tau2, 0.0), sigma2)
    n_star = sigma2 / tau2
    expected = (n_pa * x + n_star[None, :] * mu[None, :]) / (n_pa + n_star[None, :])
    assert np.allclose(shrunk, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# G5 — shrinkage limits and monotonicity, including the zero-PA closed form
# ---------------------------------------------------------------------------

def test_g5_zero_pa_side_matches_the_analytic_limit():
    """
    A hitter with ZERO PA vs LHP must get mu_L + tau_LR/(tau^2_R + v_R) * (x_R - mu_R).
    This is the 117-group case in the low stratum and the main source of C.2's edge,
    so the numeric stand-in for an infinite variance is checked against the algebra.
    """
    mu, tau2, sigma2 = np.array(REAL_MU), np.array(REAL_TAU2), np.array(REAL_SIGMA2)
    rho = 0.85
    matrix = c2.sigma_matrix(tau2, rho)
    x_r, n_r = 0.420, 600.0

    shrunk = c2.posterior_mean(np.array([[np.nan, x_r]]), np.array([[0.0, n_r]]), mu, matrix, sigma2)

    v_r = sigma2[1] / n_r
    expected_l = mu[0] + matrix[0, 1] / (tau2[1] + v_r) * (x_r - mu[1])
    expected_r = mu[1] + tau2[1] / (tau2[1] + v_r) * (x_r - mu[1])
    assert abs(shrunk[0, 0] - expected_l) < 1e-10
    assert abs(shrunk[0, 1] - expected_r) < 1e-10
    # and it must be a real transfer of information, not a fallback to the league mean
    assert shrunk[0, 0] > mu[0] + 0.02


def test_g5_no_data_at_all_returns_the_prior_mean():
    mu, tau2, sigma2 = np.array(REAL_MU), np.array(REAL_TAU2), np.array(REAL_SIGMA2)
    shrunk = c2.posterior_mean(np.array([[np.nan, np.nan]]), np.array([[0.0, 0.0]]),
                               mu, c2.sigma_matrix(tau2, 0.85), sigma2)
    assert np.allclose(shrunk, mu, atol=1e-9)


def test_g5_shrinkage_is_monotone_in_exposure():
    """More PA on a side must move that side's estimate monotonically toward what he did."""
    mu, tau2, sigma2 = np.array(REAL_MU), np.array(REAL_TAU2), np.array(REAL_SIGMA2)
    matrix = c2.sigma_matrix(tau2, 0.85)
    exposures = np.array([1.0, 10.0, 50.0, 200.0, 1000.0, 100_000.0])
    x = np.column_stack([np.full(len(exposures), 0.450), np.full(len(exposures), 0.320)])
    n_pa = np.column_stack([exposures, np.full(len(exposures), 300.0)])

    shrunk = c2.posterior_mean(x, n_pa, mu, matrix, sigma2)[:, 0]
    assert np.all(np.diff(shrunk) > 0), "estimate must rise monotonically with weak-side PA"
    assert abs(shrunk[-1] - 0.450) < 1e-3, "infinite exposure must converge on the observation"


# ---------------------------------------------------------------------------
# G6 — Sigma must be positive semidefinite
# ---------------------------------------------------------------------------

def test_g6_impossible_correlation_is_rejected():
    with pytest.raises(AssertionError, match="positive semidefinite"):
        c2.sigma_matrix(np.array(REAL_TAU2), 1.5)


def test_g6_fitted_rho_is_clipped_into_range():
    """A moment estimate can exceed |1| by sampling error; the packed Sigma must not."""
    pa_df, _ = synthetic_pa_table(150, REAL_MU, REAL_TAU2, 0.99, REAL_SIGMA2, REAL_PA, seed=5)
    fitted = c2.fit(pa_df, eval_season=2024, rho_min_pa=1)["R"]
    assert -1.0 < fitted["rho"] < 1.0
    assert np.linalg.eigvalsh(fitted["sigma_matrix"]).min() >= -1e-15


def test_g6_rho_pinned_at_the_ceiling_is_flagged_and_derives_nothing():
    """
    The ceiling analog of G2's floor guard, and the bug the first real run produced:
    rho came back as exactly RHO_CLIP for all three batter types and the code happily
    printed a split constant of 55,083 off it. A clipped rho is not an estimate, so
    `rho_at_bound` must be set and `implied_split_constant` must refuse to answer.
    """
    impossible = {"L": 0.0013, "R": 0.0013}
    record = c2._assemble("R", {"L": 0.30, "R": 0.32}, impossible, {"L": 0.24, "R": 0.26},
                          covariance=0.0050, n_hitters=500, n_rho_hitters=300)

    assert record["rho_at_bound"], "an out-of-range moment estimate must be flagged"
    assert record["rho_unclipped"] > 1.0
    assert record["rho"] == c2.RHO_CLIP
    n_star_split, tau2_split = c2.implied_split_constant(record)
    assert np.isnan(n_star_split) and np.isnan(tau2_split), \
        "nothing may be derived from a bound; that is how attempt 1 printed a fake n*"

    # and a genuinely identified fit must NOT be flagged, or the guard is useless
    healthy = c2._assemble("R", {"L": 0.30, "R": 0.32}, impossible, {"L": 0.24, "R": 0.26},
                           covariance=0.0013 * 0.85, n_hitters=500, n_rho_hitters=300)
    assert not healthy["rho_at_bound"]
    assert np.isfinite(c2.implied_split_constant(healthy)[0])


# ---------------------------------------------------------------------------
# G7 — leakage boundary
# ---------------------------------------------------------------------------

def test_g7_fit_ignores_the_evaluated_season_entirely():
    """
    Poison the eval season with absurd values. If any parameter moves, the fitting
    window leaked, which would raise the bar Phase D must clear using data no
    forecaster could have had.
    """
    pa_df, _ = synthetic_pa_table(300, REAL_MU, REAL_TAU2, 0.85, REAL_SIGMA2, REAL_PA, seed=6)
    clean = c2.fit(pa_df, eval_season=2024)["R"]

    poison = pa_df[pa_df["season"] == 2023].copy()
    poison["season"] = 2024
    poison["woba_points"] = 9.99
    poisoned = c2.fit(pd.concat([pa_df, poison], ignore_index=True), eval_season=2024)["R"]

    assert np.allclose(clean["mu"], poisoned["mu"])
    assert np.allclose(clean["tau2"], poisoned["tau2"])
    assert clean["rho"] == poisoned["rho"]


def test_g7_window_excludes_seasons_before_the_trailing_span():
    """A 3-season window must not read a 4th season back."""
    pa_df, _ = synthetic_pa_table(200, REAL_MU, REAL_TAU2, 0.85, REAL_SIGMA2, REAL_PA,
                                  seed=7, seasons=(2021, 2022, 2023))
    old = pa_df[pa_df["season"] == 2021].copy()
    old["season"] = 2020
    old["woba_points"] = 9.99
    with_old = c2.fit(pd.concat([pa_df, old], ignore_index=True), eval_season=2024)["R"]
    assert np.allclose(c2.fit(pa_df, eval_season=2024)["R"]["mu"], with_old["mu"])


# ---------------------------------------------------------------------------
# G8 — determinism
# ---------------------------------------------------------------------------

def test_g8_fit_and_predict_are_deterministic():
    pa_df, _ = synthetic_pa_table(250, REAL_MU, REAL_TAU2, 0.85, REAL_SIGMA2, REAL_PA, seed=8)
    eval_rows = pa_df[pa_df["season"] == 2023].copy()
    eval_rows["season"] = 2024
    full = pd.concat([pa_df, eval_rows], ignore_index=True)

    first = c2.predict(full, 2024)
    second = c2.predict(full, 2024)
    pd.testing.assert_frame_equal(first, second)


# ---------------------------------------------------------------------------
# G9 — the covariance estimator does no noise subtraction, so the noise level
#      must not move it. This is why the bivariate parameterization is used.
# ---------------------------------------------------------------------------

def test_g9a_covariance_is_unbiased_at_every_noise_level():
    """
    Same true rho, sampling noise scaled over an 8x range, averaged over independent
    seeds. The estimate must stay centred on the truth at every scale because it
    subtracts no noise term.

    NOTE the honest limit this gate encodes: only the MEAN is stable. The estimator's
    variance still grows with the noise level (asserted below), so the bivariate
    parameterization buys freedom from bias and from catastrophic cancellation, NOT
    immunity to noise. Averaging over seeds is required; a single shared seed couples
    the realizations and shows spurious drift.

    Run with rho_min_pa=1 deliberately. RHO_MIN_PA exists to defend against a REAL-DATA
    pathology — heavy-tailed one-PA side means plus selection on exposure — that this
    synthetic does not contain, because here n is drawn independently of talent. The
    threshold's justification is the measured sensitivity table in the lab notebook;
    this gate's job is the estimator's algebra.
    """
    truth = 0.80 * np.sqrt(REAL_TAU2[0] * REAL_TAU2[1])
    spreads = []
    for scale in (0.5, 2.0, 8.0):
        sigma2 = (REAL_SIGMA2[0] * scale, REAL_SIGMA2[1] * scale)
        estimates = [
            c2.fit(synthetic_pa_table(600, REAL_MU, REAL_TAU2, 0.80, sigma2, REAL_PA,
                                      seed=200 + k)[0], eval_season=2024,
                   rho_min_pa=1)["R"]["covariance"]
            for k in range(8)
        ]
        assert abs(np.mean(estimates) - truth) < 0.25 * truth, \
            f"covariance biased at noise scale {scale}: {np.mean(estimates):.6f} vs {truth:.6f}"
        spreads.append(float(np.std(estimates)))

    assert spreads[-1] > spreads[0], \
        "estimator variance must grow with noise — a gate that hid this would overclaim"


def _naive_split_variance(pairs, sigma2):
    """
    The route the reverted C.2 attempt took: estimate Var(split) directly by
    subtracting an ASSUMED sampling-noise term. Test-only, kept here to demonstrate
    why production code does not do this.
    """
    usable = pairs[(pairs["n_L"] > 0) & (pairs["n_R"] > 0)]
    observed = float(np.var(usable["x_L"] - usable["x_R"], ddof=1))
    noise = float(np.mean(sigma2[0] / usable["n_L"] + sigma2[1] / usable["n_R"]))
    return observed - noise


def test_g9b_split_route_is_destroyed_by_a_small_sigma2_error_and_ours_is_not():
    """
    The exact failure that killed attempt 1. It plugged in one pooled within-PA
    variance (.261) where the real LHB-vs-LHP value is .2374 — a 9% error — and the
    split constant moved 16,419 -> 4,390.

    Direct Var(split) subtracts a noise term ~12x larger than the quantity it is
    trying to leave behind, so a 9% error in that term swamps the answer. The
    covariance route never takes sigma^2 as an input at all, so the same
    misspecification cannot reach it.
    """
    pa_df, _ = synthetic_pa_table(1500, REAL_MU, REAL_TAU2, 0.80, REAL_SIGMA2, REAL_PA, seed=11)
    window = pa_df[pa_df["in_denominator"]].copy()
    window = window.merge(c2.batter_types(window), on="batter", how="left")
    pairs = c2.to_pairs(c2.side_observations(window))

    correct = np.array(REAL_SIGMA2)
    misspecified = np.array([0.2610, 0.2610])  # the pooled value attempt 1 assumed
    assert abs(misspecified[0] / correct[0] - 1.0) < 0.11, "the injected error is ~9%, as it was"

    naive_correct = _naive_split_variance(pairs, correct)
    naive_wrong = _naive_split_variance(pairs, misspecified)
    truth_split = REAL_TAU2[0] + REAL_TAU2[1] - 2 * 0.80 * np.sqrt(REAL_TAU2[0] * REAL_TAU2[1])

    # a 9% sigma^2 error moves the direct estimate by more than the whole quantity
    assert abs(naive_wrong - naive_correct) > truth_split, \
        f"expected the direct route to break: {naive_correct:.6f} -> {naive_wrong:.6f}"

    # our route takes no sigma^2 argument, so the same error has nowhere to enter
    fitted = c2.fit(pa_df, eval_season=2024, rho_min_pa=1)["R"]
    _, derived_split = c2.implied_split_constant(fitted)
    assert abs(derived_split - truth_split) < 0.6 * truth_split, \
        f"derived split variance {derived_split:.6f} vs truth {truth_split:.6f}"


# ---------------------------------------------------------------------------
# G10 — batter typing, including switch hitters
# ---------------------------------------------------------------------------

def test_g10_switch_hitters_are_their_own_type():
    rows = []
    for batter, stands in [(1, ["R"] * 100), (2, ["L"] * 100),
                           (3, ["L"] * 70 + ["R"] * 30), (4, ["R"] * 199 + ["L"])]:
        for stand in stands:
            rows.append({"batter": batter, "stand": stand})
    typed = c2.batter_types(pd.DataFrame(rows)).set_index("batter")["batter_type"]

    assert typed[1] == "R"
    assert typed[2] == "L"
    assert typed[3] == "S", "a genuine switch hitter must be typed S"
    assert typed[4] == "R", "a single stray row must not create a switch hitter"


# ---------------------------------------------------------------------------
# G11 — shape assertions at the module boundary
# ---------------------------------------------------------------------------

def test_g11_rejects_wrong_shapes():
    mu, tau2, sigma2 = np.array(REAL_MU), np.array(REAL_TAU2), np.array(REAL_SIGMA2)
    matrix = c2.sigma_matrix(tau2, 0.5)
    with pytest.raises(AssertionError, match=r"\(N,2\)"):
        c2.posterior_mean(np.array([0.3, 0.3]), np.array([10.0, 10.0]), mu, matrix, sigma2)
    with pytest.raises(AssertionError, match=r"\(N,2\)"):
        c2.posterior_mean(np.array([[0.3, 0.3]]), np.array([[10.0, 10.0, 5.0]]), mu, matrix, sigma2)


def test_g11_predict_emits_the_claim1_contract():
    pa_df, _ = synthetic_pa_table(200, REAL_MU, REAL_TAU2, 0.85, REAL_SIGMA2, REAL_PA, seed=10)
    eval_rows = pa_df[pa_df["season"] == 2023].copy()
    eval_rows["season"] = 2024
    predictions = c2.predict(pd.concat([pa_df, eval_rows], ignore_index=True), 2024)

    assert list(predictions.columns) == c2.KEY + ["pred_woba"]
    assert (predictions["season"] == 2024).all()
    assert not predictions.duplicated(c2.KEY).any()
    assert np.isfinite(predictions["pred_woba"]).all()
