"""
Phase M — the measurement-ceiling apparatus (docs/phase-m-spec.md §M.0).

WHAT THIS FILE IS
-----------------
The pure statistics and the simulations behind Phase M. No file reads, no artifacts, no
argparse — everything here is a function of numbers, so §9's planted-recovery self-checks
can drive it directly. The driver that reads Phase C/E/F artifacts and writes
`results/phase_m/` is `m_report.py`.

E.15 (`e_platoon_ceiling.py`) already owns the primitives — `reliability_from_variances`,
`noise_corrected_variance`, `spearman_brown`, `weighted_variance`. They are imported, not
reimplemented, so the two phases cannot drift apart.

THE MAP THIS FILE VALIDATES
---------------------------
For hitter h with realized side denominators n_L, n_R:

    delta_obs(h) = delta_true(h) + e(h),   Var[e(h)] = s2_L/n_L + s2_R/n_R
    reliability  = tau2 / (tau2 + E[Var[e]])
    ceiling on |corr(anything, delta_obs)| = sqrt(reliability)

The ceiling step is exact for a weighted PEARSON correlation: the best available predictor
is delta_true itself, and corr(delta_true, delta_obs) = tau2 / sqrt(tau2·(tau2 + E[Var[e]]))
= sqrt(reliability). E.5 reports a weighted SPEARMAN correlation, and the transfer is NOT
free.

  MEASURED, not assumed (§9.1, seed 7, 300 draws, tau2 = 0.00059034). On a homoscedastic
  profile the rank ceiling is 0.3427 against an analytic 0.3556 — the textbook few-percent
  conservatism. On the REAL 2024 exposure profile, whose per-hitter sampling variance spans
  a factor of 36, it is 0.3693, i.e. 3.9% ABOVE the analytic value. Rank correlation is
  robust to the heavy-tailed observations the low-exposure hitters contribute, and that
  robustness buys back more than joint normality costs.

  So sqrt(reliability) is not a conservative bound on the rank correlation at these
  exposures; it is a close approximation that runs ~4% low, which makes every
  fraction-of-ceiling figure ~4% relatively HIGH. `monte_carlo_ceiling` returns the
  measured rank ceiling beside the analytic one so the artifacts can carry both, and the
  §9 test asserts the size of the gap rather than a direction that does not hold.

NEGATIVE ESTIMATES ARE NEVER CLIPPED (spec §9.2). A negative tau2 is a finding about the
noise model, not an error to be floored at zero; it comes back as a negative number with
`degenerate` set, and every caller propagates the flag.

WHAT IS NOT HERE
----------------
No estimator is invented. Route A is E.15's subtraction, Route B/B' is C.2's fit, Route C
is E.15's split-half. Phase M adds populations, strata and simulations around them.
"""

import numpy as np
import pandas as pd

from src.analysis import claim1_eval
from src.analysis.e_platoon_ceiling import (
    noise_corrected_variance,
    reliability_from_variances,
    spearman_brown,
    weighted_variance,
)

# The fragility band the route rule pre-registered: Route A is a near-total cancellation
# (0.004196 − 0.004078), so a 3% error in the noise model swings tau2 by ~100%. These
# scales demonstrate that; they never pick the route.
FRAGILITY_SCALES = (0.97, 1.03)


# ------------------------------------------------------------------ the map

def ceiling_from_variances(tau2, mean_sampling_var):
    """
    tau2 and a mean sampling variance in, reliability and rank-correlation ceiling out.

    The one place the reliability -> ceiling step is taken. Degenerate input (tau2 <= 0)
    returns the negative tau2 unchanged, a nan ceiling, and `degenerate` True — never a
    clip to zero, per spec §9.2.
    """
    tau2 = float(tau2)
    mean_sampling_var = float(mean_sampling_var)
    assert mean_sampling_var > 0, "mean sampling variance must be positive"
    if tau2 <= 0:
        return {"tau2": tau2, "mean_sampling_var": mean_sampling_var,
                "reliability": float("nan"), "ceiling_rank_corr": float("nan"),
                "degenerate": True}
    reliability = reliability_from_variances(tau2, mean_sampling_var)
    return {"tau2": tau2, "mean_sampling_var": mean_sampling_var,
            "reliability": reliability,
            "ceiling_rank_corr": float(np.sqrt(reliability)),
            "degenerate": False}


def route_a_tau2(observed_within_stand_var, mean_sampling_var, scale=1.0):
    """
    Route A: the same-season empirical subtraction, observed − sampling.

    `scale` multiplies the sampling term and exists only for the fragility band; the
    committed Route A value is the scale=1.0 call.
    """
    scaled = float(mean_sampling_var) * float(scale)
    tau2 = noise_corrected_variance(float(observed_within_stand_var), scaled)
    out = ceiling_from_variances(tau2, scaled) if tau2 > 0 else {
        "tau2": tau2, "mean_sampling_var": scaled, "reliability": float("nan"),
        "ceiling_rank_corr": float("nan"), "degenerate": True}
    out["sampling_scale"] = float(scale)
    out["observed_within_stand_var"] = float(observed_within_stand_var)
    return out


def fragility_band(observed_within_stand_var, mean_sampling_var, scales=FRAGILITY_SCALES):
    """Route A recomputed with the sampling term scaled, per the pre-registered rule §M.0.2."""
    rows = [route_a_tau2(observed_within_stand_var, mean_sampling_var, scale)
            for scale in (1.0,) + tuple(scales)]
    ceilings = [row["ceiling_rank_corr"] for row in rows]
    finite = [value for value in ceilings if np.isfinite(value)]
    return {
        "scales": [row["sampling_scale"] for row in rows],
        "tau2": [row["tau2"] for row in rows],
        "ceiling_rank_corr": ceilings,
        "ceiling_range_finite_only": [min(finite), max(finite)] if finite else [float("nan")] * 2,
        "any_degenerate": any(row["degenerate"] for row in rows),
    }


def stabilization_pa(per_pa_noise_var, tau2):
    """
    PA* = per-PA noise variance / tau2 — the exposure at which sampling variance falls to
    the true-talent variance, i.e. reliability 0.5. tau2 re-expressed in the unit baseball
    readers know; not a new estimator.

    Returns inf for tau2 <= 0 (a differential that never stabilizes), never a clip.
    """
    tau2 = float(tau2)
    assert per_pa_noise_var > 0, "per-PA noise variance must be positive"
    if tau2 <= 0:
        return float("inf")
    return float(per_pa_noise_var) / tau2


# ------------------------------------------------------------------ simulation

def simulate_differentials(tau2, sampling_var, rng):
    """
    Plant a known tau2 on a real exposure profile.

    `sampling_var` is the per-hitter Var[e] from the real frame, so the simulated hitters
    carry the actual 2024 exposure skew rather than a tidy homoscedastic one — the point
    of the check is that the map survives that skew.
    """
    sampling_var = np.asarray(sampling_var, dtype="float64")
    assert (sampling_var > 0).all(), "a simulated hitter has non-positive sampling variance"
    assert tau2 >= 0, "cannot plant a negative tau2"
    true = rng.normal(0.0, np.sqrt(tau2), size=len(sampling_var))
    observed = true + rng.normal(0.0, np.sqrt(sampling_var))
    return true, observed


def monte_carlo_ceiling(tau2, sampling_var, weight, n_draws=200, seed=0):
    """
    The load-bearing §9 check, as a function: does sqrt(reliability) actually bound the
    achievable correlation?

    Simulates hitters at a known tau2, correlates the BEST POSSIBLE predictor (true skill
    itself) against the simulated observation, and returns the mean weighted Pearson and
    weighted Spearman over draws beside the analytic ceiling. Pearson matches the analytic
    ceiling exactly; Spearman does not, and `mc_spearman_mean` is the measured rank ceiling
    callers should report beside it (see the module docstring for why the sign of that gap
    depends on the exposure skew).
    """
    rng = np.random.default_rng(seed)
    weight = np.asarray(weight, dtype="float64")
    sampling_var = np.asarray(sampling_var, dtype="float64")
    analytic = ceiling_from_variances(
        tau2, float(np.average(sampling_var, weights=weight)))
    pearson, spearman = [], []
    for _ in range(n_draws):
        true, observed = simulate_differentials(tau2, sampling_var, rng)
        pearson.append(claim1_eval.weighted_pearson(true, observed, weight))
        spearman.append(claim1_eval.weighted_rank_correlation(observed, true, weight))
    return {
        **analytic,
        "mc_pearson_mean": float(np.mean(pearson)),
        "mc_pearson_sd": float(np.std(pearson, ddof=1)),
        "mc_spearman_mean": float(np.mean(spearman)),
        "mc_spearman_sd": float(np.std(spearman, ddof=1)),
        "n_draws": int(n_draws),
    }


def recover_route_a(tau2, sampling_var, weight, n_draws=200, seed=0):
    """
    Does Route A's subtraction get the planted tau2 back?

    Reproduces the estimator exactly as E.15 runs it: weighted variance of the simulated
    observations minus the weighted mean sampling variance. The spread over draws is
    returned because at these exposures the estimator's own sampling error is the story —
    a recovery that is unbiased but has an sd larger than the estimate is the fragility
    the route rule pre-registered.
    """
    rng = np.random.default_rng(seed)
    weight = np.asarray(weight, dtype="float64")
    sampling_var = np.asarray(sampling_var, dtype="float64")
    mean_sampling = float(np.average(sampling_var, weights=weight))
    estimates = []
    for _ in range(n_draws):
        _, observed = simulate_differentials(tau2, sampling_var, rng)
        estimates.append(noise_corrected_variance(
            weighted_variance(observed, weight), mean_sampling))
    estimates = np.asarray(estimates, dtype="float64")
    return {
        "planted_tau2": float(tau2),
        "recovered_mean": float(estimates.mean()),
        "recovered_sd": float(estimates.std(ddof=1)),
        "recovered_p2_5": float(np.percentile(estimates, 2.5)),
        "recovered_p97_5": float(np.percentile(estimates, 97.5)),
        "share_negative": float((estimates < 0).mean()),
        "n_draws": int(n_draws),
    }


# ------------------------------------------------------------------ Route C

def split_half_from_halves(delta_a, delta_b):
    """
    Route C's estimator, isolated from its data plumbing: Spearman across hitters between
    the two halves' differentials, stepped up to full length by Spearman-Brown.

    Kept separate from `e_platoon_ceiling.split_half_reliability` (which builds the halves
    from PA rows) precisely so the simulation can feed it synthetic halves and check the
    ESTIMATOR rather than the pipeline.
    """
    a, b = pd.Series(np.asarray(delta_a, "float64")), pd.Series(np.asarray(delta_b, "float64"))
    half = float(a.corr(b, method="spearman"))
    return {"half_split_spearman": half,
            "reliability_spearman_brown": float(spearman_brown(half))}


def simulate_split_half(tau2, half_sampling_var, rng, n_draws=1000):
    """
    The null distribution of Route C's estimate at a planted tau2, on real half-season
    exposures.

    `half_sampling_var` is Var[e] for ONE half — the two halves are disjoint games, so
    each carries roughly twice the full-season variance and their errors are independent.
    That independence is the whole reason split-half is a valid reliability estimator, and
    simulating it this way is what locates a value like −0.366 in a distribution.
    """
    half_sampling_var = np.asarray(half_sampling_var, dtype="float64")
    assert (half_sampling_var > 0).all(), "a simulated half has non-positive sampling variance"
    draws = []
    for _ in range(n_draws):
        true = rng.normal(0.0, np.sqrt(tau2), size=len(half_sampling_var)) if tau2 > 0 \
            else np.zeros(len(half_sampling_var))
        a = true + rng.normal(0.0, np.sqrt(half_sampling_var))
        b = true + rng.normal(0.0, np.sqrt(half_sampling_var))
        draws.append(split_half_from_halves(a, b)["reliability_spearman_brown"])
    draws = np.asarray(draws, dtype="float64")
    finite = draws[np.isfinite(draws)]
    assert len(finite) > 0.9 * n_draws, "the split-half simulation degenerated"
    return {
        "planted_tau2": float(tau2),
        "n_draws": int(n_draws),
        "n_hitters": int(len(half_sampling_var)),
        "mean": float(finite.mean()),
        "p2_5": float(np.percentile(finite, 2.5)),
        "p97_5": float(np.percentile(finite, 97.5)),
        "draws": finite,
    }


def locate_in_simulation(value, simulation):
    """Where an observed Route C estimate falls in a simulated null: percentile and inside/outside."""
    draws = simulation["draws"]
    return {
        "value": float(value),
        "percentile": float((draws < value).mean() * 100.0),
        "inside_95": bool(simulation["p2_5"] <= value <= simulation["p97_5"]),
    }


# ------------------------------------------------------------------ paired bootstrap
#
# Added Pass A1. The review of 2026-08-31 found that no achieved correlation in Phase M
# carried an interval, so a 49.3% fraction of ceiling was quoted as a point estimate while
# its own sampling error was wider than the A-to-B' bracket it sat inside.
#
# The unit resampled is the HITTER. On the M.1 differential frame a hitter contributes
# exactly one row -- both of his sides are already collapsed into delta_obs -- so row
# resampling IS cluster resampling here, unlike `claim1_eval.batter_clusters`, which
# exists because a claim-1 eval frame carries two rows per hitter. `paired_rank_bootstrap`
# asserts the one-row-per-hitter property rather than assuming it.
#
# Every model is scored on the SAME resampled hitters, so the shared target noise cancels
# in a difference and the paired interval is far tighter than the difference of two
# marginal intervals would suggest.


def within_stand_residual(values, weight, stand):
    """Subtract the weighted mean inside each stand — E.5's residualisation, in numpy."""
    values = np.asarray(values, dtype="float64")
    weight = np.asarray(weight, dtype="float64")
    stand = np.asarray(stand)
    out = values.copy()
    for side in np.unique(stand):
        mask = stand == side
        out[mask] = values[mask] - np.average(values[mask], weights=weight[mask])
    return out


def within_stand_rank_correlation(delta_obs, delta_pred, weight, stand):
    """
    E.5's within-stand weighted Spearman, as a pure array function.

    `e_eval.platoon_decomposition` is the reference implementation and stays the one the
    committed artifacts are produced by. This is the same arithmetic without the pandas
    frame rebuild, because the bootstrap calls it thousands of times. `tests/test_m_ceiling`
    asserts the two agree to floating point on the real frame — a translation check, not a
    second estimator.
    """
    weight = np.asarray(weight, dtype="float64")
    return float(claim1_eval.weighted_rank_correlation(
        within_stand_residual(delta_obs, weight, stand),
        within_stand_residual(delta_pred, weight, stand),
        weight))


def paired_rank_bootstrap(frame, columns, n_boot=2000, seed=0, ci=(2.5, 97.5)):
    """
    Resample hitters once per replicate and score EVERY model on that resample.

    frame: one row per hitter, with `delta_obs`, `weight`, `stand`, and each column in
    `columns`. Returns (draws, table): `draws` is (n_boot, len(columns)) so the caller can
    form any paired contrast from the same replicates, and `table` carries the point
    estimate and percentile interval per column.

    A replicate drawing a single stand is kept: the residualisation degenerates to centring
    and the statistic is still defined. A replicate yielding fewer than three distinct
    predictions gives NaN and is dropped, with `n_draws` reporting the survivors so a
    mostly-degenerate column cannot show a spuriously tight interval.
    """
    assert frame["batter"].is_unique, \
        "paired_rank_bootstrap resamples ROWS as hitters; this frame has repeated batters"
    obs = frame["delta_obs"].to_numpy(dtype="float64")
    weight = frame["weight"].to_numpy(dtype="float64")
    stand = frame["stand"].to_numpy()
    predictions = {name: frame[name].to_numpy(dtype="float64") for name in columns}
    n = len(frame)
    rng = np.random.default_rng(seed)

    draws = np.empty((int(n_boot), len(columns)), dtype="float64")
    for b in range(int(n_boot)):
        pick = rng.integers(0, n, n)
        for j, name in enumerate(columns):
            draws[b, j] = within_stand_rank_correlation(
                obs[pick], predictions[name][pick], weight[pick], stand[pick])

    rows = []
    for j, name in enumerate(columns):
        column = draws[:, j]
        finite = column[np.isfinite(column)]
        assert len(finite) > n_boot // 2, f"{name}: most bootstrap replicates degenerated"
        rows.append({
            "column": name,
            "n_hitters": int(n),
            "n_draws": int(len(finite)),
            "point": within_stand_rank_correlation(obs, predictions[name], weight, stand),
            "ci_low": float(np.percentile(finite, ci[0])),
            "ci_high": float(np.percentile(finite, ci[1])),
        })
    return draws, pd.DataFrame(rows)


def paired_contrast(draws, columns, a, b, point_a, point_b, ci=(2.5, 97.5)):
    """
    One paired difference from an existing `paired_rank_bootstrap` draw matrix.

    Sign convention follows `claim1_eval.paired_rank_difference`: positive favours A,
    because a higher rank correlation is better. `favours_a_share` is the share of
    replicates in which A ranks better — the same meaning it carries there.
    """
    index = {name: j for j, name in enumerate(columns)}
    difference = draws[:, index[a]] - draws[:, index[b]]
    difference = difference[np.isfinite(difference)]
    return {
        "contrast": f"{a} minus {b}",
        "difference": float(point_a - point_b),
        "ci_low": float(np.percentile(difference, ci[0])),
        "ci_high": float(np.percentile(difference, ci[1])),
        "favours_a_share": float(np.mean(difference > 0)),
        "n_draws": int(len(difference)),
    }
