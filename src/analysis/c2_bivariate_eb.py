"""
Phase C.2 — bivariate empirical-Bayes platoon regression (architecture §3 Phase C.2).

The incumbent. Frozen rule 1 says no deep-learning claim survives without beating
empirical-Bayes platoon regression out-of-sample, especially in low exposure. C.2's
job is to be the STRONGEST model that uses only (a) this hitter's own outcome data
and (b) league-level pooling — everything except the one mechanism Phase D claims.
Whatever Phase D wins over C.2 is attributable to cross-hitter process structure and
nothing else, which is what makes the thesis falsifiable.

  raw -> bucketed      value of shrinkage AT ALL          (C.1)
  bucketed -> C.2      value of doing shrinkage PROPERLY  (here)
  C.2 -> Phase D       value of cross-hitter structure    (the thesis)

WHAT IS SHRUNK, AND WHY THIS PARAMETERIZATION
---------------------------------------------
The literature (The Book, p. 157 via Tango) shrinks the SPLIT: regress a hitter's
observed (wOBA vs LHP - wOBA vs RHP) toward the league split, weighting his own
number by PA faced vs LHP and the league by 2200 (RHB) / 1000 (LHB).

We shrink the two sides JOINTLY instead. These are the same model in rotated
coordinates -- (theta_L, theta_R) vs (mean, difference) -- so nothing is given up.
What changes is which parameters are fixed by hand and how they are estimated:

  1. The Book FIXES the correlation between a hitter's two sides. Its constant is
     equivalent to an assumption about rho: with n*_d = n*_side / (2(1-rho)), the
     2200 constant asserts rho ~= 0.95 and 1000 asserts rho ~= 0.89 (derivation in
     the lab notebook; approximate, assumes tau_L = tau_R). We estimate rho instead.

  2. Identification. Estimating Var(split) requires subtracting a sampling-noise
     term that DOMINATES it: for a hitter with 30 PA vs LHP the noise variance is
     ~.237/30 = .0079 against a plausible signal of ~.015^2 = .00023, a 35x
     subtraction. A 9% error in the noise constant moved our first attempt's
     estimate from 16,419 to 4,390. The covariance route has no such term: a
     hitter's PA vs LHP and vs RHP are DISJOINT events, so their sampling noises
     are independent and

         Cov(x_L, x_R) = Cov(theta_L, theta_R)   exactly, no subtraction.

     The split variance is then DERIVED as tau^2_L + tau^2_R - 2*tau_LR, assembled
     from three separately well-identified estimates instead of one catastrophic
     difference. That is the whole reason this parameterization works where the
     first C.2 attempt did not.

     What this does and does NOT buy (gated in test_g9a / test_g9b): it removes the
     bias amplification and the dependence on a plugged-in sigma^2 constant -- a 9%
     error in that constant moved attempt 1's answer from 16,419 to 4,390, and there
     is no such input here. It does NOT make the estimate immune to noise. The
     covariance estimator's VARIANCE still grows with the noise level, so the interval
     on rho is wide and must be reported, never suppressed.

  3. Zero weak-side exposure. Where a hitter has PA against one hand only, the limit
     is theta_L = mu_L + tau_LR/(tau^2_R + v_R) * (x_R - mu_R): his PA vs RHP inform
     his vs-LHP projection in proportion to the measured rho, where C.1 can only
     return the league average.

     MEASURED SCOPE, corrected after the first real run: this affects far fewer rows
     than the design assumed. Of 1,005 hitters in the 2024 window, 955 have PA against
     BOTH hands and only 50 have one side at zero. Of the 117 zero-prior-PA groups in
     the low stratum, 115 have no prior MLB PA vs EITHER hand -- they are debuts, not
     platooned veterans, so there is nothing to borrow from and rho cannot move them
     (verified: their predictions are identical at rho=0 and rho=0.85). C.2's measured
     advantage over C.1 therefore comes from replacing C.1's step-function shrinkage
     with the continuous n/(n+n*) weight, NOT from cross-side borrowing. The borrowing
     is correct and cheap but it is not where the win came from; claiming otherwise
     would misattribute the result.

Grouping is by batter TYPE (L / R / switch), not by a pooled constant. Switch
hitters get their own mu, Sigma and sigma^2 because they take the platoon-advantage
cell on BOTH sides and their split structure is genuinely different. Within-PA
variance is per (type, hand) -- the first attempt pooled it by pitcher hand only,
used .261 where the real LHB-vs-LHP value is .2374, and that single assumption drove
the entire wrong answer.

Distributional assumption: none beyond two moments. The posterior-mean formula is
read as the BLUP (best linear unbiased predictor), which needs only the first and
second moments, not normality. wOBA over 5 PA is nowhere near normal, so this
reading is load-bearing, not a formality.

Leakage discipline: every parameter and every input is drawn from seasons strictly
before the evaluated season. The one exception is batter HANDEDNESS for hitters with
no window PA at all, which is read from the eval season -- a static roster attribute
known at projection time, never an outcome. `predict` asserts and reports this.
"""

import numpy as np
import pandas as pd

from src.analysis.c1_trailing import TRAILING_SEASONS, trailing_window
from src.analysis.stabilization import hitter_stats, variance_components
from src.data.eval_targets import drop_pitcher_batters

KEY = ["batter", "season", "p_throws"]
SIDES = ("L", "R")
BATTER_TYPES = ("L", "R", "S")

# A batter counts as a switch hitter when his minority stand is at least this share of
# his window PA. Genuine switch hitters sit near 25-30% (LHP are ~25% of PA); a
# mislabeled row or two sits far below 1%, so the threshold separates cleanly and does
# not misclassify a low-exposure switch hitter the way a raw PA floor would.
SWITCH_MIN_MINORITY_SHARE = 0.10

# A hitter needs PA against BOTH hands to contribute a pair. Kept at 1 for mu, tau^2
# and sigma^2: the ANOVA handles unequal group sizes correctly, and B.1 established
# that restricting to durable regulars halves the between-hitter signal variance
# (survivorship), so the full population is the right one for those.
MIN_PAIR_PA = 1

# The COVARIANCE is restricted further, and only the covariance. A hitter with 1 PA
# vs a hand has a side mean that is a single wOBA event, variance ~.24 against a
# tau^2 of ~.001 -- 240x. Sample covariance is not robust to those points, and they
# also carry the strongest selection (a hitter with 1 weak-side PA and 400 strong-side
# PA got the 400 by performing). Measured: rho comes out at 4.74 with no threshold,
# 1.09 at 25, and settles in .85-1.10 from 50 up. Restricting a CORRELATION to the
# durable subpopulation is a far milder assumption than restricting a VARIANCE, which
# is why the diagonal keeps the full population and only this is cut.
RHO_MIN_PA = 50

# Stand-in for an infinite sampling variance when a hitter has zero PA vs a hand.
# Large enough that the numeric answer matches the analytic limit to ~1e-13 (gated),
# small enough to keep the 2x2 solve well conditioned in float64.
ZERO_PA_VARIANCE = 1e9

# rho is clipped just inside the unit interval: |rho| = 1 makes Sigma singular, and a
# moment estimate can land outside [-1, 1] by sampling error alone.
RHO_CLIP = 0.999

# The Book's constants (p. 157, via Tango on insidethebook.com, 2009), converted to
# the rho they are equivalent to. Reported as a REFERENCE ROW only -- never fitted,
# never used to set our estimate. See the module docstring for the conversion.
BOOK_SPLIT_CONSTANT = {"R": 2200.0, "L": 1000.0}


def batter_types(window):
    """
    Classify each batter as L, R, or S (switch) from the stands he actually used.
    window: PA-level rows. Returns a frame of batter/batter_type.
    """
    counts = window.groupby(["batter", "stand"]).size().unstack(fill_value=0)
    for stand in SIDES:
        if stand not in counts.columns:
            counts[stand] = 0
    total = counts["L"] + counts["R"]
    minority_share = counts[["L", "R"]].min(axis=1) / total
    majority = counts[["L", "R"]].idxmax(axis=1)
    batter_type = np.where(minority_share >= SWITCH_MIN_MINORITY_SHARE, "S", majority)
    return pd.DataFrame({"batter": counts.index, "batter_type": batter_type})


def side_observations(window):
    """
    One row per (batter, pitcher hand): his window wOBA and PA vs that hand.
    window: denominator-only PA rows carrying batter_type.
    """
    grouped = window.groupby(["batter", "batter_type", "p_throws"], observed=True)
    out = grouped.agg(x=("woba_points", "mean"), n=("woba_points", "size")).reset_index()
    return out


def to_pairs(side_obs):
    """
    Pivot side observations to one row per batter with both sides side by side.
    Returns batter/batter_type/x_L/n_L/x_R/n_R; sides with no PA get x=nan, n=0.
    """
    wide = side_obs.pivot_table(index=["batter", "batter_type"], columns="p_throws",
                                values=["x", "n"], observed=True)
    wide.columns = [f"{value}_{hand}" for value, hand in wide.columns]
    wide = wide.reset_index()
    for hand in SIDES:
        for value, fill in (("x", np.nan), ("n", 0.0)):
            column = f"{value}_{hand}"
            if column not in wide.columns:
                wide[column] = fill
            wide[column] = wide[column].fillna(fill)
    return wide[["batter", "batter_type", "x_L", "n_L", "x_R", "n_R"]]


def cross_side_covariance(pairs):
    """
    Between-hitter covariance of talent vs LHP and vs RHP, from the sample covariance
    of the two observed wOBAs. Unbiased with NO noise subtraction: a hitter's PA vs
    LHP and vs RHP are disjoint, so their sampling errors are independent and the
    cross term vanishes in expectation. Unweighted, because any precision weight
    based on PA correlates with talent (Brown 2008, remark 3b) and would bias it.
    Returns nan if under 2 usable pairs.
    """
    usable = pairs[np.isfinite(pairs["x_L"]) & np.isfinite(pairs["x_R"])]
    if len(usable) < 2:
        return np.nan
    return float(np.cov(usable["x_L"].to_numpy(), usable["x_R"].to_numpy(), ddof=1)[0, 1])


def sufficient_stats(window, min_pair_pa=MIN_PAIR_PA):
    """
    One row per hitter carrying everything both estimators need for both sides:
    n / mean / within-hitter sum of squares. Restricted to hitters with PA against
    BOTH hands, so mu, tau^2 and the covariance are all fitted on the same population
    and Sigma is internally consistent. window: denominator-only rows with batter_type.
    """
    frames = []
    for hand in SIDES:
        side = window[window["p_throws"] == hand]
        stats = hitter_stats(side.assign(_key=side["batter"]), "_key", "woba_points")
        stats["batter"] = sorted(side["batter"].unique())
        frames.append(stats.set_index("batter").add_suffix(f"_{hand}"))
    wide = frames[0].join(frames[1], how="inner").reset_index()
    types = window[["batter", "batter_type"]].drop_duplicates()
    wide = wide.merge(types, on="batter", how="left")
    return wide[(wide["n_L"] >= min_pair_pa) & (wide["n_R"] >= min_pair_pa)].reset_index(drop=True)


def fit_from_stats(stats, rho_min_pa=RHO_MIN_PA):
    """
    Estimate (mu, Sigma, sigma^2) per batter type from the sufficient-statistic table.
    Separated from `fit` so the bootstrap can resample hitters without re-reading PA.
    The diagonal uses every hitter; the covariance uses only those with rho_min_pa PA
    on both sides (see RHO_MIN_PA).
    """
    params = {}
    for batter_type in BATTER_TYPES:
        type_stats = stats[stats["batter_type"] == batter_type]
        if len(type_stats) < 2:
            continue
        mu, tau2, sigma2 = {}, {}, {}
        for hand in SIDES:
            columns = {f"n_{hand}": "n", f"mean_{hand}": "mean", f"ss_{hand}": "ss"}
            side = type_stats[list(columns)].rename(columns=columns)
            # PA-weighted grand mean is the cell's league wOBA
            mu[hand] = float((side["n"] * side["mean"]).sum() / side["n"].sum())
            signal_var, noise_var = variance_components(side)
            tau2[hand] = float(signal_var)
            sigma2[hand] = float(noise_var)

        resolved = type_stats[(type_stats["n_L"] >= rho_min_pa) & (type_stats["n_R"] >= rho_min_pa)]
        covariance = cross_side_covariance(
            resolved.rename(columns={"mean_L": "x_L", "mean_R": "x_R"}))
        params[batter_type] = _assemble(batter_type, mu, tau2, sigma2, covariance,
                                        len(type_stats), len(resolved))
    assert params, "no batter type had enough paired hitters to fit"
    return params


def _fitting_window(pa_df, eval_season, n_seasons):
    """The denominator-only, hitters-only trailing window with batter types attached."""
    # pitchers taking their own turn at bat contaminate every prior (decision log
    # 2026-07-27); the league mean a projection shrinks toward is exactly such a prior
    window = trailing_window(drop_pitcher_batters(pa_df), eval_season, n_seasons)
    window = window[window["in_denominator"]]
    assert window["season"].max() < eval_season, "fitting window leaked the evaluated season"
    return window.merge(batter_types(window), on="batter", how="left")


def fit(pa_df, eval_season, n_seasons=TRAILING_SEASONS, min_pair_pa=MIN_PAIR_PA,
        rho_min_pa=RHO_MIN_PA):
    """
    Estimate the prior (mu, Sigma) and the sampling variance sigma^2 per batter type.
    pa_df: the PA-level eval-target table; eval_season: the season to be projected.
    Returns a dict keyed by batter type.
    """
    window = _fitting_window(pa_df, eval_season, n_seasons)
    return fit_from_stats(sufficient_stats(window, min_pair_pa), rho_min_pa)


def bootstrap_rho(pa_df, eval_season, n_seasons=TRAILING_SEASONS, min_pair_pa=MIN_PAIR_PA,
                  rho_min_pa=RHO_MIN_PA, n_boot=400, seed=0, ci=(2.5, 97.5)):
    """
    Percentile CI for rho per batter type, resampling HITTERS (the independent unit).
    The point estimate alone cannot answer whether rho differs from the value The
    Book's constant implies, which is the question this baseline exists to settle.
    Returns {batter_type: (point, lo, hi, share_at_bound)}; share_at_bound is the
    fraction of resamples that hit the clip, and a high value means rho is not
    identified from above rather than that it is close to 1.
    """
    stats = sufficient_stats(_fitting_window(pa_df, eval_season, n_seasons), min_pair_pa)
    point = {t: p["rho"] for t, p in fit_from_stats(stats, rho_min_pa).items()}
    rng = np.random.default_rng(seed)

    draws = {t: [] for t in point}
    bounded = {t: [] for t in point}
    for _ in range(n_boot):
        sample = stats.iloc[rng.integers(0, len(stats), len(stats))]
        for batter_type, record in fit_from_stats(sample, rho_min_pa).items():
            if batter_type in draws:
                draws[batter_type].append(record["rho"])
                bounded[batter_type].append(record["rho_at_bound"])
    return {t: (point[t], float(np.percentile(d, ci[0])), float(np.percentile(d, ci[1])),
                float(np.mean(bounded[t])))
            for t, d in draws.items() if d}


def _assemble(batter_type, mu, tau2, sigma2, covariance, n_hitters, n_rho_hitters=0):
    """
    Pack the per-type estimates into a parameter record, clipping rho into range.

    `rho_at_bound` is the ceiling analog of the floor guard the reverted C.2 attempt
    lacked: a moment estimate that lands outside [-1, 1] has been silenced by the clip
    and is NOT an estimate. Anything derived from it must refuse to produce a number
    rather than print the bound as though it were a result.
    """
    tau_product = np.sqrt(tau2["L"] * tau2["R"])
    rho_raw = float(covariance / tau_product) if tau_product > 0 else np.nan
    rho = float(np.clip(rho_raw, -RHO_CLIP, RHO_CLIP)) if np.isfinite(rho_raw) else 0.0
    return {
        "batter_type": batter_type,
        "n_hitters": int(n_hitters),
        "n_rho_hitters": int(n_rho_hitters),
        "mu": np.array([mu["L"], mu["R"]], dtype=float),
        "tau2": np.array([tau2["L"], tau2["R"]], dtype=float),
        "sigma2": np.array([sigma2["L"], sigma2["R"]], dtype=float),
        "covariance": float(covariance),
        "rho": rho,
        "rho_unclipped": rho_raw,
        "rho_at_bound": bool(not np.isfinite(rho_raw) or abs(rho_raw) >= RHO_CLIP),
        "sigma_matrix": sigma_matrix(np.array([tau2["L"], tau2["R"]]), rho),
    }


def sigma_matrix(tau2, rho):
    """Between-hitter covariance matrix from the two variances and their correlation."""
    off_diagonal = rho * np.sqrt(tau2[0] * tau2[1])
    matrix = np.array([[tau2[0], off_diagonal], [off_diagonal, tau2[1]]], dtype=float)
    eigenvalues = np.linalg.eigvalsh(matrix)
    assert eigenvalues.min() >= -1e-15, f"Sigma is not positive semidefinite: {eigenvalues}"
    return matrix


def posterior_mean(x, n_pa, mu, matrix, sigma2):
    """
    Empirical-Bayes posterior mean (BLUP) for every hitter at once.
    x: (N,2) observed side-specific wOBA, nan allowed where n_pa is 0.
    n_pa: (N,2) PA vs each hand. mu: (2,). matrix: (2,2) Sigma. sigma2: (2,).
    Returns (N,2) shrunk estimates.
    """
    x, n_pa = np.asarray(x, dtype=float), np.asarray(n_pa, dtype=float)
    assert x.shape == n_pa.shape and x.ndim == 2 and x.shape[1] == 2, \
        f"x and n_pa must both be (N,2), got {x.shape} and {n_pa.shape}"
    assert mu.shape == (2,) and matrix.shape == (2, 2) and sigma2.shape == (2,), \
        f"parameter shapes wrong: mu {mu.shape}, Sigma {matrix.shape}, sigma2 {sigma2.shape}"

    observed = n_pa > 0
    # a side with no PA contributes no evidence: infinite sampling variance, and its
    # residual is zeroed so a nan cannot propagate through the solve
    sampling_var = np.where(observed, sigma2[None, :] / np.maximum(n_pa, 1.0), ZERO_PA_VARIANCE)
    residual = np.where(observed, np.nan_to_num(x) - mu[None, :], 0.0)

    n_rows = len(x)
    total = np.broadcast_to(matrix, (n_rows, 2, 2)).copy()
    total[:, 0, 0] += sampling_var[:, 0]
    total[:, 1, 1] += sampling_var[:, 1]

    weighted = np.linalg.solve(total, residual[:, :, None])[:, :, 0]
    out = mu[None, :] + weighted @ matrix
    assert np.isfinite(out).all(), "posterior mean produced non-finite values"
    return out


def implied_split_constant(params):
    """
    The split-regression constant this fit implies, for comparison with The Book's
    2200 (RHB) / 1000 (LHB). n*_d = sigma^2_d / tau^2_d with tau^2_d the DERIVED split
    variance tau^2_L + tau^2_R - 2*tau_LR. Reported against the literature, never used
    to set anything. Returns (n_star_split, tau2_split).
    """
    # rho pinned at the clip is not an estimate, so nothing derived from it is either.
    # Printing a confident constant off a bound is exactly how the reverted attempt
    # produced its fake n* of ~46,878.
    if params["rho_at_bound"]:
        return np.nan, np.nan

    tau2 = params["tau2"]
    tau2_split = float(tau2[0] + tau2[1] - 2.0 * params["rho"] * np.sqrt(tau2[0] * tau2[1]))
    if tau2_split <= 0:
        return np.inf, tau2_split
    # The Book weights the hitter's own split by PA faced vs LHP, i.e. it treats the
    # strong side as known, so the relevant per-observation variance is the weak side's
    return float(params["sigma2"][0] / tau2_split), tau2_split


def predict(pa_df, eval_season, params=None, n_seasons=TRAILING_SEASONS, rho_override=None):
    """
    Project side-specific wOBA for every hitter-hand active in eval_season.
    params: output of `fit`; refit on the trailing window when omitted.
    rho_override: {batter_type: rho} to score a fixed-correlation variant (used only
    for The Book reference row); None uses the estimated rho.
    Returns batter/season/p_throws/pred_woba, ready for claim1_eval.
    """
    params = params or fit(pa_df, eval_season, n_seasons)
    hitters_only = drop_pitcher_batters(pa_df)
    window = trailing_window(hitters_only, eval_season, n_seasons)
    window = window[window["in_denominator"]]
    window = window.merge(batter_types(window), on="batter", how="left")
    pairs = to_pairs(side_observations(window))

    # the groups needing a projection are those active in the evaluated season
    active = hitters_only[hitters_only["season"] == eval_season]
    targets = active[["batter", "p_throws"]].drop_duplicates()
    known = targets[["batter"]].drop_duplicates().merge(pairs, on="batter", how="left")

    # handedness for hitters with no window PA: a static roster attribute known at
    # projection time, never an outcome. Nothing else is read from the eval season.
    debut_types = batter_types(active[active["batter"].isin(known[known["batter_type"].isna()]["batter"])])
    known = known.merge(debut_types.rename(columns={"batter_type": "debut_type"}), on="batter", how="left")
    n_debut = int(known["batter_type"].isna().sum())
    known["batter_type"] = known["batter_type"].fillna(known["debut_type"]).fillna("R")
    known[["n_L", "n_R"]] = known[["n_L", "n_R"]].fillna(0.0)

    predictions = []
    for batter_type, group in known.groupby("batter_type"):
        record = params.get(batter_type) or params[max(params, key=lambda t: params[t]["n_hitters"])]
        matrix = record["sigma_matrix"]
        if rho_override is not None:
            matrix = sigma_matrix(record["tau2"], rho_override[batter_type])
        shrunk = posterior_mean(group[["x_L", "x_R"]].to_numpy(), group[["n_L", "n_R"]].to_numpy(),
                                record["mu"], matrix, record["sigma2"])
        predictions.append(pd.DataFrame({"batter": group["batter"].to_numpy(),
                                         "L": shrunk[:, 0], "R": shrunk[:, 1]}))

    long = pd.concat(predictions, ignore_index=True).melt(
        id_vars="batter", var_name="p_throws", value_name="pred_woba")
    out = targets.merge(long, on=["batter", "p_throws"], how="left")
    out["season"] = eval_season
    assert out["pred_woba"].notna().all(), "predictions contain nulls"
    out.attrs["n_debut_handedness_from_eval_season"] = n_debut
    return out[KEY + ["pred_woba"]]
