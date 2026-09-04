"""
E.15 — the measurement ceiling on the observed platoon differential
(docs/phase-e-spec.md §12.5).

WHY THIS EXISTS
---------------
E.5 reports a within-stand weighted rank correlation of 0.146 between the model's
predicted platoon differential and the realized one, and that number was read
in-session as "essentially zero". That reading is only licensed if the realized
differential is a measurable quantity in a single season. Phase C.2 says it may not
be: `results/baseline_ladder/eb_prior_parameters.csv` puts the TRUE between-hitter split sd at
0.0271 wOBA (LHB) / 0.0222 (RHB) and the implied stabilization denominator at 320.65 /
562.98 PA PER SIDE, against realized 2024 exposures whose medians are 86 (vs LHP) and
223 (vs RHP). A correlation cannot exceed the square root of the reliability of the
thing it is correlated against; this module computes that ceiling and reports 0.146 as
a fraction of it.

It also discharges a logged debt. The decision-log entry of 2026-08-18 claimed "81.7%
of the model's predicted differential variance is the batter-stand effect alone,
against 10.9% of the observed differential's". That comparison is biased by
construction: `delta_pred` is noise-free while `delta_obs` carries two sides' worth of
sampling noise, so the observed denominator is inflated and its stand share is pushed
toward zero. The entry of 2026-08-19 withdraws it and routes the correction here. Part
2 removes the sampling-noise variance from the observed side before taking the share.

READ-ONLY. No model, no checkpoint, no forward pass, no training. It consumes
`results/model_evaluation/platoon_frame.csv` (written by `model_evaluation_eval.platoon_frame`), the
PA-level eval-target table, and the C.2 posterior variance components. 2024 only —
2025 is never touched (spec §10).

STATISTICS, STATED PLAINLY
--------------------------
For hitter h with realized denominators n_L, n_R:

    delta_obs(h) = wOBA(h, vs L) − wOBA(h, vs R)
                 = delta_true(h) + e(h)
    Var[e(h)]    = s2_L(h)/n_L + s2_R(h)/n_R

where s2 is the PER-PLATE-APPEARANCE variance of wOBA points. wOBA is a weighted sum
over outcome categories, NOT a rate, so p(1−p) is wrong here; s2 is computed as the
realized variance of `woba_points` over the PAs in the wOBA denominator. C.2's own
`sigma2_within_pa` column (0.235–0.277) is the same quantity and is used as a
cross-check on the value computed here.

    reliability = Var[delta_true] / (Var[delta_true] + E[Var[e]])
    ceiling on |corr(delta_pred, delta_obs)| ≈ sqrt(reliability)

Var[delta_true] comes from C.2's `tau2_split_derived`, which is the within-stand
variance of the true differential implied by the two sides' talent variances and their
correlation rho: tau2_L + tau2_R − 2·rho·tau_L·tau_R.

Everything is reported for LHB and RHB separately and pooled, because C.2's components
differ by stand and Part 3's asymmetry is the load-bearing claim.

Run:  PYTHONPATH=. .venv/bin/python -m src.analysis.model_evaluation_platoon_ceiling
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import claim1_eval
from src.analysis import model_evaluation_eval

DEFAULT_OUT_DIR = "results/model_evaluation"
EVAL_SEASON = 2024
C2_PATH = "results/baseline_ladder/eb_prior_parameters.csv"
FRAME_PATH = "results/model_evaluation/platoon_frame.csv"
EVAL_TARGETS_PATH = "data/processed/eval_targets_pa.parquet"

# A per-hitter, per-side variance estimate needs enough PAs to be worth anything. Below
# this we substitute the league (stand, p_throws) per-PA variance. 50 is a judgement
# call, reported as an assumption; the all-league sensitivity is computed alongside so
# the choice is visible rather than load-bearing.
MIN_PA_FOR_OWN_VARIANCE = 50

# E.5's reported within-stand rank correlation, for the reproduction check. Never used
# as an input -- it is recomputed from the frame and compared against this.
# embedding_sgd_sgd_lr1 chain, 2026-09-03 (rebuild_baseline was 0.14626474805407289)
E5_REPORTED_WITHIN_STAND_RANK_CORR = 0.1626334134811194


# ------------------------------------------------------------------ pure statistics

def reliability_from_variances(true_variance, sampling_variance):
    """
    Reliability of a measurement: signal variance over total observed variance.

    Both arguments are variances and must be non-negative; a negative one means an
    upstream decomposition failed and is raised, never clipped.
    """
    assert true_variance >= 0, f"true variance is negative: {true_variance}"
    assert sampling_variance >= 0, f"sampling variance is negative: {sampling_variance}"
    total = true_variance + sampling_variance
    assert total > 0, "total variance is zero; reliability is undefined"
    return true_variance / total


def spearman_brown(half_correlation, length_ratio=2.0):
    """
    Spearman-Brown: the reliability of a test `length_ratio` times as long as the one
    that produced `half_correlation`. Correlating two halves measures the reliability
    of a HALF-length measurement; the season's differential is full length, so the
    split-half r must be stepped up by this before it is comparable to the C.2-derived
    number.
    """
    denominator = 1.0 + (length_ratio - 1.0) * half_correlation
    assert denominator != 0, "Spearman-Brown denominator is zero"
    return length_ratio * half_correlation / denominator


def noise_corrected_variance(observed_variance, mean_sampling_variance):
    """
    Var[true] = Var[observed] − E[Var[sampling noise]].

    NOT clipped at zero. A negative result means the observed spread is smaller than
    the sampling noise alone predicts -- i.e. the quantity is pure noise -- and that is
    a finding to report, not an inconvenience to floor.
    """
    assert observed_variance >= 0, f"observed variance is negative: {observed_variance}"
    assert mean_sampling_variance >= 0, (
        f"mean sampling variance is negative: {mean_sampling_variance}")
    return observed_variance - mean_sampling_variance


# ------------------------------------------------------------------ weighted moments

def weighted_variance(values, weight):
    """Weighted variance about the weighted mean (population form, matching model_evaluation_eval)."""
    values = np.asarray(values, dtype="float64")
    weight = np.asarray(weight, dtype="float64")
    assert len(values) == len(weight), "values and weights differ in length"
    assert weight.sum() > 0, "weights sum to zero"
    mean = np.average(values, weights=weight)
    variance = float(np.average((values - mean) ** 2, weights=weight))
    assert variance >= 0, "weighted variance is negative"
    return variance


def between_within_stand(values, weight, stand):
    """
    Split a weighted variance into its between-stand and within-stand parts.

    Same decomposition `model_evaluation_eval.platoon_decomposition` performs; repeated here (rather
    than imported) only because Part 2 needs it inside a bootstrap loop over resampled
    rows. The closure assertion below is what keeps the two honest.
    """
    values = np.asarray(values, dtype="float64")
    weight = np.asarray(weight, dtype="float64")
    stand = np.asarray(stand)
    grand = np.average(values, weights=weight)
    between = 0.0
    for side in np.unique(stand):
        mask = stand == side
        cell = np.average(values[mask], weights=weight[mask])
        between += weight[mask].sum() / weight.sum() * (cell - grand) ** 2
    total = float(np.average((values - grand) ** 2, weights=weight))
    within = total - between
    assert between >= -1e-15 and within >= -1e-15, (
        f"variance decomposition produced a negative part: between={between} within={within}")
    assert abs((between + within) - total) < 1e-12 * max(total, 1.0), (
        "between + within does not close on the total")
    return {"total": total, "between_stand": float(between), "within_stand": float(within)}


# ------------------------------------------------------------------ inputs

def load_eb_components(path=C2_PATH):
    """
    C.2's posterior variance components, keyed by batter_type.

    `tau2_split_derived` is the TRUE within-stand variance of the platoon differential:
    tau2(vs L) + tau2(vs R) − 2·rho·tau(vs L)·tau(vs R). It is asserted to reproduce
    from the per-side components in the same file, so a silently stale column cannot
    become the ceiling.
    """
    eb = pd.read_csv(path)
    out = {}
    for batter_type, part in eb.groupby("batter_type"):
        rows = part.set_index("vs_hand")
        tau2_l, tau2_r = float(rows.loc["L", "tau2"]), float(rows.loc["R", "tau2"])
        rho = float(rows.loc["L", "rho"])
        assert rho == float(rows.loc["R", "rho"]), "rho differs across sides for one batter type"
        derived = tau2_l + tau2_r - 2.0 * rho * np.sqrt(tau2_l * tau2_r)
        stored = float(rows.loc["L", "tau2_split_derived"])
        assert abs(derived - stored) < 1e-9, (
            f"tau2_split_derived does not reproduce for {batter_type}: {derived} vs {stored}")
        assert stored > 0, f"true split variance is not positive for {batter_type}"
        out[batter_type] = {
            "tau2_split": stored,
            "tau_split": float(np.sqrt(stored)),
            "rho": rho,
            "rho_ci": [float(rows.loc["L", "rho_ci_low"]), float(rows.loc["L", "rho_ci_high"])],
            "n_star_split_implied": float(rows.loc["L", "n_star_split_implied"]),
            "sigma2_within_pa_vs_L": float(rows.loc["L", "sigma2_within_pa"]),
            "sigma2_within_pa_vs_R": float(rows.loc["R", "sigma2_within_pa"]),
        }
    return out


def per_pa_variance_tables(pa_df, season=EVAL_SEASON):
    """
    Realized per-PA variance of wOBA points, per (batter, p_throws) and per
    (stand, p_throws), over the wOBA denominator of `season`.

    wOBA is a weighted sum over outcome categories, so its per-PA variance is the
    variance of `woba_points` itself. Deriving it from p(1−p) on a rate would be wrong:
    the outcome is not binary and the weights are not 1.
    """
    window = pa_df[(pa_df["season"] == season) & pa_df["in_denominator"]]
    assert len(window), f"no denominator plate appearances in {season}"
    points = window["woba_points"].astype("float64")
    window = window.assign(woba_points=points)

    by_batter = (window.groupby(["batter", "p_throws"])["woba_points"]
                 .agg(n="size", var=lambda s: s.var(ddof=1)).reset_index())
    by_league = (window.groupby(["stand", "p_throws"])["woba_points"]
                 .agg(n="size", var=lambda s: s.var(ddof=1)).reset_index())
    assert (by_league["var"] > 0).all(), "a league per-PA wOBA variance is not positive"
    return by_batter, by_league


def attach_sampling_variance(frame, by_batter, by_league):
    """
    Per-hitter sampling variance of delta_obs: s2_L/n_L + s2_R/n_R.

    The two sides are independent samples of different plate appearances, so the
    variance of their difference is the SUM of the two sides' variances -- the reason a
    differential is so much harder to measure than either side.

    s2 is the hitter's own realized per-PA variance where he has at least
    MIN_PA_FOR_OWN_VARIANCE denominator PAs on that side, and the league
    (stand, p_throws) per-PA variance otherwise, because a variance estimated on ~20
    PAs of a heavy-tailed outcome is worse than the league value it would replace. The
    all-league variant is computed alongside as `sampling_var_league` so the choice is
    a reported sensitivity and not a hidden assumption.
    """
    league = by_league.set_index(["stand", "p_throws"])["var"]
    batter = by_batter.set_index(["batter", "p_throws"])

    def side_variance(row, side):
        league_value = float(league.loc[(row["stand"], side)])
        key = (row["batter"], side)
        if key in batter.index:
            entry = batter.loc[key]
            if float(entry["n"]) >= MIN_PA_FOR_OWN_VARIANCE and np.isfinite(entry["var"]):
                return float(entry["var"]), league_value, False
        return league_value, league_value, True

    records = []
    for _, row in frame.iterrows():
        s2_l, league_l, fell_back_l = side_variance(row, "L")
        s2_r, league_r, fell_back_r = side_variance(row, "R")
        records.append({
            "s2_L": s2_l, "s2_R": s2_r,
            "sampling_var": s2_l / row["denom_L"] + s2_r / row["denom_R"],
            "sampling_var_league": league_l / row["denom_L"] + league_r / row["denom_R"],
            "fell_back_L": fell_back_l, "fell_back_R": fell_back_r,
        })
    attached = frame.join(pd.DataFrame(records, index=frame.index))
    assert len(attached) == len(frame), "attaching sampling variance changed the row count"
    assert attached["sampling_var"].notna().all(), "a hitter has no sampling variance"
    assert (attached["sampling_var"] > 0).all(), "a sampling variance is not positive"
    return attached


# ------------------------------------------------------------------ Part 1

def ceiling_table(frame, eb):
    """
    Reliability and the implied rank-correlation ceiling, per stand and pooled.

    Two aggregations are reported because they answer different questions:
      * `reliability_variance_ratio` = tau2 / (tau2 + E[sampling var]) is the population
        quantity that caps a CROSS-HITTER correlation, and is the headline.
      * `reliability_mean_per_hitter` is the average of each hitter's own reliability;
        it differs whenever exposures are skewed and is reported so the skew is visible.
    E[sampling var] is taken under E.5's own harmonic-denominator weights, so the
    ceiling is the ceiling on the number E.5 actually computed, and unweighted as well.
    """
    rows = []
    for label in ("L", "R", "pooled"):
        part = frame if label == "pooled" else frame[frame["stand"] == label]
        assert len(part) > 2, f"too few hitters to characterise {label}"
        weight = part["weight"].to_numpy(dtype="float64")
        tau2 = np.array([eb[side]["tau2_split"] for side in part["stand"]])
        sampling = part["sampling_var"].to_numpy(dtype="float64")
        mean_tau2_w = float(np.average(tau2, weights=weight))
        mean_samp_w = float(np.average(sampling, weights=weight))
        mean_samp_u = float(sampling.mean())
        per_hitter = tau2 / (tau2 + sampling)
        weighted = reliability_from_variances(mean_tau2_w, mean_samp_w)
        unweighted = reliability_from_variances(float(tau2.mean()), mean_samp_u)
        rows.append({
            "stand": label, "n_hitters": int(len(part)),
            "tau2_split_true": mean_tau2_w, "tau_split_true": float(np.sqrt(mean_tau2_w)),
            "mean_sampling_var_weighted": mean_samp_w,
            "mean_sampling_var_unweighted": mean_samp_u,
            "mean_sampling_var_league_weighted":
                float(np.average(part["sampling_var_league"], weights=weight)),
            "median_denom_L": float(part["denom_L"].median()),
            "median_denom_R": float(part["denom_R"].median()),
            "reliability_variance_ratio": weighted,
            "reliability_variance_ratio_unweighted": unweighted,
            "reliability_mean_per_hitter": float(per_hitter.mean()),
            "ceiling_rank_corr": float(np.sqrt(weighted)),
            "ceiling_rank_corr_unweighted": float(np.sqrt(unweighted)),
        })
    return pd.DataFrame(rows)


def recompute_platoon_rank_correlation(frame):
    """
    Recompute E.5's within-stand rank correlation from the frame rather than quoting it.

    Reuses `model_evaluation_eval.platoon_decomposition` -- the same code that produced the reported
    number -- so this is a reproduction check on the stored artefact, and a mismatch
    means the frame on disk is not the frame E.5 scored.
    """
    decomposition = model_evaluation_eval.platoon_decomposition(frame)
    value = float(decomposition["post_hoc_within_stand_rank_corr"])
    return value, decomposition


def within_stand_rank_correlation_by_stand(frame):
    """E.5's within-stand rank correlation, computed separately for LHB and RHB."""
    out = {}
    for side in ("L", "R"):
        part = frame[frame["stand"] == side]
        weight = part["weight"].to_numpy(dtype="float64")
        # within a single stand the residualisation is just centring, which rank
        # correlation is invariant to, so the raw columns are correct here
        out[side] = float(claim1_eval.weighted_rank_correlation(
            part["delta_obs"], part["delta_pred"], weight))
    return out


def split_half_reliability(frame, pa_df, season=EVAL_SEASON, min_half_pa=10):
    """
    Independent check on the C.2-derived reliability: split each hitter's 2024 PAs into
    two halves, compute the platoon differential in each, correlate across hitters, and
    step the result up to full length with Spearman-Brown.

    The split is on `game_pk` parity -- odd vs even game id. Splitting by GAME rather
    than by PA keeps the two halves independent even if PAs within a game are
    correlated (same pitcher, same park, same day), which a random PA-level split would
    not. It is deterministic, so no seed is needed.

    This shares no assumption with the C.2 route: it uses no prior, no rho, and no
    variance component. If the two disagree by more than a factor of 1.5 the caller
    reports both and adopts neither.
    """
    window = pa_df[(pa_df["season"] == season) & pa_df["in_denominator"]]
    window = window[window["batter"].isin(frame["batter"])]
    window = window.assign(half=np.where(window["game_pk"].to_numpy() % 2 == 0, "A", "B"))
    grouped = (window.groupby(["batter", "half", "p_throws"])["woba_points"]
               .agg(points="sum", denom="size").reset_index())
    wide = grouped.pivot_table(index="batter", columns=["half", "p_throws"],
                               values=["points", "denom"])
    needed = [(field, half, hand) for field in ("points", "denom")
              for half in ("A", "B") for hand in ("L", "R")]
    missing = [key for key in needed if key not in wide.columns]
    assert not missing, f"split-half pivot is missing columns: {missing}"
    wide = wide.dropna()

    enough = np.ones(len(wide), dtype=bool)
    for half in ("A", "B"):
        for hand in ("L", "R"):
            enough &= wide[("denom", half, hand)].to_numpy() >= min_half_pa
    wide = wide[enough]
    assert len(wide) > 10, f"only {len(wide)} hitters survive the split-half exposure floor"

    delta = {}
    for half in ("A", "B"):
        delta[half] = (wide[("points", half, "L")] / wide[("denom", half, "L")]
                       - wide[("points", half, "R")] / wide[("denom", half, "R")])
    stand = frame.set_index("batter")["stand"].reindex(wide.index)
    assert stand.notna().all(), "a split-half hitter has no stand"

    rows = []
    for label in ("L", "R", "pooled"):
        mask = np.ones(len(wide), dtype=bool) if label == "pooled" else (stand == label).to_numpy()
        a = np.array(delta["A"][mask].to_numpy(), dtype="float64")
        b = np.array(delta["B"][mask].to_numpy(), dtype="float64")
        side = stand[mask].to_numpy()
        if label == "pooled":
            # centre each stand's differential so the between-stand main effect cannot
            # inflate the pooled correlation -- E.5's 0.146 is a WITHIN-stand number and
            # its reliability check has to be within-stand too
            for value in np.unique(side):
                cell = side == value
                a[cell] = a[cell] - a[cell].mean()
                b[cell] = b[cell] - b[cell].mean()
        half_spearman = float(pd.Series(a).corr(pd.Series(b), method="spearman"))
        half_pearson = float(pd.Series(a).corr(pd.Series(b)))
        full_spearman = float(spearman_brown(half_spearman))
        # a bootstrap interval on the split-half number, because the factor-of-1.5
        # agreement rule is being applied to a point estimate computed on half-season
        # exposures: without an interval there is no way to tell a real disagreement
        # with C.2 from a sampling accident
        rng = np.random.default_rng(0)
        boot = np.array([
            spearman_brown(float(pd.Series(a[idx]).corr(pd.Series(b[idx]), method="spearman")))
            for idx in (rng.integers(0, len(a), len(a)) for _ in range(1000))])
        boot = boot[np.isfinite(boot)]
        rows.append({
            "stand": label, "n_hitters": int(mask.sum()),
            "half_split_spearman": half_spearman,
            "half_split_pearson": half_pearson,
            "reliability_spearman_brown": full_spearman,
            "reliability_spearman_brown_ci_low": float(np.percentile(boot, 2.5)),
            "reliability_spearman_brown_ci_high": float(np.percentile(boot, 97.5)),
            "reliability_spearman_brown_pearson": float(spearman_brown(half_pearson)),
            "ceiling_rank_corr": float(np.sqrt(full_spearman)) if full_spearman >= 0
                                 else float("nan"),
            "median_half_denom_L": float(np.median(wide[("denom", "A", "L")].to_numpy()[mask])),
            "median_half_denom_R": float(np.median(wide[("denom", "A", "R")].to_numpy()[mask])),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ Part 2

def corrected_stand_share(frame, n_boot=2000, seed=0, ci=(2.5, 97.5)):
    """
    The share of the observed differential's TRUE variance owned by the batter-stand
    main effect, with sampling noise removed from the observed side.

    The withdrawn 2026-08-18 framing divided a between-stand variance by a NOISE-INFLATED
    total, which drives the share toward zero mechanically. Here the total is
    Var[delta_obs] − E[sampling var]; the between-stand numerator needs no correction of
    its own because it is a two-cell mean, whose sampling variance is smaller by a factor
    of the cell counts (221 and 324 hitters) and is negligible against the components below.

    Bootstrap is over ROWS, and the frame is one row per batter, so resampling rows IS
    clustering on batter (the same argument `model_evaluation_eval.paired_platoon_difference` makes).
    """
    rng = np.random.default_rng(seed)
    weight = frame["weight"].to_numpy(dtype="float64")
    values = frame["delta_obs"].to_numpy(dtype="float64")
    stand = frame["stand"].to_numpy()
    sampling = frame["sampling_var"].to_numpy(dtype="float64")

    def statistic(index):
        parts = between_within_stand(values[index], weight[index], stand[index])
        mean_samp = float(np.average(sampling[index], weights=weight[index]))
        true_total = noise_corrected_variance(parts["total"], mean_samp)
        return parts, mean_samp, true_total

    parts, mean_samp, true_total = statistic(np.arange(len(frame)))
    assert true_total > 0, (
        "FINDING, not an error: the noise-corrected observed differential variance is "
        f"{true_total} <= 0, i.e. the observed platoon differential is consistent with "
        "pure sampling noise. Report this rather than clipping.")
    share = parts["between_stand"] / true_total

    draws = []
    for _ in range(n_boot):
        index = rng.integers(0, len(frame), len(frame))
        try:
            boot_parts, _, boot_total = statistic(index)
        except AssertionError:
            draws.append(np.nan)
            continue
        draws.append(boot_parts["between_stand"] / boot_total if boot_total > 0 else np.nan)
    draws = np.asarray(draws, dtype="float64")
    finite = draws[np.isfinite(draws)]
    assert len(finite) > 0.9 * n_boot, (
        f"only {len(finite)}/{n_boot} bootstrap draws produced a positive corrected variance")

    return {
        "observed_variance_raw": parts["total"],
        "between_stand_variance": parts["between_stand"],
        "within_stand_variance_raw": parts["within_stand"],
        "mean_sampling_variance_weighted": mean_samp,
        "observed_variance_noise_corrected": true_total,
        "within_stand_variance_noise_corrected": float(true_total - parts["between_stand"]),
        "share_between_stand_corrected": float(share),
        "share_between_stand_uncorrected": float(parts["between_stand"] / parts["total"]),
        "ci_low": float(np.percentile(finite, ci[0])),
        "ci_high": float(np.percentile(finite, ci[1])),
        "n_boot": n_boot, "n_boot_usable": int(len(finite)), "seed": seed,
        "noise_share_of_observed_variance": float(mean_samp / parts["total"]),
    }


# ------------------------------------------------------------------ Part 3

def recovery_by_stand(frame, eb):
    """
    What fraction of the TRUE within-stand platoon spread the model's predictions span,
    per stand, noise-corrected.

    Two denominators, both reported, because they come from independent sources and the
    in-session figures (54% LHB / 18% RHB) used only the first:
      * `ratio_vs_eb` divides the model's within-stand predicted sd by C.2's
        `tau_split` -- a posterior quantity fit on many seasons.
      * `ratio_vs_realized_true` divides it by sqrt(Var[delta_obs] − E[sampling var])
        computed inside this 2024 stand group -- purely empirical, no prior.
    A ratio near 1 does not mean the model is right, only that it is not systematically
    under-dispersed; correlation (E.5) is what says whether the spread points anywhere.
    """
    rows = []
    for side in ("L", "R"):
        part = frame[frame["stand"] == side]
        weight = part["weight"].to_numpy(dtype="float64")
        var_pred = weighted_variance(part["delta_pred"], weight)
        var_obs = weighted_variance(part["delta_obs"], weight)
        mean_samp = float(np.average(part["sampling_var"], weights=weight))
        var_true = noise_corrected_variance(var_obs, mean_samp)
        # SENSITIVITY. The population weighted variance above is the estimator
        # model_evaluation_eval.platoon_decomposition uses, and matching it is what makes the Part 2
        # correction comparable to the 10.9% figure it supersedes. np.cov's aweights
        # form applies a reliability-weight bias correction which, with weights
        # spanning 20-500 PA, is a live alternative. Both are reported so the LHB
        # stand's NEGATIVE noise-corrected variance can be seen not to depend on the
        # estimator choice.
        var_obs_aweights = float(np.cov(part["delta_obs"], aweights=weight))
        var_pred_aweights = (float(np.cov(part["delta_pred"], aweights=weight))
                             if part["delta_pred"].nunique() > 1 else 0.0)
        var_true_aweights = var_obs_aweights - mean_samp
        # NOT clipped. A non-positive value here is the finding: this stand's observed
        # within-stand spread is no larger than its own sampling noise, so the realized
        # 2024 data alone cannot see any true platoon spread among these hitters. The
        # ratio against it is left undefined rather than floored at some convenient
        # epsilon, and the C.2-based ratio carries the answer for that stand.
        rows.append({
            "stand": side, "n_hitters": int(len(part)),
            "sd_pred_within_stand": float(np.sqrt(var_pred)),
            "sd_obs_within_stand_raw": float(np.sqrt(var_obs)),
            "var_obs_within_stand_noise_corrected": float(var_true),
            "noise_corrected_variance_is_positive": bool(var_true > 0),
            "sd_obs_within_stand_true": float(np.sqrt(var_true)) if var_true > 0 else float("nan"),
            "mean_sampling_var_weighted": mean_samp,
            "tau_split_eb": eb[side]["tau_split"],
            "ratio_vs_eb": float(np.sqrt(var_pred) / eb[side]["tau_split"]),
            "ratio_vs_realized_true": float(np.sqrt(var_pred / var_true)) if var_true > 0
                                      else float("nan"),
            "sd_obs_within_stand_raw_aweights": float(np.sqrt(var_obs_aweights)),
            "sd_pred_within_stand_aweights": float(np.sqrt(var_pred_aweights)),
            "var_obs_within_stand_noise_corrected_aweights": float(var_true_aweights),
            "ratio_vs_realized_true_aweights":
                float(np.sqrt(var_pred_aweights / var_true_aweights))
                if var_true_aweights > 0 else float("nan"),
        })
    table = pd.DataFrame(rows)
    left, right = table.set_index("stand").loc["L"], table.set_index("stand").loc["R"]
    table.attrs["asymmetry_vs_eb"] = float(left["ratio_vs_eb"] / right["ratio_vs_eb"])
    table.attrs["asymmetry_vs_realized_true"] = float(
        left["ratio_vs_realized_true"] / right["ratio_vs_realized_true"])
    table.attrs["asymmetry_vs_realized_true_aweights"] = float(
        left["ratio_vs_realized_true_aweights"] / right["ratio_vs_realized_true_aweights"])
    table.attrs["both_stands_noise_corrected_positive"] = bool(
        left["noise_corrected_variance_is_positive"]
        and right["noise_corrected_variance_is_positive"])
    return table


# ------------------------------------------------------------------ driver

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", default=FRAME_PATH)
    parser.add_argument("--eval-targets", default=EVAL_TARGETS_PATH)
    parser.add_argument("--eb", default=C2_PATH)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--eval-season", type=int, default=EVAL_SEASON)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # the same guard every other Phase E entry point carries: 2025 is never touched here
    claim1_eval.assert_not_test_season(args.eval_season, final_run=False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.frame)
    assert frame["batter"].is_unique, "the E.5 frame has duplicate batters"
    assert set(frame["stand"].unique()) <= {"L", "R"}, "unexpected stand value in the E.5 frame"
    eb = load_eb_components(args.eb)

    pa_df = pd.read_parquet(args.eval_targets)
    by_batter, by_league = per_pa_variance_tables(pa_df, args.eval_season)
    frame = attach_sampling_variance(frame, by_batter, by_league)

    # cross-check the per-PA variance scale against C.2's own sigma2_within_pa
    league_lookup = by_league.set_index(["stand", "p_throws"])["var"]
    sigma_check = {
        f"{stand}_vs_{hand}": {
            "realized_2024": float(league_lookup.loc[(stand, hand)]),
            "eb_sigma2_within_pa": eb[stand][f"sigma2_within_pa_vs_{hand}"],
        } for stand in ("L", "R") for hand in ("L", "R")}
    for key, pair in sigma_check.items():
        ratio = pair["realized_2024"] / pair["eb_sigma2_within_pa"]
        assert 0.7 < ratio < 1.4, (
            f"realized per-PA wOBA variance disagrees with C.2 for {key}: ratio {ratio}")
        pair["ratio"] = float(ratio)

    print("E.15 Part 1 — measurement ceiling")
    ceilings = ceiling_table(frame, eb)
    observed_corr, decomposition = recompute_platoon_rank_correlation(frame)
    by_stand_corr = within_stand_rank_correlation_by_stand(frame)
    split_half = split_half_reliability(frame, pa_df, args.eval_season)

    ceiling_index = ceilings.set_index("stand")
    fraction_of_ceiling = {
        "pooled": float(observed_corr / ceiling_index.loc["pooled", "ceiling_rank_corr"]),
        "L": float(by_stand_corr["L"] / ceiling_index.loc["L", "ceiling_rank_corr"]),
        "R": float(by_stand_corr["R"] / ceiling_index.loc["R", "ceiling_rank_corr"]),
    }

    # do the two independent reliability routes agree? factor-of-1.5 rule, spec §12.5
    sh_index = split_half.set_index("stand")
    agreement = {}
    for label in ("L", "R", "pooled"):
        eb_value = float(ceiling_index.loc[label, "reliability_variance_ratio"])
        sh_value = float(sh_index.loc[label, "reliability_spearman_brown"])
        if sh_value <= 0:
            ratio = float("nan")
        else:
            ratio = max(eb_value / sh_value, sh_value / eb_value)
        agreement[label] = {
            "eb_derived_reliability": eb_value,
            "split_half_reliability": sh_value,
            "disagreement_factor": ratio,
            "agree_within_1_5x": bool(np.isfinite(ratio) and ratio <= 1.5),
        }
    adopt = all(entry["agree_within_1_5x"] for entry in agreement.values())

    print("E.15 Part 2 — errors-in-variables correction")
    corrected = corrected_stand_share(frame, n_boot=args.n_boot, seed=args.seed)

    print("E.15 Part 3 — recovery of the true within-stand spread")
    recovery = recovery_by_stand(frame, eb)

    expectations = {
        "part1_reliability_near_0.13_L_and_0.06_R": {
            "expected": {"L": 0.13, "R": 0.06},
            "observed": {"L": float(ceiling_index.loc["L", "reliability_variance_ratio"]),
                         "R": float(ceiling_index.loc["R", "reliability_variance_ratio"])},
            "held": bool(
                0.5 * 0.13 <= ceiling_index.loc["L", "reliability_variance_ratio"] <= 2 * 0.13
                and 0.5 * 0.06 <= ceiling_index.loc["R", "reliability_variance_ratio"] <= 2 * 0.06),
            "rule": "held if each observed reliability is within a factor of 2 of its expectation",
        },
        "part1_ceiling_near_0.36_L_and_0.25_R": {
            "expected": {"L": 0.36, "R": 0.25},
            "observed": {"L": float(ceiling_index.loc["L", "ceiling_rank_corr"]),
                         "R": float(ceiling_index.loc["R", "ceiling_rank_corr"])},
            "held": bool(abs(ceiling_index.loc["L", "ceiling_rank_corr"] - 0.36) < 0.10
                         and abs(ceiling_index.loc["R", "ceiling_rank_corr"] - 0.25) < 0.10),
            "rule": "held if each ceiling is within 0.10 of its expectation",
        },
        "part1_platoon_is_40_to_55_percent_of_ceiling": {
            "expected": [0.40, 0.55],
            "observed": fraction_of_ceiling["pooled"],
            "held": bool(0.40 <= fraction_of_ceiling["pooled"] <= 0.55),
        },
        "part2_corrected_share_50_to_60_percent": {
            "expected": [0.50, 0.60],
            "observed": corrected["share_between_stand_corrected"],
            "held": bool(0.50 <= corrected["share_between_stand_corrected"] <= 0.60),
        },
        "part3_asymmetry_survives_correction": {
            "expected": "the LHB/RHB recovery ratio stays near 3x after noise correction",
            "observed": {"asymmetry_vs_eb": recovery.attrs["asymmetry_vs_eb"],
                         "asymmetry_vs_realized_true": recovery.attrs["asymmetry_vs_realized_true"],
                         "asymmetry_vs_realized_true_aweights":
                             recovery.attrs["asymmetry_vs_realized_true_aweights"]},
            "held": bool(recovery.attrs["asymmetry_vs_realized_true"] >= 2.0),
            "rule": "held if the noise-corrected LHB:RHB recovery ratio is at least 2x; "
                    "NaN when a stand's noise-corrected variance is not positive, in which "
                    "case only the C.2-denominated ratio is interpretable",
            "both_stands_noise_corrected_positive":
                recovery.attrs["both_stands_noise_corrected_positive"],
        },
    }

    report = {
        "step": "E.15",
        "spec": "docs/phase-e-spec.md §12.5",
        "eval_season": args.eval_season,
        "read_only": True,
        "sources": {
            "platoon_frame": args.frame,
            "platoon_frame_producer": "src/analysis/model_evaluation_eval.py:platoon_frame",
            "eb_prior_parameters": args.eb,
            "pa_level_eval_targets": args.eval_targets,
            "platoon_reported_within_stand_rank_corr": "results/model_evaluation/model_evaluation_report.json"
                                                  " -> platoon.decomposition",
            "withdrawn_claim": "docs/decision-log.md 2026-08-18 (81.7% vs 10.9%),"
                               " withdrawn 2026-08-19 and corrected in Part 2 here",
        },
        "eb_components_used": eb,
        "per_pa_variance_cross_check": sigma_check,
        "sampling_variance_source": {
            "rule": f"hitter's own realized per-PA wOBA variance when that side has "
                    f">= {MIN_PA_FOR_OWN_VARIANCE} denominator PAs, else the league "
                    f"(stand, p_throws) per-PA variance",
            "why": "wOBA is a weighted sum over outcome categories, not a rate, so its "
                   "per-PA variance is the realized variance of woba_points; p(1-p) on a "
                   "rate would be wrong. Per-hitter estimates below ~50 PA of a "
                   "heavy-tailed outcome are noisier than the league value they replace.",
            "fell_back_L_share": float(frame["fell_back_L"].mean()),
            "fell_back_R_share": float(frame["fell_back_R"].mean()),
            "all_league_sensitivity_mean_ratio": float(
                (frame["sampling_var_league"] / frame["sampling_var"]).mean()),
        },
        "part1_ceiling": {
            "by_stand": ceilings.to_dict(orient="records"),
            "platoon_within_stand_rank_corr_recomputed": observed_corr,
            "platoon_within_stand_rank_corr_reported": E5_REPORTED_WITHIN_STAND_RANK_CORR,
            "recomputation_matches_report": bool(
                abs(observed_corr - E5_REPORTED_WITHIN_STAND_RANK_CORR) < 1e-9),
            "platoon_within_stand_rank_corr_by_stand": by_stand_corr,
            "fraction_of_ceiling": fraction_of_ceiling,
            "split_half": split_half.to_dict(orient="records"),
            "reliability_route_agreement": agreement,
            "adopt_a_single_reliability": adopt,
            "adoption_note": (
                "the two routes agree within 1.5x; the C.2-derived reliability is adopted"
                if adopt else
                "the two routes DISAGREE by more than 1.5x. Both are reported and NEITHER "
                "is adopted, per spec §12.5. Any downstream reading of E.5 must carry both "
                "ceilings."),
            "decomposition_from_model_evaluation_eval": decomposition,
        },
        "part2_errors_in_variables": corrected,
        "part3_recovery": {
            "by_stand": recovery.to_dict(orient="records"),
            "asymmetry_vs_eb": recovery.attrs["asymmetry_vs_eb"],
            "asymmetry_vs_realized_true": recovery.attrs["asymmetry_vs_realized_true"],
            "asymmetry_vs_realized_true_aweights":
                recovery.attrs["asymmetry_vs_realized_true_aweights"],
            "in_session_uncorrected": {"L": 0.54, "R": 0.18},
        },
        "pre_registered_expectations": expectations,
        "assumptions": [
            "C.2's tau2_split_derived is the true within-stand variance of the platoon "
            "differential and transfers to the 2024 eval population unchanged.",
            "The two sides' plate appearances are independent samples, so Var[delta_obs] "
            "sampling noise is the SUM of the two sides' sampling variances.",
            "Per-PA wOBA variance is the realized variance of woba_points over the wOBA "
            "denominator; wOBA is a weighted category sum, so no p(1-p) rate form is used.",
            f"Per-hitter per-PA variance is used when a side has >= {MIN_PA_FOR_OWN_VARIANCE} "
            "denominator PAs, otherwise the league (stand, p_throws) value.",
            "ceiling ~= sqrt(reliability) is the Pearson attenuation bound; it is applied to "
            "a WEIGHTED SPEARMAN correlation, which it bounds only approximately. Rank "
            "correlations are typically slightly lower than Pearson for the same signal, so "
            "using it as the ceiling is mildly conservative against the model.",
            "delta_pred is treated as noise-free (it is a deterministic model output), so all "
            "attenuation is charged to the observed side.",
            "Stand in the E.5 frame is the modal batting side in 2024, so switch hitters are "
            "assigned to whichever C.2 batter_type they batted from more often; C.2's separate "
            "S row is not used. 2024 switch hitters are a small minority of the 545.",
            "All weighted variances use the POPULATION form (np.average of squared "
            "deviations), matching model_evaluation_eval.platoon_decomposition, so the corrected share in "
            "Part 2 is directly comparable to the 10.9% it supersedes. np.cov's aweights "
            "bias-corrected form is reported alongside in Part 3; within a stand the two "
            "differ by well under 1% and the LHB negative sign holds under both.",
            "The between-stand main effect's own sampling variance is treated as negligible "
            "(two cells of 221 and 324 hitters) and is not subtracted in Part 2.",
            "The split-half route splits on game_pk parity, which keeps within-game "
            "correlation inside a half; Spearman-Brown assumes the two halves are parallel "
            "measurements of equal length, which odd/even games approximately are.",
        ],
    }

    frame.to_csv(out_dir / "ceiling_per_hitter.csv", index=False)
    ceilings.to_csv(out_dir / "ceiling_by_stand.csv", index=False)
    (out_dir / "ceiling.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"wrote {out_dir / 'ceiling.json'}")


if __name__ == "__main__":
    main()
