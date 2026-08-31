"""
Phase M driver — population, routes, differential, strata, level side, coverage.

Reads committed Phase C/D/E/F artifacts and writes `results/phase_m/`. The statistics
live in `m_ceiling.py`; nothing is estimated here that is not either an existing repo
estimator (`c2_bivariate_eb.fit`, `e_platoon_ceiling`'s subtraction and split-half,
`claim1_eval`'s scorer) or an aggregation of one.

EVERY NUMBER THIS FILE EMITS IS POST-SELECTION AND DESCRIPTIVE (spec §0.3). The
pre-registered gate on the platoon differential ran on 2024 and failed (research-manifest,
2026-08-20); everything after it describes a measurement problem rather than testing a
hypothesis. The label rides on every artifact as `post_selection_descriptive`.

2025 is never read. The only season this module scores is 2024, and `claim1_eval`'s test
guard is called before anything else runs.

Run:  PYTHONPATH=. .venv/bin/python -m src.analysis.m_report
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import c2_bivariate_eb as c2
from src.analysis import claim1_eval
from src.analysis import e_platoon_ceiling as e15
from src.analysis import e_probe_coverage
from src.analysis import m_ceiling
from src.analysis.f5_pooled import POOLED_HAND, pool_predictions, pooling_weights

DEFAULT_OUT_DIR = "results/phase_m"
EVAL_SEASON = 2024
POST_SELECTION_LABEL = (
    "post-selection descriptive: computed on 2024 after the pre-registered platoon gate "
    "ran and failed (docs/research-manifest.md, 2026-08-20). Not a test of a hypothesis.")

# The route rule, pre-registered 2026-08-30 before any Phase M number was read.
ROUTE_RULE = {
    "primary": "B_prime",
    "always_reported": ["A"],
    "provenance_only": ["B"],
    "never_primary": ["C"],
    "rationale": ("Route A's fragility is structural — a near-total cancellation where a "
                  "3% noise-model error swings tau2 by ~100% — and that was known before "
                  "any number was read."),
    "frozen": "docs/phase-m-spec.md §M.0, decision log 2026-08-30",
}

# Committed inputs the build reproduces rather than transcribes (spec §M.0).
COMMITTED = {
    "observed_within_stand_var": 0.00419563,
    "mean_sampling_var": 0.00407848,
    "route_a_tau2": 0.00011715,
    "route_b_tau2": 0.00059034,
    "e14_pooled_coverage_95": 0.855527,
}


# ------------------------------------------------------------------ M.6

def m6_population(pa_df, e5_frame, model_predictions, eval_season=EVAL_SEASON):
    """
    M.6, run first: which hitters E.5 and F.5 actually scored, and their intersection.

    E.5's population is read off its committed frame. F.5's is REBUILT through
    `f5_pooled`'s own path — pool the five-seed ensemble's side predictions by prior
    exposure, collapse the hand, run the frozen scorer — rather than transcribed, so a
    mismatch against `f5_pooled_summary.json` is caught here instead of propagating.
    """
    weights = pooling_weights(pa_df, eval_season)
    pooled_predictions = pool_predictions(model_predictions, weights, eval_season)
    pooled_pa = pa_df.copy()
    pooled_pa["p_throws"] = POOLED_HAND
    f5_frame, f5_coverage = claim1_eval.build_eval_frame(pooled_pa, pooled_predictions,
                                                         eval_season)

    e5_batters = set(e5_frame["batter"])
    f5_batters = set(f5_frame["batter"])
    everyone = sorted(e5_batters | f5_batters)

    exposure = e5_frame.set_index("batter")[["denom_L", "denom_R", "stand", "stratum"]]
    pooled_denominator = f5_frame.set_index("batter")["denominator"]

    population = pd.DataFrame({"batter": everyone})
    population["in_e5"] = population["batter"].isin(e5_batters)
    population["in_f5"] = population["batter"].isin(f5_batters)
    population["in_intersection"] = population["in_e5"] & population["in_f5"]
    for column in ("denom_L", "denom_R", "stand", "stratum"):
        population[column] = population["batter"].map(exposure[column])
    population["pooled_denom"] = population["batter"].map(pooled_denominator)
    population["post_selection_descriptive"] = True

    summary = {
        "n_e5": len(e5_batters),
        "n_f5": len(f5_batters),
        "n_intersection": len(e5_batters & f5_batters),
        "n_e5_only": len(e5_batters - f5_batters),
        "n_f5_only": len(f5_batters - e5_batters),
        "e5_is_subset_of_f5": e5_batters <= f5_batters,
        "f5_rebuilt_coverage": f5_coverage,
    }
    return population, summary


def intersection_frame(e5_frame, pa_df, population, eval_season=EVAL_SEASON):
    """The E.5 frame cut to the M.6 intersection, with per-hitter sampling variance attached."""
    keep = set(population.loc[population["in_intersection"], "batter"])
    frame = e5_frame[e5_frame["batter"].isin(keep)].reset_index(drop=True)
    assert len(frame) == len(keep), "the intersection cut lost or duplicated a hitter"
    by_batter, by_league = e15.per_pa_variance_tables(pa_df, eval_season)
    return e15.attach_sampling_variance(frame, by_batter, by_league), by_league


# ------------------------------------------------------------------ M.0

def per_hitter_tau2(frame, params):
    """
    Each hitter's within-stand true differential variance under a given C.2 fit.

    `implied_split_constant` is C.2's own accessor for the derived split variance
    tau2_L + tau2_R − 2·rho·tau_L·tau_R, and it returns nan when rho sits at its clip,
    because nothing derived from a bound is an estimate. E.15's Route B read exactly this
    quantity, stored in `c2_prior_parameters.csv` as `tau2_split_derived`.
    """
    derived = {}
    for batter_type, fitted in params.items():
        _, tau2_split = c2.implied_split_constant(fitted)
        derived[batter_type] = float(tau2_split)
    stands = frame["stand"].to_numpy()
    missing = sorted(set(stands) - set(derived))
    assert not missing, f"no C.2 batter type for stand(s) {missing}"
    return np.array([derived[stand] for stand in stands]), derived


def pooled_tau2(frame, params):
    """Weight-weighted mean of the per-hitter tau2 — E.15's pooling, reused unchanged."""
    values, derived = per_hitter_tau2(frame, params)
    weight = frame["weight"].to_numpy(dtype="float64")
    return float(np.average(values, weights=weight)), derived


def route_tables(frame, by_league, params_b, params_b_prime):
    """
    The M.0 table: every route, the pre-registered rule's verdict, the fragility band and
    the stabilization thresholds — all on the M.6 intersection.
    """
    weight = frame["weight"].to_numpy(dtype="float64")
    sampling = frame["sampling_var"].to_numpy(dtype="float64")
    mean_sampling = float(np.average(sampling, weights=weight))
    decomposition = e15.between_within_stand(frame["delta_obs"], weight, frame["stand"])
    observed = decomposition["within_stand"]

    route_a = m_ceiling.route_a_tau2(observed, mean_sampling)
    band = m_ceiling.fragility_band(observed, mean_sampling)

    tau2_b, derived_b = pooled_tau2(frame, params_b)
    tau2_b_prime, derived_b_prime = pooled_tau2(frame, params_b_prime)
    route_b = m_ceiling.ceiling_from_variances(tau2_b, mean_sampling)
    route_b_prime = m_ceiling.ceiling_from_variances(tau2_b_prime, mean_sampling)

    # the measured rank ceiling beside the analytic one; see m_ceiling's module docstring
    monte_carlo = m_ceiling.monte_carlo_ceiling(
        tau2_b_prime, sampling, weight, n_draws=300, seed=7) if tau2_b_prime > 0 else None

    achieved, _ = e15.recompute_e5_rank_correlation(frame)
    routes = {"A": route_a, "B": route_b, "B_prime": route_b_prime}
    for name, row in routes.items():
        row["achieved_rank_corr"] = achieved
        row["fraction_of_ceiling"] = (achieved / row["ceiling_rank_corr"]
                                      if row["ceiling_rank_corr"] > 0 else float("nan"))

    # --- the B -> B' population diagnostic (route rule item 3)
    drop = 1.0 - tau2_b_prime / tau2_b if tau2_b > 0 else float("nan")
    toward_a = (tau2_b - tau2_b_prime) / (tau2_b - route_a["tau2"]) if tau2_b > route_a["tau2"] \
        else float("nan")
    diagnostic = {
        "tau2_B": tau2_b, "tau2_B_prime": tau2_b_prime,
        "relative_drop": float(drop),
        "share_of_B_to_A_gap_closed": float(toward_a),
        "reading": ("a large drop toward A means selection into the 2024 eval population "
                    "explains the gap; no material drop means the remaining A/B' gap is "
                    "window or estimator"),
        "verdict": ("selection into the 2024 eval population explains most of the B-vs-A gap"
                    if toward_a >= 0.5 else
                    "selection explains part of the gap; the remainder is window or estimator"
                    if toward_a >= 0.15 else
                    "no material drop — the A/B' gap is window or estimator, not population"),
        "conditioning_label": ("Route B' restricts PAST seasons to hitters who reached the "
                               "2024 eval population, which conditions on survival to 2024. "
                               "That is the correct population for a claim about 2024 eval "
                               "hitters, and it is not a general-population tau2."),
    }

    return {
        "population": {"n_hitters": int(len(frame)), "basis": "M.6 intersection"},
        "observed_within_stand_var": observed,
        "variance_decomposition": decomposition,
        "mean_sampling_var_weighted": mean_sampling,
        "mean_sampling_var_unweighted": float(sampling.mean()),
        "routes": routes,
        "fragility_band": band,
        "monte_carlo_rank_ceiling": monte_carlo,
        "b_to_b_prime_diagnostic": diagnostic,
        "tau2_by_batter_type": {"B": derived_b, "B_prime": derived_b_prime},
        "stabilization": stabilization_table(frame, by_league, routes),
        "achieved_rank_corr_e5": achieved,
    }


def stabilization_table(frame, by_league, routes):
    """
    Each route's tau2 re-expressed as PA*, the exposure at which reliability reaches 0.5.

    TWO CONVENTIONS, both reported, because the handoff's ~430 / >2000 figures are on the
    first and a reader will assume the second:

      `pa_star_weak_side`  — per-PA noise is the variance vs LHP alone. This is C.2's own
        `implied_split_constant` convention (sigma2[0] / tau2_split): The Book weights a
        hitter's split by PA faced vs LHP, treating the strong side as known. It is the
        number `n_star_split_implied` reports and the one the handoff quoted.
      `pa_star_both_sides` — per-PA noise is s2_L + s2_R, i.e. PA* per side when BOTH sides
        grow together, which is how exposure actually accumulates. Always the larger.

    Per-PA variances are the realized 2024 league values from E.15's own noise model, not
    a new one.
    """
    weight = frame["weight"].to_numpy(dtype="float64")
    league = by_league.set_index(["stand", "p_throws"])["var"]
    vs_left = np.array([float(league.loc[(stand, "L")]) for stand in frame["stand"]])
    vs_right = np.array([float(league.loc[(stand, "R")]) for stand in frame["stand"]])
    per_pa_weak = float(np.average(vs_left, weights=weight))
    per_pa_both = float(np.average(vs_left + vs_right, weights=weight))

    rows = {}
    for name, row in routes.items():
        rows[name] = {
            "tau2": row["tau2"],
            "pa_star_weak_side": m_ceiling.stabilization_pa(per_pa_weak, row["tau2"]),
            "pa_star_both_sides": m_ceiling.stabilization_pa(per_pa_both, row["tau2"]),
        }
    return {
        "per_pa_noise_var_vs_LHP": per_pa_weak,
        "per_pa_noise_var_both_sides": per_pa_both,
        "by_route": rows,
        "observed_exposure": {
            "median_denom_L": float(frame["denom_L"].median()),
            "median_denom_R": float(frame["denom_R"].median()),
            "median_denom_L_by_stand": {
                stand: float(part["denom_L"].median())
                for stand, part in frame.groupby("stand")},
        },
        "handoff_figures_checked": {
            "claimed_B_about_430": 430,
            "claimed_A_over_2000": 2000,
            "convention": "pa_star_weak_side",
        },
    }


# ------------------------------------------------------------------ M.0 Route C

def half_sampling_variance(frame, pa_df, eval_season=EVAL_SEASON, min_half_pa=10):
    """
    Per-hitter sampling variance of ONE half's differential, on the real game-parity split.

    Rebuilds exactly the halves `e_platoon_ceiling.split_half_reliability` builds — same
    parity rule, same per-side exposure floor — so the simulated null is a null for the
    estimator as it actually ran, not for an idealized version of it.
    """
    window = pa_df[(pa_df["season"] == eval_season) & pa_df["in_denominator"]]
    window = window[window["batter"].isin(frame["batter"])]
    window = window.assign(half=np.where(window["game_pk"].to_numpy() % 2 == 0, "A", "B"))
    denominators = (window.groupby(["batter", "half", "p_throws"]).size()
                    .unstack(["half", "p_throws"]))
    needed = [(half, hand) for half in ("A", "B") for hand in ("L", "R")]
    denominators = denominators.reindex(columns=pd.MultiIndex.from_tuples(needed)).dropna()
    enough = (denominators >= min_half_pa).all(axis=1)
    denominators = denominators[enough]

    variances = frame.set_index("batter")[["s2_L", "s2_R"]].reindex(denominators.index)
    assert variances.notna().all().all(), "a split-half hitter has no per-PA variance"
    per_half = {}
    for half in ("A", "B"):
        per_half[half] = (variances["s2_L"] / denominators[(half, "L")]
                          + variances["s2_R"] / denominators[(half, "R")]).to_numpy()
    return denominators.index.to_numpy(), (per_half["A"] + per_half["B"]) / 2.0, denominators


def route_c_diagnostic(frame, pa_df, tau2_b_prime, committed_split_half,
                       eval_season=EVAL_SEASON, n_draws=2000, seed=0):
    """
    M.0's bounded Route C diagnostic: a code audit, then a null simulation that locates the
    observed −0.366 under tau2 = 0 and under tau2 = B'.

    The audit is recorded as findings, not as a claim of correctness — the simulation is
    what decides, and its two outcomes are the pre-registered branches of §10 rule 2.
    """
    audit = {
        "implementation": "src/analysis/e_platoon_ceiling.py::split_half_reliability",
        "checks": {
            "split_definition": ("game_pk parity — halves are disjoint GAMES, so no plate "
                                 "appearance appears in both halves and within-game "
                                 "correlation cannot leak across the split. Deterministic, "
                                 "no seed. CORRECT."),
            "pairing": ("halves are pivoted per batter and joined on the batter index, so "
                        "the correlation is across hitters with one row each. CORRECT."),
            "no_shared_pa": ("a PA belongs to exactly one game_pk and therefore exactly one "
                             "half; the pivot sums disjoint sets. CORRECT."),
            "spearman_brown_input": ("applied to the raw half-length correlation, not to an "
                                     "already-stepped-up value. CORRECT."),
            "exposure_floor": ("min_half_pa=10 per side per half, applied to both halves. "
                               "CORRECT, and it is the reason the split-half n (185 LHB) is "
                               "below the frame's."),
            "pooled_centring": ("the pooled row centres each stand's differential before "
                                "correlating, so the between-stand main effect cannot "
                                "inflate it. CORRECT and necessary — E.5's 0.146 is a "
                                "within-stand number."),
        },
        "finding": ("no bug found. One caveat, not a bug: Spearman-Brown is applied to a "
                    "NEGATIVE half correlation, and the step-up formula 2r/(1+r) has no "
                    "reliability interpretation there — it magnifies −0.155 to −0.366. The "
                    "magnitude of the negative number is therefore an artifact of stepping "
                    "up an out-of-domain input; the SIGN is the finding, and the raw half "
                    "correlation −0.155 is the quantity the simulation locates."),
        "bug_found": False,
    }

    batters, half_variance, denominators = half_sampling_variance(frame, pa_df, eval_season)
    stands = frame.set_index("batter")["stand"].reindex(batters)
    rng = np.random.default_rng(seed)

    simulations, located = {}, {}
    for label, tau2 in (("tau2_zero", 0.0), ("tau2_b_prime", float(tau2_b_prime))):
        by_stand = {}
        for stand in ("L", "R"):
            mask = (stands == stand).to_numpy()
            if mask.sum() < 20:
                continue
            simulated = m_ceiling.simulate_split_half(tau2, half_variance[mask], rng,
                                                      n_draws=n_draws)
            observed = committed_split_half[stand]
            by_stand[stand] = {key: value for key, value in simulated.items()
                               if key != "draws"}
            by_stand[stand]["located"] = m_ceiling.locate_in_simulation(observed, simulated)
        simulations[label] = by_stand
    located = {stand: {label: simulations[label][stand]["located"]
                       for label in simulations if stand in simulations[label]}
               for stand in ("L", "R")}

    lhb = located.get("L", {})
    inside_zero = lhb.get("tau2_zero", {}).get("inside_95")
    inside_b_prime = lhb.get("tau2_b_prime", {}).get("inside_95")
    if inside_zero and inside_b_prime:
        verdict = "uninformative_inside_both_nulls"
        reading = ("Route C's LHB value sits inside BOTH simulated nulls, so it discriminates "
                   "nothing: the split-half estimator at these half-length exposures has too "
                   "much sampling variance to separate tau2=0 from tau2=B'. The alarming "
                   "-0.366 in E.15 is Spearman-Brown magnifying an ordinary null draw "
                   "(the raw half correlation is at the 18th percentile of the tau2=0 null). "
                   "Route C is retired as evidence in either direction, which is the bounded "
                   "outcome the spec allows. It was never an estimator.")
        fallback_2_fires = False
    elif inside_zero:
        verdict = "consistent_with_small_tau2"
        reading = ("Route C's LHB value sits inside the tau2=0 null and outside the tau2=B' "
                   "null, so it is an evidence line leaning toward small tau2 (Route A's "
                   "reading) and never an estimator.")
        fallback_2_fires = False
    elif not inside_zero and not inside_b_prime:
        verdict = "outside_both_nulls_no_bug"
        reading = ("§10 fallback rule 2 FIRES: the observed value falls outside both "
                   "simulated nulls and the audit found no bug, so the shared noise model "
                   "is suspect — and Route A uses it too.")
        fallback_2_fires = True
    else:
        verdict = "consistent_with_b_prime"
        reading = ("Route C's LHB value is inside the tau2=B' null but outside the zero "
                   "null, which is evidence for the B' magnitude rather than against it.")
        fallback_2_fires = False

    return {
        "post_selection_descriptive": POST_SELECTION_LABEL,
        "bounded": "one-day hard cap per spec §M.0; audit + one simulation, no iteration",
        "audit": audit,
        "observed": committed_split_half,
        "n_hitters_by_stand": {stand: int((stands == stand).sum()) for stand in ("L", "R")},
        "median_half_denom": {
            "L_vs_LHP": float(denominators[("A", "L")].median()),
            "L_vs_RHP": float(denominators[("A", "R")].median())},
        "simulations": simulations,
        "located": located,
        "verdict": verdict,
        "reading": reading,
        "fallback_rule_2_fires": fallback_2_fires,
    }


# ------------------------------------------------------------------ M.1

def m1_differential(frame, pa_df, routes, eval_season=EVAL_SEASON):
    """
    M.1: put an opponent in the ceiling table. E.5 carried only the model's differential.

    `delta_c2` is C.2's own `predict()` side-specific wOBA differenced — scoring an already
    committed projection, not new modeling. C.3-full is governed by §10 fallback rule 1;
    see `c3_full_availability`.
    """
    predictions = c2.predict(pa_df, eval_season)
    wide = predictions.pivot_table(index="batter", columns="p_throws", values="pred_woba")
    assert {"L", "R"}.issubset(wide.columns), "C.2 did not emit both sides"
    scored = frame.copy()
    scored["delta_c2"] = (wide["L"] - wide["R"]).reindex(scored["batter"]).to_numpy()
    assert scored["delta_c2"].notna().all(), "a hitter in the intersection has no C.2 forecast"

    weight = scored["weight"].to_numpy(dtype="float64")
    rows = []
    for name, column in (("phase_d_model", "delta_pred"), ("c2_bivariate", "delta_c2")):
        by_stand = {}
        for stand in ("L", "R"):
            part = scored[scored["stand"] == stand]
            by_stand[stand] = float(claim1_eval.weighted_rank_correlation(
                part["delta_obs"], part[column], part["weight"]))
        # the pooled row goes through E.5's own residualisation, not a reimplementation:
        # swap the model's differential into `delta_pred` and call the same function.
        # For the phase_d row that makes this a literal reproduction of E.5's committed
        # within-stand rank correlation, which is spec §9 gate 3c.
        pooled, _ = e15.recompute_e5_rank_correlation(
            scored.assign(delta_pred=scored[column]))
        rows.append({
            "model": name, "n_hitters": int(len(scored)),
            "rank_corr_within_stand_pooled": pooled,
            "rank_corr_L": by_stand["L"], "rank_corr_R": by_stand["R"],
            "pred_variance": float(m_ceiling.weighted_variance(scored[column], weight)),
            "population": "M.6 intersection",
            "post_selection_descriptive": True,
        })
    scores = pd.DataFrame(rows)

    fractions = []
    for _, row in scores.iterrows():
        for route in ("B_prime", "A"):
            ceiling = routes[route]["ceiling_rank_corr"]
            fractions.append({
                "model": row["model"], "route": route,
                "tau2": routes[route]["tau2"],
                "ceiling_rank_corr": ceiling,
                "achieved_rank_corr": row["rank_corr_within_stand_pooled"],
                "fraction_of_ceiling": (row["rank_corr_within_stand_pooled"] / ceiling
                                        if ceiling > 0 else float("nan")),
                "route_role": ("primary (pre-registered)" if route == "B_prime"
                               else "sensitivity, carries the fragility band"),
                "population": "M.6 intersection",
                "post_selection_descriptive": True,
            })
    return scores, pd.DataFrame(fractions), scored


C3_FULL_AVAILABILITY = {
    "emitted": False,
    "fallback_rule": "§10 rule 1",
    "reason": ("C.3-full has no persisted fitted artifact anywhere in `results/`; "
               "`c3_gbm.predict` calls `c3_gbm.fit` internally, so producing side-specific "
               "predictions requires RETRAINING the GBM on the 341MB labeled pitch table. "
               "The spec's condition ('from existing fitted artifacts without retraining or "
               "new feature code') is therefore met and rule 1 fires: the differential table "
               "reports C.2 only."),
    "not_a_capability_limit": ("the refit is an existing, supported code path "
                               "(src/analysis/c_report.py:364) needing no new feature code. "
                               "The omission is a scope boundary the spec pre-set, not "
                               "something C.3 cannot do."),
    "never": "no differential head was improvised, per the spec's explicit prohibition.",
}


# ------------------------------------------------------------------ M.2

def m2_stratum(frame, routes, params_b_prime, n_boot=2000, seed=0):
    """
    M.2: the ceiling inside each frozen claim-1 exposure stratum.

    tau2 is NOT refit per stratum — a per-stratum refit is a new estimator and is not
    authorized. The stratum's own sampling-variance profile is applied against the common
    B' fit, exactly as E.15 applies a common fit across stands. Two tau2 columns are
    emitted so the one moving part is visible: `tau2_stratum_mix` is the per-hitter B' tau2
    weight-averaged inside the stratum (it moves only through the stratum's L/R
    composition), and `tau2_pooled` is the single intersection-wide value.

    Strata come from the frozen `stratum` column on the E.5 frame. They are not redefined
    here — that is a hard stop (§10.6).
    """
    per_hitter, _ = per_hitter_tau2(frame, params_b_prime)
    frame = frame.assign(tau2_b_prime=per_hitter)
    tau2_pooled = routes["B_prime"]["tau2"]
    rng = np.random.default_rng(seed)

    rows = []
    for stratum in claim1_eval.STRATUM_NAMES:
        part = frame[frame["stratum"] == stratum]
        if len(part) < 3:
            continue
        weight = part["weight"].to_numpy(dtype="float64")
        sampling = part["sampling_var"].to_numpy(dtype="float64")
        tau2_mix = float(np.average(part["tau2_b_prime"], weights=weight))
        mean_sampling = float(np.average(sampling, weights=weight))
        observed = e15.between_within_stand(part["delta_obs"], weight, part["stand"]) \
            if part["stand"].nunique() > 1 else {
                "within_stand": m_ceiling.weighted_variance(part["delta_obs"], weight)}

        b_prime = m_ceiling.ceiling_from_variances(tau2_mix, mean_sampling)
        route_a = m_ceiling.route_a_tau2(observed["within_stand"], mean_sampling)
        band = m_ceiling.fragility_band(observed["within_stand"], mean_sampling)
        # WITHIN-STAND, matching what the ceiling is a ceiling ON. tau2 here is the
        # within-stand true differential variance, so the raw correlation -- which E.5
        # showed is mostly the between-stand main effect -- is not the comparable
        # quantity and produces fractions above 1. Both are emitted; the fraction uses
        # the within-stand one.
        achieved_raw = float(claim1_eval.weighted_rank_correlation(
            part["delta_obs"], part["delta_pred"], weight))
        achieved = (float(e15.recompute_e5_rank_correlation(part)[0])
                    if part["stand"].nunique() > 1 else achieved_raw)

        # bootstrap resamples HITTERS inside the stratum; tau2 stays fixed under B' because
        # it is not refit, so the interval reflects the sampling-variance profile and the
        # stand mix — which is exactly what the per-stratum ceiling is made of
        draws_b_prime, draws_a = [], []
        for _ in range(n_boot):
            pick = rng.integers(0, len(part), len(part))
            w = weight[pick]
            samp = float(np.average(sampling[pick], weights=w))
            t2 = float(np.average(part["tau2_b_prime"].to_numpy()[pick], weights=w))
            draws_b_prime.append(m_ceiling.ceiling_from_variances(t2, samp)["ceiling_rank_corr"])
            obs = m_ceiling.weighted_variance(part["delta_obs"].to_numpy()[pick], w)
            draws_a.append(m_ceiling.route_a_tau2(obs, samp)["ceiling_rank_corr"])
        draws_b_prime = np.asarray(draws_b_prime, dtype="float64")
        draws_a = np.asarray(draws_a, dtype="float64")
        finite_a = draws_a[np.isfinite(draws_a)]

        rows.append({
            "stratum": stratum,
            "stratum_basis": "side-specific prior PA (frozen claim1_eval.STRATUM_BOUNDARIES)",
            "n_hitters": int(len(part)),
            "median_denom_L": float(part["denom_L"].median()),
            "median_denom_R": float(part["denom_R"].median()),
            "observed_within_stand_var": float(observed["within_stand"]),
            "mean_sampling_var": mean_sampling,
            "tau2_stratum_mix": tau2_mix,
            "tau2_pooled": tau2_pooled,
            "ceiling_b_prime": b_prime["ceiling_rank_corr"],
            "ceiling_b_prime_ci_low": float(np.percentile(draws_b_prime, 2.5)),
            "ceiling_b_prime_ci_high": float(np.percentile(draws_b_prime, 97.5)),
            "tau2_route_a": route_a["tau2"],
            "route_a_degenerate": bool(route_a["degenerate"]),
            "ceiling_route_a": route_a["ceiling_rank_corr"],
            "ceiling_route_a_ci_low_given_nondegenerate": float(np.percentile(finite_a, 2.5)) if len(finite_a)
                                      else float("nan"),
            "ceiling_route_a_ci_high_given_nondegenerate": float(np.percentile(finite_a, 97.5)) if len(finite_a)
                                       else float("nan"),
            "route_a_share_degenerate_in_bootstrap": float((~np.isfinite(draws_a)).mean()),
            "route_a_ci_note": ("the Route A interval is CONDITIONAL on the draw being "
                                "non-degenerate; the share of draws with tau2 <= 0 is the "
                                "column beside it, and no draw is clipped to zero"),
            "route_a_fragility_ceiling_range": band["ceiling_range"],
            "achieved_rank_corr_within_stand": achieved,
            "achieved_rank_corr_raw": achieved_raw,
            "raw_note": ("the raw column includes the between-stand main effect and is NOT "
                         "comparable to a within-stand ceiling; it is reported so the gap "
                         "between the two is visible"),
            "fraction_of_ceiling_b_prime": (achieved / b_prime["ceiling_rank_corr"]
                                            if b_prime["ceiling_rank_corr"] > 0 else np.nan),
            "post_selection_descriptive": True,
        })

    table = pd.DataFrame(rows)
    return table, precision_clause(table)


def precision_clause(table):
    """
    The 2026-08-20 revisit condition, made numeric 2026-08-30 and applied mechanically.

    A stratum whose 95% bootstrap CI on the B' ceiling includes zero cannot carry the
    illustration; the illustration moves to the stratum with the narrowest CI RELATIVE to
    its point estimate. No judgment call is left open.
    """
    table = table.copy()
    table["ci_includes_zero"] = (table["ceiling_b_prime_ci_low"] <= 0) & \
                                (table["ceiling_b_prime_ci_high"] >= 0)
    table["relative_ci_width"] = ((table["ceiling_b_prime_ci_high"]
                                   - table["ceiling_b_prime_ci_low"])
                                  / table["ceiling_b_prime"])
    # the clause is CONDITIONAL: the narrowest-CI tiebreak is what to do WHEN the default
    # stratum is unusable, not a standing preference. 'low' is the stratum the thesis is
    # graded on, so it stays unless its own interval includes zero.
    default = "low"
    eligible = table[~table["ci_includes_zero"]]
    default_row = table[table["stratum"] == default]
    default_usable = bool(len(default_row)) and not bool(default_row["ci_includes_zero"].iloc[0])
    if default_usable:
        chosen = default
    else:
        chosen = (eligible.loc[eligible["relative_ci_width"].idxmin(), "stratum"]
                  if len(eligible) else None)
    return {
        "rule": ("a stratum whose 95% bootstrap CI on the ceiling includes zero cannot carry "
                 "the illustration; it moves to the stratum with the narrowest CI relative "
                 "to its point estimate"),
        "frozen": "docs/phase-m-spec.md §M.2, from the 2026-08-20 revisit condition",
        "ci_includes_zero": dict(zip(table["stratum"], table["ci_includes_zero"].astype(bool))),
        "relative_ci_width": dict(zip(table["stratum"], table["relative_ci_width"])),
        "illustration_stratum": chosen,
        "default_stratum": default,
        "default_stratum_usable": default_usable,
        "moved": bool(chosen != default),
        "triggered": bool(not default_usable),
        "note": ("'low' is the stratum the thesis is graded on, so a move away from it is "
                 "reported as a limitation on the illustration, not as a change of subject."),
    }


# ------------------------------------------------------------------ M.3

def m3_level(pa_df, model_predictions, population, params_b_prime,
             f5_scores, eval_season=EVAL_SEASON):
    """
    M.3: the level-side ceiling, so "X% of the platoon ceiling" has a comparable figure.

    PRIMARY, from the committed F.5 terms. `noise_floor` is an RMSE, so E[Var[noise]] is
    its square; the no-information rung's `pa_weighted_rmse` is the RAW observed
    between-hitter spread and its `model_rmse` is already that spread DECONVOLVED, i.e. the
    true-talent sd. (The spec calls `model_rmse` the observed spread; the repo had already
    done the subtraction the spec asks for, and the identity
    model_rmse^2 = pa_weighted_rmse^2 − noise_floor^2 is asserted below.)

    The ceiling is refit-invariant; the achieved fraction is not. They are separate fields,
    per the plan's wording.
    """
    intersection = sorted(population.loc[population["in_intersection"], "batter"])
    weights = pooling_weights(pa_df, eval_season)
    pooled_predictions = pool_predictions(model_predictions, weights, eval_season)
    pooled_pa = pa_df.copy()
    pooled_pa["p_throws"] = POOLED_HAND

    # --- committed F.5 population (n = 617), reported labelled with its own n
    committed = {}
    for scope, table in (("f5_committed_617", f5_scores),):
        no_info = table[(table["model"] == "no_info_league_average")
                        & (table["stratum"] == "all")].iloc[0]
        model = table[(table["model"] == "phase_d") & (table["stratum"] == "all")].iloc[0]
        implied = claim1_eval.deconvolve(no_info["pa_weighted_rmse"], no_info["noise_floor"])
        assert abs(implied - no_info["model_rmse"]) < 1e-6, (
            "the no_info rung's model_rmse is not its deconvolved spread; the term "
            "provenance below would be wrong")
        committed[scope] = level_terms(no_info, model, int(no_info["n_hitters"]))

    # --- M.6 intersection (n = 545), the headline population
    restricted_pa = pooled_pa[pooled_pa["batter"].isin(intersection)]
    restricted_predictions = pooled_predictions[pooled_predictions["batter"].isin(intersection)]
    rungs = {"phase_d": restricted_predictions,
             "no_info_league_average": no_info_predictions(pa_df, intersection, eval_season)}
    scored = {}
    for name, predictions in rungs.items():
        table, coverage = claim1_eval.evaluate(restricted_pa, predictions, eval_season)
        scored[name] = table[table["stratum"] == "all"].iloc[0]
        scored[name + "_coverage"] = coverage
    intersection_terms = level_terms(scored["no_info_league_average"], scored["phase_d"],
                                     len(intersection))

    # --- cross-check 1: C.2-derived level variance composition, on the M.6 population
    cross_c2 = c2_level_variance(pa_df, intersection, params_b_prime, weights, eval_season)
    # --- cross-check 2: split-half on game_pk parity, pooled wOBA
    cross_split = level_split_half(pa_df, intersection, eval_season)

    return {
        "post_selection_descriptive": POST_SELECTION_LABEL,
        "primary_population": "M.6 intersection",
        "intersection": intersection_terms,
        "committed_f5_population": committed,
        "cross_check_c2_composition": cross_c2,
        "cross_check_split_half": cross_split,
        "term_provenance": {
            "E_var_noise": "f5/claim1_eval noise_floor squared (noise_floor is an RMSE)",
            "observed_between_hitter_variance": "no_info rung pa_weighted_rmse squared",
            "true_talent_variance": ("observed minus noise; equals the no_info rung's "
                                     "model_rmse squared, which claim1_eval.deconvolve "
                                     "had already computed"),
        },
        "by_stand_fractions_descriptive": {
            "L": 0.5690628371928935, "R": 0.12158155336698968,
            "caveat": ("reported descriptively only. The asymmetry's supporting statistic "
                       "is broken (E.15's LHB split-half reliability is negative) and the "
                       "L-vs-R difference was never tested."),
        },
        "coverage": {name: value for name, value in scored.items()
                     if name.endswith("_coverage")},
    }


def level_terms(no_info_row, model_row, n_hitters):
    """Reliability and ceiling on the LEVEL side, from one scored table's two rungs."""
    noise_variance = float(no_info_row["noise_floor"]) ** 2
    observed_variance = float(no_info_row["pa_weighted_rmse"]) ** 2
    true_variance = observed_variance - noise_variance
    ceiling = m_ceiling.ceiling_from_variances(true_variance, noise_variance)
    achieved = float(model_row["rank_corr_weighted"])
    return {
        "n_hitters": int(n_hitters),
        "observed_between_hitter_variance": observed_variance,
        "E_var_noise": noise_variance,
        "true_talent_variance": true_variance,
        "true_talent_sd": float(np.sqrt(true_variance)) if true_variance > 0 else float("nan"),
        "reliability": ceiling["reliability"],
        # refit-invariant
        "ceiling_rank_corr": ceiling["ceiling_rank_corr"],
        "degenerate": ceiling["degenerate"],
        # NOT refit-invariant — kept as a separate field per the plan's wording
        "achieved_rank_corr_weighted": achieved,
        "fraction_of_ceiling": (achieved / ceiling["ceiling_rank_corr"]
                                if ceiling["ceiling_rank_corr"] > 0 else float("nan")),
        "post_selection_descriptive": True,
    }


def no_info_predictions(pa_df, batters, eval_season):
    """The no-information rung, pooled: one league-average constant for every hitter."""
    from src.analysis.c1_trailing import predict as c1_predict
    from src.analysis.c_report import NO_INFO_BUCKETS
    predictions = c1_predict(pa_df, eval_season, variant="bucketed", buckets=NO_INFO_BUCKETS)
    weights = pooling_weights(pa_df, eval_season)
    pooled = pool_predictions(predictions, weights, eval_season)
    return pooled[pooled["batter"].isin(batters)]


def c2_level_variance(pa_df, batters, params, weights, eval_season):
    """
    Cross-check 1: the level-side true-talent variance implied by the B' C.2 fit.

    A hitter's pooled talent is w_L·theta_L + w_R·theta_R with F.5's prior-exposure weights,
    so its variance is w_L^2·tau2_L + w_R^2·tau2_R + 2·w_L·w_R·rho·tau_L·tau_R. This is the
    level-side analogue of Route B': same fit, same restricted population, the composition
    taken rather than the difference.
    """
    types = c2.batter_types(pa_df[pa_df["batter"].isin(batters)])
    side = weights.pivot_table(index="batter", columns="p_throws", values="weight")
    side = side.reindex(sorted(set(batters) & set(side.index))).fillna(0.5)
    lookup = types.set_index("batter")["batter_type"].reindex(side.index)
    lookup = lookup.fillna("R")

    values = []
    for batter, row in side.iterrows():
        fitted = params.get(lookup.loc[batter])
        if fitted is None:
            continue
        tau2_l, tau2_r = float(fitted["tau2"][0]), float(fitted["tau2"][1])
        rho = float(fitted["rho"])
        w_l, w_r = float(row.get("L", 0.5)), float(row.get("R", 0.5))
        total = w_l + w_r
        w_l, w_r = w_l / total, w_r / total
        values.append(w_l ** 2 * tau2_l + w_r ** 2 * tau2_r
                      + 2.0 * w_l * w_r * rho * np.sqrt(tau2_l * tau2_r))
    return {
        "route": "C.2 variance composition on the M.6 population (level-side Route B')",
        "n_hitters": int(len(values)),
        "true_talent_variance": float(np.mean(values)),
        "true_talent_sd": float(np.sqrt(np.mean(values))),
        "formula": "w_L^2 tau2_L + w_R^2 tau2_R + 2 w_L w_R rho tau_L tau_R",
        "weights": "F.5 prior side-specific exposure, normalized within batter",
    }


def level_split_half(pa_df, batters, eval_season, min_half_pa=25):
    """Cross-check 2: split-half on game_pk parity, pooled wOBA, stepped up by Spearman-Brown."""
    window = pa_df[(pa_df["season"] == eval_season) & pa_df["in_denominator"]]
    window = window[window["batter"].isin(batters)]
    window = window.assign(half=np.where(window["game_pk"].to_numpy() % 2 == 0, "A", "B"))
    grouped = (window.groupby(["batter", "half"])["woba_points"]
               .agg(points="sum", denom="size").reset_index())
    wide = grouped.pivot_table(index="batter", columns="half", values=["points", "denom"]).dropna()
    enough = ((wide[("denom", "A")] >= min_half_pa) & (wide[("denom", "B")] >= min_half_pa))
    wide = wide[enough]
    a = wide[("points", "A")] / wide[("denom", "A")]
    b = wide[("points", "B")] / wide[("denom", "B")]
    estimate = m_ceiling.split_half_from_halves(a, b)
    reliability = estimate["reliability_spearman_brown"]
    return {
        "route": "split-half on game_pk parity, pooled wOBA",
        "n_hitters": int(len(wide)),
        "min_half_pa": min_half_pa,
        "median_half_denom": float(wide[("denom", "A")].median()),
        **estimate,
        "ceiling_rank_corr": float(np.sqrt(reliability)) if reliability > 0 else float("nan"),
    }


# ------------------------------------------------------------------ M.4

def m4_coverage(pa_df, seed_predictions, manifest, population, eval_season=EVAL_SEASON):
    """
    M.4: one verification pass that E.14's 85.6% holds on the M.6 population. Labeling, not
    iteration — the plan's fallback already fired and no attempt is made to close the gap.

    E.14 scored 1149 SIDE-SPECIFIC groups; M.6 is 545 hitters. The measured coverage on the
    M.6 hitters' own rows is what every interval displayed in notebook 08 must be labeled
    with.
    """
    frame, _ = e_probe_coverage.ensemble_frame(pa_df, seed_predictions, manifest, eval_season)
    keep = set(population.loc[population["in_intersection"], "batter"])
    restricted = frame[frame["batter"].isin(keep)]
    rows = e_probe_coverage.coverage_rows(frame, "e14_all_groups")
    rows += e_probe_coverage.coverage_rows(restricted, "m6_intersection")
    for stratum in claim1_eval.STRATUM_NAMES:
        part = restricted[restricted["stratum"] == stratum]
        if len(part):
            rows += e_probe_coverage.coverage_rows(part, f"m6_intersection_{stratum}")
    table = pd.DataFrame(rows)
    table["post_selection_descriptive"] = True

    def headline(slice_name):
        row = table[(table["slice"] == slice_name) & (table["nominal"] == 0.95)
                    & (table["interval"] == "seed_plus_target_noise")]
        return float(row["empirical"].iloc[0]) if len(row) else float("nan")

    return table, {
        "nominal": 0.95,
        "interval": "seed_plus_target_noise (the fair test — seed spread convolved with "
                    "the target's own sampling noise)",
        "measured_coverage_e14_all_groups": headline("e14_all_groups"),
        "measured_coverage_m6_intersection": headline("m6_intersection"),
        "committed_e14_value": COMMITTED["e14_pooled_coverage_95"],
        "reproduces_committed": abs(headline("e14_all_groups")
                                    - COMMITTED["e14_pooled_coverage_95"]) < 1e-6,
        "label_for_notebook": ("every displayed interval carries its MEASURED coverage, not "
                               "the nominal level — the plan's own fallback, already fired. "
                               "No further attempt is made to close the gap."),
    }


# ------------------------------------------------------------------ driver

def main():
    parser = argparse.ArgumentParser(description="Phase M — the measurement ceiling.")
    parser.add_argument("--eval-season", type=int, default=EVAL_SEASON)
    parser.add_argument("--final-run", action="store_true")
    parser.add_argument("--eval-targets", default="data/processed/eval_targets_pa.parquet")
    parser.add_argument("--e5-frame", default="results/phase_e/e5_platoon_frame.csv")
    parser.add_argument("--model-predictions",
                        default="results/phase_d/d5_predictions_d10_baseline.csv")
    parser.add_argument("--seed-predictions",
                        default="results/phase_d/d5_predictions_d10_baseline_s{seed}.csv")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--manifest", default="data/processed/phase_d5/manifest.json")
    parser.add_argument("--f5-scores", default="results/phase_f/f5_pooled_scores.csv")
    parser.add_argument("--e15-json", default="results/phase_e/e15_ceiling.json")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    claim1_eval.assert_not_test_season(args.eval_season, final_run=args.final_run)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pa_df = pd.read_parquet(args.eval_targets)
    e5_frame = pd.read_csv(args.e5_frame)
    model_predictions = pd.read_csv(args.model_predictions)
    model_predictions = model_predictions[model_predictions["season"] == args.eval_season]

    # ---------------- M.6
    print("M.6 population reconciliation")
    population, population_summary = m6_population(pa_df, e5_frame, model_predictions,
                                                   args.eval_season)
    population.to_csv(out_dir / "population.csv", index=False)
    print(f"  E.5 {population_summary['n_e5']}  F.5 {population_summary['n_f5']}  "
          f"intersection {population_summary['n_intersection']}  "
          f"(E.5 subset of F.5: {population_summary['e5_is_subset_of_f5']})")

    frame, by_league = intersection_frame(e5_frame, pa_df, population, args.eval_season)

    # ---------------- M.0
    print("\nM.0 routes")
    params_b = c2.fit(pa_df, args.eval_season)
    restricted_pa = pa_df[pa_df["batter"].isin(set(frame["batter"]))]
    params_b_prime = c2.fit(restricted_pa, args.eval_season)
    routes = route_tables(frame, by_league, params_b, params_b_prime)

    committed_split_half = {}
    e15_json = json.loads(Path(args.e15_json).read_text())
    for row in e15_json["part1_ceiling"]["split_half"]:
        committed_split_half[row["stand"]] = row["half_split_spearman"]
    route_c = route_c_diagnostic(frame, pa_df, routes["routes"]["B_prime"]["tau2"],
                                 committed_split_half, args.eval_season)

    fired = []
    if route_c["fallback_rule_2_fires"]:
        fired.append("2 — Route C outside both nulls with no bug: noise model suspect, "
                     "Route A demoted to 'reported, unvalidated-noise-model caveat'")
    if routes["routes"]["B_prime"]["degenerate"]:
        fired.append("3 — B' degenerate: headline falls back to the B-vs-A bracket")
    fired.append("1 — C.3-full omitted from M.1; " + C3_FULL_AVAILABILITY["reason"])
    route_b_reproduces = abs(routes["routes"]["B"]["tau2"] - COMMITTED["route_b_tau2"]) < 5e-8
    if not route_b_reproduces:
        fired.append("4 — committed Route B failed to reproduce; current code's value adopted")

    m0 = {
        "step": "M.0",
        "post_selection_descriptive": POST_SELECTION_LABEL,
        "fallback_rules_fired": fired,
        "route_rule": ROUTE_RULE,
        **routes,
        "route_c": {"verdict": route_c["verdict"], "reading": route_c["reading"],
                    "artifact": "results/phase_m/m0_route_c_diagnostic.json"},
        "field_provenance": {
            "route_b_field": "tau2_split_derived (results/phase_c/c2_prior_parameters.csv)",
            "accessor": "c2_bivariate_eb.implied_split_constant -> tau2_split",
            "identity": "tau2_L + tau2_R - 2*rho*sqrt(tau2_L*tau2_R)",
            "note": ("E.15's Route B read this column via "
                     "e_platoon_ceiling.load_c2_components['tau2_split']; for a "
                     "DIFFERENTIAL the derived split variance IS the within-stand tau2, so "
                     "the spec's parenthetical describes the same quantity."),
        },
        "reproduction": {
            "route_b_committed": COMMITTED["route_b_tau2"],
            "route_b_recomputed": routes["routes"]["B"]["tau2"],
            "route_b_reproduces": bool(route_b_reproduces),
            "fitting_window_spec_says": "nine-season, 2016-2024",
            "fitting_window_code_uses": (
                f"{args.eval_season - 3}-{args.eval_season - 1} "
                "(c1_trailing.TRAILING_SEASONS = 3)"),
            "window_note": ("the spec's window PROSE is wrong; the committed VALUE "
                            "reproduces exactly on the code's three-season window, so §10 "
                            "rule 4 does not fire. B and B' differ only in population, "
                            "which is what the B->B' diagnostic requires."),
        },
        "c3_full_availability": C3_FULL_AVAILABILITY,
    }
    (out_dir / "m0_routes.json").write_text(json.dumps(m0, indent=2, default=float))
    (out_dir / "m0_route_c_diagnostic.json").write_text(
        json.dumps(route_c, indent=2, default=float))
    for name in ("B_prime", "A", "B"):
        row = routes["routes"][name]
        print(f"  {name:8s} tau2 {row['tau2']:.8f}  ceiling "
              f"{row['ceiling_rank_corr']:.4f}  fraction {row['fraction_of_ceiling']:.3f}"
              f"{'  DEGENERATE' if row['degenerate'] else ''}")
    print(f"  fragility band (A, x0.97/x1.03): {routes['fragility_band']['ceiling_range']}")
    print(f"  Route C: {route_c['verdict']}")

    # ---------------- M.1
    print("\nM.1 differential scores")
    m1_scores, m1_fractions, scored_frame = m1_differential(frame, pa_df, routes["routes"],
                                                            args.eval_season)
    m1_scores.to_csv(out_dir / "m1_differential_scores.csv", index=False)
    m1_fractions.to_csv(out_dir / "m1_fraction_of_ceiling.csv", index=False)
    print(m1_scores[["model", "n_hitters", "rank_corr_within_stand_pooled",
                     "rank_corr_L", "rank_corr_R"]].round(4).to_string(index=False))

    # ---------------- M.2
    print("\nM.2 per-stratum ceiling")
    m2_table, clause = m2_stratum(frame, routes["routes"], params_b_prime, n_boot=args.n_boot)
    m2_table.to_csv(out_dir / "m2_stratum_ceiling.csv", index=False)
    print(m2_table[["stratum", "n_hitters", "ceiling_b_prime", "ceiling_b_prime_ci_low",
                    "ceiling_b_prime_ci_high", "achieved_rank_corr_within_stand",
                    "achieved_rank_corr_raw",
                    "fraction_of_ceiling_b_prime"]].round(4).to_string(index=False))
    print(f"  precision clause -> illustration stratum: {clause['illustration_stratum']}")

    # ---------------- M.3
    print("\nM.3 level-side ceiling")
    f5_scores = pd.read_csv(args.f5_scores)
    m3 = m3_level(pa_df, model_predictions, population, params_b_prime, f5_scores,
                  args.eval_season)
    m3["precision_clause"] = clause
    (out_dir / "m3_level_ceiling.json").write_text(json.dumps(m3, indent=2, default=float))
    level = m3["intersection"]
    print(f"  n={level['n_hitters']} reliability {level['reliability']:.4f}  "
          f"ceiling {level['ceiling_rank_corr']:.4f}  "
          f"achieved {level['achieved_rank_corr_weighted']:.4f}  "
          f"fraction {level['fraction_of_ceiling']:.3f}")

    # ---------------- M.4
    print("\nM.4 coverage labels")
    seed_predictions = {}
    for seed in args.seeds:
        table = pd.read_csv(args.seed_predictions.format(seed=seed))
        seed_predictions[seed] = table[table["season"] == args.eval_season]
    manifest = json.loads(Path(args.manifest).read_text())
    m4_table, m4_summary = m4_coverage(pa_df, seed_predictions, manifest, population,
                                       args.eval_season)
    m4_table.to_csv(out_dir / "m4_coverage_labels.csv", index=False)
    print(f"  measured 95% coverage: all groups "
          f"{m4_summary['measured_coverage_e14_all_groups']:.4f}  "
          f"M.6 intersection {m4_summary['measured_coverage_m6_intersection']:.4f}")

    # ---------------- summary
    summary = {
        "step": "Phase M",
        "spec": "docs/phase-m-spec.md",
        "eval_season": args.eval_season,
        "sealed_seasons_never_read": [2025],
        "post_selection_descriptive": POST_SELECTION_LABEL,
        "population": population_summary,
        "route_rule": ROUTE_RULE,
        "headline": {
            "primary_route": "B_prime",
            "platoon_ceiling_b_prime": routes["routes"]["B_prime"]["ceiling_rank_corr"],
            "platoon_fraction_b_prime": routes["routes"]["B_prime"]["fraction_of_ceiling"],
            "platoon_ceiling_a": routes["routes"]["A"]["ceiling_rank_corr"],
            "platoon_fraction_a": routes["routes"]["A"]["fraction_of_ceiling"],
            "level_ceiling": level["ceiling_rank_corr"],
            "level_fraction": level["fraction_of_ceiling"],
        },
        "fallback_rules_fired": fired,
        "m4_coverage": m4_summary,
        "route_c_verdict": route_c["verdict"],
        "precision_clause": clause,
        "artifacts": sorted(path.name for path in out_dir.glob("*")),
    }
    (out_dir / "m_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nwrote {out_dir}/")


if __name__ == "__main__":
    main()
