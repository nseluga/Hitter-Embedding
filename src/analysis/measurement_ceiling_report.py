"""
Phase M driver — population, routes, differential, strata, level side, coverage.

Reads committed Phase C/D/E/F artifacts and writes `results/measurement_ceiling/`. The statistics
live in `measurement_ceiling_stats.py`; nothing is estimated here that is not either an existing repo
estimator (`eb_bivariate_eb.fit`, `model_evaluation_platoon_ceiling`'s subtraction and split-half,
`claim1_eval`'s scorer) or an aggregation of one.

EVERY NUMBER THIS FILE EMITS IS POST-SELECTION AND DESCRIPTIVE (spec §0.3). The
pre-registered gate on the platoon differential ran on 2024 and failed (research-manifest,
2026-08-20); everything after it describes a measurement problem rather than testing a
hypothesis. The label rides on every artifact as `post_selection_descriptive`.

2025 is never read. The only season this module scores is 2024, and `claim1_eval`'s test
guard is called before anything else runs.

Run:  PYTHONPATH=. .venv/bin/python -m src.analysis.measurement_ceiling_report
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import baseline_ladder_bivariate_eb as eb
from src.analysis import claim1_eval
from src.analysis import model_evaluation_platoon_ceiling as platoon_ceiling
from src.analysis import model_evaluation_probe_coverage
from src.analysis import measurement_ceiling_stats
from src.analysis.process_calibration_pooled import POOLED_HAND, pool_predictions, pooling_weights

DEFAULT_OUT_DIR = "results/measurement_ceiling"
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
    "coverage_pooled_coverage_95": 0.855527,
}


# ------------------------------------------------------------------ M.6

def m6_population(pa_df, platoon_frame, model_predictions, eval_season=EVAL_SEASON):
    """
    M.6, run first: which hitters E.5 and F.5 actually scored, and their intersection.

    E.5's population is read off its committed frame. F.5's is REBUILT through
    `pooled`'s own path — pool the five-seed ensemble's side predictions by prior
    exposure, collapse the hand, run the frozen scorer — rather than transcribed, so a
    mismatch against `pooled_summary.json` is caught here instead of propagating.
    """
    weights = pooling_weights(pa_df, eval_season)
    pooled_predictions = pool_predictions(model_predictions, weights, eval_season)
    pooled_pa = pa_df.copy()
    pooled_pa["p_throws"] = POOLED_HAND
    pooled_frame, pooled_coverage = claim1_eval.build_eval_frame(pooled_pa, pooled_predictions,
                                                         eval_season)

    platoon_batters = set(platoon_frame["batter"])
    pooled_batters = set(pooled_frame["batter"])
    everyone = sorted(platoon_batters | pooled_batters)

    exposure = platoon_frame.set_index("batter")[["denom_L", "denom_R", "stand", "stratum"]]
    pooled_denominator = pooled_frame.set_index("batter")["denominator"]

    population = pd.DataFrame({"batter": everyone})
    population["in_platoon"] = population["batter"].isin(platoon_batters)
    population["in_pooled"] = population["batter"].isin(pooled_batters)
    population["in_intersection"] = population["in_platoon"] & population["in_pooled"]
    for column in ("denom_L", "denom_R", "stand", "stratum"):
        population[column] = population["batter"].map(exposure[column])
    population["pooled_denom"] = population["batter"].map(pooled_denominator)
    population["post_selection_descriptive"] = True

    summary = {
        "n_platoon": len(platoon_batters),
        "n_pooled": len(pooled_batters),
        "n_intersection": len(platoon_batters & pooled_batters),
        "n_platoon_only": len(platoon_batters - pooled_batters),
        "n_pooled_only": len(pooled_batters - platoon_batters),
        "platoon_is_subset_of_pooled": platoon_batters <= pooled_batters,
        "pooled_rebuilt_coverage": pooled_coverage,
    }
    return population, summary


def intersection_frame(platoon_frame, pa_df, population, eval_season=EVAL_SEASON):
    """The E.5 frame cut to the M.6 intersection, with per-hitter sampling variance attached."""
    keep = set(population.loc[population["in_intersection"], "batter"])
    frame = platoon_frame[platoon_frame["batter"].isin(keep)].reset_index(drop=True)
    assert len(frame) == len(keep), "the intersection cut lost or duplicated a hitter"
    by_batter, by_league = platoon_ceiling.per_pa_variance_tables(pa_df, eval_season)
    return platoon_ceiling.attach_sampling_variance(frame, by_batter, by_league), by_league


# ------------------------------------------------------------------ M.0

def per_hitter_tau2(frame, params):
    """
    Each hitter's within-stand true differential variance under a given C.2 fit.

    `implied_split_constant` is C.2's own accessor for the derived split variance
    tau2_L + tau2_R − 2·rho·tau_L·tau_R, and it returns nan when rho sits at its clip,
    because nothing derived from a bound is an estimate. E.15's Route B read exactly this
    quantity, stored in `eb_prior_parameters.csv` as `tau2_split_derived`.
    """
    derived = {}
    for batter_type, fitted in params.items():
        _, tau2_split = eb.implied_split_constant(fitted)
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
    decomposition = platoon_ceiling.between_within_stand(frame["delta_obs"], weight, frame["stand"])
    observed = decomposition["within_stand"]

    route_a = measurement_ceiling_stats.route_a_tau2(observed, mean_sampling)
    band = measurement_ceiling_stats.fragility_band(observed, mean_sampling)

    tau2_b, derived_b = pooled_tau2(frame, params_b)
    tau2_b_prime, derived_b_prime = pooled_tau2(frame, params_b_prime)
    route_b = measurement_ceiling_stats.ceiling_from_variances(tau2_b, mean_sampling)
    route_b_prime = measurement_ceiling_stats.ceiling_from_variances(tau2_b_prime, mean_sampling)

    # the measured rank ceiling beside the analytic one; see measurement_ceiling's module docstring
    monte_carlo = measurement_ceiling_stats.monte_carlo_ceiling(
        tau2_b_prime, sampling, weight, n_draws=300, seed=7) if tau2_b_prime > 0 else None

    achieved, _ = platoon_ceiling.recompute_platoon_rank_correlation(frame)
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
        "achieved_rank_corr_platoon": achieved,
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
            "pa_star_weak_side": measurement_ceiling_stats.stabilization_pa(per_pa_weak, row["tau2"]),
            "pa_star_both_sides": measurement_ceiling_stats.stabilization_pa(per_pa_both, row["tau2"]),
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

    Rebuilds exactly the halves `model_evaluation_platoon_ceiling.split_half_reliability` builds — same
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
        "implementation": "src/analysis/model_evaluation_platoon_ceiling.py::split_half_reliability",
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
            simulated = measurement_ceiling_stats.simulate_split_half(tau2, half_variance[mask], rng,
                                                      n_draws=n_draws)
            observed = committed_split_half[stand]
            by_stand[stand] = {key: value for key, value in simulated.items()
                               if key != "draws"}
            by_stand[stand]["located"] = measurement_ceiling_stats.locate_in_simulation(observed, simulated)
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

# The differential columns scored in M.1 and M.2, in report order. `delta_pred` is the
# model_v1 model's, already on the E.5 frame; the other two are attached from committed
# side-specific projections and differenced. Every one of them is SCORING an existing
# forecast — no differential head is fitted anywhere in Phase M.
DIFFERENTIAL_MODELS = (
    ("model_v1_model", "delta_pred"),
    ("eb_bivariate", "delta_eb"),
    ("gbm_full", "delta_c3full"),
)
REFERENCE_MODEL = "model_v1_model"


def gbm_full_differential(pa_df, cache_path, pitch_events, eval_season=EVAL_SEASON, seed=0):
    """
    C.3-full's side-specific projections, from cache or from the supported refit path.

    Pass A1 discharges §10 fallback rule 1. The rule fired at Phase M because
    `gbm.predict` fits inside the call and no fitted C.3 artifact was persisted, so a
    differential needed a retrain. The retrain is the loop at `baseline_ladder_report.py:364` and takes
    seconds, so this function runs it once and PERSISTS the predictions — after which the
    condition rule 1 tested ("no side-specific predictions without retraining") is false
    for every later pass.

    Returns a batter-indexed Series of predicted wOBA(vs L) − wOBA(vs R).
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        predictions = pd.read_csv(cache_path)
    else:
        # imported here: gbm pulls lightgbm, and every other Phase M path runs without it
        from src.analysis import baseline_ladder_gbm as gbm
        from src.analysis.baseline_ladder_report import PITCH_COLUMNS
        process_seasons = gbm.season_process(
            pd.read_parquet(pitch_events, columns=PITCH_COLUMNS))
        predictions = gbm.predict(pa_df, process_seasons, eval_season,
                                 feature_set="full", seed=seed)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(cache_path, index=False)
    predictions = predictions[predictions["season"] == eval_season]
    wide = predictions.pivot_table(index="batter", columns="p_throws", values="pred_woba")
    assert {"L", "R"}.issubset(wide.columns), "C.3-full did not emit both sides"
    # a hitter C.3-full forecast against only one hand has no differential at all; dropping
    # him here keeps the absence explicit, and `attach_differentials` asserts that nobody in
    # the M.1 population lands in that set
    return (wide["L"] - wide["R"]).dropna()


def eb_differential(pa_df, eval_season=EVAL_SEASON):
    """C.2's predicted wOBA(vs L) − wOBA(vs R), batter-indexed. Scoring a committed
    forecast, not fitting a differential head."""
    wide = eb.predict(pa_df, eval_season).pivot_table(
        index="batter", columns="p_throws", values="pred_woba")
    assert {"L", "R"}.issubset(wide.columns), "C.2 did not emit both sides"
    return wide["L"] - wide["R"]


def attach_differentials(frame, eb_delta, gbm_full_delta):
    """Put every opponent's differential on the M.1 frame, one column each."""
    scored = frame.copy()
    scored["delta_eb"] = eb_delta.reindex(scored["batter"]).to_numpy()
    scored["delta_c3full"] = gbm_full_delta.reindex(scored["batter"]).to_numpy()
    for _, column in DIFFERENTIAL_MODELS:
        assert scored[column].notna().all(), \
            f"a hitter in the intersection has no {column} forecast"
    return scored


def differential(scored, routes, n_boot=2000, seed=0):
    """
    M.1: put every Phase C opponent in the ceiling table, each achieved value with an
    interval. E.5 carried only the model's differential and no interval at all.

    Both gaps are the 2026-08-31 review's findings. A bare 49.3% fraction of ceiling was
    quoted against an A-to-B′ bracket narrower than the fraction's own sampling error, and
    a table with no opponent cannot answer the question frozen rule 1 asks.

    One shared bootstrap draw matrix serves the marginal intervals, the fractions (the
    ceiling is held FIXED — τ² is not refit per replicate, so the interval is the sampling
    error of the numerator alone, and the artifact says so) and the paired contrasts
    against `REFERENCE_MODEL`.
    """
    names = [name for name, _ in DIFFERENTIAL_MODELS]
    columns = [column for _, column in DIFFERENTIAL_MODELS]
    weight = scored["weight"].to_numpy(dtype="float64")
    draws, boot = measurement_ceiling_stats.paired_rank_bootstrap(scored, columns, n_boot=n_boot, seed=seed)
    boot = boot.set_index("column")

    rows = []
    for name, column in DIFFERENTIAL_MODELS:
        by_stand = {}
        for stand in ("L", "R"):
            part = scored[scored["stand"] == stand]
            by_stand[stand] = float(claim1_eval.weighted_rank_correlation(
                part["delta_obs"], part[column], part["weight"]))
        # the pooled row goes through E.5's own residualisation, not a reimplementation:
        # swap the model's differential into `delta_pred` and call the same function.
        # For the model_v1 row that makes this a literal reproduction of E.5's committed
        # within-stand rank correlation, which is spec §9 gate 3c.
        pooled, _ = platoon_ceiling.recompute_platoon_rank_correlation(
            scored.assign(delta_pred=scored[column]))
        assert abs(pooled - boot.loc[column, "point"]) < 1e-12, \
            f"{column}: the bootstrap's fast statistic disagrees with E.5's own"
        contrast = (None if name == REFERENCE_MODEL else measurement_ceiling_stats.paired_contrast(
            draws, columns, column, dict(DIFFERENTIAL_MODELS)[REFERENCE_MODEL],
            pooled, boot.loc[dict(DIFFERENTIAL_MODELS)[REFERENCE_MODEL], "point"]))
        rows.append({
            "model": name, "n_hitters": int(len(scored)),
            "rank_corr_within_stand_pooled": pooled,
            "rank_corr_ci_low": boot.loc[column, "ci_low"],
            "rank_corr_ci_high": boot.loc[column, "ci_high"],
            "rank_corr_L": by_stand["L"], "rank_corr_R": by_stand["R"],
            "pred_variance": float(measurement_ceiling_stats.weighted_variance(scored[column], weight)),
            "paired_diff_vs_reference": contrast["difference"] if contrast else 0.0,
            "paired_diff_ci_low": contrast["ci_low"] if contrast else np.nan,
            "paired_diff_ci_high": contrast["ci_high"] if contrast else np.nan,
            "paired_share_favouring_this_model": (contrast["favours_a_share"] if contrast
                                                  else np.nan),
            "reference_model": REFERENCE_MODEL,
            "n_boot": int(n_boot),
            "population": "M.6 intersection",
            "post_selection_descriptive": True,
        })
    scores = pd.DataFrame(rows)

    fractions = []
    for index, (name, column) in enumerate(DIFFERENTIAL_MODELS):
        achieved = scores.loc[index, "rank_corr_within_stand_pooled"]
        for route in ("B_prime", "A"):
            ceiling = routes[route]["ceiling_rank_corr"]
            usable = ceiling > 0
            fractions.append({
                "model": name, "route": route,
                "tau2": routes[route]["tau2"],
                "ceiling_rank_corr": ceiling,
                "achieved_rank_corr": achieved,
                "achieved_ci_low": scores.loc[index, "rank_corr_ci_low"],
                "achieved_ci_high": scores.loc[index, "rank_corr_ci_high"],
                "fraction_of_ceiling": achieved / ceiling if usable else float("nan"),
                "fraction_ci_low": (scores.loc[index, "rank_corr_ci_low"] / ceiling
                                    if usable else float("nan")),
                "fraction_ci_high": (scores.loc[index, "rank_corr_ci_high"] / ceiling
                                     if usable else float("nan")),
                "interval_note": ("hitter-level paired bootstrap on the NUMERATOR only; "
                                  "the ceiling is held fixed, so this understates total "
                                  "uncertainty by the width of the route disagreement"),
                "route_role": ("primary (pre-registered)" if route == "B_prime"
                               else "sensitivity, carries the fragility band"),
                "population": "M.6 intersection",
                "post_selection_descriptive": True,
            })
    return scores, pd.DataFrame(fractions), draws, columns


C3_FULL_AVAILABILITY = {
    "emitted": True,
    "fallback_rule": "§10 rule 1 — FIRED at Phase M (2026-08-30), DISCHARGED in Pass A1",
    "why_it_fired": ("C.3-full had no persisted fitted artifact anywhere in `results/`, and "
                     "`gbm.predict` calls `gbm.fit` internally, so side-specific "
                     "predictions required retraining the GBM on the 341MB labeled pitch "
                     "table — the condition rule 1 names."),
    "why_it_is_discharged": ("the retrain is the supported path at `src/analysis/baseline_ladder_report.py:364` "
                             "and costs 7 seconds, not the session it was scoped as. Pass A1 ran "
                             "it once and persisted the output to "
                             "`results/measurement_ceiling/differential_gbm_full_predictions.csv`, so the rule's "
                             "condition is false for every later pass."),
    "reproduction": ("the refit reproduces the committed Phase C `gbm_full` claim-1 row to "
                     "1.0e-9 on RMSE and 5.6e-17 on rank correlation in every stratum, so the "
                     "2026-08-31 revisit condition (absence becomes a capability limit if the "
                     "refit cannot reproduce Phase C) does not fire."),
    "predictions": "results/measurement_ceiling/differential_gbm_full_predictions.csv",
    "never": "no differential head was improvised, per the spec's explicit prohibition.",
}


# ------------------------------------------------------------------ M.2

def stratum_ceiling_stratum(frame, routes, params_b_prime, n_boot=2000, seed=0):
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
        observed = platoon_ceiling.between_within_stand(part["delta_obs"], weight, part["stand"]) \
            if part["stand"].nunique() > 1 else {
                "within_stand": measurement_ceiling_stats.weighted_variance(part["delta_obs"], weight)}

        b_prime = measurement_ceiling_stats.ceiling_from_variances(tau2_mix, mean_sampling)
        route_a = measurement_ceiling_stats.route_a_tau2(observed["within_stand"], mean_sampling)
        band = measurement_ceiling_stats.fragility_band(observed["within_stand"], mean_sampling)
        # WITHIN-STAND, matching what the ceiling is a ceiling ON. tau2 here is the
        # within-stand true differential variance, so the raw correlation -- which E.5
        # showed is mostly the between-stand main effect -- is not the comparable
        # quantity and produces fractions above 1. Both are emitted; the fraction uses
        # the within-stand one.
        #
        # Pass A1: every opponent is scored here, not just the model, and every achieved
        # value carries a hitter-level paired bootstrap interval. Frozen rule 1 grades the
        # thesis in the LOW stratum against baselines, so a stratum table with a ceiling
        # and no opponent cannot answer the question the rule asks.
        achieved_raw = float(claim1_eval.weighted_rank_correlation(
            part["delta_obs"], part["delta_pred"], weight))
        achieved = (float(platoon_ceiling.recompute_platoon_rank_correlation(part)[0])
                    if part["stand"].nunique() > 1 else achieved_raw)
        columns = [column for _, column in DIFFERENTIAL_MODELS]
        reference_column = dict(DIFFERENTIAL_MODELS)[REFERENCE_MODEL]
        model_draws, model_boot = measurement_ceiling_stats.paired_rank_bootstrap(
            part, columns, n_boot=n_boot, seed=seed)
        model_boot = model_boot.set_index("column")
        assert abs(model_boot.loc[reference_column, "point"] - achieved) < 1e-12, \
            f"{stratum}: the bootstrap statistic disagrees with E.5's own residualisation"
        achieved_columns = {}
        for name, column in DIFFERENTIAL_MODELS:
            point = float(model_boot.loc[column, "point"])
            achieved_columns[f"achieved_{name}"] = point
            achieved_columns[f"achieved_{name}_ci_low"] = float(model_boot.loc[column, "ci_low"])
            achieved_columns[f"achieved_{name}_ci_high"] = float(model_boot.loc[column, "ci_high"])
            achieved_columns[f"fraction_of_ceiling_b_prime_{name}"] = (
                point / b_prime["ceiling_rank_corr"] if b_prime["ceiling_rank_corr"] > 0
                else np.nan)
            if name == REFERENCE_MODEL:
                continue
            contrast = measurement_ceiling_stats.paired_contrast(
                model_draws, columns, column, reference_column, point,
                float(model_boot.loc[reference_column, "point"]))
            achieved_columns[f"paired_diff_{name}_minus_reference"] = contrast["difference"]
            achieved_columns[f"paired_diff_{name}_ci_low"] = contrast["ci_low"]
            achieved_columns[f"paired_diff_{name}_ci_high"] = contrast["ci_high"]
            achieved_columns[f"paired_share_favouring_{name}"] = contrast["favours_a_share"]

        # bootstrap resamples HITTERS inside the stratum; tau2 stays fixed under B' because
        # it is not refit, so the interval reflects the sampling-variance profile and the
        # stand mix — which is exactly what the per-stratum ceiling is made of
        draws_b_prime, draws_a = [], []
        for _ in range(n_boot):
            pick = rng.integers(0, len(part), len(part))
            w = weight[pick]
            samp = float(np.average(sampling[pick], weights=w))
            t2 = float(np.average(part["tau2_b_prime"].to_numpy()[pick], weights=w))
            draws_b_prime.append(measurement_ceiling_stats.ceiling_from_variances(t2, samp)["ceiling_rank_corr"])
            obs = measurement_ceiling_stats.weighted_variance(part["delta_obs"].to_numpy()[pick], w)
            draws_a.append(measurement_ceiling_stats.route_a_tau2(obs, samp)["ceiling_rank_corr"])
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
            "route_a_fragility_ceiling_range_finite_only": band["ceiling_range_finite_only"],
            "achieved_rank_corr_within_stand": achieved,
            "achieved_rank_corr_raw": achieved_raw,
            "raw_note": ("the raw column includes the between-stand main effect and is NOT "
                         "comparable to a within-stand ceiling; it is reported so the gap "
                         "between the two is visible"),
            "fraction_of_ceiling_b_prime": (achieved / b_prime["ceiling_rank_corr"]
                                            if b_prime["ceiling_rank_corr"] > 0 else np.nan),
            **achieved_columns,
            "n_boot": int(n_boot),
            "reference_model": REFERENCE_MODEL,
            "achieved_interval_note": ("hitter-level paired bootstrap inside the stratum; the "
                                       "ceiling columns have their own interval and the two are "
                                       "not combined"),
            "post_selection_descriptive": True,
        })

    table = pd.DataFrame(rows)
    return table, precision_clause(table)


# ------------------------------------------------------------------ the main exhibit

EXHIBIT_COLUMNS = ("pooled",) + tuple(claim1_eval.STRATUM_NAMES)
BUILD_STAMP = "pre_hyperparameter_tuning"
BUILD_STAMP_NOTE = (
    "scored on the rebuild-baseline build that every pre-selection run used: lr 1e-3, "
    "warmup 0. Regenerate with `PYTHONPATH=. .venv/bin/python -m "
    "src.analysis.measurement_ceiling_report` if the selection stage ever promotes an arm.")
# The committed route table simulates the rank ceiling at 300 draws from seed 7
# (`route_tables`, `results/measurement_ceiling/routes.json` -> monte_carlo_rank_ceiling).
# The exhibit reuses those settings EXACTLY so its pooled denominator reproduces the
# committed 0.30931 rather than landing a Monte Carlo error away from it, and so the three
# strata are simulated the same way the pooled cell is. The seed is deliberately not the
# bootstrap's: the ceiling simulation and the hitter resample are independent.
MC_CEILING_DRAWS = 300
MC_CEILING_SEED = 7

# What the superseded stratum table carried that no exhibit cell needs: the route A
# fragility apparatus, the tau2 bookkeeping, and the raw-vs-within-stand gap. Kept in their
# own artifact so replacing two files with one exhibit loses no committed diagnostic.
ROUTE_DIAGNOSTIC_COLUMNS = (
    "stratum", "stratum_basis", "n_hitters", "median_denom_L", "median_denom_R",
    "observed_within_stand_var", "mean_sampling_var", "tau2_stratum_mix", "tau2_pooled",
    "ceiling_b_prime", "ceiling_b_prime_ci_low", "ceiling_b_prime_ci_high",
    "tau2_route_a", "route_a_degenerate", "ceiling_route_a",
    "ceiling_route_a_ci_low_given_nondegenerate", "ceiling_route_a_ci_high_given_nondegenerate",
    "route_a_share_degenerate_in_bootstrap", "route_a_ci_note",
    "route_a_fragility_ceiling_range_finite_only",
    "achieved_rank_corr_raw", "raw_note", "n_boot", "post_selection_descriptive")


def exhibit_cell(part, columns, tau2_mix, n_boot, seed, mc_draws=MC_CEILING_DRAWS,
                 mc_seed=MC_CEILING_SEED):
    """
    Every number one cell of the main exhibit carries, for one subset of hitters.

    part: the M.1 frame cut to this subset. Returns (per-model rows, denominators dict).

    Both fractions are oriented the SAME way -- higher is better, 1.0 means the bound is
    reached -- so one sign convention serves both metrics and `paired_contrast` needs no
    variant. Rank is achieved / ceiling; RMSE is floor / achieved, because a lower error is
    a better one.

    Both denominators are held FIXED across bootstrap replicates, the convention M.1
    already reports under: the interval is the sampling error of the numerator alone and
    understates total uncertainty by the width of the route disagreement.
    """
    weight = part["weight"].to_numpy(dtype="float64")
    sampling = part["sampling_var"].to_numpy(dtype="float64")

    # the rank denominator is the SIMULATED Spearman ceiling, not the analytic Pearson
    # closed form (decision log 2026-08-31): the analytic bound is exact for Pearson and
    # runs ~4% low against a Spearman numerator at this exposure skew. Run per subset so
    # the pooled cell and the stratum cells share one denominator definition.
    monte_carlo = measurement_ceiling_stats.monte_carlo_ceiling(
        tau2_mix, sampling, weight, n_draws=mc_draws, seed=mc_seed)
    ceiling = monte_carlo["mc_spearman_mean"]
    floor = measurement_ceiling_stats.differential_noise_floor(sampling, weight)

    rank_draws, rmse_draws, boot = measurement_ceiling_stats.paired_differential_bootstrap(
        part, columns, n_boot=n_boot, seed=seed)
    boot = boot.set_index("column")
    # both draw matrices become fractions on the shared replicates, so a paired contrast on
    # either metric is formed from the same resampled hitters
    rank_fraction_draws = rank_draws / ceiling
    rmse_fraction_draws = floor / rmse_draws
    reference_column = dict(DIFFERENTIAL_MODELS)[REFERENCE_MODEL]

    rows = []
    for name, column in DIFFERENTIAL_MODELS:
        rank_point = float(boot.loc[column, "rank_point"])
        rmse_point = float(boot.loc[column, "rmse_point"])
        rank_fraction = rank_point / ceiling
        rmse_fraction = floor / rmse_point
        row = {
            "model": name,
            "n_hitters": int(len(part)),
            "rank_corr_within_stand": rank_point,
            "rank_corr_ci_low": float(boot.loc[column, "rank_ci_low"]),
            "rank_corr_ci_high": float(boot.loc[column, "rank_ci_high"]),
            "rank_ceiling_mc_spearman": ceiling,
            "rank_ceiling_mc_standard_error": (monte_carlo["mc_spearman_sd"]
                                               / np.sqrt(monte_carlo["n_draws"])),
            "rank_fraction_of_ceiling": rank_fraction,
            "rank_fraction_ci_low": float(boot.loc[column, "rank_ci_low"]) / ceiling,
            "rank_fraction_ci_high": float(boot.loc[column, "rank_ci_high"]) / ceiling,
            "pa_weighted_rmse": rmse_point,
            "rmse_ci_low": float(boot.loc[column, "rmse_ci_low"]),
            "rmse_ci_high": float(boot.loc[column, "rmse_ci_high"]),
            "rmse_noise_floor": floor,
            "rmse_deconvolved": claim1_eval.deconvolve(rmse_point, floor),
            "rmse_fraction_of_floor": rmse_fraction,
            # the interval inverts: the widest RMSE draw is the smallest fraction
            "rmse_fraction_ci_low": floor / float(boot.loc[column, "rmse_ci_high"]),
            "rmse_fraction_ci_high": floor / float(boot.loc[column, "rmse_ci_low"]),
        }
        if name != REFERENCE_MODEL:
            rank_contrast = measurement_ceiling_stats.paired_contrast(
                rank_fraction_draws, columns, column, reference_column,
                rank_fraction, float(boot.loc[reference_column, "rank_point"]) / ceiling)
            rmse_contrast = measurement_ceiling_stats.paired_contrast(
                rmse_fraction_draws, columns, column, reference_column,
                rmse_fraction, floor / float(boot.loc[reference_column, "rmse_point"]))
            row.update({
                "rank_paired_diff_vs_reference": rank_contrast["difference"],
                "rank_paired_ci_low": rank_contrast["ci_low"],
                "rank_paired_ci_high": rank_contrast["ci_high"],
                "rank_paired_share_favouring_this_model": rank_contrast["favours_a_share"],
                "rmse_paired_diff_vs_reference": rmse_contrast["difference"],
                "rmse_paired_ci_low": rmse_contrast["ci_low"],
                "rmse_paired_ci_high": rmse_contrast["ci_high"],
                "rmse_paired_share_favouring_this_model": rmse_contrast["favours_a_share"],
            })
        rows.append(row)
    return rows, {"mc_ceiling": monte_carlo, "noise_floor": floor, "tau2_mix": tau2_mix}


def differential_exhibit(frame, routes, params_b_prime, n_boot=2000, seed=0):
    """
    THE main exhibit. One table, three models by four columns, both claim-1 metrics per cell.

    Rows are the model and the two Phase C opponents frozen rule 1 grades against; columns
    are the pooled population and the three frozen exposure strata. Every cell carries the
    fraction of the RMSE noise floor and the fraction of the Monte Carlo Spearman ceiling,
    each with a hitter-level paired bootstrap interval, plus the paired contrast against
    the model.

    Replaces `differential_scores.csv` and `stratum_ceiling_stratum_ceiling.csv`. The route
    and tau2 diagnostics those files carried, which no cell of this table needs, move to
    `differential_route_diagnostics.csv` rather than being dropped.

    Strata are read from the frozen `stratum` column and are never redefined here (spec §10.6).
    """
    per_hitter, _ = per_hitter_tau2(frame, params_b_prime)
    frame = frame.assign(tau2_b_prime=per_hitter)
    columns = [column for _, column in DIFFERENTIAL_MODELS]

    rows, denominators = [], []
    for name in EXHIBIT_COLUMNS:
        part = frame if name == "pooled" else frame[frame["stratum"] == name]
        part = part.reset_index(drop=True)
        assert len(part) >= 3, f"exhibit column {name!r} has too few hitters to score"
        weight = part["weight"].to_numpy(dtype="float64")
        # tau2 is NOT refit per column -- the common B' fit is applied against the column's
        # own sampling-variance profile, exactly as M.2 did it. Only the L/R composition moves.
        tau2_mix = (routes["B_prime"]["tau2"] if name == "pooled"
                    else float(np.average(part["tau2_b_prime"], weights=weight)))
        cells, denominator = exhibit_cell(part, columns, tau2_mix, n_boot, seed)
        for cell in cells:
            rows.append({"exhibit_column": name, **cell,
                         "reference_model": REFERENCE_MODEL,
                         "n_boot": int(n_boot),
                         "mc_ceiling_draws": denominator["mc_ceiling"]["n_draws"],
                         "rank_ceiling_analytic_pearson":
                             denominator["mc_ceiling"]["ceiling_rank_corr"],
                         "build": BUILD_STAMP,
                         "population": "M.6 intersection",
                         "post_selection_descriptive": True})
        denominators.append({
            "exhibit_column": name,
            "n_hitters": int(len(part)),
            "tau2_b_prime_mix": tau2_mix,
            "tau2_b_prime_pooled": routes["B_prime"]["tau2"],
            "mean_sampling_var": float(np.average(part["sampling_var"], weights=weight)),
            "rank_ceiling_mc_spearman": denominator["mc_ceiling"]["mc_spearman_mean"],
            "rank_ceiling_mc_spearman_sd": denominator["mc_ceiling"]["mc_spearman_sd"],
            "rank_ceiling_mc_standard_error": (
                denominator["mc_ceiling"]["mc_spearman_sd"]
                / np.sqrt(denominator["mc_ceiling"]["n_draws"])),
            "rank_ceiling_analytic_pearson": denominator["mc_ceiling"]["ceiling_rank_corr"],
            "rmse_noise_floor": denominator["noise_floor"],
            "reliability": denominator["mc_ceiling"]["reliability"],
            "denominator_note": (
                "both denominators are held FIXED across bootstrap replicates, so every "
                "interval is the sampling error of the NUMERATOR alone and understates "
                "total uncertainty by the width of the route disagreement. The rank "
                "denominator is itself simulated, and its own Monte Carlo standard error "
                "is the column beside it -- also not in any reported interval."),
            "build": BUILD_STAMP,
            "build_note": BUILD_STAMP_NOTE,
            "post_selection_descriptive": True,
        })
    return pd.DataFrame(rows), pd.DataFrame(denominators)


# ------------------------------------------------------------------ min_eval_pa sensitivity

MIN_EVAL_PA_FLOORS = (5, 10, 25)


def min_eval_pa_sensitivity(pa_df, model_predictions, eb_delta, gbm_full_delta,
                            floors=MIN_EVAL_PA_FLOORS, eval_season=EVAL_SEASON, n_boot=2000,
                            seed=0):
    """
    Rebuild the whole M.0/M.1 chain at each PA floor and report what moves.

    The 2026-08-31 decision keeps the reported population at the committed floor
    (`claim1_eval.MIN_EVAL_PA` = 10) and reports the floor as a SENSITIVITY. So this
    function changes nothing upstream: it re-derives a parallel population per floor,
    refits B' on that population, and re-scores, leaving the headline untouched.

    `claim1_eval.MIN_EVAL_PA_SENSITIVITY` is deliberately NOT reused. That tuple is Phase
    C's, its committed artifact is `results/baseline_ladder/baseline_ladder_min_eval_pa_sensitivity.csv`, and
    editing it would silently move a committed Phase C result. Phase M's floors are its own.

    A lower floor admits hitters whose differential is measured on very few PA against one
    hand, which raises the mean sampling variance and therefore LOWERS the B' ceiling while
    also adding noise to the achieved correlation. Both move the same way, so the fraction
    of ceiling is the quantity worth reading here, not either term alone.
    """
    from src.analysis import model_evaluation_eval

    by_batter, by_league = platoon_ceiling.per_pa_variance_tables(pa_df, eval_season)
    weights = pooling_weights(pa_df, eval_season)
    pooled_predictions = pool_predictions(model_predictions, weights, eval_season)
    pooled_pa = pa_df.copy()
    pooled_pa["p_throws"] = POOLED_HAND

    rows = []
    for floor in floors:
        platoon_floor, _ = model_evaluation_eval.platoon_frame(pa_df, model_predictions, eval_season,
                                           min_eval_pa=floor)
        pooled_floor, _ = claim1_eval.build_eval_frame(pooled_pa, pooled_predictions, eval_season,
                                                   min_eval_pa=floor)
        keep = set(platoon_floor["batter"]) & set(pooled_floor["batter"])
        frame = platoon_floor[platoon_floor["batter"].isin(keep)].reset_index(drop=True)
        frame = platoon_ceiling.attach_sampling_variance(frame, by_batter, by_league)
        frame["delta_eb"] = eb_delta.reindex(frame["batter"]).to_numpy()
        frame["delta_c3full"] = gbm_full_delta.reindex(frame["batter"]).to_numpy()
        # a floor below the committed one admits hitters no Phase C opponent forecast covers
        covered = frame["delta_eb"].notna() & frame["delta_c3full"].notna()

        restricted_pa = pa_df[pa_df["batter"].isin(keep)]
        params = eb.fit(restricted_pa, eval_season)
        weight = frame["weight"].to_numpy(dtype="float64")
        sampling = frame["sampling_var"].to_numpy(dtype="float64")
        mean_sampling = float(np.average(sampling, weights=weight))
        tau2, _ = pooled_tau2(frame, params)
        b_prime = measurement_ceiling_stats.ceiling_from_variances(tau2, mean_sampling)

        scored = frame[covered].reset_index(drop=True)
        columns = [column for _, column in DIFFERENTIAL_MODELS]
        _, boot = measurement_ceiling_stats.paired_rank_bootstrap(scored, columns, n_boot=n_boot, seed=seed)
        boot = boot.set_index("column")
        ceiling = b_prime["ceiling_rank_corr"]
        for name, column in DIFFERENTIAL_MODELS:
            point = float(boot.loc[column, "point"])
            rows.append({
                "min_eval_pa": int(floor),
                "is_committed_floor": floor == claim1_eval.MIN_EVAL_PA,
                "model": name,
                "n_hitters_population": int(len(frame)),
                "n_hitters_scored": int(len(scored)),
                "n_hitters_without_baseline_ladder_forecast": int((~covered).sum()),
                "mean_sampling_var": mean_sampling,
                "tau2_b_prime": tau2,
                "ceiling_b_prime": ceiling,
                "achieved_rank_corr": point,
                "achieved_ci_low": float(boot.loc[column, "ci_low"]),
                "achieved_ci_high": float(boot.loc[column, "ci_high"]),
                "fraction_of_ceiling_b_prime": point / ceiling if ceiling > 0 else np.nan,
                "n_boot": int(n_boot),
                "role": "sensitivity — the reported population is unchanged (2026-08-31)",
                "post_selection_descriptive": True,
            })
    return pd.DataFrame(rows)


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

def level_ceiling_level(pa_df, model_predictions, population, params_b_prime,
             pooled_scores, eval_season=EVAL_SEASON):
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
    for scope, table in (("pooled_committed_617", pooled_scores),):
        no_info = table[(table["model"] == "no_info_league_average")
                        & (table["stratum"] == "all")].iloc[0]
        model = table[(table["model"] == "model_v1") & (table["stratum"] == "all")].iloc[0]
        implied = claim1_eval.deconvolve(no_info["pa_weighted_rmse"], no_info["noise_floor"])
        assert abs(implied - no_info["model_rmse"]) < 1e-6, (
            "the no_info rung's model_rmse is not its deconvolved spread; the term "
            "provenance below would be wrong")
        committed[scope] = level_terms(no_info, model, int(no_info["n_hitters"]))

    # --- M.6 intersection (n = 545), the headline population
    restricted_pa = pooled_pa[pooled_pa["batter"].isin(intersection)]
    restricted_predictions = pooled_predictions[pooled_predictions["batter"].isin(intersection)]
    rungs = {"model_v1": restricted_predictions,
             "no_info_league_average": no_info_predictions(pa_df, intersection, eval_season)}
    scored = {}
    for name, predictions in rungs.items():
        table, coverage = claim1_eval.evaluate(restricted_pa, predictions, eval_season)
        scored[name] = table[table["stratum"] == "all"].iloc[0]
        scored[name + "_coverage"] = coverage
    intersection_terms = level_terms(scored["no_info_league_average"], scored["model_v1"],
                                     len(intersection))

    # --- cross-check 1: C.2-derived level variance composition, on the M.6 population
    cross_eb = eb_level_variance(pa_df, intersection, params_b_prime, weights, eval_season)
    # --- cross-check 2: split-half on game_pk parity, pooled wOBA
    cross_split = level_split_half(pa_df, intersection, eval_season)

    return {
        "post_selection_descriptive": POST_SELECTION_LABEL,
        "primary_population": "M.6 intersection",
        "intersection": intersection_terms,
        "committed_pooled_population": committed,
        "cross_check_eb_composition": cross_eb,
        "cross_check_split_half": cross_split,
        "term_provenance": {
            "E_var_noise": "pooled/claim1_eval noise_floor squared (noise_floor is an RMSE)",
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
    ceiling = measurement_ceiling_stats.ceiling_from_variances(true_variance, noise_variance)
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
    from src.analysis.baseline_ladder_trailing import predict as trailing_predict
    from src.analysis.baseline_ladder_report import NO_INFO_BUCKETS
    predictions = trailing_predict(pa_df, eval_season, variant="bucketed", buckets=NO_INFO_BUCKETS)
    weights = pooling_weights(pa_df, eval_season)
    pooled = pool_predictions(predictions, weights, eval_season)
    return pooled[pooled["batter"].isin(batters)]


def eb_level_variance(pa_df, batters, params, weights, eval_season):
    """
    Cross-check 1: the level-side true-talent variance implied by the B' C.2 fit.

    A hitter's pooled talent is w_L·theta_L + w_R·theta_R with F.5's prior-exposure weights,
    so its variance is w_L^2·tau2_L + w_R^2·tau2_R + 2·w_L·w_R·rho·tau_L·tau_R. This is the
    level-side analogue of Route B': same fit, same restricted population, the composition
    taken rather than the difference.
    """
    types = eb.batter_types(pa_df[pa_df["batter"].isin(batters)])
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
    estimate = measurement_ceiling_stats.split_half_from_halves(a, b)
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

def coverage_labels_coverage(pa_df, seed_predictions, manifest, population, eval_season=EVAL_SEASON):
    """
    M.4: one verification pass that E.14's 85.6% holds on the M.6 population. Labeling, not
    iteration — the plan's fallback already fired and no attempt is made to close the gap.

    E.14 scored 1149 SIDE-SPECIFIC groups; M.6 is 545 hitters. The measured coverage on the
    M.6 hitters' own rows is what every interval displayed in notebook 08 must be labeled
    with.
    """
    frame, _ = model_evaluation_probe_coverage.ensemble_frame(pa_df, seed_predictions, manifest, eval_season)
    keep = set(population.loc[population["in_intersection"], "batter"])
    restricted = frame[frame["batter"].isin(keep)]
    rows = model_evaluation_probe_coverage.coverage_rows(frame, "coverage_all_groups")
    rows += model_evaluation_probe_coverage.coverage_rows(restricted, "m6_intersection")
    for stratum in claim1_eval.STRATUM_NAMES:
        part = restricted[restricted["stratum"] == stratum]
        if len(part):
            rows += model_evaluation_probe_coverage.coverage_rows(part, f"m6_intersection_{stratum}")
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
        "measured_coverage_all_groups": headline("coverage_all_groups"),
        "measured_coverage_m6_intersection": headline("m6_intersection"),
        "committed_coverage_value": COMMITTED["coverage_pooled_coverage_95"],
        "reproduces_committed": abs(headline("coverage_all_groups")
                                    - COMMITTED["coverage_pooled_coverage_95"]) < 1e-6,
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
    parser.add_argument("--platoon-frame", default="results/model_evaluation/platoon_frame.csv")
    parser.add_argument("--model-predictions",
                        default="results/model_v1/model_v1_predictions_embedding_sgd_sgd_lr1.csv")
    parser.add_argument("--seed-predictions",
                        default="results/model_v1/model_v1_predictions_embedding_sgd_sgd_lr1_s{seed}.csv")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--manifest", default="data/processed/phase_d5/manifest.json")
    parser.add_argument("--pooled-scores", default="results/process_calibration/pooled_scores.csv")
    parser.add_argument("--ceiling-json", default="results/model_evaluation/ceiling.json")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--pitch-events", default="data/processed/pitch_events.parquet")
    parser.add_argument("--gbm-full-cache",
                        default="results/measurement_ceiling/differential_gbm_full_predictions.csv")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    claim1_eval.assert_not_test_season(args.eval_season, final_run=args.final_run)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pa_df = pd.read_parquet(args.eval_targets)
    platoon_frame = pd.read_csv(args.platoon_frame)
    model_predictions = pd.read_csv(args.model_predictions)
    model_predictions = model_predictions[model_predictions["season"] == args.eval_season]

    # ---------------- M.6
    print("M.6 population reconciliation")
    population, population_summary = m6_population(pa_df, platoon_frame, model_predictions,
                                                   args.eval_season)
    population.to_csv(out_dir / "population.csv", index=False)
    print(f"  E.5 {population_summary['n_platoon']}  F.5 {population_summary['n_pooled']}  "
          f"intersection {population_summary['n_intersection']}  "
          f"(E.5 subset of F.5: {population_summary['platoon_is_subset_of_pooled']})")

    frame, by_league = intersection_frame(platoon_frame, pa_df, population, args.eval_season)

    # ---------------- M.0
    print("\nM.0 routes")
    params_b = eb.fit(pa_df, args.eval_season)
    restricted_pa = pa_df[pa_df["batter"].isin(set(frame["batter"]))]
    params_b_prime = eb.fit(restricted_pa, args.eval_season)
    routes = route_tables(frame, by_league, params_b, params_b_prime)

    committed_split_half = {}
    ceiling_json = json.loads(Path(args.ceiling_json).read_text())
    for row in ceiling_json["part1_ceiling"]["split_half"]:
        committed_split_half[row["stand"]] = row["half_split_spearman"]
    route_c = route_c_diagnostic(frame, pa_df, routes["routes"]["B_prime"]["tau2"],
                                 committed_split_half, args.eval_season)

    fired = []
    if route_c["fallback_rule_2_fires"]:
        fired.append("2 — Route C outside both nulls with no bug: noise model suspect, "
                     "Route A demoted to 'reported, unvalidated-noise-model caveat'")
    if routes["routes"]["B_prime"]["degenerate"]:
        fired.append("3 — B' degenerate: headline falls back to the B-vs-A bracket")
    # §10 rule 1 fired at Phase M and is DISCHARGED in Pass A1: C.3-full is refit and scored
    # in M.1 and M.2 below, so the rule is recorded as discharged rather than as firing.
    assert C3_FULL_AVAILABILITY["emitted"], "rule 1 must fire again if C.3-full is not emitted"
    route_b_reproduces = abs(routes["routes"]["B"]["tau2"] - COMMITTED["route_b_tau2"]) < 5e-8
    if not route_b_reproduces:
        fired.append("4 — committed Route B failed to reproduce; current code's value adopted")

    routes = {
        "step": "M.0",
        "post_selection_descriptive": POST_SELECTION_LABEL,
        "fallback_rules_fired": fired,
        "route_rule": ROUTE_RULE,
        **routes,
        "route_c": {"verdict": route_c["verdict"], "reading": route_c["reading"],
                    "artifact": "results/measurement_ceiling/routes_route_c_diagnostic.json"},
        "field_provenance": {
            "route_b_field": "tau2_split_derived (results/baseline_ladder/eb_prior_parameters.csv)",
            "accessor": "eb_bivariate_eb.implied_split_constant -> tau2_split",
            "identity": "tau2_L + tau2_R - 2*rho*sqrt(tau2_L*tau2_R)",
            "note": ("E.15's Route B read this column via "
                     "model_evaluation_platoon_ceiling.load_eb_components['tau2_split']; for a "
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
                "(trailing.TRAILING_SEASONS = 3)"),
            "window_note": ("the spec's window PROSE is wrong; the committed VALUE "
                            "reproduces exactly on the code's three-season window, so §10 "
                            "rule 4 does not fire. B and B' differ only in population, "
                            "which is what the B->B' diagnostic requires."),
        },
        "gbm_full_availability": C3_FULL_AVAILABILITY,
    }
    (out_dir / "routes.json").write_text(json.dumps(routes, indent=2, default=float))
    (out_dir / "routes_route_c_diagnostic.json").write_text(
        json.dumps(route_c, indent=2, default=float))
    for name in ("B_prime", "A", "B"):
        row = routes["routes"][name]
        print(f"  {name:8s} tau2 {row['tau2']:.8f}  ceiling "
              f"{row['ceiling_rank_corr']:.4f}  fraction {row['fraction_of_ceiling']:.3f}"
              f"{'  DEGENERATE' if row['degenerate'] else ''}")
    print(f"  fragility band (A, x0.97/x1.03): {routes['fragility_band']['ceiling_range_finite_only']}")
    print(f"  Route C: {route_c['verdict']}")

    # ---------------- M.1
    print("\nM.1 differential scores")
    eb_delta = eb_differential(pa_df, args.eval_season)
    gbm_full_delta = gbm_full_differential(pa_df, args.gbm_full_cache, args.pitch_events,
                                         args.eval_season)
    frame = attach_differentials(frame, eb_delta, gbm_full_delta)
    differential_scores, differential_fractions, _, _ = differential(frame, routes["routes"],
                                                    n_boot=args.n_boot)
    # `differential_scores.csv` is superseded by the exhibit and is no longer written; the
    # route-by-route fraction table stays, because it is the artifact the A-to-B' bracket
    # is read off and the exhibit reports one route only.
    differential_fractions.to_csv(out_dir / "differential_fraction_of_ceiling.csv", index=False)
    print(differential_scores[["model", "n_hitters", "rank_corr_within_stand_pooled",
                     "rank_corr_ci_low", "rank_corr_ci_high",
                     "paired_diff_vs_reference"]].round(4).to_string(index=False))

    # ---------------- M.1 sensitivity: the PA floor
    print("\nM.1 sensitivity — min_eval_pa")
    sensitivity = min_eval_pa_sensitivity(pa_df, model_predictions, eb_delta, gbm_full_delta,
                                          eval_season=args.eval_season, n_boot=args.n_boot)
    sensitivity.to_csv(out_dir / "differential_min_eval_pa_sensitivity.csv", index=False)
    print(sensitivity[["min_eval_pa", "model", "n_hitters_scored", "ceiling_b_prime",
                       "achieved_rank_corr",
                       "fraction_of_ceiling_b_prime"]].round(4).to_string(index=False))

    # ---------------- M.2
    print("\nM.2 per-stratum ceiling")
    stratum_ceiling_table, clause = stratum_ceiling_stratum(frame, routes["routes"], params_b_prime, n_boot=args.n_boot)
    # the achieved/fraction columns move into the exhibit; the route and tau2 diagnostics no
    # cell of the exhibit needs are kept here rather than dropped
    stratum_ceiling_table[[c for c in ROUTE_DIAGNOSTIC_COLUMNS
                           if c in stratum_ceiling_table.columns]].to_csv(
        out_dir / "differential_route_diagnostics.csv", index=False)
    print(stratum_ceiling_table[["stratum", "n_hitters", "ceiling_b_prime",
                    "tau2_route_a", "route_a_degenerate",
                    "achieved_rank_corr_raw"]].round(4).to_string(index=False))
    print(f"  precision clause -> illustration stratum: {clause['illustration_stratum']}")

    # ---------------- the main exhibit
    print("\nMain exhibit — three models x pooled + three strata, both claim-1 metrics")
    exhibit, exhibit_denominators = differential_exhibit(
        frame, routes["routes"], params_b_prime, n_boot=args.n_boot)
    exhibit.to_csv(out_dir / "differential_exhibit.csv", index=False)
    exhibit_denominators.to_csv(out_dir / "differential_exhibit_denominators.csv", index=False)
    print(exhibit[["exhibit_column", "model", "n_hitters", "rank_fraction_of_ceiling",
                   "rank_fraction_ci_low", "rank_fraction_ci_high",
                   "rmse_fraction_of_floor", "rmse_fraction_ci_low",
                   "rmse_fraction_ci_high"]].round(4).to_string(index=False))
    print(f"  build stamp: {BUILD_STAMP}")

    # ---------------- M.3
    print("\nM.3 level-side ceiling")
    pooled_scores = pd.read_csv(args.pooled_scores)
    level_ceiling = level_ceiling_level(pa_df, model_predictions, population, params_b_prime, pooled_scores,
                  args.eval_season)
    level_ceiling["precision_clause"] = clause
    (out_dir / "level_ceiling_level_ceiling.json").write_text(json.dumps(level_ceiling, indent=2, default=float))
    level = level_ceiling["intersection"]
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
    coverage_labels_table, coverage_labels_summary = coverage_labels_coverage(pa_df, seed_predictions, manifest, population,
                                       args.eval_season)
    coverage_labels_table.to_csv(out_dir / "coverage_labels_coverage_labels.csv", index=False)
    print(f"  measured 95% coverage: all groups "
          f"{coverage_labels_summary['measured_coverage_all_groups']:.4f}  "
          f"M.6 intersection {coverage_labels_summary['measured_coverage_m6_intersection']:.4f}")

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
        "coverage_labels_coverage": coverage_labels_summary,
        "route_c_verdict": route_c["verdict"],
        "precision_clause": clause,
        "artifacts": sorted(path.name for path in out_dir.glob("*")),
    }
    (out_dir / "measurement_ceiling_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nwrote {out_dir}/")


if __name__ == "__main__":
    main()
