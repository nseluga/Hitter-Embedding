"""
Phase M reproduction gates (spec §9.3) plus the driver's structural invariants.

The gates are the reason no Phase M number was committed before this file ran. Each one
asks the same question in a different place: does the code, run now, produce the number
already written down? A gate failure means the artifact on disk and the code in the repo
have drifted apart, and every downstream figure inherits the drift.

They refit C.2 on the real trailing window and score the real eval frame.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis import c2_bivariate_eb as c2
from src.analysis import claim1_eval
from src.analysis import e_platoon_ceiling as e15
from src.analysis import m_ceiling, m_report

REPO = Path(__file__).resolve().parents[1]
EVAL_SEASON = 2024

# Committed values these gates reproduce. Transcribed once, here, and nowhere else.
COMMITTED_ROUTE_B_TAU2 = 0.00059034
COMMITTED_TAU2_SPLIT_DERIVED = {
    "L": 0.0007322758887773689,
    "R": 0.0004917731167240981,
    "S": 0.000588220547130464,
}
COMMITTED_E5_WITHIN_STAND_RANK_CORR = e15.E5_REPORTED_WITHIN_STAND_RANK_CORR
COMMITTED_E14_POOLED_COVERAGE_95 = 0.855527


@pytest.fixture(scope="module")
def pa_df():
    return pd.read_parquet(REPO / "data/processed/eval_targets_pa.parquet")


@pytest.fixture(scope="module")
def e5_frame():
    return pd.read_csv(REPO / "results/phase_e/e5_platoon_frame.csv")


@pytest.fixture(scope="module")
def model_predictions():
    table = pd.read_csv(REPO / "results/phase_d/d5_predictions_d10_baseline.csv")
    return table[table["season"] == EVAL_SEASON]


@pytest.fixture(scope="module")
def population(pa_df, e5_frame, model_predictions):
    return m_report.m6_population(pa_df, e5_frame, model_predictions, EVAL_SEASON)


@pytest.fixture(scope="module")
def intersection(pa_df, e5_frame, population):
    return m_report.intersection_frame(e5_frame, pa_df, population[0], EVAL_SEASON)


@pytest.fixture(scope="module")
def c2_delta(pa_df):
    return m_report.c2_differential(pa_df, EVAL_SEASON)


@pytest.fixture(scope="module")
def c3_full_delta(pa_df):
    """Reads the committed Pass A1 cache. If it is missing the refit runs, which is the
    point: the artifact is reproducible, not a hand-placed file."""
    return m_report.c3_full_differential(
        pa_df, REPO / "results/phase_m/m1_c3_full_predictions.csv",
        REPO / "data/processed/pitch_events.parquet", EVAL_SEASON)


@pytest.fixture(scope="module")
def scored_frame(intersection, c2_delta, c3_full_delta):
    frame, _ = intersection
    return m_report.attach_differentials(frame, c2_delta, c3_full_delta)


# ---------------------------------------------------------------- §9.3a

def test_unrestricted_c2_refit_reproduces_the_committed_route_b_tau2(pa_df):
    """
    Gate 3a. Route B is the committed C.2 fit, and every ceiling that reads it inherits it.
    If a refit today does not land on the number in `c2_prior_parameters.csv`, the fit has
    drifted and §10 rule 4 fires — so this must run before B, B', or the B->B' diagnostic
    is believed.
    """
    params = c2.fit(pa_df, EVAL_SEASON)
    for batter_type, expected in COMMITTED_TAU2_SPLIT_DERIVED.items():
        _, tau2_split = c2.implied_split_constant(params[batter_type])
        assert tau2_split == pytest.approx(expected, abs=1e-12), \
            f"C.2 refit no longer reproduces the committed tau2_split for {batter_type}"


def test_committed_prior_parameter_file_matches_the_refit(pa_df):
    """The same gate from the other side: the CSV on disk against a live refit."""
    stored = pd.read_csv(REPO / "results/phase_c/c2_prior_parameters.csv")
    params = c2.fit(pa_df, EVAL_SEASON)
    for _, row in stored.iterrows():
        _, tau2_split = c2.implied_split_constant(params[row["batter_type"]])
        assert tau2_split == pytest.approx(row["tau2_split_derived"], abs=1e-12)


def test_pooled_route_b_lands_on_the_spec_value(pa_df, e5_frame, population):
    """
    The pooled Route B tau2 the spec commits (0.00059034) is the per-hitter weight-average
    of the derived split variances, so it is a property of the fit AND the population.
    """
    frame, _ = m_report.intersection_frame(e5_frame, pa_df, population[0], EVAL_SEASON)
    tau2, _ = m_report.pooled_tau2(frame, c2.fit(pa_df, EVAL_SEASON))
    assert tau2 == pytest.approx(COMMITTED_ROUTE_B_TAU2, abs=5e-8)


# ---------------------------------------------------------------- §9.3b

def test_e15_committed_pooled_numbers_reproduce_from_the_artifact():
    """
    Gate 3b, arithmetic half: E.15's stored reliability and ceiling are consistent with its
    own stored tau2 and sampling variance. A stored ceiling that does not follow from the
    stored variances means the artifact was assembled from two different runs.
    """
    stored = json.loads((REPO / "results/phase_e/e15_ceiling.json").read_text())
    rows = stored["part1_ceiling"]["by_stand"]
    assert {row["stand"] for row in rows} >= {"L", "R", "pooled"}
    for row in rows:
        recomputed = m_ceiling.ceiling_from_variances(row["tau2_split_true"],
                                                      row["mean_sampling_var_weighted"])
        assert recomputed["reliability"] == pytest.approx(row["reliability_variance_ratio"],
                                                          abs=1e-12), row["stand"]
        assert recomputed["ceiling_rank_corr"] == pytest.approx(row["ceiling_rank_corr"],
                                                                abs=1e-12), row["stand"]


# ---------------------------------------------------------------- §9.3c

def test_m1_reproduces_e5s_committed_within_stand_rank_correlation(intersection):
    """
    Gate 3c. M.1 adds an opponent to E.5's table; the model's own row must come out
    IDENTICAL to what E.5 reported, or M.1 is scoring a different thing and the C.2
    comparison beside it is meaningless.
    """
    frame, _ = intersection
    achieved, _ = e15.recompute_e5_rank_correlation(frame)
    assert achieved == pytest.approx(COMMITTED_E5_WITHIN_STAND_RANK_CORR, abs=1e-12)


def test_m1_scores_every_opponent_on_identical_rows(scored_frame):
    routes = {"A": {"tau2": 1e-4, "ceiling_rank_corr": 0.1},
              "B_prime": {"tau2": 4e-4, "ceiling_rank_corr": 0.3}}
    scores, fractions, draws, columns = m_report.m1_differential(scored_frame, routes,
                                                                 n_boot=200)
    assert set(scores["model"]) == {"phase_d_model", "c2_bivariate", "c3_gbm_full"}
    assert scores["n_hitters"].nunique() == 1, "the rungs were scored on different rows"
    model_row = scores[scores["model"] == "phase_d_model"].iloc[0]
    assert model_row["rank_corr_within_stand_pooled"] == pytest.approx(
        COMMITTED_E5_WITHIN_STAND_RANK_CORR, abs=1e-12)
    assert len(fractions) == 6, "every rung must be reported under both B' and A"
    assert draws.shape == (200, 3)
    assert columns == ["delta_pred", "delta_c2", "delta_c3full"]


def test_m1_every_achieved_value_carries_an_interval_that_brackets_it(scored_frame):
    """The review of 2026-08-31: a point estimate with no interval is what this pass
    exists to stop shipping."""
    routes = {"A": {"tau2": 1e-4, "ceiling_rank_corr": 0.1},
              "B_prime": {"tau2": 4e-4, "ceiling_rank_corr": 0.3}}
    scores, fractions, _, _ = m_report.m1_differential(scored_frame, routes, n_boot=200)
    for _, row in scores.iterrows():
        assert row["rank_corr_ci_low"] < row["rank_corr_within_stand_pooled"] \
            < row["rank_corr_ci_high"], row["model"]
    assert fractions[["fraction_ci_low", "fraction_ci_high"]].notna().all().all()


def test_the_reference_model_has_a_zero_paired_difference_with_itself(scored_frame):
    routes = {"A": {"tau2": 1e-4, "ceiling_rank_corr": 0.1},
              "B_prime": {"tau2": 4e-4, "ceiling_rank_corr": 0.3}}
    scores, _, _, _ = m_report.m1_differential(scored_frame, routes, n_boot=100)
    reference = scores[scores["model"] == m_report.REFERENCE_MODEL].iloc[0]
    assert reference["paired_diff_vs_reference"] == 0.0
    assert np.isnan(reference["paired_share_favouring_this_model"])


def test_c3_full_reproduces_its_committed_phase_c_claim1_row(pa_df, c3_full_delta):
    """
    The 2026-08-31 revisit condition on the differential-table decision: absence of
    C.3-full becomes a capability limit only if the refit cannot reproduce its committed
    Phase C parameters. It reproduces, so the condition does not fire.
    """
    cached = pd.read_csv(REPO / "results/phase_m/m1_c3_full_predictions.csv")
    cached = cached[cached["season"] == EVAL_SEASON]
    committed = pd.read_csv(REPO / "results/phase_c/c_claim1_scores.csv")
    committed = committed[committed["model"] == "c3_gbm_full"]
    assert len(committed), "Phase C has no committed c3_gbm_full row to reproduce"
    rescored, _ = claim1_eval.evaluate(pa_df, cached, EVAL_SEASON)
    for _, row in committed.iterrows():
        mine = rescored[rescored["stratum"] == row["stratum"]]
        if not len(mine):
            continue
        assert float(mine.iloc[0]["pa_weighted_rmse"]) == pytest.approx(
            float(row["pa_weighted_rmse"]), abs=1e-6), row["stratum"]
        assert float(mine.iloc[0]["rank_corr_weighted"]) == pytest.approx(
            float(row["rank_corr_weighted"]), abs=1e-9), row["stratum"]
    assert c3_full_delta.notna().all() and len(c3_full_delta) > 500


def test_the_pa_floor_sensitivity_leaves_the_reported_population_alone(pa_df,
                                                                      model_predictions,
                                                                      c2_delta,
                                                                      c3_full_delta,
                                                                      intersection):
    """Item 2 of Pass A1. The floor is a SENSITIVITY: the committed floor's row must still
    describe the same population M.1 reports on."""
    frame, _ = intersection
    table = m_report.min_eval_pa_sensitivity(pa_df, model_predictions, c2_delta,
                                             c3_full_delta, floors=(5, 10, 25),
                                             n_boot=100)
    assert set(table["min_eval_pa"]) == {5, 10, 25}
    committed = table[table["is_committed_floor"]]
    assert len(committed), "no row is marked as the committed floor"
    assert committed["n_hitters_population"].iloc[0] == len(frame)
    assert claim1_eval.MIN_EVAL_PA_SENSITIVITY == (10, 25, 50), \
        "Phase C's own floor tuple was edited; its committed artifact would move"
    # a lower floor admits worse-measured hitters, so sampling variance rises monotonically
    by_floor = table.drop_duplicates("min_eval_pa").set_index("min_eval_pa")
    assert by_floor.loc[5, "mean_sampling_var"] > by_floor.loc[25, "mean_sampling_var"]


# ---------------------------------------------------------------- population

def test_m6_rebuilds_f5s_committed_coverage(population):
    """
    F.5's population is rebuilt through f5_pooled's own path rather than transcribed, so a
    silent change in the pooling or the scorer shows up here instead of in the intersection.
    """
    _, summary = population
    committed = json.loads((REPO / "results/phase_f/f5_pooled_summary.json").read_text())
    expected = committed["coverage"] if "coverage" in committed else committed
    rebuilt = summary["f5_rebuilt_coverage"]
    assert rebuilt["scored_groups"] == summary["n_f5"]
    for field in ("actual_groups", "scored_groups", "dropped_below_min_eval_pa"):
        if field in expected:
            assert rebuilt[field] == expected[field], f"F.5 rebuild differs on {field}"


def test_the_intersection_is_e5s_population(population):
    _, summary = population
    assert summary["e5_is_subset_of_f5"], (
        "E.5 is no longer a subset of F.5; M.0-M.2 can no longer be read on E.5's frame "
        "without re-restriction")
    assert summary["n_intersection"] == summary["n_e5"]


def test_the_intersection_never_touches_the_frozen_strata(intersection, e5_frame):
    frame, _ = intersection
    original = e5_frame.set_index("batter")["stratum"]
    assert (frame.set_index("batter")["stratum"] == original.reindex(frame["batter"])).all(), \
        "a stratum label changed; strata are frozen (spec §10.6)"
    assert claim1_eval.STRATUM_BOUNDARIES == (113, 452)


# ---------------------------------------------------------------- structure

def test_route_b_prime_is_not_degenerate_and_sits_between_a_and_b(pa_df, intersection):
    """
    The pre-registered primary route has to be usable. A degenerate B' is §10 rule 3 and
    changes the headline, so it is asserted here rather than discovered in the artifact.
    """
    frame, by_league = intersection
    params_b = c2.fit(pa_df, EVAL_SEASON)
    restricted = pa_df[pa_df["batter"].isin(set(frame["batter"]))]
    routes = m_report.route_tables(frame, by_league, params_b, c2.fit(restricted, EVAL_SEASON))
    b_prime = routes["routes"]["B_prime"]
    assert not b_prime["degenerate"]
    assert routes["routes"]["A"]["tau2"] < b_prime["tau2"] < routes["routes"]["B"]["tau2"]
    assert 0.0 < b_prime["ceiling_rank_corr"] < 1.0


def test_the_fragility_band_is_wide_enough_to_matter(intersection):
    """
    Route A is reported WITH its band because the band is the point: a 3% error in the
    noise model moves the ceiling by tens of percent. If that ever stopped being true,
    the route rule's rationale would need revisiting.
    """
    frame, _ = intersection
    weight = frame["weight"].to_numpy(dtype="float64")
    sampling = frame["sampling_var"].to_numpy(dtype="float64")
    observed = e15.between_within_stand(frame["delta_obs"], weight, frame["stand"])
    band = m_ceiling.fragility_band(observed["within_stand"],
                                    float(np.average(sampling, weights=weight)))
    low, high = band["ceiling_range_finite_only"]
    assert high / low > 1.2, "the fragility band collapsed; the route rule's premise changed"


def test_the_precision_clause_moves_off_a_stratum_whose_ci_includes_zero():
    """
    The M.2 clause is applied mechanically, so it is tested on a synthetic table where the
    answer is known — not on the real one, where it would just restate the result.
    """
    table = pd.DataFrame({
        "stratum": ["low", "medium", "high"],
        "ceiling_b_prime": [0.20, 0.30, 0.32],
        "ceiling_b_prime_ci_low": [-0.01, 0.28, 0.20],
        "ceiling_b_prime_ci_high": [0.40, 0.32, 0.44],
    })
    clause = m_report.precision_clause(table)
    assert clause["ci_includes_zero"]["low"] is True
    assert clause["illustration_stratum"] == "medium"
    assert clause["moved"] is True
    assert clause["triggered"] is True


def test_the_precision_clause_stays_on_low_when_low_is_usable():
    """
    The clause is conditional. 'low' is the stratum the thesis is graded on, so a merely
    narrower interval elsewhere must NOT move the illustration -- only 'low' being unusable
    does. Here 'medium' has the wider interval and 'low' is fine, and the tie-break must
    stay dormant.
    """
    table = pd.DataFrame({
        "stratum": ["low", "medium"],
        "ceiling_b_prime": [0.20, 0.30],
        "ceiling_b_prime_ci_low": [0.19, 0.28],
        "ceiling_b_prime_ci_high": [0.21, 0.31],
    })
    clause = m_report.precision_clause(table)
    assert clause["illustration_stratum"] == "low"
    assert clause["moved"] is False
    assert clause["triggered"] is False
    assert clause["relative_ci_width"]["medium"] < clause["relative_ci_width"]["low"], \
        "the fixture must have a narrower medium, or it does not test the tie-break"


def test_c3_full_is_now_emitted_and_rule_1_is_recorded_as_discharged():
    """Pass A1 discharges §10 fallback rule 1. If C.3-full ever stops being emitted the
    rule fires again, and this test is what makes that a failure rather than a silence."""
    record = m_report.C3_FULL_AVAILABILITY
    assert record["emitted"] is True
    assert "DISCHARGED" in record["fallback_rule"]
    assert record["predictions"].endswith("m1_c3_full_predictions.csv")
    assert (REPO / record["predictions"]).exists()


def test_the_route_rule_names_b_prime_primary_and_never_promotes_route_c():
    rule = m_report.ROUTE_RULE
    assert rule["primary"] == "B_prime"
    assert "A" in rule["always_reported"]
    assert rule["provenance_only"] == ["B"]
    assert rule["never_primary"] == ["C"]
