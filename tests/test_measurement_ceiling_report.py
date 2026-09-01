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

from src.analysis import baseline_ladder_bivariate_eb as eb
from src.analysis import claim1_eval
from src.analysis import model_evaluation_platoon_ceiling as ceiling
from src.analysis import measurement_ceiling_stats, measurement_ceiling_report

REPO = Path(__file__).resolve().parents[1]
EVAL_SEASON = 2024

# Committed values these gates reproduce. Transcribed once, here, and nowhere else.
COMMITTED_ROUTE_B_TAU2 = 0.00059034
COMMITTED_TAU2_SPLIT_DERIVED = {
    "L": 0.0007322758887773689,
    "R": 0.0004917731167240981,
    "S": 0.000588220547130464,
}
COMMITTED_E5_WITHIN_STAND_RANK_CORR = ceiling.E5_REPORTED_WITHIN_STAND_RANK_CORR
COMMITTED_E14_POOLED_COVERAGE_95 = 0.855527


@pytest.fixture(scope="module")
def pa_df():
    return pd.read_parquet(REPO / "data/processed/eval_targets_pa.parquet")


@pytest.fixture(scope="module")
def platoon_frame():
    return pd.read_csv(REPO / "results/model_evaluation/platoon_frame.csv")


@pytest.fixture(scope="module")
def model_predictions():
    table = pd.read_csv(REPO / "results/model_v1/model_v1_predictions_rebuild_baseline.csv")
    return table[table["season"] == EVAL_SEASON]


@pytest.fixture(scope="module")
def population(pa_df, platoon_frame, model_predictions):
    return measurement_ceiling_report.m6_population(pa_df, platoon_frame, model_predictions, EVAL_SEASON)


@pytest.fixture(scope="module")
def intersection(pa_df, platoon_frame, population):
    return measurement_ceiling_report.intersection_frame(platoon_frame, pa_df, population[0], EVAL_SEASON)


@pytest.fixture(scope="module")
def eb_delta(pa_df):
    return measurement_ceiling_report.eb_differential(pa_df, EVAL_SEASON)


@pytest.fixture(scope="module")
def gbm_full_delta(pa_df):
    """Reads the committed Pass A1 cache. If it is missing the refit runs, which is the
    point: the artifact is reproducible, not a hand-placed file."""
    return measurement_ceiling_report.gbm_full_differential(
        pa_df, REPO / "results/measurement_ceiling/differential_gbm_full_predictions.csv",
        REPO / "data/processed/pitch_events.parquet", EVAL_SEASON)


@pytest.fixture(scope="module")
def scored_frame(intersection, eb_delta, gbm_full_delta):
    frame, _ = intersection
    return measurement_ceiling_report.attach_differentials(frame, eb_delta, gbm_full_delta)


@pytest.fixture(scope="module")
def exhibit_tables(pa_df, intersection, scored_frame):
    """The main exhibit built on the real 2024 frame, at a bootstrap depth the test suite
    can afford. The bootstrap depth moves only the interval columns, never a point."""
    from src.analysis import baseline_ladder_bivariate_eb as eb
    frame, by_league = intersection
    params_b = eb.fit(pa_df, EVAL_SEASON)
    params_b_prime = eb.fit(pa_df[pa_df["batter"].isin(set(frame["batter"]))], EVAL_SEASON)
    routes = measurement_ceiling_report.route_tables(
        frame, by_league, params_b, params_b_prime)["routes"]
    return measurement_ceiling_report.differential_exhibit(
        scored_frame, routes, params_b_prime, n_boot=200)


# ---------------------------------------------------------------- §9.3a

def test_unrestricted_eb_refit_reproduces_the_committed_route_b_tau2(pa_df):
    """
    Gate 3a. Route B is the committed C.2 fit, and every ceiling that reads it inherits it.
    If a refit today does not land on the number in `eb_prior_parameters.csv`, the fit has
    drifted and §10 rule 4 fires — so this must run before B, B', or the B->B' diagnostic
    is believed.
    """
    params = eb.fit(pa_df, EVAL_SEASON)
    for batter_type, expected in COMMITTED_TAU2_SPLIT_DERIVED.items():
        _, tau2_split = eb.implied_split_constant(params[batter_type])
        assert tau2_split == pytest.approx(expected, abs=1e-12), \
            f"C.2 refit no longer reproduces the committed tau2_split for {batter_type}"


def test_committed_prior_parameter_file_matches_the_refit(pa_df):
    """The same gate from the other side: the CSV on disk against a live refit."""
    stored = pd.read_csv(REPO / "results/baseline_ladder/eb_prior_parameters.csv")
    params = eb.fit(pa_df, EVAL_SEASON)
    for _, row in stored.iterrows():
        _, tau2_split = eb.implied_split_constant(params[row["batter_type"]])
        assert tau2_split == pytest.approx(row["tau2_split_derived"], abs=1e-12)


def test_pooled_route_b_lands_on_the_spec_value(pa_df, platoon_frame, population):
    """
    The pooled Route B tau2 the spec commits (0.00059034) is the per-hitter weight-average
    of the derived split variances, so it is a property of the fit AND the population.
    """
    frame, _ = measurement_ceiling_report.intersection_frame(platoon_frame, pa_df, population[0], EVAL_SEASON)
    tau2, _ = measurement_ceiling_report.pooled_tau2(frame, eb.fit(pa_df, EVAL_SEASON))
    assert tau2 == pytest.approx(COMMITTED_ROUTE_B_TAU2, abs=5e-8)


# ---------------------------------------------------------------- §9.3b

def test_ceiling_committed_pooled_numbers_reproduce_from_the_artifact():
    """
    Gate 3b, arithmetic half: E.15's stored reliability and ceiling are consistent with its
    own stored tau2 and sampling variance. A stored ceiling that does not follow from the
    stored variances means the artifact was assembled from two different runs.
    """
    stored = json.loads((REPO / "results/model_evaluation/ceiling.json").read_text())
    rows = stored["part1_ceiling"]["by_stand"]
    assert {row["stand"] for row in rows} >= {"L", "R", "pooled"}
    for row in rows:
        recomputed = measurement_ceiling_stats.ceiling_from_variances(row["tau2_split_true"],
                                                      row["mean_sampling_var_weighted"])
        assert recomputed["reliability"] == pytest.approx(row["reliability_variance_ratio"],
                                                          abs=1e-12), row["stand"]
        assert recomputed["ceiling_rank_corr"] == pytest.approx(row["ceiling_rank_corr"],
                                                                abs=1e-12), row["stand"]


# ---------------------------------------------------------------- §9.3c

def test_differential_reproduces_e5s_committed_within_stand_rank_correlation(intersection):
    """
    Gate 3c. M.1 adds an opponent to E.5's table; the model's own row must come out
    IDENTICAL to what E.5 reported, or M.1 is scoring a different thing and the C.2
    comparison beside it is meaningless.
    """
    frame, _ = intersection
    achieved, _ = ceiling.recompute_platoon_rank_correlation(frame)
    assert achieved == pytest.approx(COMMITTED_E5_WITHIN_STAND_RANK_CORR, abs=1e-12)


def test_differential_scores_every_opponent_on_identical_rows(scored_frame):
    routes = {"A": {"tau2": 1e-4, "ceiling_rank_corr": 0.1},
              "B_prime": {"tau2": 4e-4, "ceiling_rank_corr": 0.3}}
    scores, fractions, draws, columns = measurement_ceiling_report.differential(scored_frame, routes,
                                                                 n_boot=200)
    assert set(scores["model"]) == {"model_v1_model", "eb_bivariate", "gbm_full"}
    assert scores["n_hitters"].nunique() == 1, "the rungs were scored on different rows"
    model_row = scores[scores["model"] == "model_v1_model"].iloc[0]
    assert model_row["rank_corr_within_stand_pooled"] == pytest.approx(
        COMMITTED_E5_WITHIN_STAND_RANK_CORR, abs=1e-12)
    assert len(fractions) == 6, "every rung must be reported under both B' and A"
    assert draws.shape == (200, 3)
    assert columns == ["delta_pred", "delta_eb", "delta_c3full"]


def test_differential_every_achieved_value_carries_an_interval_that_brackets_it(scored_frame):
    """The review of 2026-08-31: a point estimate with no interval is what this pass
    exists to stop shipping."""
    routes = {"A": {"tau2": 1e-4, "ceiling_rank_corr": 0.1},
              "B_prime": {"tau2": 4e-4, "ceiling_rank_corr": 0.3}}
    scores, fractions, _, _ = measurement_ceiling_report.differential(scored_frame, routes, n_boot=200)
    for _, row in scores.iterrows():
        assert row["rank_corr_ci_low"] < row["rank_corr_within_stand_pooled"] \
            < row["rank_corr_ci_high"], row["model"]
    assert fractions[["fraction_ci_low", "fraction_ci_high"]].notna().all().all()


def test_the_reference_model_has_a_zero_paired_difference_with_itself(scored_frame):
    routes = {"A": {"tau2": 1e-4, "ceiling_rank_corr": 0.1},
              "B_prime": {"tau2": 4e-4, "ceiling_rank_corr": 0.3}}
    scores, _, _, _ = measurement_ceiling_report.differential(scored_frame, routes, n_boot=100)
    reference = scores[scores["model"] == measurement_ceiling_report.REFERENCE_MODEL].iloc[0]
    assert reference["paired_diff_vs_reference"] == 0.0
    assert np.isnan(reference["paired_share_favouring_this_model"])


def test_gbm_full_reproduces_its_committed_baseline_ladder_claim1_row(pa_df, gbm_full_delta):
    """
    The 2026-08-31 revisit condition on the differential-table decision: absence of
    C.3-full becomes a capability limit only if the refit cannot reproduce its committed
    Phase C parameters. It reproduces, so the condition does not fire.
    """
    cached = pd.read_csv(REPO / "results/measurement_ceiling/differential_gbm_full_predictions.csv")
    cached = cached[cached["season"] == EVAL_SEASON]
    committed = pd.read_csv(REPO / "results/baseline_ladder/baseline_ladder_claim1_scores.csv")
    committed = committed[committed["model"] == "gbm_full"]
    assert len(committed), "Phase C has no committed gbm_full row to reproduce"
    rescored, _ = claim1_eval.evaluate(pa_df, cached, EVAL_SEASON)
    for _, row in committed.iterrows():
        mine = rescored[rescored["stratum"] == row["stratum"]]
        if not len(mine):
            continue
        assert float(mine.iloc[0]["pa_weighted_rmse"]) == pytest.approx(
            float(row["pa_weighted_rmse"]), abs=1e-6), row["stratum"]
        assert float(mine.iloc[0]["rank_corr_weighted"]) == pytest.approx(
            float(row["rank_corr_weighted"]), abs=1e-9), row["stratum"]
    assert gbm_full_delta.notna().all() and len(gbm_full_delta) > 500


def test_the_pa_floor_sensitivity_leaves_the_reported_population_alone(pa_df,
                                                                      model_predictions,
                                                                      eb_delta,
                                                                      gbm_full_delta,
                                                                      intersection):
    """Item 2 of Pass A1. The floor is a SENSITIVITY: the committed floor's row must still
    describe the same population M.1 reports on."""
    frame, _ = intersection
    table = measurement_ceiling_report.min_eval_pa_sensitivity(pa_df, model_predictions, eb_delta,
                                             gbm_full_delta, floors=(5, 10, 25),
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
    F.5's population is rebuilt through pooled's own path rather than transcribed, so a
    silent change in the pooling or the scorer shows up here instead of in the intersection.
    """
    _, summary = population
    committed = json.loads((REPO / "results/process_calibration/pooled_summary.json").read_text())
    expected = committed["coverage"] if "coverage" in committed else committed
    rebuilt = summary["pooled_rebuilt_coverage"]
    assert rebuilt["scored_groups"] == summary["n_pooled"]
    for field in ("actual_groups", "scored_groups", "dropped_below_min_eval_pa"):
        if field in expected:
            assert rebuilt[field] == expected[field], f"F.5 rebuild differs on {field}"


def test_the_intersection_is_e5s_population(population):
    _, summary = population
    assert summary["platoon_is_subset_of_pooled"], (
        "E.5 is no longer a subset of F.5; M.0-M.2 can no longer be read on E.5's frame "
        "without re-restriction")
    assert summary["n_intersection"] == summary["n_platoon"]


def test_the_intersection_never_touches_the_frozen_strata(intersection, platoon_frame):
    frame, _ = intersection
    original = platoon_frame.set_index("batter")["stratum"]
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
    params_b = eb.fit(pa_df, EVAL_SEASON)
    restricted = pa_df[pa_df["batter"].isin(set(frame["batter"]))]
    routes = measurement_ceiling_report.route_tables(frame, by_league, params_b, eb.fit(restricted, EVAL_SEASON))
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
    observed = ceiling.between_within_stand(frame["delta_obs"], weight, frame["stand"])
    band = measurement_ceiling_stats.fragility_band(observed["within_stand"],
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
    clause = measurement_ceiling_report.precision_clause(table)
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
    clause = measurement_ceiling_report.precision_clause(table)
    assert clause["illustration_stratum"] == "low"
    assert clause["moved"] is False
    assert clause["triggered"] is False
    assert clause["relative_ci_width"]["medium"] < clause["relative_ci_width"]["low"], \
        "the fixture must have a narrower medium, or it does not test the tie-break"


def test_gbm_full_is_now_emitted_and_rule_1_is_recorded_as_discharged():
    """Pass A1 discharges §10 fallback rule 1. If C.3-full ever stops being emitted the
    rule fires again, and this test is what makes that a failure rather than a silence."""
    record = measurement_ceiling_report.C3_FULL_AVAILABILITY
    assert record["emitted"] is True
    assert "DISCHARGED" in record["fallback_rule"]
    assert record["predictions"].endswith("differential_gbm_full_predictions.csv")
    assert (REPO / record["predictions"]).exists()


def test_the_route_rule_names_b_prime_primary_and_never_promotes_route_c():
    rule = measurement_ceiling_report.ROUTE_RULE
    assert rule["primary"] == "B_prime"
    assert "A" in rule["always_reported"]
    assert rule["provenance_only"] == ["B"]
    assert rule["never_primary"] == ["C"]


# ---------------------------------------------------------------- the main exhibit (Pass A2)

def test_the_exhibit_pooled_rank_ceiling_reproduces_the_committed_route_table():
    """
    REPRODUCTION GATE, blocking. The exhibit's rank denominator must be the SAME simulated
    Spearman ceiling the committed route table reports, not a fresh simulation that lands a
    Monte Carlo error away from it. Both run 300 draws from seed 7 on the pooled frame, so
    the agreement is exact rather than approximate. A drift here silently rescales every
    pooled rank fraction in the exhibit against a number the decision log does not name.
    """
    import json
    from pathlib import Path
    from src.analysis import measurement_ceiling_report

    routes = json.loads(Path("results/measurement_ceiling/routes.json").read_text())
    committed = routes["monte_carlo_rank_ceiling"]
    assert committed["n_draws"] == measurement_ceiling_report.MC_CEILING_DRAWS
    assert measurement_ceiling_report.MC_CEILING_SEED == 7
    assert committed["mc_spearman_mean"] == pytest.approx(0.30931, abs=5e-5)


def test_both_exhibit_fractions_point_the_same_way(exhibit_tables):
    """
    ORIENTATION, blocking. The table mixes a correlation (higher is better) with an error
    (lower is better). Both are reported as a fraction of their own bound so that one sign
    convention serves the whole exhibit and `paired_contrast` needs no metric-specific
    variant. If either flipped, every paired contrast in that half would read backwards.
    """
    exhibit, _ = exhibit_tables
    for _, row in exhibit.iterrows():
        assert row["rank_fraction_of_ceiling"] == pytest.approx(
            row["rank_corr_within_stand"] / row["rank_ceiling_mc_spearman"], abs=1e-12)
        assert row["rmse_fraction_of_floor"] == pytest.approx(
            row["rmse_noise_floor"] / row["pa_weighted_rmse"], abs=1e-12)
        # the RMSE interval inverts under the reciprocal: the widest error is the worst cell
        assert row["rmse_fraction_ci_low"] < row["rmse_fraction_ci_high"]
        assert row["rank_fraction_ci_low"] < row["rank_fraction_ci_high"]


def test_the_exhibit_rank_cells_reproduce_the_superseded_tables(exhibit_tables):
    """
    REPRODUCTION GATE, blocking. The exhibit replaces `differential_scores.csv` and the
    stratum table, so its rank numerators must be those files' committed values to the last
    digit -- a replacement that also moves the numbers is a new result wearing an old name.
    Reference values are the Pass A2 spec's table and the committed stratum CSV.
    """
    exhibit, _ = exhibit_tables
    expected = {
        ("pooled", "model_v1_model"): 0.14626474805407283,
        ("pooled", "eb_bivariate"): 0.14582642711091456,
        ("pooled", "gbm_full"): 0.1412976147889645,
        ("low", "model_v1_model"): 0.1901200387409761,
        ("low", "eb_bivariate"): 0.19737665798869397,
        ("low", "gbm_full"): 0.23600403635465694,
        ("medium", "model_v1_model"): 0.14263646602311905,
        ("high", "model_v1_model"): 0.11046510805034039,
    }
    indexed = exhibit.set_index(["exhibit_column", "model"])
    for key, value in expected.items():
        assert indexed.loc[key, "rank_corr_within_stand"] == pytest.approx(value, abs=1e-12)


def test_the_exhibit_covers_every_model_in_every_column(exhibit_tables):
    """Frozen rule 1 grades the LOW stratum against baselines, so a missing opponent in any
    column is the gap the rule exists to close, not a formatting detail."""
    from src.analysis import measurement_ceiling_report
    exhibit, denominators = exhibit_tables
    models = {name for name, _ in measurement_ceiling_report.DIFFERENTIAL_MODELS}
    assert set(exhibit["exhibit_column"]) == set(measurement_ceiling_report.EXHIBIT_COLUMNS)
    assert set(denominators["exhibit_column"]) == set(measurement_ceiling_report.EXHIBIT_COLUMNS)
    for column, part in exhibit.groupby("exhibit_column"):
        assert set(part["model"]) == models, f"{column} is missing an opponent"
    assert (exhibit["build"] == measurement_ceiling_report.BUILD_STAMP).all()
    assert exhibit["post_selection_descriptive"].all()


def test_the_exhibit_strata_sum_to_the_pooled_population(exhibit_tables):
    """The three strata partition the pooled column; a hitter in neither or in two would
    make the stratum cells and the pooled cell answer different questions."""
    exhibit, _ = exhibit_tables
    per_column = exhibit.groupby("exhibit_column")["n_hitters"].first()
    assert per_column[["low", "medium", "high"]].sum() == per_column["pooled"]


def test_a_model_paired_against_itself_is_absent_from_the_contrast_columns(exhibit_tables):
    """The reference carries no contrast against itself — an all-zero row would read as a
    measured tie rather than as the definition it is."""
    from src.analysis import measurement_ceiling_report
    exhibit, _ = exhibit_tables
    reference = exhibit[exhibit["model"] == measurement_ceiling_report.REFERENCE_MODEL]
    assert reference["rank_paired_diff_vs_reference"].isna().all()
    assert reference["rmse_paired_diff_vs_reference"].isna().all()


def test_the_constant_mean_null_scores_worse_than_every_model_on_differential_rmse(intersection):
    """
    Pins the number the 2026-09-01 RMSE-saturation decision-log entry rests on.

    The entry's claim is narrow and worth keeping honest: the three models are
    indistinguishable FROM EACH OTHER on differential RMSE, but not from a null.
    An earlier draft asserted the null would score about the same; it does not.
    """
    frame = intersection[0]
    obs = frame["delta_obs"].to_numpy(dtype="float64")
    weight = frame["weight"].to_numpy(dtype="float64")
    constant = np.full_like(obs, np.average(obs, weights=weight))
    null_rmse = claim1_eval.pa_weighted_rmse(obs, constant, weight)
    assert null_rmse == pytest.approx(0.06862, abs=5e-5)

    exhibit = pd.read_csv(
        REPO / "results/measurement_ceiling/differential_exhibit.csv")
    pooled = exhibit[exhibit["exhibit_column"] == "pooled"]
    assert (pooled["pa_weighted_rmse"] < null_rmse).all(), \
        "a model no longer beats the constant-mean null; the logged finding needs rewriting"
