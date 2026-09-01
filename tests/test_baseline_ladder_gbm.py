"""
Verification gates for the C.3 GBM baseline.

C.3 is the first Phase C baseline that LEARNS from the training seasons rather
than deriving a closed form, which changes what can silently go wrong. The two
failure classes that matter:

  - rolling-window leakage. Every feature is a trailing aggregate, and the
    classic silent bug is a window that includes its own target season. G1
    injects an extreme season-s performance and asserts not one feature moves.
  - a model that is not actually wired to its inputs. A GBM trained on
    misaligned features still fits, still early-stops, still prints a plausible
    RMSE. G2 (overfit a tiny batch) and G5 (decode a feature row by hand) are
    the checks that fail loudly when the join is wrong.

G8 and G9 are regression gates for two bugs caught during the build: a label
shuffle that moved the model's intercept instead of only breaking the
feature->target link, and `stand` taken per season rather than per
(season, pitcher hand), which labels every switch hitter a right-handed batter
because he faces more RHP than LHP.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis import c3_gbm as c3

BATTER_ID_BASE = 100_000
PITCHER_ID_BASE = 900_000


def pa_table(records):
    """PA-level eval-target rows from (batter, season, p_throws, stand, woba_points) tuples."""
    frame = pd.DataFrame(records, columns=["batter", "season", "p_throws", "stand", "woba_points"])
    frame["in_denominator"] = True
    frame["pitcher"] = PITCHER_ID_BASE
    frame["at_bat_number"] = np.arange(len(frame))
    frame["game_pk"] = 1
    return frame


def synthetic_pa(n_hitters=60, seasons=(2020, 2021, 2022, 2023, 2024), n_pa=40,
                 seed=0, stand="R"):
    """
    A PA table where each hitter has a fixed true talent per side, so a model
    with working wiring can recover an ordering and a broken one cannot.
    """
    rng = np.random.default_rng(seed)
    talent = {"L": rng.normal(0.32, 0.05, n_hitters), "R": rng.normal(0.32, 0.05, n_hitters)}
    records = []
    for index in range(n_hitters):
        for season in seasons:
            for hand in ("L", "R"):
                points = rng.normal(talent[hand][index], 0.5, n_pa)
                records.extend(
                    (BATTER_ID_BASE + index, season, hand, stand, float(value))
                    for value in points
                )
    return pa_table(records)


def pitch_table(records):
    """Labeled-pitch rows from dicts, with the columns season_process reads."""
    frame = pd.DataFrame(records)
    defaults = {"pitch_type": "FF", "strikes": 0, "swing": False, "contact": False,
                "ev": np.nan, "la": np.nan, "spray": np.nan, "stand": "R"}
    for column, value in defaults.items():
        if column not in frame:
            frame[column] = value
    return frame


def synthetic_pitches(pa_frame, seed=0):
    """One pitch per PA, with process outcomes correlated to nothing in particular —
    enough to exercise the aggregation and the join, not to carry signal."""
    rng = np.random.default_rng(seed)
    n = len(pa_frame)
    return pitch_table({
        "batter": pa_frame["batter"].to_numpy(),
        "season": pa_frame["season"].to_numpy(),
        "p_throws": pa_frame["p_throws"].to_numpy(),
        "stand": pa_frame["stand"].to_numpy(),
        "pitch_type": rng.choice(["FF", "SL", "CH"], n),
        "strikes": rng.integers(0, 3, n),
        "swing": rng.random(n) < 0.5,
        "contact": rng.random(n) < 0.7,
        "ev": np.where(rng.random(n) < 0.3, rng.normal(88, 12, n), np.nan),
        "la": np.where(rng.random(n) < 0.3, rng.normal(12, 20, n), np.nan),
        "spray": np.where(rng.random(n) < 0.3, rng.normal(5, 25, n), np.nan),
    })


def training_stub(n_rows=40, n_features=None, seed=0):
    """A tiny training frame with a deterministic target — used by the overfit gate."""
    rng = np.random.default_rng(seed)
    columns = c3.FEATURE_SETS["full"]
    frame = pd.DataFrame(rng.normal(size=(n_rows, len(columns))), columns=columns)
    frame["season"] = np.repeat([2021, 2022, 2023], np.ceil(n_rows / 3).astype(int))[:n_rows]
    frame["pa"] = 100.0
    frame["target_woba"] = 0.3 + 0.1 * frame["own_woba"] - 0.05 * frame["whiff_rate"]
    return frame


# ---------------------------------------------------------------------------
# G1 — split boundary: no feature may see its own target season or later
# ---------------------------------------------------------------------------

def test_g1_features_ignore_the_target_season_entirely():
    pa = synthetic_pa(n_hitters=20, seasons=(2021, 2022, 2023), n_pa=30)
    pitches = synthetic_pitches(pa)

    poisoned = pa.copy()
    mask = poisoned["season"] == 2023
    poisoned.loc[mask, "woba_points"] = 9.0  # every 2023 PA a grand slam

    groups = pa[["batter", "p_throws"]].drop_duplicates()
    base = c3.window_features(groups, c3.season_outcome(pa), c3.season_process(pitches), 2023)
    after = c3.window_features(groups, c3.season_outcome(poisoned),
                               c3.season_process(pitches), 2023)
    pd.testing.assert_frame_equal(base, after)


def test_g1_training_frame_excludes_the_evaluated_season():
    pa = synthetic_pa(n_hitters=20, seasons=(2020, 2021, 2022, 2023, 2024), n_pa=25)
    training = c3.build_training_frame(pa, c3.season_outcome(pa),
                                       c3.season_process(synthetic_pitches(pa)),
                                       eval_season=2024, min_target_season=2021)
    assert training["season"].max() == 2023
    assert 2024 not in set(training["season"])


def test_g1_window_excludes_seasons_before_the_trailing_span():
    pa = synthetic_pa(n_hitters=15, seasons=(2018, 2022, 2023), n_pa=30)
    groups = pa[["batter", "p_throws"]].drop_duplicates()
    features = c3.window_features(groups, c3.season_outcome(pa),
                                  c3.season_process(synthetic_pitches(pa)), 2024,
                                  n_seasons=2)
    # only 2022-2023 are inside a 2-season window ending at 2024; 2018 must not count
    assert features["seasons_seen"].max() == 2


# ---------------------------------------------------------------------------
# G2 — overfit a tiny batch: broken wiring cannot fit data it is not connected to
# ---------------------------------------------------------------------------

def test_g2_overfits_a_tiny_batch():
    training = training_stub(n_rows=40)
    unregularized = dict(c3.PARAMS, max_depth=8, learning_rate=0.3, subsample=1.0,
                         colsample_bytree=1.0, min_child_weight=0.0, reg_lambda=0.0,
                         n_estimators=400)
    model, _ = c3.fit(training, feature_set="full", inner_val_season=2023,
                      params=unregularized, n_rounds=400)
    predicted = model.predict(training[c3.FEATURE_SETS["full"]].astype(float))
    assert np.sqrt(np.mean((predicted - training["target_woba"]) ** 2)) < 1e-3


# ---------------------------------------------------------------------------
# G3 — determinism: the same seed must reproduce the same predictions
# ---------------------------------------------------------------------------

def test_g3_predictions_are_deterministic():
    pa = synthetic_pa(n_hitters=40, n_pa=20)
    process = c3.season_process(synthetic_pitches(pa))
    first = c3.predict(pa, process, 2024, seed=0)
    second = c3.predict(pa, process, 2024, seed=0)
    pd.testing.assert_frame_equal(first, second)


def test_g3_a_different_seed_is_allowed_to_differ():
    """Guard against a seed that is silently ignored — subsampling must respond to it."""
    pa = synthetic_pa(n_hitters=40, n_pa=20)
    process = c3.season_process(synthetic_pitches(pa))
    first = c3.predict(pa, process, 2024, seed=0)
    second = c3.predict(pa, process, 2024, seed=17)
    assert not np.allclose(first["pred_woba"], second["pred_woba"])


# ---------------------------------------------------------------------------
# G4 — loss scale: a constant predictor must score exactly the target's spread
# ---------------------------------------------------------------------------

def test_g4_constant_prediction_scores_the_weighted_sd():
    from src.analysis.claim1_eval import pa_weighted_rmse

    training = training_stub(n_rows=60)
    weight = training["pa"].to_numpy(dtype=float)
    actual = training["target_woba"].to_numpy()
    mean = np.average(actual, weights=weight)
    expected = np.sqrt(np.average((actual - mean) ** 2, weights=weight))
    assert pa_weighted_rmse(actual, np.full_like(actual, mean), weight) == pytest.approx(expected)


def test_g4_fitted_model_beats_the_constant_on_a_learnable_target():
    training = training_stub(n_rows=300)
    model, _ = c3.fit(training, feature_set="full", inner_val_season=2023)
    predicted = model.predict(training[c3.FEATURE_SETS["full"]].astype(float))
    actual = training["target_woba"].to_numpy()
    assert np.sqrt(np.mean((predicted - actual) ** 2)) < np.std(actual)


# ---------------------------------------------------------------------------
# G5 — decode a feature row by hand
# ---------------------------------------------------------------------------

def test_g5_feature_row_matches_hand_computation():
    pa = pa_table([(BATTER_ID_BASE, 2023, "R", "R", 0.9)] * 10)
    pitches = pitch_table([
        # 4 fastballs: 3 swings, 1 whiff; one ball in play at 100 mph
        {"batter": BATTER_ID_BASE, "season": 2023, "p_throws": "R", "pitch_type": "FF",
         "swing": True, "contact": True, "ev": 100.0, "la": 10.0, "spray": 5.0},
        {"batter": BATTER_ID_BASE, "season": 2023, "p_throws": "R", "pitch_type": "FF",
         "swing": True, "contact": True},
        {"batter": BATTER_ID_BASE, "season": 2023, "p_throws": "R", "pitch_type": "FF",
         "swing": True, "contact": False, "strikes": 2},
        {"batter": BATTER_ID_BASE, "season": 2023, "p_throws": "R", "pitch_type": "FF",
         "swing": False, "contact": False},
        # 2 sliders: 1 swing, 1 whiff
        {"batter": BATTER_ID_BASE, "season": 2023, "p_throws": "R", "pitch_type": "SL",
         "swing": True, "contact": False},
        {"batter": BATTER_ID_BASE, "season": 2023, "p_throws": "R", "pitch_type": "SL",
         "swing": False, "contact": False},
    ])
    groups = pd.DataFrame({"batter": [BATTER_ID_BASE], "p_throws": ["R"]})
    row = c3.window_features(groups, c3.season_outcome(pa), c3.season_process(pitches), 2024).iloc[0]

    assert row["own_pa"] == 10 and row["own_woba"] == pytest.approx(0.9)
    assert row["own_pitches"] == 6
    assert row["swing_rate"] == pytest.approx(4 / 6)
    assert row["whiff_rate"] == pytest.approx(2 / 4)
    assert row["whiff_rate_fastball"] == pytest.approx(1 / 3)
    assert row["whiff_rate_breaking"] == pytest.approx(1 / 1)
    assert row["ev_mean"] == pytest.approx(100.0)
    assert row["hard_hit_rate"] == pytest.approx(1.0)
    assert row["whiff_rate_two_strike"] == pytest.approx(1.0)
    assert np.isnan(row["whiff_rate_offspeed"])  # no offspeed swings, not a zero
    assert np.isnan(row["opp_woba"]) and row["opp_pa"] == 0


def test_g5_window_rate_is_pooled_not_a_mean_of_season_rates():
    """A 1-pitch season must not weigh as much as a 100-pitch one."""
    pitches = pitch_table(
        [{"batter": BATTER_ID_BASE, "season": 2022, "p_throws": "R",
          "swing": True, "contact": True}] * 99
        + [{"batter": BATTER_ID_BASE, "season": 2022, "p_throws": "R",
            "swing": True, "contact": False}]
        + [{"batter": BATTER_ID_BASE, "season": 2023, "p_throws": "R",
            "swing": True, "contact": False}]
    )
    pa = pa_table([(BATTER_ID_BASE, 2022, "R", "R", 0.3),
                   (BATTER_ID_BASE, 2023, "R", "R", 0.3)])
    groups = pd.DataFrame({"batter": [BATTER_ID_BASE], "p_throws": ["R"]})
    row = c3.window_features(groups, c3.season_outcome(pa), c3.season_process(pitches), 2024).iloc[0]
    assert row["whiff_rate"] == pytest.approx(2 / 101)   # pooled
    assert row["whiff_rate"] != pytest.approx(0.505)     # mean of 0.01 and 1.00


def test_g5_intentional_balls_are_excluded_from_process_rates():
    pitches = pitch_table(
        [{"batter": BATTER_ID_BASE, "season": 2023, "p_throws": "R",
          "swing": True, "contact": True}] * 4
        + [{"batter": BATTER_ID_BASE, "season": 2023, "p_throws": "R",
            "pitch_type": "IN", "swing": False}] * 4
    )
    pa = pa_table([(BATTER_ID_BASE, 2023, "R", "R", 0.3)])
    groups = pd.DataFrame({"batter": [BATTER_ID_BASE], "p_throws": ["R"]})
    row = c3.window_features(groups, c3.season_outcome(pa), c3.season_process(pitches), 2024).iloc[0]
    assert row["own_pitches"] == 4 and row["swing_rate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# G6 — missingness is explicit, and no hitter is silently dropped
# ---------------------------------------------------------------------------

def test_g6_a_hitter_with_no_history_gets_nan_features_not_zero_rates():
    pa = pa_table([(BATTER_ID_BASE, 2023, "R", "R", 0.3)] * 5
                  + [(BATTER_ID_BASE + 1, 2024, "R", "R", 0.3)] * 5)
    groups = pd.DataFrame({"batter": [BATTER_ID_BASE + 1], "p_throws": ["R"]})
    row = c3.window_features(groups, c3.season_outcome(pa),
                             c3.season_process(pitch_table([
                                 {"batter": BATTER_ID_BASE, "season": 2023, "p_throws": "R"}])),
                             2024).iloc[0]
    assert row["own_pa"] == 0 and np.isnan(row["own_woba"])
    assert np.isnan(row["whiff_rate"]) and np.isnan(row["ev_mean"])
    assert row["seasons_seen"] == 0


def test_g6_every_active_group_receives_a_finite_prediction():
    pa = synthetic_pa(n_hitters=40, n_pa=20)
    # a debutant with no prior seasons at all — the low-exposure population
    debut = pa_table([(BATTER_ID_BASE + 999, 2024, "R", "R", 0.4)] * 30)
    pa = pd.concat([pa, debut], ignore_index=True)
    process = c3.season_process(synthetic_pitches(pa))

    predictions = c3.predict(pa, process, 2024)
    active = pa[pa["season"] == 2024][["batter", "p_throws"]].drop_duplicates()
    assert len(predictions) == len(active)
    assert np.isfinite(predictions["pred_woba"]).all()
    assert (predictions["batter"] == BATTER_ID_BASE + 999).any()


# ---------------------------------------------------------------------------
# G7 — the claim-1 contract and the feature-matrix shape
# ---------------------------------------------------------------------------

def test_g7_predict_emits_the_claim1_contract():
    pa = synthetic_pa(n_hitters=30, n_pa=20)
    predictions = c3.predict(pa, c3.season_process(synthetic_pitches(pa)), 2024)
    assert list(predictions.columns) == c3.KEY + ["pred_woba"]
    assert predictions["season"].nunique() == 1 and predictions["season"].iloc[0] == 2024
    assert not predictions.duplicated(c3.KEY).any()


def test_g7_feature_sets_are_nested_and_the_matrix_shape_is_asserted():
    assert set(c3.FEATURE_SETS["outcome"]).issubset(c3.FEATURE_SETS["full"])
    training = training_stub(n_rows=20)
    assert c3._matrix(training, "outcome").shape == (20, len(c3.FEATURE_SETS["outcome"]))
    assert c3._matrix(training, "full").shape == (20, len(c3.FEATURE_SETS["full"]))
    with pytest.raises(AssertionError):
        c3.fit(training, feature_set="full", inner_val_season=1999)


def test_inner_val_season_defaults_to_the_latest_training_season():
    """
    Re-scoring on a different evaluated season must move the inner-val season with
    it. A constant would early-stop on an interior season while fitting on newer
    rows — not a leak, but it silently stops matching the documented protocol.
    """
    training = training_stub(n_rows=60)
    latest = int(training["season"].max())
    _, derived_rounds = c3.fit(training, feature_set="outcome", seed=0)
    _, latest_rounds = c3.fit(training, feature_set="outcome",
                              inner_val_season=latest, seed=0)
    assert derived_rounds == latest_rounds

    # and it must track the frame: drop the last season and the default moves with it
    earlier = training[training["season"] < latest]
    _, earlier_rounds = c3.fit(earlier, feature_set="outcome", seed=0)
    _, explicit_rounds = c3.fit(earlier, feature_set="outcome",
                                inner_val_season=int(earlier["season"].max()), seed=0)
    assert earlier_rounds == explicit_rounds


# ---------------------------------------------------------------------------
# G8 — the label-shuffle null must not hand the null model a free advantage
# ---------------------------------------------------------------------------

def test_g8_shuffle_preserves_the_intercept():
    """
    Regression gate for a bug in the FIRST version of this diagnostic. Permuting
    the target while leaving PA weights in place changes the PA-weighted mean of
    the labels, which is the model's intercept — on the real table it moved .3186
    to .2836, close enough to the low-exposure population's true level that the
    null model "beat" C.2 in the headline stratum on that alone. The shuffled
    null then reads as "C.3's margin is indistinguishable from noise", which is
    the opposite of the truth.
    """
    training = training_stub(n_rows=200)
    training["pa"] = np.linspace(10, 600, len(training))  # weights must vary to bite
    training["target_woba"] = 0.25 + 0.0001 * training["pa"]  # ...and correlate with the target

    shuffled = c3.shuffled_training(training, seed=0)
    weighted = lambda f: np.average(f["target_woba"], weights=f["pa"])
    assert weighted(shuffled) == pytest.approx(weighted(training))
    assert not np.allclose(shuffled["target_woba"], training["target_woba"]), "nothing was shuffled"


def test_g8_shuffle_destroys_the_feature_target_link():
    training = training_stub(n_rows=300)
    shuffled = c3.shuffled_training(training, seed=0)
    real_corr = abs(training["own_woba"].corr(training["target_woba"]))
    shuffled_corr = abs(shuffled["own_woba"].corr(shuffled["target_woba"]))
    assert real_corr > 0.5 and shuffled_corr < 0.2


# ---------------------------------------------------------------------------
# G9 — switch hitters (regression gate for a bug caught during the build)
# ---------------------------------------------------------------------------

def test_g9_switch_hitters_are_flagged_not_folded_into_right_handers():
    # a switch hitter faces far more RHP: a per-season mode would call him a RHB
    records = ([(BATTER_ID_BASE, 2023, "R", "L", 0.3)] * 90
               + [(BATTER_ID_BASE, 2023, "L", "R", 0.3)] * 10
               + [(BATTER_ID_BASE + 1, 2023, "R", "L", 0.3)] * 50)
    pa = pa_table(records)
    groups = pa[["batter", "p_throws"]].drop_duplicates()
    features = c3.window_features(groups, c3.season_outcome(pa),
                                  c3.season_process(synthetic_pitches(pa)), 2024)
    switch = features[features["batter"] == BATTER_ID_BASE]
    pure_lefty = features[features["batter"] == BATTER_ID_BASE + 1]
    assert (switch["bats_switch"] == 1.0).all()
    assert (switch["bats_left"] == 0.0).all()
    assert (pure_lefty["bats_switch"] == 0.0).all()
    assert (pure_lefty["bats_left"] == 1.0).all()
