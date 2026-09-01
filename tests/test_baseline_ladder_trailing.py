"""
Verification gates for the C.1 trailing-average baselines.

Closed-form checks on hand-computable cases, plus the leakage boundary. A baseline
that silently peeks at the evaluated season would look excellent and quietly raise
the bar the Phase D model has to clear, so the window guard matters as much here as
in the model code.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis import c1_trailing as c1


def pa_table(rows):
    """Build a PA-level eval-target table from (batter, season, hand, n_pa, woba_points) tuples."""
    records = []
    pa_id = 0
    for batter, season, hand, n_pa, points in rows:
        for _ in range(n_pa):
            pa_id += 1
            records.append({
                "batter": batter, "season": season, "p_throws": hand,
                "woba_points": points, "in_denominator": True,
                "pitcher": 900001, "game_pk": pa_id, "at_bat_number": 1,
            })
    return pd.DataFrame(records)


# ---- gate: the trailing window is exactly 3 prior seasons, never the eval season ----

def test_window_excludes_eval_season_and_older_seasons():
    pa_df = pa_table([
        (1, 2019, "L", 10, 0.5),   # too old for a 3-season window
        (1, 2021, "L", 10, 0.5),
        (1, 2022, "L", 10, 0.5),
        (1, 2023, "L", 10, 0.5),
        (1, 2024, "L", 10, 0.9),   # the evaluated season
    ])
    window = c1.trailing_window(pa_df, eval_season=2024, n_seasons=3)
    assert sorted(window["season"].unique()) == [2021, 2022, 2023]
    assert window["season"].max() < 2024


def test_window_with_no_prior_seasons_fails_loud():
    pa_df = pa_table([(1, 2024, "L", 10, 0.5)])
    with pytest.raises(AssertionError):
        c1.trailing_window(pa_df, eval_season=2024, n_seasons=3)


# ---- gate: bucket weights are a correct step function ----

def test_bucket_weight_step_function():
    pa = pd.Series([0, 49, 50, 199, 200, 5000])
    assert list(c1.bucket_weight(pa)) == [0.0, 0.0, 0.5, 0.5, 1.0, 1.0]


# ---- gate: the two variants differ exactly where shrinkage bites ----

def small_sample_case():
    """One hot-hitting small-sample hitter, plus filler that sets the league average."""
    rows = [(1, 2023, "L", 30, 0.450)]                  # 30 PA, .450 -> noise
    rows += [(h, 2023, "L", 200, 0.300) for h in range(2, 12)]  # league ~ .300
    rows += [(1, 2024, "L", 100, 0.0), (2, 2024, "L", 100, 0.0)]
    return pa_table(rows)


def test_raw_variant_does_not_shrink():
    preds = c1.predict(small_sample_case(), eval_season=2024, variant="raw")
    hitter = preds.set_index("batter").loc[1, "pred_woba"]
    assert hitter == pytest.approx(0.450), "raw must predict the observed trailing wOBA unshrunk"


def test_bucketed_variant_shrinks_small_samples_toward_league():
    pa_df = small_sample_case()
    preds = c1.predict(pa_df, eval_season=2024, variant="bucketed")
    hitter = preds.set_index("batter").loc[1, "pred_woba"]
    # 30 PA falls in the <50 bucket -> weight 0 -> entirely league average
    window = c1.trailing_window(pa_df, 2024)
    league = c1.league_average(window)["L"]
    assert hitter == pytest.approx(league)
    assert hitter < 0.450, "shrinkage must pull a lucky small sample down"


def test_bucketed_matches_raw_for_high_exposure_hitters():
    """At 200+ PA the bucket weight is 1.0, so the two variants must agree exactly."""
    rows = [(1, 2023, "L", 400, 0.380)]
    rows += [(h, 2023, "L", 200, 0.300) for h in range(2, 12)]
    rows += [(1, 2024, "L", 100, 0.0)]
    pa_df = pa_table(rows)
    raw = c1.predict(pa_df, 2024, variant="raw").set_index("batter").loc[1, "pred_woba"]
    bucketed = c1.predict(pa_df, 2024, variant="bucketed").set_index("batter").loc[1, "pred_woba"]
    assert raw == pytest.approx(bucketed) == pytest.approx(0.380)


def test_blend_is_closed_form_in_the_middle_bucket():
    rows = [(1, 2023, "L", 100, 0.400)]                  # 100 PA -> weight 0.5
    rows += [(h, 2023, "L", 200, 0.300) for h in range(2, 12)]
    rows += [(1, 2024, "L", 100, 0.0)]
    pa_df = pa_table(rows)
    league = c1.league_average(c1.trailing_window(pa_df, 2024))["L"]
    got = c1.predict(pa_df, 2024, variant="bucketed").set_index("batter").loc[1, "pred_woba"]
    assert got == pytest.approx(0.5 * 0.400 + 0.5 * league)


# ---- gate: hitters with no prior history get the league average, not a null ----

def test_debutant_gets_side_specific_league_average():
    rows = [(h, 2023, "L", 200, 0.300) for h in range(2, 12)]
    rows += [(h, 2023, "R", 200, 0.340) for h in range(2, 12)]
    rows += [(99, 2024, "L", 100, 0.0), (99, 2024, "R", 100, 0.0)]  # no prior PA at all
    pa_df = pa_table(rows)
    league = c1.league_average(c1.trailing_window(pa_df, 2024))
    for variant in ("raw", "bucketed"):
        preds = c1.predict(pa_df, 2024, variant=variant).set_index(["batter", "p_throws"])
        assert preds.loc[(99, "L"), "pred_woba"] == pytest.approx(league["L"])
        assert preds.loc[(99, "R"), "pred_woba"] == pytest.approx(league["R"])
    assert league["L"] != pytest.approx(league["R"]), "fallback must be side-specific"


# ---- gate: output contract matches what claim1_eval consumes ----

def test_prediction_frame_contract():
    preds = c1.predict(small_sample_case(), eval_season=2024, variant="bucketed")
    assert list(preds.columns) == c1.KEY + ["pred_woba"]
    assert (preds["season"] == 2024).all()
    assert not preds.duplicated(c1.KEY).any()
    assert np.isfinite(preds["pred_woba"]).all()


def test_unknown_variant_fails_loud():
    with pytest.raises(AssertionError):
        c1.predict(small_sample_case(), 2024, variant="magic")
