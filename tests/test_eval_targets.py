"""
Unit tests for the wOBA eval-target build. Synthetic fixtures only, no snapshot
dependency. The core gate is a hand-computed wOBA: the code must reproduce the
FanGraphs formula exactly, not a plausible neighbor of it.
"""

import pandas as pd
import pytest

from src.data import eval_targets as et

# a small, round weight set so the hand-computed expectation is transparent
WEIGHTS = {"2020": {"wBB": 0.7, "wHBP": 0.7, "w1B": 0.9, "w2B": 1.25,
                    "w3B": 1.6, "wHR": 2.0, "league_woba": 0.320}}


def make_pa(events, season=2020, batter=1, p_throws="R"):
    return {"game_pk": 1, "at_bat_number": 1, "batter": batter, "pitcher": 9,
            "p_throws": p_throws, "stand": "L", "events": events, "season": season}


def frame(events_list, **kw):
    return pd.DataFrame([make_pa(e, **kw) for e in events_list])


def prepared(events_list, **kw):
    return et.add_woba_points(et.categorize(frame(events_list, **kw)), WEIGHTS)


def test_woba_matches_hand_computation():
    # numerator = w1B + w2B + wBB(uBB) + wHBP = 0.9 + 1.25 + 0.7 + 0.7 = 3.55
    # denominator = single, double, uBB, HBP, SF, strikeout(OUT) = 6
    #   (intent_walk, sac_bunt, catcher_interf excluded)
    events = ["single", "double", "walk", "intent_walk", "hit_by_pitch",
              "sac_fly", "sac_bunt", "strikeout", "catcher_interf"]
    agg = et.aggregate(prepared(events))
    assert len(agg) == 1
    assert agg.loc[0, "denominator"] == 6
    assert agg.loc[0, "woba"] == pytest.approx(3.55 / 6, rel=1e-9)


def test_unknown_event_raises():
    with pytest.raises(ValueError, match="unknown events"):
        et.categorize(frame(["single", "made_up_event"]))


def test_non_batting_events_are_not_completed_categories():
    # non-batting terminal rows must be dropped upstream, never mapped
    assert not (et.NON_BATTING_EVENTS & set(et.EVENT_TO_CATEGORY))
    for event in et.NON_BATTING_EVENTS:
        with pytest.raises(ValueError):
            et.categorize(frame([event]))


def test_intentional_walk_excluded_from_numerator_and_denominator():
    df = prepared(["intent_walk"])
    assert df.loc[0, "woba_points"] == 0.0
    assert bool(df.loc[0, "in_denominator"]) is False


def test_unintentional_walk_credited_and_in_denominator():
    df = prepared(["walk"])
    assert df.loc[0, "woba_points"] == pytest.approx(0.7)
    assert bool(df.loc[0, "in_denominator"]) is True


def test_sac_fly_zero_points_but_in_denominator():
    df = prepared(["sac_fly"])
    assert df.loc[0, "woba_points"] == 0.0
    assert bool(df.loc[0, "in_denominator"]) is True


def test_sac_bunt_and_interference_excluded_from_denominator():
    df = prepared(["sac_bunt", "catcher_interf"])
    assert (df["woba_points"] == 0.0).all()
    assert (~df["in_denominator"]).all()


def test_home_run_uses_season_weight():
    df = prepared(["home_run"])
    assert df.loc[0, "woba_points"] == pytest.approx(2.0)


def test_aggregate_splits_by_pitcher_hand():
    rows = frame(["home_run"], p_throws="R")
    rows = pd.concat([rows, frame(["strikeout"], p_throws="L")], ignore_index=True)
    agg = et.aggregate(et.add_woba_points(et.categorize(rows), WEIGHTS))
    by_hand = dict(zip(agg["p_throws"], agg["woba"]))
    assert by_hand["R"] == pytest.approx(2.0)   # lone HR / denom 1
    assert by_hand["L"] == pytest.approx(0.0)   # lone strikeout / denom 1


# ---- gates: pitcher-batter identification (hitter-talent quantities only) ----

def _pa_rows(rows):
    """(batter, pitcher, season, n) -> PA-level frame with unique PA keys."""
    records, pa_id = [], 0
    for batter, pitcher, season, n in rows:
        for _ in range(n):
            pa_id += 1
            records.append({"batter": batter, "pitcher": pitcher, "season": season,
                            "game_pk": pa_id, "at_bat_number": 1,
                            "woba_points": 0.3, "in_denominator": True})
    return pd.DataFrame(records)


def test_pitcher_taking_his_own_at_bats_is_excluded():
    """Faces 300 batters, takes 12 PA -> an NL pitcher hitting. Not a hitter."""
    df = _pa_rows([(600, 500, 2019, 300), (500, 998, 2019, 12)])
    assert (2019, 500) in et.primarily_pitchers(df)
    assert 500 not in set(et.drop_pitcher_batters(df)["batter"])


def test_two_way_player_is_retained_as_a_hitter():
    """Faces 300 batters AND takes 500 PA -> Ohtani. Both conditions required."""
    df = _pa_rows([(700, 999, 2021, 500), (800, 700, 2021, 300)])
    assert (2021, 700) not in et.primarily_pitchers(df)
    assert 700 in set(et.drop_pitcher_batters(df)["batter"])


def test_position_player_who_mopped_up_is_retained():
    """Faces 6 batters in a blowout, takes 500 PA -> a hitter, not a pitcher."""
    df = _pa_rows([(750, 999, 2019, 500), (860, 750, 2019, 6)])
    assert (2019, 750) not in et.primarily_pitchers(df)
    assert 750 in set(et.drop_pitcher_batters(df)["batter"])


def test_exclusion_is_per_season_not_career():
    """A player who pitches one season and hits the next is judged season by season."""
    df = _pa_rows([(600, 500, 2019, 300), (500, 998, 2019, 12),
                   (500, 998, 2021, 400)])
    excluded = et.primarily_pitchers(df)
    assert (2019, 500) in excluded and (2021, 500) not in excluded


# --- the second target: xwOBA (D5-R8) ------------------------------------------------

def with_xwoba(events_list, xwoba_values, **kw):
    """A prepared frame carrying Statcast's xwOBA field, one value per event."""
    df = frame(events_list, **kw)
    df[et.XWOBA_FIELD] = xwoba_values
    return et.add_woba_points(et.categorize(df), WEIGHTS)


def test_xwoba_field_is_dropped_and_replaced_by_its_two_columns():
    df = with_xwoba(["single", "strikeout"], [0.55, 0.0])
    assert et.XWOBA_FIELD not in df.columns, "the raw field must not survive into the table"
    assert list(df["xwoba_points"]) == [0.55, 0.0]
    assert list(df["in_xwoba_denominator"]) == [True, True]


def test_an_unestimated_batted_ball_leaves_only_the_xwoba_denominator():
    """
    Statcast fails to estimate a small share of batted balls. Those plate appearances stay in
    the wOBA denominator -- they really happened -- and leave the xwOBA one. Imputing them
    would violate the Phase B missingness rule.
    """
    df = with_xwoba(["single", "double"], [0.55, None])
    assert list(df["in_denominator"]) == [True, True]
    assert list(df["in_xwoba_denominator"]) == [True, False]

    agg = et.aggregate(df)
    assert agg.loc[0, "denominator"] == 2
    assert agg.loc[0, "xwoba_denominator"] == 1
    assert agg.loc[0, "xwoba"] == pytest.approx(0.55)


def test_a_group_with_nothing_estimated_gets_a_null_xwoba_not_a_zero():
    """A zero would read as a hitter who produced nothing, which is a different claim."""
    df = with_xwoba(["double"], [None])
    agg = et.aggregate(df)
    assert agg.loc[0, "xwoba_denominator"] == 0
    assert pd.isna(agg.loc[0, "xwoba"])


def test_non_denominator_categories_are_out_of_both_denominators():
    df = with_xwoba(["intent_walk", "sac_bunt", "catcher_interf"], [None, None, None])
    assert not df["in_denominator"].any()
    assert not df["in_xwoba_denominator"].any()


def test_xwoba_coverage_counts_what_the_second_target_gives_up():
    df = with_xwoba(["single", "double", "triple", "intent_walk"], [0.5, None, 0.9, None])
    report = et.xwoba_coverage(df)["2020"]
    assert report["denominator"] == 3          # intent_walk is outside both
    assert report["xwoba_denominator"] == 2
    assert report["dropped"] == 1
    assert report["coverage"] == pytest.approx(2 / 3, abs=1e-5)


def test_aggregate_still_works_without_the_second_target():
    """Frames built before D5-R8, and every hand-built test frame, must keep scoring."""
    agg = et.aggregate(prepared(["single", "strikeout"]))
    assert "xwoba" not in agg.columns
    assert agg.loc[0, "denominator"] == 2
