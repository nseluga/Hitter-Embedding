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
