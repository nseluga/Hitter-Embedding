"""Unit tests for the Statcast cleaning pipeline. Synthetic fixtures only, no snapshot dependency."""

import pandas as pd
import pytest

from src.data import clean


def make_pitch(**overrides):
    """A single valid regular-season pitch row that survives every filter by default."""
    row = {
        "game_type": "R", "game_pk": 1, "at_bat_number": 1, "pitch_number": 1,
        "game_date": "2024-04-01", "season": 2024, "batter": 100, "pitcher": 200,
        "description": "ball", "des": "", "events": None,
        "pitch_type": "FF", "release_speed": 93.0, "plate_x": 0.0, "plate_z": 2.5,
        "pfx_x": 0.5, "pfx_z": 1.2,
        "release_spin_rate": 2200.0, "spin_axis": 200.0, "effective_speed": 93.5,
        "release_extension": 6.5, "release_pos_x": -1.0, "release_pos_y": 54.0, "release_pos_z": 6.0,
    }
    row.update(overrides)
    return row


def frame(rows):
    return pd.DataFrame(rows)


def test_filter_regular_season_keeps_only_R():
    df = frame([make_pitch(game_type="R"), make_pitch(game_type="S"), make_pitch(game_type="D")])
    assert set(clean.filter_regular_season(df)["game_type"]) == {"R"}


def test_drop_deprecated_removes_dead_columns_only():
    df = frame([make_pitch()])
    df["spin_dir"] = None
    df["age_bat_legacy"] = 24
    result = clean.drop_deprecated_columns(df)
    assert "spin_dir" not in result.columns
    assert "age_bat_legacy" not in result.columns
    assert "release_speed" in result.columns


def test_deduplicate_removes_repeated_pitch_key():
    df = frame([make_pitch(), make_pitch(), make_pitch(pitch_number=2)])
    result = clean.deduplicate(df)
    assert len(result) == 2


def test_position_player_pitcher_removed_real_pitcher_kept(monkeypatch):
    monkeypatch.setattr(clean, "BATTER_PA_THRESHOLD", 3)
    monkeypatch.setattr(clean, "POSITION_PLAYER_MAX_BATTERS_FACED", 3)

    rows = []
    # position player 900 bats in 3 PAs (pitcher 200 faces those 3), then pitches 1 PA
    for at_bat in (1, 2, 3):
        rows.append(make_pitch(game_pk=10, at_bat_number=at_bat, batter=900, pitcher=200))
    rows.append(make_pitch(game_pk=11, at_bat_number=1, batter=101, pitcher=900))
    df = frame(rows)

    result = clean.filter_position_player_pitchers(df)
    assert 900 not in set(result["pitcher"])   # position-player pitching removed
    assert 200 in set(result["pitcher"])        # real pitcher (faced 3) kept


def test_filter_noncompetitive_rules():
    df = frame([
        make_pitch(description="ball"),                       # keep
        make_pitch(description="intent_ball"),                # keep (real take/walk)
        make_pitch(description="pitchout"),                   # drop
        make_pitch(description="automatic_ball"),             # drop
        make_pitch(description="foul_bunt"),                  # drop
        make_pitch(description="hit_into_play", des="Joe bunts into a double play"),  # drop (in-play bunt)
    ])
    result = clean.filter_noncompetitive(df)
    kept = set(result["description"])
    assert kept == {"ball", "intent_ball"}


def test_validate_core_context_drops_null_and_impossible():
    df = frame([
        make_pitch(),                       # keep
        make_pitch(pitch_type=None),        # drop: null core
        make_pitch(plate_x=10.0),           # drop: impossible location
        make_pitch(release_speed=15.0),     # drop: impossible velocity
        make_pitch(plate_z=-50.0),          # drop: impossible height
    ])
    result = clean.validate_core_context(df)
    assert len(result) == 1


def test_missingness_indicator_added_and_field_kept():
    df = frame([make_pitch(release_spin_rate=None), make_pitch(release_spin_rate=2100.0)])
    result = clean.add_missingness_indicators(df)
    assert "release_spin_rate_missing" in result.columns
    assert result["release_spin_rate_missing"].tolist() == [True, False]
    assert "release_spin_rate" in result.columns   # field retained, not dropped


def test_sort_pitches_orders_by_date_then_key():
    df = frame([
        make_pitch(game_date="2024-04-02", game_pk=5, at_bat_number=1, pitch_number=1),
        make_pitch(game_date="2024-04-01", game_pk=9, at_bat_number=2, pitch_number=3),
    ])
    result = clean.sort_pitches(df)
    assert result["game_date"].tolist() == ["2024-04-01", "2024-04-02"]


def test_clean_report_reconciles(tmp_path):
    snapshot = tmp_path / "snapshot_test"
    snapshot.mkdir()
    rows = [make_pitch(pitch_number=n) for n in range(1, 6)] + [make_pitch(game_type="S")]
    frame(rows).to_parquet(snapshot / "season=2024.parquet", index=False)

    table, report = clean.clean(snapshot)
    stages = dict(report)
    assert stages["regular_season"] == 5          # the spring-training row dropped
    assert stages["final"] == len(table)
    counts = [rows for _, rows in report]
    assert counts == sorted(counts, reverse=True)  # attrition is monotone non-increasing
