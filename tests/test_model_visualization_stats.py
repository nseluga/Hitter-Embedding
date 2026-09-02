"""
Verification gates for the Phase V observable-stat panel (docs/phase-v-spec.md §1).

Synthetic small frames, not the real 7.3M-row parquet: rates land in [0,1], chase
uses only zones 11-14, pull_rate's orientation flips with stand (spray is already
hand-mirrored upstream, so this is really a regression check on that reuse), seasons
outside 2015-2023 are dropped, and the real vocabulary join covers all 1,762 rows.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.analysis import model_visualization_stats as mvs


def synthetic_pitch_frame():
    """A tiny pitch table covering swings, takes, zones, and both pull directions."""
    rows = [
        # batter 1 (R), several takes and swings, one chase, one in-zone swing
        {"batter": 1, "stand": "R", "season": 2018, "swing": 0, "contact": pd.NA, "zone": 5, "ev": np.nan, "la": np.nan, "spray": np.nan, "bat_speed": np.nan},
        {"batter": 1, "stand": "R", "season": 2018, "swing": 1, "contact": 1, "zone": 5, "ev": 95.0, "la": 20, "spray": -20.0, "bat_speed": np.nan},
        {"batter": 1, "stand": "R", "season": 2019, "swing": 1, "contact": 0, "zone": 1, "ev": np.nan, "la": np.nan, "spray": np.nan, "bat_speed": np.nan},
        {"batter": 1, "stand": "R", "season": 2019, "swing": 1, "contact": 0, "zone": 12, "ev": np.nan, "la": np.nan, "spray": np.nan, "bat_speed": np.nan},
        # batter 2 (L), pulled ball
        {"batter": 2, "stand": "L", "season": 2020, "swing": 1, "contact": 1, "zone": 3, "ev": 100.0, "la": 10, "spray": 25.0, "bat_speed": 70.0, },
        {"batter": 2, "stand": "L", "season": 2020, "swing": 0, "contact": pd.NA, "zone": 3, "ev": np.nan, "la": np.nan, "spray": np.nan, "bat_speed": np.nan},
        # a row outside the training window, must be dropped by load_pitch_frame
        {"batter": 1, "stand": "R", "season": 2024, "swing": 1, "contact": 1, "zone": 5, "ev": 90.0, "la": 15, "spray": 5.0, "bat_speed": 71.0},
    ]
    return pd.DataFrame(rows)


def test_season_filter_drops_out_of_window_rows(tmp_path):
    frame = synthetic_pitch_frame()
    path = tmp_path / "pitches.parquet"
    frame.to_parquet(path)

    filtered = mvs.load_pitch_frame(str(path))
    assert filtered["season"].between(2015, 2023).all()
    assert 2024 not in set(filtered["season"])
    assert len(filtered) == len(frame) - 1


def test_rates_in_unit_interval_and_chase_uses_only_11_14():
    frame = synthetic_pitch_frame()
    frame = frame[frame["season"].between(2015, 2023)]
    stats = mvs.per_batter_pitch_stats(frame)

    rate_columns = ["swing_rate", "whiff_rate", "contact_rate", "chase_rate", "zone_swing_rate", "pull_rate"]
    for column in rate_columns:
        values = stats[column].dropna()
        assert ((values >= 0) & (values <= 1)).all(), f"{column} outside [0,1]: {values.tolist()}"

    # ground truth computed directly from the raw rows, independent of the module,
    # so this fails if chase_rate or zone_swing_rate leaks rows from the other zone set
    batter_1 = frame[frame["batter"] == 1]
    expected_chase = batter_1[batter_1["zone"].isin(mvs.CHASE_ZONES)]["swing"].mean()
    expected_in_zone = batter_1[batter_1["zone"].isin(mvs.IN_ZONE_ZONES)]["swing"].mean()
    result = stats.set_index("batter").loc[1]
    assert result["chase_rate"] == pytest.approx(expected_chase)
    assert result["zone_swing_rate"] == pytest.approx(expected_in_zone)
    # the two zone sets are disjoint, so a leak would make them equal on this data
    assert expected_chase != expected_in_zone


def test_pull_rate_orientation_flips_with_stand():
    # spray is already hand-mirrored (positive = pull) per src.data.labels; this pins
    # that the module reads the sign as-is rather than re-deriving or re-flipping it
    pulled = pd.DataFrame([
        {"batter": 9, "stand": "L", "season": 2020, "swing": 1, "contact": 1, "zone": 5,
         "ev": 100.0, "la": 10, "spray": 20.0, "bat_speed": np.nan},
    ])
    not_pulled = pd.DataFrame([
        {"batter": 9, "stand": "L", "season": 2020, "swing": 1, "contact": 1, "zone": 5,
         "ev": 100.0, "la": 10, "spray": -20.0, "bat_speed": np.nan},
    ])
    assert mvs.per_batter_pitch_stats(pulled).set_index("batter").loc[9, "pull_rate"] == 1.0
    assert mvs.per_batter_pitch_stats(not_pulled).set_index("batter").loc[9, "pull_rate"] == 0.0


def synthetic_pa_frame():
    """A tiny PA-level eval-target table: one batter faces both hands, unbalanced."""
    records = []
    pa_id = 0

    def add(batter, season, hand, n_pa, woba, stand):
        nonlocal pa_id
        for _ in range(n_pa):
            pa_id += 1
            records.append({
                "batter": batter, "season": season, "p_throws": hand, "stand": stand,
                "woba_points": woba, "in_denominator": True,
                "pitcher": 900001, "game_pk": pa_id, "at_bat_number": 1,
            })

    # batter 3, LHB: better vs RHP (0.400) than vs LHP (0.200) -> obs_platoon_diff positive
    add(3, 2019, "R", 20, 0.400, "L")
    add(3, 2019, "L", 5, 0.200, "L")
    return pd.DataFrame(records)


def test_obs_platoon_diff_sign_for_lhb():
    pa_df = synthetic_pa_frame()
    result = mvs.per_batter_exposure_and_platoon(pa_df).set_index("batter")
    assert result.loc[3, "obs_platoon_diff"] > 0
    assert result.loc[3, "prior_pa_L"] == pytest.approx(5.0)
    assert result.loc[3, "prior_pa_R"] == pytest.approx(20.0)
    assert result.loc[3, "n_pa_L"] == result.loc[3, "prior_pa_L"]
    assert result.loc[3, "stratum"] == "low"  # min(5, 20) = 5 < 113


def test_vocabulary_coverage_is_1762_rows():
    vocab = mvs.load_vocabulary()
    assert len(vocab) == 1762
    assert vocab["batter"].is_unique
    assert vocab["embedding_index"].min() >= 1
