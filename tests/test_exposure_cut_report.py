"""
Unit tests for the exposure-cut report's pure functions, on small synthetic frames
(no real parquet -- matches tests/test_stabilization.py's pattern). Confirms the
correlation/ratio/population logic, not the claim-1 machinery it reuses (that is
already covered by test_claim1_eval.py / test_stabilization.py).
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis import exposure_cut_report as report


def synthetic_labeled(n_hitters=200, seed=0):
    """batter x pitches vs LHP, with a per-hitter swing rate so hitter_stats has signal."""
    rng = np.random.default_rng(seed)
    rows = []
    for batter in range(n_hitters):
        n_pitches = rng.integers(20, 400)
        talent = rng.normal(0.45, 0.08)
        swings = rng.random(n_pitches) < np.clip(talent, 0.05, 0.95)
        rows.extend({"batter": batter, "season": 2020, "p_throws": "L", "swing": int(s)}
                    for s in swings)
    return pd.DataFrame(rows)


def synthetic_pa(labeled, pitches_per_pa=3.9, seed=0):
    """PA table with roughly pitches/pitches_per_pa completed PA per hitter vs LHP."""
    rng = np.random.default_rng(seed)
    counts = labeled.groupby("batter").size()
    rows = []
    for batter, n_pitches in counts.items():
        n_pa = max(1, round(n_pitches / pitches_per_pa))
        rows.extend({"batter": batter, "season": 2020, "p_throws": "L",
                     "woba_points": rng.normal(0.3, 0.2), "in_denominator": True}
                    for _ in range(n_pa))
    return pd.DataFrame(rows)


def test_pitch_pa_correlation_recovers_known_ratio():
    labeled = synthetic_labeled()
    pa_df = synthetic_pa(labeled, pitches_per_pa=3.9)
    r, ratio, n_matched = report.pitch_pa_correlation(labeled, pa_df, max_train_season=2023)
    assert n_matched == 200
    assert r > 0.9  # PA count is pitches/3.9 rounded -- near-perfect correlation by construction
    assert ratio == pytest.approx(3.9, rel=0.1)


def test_pitch_unit_stabilization_returns_finite_point_with_signal():
    labeled = synthetic_labeled(n_hitters=300, seed=1)
    point, lo, hi, n_hitters = report.pitch_unit_stabilization(labeled, max_train_season=2023, seed=1)
    assert n_hitters == 300
    assert np.isfinite(point) and point > 0
    assert lo <= point <= hi or not np.isfinite(hi)


def test_population_table_counts_and_bands_by_prior_pa():
    # 3 hitters: 0 prior PA (low), 200 prior PA (medium), 500 prior PA (high) vs LHP
    pa_rows = []
    at_bat = 0
    for batter, prior_pa in [(1, 0), (2, 200), (3, 500)]:
        for season, pa_count in [(2019, prior_pa), (2020, 10)]:  # 2020 = eval season
            for _ in range(pa_count if pa_count else 1):
                at_bat += 1
                pa_rows.append({"batter": batter, "season": season, "p_throws": "L",
                                "woba_points": 0.3, "in_denominator": True,
                                "game_pk": at_bat, "at_bat_number": 1, "pitcher": 999})
    pa_df = pd.DataFrame(pa_rows)
    # give the zero-prior-PA hitter no 2019 rows at all (true cold start)
    pa_df = pa_df[~((pa_df["batter"] == 1) & (pa_df["season"] == 2019))]

    result = report.population_table(pa_df, eval_season=2020, boundaries=(113, 452))
    assert result["n_hitters"] == 3
    assert result["by_stratum"] == {"low": 1, "medium": 1, "high": 1}
    assert result["n_above_low_cut"] == 2
