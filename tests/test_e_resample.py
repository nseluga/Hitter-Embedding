"""
E.7's draw audit (src/analysis/e_resample.py).

The module's whole output is a difference of three means that all live in [0, 1], so every
failure mode here returns a publishable number: a mis-normalised pitcher weight, an `exact`
reference built from the wrong pitcher's cell, or a `league` row that quietly inherits the
pool weighting instead of staying unweighted. The fixture below is homogeneous BY
CONSTRUCTION -- every pitcher throws one location, so the sampled draw and the pitcher's own
cell must agree exactly -- which makes each of the three references hand-computable.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis import e_resample
from src.model import query, query_tables as qt

# grid bins the fixture's two locations land in: plate_x -1.0 -> floor((-1.0+2.5)/0.1) = 15,
# plate_x +1.0 -> floor(( 1.0+2.5)/0.1) = 35, plate_z 2.0 -> floor(2.0/0.1) = 20
BIN_A, BIN_B, BIN_Z = 15, 35, 20
TRAIN_SEASONS = [2021]


def _surfaces():
    """P(ball, called strike, hbp | take) that depends only on which of the two bins it is."""
    out = {}
    for stand in ("L", "R"):
        for throws in ("L", "R"):
            surface = np.zeros((50, 50, 3), dtype="float64")
            surface[..., 0], surface[..., 1] = 0.5, 0.5
            surface[BIN_A, BIN_Z] = (0.80, 0.15, 0.05)
            surface[BIN_B, BIN_Z] = (0.40, 0.55, 0.05)
            out[(stand, throws)] = surface
    return out


def _frame(rows_per_cell=6):
    """
    Two right-handed pitchers, each with `rows_per_cell` rows in every (stand, count) cell.
    Pitcher 100 throws only at bin A and is in the zone; 200 only at bin B and is not.
    """
    records = []
    pk = 0
    for pitcher, plate_x, zone in ((100, -1.0, 5), (200, 1.0, 12)):
        for stand in ("L", "R"):
            for balls, strikes in query.COUNT_STATES:
                for _ in range(rows_per_cell):
                    pk += 1
                    records.append({
                        "batter": 1, "season": TRAIN_SEASONS[0], "pitcher": pitcher,
                        "description": "ball", "zone": zone, "balls": balls,
                        "strikes": strikes, "stand": stand, "p_throws": "R",
                        "plate_x": plate_x, "plate_z": 2.0,
                        "game_pk": pk, "at_bat_number": 1,
                    })
    return pd.DataFrame(records)


def _tables(frame, weights=(0.75, 0.25)):
    train_mask = np.isin(frame["season"].to_numpy(), TRAIN_SEASONS)
    repertoire = qt.build_repertoire(frame, train_mask)
    slots = np.searchsorted(repertoire.pitcher_ids, np.array([100, 200]))
    return {"take": _surfaces(), "repertoire": repertoire,
            "pitcher_weights": {"R": (slots, np.array(weights, dtype="float64"))},
            "_train_seasons": TRAIN_SEASONS}


# ---------------------------------------------------------------- _take_means

def test_take_means_is_the_plain_row_mean_of_the_surface():
    """Half the rows at bin A (P(ball)=0.80) and half at bin B (0.40) -> 0.60, and 50% in zone."""
    frame = _frame()
    tables = _tables(frame)
    rows = np.concatenate([np.flatnonzero(frame["pitcher"].to_numpy() == 100)[:4],
                           np.flatnonzero(frame["pitcher"].to_numpy() == 200)[:4]])
    out = e_resample._take_means(tables, frame, rows, None)
    assert out["ball"] == pytest.approx(0.60)             # (0.80 + 0.40) / 2
    assert out["called_strike"] == pytest.approx(0.35)    # (0.15 + 0.55) / 2
    assert out["hbp"] == pytest.approx(0.05)
    assert out["in_zone"] == pytest.approx(0.5)           # zone 5 vs zone 12


def test_take_means_of_an_empty_selection_is_nan_not_zero():
    """An empty cell has no rate; reporting 0.0 would drag a weighted mean toward zero."""
    frame = _frame()
    out = e_resample._take_means(_tables(frame), frame, np.array([], dtype="int64"), None)
    assert set(out) == {"ball", "called_strike", "hbp", "in_zone"}
    assert all(np.isnan(value) for value in out.values())


def test_take_means_renormalises_after_a_count_offset():
    """
    Bin A is (0.80, 0.15, 0.05). Doubling the ball channel gives (1.60, 0.15, 0.05), which
    sums to 1.80, so the renormalised ball rate is 1.60/1.80 = 0.888..., not 1.60.
    """
    frame = _frame()
    rows = np.flatnonzero(frame["pitcher"].to_numpy() == 100)[:3]
    offsets = np.ones((4, 3, 3), dtype="float64")
    offsets[..., 0] = 2.0
    out = e_resample._take_means(_tables(frame), frame, rows, offsets)
    assert out["ball"] == pytest.approx(1.60 / 1.80)
    assert out["ball"] + out["called_strike"] + out["hbp"] == pytest.approx(1.0)


# ---------------------------------------------------------------- resampler_audit

def test_audit_weights_the_pool_and_leaves_league_unweighted():
    """
    Both pitchers fill every cell with exactly 6 identical rows, so the draw can only return
    those rows: sampled must equal exact to the bit, and the draw gap must be exactly zero.
    exact is the 0.75/0.25 batters-faced mean: 0.75*0.80 + 0.25*0.40 = 0.70.
    league is the raw row mean over the train window, 6 rows each: (0.80 + 0.40)/2 = 0.60.
    """
    frame = _frame()
    table = e_resample.resampler_audit(_tables(frame), frame, seed=0)
    assert len(table) == 2 * len(query.COUNT_STATES)      # one hand x two stands x 12 counts
    assert np.allclose(table["sampled_ball"], 0.70)
    assert np.allclose(table["exact_ball"], 0.70)
    assert np.allclose(table["league_ball"], 0.60)
    assert np.allclose(table["draw_gap_ball"], 0.0), "a homogeneous cell cannot produce a draw gap"
    assert np.allclose(table["pool_gap_ball"], 0.10)      # 0.70 - 0.60


def test_audit_in_zone_share_follows_the_same_two_weightings():
    """Pitcher 100 is always in the zone and 200 never is, so in_zone IS the weight itself."""
    frame = _frame()
    table = e_resample.resampler_audit(_tables(frame, weights=(0.9, 0.1)), frame, seed=0)
    assert np.allclose(table["sampled_in_zone"], 0.9)
    assert np.allclose(table["exact_in_zone"], 0.9)
    assert np.allclose(table["league_in_zone"], 0.5)      # 6 in-zone rows against 6 out
    assert np.allclose(table["n_league_pitches"], 12)     # per (stand, count): 6 + 6
    assert np.allclose(table["n_exact_pitches"], 12)


def test_audit_counts_the_backoff_levels_it_actually_used():
    """
    Every cell holds exactly `DEFAULT_N_PITCHES` rows, so `Repertoire.sample` returns the
    exact cell every time: 2 pitchers x 12 counts = 24 exact draws per (hand, stand) block
    and nothing at any wider level.
    """
    frame = _frame(rows_per_cell=query.DEFAULT_N_PITCHES)
    table = e_resample.resampler_audit(_tables(frame), frame, seed=0)
    assert (table["backoff_exact"] == 2 * len(query.COUNT_STATES)).all()
    assert (table[["backoff_strikes", "backoff_stand", "backoff_empty"]] == 0).all().all()
    assert (table["n_pitchers"] == 2).all()
