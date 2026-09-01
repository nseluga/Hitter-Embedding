"""
E.9's pairing and scoring helpers (src/analysis/model_evaluation_take_mass.py).

The estimand is a PAIRED difference, and pairing is exactly the thing that fails silently:
if a pitcher missing an exact cell survives into the panel, his drawn side is compared
against a substitute cell and the gap is manufactured rather than measured. The other half
is `score_take_mass`, which de-duplicates pitch rows before the forward pass and then scatters
the answer back -- a wrong inverse map would still return a well-shaped array of plausible
take masses.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.analysis import model_evaluation_take_mass
from src.model import query, query_tables as qt

TRAIN_SEASONS = [2021]
N_COUNTS = len(query.COUNT_STATES)


def _frame(cells):
    """
    A pitch frame from {pitcher: {(stand, balls, strikes): n_rows}}; cells left out are empty.
    Every row is distinguishable by its own index, which is what the row-provenance test needs.
    """
    records = []
    pk = 0
    for pitcher, spec in cells.items():
        for (stand, balls, strikes), n_rows in spec.items():
            for _ in range(n_rows):
                pk += 1
                records.append({
                    "batter": 1, "season": TRAIN_SEASONS[0], "pitcher": pitcher,
                    "description": "ball", "zone": 5, "balls": balls, "strikes": strikes,
                    "stand": stand, "p_throws": "R", "plate_x": 0.0, "plate_z": 2.0,
                    "game_pk": pk, "at_bat_number": 1,
                })
    return pd.DataFrame(records)


def _full(pitcher, n_rows, stand="R"):
    return {pitcher: {(stand, b, s): n_rows for b, s in query.COUNT_STATES}}


def _repertoire(frame):
    return qt.build_repertoire(frame, np.isin(frame["season"].to_numpy(), TRAIN_SEASONS))


# ---------------------------------------------------------------- paired_rows

def test_paired_rows_shapes_are_six_drawn_and_n_exact_per_count():
    frame = _frame({**_full(100, 8), **_full(200, 8)})
    repertoire = _repertoire(frame)
    drawn, exact, keep = model_evaluation_take_mass.paired_rows(repertoire, [0, 1], 1, 5,
                                                 np.random.default_rng(0))
    assert keep.tolist() == [True, True]
    assert drawn.shape == (2, N_COUNTS, 6)       # the composition's 6 rows per cell
    assert exact.shape == (2, N_COUNTS, 5)       # n_exact reference draws per cell


def test_a_pitcher_missing_one_exact_cell_is_dropped_entirely():
    """
    Pitcher 200 has every cell but 3-2. Backoff would still hand him six DRAWN rows there,
    so nothing upstream would notice; the pairing rule is what removes him, and it removes
    all twelve of his counts, not just the hole.
    """
    thin = _full(200, 8)
    del thin[200][("R", 3, 2)]
    frame = _frame({**_full(100, 8), **thin})
    repertoire = _repertoire(frame)
    assert len(repertoire.rows(1, 1, 3, 2)) == 0
    assert len(repertoire.sample(1, 1, 3, 2, 6, np.random.default_rng(0))[0]) == 6

    drawn, exact, keep = model_evaluation_take_mass.paired_rows(repertoire, [0, 1], 1, 5,
                                                 np.random.default_rng(0))
    assert keep.tolist() == [True, False]
    assert drawn.shape == (1, N_COUNTS, 6) and exact.shape == (1, N_COUNTS, 5)


def test_exact_rows_come_only_from_that_pitchers_own_cell():
    """
    Pitcher 100 owns rows 0..95 and pitcher 200 rows 96..191 (8 per cell, 12 counts each).
    Every reference row for slot 0 must land in slot 0's own cell -- the point of the panel.
    """
    frame = _frame({**_full(100, 8), **_full(200, 8)})
    repertoire = _repertoire(frame)
    _, exact, _ = model_evaluation_take_mass.paired_rows(repertoire, [0, 1], 1, 5, np.random.default_rng(1))
    for position, (balls, strikes) in enumerate(query.COUNT_STATES):
        own = set(repertoire.rows(0, 1, balls, strikes).tolist())
        assert len(own) == 8
        assert set(exact[0, position].tolist()) <= own


def test_a_thin_cell_is_drawn_with_replacement_to_stay_rectangular():
    """One row in the cell and 5 reference draws asked for: all 5 must be that same row."""
    frame = _frame(_full(100, 1))
    repertoire = _repertoire(frame)
    _, exact, keep = model_evaluation_take_mass.paired_rows(repertoire, [0], 1, 5, np.random.default_rng(0))
    assert keep.tolist() == [True]
    assert exact.shape == (1, N_COUNTS, 5)
    for position, (balls, strikes) in enumerate(query.COUNT_STATES):
        only = repertoire.rows(0, 1, balls, strikes)
        assert len(only) == 1
        assert (exact[0, position] == only[0]).all()


# ---------------------------------------------------------------- score_take_mass

class _LogitIsHitterPlusContext:
    """`_trunk`'s attribute surface, arranged so the swing logit is hitter_row + context[0]."""

    interaction = None

    def embedding(self, hitter):
        return hitter.to(torch.float64).unsqueeze(1)

    def context_tower(self, context):
        return context.to(torch.float64)

    def trunk(self, stacked):
        return stacked.sum(dim=1, keepdim=True)

    def head_swing(self, trunk):
        return trunk


def test_take_mass_is_one_minus_the_swing_probability_per_row():
    """
    Context row 0 carries logit 0 -> p 0.5 -> take mass 0.5; row 1 carries ln(3) -> p 0.75
    -> take mass 0.25. The repeated pitch row must come back with the same number, and the
    output must keep the caller's (pitcher, count) shape.
    """
    context = np.array([[0.0], [np.log(3.0)]], dtype="float32")
    pitch_rows = np.array([[[0, 1], [1, 1]]])            # (1 pitcher, 2 counts, 2 pitches)
    out = model_evaluation_take_mass.score_take_mass([_LogitIsHitterPlusContext()], context, [0], pitch_rows)
    assert out.shape == pitch_rows.shape
    assert np.allclose(out, [[[0.5, 0.25], [0.25, 0.25]]])


def test_take_mass_averages_over_hitters_and_over_the_ensemble():
    """
    Two hitters shift the logit by their row id, so pitch row 0 (context 0) is scored at
    logit 0 and logit 1: (1 - 0.5 + 1 - 0.7310585786) / 2 = 0.3844707107.
    Two identical models must not move that -- probabilities are averaged, not summed.
    """
    context = np.zeros((1, 1), dtype="float32")
    pitch_rows = np.array([[[0]]])
    one_model = model_evaluation_take_mass.score_take_mass([_LogitIsHitterPlusContext()], context, [0, 1],
                                            pitch_rows)
    two_models = model_evaluation_take_mass.score_take_mass(
        [_LogitIsHitterPlusContext(), _LogitIsHitterPlusContext()], context, [0, 1], pitch_rows)
    assert one_model[0, 0, 0] == pytest.approx(0.3844707107, abs=1e-9)
    assert two_models[0, 0, 0] == pytest.approx(0.3844707107, abs=1e-9)


def test_take_mass_batching_does_not_change_the_answer():
    """Seven distinct pitch rows, scored whole and in chunks of three."""
    context = np.linspace(-1.0, 1.0, 7, dtype="float32").reshape(7, 1)
    pitch_rows = np.arange(7).reshape(1, 7, 1)
    whole = model_evaluation_take_mass.score_take_mass([_LogitIsHitterPlusContext()], context, [0],
                                        pitch_rows, batch=64)
    chunked = model_evaluation_take_mass.score_take_mass([_LogitIsHitterPlusContext()], context, [0],
                                          pitch_rows, batch=3)
    assert np.array_equal(whole, chunked)
    # the middle row has context 0 and hitter 0, so its take mass is exactly 1 - 0.5
    assert whole[0, 3, 0] == pytest.approx(0.5)
