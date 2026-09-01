"""
E.6's data-free arithmetic (src/analysis/model_evaluation_swing.py).

Two things in this module can be wrong while the CSV still looks like a calibration table:
the per-cell grouping can pool rows it should have split (a pooled mean is always a plausible
swing rate), and the ensemble can average LOGITS instead of PROBABILITIES, which is the
convention `query.expected_woba` fixes and which moves the number in the same direction a
miscalibrated head would. Both are pinned here on hand-computable fixtures.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.analysis import model_evaluation_swing


def _pitches(swing, p_swing, balls, strikes):
    return pd.DataFrame({"swing": np.asarray(swing, dtype="float64"),
                         "p_swing": np.asarray(p_swing, dtype="float64"),
                         "balls": balls, "strikes": strikes})


# ---------------------------------------------------------------- calibration_table

def test_calibration_gap_and_relative_gap_are_closed_form():
    """One cell, four pitches: observed 0.5, predicted 0.6, so gap +0.1 and relative +0.2."""
    frame = _pitches(swing=[1, 1, 0, 0], p_swing=[0.7, 0.7, 0.5, 0.5],
                     balls=0, strikes=0)
    out = model_evaluation_swing.calibration_table(frame, ("balls", "strikes"), "count")
    assert len(out) == 1
    row = out.iloc[0]
    assert row["n_pitches"] == 4
    assert row["observed"] == pytest.approx(0.5)          # (1+1+0+0)/4
    assert row["predicted"] == pytest.approx(0.6)         # (0.7+0.7+0.5+0.5)/4
    assert row["gap"] == pytest.approx(0.1)               # 0.6 - 0.5
    assert row["relative_gap"] == pytest.approx(0.2)      # 0.1 / 0.5
    assert row["grouping"] == "count"


def test_calibration_splits_cells_rather_than_pooling_them():
    """
    The two counts have opposite gaps; pooled they cancel exactly. A grouping bug that
    collapsed them would report a perfectly calibrated head over a table with no cell in it.
    """
    frame = pd.concat([
        _pitches(swing=[1, 0], p_swing=[1.0, 0.4], balls=0, strikes=0),   # obs .5 pred .7
        _pitches(swing=[1, 0], p_swing=[0.6, 0.0], balls=3, strikes=0),   # obs .5 pred .3
    ], ignore_index=True)
    out = model_evaluation_swing.calibration_table(frame, ("balls", "strikes"), "count").set_index("balls")
    assert len(out) == 2
    assert out.loc[0, "gap"] == pytest.approx(0.2)        # 0.7 - 0.5
    assert out.loc[3, "gap"] == pytest.approx(-0.2)       # 0.3 - 0.5
    assert out.loc[0, "n_pitches"] == 2 and out.loc[3, "n_pitches"] == 2
    pooled = model_evaluation_swing.calibration_table(frame.assign(all="all"), ("all",), "overall")
    assert pooled.iloc[0]["gap"] == pytest.approx(0.0), "the fixture's cells must cancel pooled"


def test_calibration_cells_are_unweighted_within_the_cell():
    """
    Each cell's mean is over its own rows, so an 8-row cell and a 2-row cell keep their own
    numbers -- the row count rides along in n_pitches for the caller to weight with later.
    """
    frame = pd.concat([
        _pitches(swing=[1] * 8, p_swing=[0.5] * 8, balls=0, strikes=0),
        _pitches(swing=[0, 0], p_swing=[0.1, 0.3], balls=1, strikes=0),
    ], ignore_index=True)
    out = model_evaluation_swing.calibration_table(frame, ("balls", "strikes"), "count").set_index("balls")
    assert out.loc[0, "observed"] == pytest.approx(1.0)
    assert out.loc[0, "predicted"] == pytest.approx(0.5)
    assert out.loc[0, "relative_gap"] == pytest.approx(-0.5)   # (0.5 - 1.0) / 1.0
    assert out.loc[1, "observed"] == 0.0
    assert out.loc[1, "predicted"] == pytest.approx(0.2)       # (0.1 + 0.3)/2
    assert np.isinf(out.loc[1, "relative_gap"]), "a zero observed rate divides, it does not hide"


# ---------------------------------------------------------------- ensemble_swing

class _ConstantSwing:
    """A stand-in with `_trunk`'s attribute surface whose swing head returns a fixed logit."""

    interaction = None

    def __init__(self, logit):
        self.logit = float(logit)

    def embedding(self, hitter):
        return torch.zeros(len(hitter), 1)

    def context_tower(self, context):
        return torch.zeros(len(context), 1)

    def trunk(self, stacked):
        return stacked[:, :1]

    def head_swing(self, trunk):
        return torch.full_like(trunk, self.logit)


def test_ensemble_averages_probabilities_not_logits():
    """
    sigmoid(0) = 0.5 and sigmoid(ln 3) = 0.75, so the probability mean is 0.625.
    Averaging the logits first gives sigmoid((0 + 1.0986)/2) = sigmoid(0.5493) = 0.6339,
    which is the convention `expected_woba` forbids.
    """
    hitter = np.zeros(5, dtype="int64")
    context = np.zeros((5, 3), dtype="float32")
    models = [_ConstantSwing(0.0), _ConstantSwing(np.log(3.0))]
    p = model_evaluation_swing.ensemble_swing(models, hitter, context)
    assert p.shape == (5,)
    assert np.allclose(p, 0.625)
    assert not np.allclose(p, 0.6339745962155614), "logits were averaged instead of probabilities"


def test_ensemble_batching_does_not_change_the_answer():
    """A batch boundary must not re-weight anything: 7 rows in one pass and in chunks of 3."""
    hitter = np.arange(7, dtype="int64")
    context = np.zeros((7, 2), dtype="float32")
    models = [_ConstantSwing(0.0), _ConstantSwing(np.log(3.0))]
    whole = model_evaluation_swing.ensemble_swing(models, hitter, context, batch=64)
    chunked = model_evaluation_swing.ensemble_swing(models, hitter, context, batch=3)
    assert np.array_equal(whole, chunked)
    assert np.allclose(whole, 0.625)   # same two constants as above
