"""
E.12's one data helper (src/analysis/e_level_confound.py).

`arm_mean` decides where an arm's output distribution sits, and every correlation and every
scale check in the module is built on it. Two ways it can be quietly wrong: it can weight
(the docstring promises it does not -- a PA-weighted mean answers a different question and
would move each arm by a different amount), and it can read the wrong arm's table, which
produces a perfectly plausible number attached to the wrong row.
"""

import pandas as pd
import pytest

from src.analysis import e_level_confound


def _write_arm(directory, arm, pred, pa=None):
    frame = pd.DataFrame({"batter": range(len(pred)), "season": 2024, "p_throws": "R",
                          "pred_woba": pred})
    if pa is not None:
        frame["pa"] = pa
    frame.to_csv(directory / f"d5_predictions_d10_{arm}.csv", index=False)


@pytest.fixture
def phase_d(tmp_path, monkeypatch):
    monkeypatch.setattr(e_level_confound, "PHASE_D", tmp_path)
    return tmp_path


def test_arm_mean_is_the_unweighted_mean_and_the_row_count(phase_d):
    """(0.300 + 0.320 + 0.340 + 0.360) / 4 = 0.330, over 4 rows."""
    _write_arm(phase_d, "baseline", [0.300, 0.320, 0.340, 0.360])
    mean, n = e_level_confound.arm_mean("baseline")
    assert mean == pytest.approx(0.330)
    assert n == 4


def test_arm_mean_ignores_playing_time(phase_d):
    """
    The same four predictions with 600 PA on the highest one. Unweighted the answer is still
    0.330; a PA-weighted mean would be 0.3557, and the module's question is where the arm's
    OUTPUT sits, not where the population average does.
    """
    _write_arm(phase_d, "weighted", [0.300, 0.320, 0.340, 0.360], pa=[10, 10, 10, 600])
    mean, _ = e_level_confound.arm_mean("weighted")
    assert mean == pytest.approx(0.330)
    # (10*.300 + 10*.320 + 10*.340 + 600*.360) / 630 = 224.4 / 630 = 0.356190...
    assert mean != pytest.approx(224.4 / 630)


def test_each_arm_reads_its_own_table(phase_d):
    """A path that fell through to a shared file would make every arm's level identical."""
    _write_arm(phase_d, "baseline", [0.300, 0.300])
    _write_arm(phase_d, "invfreq", [0.250, 0.250, 0.250])
    assert e_level_confound.arm_mean("baseline") == (pytest.approx(0.300), 2)
    assert e_level_confound.arm_mean("invfreq") == (pytest.approx(0.250), 3)


def test_a_table_without_predictions_fails_loud(phase_d):
    """A renamed column must raise, not return the mean of whatever else is in the file."""
    pd.DataFrame({"batter": [1, 2], "woba": [0.3, 0.4]}).to_csv(
        phase_d / "d5_predictions_d10_broken.csv", index=False)
    with pytest.raises(AssertionError):
        e_level_confound.arm_mean("broken")
