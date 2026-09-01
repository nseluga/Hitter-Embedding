"""
E.14 unit tests — pure logic only (interval construction, coverage arithmetic, the
binomial interval). Nothing here reads a checkpoint or the PA table; the data-facing
paths are guarded by the module's own assertions.
"""

import numpy as np
import pandas as pd

from src.analysis import e_probe_coverage as e14


def test_covered_widens_with_target_noise():
    """
    The fair interval must be a superset of the raw one: a group missed by the seed
    spread alone but inside the noise-convolved interval is the whole point of report (b).
    Constructed so the residual (0.05) sits outside 1*seed_sd but inside sqrt(sd^2+var).
    """
    frame = pd.DataFrame({"woba": [0.35], "pred_woba": [0.30],
                          "seed_sd": [0.01], "noise_var": [0.06 ** 2]})
    assert not e14.covered(frame, 1.0, widen=False)[0]
    assert e14.covered(frame, 1.0, widen=True)[0]


def test_covered_is_the_symmetric_z_interval():
    """Cover is |obs - mean| <= z*sd, symmetric about the mean on both sides."""
    frame = pd.DataFrame({
        "woba": [0.319, 0.281, 0.321, 0.279],
        "pred_woba": [0.300] * 4,
        "seed_sd": [0.010] * 4,
        "noise_var": [0.0] * 4,
    })
    assert list(e14.covered(frame, 2.0, widen=False)) == [True, True, False, False]


def test_coverage_rows_arithmetic_and_shape():
    """
    Empirical rate, gap against nominal, and half width are read off the same rows the
    reliability diagram plots. Ten groups, all exactly at the mean, so every interval
    covers and the empirical rate is 1.0 at every nominal level.
    """
    frame = pd.DataFrame({"woba": [0.3] * 10, "pred_woba": [0.3] * 10,
                          "seed_sd": [0.02] * 10, "noise_var": [0.0] * 10})
    rows = e14.coverage_rows(frame, "unit")
    assert len(rows) == 2 * len(e14.Z_BY_LEVEL)
    for row in rows:
        assert row["n_groups"] == 10
        assert row["empirical"] == 1.0
        assert np.isclose(row["gap"], 1.0 - row["nominal"])
        assert np.isclose(row["mean_half_width"], row["z"] * 0.02)


def test_wilson_interval_stays_in_the_unit_interval():
    """
    The reason Wilson is used rather than Wald: at an empirical rate of 1.0 the Wald
    bound is exactly 1.0 with zero width, and at 0.0 it leaves the interval. Wilson
    brackets the estimate from inside on both ends.
    """
    low, high = e14.wilson_interval(20, 20)
    assert 0.0 < low < 1.0 and high == 1.0
    low, high = e14.wilson_interval(0, 20)
    assert low == 0.0 and 0.0 < high < 1.0
    low, high = e14.wilson_interval(10, 20)
    assert low < 0.5 < high
