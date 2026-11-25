import pandas as pd
import pytest

from src.analysis.surfaces_observed import agreement, observed_cells


def test_observed_delta_is_anchor_mean_minus_cell_league_mean():
    df = pd.DataFrame({"batter": [1, 1, 2, 2], "p_throws": ["L"] * 4,
                       "x_bin": [0] * 4, "z_bin": [0] * 4, "value": [0.6, 0.4, 0.2, 0.0]})
    out = observed_cells(df, anchors={1: "A"})
    assert out["n"].iat[0] == 2
    assert out["delta"].iat[0] == pytest.approx(0.5 - 0.3)


def test_agreement_drops_thin_cells():
    keys = {"anchor": [1, 1, 1], "p_throws": ["L"] * 3, "x_bin": [0, 1, 2], "z_bin": [0] * 3}
    model = pd.DataFrame({**keys, "delta": [0.1, 0.2, 0.3]})
    obs = pd.DataFrame({**keys, "delta": [0.1, 0.2, -9.0], "n": [30, 30, 5]})
    got = agreement(model, obs, min_bip=20)
    assert got["n_cells"] == 2 and got["r"] == pytest.approx(1.0)
