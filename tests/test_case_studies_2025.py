import numpy as np
import pandas as pd

from src.analysis.case_studies_2025 import grade, tally


def test_grade_picks_closer_estimator_and_noise_band():
    frame = pd.DataFrame({
        "batter": [1, 2], "delta_obs": [0.05, -0.02], "delta_pred": [0.04, 0.03],
        "delta_eb": [-0.01, -0.01], "denom_L": [100.0, 10000.0], "denom_R": [100.0, 10000.0],
        "weight": [1.0, 1.0]})
    g = grade(frame, woba_sd=0.5)
    assert list(g["closer"]) == ["model", "eb"]
    assert list(g["model_sign_right"]) == [True, False]
    assert np.isclose(g["se_obs"].iloc[0], 0.5 * np.sqrt(0.02))
    # row 1: |0.05 - 0.015| / 0.0707 < 1.96 -> within noise; row 2: |-0.02 - 0.01| / 0.00707 > 1.96
    assert list(g["within_noise"]) == [True, False]
    t = tally(g)
    assert (t["model_closer"], t["eb_closer"]) == (1, 1)
