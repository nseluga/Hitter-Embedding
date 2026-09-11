import numpy as np
import pandas as pd

from src.analysis import pitcher_type_query as ptq


def test_attributes_use_train_seasons_only():
    rows = []
    for season, speed in ((2020, 90.0), (2024, 99.0)):
        for i in range(300):
            rows.append({"pitcher": 1, "p_throws": "R", "season": season,
                         "pitch_type": "FF" if i % 2 else "SL",
                         "release_speed": speed, "plate_z": 2.5})
    attrs = ptq.pitcher_attributes(pd.DataFrame(rows), [2020])
    assert attrs["fb_velo"].iat[0] == 90.0
    assert attrs["brk_share"].iat[0] == 0.5


def test_weighted_terciles_split_weight_in_thirds():
    values = np.arange(9, dtype=float)
    labels = ptq.weighted_terciles(values, np.ones(9))
    assert labels.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    # one pitcher carrying half the weight fills more than a tercile on its own
    heavy = ptq.weighted_terciles(values, np.r_[np.ones(8), 8.0])
    assert heavy[-1] == 2 and (heavy == 2).sum() == 1
