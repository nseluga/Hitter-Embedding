"""
Verification gates for Phase V.3/V.4 (docs/phase-v-spec.md). Pure synthetic-data logic,
no parquet or checkpoint reads.
"""

import numpy as np
import pandas as pd
import torch

from src.analysis import model_visualization_heads as mvh


def test_partial_correlation_removes_pure_confounder():
    """a and b are both linear functions of a confounder plus noise-free; partialling
    the confounder out should send r toward zero even though raw r is high."""
    rng = np.random.default_rng(0)
    confound = rng.normal(size=500)
    # independent noise keeps the residuals from being exactly zero, which would make
    # the correlation of two all-zero vectors numerically undefined rather than ~0
    a = 3.0 * confound + rng.normal(scale=1e-3, size=500)
    b = -2.0 * confound + rng.normal(scale=1e-3, size=500)
    r_raw, _ = mvh.paired_pearson(a, b)
    r_partial, _ = mvh.paired_pearson(a, b, confound=confound)
    assert abs(r_raw) > 0.99
    assert abs(r_partial) < 0.1


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(1)
    confound = rng.normal(size=300)
    a = confound + rng.normal(scale=0.5, size=300)
    b = confound + rng.normal(scale=0.5, size=300)
    point, lo, hi, n = mvh.bootstrap_ci_partial(a, b, confound, n_boot=200, seed=2)
    assert lo <= point <= hi
    assert n == 300


def test_level_query_averages_sides_present():
    """A hitter queried on both L and R averages both; a hitter with only one side
    keeps that side's value untouched."""
    predictions = pd.DataFrame({
        "batter": [1, 1, 2],
        "p_throws": ["L", "R", "L"],
        "pred_woba": [0.300, 0.340, 0.280],
    })
    predictions.to_csv("/tmp/_test_level_query_predictions.csv", index=False)
    original_path = mvh.LEVEL_QUERY_PREDICTIONS_PATH
    mvh.LEVEL_QUERY_PREDICTIONS_PATH = "/tmp/_test_level_query_predictions.csv"
    try:
        hitters = pd.DataFrame({"batter": [1, 2], "stratum": ["high", "low"],
                                "log_prior_pa": [5.0, 2.0], "woba_level": [0.32, 0.29]})
        names = pd.DataFrame({"batter": [1, 2], "embedding_index": [1, 2],
                              "name": ["A", "B"], "stand": ["R", "L"]})
        names.to_csv("/tmp/_test_names.csv", index=False)
        original_names = mvh.NAMES_PATH
        mvh.NAMES_PATH = "/tmp/_test_names.csv"
        try:
            level_query_df = mvh.build_level_query(hitters)
        finally:
            mvh.NAMES_PATH = original_names
    finally:
        mvh.LEVEL_QUERY_PREDICTIONS_PATH = original_path

    row1 = level_query_df.set_index("batter").loc[1, "level_query"]
    row2 = level_query_df.set_index("batter").loc[2, "level_query"]
    assert abs(row1 - 0.320) < 1e-9
    assert abs(row2 - 0.280) < 1e-9


def test_reference_sampler_deterministic_and_train_seasons_only():
    tensors = {"season": torch.tensor([2015, 2016, 2024, 2025, 2018, 2023, 2024])}
    manifest = {"train_seasons": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]}
    idx_a = mvh.sample_reference_context(tensors, manifest, n=3, seed=5)
    idx_b = mvh.sample_reference_context(tensors, manifest, n=3, seed=5)
    assert np.array_equal(idx_a, idx_b)
    chosen_seasons = tensors["season"].numpy()[idx_a]
    assert set(chosen_seasons.tolist()) <= set(manifest["train_seasons"])


def test_loadings_frame_is_long_form():
    coords = pd.DataFrame({"batter": [1, 2, 3], "embedding_index": [1, 2, 3],
                           "pc1": [0.1, 0.2, -0.1], "pc2": [-0.1, 0.05, 0.3],
                           "pc3": [0.0, 0.1, 0.2], "pc4": [0.2, -0.2, 0.1]})
    marginals_df = pd.DataFrame({"batter": [1, 2, 3], "embedding_index": [1, 2, 3],
                                 "swing": [0.4, 0.5, 0.6], "contact": [0.7, 0.8, 0.75],
                                 "split_inplay": [0.2, 0.25, 0.3], "ev": [88.0, 90.0, 91.0],
                                 "la": [12.0, 10.0, 14.0], "spray": [1.0, -1.0, 0.5]})
    hitters = pd.DataFrame({"batter": [1, 2, 3], "embedding_index": [1, 2, 3],
                            "swing_rate": [0.4, 0.5, 0.6], "whiff_rate": [0.2, 0.25, 0.3],
                            "contact_rate": [0.7, 0.8, 0.75], "chase_rate": [0.1, 0.2, 0.15],
                            "zone_swing_rate": [0.6, 0.65, 0.7], "ev_mean": [88.0, 90.0, 91.0],
                            "ev_p90": [100.0, 102.0, 101.0], "la_mean": [12.0, 10.0, 14.0],
                            "bat_speed_mean": [70.0, 71.0, 72.0], "pull_rate": [0.3, 0.35, 0.4],
                            "woba_level": [0.31, 0.33, 0.30], "obs_platoon_diff": [0.01, -0.02, 0.03],
                            "log_prior_pa": [4.0, 5.0, 6.0]})
    loadings, merged = mvh.build_loadings(coords, ["pc1", "pc2", "pc3", "pc4"],
                                          marginals_df, hitters)
    assert set(loadings.columns) == {"pc", "variable", "r_raw", "r_partial", "n"}
    assert set(loadings["pc"]) == {"pc1", "pc2", "pc3", "pc4"}
    n_variables = len(mvh.HEAD_MARGINALS) + len(mvh.STAT_VARIABLES) + 1
    assert len(loadings) == 4 * n_variables
