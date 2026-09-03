import math

import numpy as np
import pandas as pd

from src.analysis.model_visualization_anchors import (
    aggregate_type_count,
    bin_index,
    cell_delta,
    cosine_neighbours,
    project_runtime,
    zscore_candidates,
)


def test_cosine_neighbours_excludes_anchor_and_row_zero():
    rng = np.random.default_rng(0)
    embedding = rng.normal(size=(10, 4))
    neighbours = cosine_neighbours(embedding, anchor_row=3, k=8)
    assert 3 not in neighbours["row"].to_numpy()
    assert 0 not in neighbours["row"].to_numpy()
    # similarity should be sorted descending
    assert (neighbours["similarity_or_distance"].diff().dropna() <= 1e-12).all()


def test_zscore_candidates_drops_nan_and_low_exposure():
    df = pd.DataFrame({
        "embedding_index": [1, 2, 3, 4],
        "swing_rate": [0.4, np.nan, 0.5, 0.3],
        "whiff_rate": [0.2, 0.2, 0.2, 0.2],
        "contact_rate": [0.7, 0.7, 0.7, 0.7],
        "chase_rate": [0.3, 0.3, 0.3, 0.3],
        "zone_swing_rate": [0.6, 0.6, 0.6, 0.6],
        "ev_mean": [88.0, 88.0, 88.0, 88.0],
        "ev_p90": [100.0, 100.0, 100.0, 100.0],
        "la_mean": [12.0, 12.0, 12.0, 12.0],
        "pull_rate": [0.4, 0.4, 0.4, 0.4],
        "woba_level": [0.32, 0.32, 0.32, 0.32],
        # row 4 has plenty of PA but is missing a stat; row 3 has too few PA
        "log_prior_pa": [math.log(500), math.log(500), math.log(20), math.log(500)],
    })
    filtered, z = zscore_candidates(df)
    kept = set(filtered["embedding_index"])
    assert kept == {1, 4}
    assert z.shape == (2, 10)


def test_cosine_neighbours_with_allowed_rows_never_returns_disallowed_row():
    rng = np.random.default_rng(1)
    embedding = rng.normal(size=(10, 4))
    allowed_rows = {1, 2, 5, 6}
    neighbours = cosine_neighbours(embedding, anchor_row=1, k=8, allowed_rows=allowed_rows)
    returned = set(neighbours["row"].to_numpy())
    assert returned <= allowed_rows
    assert returned <= {2, 5, 6}  # anchor row 1 itself is still excluded


def test_zscore_candidates_strata_restricts_pool_and_zscores_within_it():
    df = pd.DataFrame({
        "embedding_index": [1, 2, 3, 4],
        "stratum": ["high", "high", "low", "low"],
        "swing_rate": [0.4, 0.6, 0.5, 0.3],
        "whiff_rate": [0.2, 0.2, 0.2, 0.2],
        "contact_rate": [0.7, 0.7, 0.7, 0.7],
        "chase_rate": [0.3, 0.3, 0.3, 0.3],
        "zone_swing_rate": [0.6, 0.6, 0.6, 0.6],
        "ev_mean": [88.0, 88.0, 88.0, 88.0],
        "ev_p90": [100.0, 100.0, 100.0, 100.0],
        "la_mean": [12.0, 12.0, 12.0, 12.0],
        "pull_rate": [0.4, 0.4, 0.4, 0.4],
        "woba_level": [0.32, 0.32, 0.32, 0.32],
        "log_prior_pa": [math.log(500)] * 4,
    })
    filtered, z = zscore_candidates(df, strata={"high"})
    assert set(filtered["embedding_index"]) == {1, 2}
    # z-scored within the restricted 2-row pool: mean of the only varying column is ~0
    swing_col = list(df.columns[df.columns == "swing_rate"])[0]
    swing_idx = [c for c in df.columns if c not in
                ("embedding_index", "stratum", "log_prior_pa")].index("swing_rate")
    assert abs(z[:, swing_idx].mean()) < 1e-9


def test_bin_index_edge_point_lands_in_correct_bin():
    edges = np.linspace(-1.5, 1.5, 6)  # 5 bins
    # the right edge of the grid falls in the last bin, not a nonexistent 6th bin
    idx = bin_index(np.array([-1.5, 1.5, 0.0]), edges)
    assert idx[0] == 0
    assert idx[1] == 4
    assert idx[2] == 2


def test_cell_delta_is_anchor_minus_league():
    anchor = pd.DataFrame({"family": ["fastball"], "balls": [0], "strikes": [0],
                          "n_pitches": [100], "p_swing": [0.6], "p_contact": [0.8],
                          "q": [0.3]})
    league = pd.DataFrame({"family": ["fastball"], "balls": [0], "strikes": [0],
                          "n_pitches": [1000], "p_swing": [0.5], "p_contact": [0.75],
                          "q": [0.28]})
    delta = cell_delta(anchor, league, ["family", "balls", "strikes"])
    row = delta.set_index("quantity")
    assert math.isclose(row.loc["p_swing", "delta"], 0.1, abs_tol=1e-9)
    assert math.isclose(row.loc["p_contact", "delta"], 0.05, abs_tol=1e-9)
    assert math.isclose(row.loc["q", "delta"], 0.02, abs_tol=1e-9)


def test_aggregate_type_count_groups_before_delta():
    family = np.array(["fastball", "fastball", "breaking"])
    balls = np.array([0, 0, 1])
    strikes = np.array([0, 0, 2])
    values = {"p_swing": np.array([0.4, 0.6, 0.9]), "p_contact": np.array([0.8, 0.8, 0.7]),
              "q": np.array([0.2, 0.4, 0.1])}
    grouped = aggregate_type_count(family, balls, strikes, values)
    fastball_row = grouped[grouped["family"] == "fastball"].iloc[0]
    assert fastball_row["n_pitches"] == 2
    assert math.isclose(fastball_row["p_swing"], 0.5, abs_tol=1e-9)


def test_project_runtime_scales_linearly():
    seconds = project_runtime(seconds_per_unit_batch=10.0, unit_batch_size=5000,
                              total_units=210 * 40000)
    expected = 10.0 / 5000 * 210 * 40000
    assert math.isclose(seconds, expected)
    assert seconds > 0
