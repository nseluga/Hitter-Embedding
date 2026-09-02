import pandas as pd

from src.analysis.model_visualization_disagreement import build_disagreement, summarize


def _platoon_frame():
    return pd.DataFrame({
        "batter": [1, 2, 3, 4, 5],
        "delta_pred": [0.05, 0.02, -0.01, 0.03, -0.02],
        "stand": ["L", "L", "L", "R", "R"],
        "stratum": ["high", "high", "low", "high", "low"],
        "prior_pa": [500, 400, 100, 600, 90],
    })


def _eb_frame(overrides=None):
    base = pd.DataFrame({
        "batter": [1, 2, 3, 4, 5],
        "delta_eb": [0.04, 0.03, -0.02, 0.01, -0.03],
    })
    if overrides:
        base = pd.concat([base, pd.DataFrame(overrides)], ignore_index=True)
    return base


def _names():
    return pd.DataFrame({"batter": [1, 2, 3, 4, 5], "name": ["A", "B", "C", "D", "E"]})


def test_rank_one_is_most_positive_delta():
    disagreement, _ = build_disagreement(_platoon_frame(), _eb_frame(), _names())
    top_l = disagreement[disagreement["batter"] == 1]
    assert top_l["rank_model"].iloc[0] == 1
    top_eb_l = disagreement[disagreement["batter"] == 1]
    assert top_eb_l["rank_eb"].iloc[0] == 1


def test_ranking_is_separate_per_stand():
    disagreement, _ = build_disagreement(_platoon_frame(), _eb_frame(), _names())
    # batter 4 is the only R hitter with a positive delta_pred among R hitters -> rank 1
    r_rows = disagreement[disagreement["stand"] == "R"]
    assert set(r_rows["rank_model"]) == {1, 2}
    assert r_rows[r_rows["batter"] == 4]["rank_model"].iloc[0] == 1


def test_inner_join_drops_unmatched():
    # 10 platoon_frame rows, only 8 have an EB estimate -> 20% loss stays under the gate
    platoon_frame = pd.concat([_platoon_frame(), pd.DataFrame({
        "batter": [6, 7, 8, 9, 10], "delta_pred": [0.01, 0.02, 0.03, 0.04, 0.05],
        "stand": ["L", "L", "R", "R", "R"], "stratum": ["low"] * 5, "prior_pa": [50] * 5,
    })], ignore_index=True)
    eb = _eb_frame(overrides={"batter": [6, 7, 8], "delta_eb": [0.0, 0.01, 0.02]})
    names = pd.concat([_names(), pd.DataFrame({"batter": range(6, 11), "name": list("FGHIJ")})],
                       ignore_index=True)
    disagreement, counts = build_disagreement(platoon_frame, eb, names)
    assert set(disagreement["batter"]) == {1, 2, 3, 4, 5, 6, 7, 8}
    assert counts["n_before"] == 10
    assert counts["n_after"] == 8
    assert counts["n_dropped"] == 2


def test_rank_gap_sign():
    disagreement, _ = build_disagreement(_platoon_frame(), _eb_frame(), _names())
    row = disagreement[disagreement["batter"] == 4].iloc[0]
    assert row["rank_gap"] == row["rank_model"] - row["rank_eb"]


def test_summary_has_stratum_and_pooled_keys():
    disagreement, _ = build_disagreement(_platoon_frame(), _eb_frame(), _names())
    summary = summarize(disagreement)
    assert "pooled" in summary
    assert set(disagreement["stratum"]) <= set(summary.keys())
    for stat in summary.values():
        assert {"n_hitters", "spearman", "rank_gap_sd"} <= set(stat.keys())
