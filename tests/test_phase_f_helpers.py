"""Pure helpers behind the Phase F modules. No checkpoints, no parquets, no GPU."""

import json

import numpy as np
import pandas as pd
import pytest

from src.analysis import provenance
from src.analysis.f5_pooled import POOLED_HAND, pool_predictions, pooling_weights


def write_build(tmp_path, name, edges):
    """Minimal tensor-build directory: just the manifest the stamper reads."""
    build = tmp_path / name
    build.mkdir()
    (build / "manifest.json").write_text(json.dumps({
        "train_seasons": [2015, 2016], "quality_bin_edges": edges}))
    return build


def test_manifest_digest_is_stable_and_content_sensitive(tmp_path):
    same_a = write_build(tmp_path, "a", {"ev": [0, 1]})
    same_b = write_build(tmp_path, "b", {"ev": [0, 1]})
    other = write_build(tmp_path, "c", {"ev": [0, 2]})
    assert provenance.manifest_digest(same_a) == provenance.manifest_digest(same_b)
    assert provenance.manifest_digest(same_a) != provenance.manifest_digest(other)


def test_stamp_carries_the_fields_a_reader_needs(tmp_path):
    build = write_build(tmp_path, "d5", {"ev": [0, 1], "la": [0, 1], "spray": [0, 1]})
    stamp = provenance.stamp(build, arm="d10_baseline", seeds=[0, 1], eval_season=2024)
    assert stamp["train_seasons"] == [2015, 2016]
    assert stamp["quality_bin_edges_present"] == ["ev", "la", "spray"]
    assert stamp["arm"] == "d10_baseline"
    assert stamp["eval_season"] == 2024
    assert len(stamp["manifest_sha256"]) == 64


def test_assert_quality_bins_blocks_a_mismatched_build(tmp_path):
    edges = {"ev": [0, 1], "la": [0, 1], "spray": [0, 1]}
    reference = write_build(tmp_path, "ref", edges)
    matching = write_build(tmp_path, "match", edges)
    provenance.assert_quality_bins(matching, reference_data_dir=reference)
    mismatched = write_build(tmp_path, "bad", {**edges, "ev": [0, 9]})
    with pytest.raises(AssertionError):
        provenance.assert_quality_bins(mismatched, reference_data_dir=reference)


def pa_frame():
    """Two prior seasons of exposure plus the eval season, two batters, both hands."""
    rows = []
    for season in (2022, 2023, 2024):
        for batter, versus_right in ((1, 30), (2, 10)):
            rows.append({"batter": batter, "season": season, "p_throws": "R",
                         "in_denominator": versus_right, "woba_value": 0.3 * versus_right})
            rows.append({"batter": batter, "season": season, "p_throws": "L",
                         "in_denominator": 10, "woba_value": 0.3 * 10})
    return pd.DataFrame(rows)


def test_pooling_weights_are_prior_shares_that_sum_to_one():
    weights = pooling_weights(pa_frame(), 2024)
    totals = weights.groupby("batter")["weight"].sum()
    assert np.allclose(totals.to_numpy(), 1.0)
    # batter 1 saw 30 vs R against 10 vs L in each of two prior seasons -> 0.75 / 0.25
    versus_right = weights[(weights["batter"] == 1) & (weights["p_throws"] == "R")]
    assert versus_right["weight"].iloc[0] == pytest.approx(0.75)


def test_pooling_ignores_eval_season_exposure():
    """The eval season must not move the weight; that would condition on the answer key."""
    baseline = pooling_weights(pa_frame(), 2024)
    inflated = pa_frame()
    eval_rows = (inflated["season"] == 2024) & (inflated["p_throws"] == "L")
    inflated.loc[eval_rows, "in_denominator"] = 500
    assert np.allclose(baseline["weight"].to_numpy(),
                       pooling_weights(inflated, 2024)["weight"].to_numpy())


def test_pool_predictions_is_the_weighted_mean_of_the_two_sides():
    weights = pooling_weights(pa_frame(), 2024)
    predictions = pd.DataFrame({
        "batter": [1, 1], "season": [2024, 2024], "p_throws": ["R", "L"],
        "pred_woba": [0.400, 0.200]})
    pooled = pool_predictions(predictions, weights, 2024)
    assert len(pooled) == 1
    assert pooled["p_throws"].iloc[0] == POOLED_HAND
    assert pooled["pred_woba"].iloc[0] == pytest.approx(0.75 * 0.400 + 0.25 * 0.200)


def test_pool_predictions_falls_back_to_an_even_split_without_prior_exposure():
    weights = pooling_weights(pa_frame(), 2024)
    predictions = pd.DataFrame({
        "batter": [99, 99], "season": [2024, 2024], "p_throws": ["R", "L"],
        "pred_woba": [0.400, 0.200]})
    pooled = pool_predictions(predictions, weights, 2024)
    assert pooled["pred_woba"].iloc[0] == pytest.approx(0.300)
