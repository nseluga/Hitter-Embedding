"""
Post-V item 2 -- cold-start debut prior (deliverables A-D). Synthetic models/frames
only, no checkpoint reads: `_append_cold_start_row`/`_restore_cold_start_row` and
`query.predict`'s `cold_start_prior` branch are exercised through a tiny real
HitterEmbeddingV1 (mirroring tests/test_query.py's `build_models`), with every heavy
per-pitcher-panel helper in `query.predict` stubbed out -- the thing under test is the
row-0 remap and the try/finally restore, not the panel sampling machinery, which has
its own gates in test_query.py. `posterior_mean`'s per-row `mu` override is tested
directly, since it's the exact mechanism `baseline_ladder_bivariate_eb.predict`'s
`debut_mu` is built on.
"""

import inspect

import numpy as np
import pandas as pd
import pytest
import torch

from src.analysis.baseline_ladder_bivariate_eb import posterior_mean, sigma_matrix
from src.analysis.cold_start_prior_diagnostic import low_stratum_means
from src.model import query
from src.model.v1 import HitterEmbeddingV1

N_BINS = 8
N_CONTEXT = 6
N_HITTERS = 5  # rows 0..5 -> 6 total embedding rows


def build_model(seed=0):
    torch.manual_seed(seed)
    model = HitterEmbeddingV1(N_HITTERS, N_CONTEXT, n_bins=N_BINS)
    model.eval()
    return model


# --- low_stratum_means: mean-vector construction, per stand ----------------------------

def test_low_stratum_means_averages_only_low_stratum_rows_within_each_stand():
    embeddings = np.arange(7 * 4, dtype=float).reshape(7, 4)  # rows 0..6
    stats = pd.DataFrame({
        "batter": [1, 2, 3, 4, 5, 6],  # 1-digit ids: not real MLBAM ids, so not pitchers
        "embedding_index": [1, 2, 3, 4, 5, 6],
        "stand": ["L", "L", "R", "R", "R", "L"],
        "stratum": ["low", "low", "low", "medium", "low", "medium"],  # row 6 excluded
    })
    means = low_stratum_means(embeddings, stats)
    assert set(means) == {"L", "R"}
    np.testing.assert_allclose(means["L"], embeddings[[1, 2]].mean(axis=0))
    np.testing.assert_allclose(means["R"], embeddings[[3, 5]].mean(axis=0))


# --- _append_cold_start_row / _restore_cold_start_row: bit-identity --------------------

def test_append_and_restore_cold_start_row_is_bit_identical():
    model = build_model()
    original_weight = model.embedding.weight.detach().clone()
    vector = np.full(N_CONTEXT if False else original_weight.shape[1], 3.5)

    original_module, new_index = query._append_cold_start_row(model, vector)
    assert new_index == original_weight.shape[0]
    assert model.embedding.weight.shape[0] == original_weight.shape[0] + 1
    np.testing.assert_allclose(model.embedding.weight[new_index].detach().numpy(), vector)
    # the trained rows, including row 0, are untouched by the append
    assert torch.equal(model.embedding.weight[:new_index], original_weight)

    query._restore_cold_start_row(model, original_module)
    assert model.embedding is original_module
    assert torch.equal(model.embedding.weight, original_weight)


def test_restore_runs_even_if_the_query_between_append_and_restore_raises():
    model = build_model()
    original_weight = model.embedding.weight.detach().clone()
    original_module, _ = query._append_cold_start_row(model, np.zeros(original_weight.shape[1]))
    try:
        try:
            raise RuntimeError("boom")
        finally:
            query._restore_cold_start_row(model, original_module)
    except RuntimeError:
        pass
    assert torch.equal(model.embedding.weight, original_weight)


# --- query.predict: cold_start_prior remaps only zero-vocabulary hitters ---------------

def _stub_predict_internals(monkeypatch):
    """
    Replace every per-pitcher-panel helper `predict` calls with a cheap stand-in, and
    capture the `hitter_rows` array `_group_woba` is actually called with -- the one
    line the cold_start_prior branch changes.
    """
    captured = []

    def fake_group_woba(models, kernels, tensors, frame, tables, points, n_bins,
                        hitter_rows, grid, weight_vector, split_head, chunk, w_bb, w_hbp,
                        use_league_split, share, unmeasured, progress=None):
        captured.append(np.asarray(hitter_rows).copy())
        n = len(hitter_rows)
        total = np.full(n, 0.3)
        used = np.ones(n)
        absorbing = {key: np.full(n, 0.25) for key in query.ABSORBING_KEYS}
        return total, used, absorbing

    monkeypatch.setattr(query, "spray_kernels", lambda model, points, n_bins, spray_mass: None)
    monkeypatch.setattr(query, "_unmeasured_terms", lambda tables, weights, split: (1.0, 0.0))
    monkeypatch.setattr(query.qt, "woba_points_table", lambda outcome, weights: np.zeros((2, 2, 2)))
    monkeypatch.setattr(query, "_panel", lambda slots, probability, n_pitchers, generator: (slots, probability))
    monkeypatch.setattr(query, "_sample_grid", lambda repertoire, slots, stand_slot, n_pitches, generator: (
        np.zeros(3), np.array([True, True, True]), np.zeros(4, dtype="int64")))
    monkeypatch.setattr(query, "_group_woba", fake_group_woba)
    monkeypatch.setattr(query.eval_targets, "load_weights", lambda: {"2024": {"wBB": 0.7, "wHBP": 0.9}})
    return captured


def _synthetic_predict_args():
    pa_df = pd.DataFrame({
        "batter": [5, 6], "season": [2024, 2024], "p_throws": ["R", "R"],
        "stand": ["L", "L"], "game_pk": [1, 2], "at_bat_number": [1, 1], "pitcher": [100, 101],
    })
    manifest = {"vocabulary": {5: 3}, "n_quality_bins": N_BINS}  # batter 6 -> vocabulary miss -> row 0
    tables = {
        "outcome": None, "spray_mass": None, "unmeasured": None, "coverage": 1.0,
        "repertoire": None, "pitcher_weights": {"R": (None, np.array([1.0, 1.0, 1.0]))},
    }
    return pa_df, manifest, tables


def test_predict_default_none_leaves_the_vocabulary_miss_at_row_zero(monkeypatch):
    captured = _stub_predict_internals(monkeypatch)
    model = build_model()
    original_weight = model.embedding.weight.detach().clone()
    pa_df, manifest, tables = _synthetic_predict_args()

    out, diagnostics = query.predict([model], {}, manifest, None, tables, pa_df, 2024)

    assert diagnostics["cold_start_prior"] is False
    (hitter_rows,) = captured
    # batter 5 is in vocabulary at row 3; batter 6 is a vocabulary miss and stays at
    # the reserved origin row 0 -- today's behaviour, unchanged
    assert sorted(hitter_rows.tolist()) == [0, 3]
    assert torch.equal(model.embedding.weight, original_weight)


def test_predict_cold_start_prior_remaps_only_the_vocabulary_miss(monkeypatch):
    captured = _stub_predict_internals(monkeypatch)
    model = build_model()
    original_weight = model.embedding.weight.detach().clone()
    n_trained_rows = original_weight.shape[0]
    pa_df, manifest, tables = _synthetic_predict_args()
    prior_vector = np.full(original_weight.shape[1], 9.0)

    out, diagnostics = query.predict([model], {}, manifest, None, tables, pa_df, 2024,
                                     cold_start_prior={"L": prior_vector})

    assert diagnostics["cold_start_prior"] is True
    (hitter_rows,) = captured
    # batter 5 (row 3, a real vocabulary entry) is untouched; batter 6 (a miss) is
    # remapped to the newly appended prior row, never to 0
    assert n_trained_rows in hitter_rows
    assert 0 not in hitter_rows
    assert 3 in hitter_rows
    # the embedding is grown-and-restored around the call, not left mutated
    assert model.embedding.weight.shape[0] == n_trained_rows
    assert torch.equal(model.embedding.weight, original_weight)


# --- posterior_mean: a per-row mu override touches only the overridden rows ------------
# (the exact mechanism baseline_ladder_bivariate_eb.predict's debut_mu is built on --
# there it overrides record["mu"] only for rows with n_L + n_R == 0)

def test_posterior_mean_per_row_mu_override_touches_only_that_row():
    x = np.array([[0.310, 0.290], [np.nan, np.nan], [0.300, 0.305]])
    n_pa = np.array([[120, 130], [0, 0], [80, 90]])
    league_mu = np.array([0.315, 0.315])
    matrix = sigma_matrix(np.array([0.001, 0.001]), 0.5)
    sigma2 = np.array([0.24, 0.24])

    baseline = posterior_mean(x, n_pa, league_mu, matrix, sigma2)

    debut_mu_value = 0.250  # a below-league debut prior
    mu_rows = np.broadcast_to(league_mu, x.shape).copy()
    zero_prior = n_pa.sum(axis=1) == 0
    mu_rows[zero_prior] = debut_mu_value
    overridden = posterior_mean(x, n_pa, mu_rows, matrix, sigma2)

    # rows WITH prior PA are bit-identical -- the override never touches them
    for row in (0, 2):
        np.testing.assert_allclose(overridden[row], baseline[row])
    # the zero-PA row moves toward the debut prior, not the league mu
    assert overridden[1, 0] < baseline[1, 0]
    assert overridden[1, 1] < baseline[1, 1]


if __name__ == "__main__":
    test_low_stratum_means_averages_only_low_stratum_rows_within_each_stand()
    test_append_and_restore_cold_start_row_is_bit_identical()
    test_restore_runs_even_if_the_query_between_append_and_restore_raises()
    test_posterior_mean_per_row_mu_override_touches_only_that_row()
    print("cold_start_prior smoke checks passed (run pytest for the monkeypatched predict() tests)")


# --- the prior is now the DEFAULT (2026-09-04) ----------------------------------------
# Deliverable A shipped the prior behind a flag. It is now on unless asked off, so the
# thing worth gating is the default itself: a query run that silently reverted to the raw
# padding row would still produce a full, plausible, wrong set of cold-start numbers.

def test_the_query_cli_turns_the_cold_start_prior_on_unless_asked_off():
    defaults = query.build_parser().parse_args([])
    assert defaults.no_cold_start_prior is False
    assert defaults.hitter_stats == query.DEFAULT_HITTER_STATS
    assert query.build_parser().parse_args(["--no-cold-start-prior"]).no_cold_start_prior


def test_default_cold_start_prior_averages_the_seeds_in_their_own_spaces(tmp_path):
    models = [build_model(seed) for seed in (0, 1)]
    stats = pd.DataFrame({
        "batter": [1, 2, 3, 4],
        "embedding_index": [1, 2, 3, 4],
        "stand": ["L", "L", "R", "R"],
        "stratum": ["low", "low", "low", "medium"],
    })
    stats_csv = tmp_path / "hitter_stats.csv"
    stats.to_csv(stats_csv, index=False)

    prior = query.default_cold_start_prior(models, stats_csv)
    assert set(prior) == {"L", "R"}

    per_seed = [low_stratum_means(m.embedding.weight.detach().numpy(), stats) for m in models]
    for stand in ("L", "R"):
        np.testing.assert_allclose(prior[stand],
                                   np.mean([m[stand] for m in per_seed], axis=0))
    # a seed's own space, not a pooled one: the two seeds disagree, so the average is
    # not either of them
    assert not np.allclose(prior["L"], per_seed[0]["L"])


def test_the_eb_ladder_gets_a_debut_prior_by_default_so_the_comparison_stays_fair(tmp_path):
    stats = pd.DataFrame({
        "stand": ["L", "L", "R", "R"],
        "stratum": ["low", "low", "low", "medium"],
        "woba_level": [0.240, 0.260, 0.300, 0.400],
    })
    stats_csv = tmp_path / "hitter_stats.csv"
    stats.to_csv(stats_csv, index=False)

    from src.analysis import baseline_ladder_bivariate_eb as eb
    from src.analysis import baseline_ladder_report as report

    debut_mu = eb.debut_mu_from_stats(stats_csv)
    assert debut_mu["L"] == (0.250, 0.250) and debut_mu["R"] == (0.300, 0.300)
    # switch hitters take the pooled low mean over ROWS, not the mean of the two stand
    # means -- the stands are unbalanced and the two differ
    assert debut_mu["S"] == pytest.approx((0.2666666, 0.2666666), abs=1e-6)

    assert "debut_mu" in inspect.signature(report.build_predictions).parameters
    source = inspect.getsource(report.main)
    assert "--no-debut-prior" in source and "debut_mu_from_stats" in source, \
        "the ladder report must build the debut prior unless --no-debut-prior is passed"


def test_low_stratum_means_drops_a_career_pitcher_batter_by_default():
    """The averaged population must not contain pitchers -- the whole point of the filter."""
    from src.analysis.embedding_structure import career_pitcher_batter_ids
    pitcher = sorted(career_pitcher_batter_ids())[0]
    embeddings = np.arange(4 * 4, dtype=float).reshape(4, 4)
    stats = pd.DataFrame({
        "batter": [1, 2, pitcher],
        "embedding_index": [1, 2, 3],
        "stand": ["R", "R", "R"],
        "stratum": ["low", "low", "low"],
    })
    np.testing.assert_allclose(low_stratum_means(embeddings, stats)["R"],
                               embeddings[[1, 2]].mean(axis=0))
    # and the escape hatch still reproduces the contaminated population
    np.testing.assert_allclose(
        low_stratum_means(embeddings, stats, exclude_pitcher_batters=False)["R"],
        embeddings[[1, 2, 3]].mean(axis=0))
