"""
D.5 verification gates (ml-engineer skill; CLAUDE.md ML verification gate).

Every failure this module guards is silent. A factored spray sum that disagrees with the
literal one still returns plausible wOBA. A chain solve that mishandles the two-strike
foul still returns a number in the right range. A table fit on the eval season still
scores. None of these raise on their own, and a poisoned claim-1 number is worse than none.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.model import query, query_tables as qt
from src.model.v1 import HitterEmbeddingV1

N_BINS = 24
N_CONTEXT = 46
N_HITTERS = 40


def build_models(n=2, seed=0, bilinear=False):
    models = []
    for offset in range(n):
        torch.manual_seed(seed + offset)
        model = HitterEmbeddingV1(N_HITTERS, N_CONTEXT, n_bins=N_BINS, bilinear=bilinear)
        model.eval()
        models.append(model)
    return models


def sample_batch(rows=17, seed=3):
    generator = torch.Generator().manual_seed(seed)
    hitter = torch.randint(0, N_HITTERS + 1, (rows,), generator=generator)
    context = torch.randn(rows, N_CONTEXT, generator=generator)
    return hitter, context


def random_points(seed=5):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(N_BINS, N_BINS, N_BINS, generator=generator) * 2.0


# --- gate 1: the fast path computes the algebra the spec says it does ------------------

@pytest.mark.parametrize("bilinear", [False, True])
def test_factored_expectation_matches_the_literal_joint(bilinear):
    """
    The (rows, 576) factored form must equal the literal (rows, 24, 24, 24) softmax.
    This is the whole basis for the 24x speedup; if it drifts, every wOBA number is wrong
    in a way nothing downstream can detect.
    """
    models = build_models(n=3, bilinear=bilinear)
    hitter, context = sample_batch()
    points = random_points()

    fast = query.expected_woba(models, hitter, context, points, N_BINS)[2]
    naive = query.expected_woba_naive(models, hitter, context, points, N_BINS)
    assert np.allclose(fast, naive, atol=1e-5), \
        f"factored and literal forms disagree by {np.abs(fast - naive).max():.2e}"


def test_expectation_is_bounded_by_the_points_table():
    """An expectation over a distribution cannot leave the table's range."""
    models = build_models()
    hitter, context = sample_batch()
    points = random_points()
    quality = query.expected_woba(models, hitter, context, points, N_BINS)[2]
    assert quality.min() >= float(points.min()) - 1e-9
    assert quality.max() <= float(points.max()) + 1e-9


def test_single_model_ensemble_is_the_model_itself():
    models = build_models(n=1)
    hitter, context = sample_batch()
    points = random_points()
    one = query.expected_woba(models, hitter, context, points, N_BINS)[2]
    two = query.expected_woba(models * 2, hitter, context, points, N_BINS)[2]
    assert np.allclose(one, two, atol=1e-9), "averaging a model with itself moved it"


# --- gate 2: the chain solve --------------------------------------------------------

def constant_aggregates(ball, strike, hbp, foul, bip, bip_points):
    keys = ("ball", "strike", "hbp", "foul", "bip", "bip_points")
    values = (ball, strike, hbp, foul, bip, bip_points)
    return {key: np.full((1, 4, 3), value, dtype="float64")
            for key, value in zip(keys, values)}


def test_every_pitch_a_ball_is_a_walk():
    """Four balls and nothing else must return exactly the walk weight."""
    aggregates = constant_aggregates(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    solved = query.solve_chain(aggregates, w_bb=0.69, w_hbp=0.72)
    assert solved[0, 0, 0] == pytest.approx(0.69)


def test_every_pitch_a_strike_is_a_strikeout():
    aggregates = constant_aggregates(0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    solved = query.solve_chain(aggregates, w_bb=0.69, w_hbp=0.72)
    assert solved[0, 0, 0] == pytest.approx(0.0)


def test_every_pitch_in_play_returns_its_own_value():
    """With all mass on balls in play the answer is the per-pitch wOBA points."""
    aggregates = constant_aggregates(0.0, 0.0, 0.0, 0.0, 1.0, 0.41)
    solved = query.solve_chain(aggregates, w_bb=0.69, w_hbp=0.72)
    assert solved[0, 0, 0] == pytest.approx(0.41)


def test_two_strike_foul_loop_matches_value_iteration():
    """
    The closed-form geometric division must equal iterating the chain to a fixed point.
    This is the one place the exact solve replaces something a simulator would truncate,
    so it is checked against the thing it replaces rather than against itself.
    """
    rng = np.random.default_rng(11)
    masses = rng.dirichlet([2, 2, 0.4, 3, 3], size=(4, 3))
    aggregates = {name: masses[..., i][None, ...] for i, name in
                  enumerate(("ball", "strike", "hbp", "foul", "bip"))}
    aggregates["bip_points"] = aggregates["bip"] * 0.38
    w_bb, w_hbp = 0.69, 0.72

    closed = query.solve_chain(aggregates, w_bb, w_hbp)

    # value iteration: the same recursion with the self-loop unrolled by brute force
    W = np.zeros((4, 3))
    for _ in range(5000):
        nxt = np.zeros((4, 3))
        for strikes in reversed(range(3)):
            for balls in reversed(range(4)):
                ball_value = w_bb if balls == 3 else W[balls + 1, strikes]
                strike_value = 0.0 if strikes == 2 else W[balls, strikes + 1]
                foul_value = W[balls, strikes] if strikes == 2 else W[balls, strikes + 1]
                nxt[balls, strikes] = (
                    aggregates["ball"][0, balls, strikes] * ball_value
                    + aggregates["strike"][0, balls, strikes] * strike_value
                    + aggregates["hbp"][0, balls, strikes] * w_hbp
                    + aggregates["foul"][0, balls, strikes] * foul_value
                    + aggregates["bip_points"][0, balls, strikes])
        W = nxt
    assert np.allclose(closed[0], W, atol=1e-9), \
        "closed-form foul loop disagrees with value iteration"


def test_solve_chain_is_vectorised_over_hitters():
    rng = np.random.default_rng(2)
    masses = rng.dirichlet([2, 2, 0.4, 3, 3], size=(6, 4, 3))
    aggregates = {name: masses[..., i] for i, name in
                  enumerate(("ball", "strike", "hbp", "foul", "bip"))}
    aggregates["bip_points"] = aggregates["bip"] * 0.4
    batched = query.solve_chain(aggregates, 0.69, 0.72)
    for row in range(6):
        single = query.solve_chain({k: v[row][None, ...] for k, v in aggregates.items()},
                                   0.69, 0.72)
        assert np.allclose(batched[row], single[0])


def test_walk_weight_bounds_a_patient_hitter():
    """A hitter who only ever walks or strikes out lands between 0 and the walk weight."""
    aggregates = constant_aggregates(0.5, 0.5, 0.0, 0.0, 0.0, 0.0)
    solved = query.solve_chain(aggregates, w_bb=0.69, w_hbp=0.72)
    assert 0.0 < solved[0, 0, 0] < 0.69


# --- gate 2b: the §8 absorbing rates (D5-R14) ----------------------------------------

def test_absorbing_rates_are_exhaustive():
    """
    The four states partition every plate appearance, so they must sum to 1 at every count.
    This is the property the check exists for: a walk/strikeout distortion is invisible in
    the wOBA level because wBB and wHBP are close, but it cannot hide in these four.
    """
    rng = np.random.default_rng(7)
    masses = rng.dirichlet([2, 2, 0.4, 3, 3], size=(5, 4, 3))
    aggregates = {name: masses[..., i] for i, name in
                  enumerate(("ball", "strike", "hbp", "foul", "bip"))}
    aggregates["bip_points"] = aggregates["bip"] * 0.38
    rates = query.absorbing_rates(aggregates)
    assert set(rates) == set(query.ABSORBING_KEYS)
    total = sum(rates.values())
    assert np.allclose(total, 1.0, atol=1e-9)
    for rate in rates.values():
        assert (rate >= -1e-12).all() and (rate <= 1.0 + 1e-12).all()


@pytest.mark.parametrize("index,key", [(0, "bb"), (1, "k"), (2, "hbp"), (4, "bip")])
def test_a_degenerate_pitch_absorbs_where_it_should(index, key):
    """All mass on one pitch outcome sends every plate appearance to one absorbing state."""
    masses = [0.0] * 5
    masses[index] = 1.0
    aggregates = constant_aggregates(*masses, 0.0)
    rates = query.absorbing_rates(aggregates)
    assert rates[key][0, 0, 0] == pytest.approx(1.0)


def test_absorbing_rates_agree_with_the_woba_solve():
    """
    Cross-check against solve_chain's own arithmetic: paying 1 for a walk, 1 for an HBP and
    nothing else must reproduce BB + HBP. If the rates came from a second implementation of
    the recursion this would be a tautology; they come from the same one, so it tests that
    the payoff substitution is what it claims to be.
    """
    rng = np.random.default_rng(19)
    masses = rng.dirichlet([2, 2, 0.4, 3, 3], size=(3, 4, 3))
    aggregates = {name: masses[..., i] for i, name in
                  enumerate(("ball", "strike", "hbp", "foul", "bip"))}
    aggregates["bip_points"] = aggregates["bip"] * 0.5
    rates = query.absorbing_rates(aggregates)
    both = query.solve_chain({**aggregates, "bip_points": np.zeros_like(aggregates["bip"])},
                             w_bb=1.0, w_hbp=1.0)
    assert np.allclose(rates["bb"] + rates["hbp"], both, atol=1e-12)


def test_a_two_strike_foul_never_absorbs_by_itself():
    """
    The self-loop must not leak: fouling forever at 0-2 with a sliver of every other outcome
    still ends the plate appearance with probability 1, and it ends by strikeout most often.
    """
    aggregates = constant_aggregates(0.001, 0.002, 0.001, 0.995, 0.001, 0.0004)
    rates = query.absorbing_rates(aggregates)
    assert sum(float(rate[0, 0, 2]) for rate in rates.values()) == pytest.approx(1.0)
    assert rates["k"][0, 0, 2] > rates["bb"][0, 0, 2]


def test_observed_absorbing_rates_exhaust_the_denominator():
    """`bip` is the remainder, so an unrecognised event string lands there rather than lost."""
    pa_df = pd.DataFrame({
        "season": [2015] * 6 + [2024],
        "in_denominator": [True] * 5 + [False, True],
        "events": ["walk", "strikeout", "strikeout_double_play", "hit_by_pitch",
                   "some_new_statcast_event", "walk", "single"],
        "stand": ["L"] * 7, "p_throws": ["R"] * 7, "batter": range(7),
    })
    rates, n_pa = query.observed_absorbing_rates(pa_df, [2015])
    assert n_pa == 5
    assert sum(rates.values()) == pytest.approx(1.0)
    assert rates == {"bb": 0.2, "k": 0.4, "hbp": 0.2, "bip": 0.2}


def test_handedness_shares_are_not_uniform_and_sum_to_one():
    """The defect this replaces was a hardcoded 0.25 in all four cells."""
    pa_df = pd.DataFrame({
        "season": [2015] * 10,
        "in_denominator": [True] * 9 + [False],
        "stand": ["L", "L", "L", "L", "R", "R", "R", "R", "R", "L"],
        "p_throws": ["L", "R", "R", "R", "L", "L", "R", "R", "R", "L"],
        "events": ["single"] * 10, "batter": range(10),
    })
    shares = query.handedness_shares(pa_df, [2015])
    assert sum(shares.values()) == pytest.approx(1.0)
    assert shares[("L", "L")] == pytest.approx(1 / 9)
    assert shares[("R", "R")] == pytest.approx(3 / 9)


def test_league_fidelity_scores_only_trained_hitters():
    """
    A cold-start hitter shares the reserved zero row, so counting one is counting the same
    embedding again -- and here the cold-start row is the one that would fail the check.
    """
    manifest = {"vocabulary": {"1": 1, "2": 2}}       # batter 3 is cold start
    predictions = pd.DataFrame({
        "batter": [1, 2, 3], "season": [2024] * 3, "p_throws": ["R"] * 3,
        "pred_woba": [0.31, 0.32, 0.90],
        "rate_bb": [0.08, 0.08, 0.50], "rate_k": [0.22, 0.22, 0.10],
        "rate_hbp": [0.01, 0.01, 0.05], "rate_bip": [0.69, 0.69, 0.35],
    })
    pa_df = pd.DataFrame({
        "batter": [1, 2, 3], "season": [2024] * 3, "p_throws": ["R"] * 3,
        "in_denominator": [True] * 3,
    })
    observed = {"bb": 0.08, "k": 0.22, "hbp": 0.01, "bip": 0.69}
    result = query.league_fidelity(predictions, pa_df, manifest, observed, 2024)
    assert result["n_hitters"] == 2, "the cold-start row was scored"
    assert result["pass"]
    assert result["rates"]["bb"]["relative_error"] == pytest.approx(0.0)


def test_league_fidelity_fails_outside_the_tolerance():
    """The HBP band is 20% and the rest 2% -- the same error must pass one and fail another."""
    manifest = {"vocabulary": {"1": 1}}
    predictions = pd.DataFrame({
        "batter": [1], "season": [2024], "p_throws": ["R"], "pred_woba": [0.31],
        "rate_bb": [0.08 * 1.10], "rate_k": [0.22], "rate_hbp": [0.01 * 1.10],
        "rate_bip": [1.0 - 0.08 * 1.10 - 0.22 - 0.01 * 1.10],
    })
    pa_df = pd.DataFrame({"batter": [1], "season": [2024], "p_throws": ["R"],
                          "in_denominator": [True]})
    observed = {"bb": 0.08, "k": 0.22, "hbp": 0.01, "bip": 0.69}
    result = query.league_fidelity(predictions, pa_df, manifest, observed, 2024)
    assert not result["rates"]["bb"]["pass"], "a 10% walk error passed a 2% band"
    assert result["rates"]["hbp"]["pass"], "a 10% HBP error failed a 20% band"
    assert not result["pass"]


# --- gate 3: the smoothing estimator -------------------------------------------------

def test_shrink_returns_the_pool_for_an_empty_cell():
    counts = np.array([[0, 0, 0], [10, 30, 60]], dtype="int64")
    shrunk = qt.shrink(counts, alpha=5.0)
    assert np.allclose(shrunk[0], [0.1, 0.3, 0.6])
    assert np.allclose(shrunk.sum(axis=1), 1.0)


def test_shrink_moves_a_thin_cell_further_than_a_thick_one():
    counts = np.array([[1, 0, 0], [1000, 0, 0], [0, 500, 500]], dtype="int64")
    shrunk = qt.shrink(counts, alpha=10.0)
    pool = counts.sum(axis=0) / counts.sum()
    thin = abs(shrunk[0, 0] - pool[0])
    thick = abs(shrunk[1, 0] - pool[0])
    assert thin < thick, "shrinkage did not weight the thick cell more heavily"


def test_shrink_is_a_no_op_at_zero_alpha():
    counts = np.array([[3, 1, 6], [10, 10, 30]], dtype="int64")
    shrunk = qt.shrink(counts, alpha=0.0)
    assert np.allclose(shrunk, counts / counts.sum(axis=1, keepdims=True))


def test_fit_alpha_is_finite_and_positive():
    rng = np.random.default_rng(7)
    counts = rng.poisson(30, size=(200, 3))
    alpha = qt.fit_alpha(counts)
    assert np.isfinite(alpha) and alpha > 0


# --- gate 4: the outcome table -------------------------------------------------------

def test_woba_points_table_uses_the_season_weights():
    table = np.zeros((2, 2, 2, len(qt.BIP_CLASSES)))
    table[..., qt.BIP_CLASSES.index("HR")] = 1.0
    weights = {"w1B": 0.9, "w2B": 1.2, "w3B": 1.5, "wHR": 2.0, "wBB": 0.7, "wHBP": 0.72}
    points = qt.woba_points_table(table, weights)
    assert np.allclose(points, 2.0), "an all-home-run cell must score the home-run weight"


def test_outs_score_zero():
    table = np.zeros((2, 2, 2, len(qt.BIP_CLASSES)))
    table[..., qt.BIP_CLASSES.index("OUT")] = 1.0
    weights = {"w1B": 0.9, "w2B": 1.2, "w3B": 1.5, "wHR": 2.0, "wBB": 0.7, "wHBP": 0.72}
    assert np.allclose(qt.woba_points_table(table, weights), 0.0)


# --- gate 5: determinism and eval-mode hygiene ---------------------------------------

def test_scoring_the_same_batch_twice_is_bit_identical():
    """
    The trunk ends in Dropout(0.1) and forward() does not set mode, so a model left in
    train mode would return a different number every call and nothing would raise.
    """
    models = build_models(n=2)
    hitter, context = sample_batch()
    points = random_points()
    first = query.expected_woba(models, hitter, context, points, N_BINS)
    second = query.expected_woba(models, hitter, context, points, N_BINS)
    for a, b in zip(first, second):
        assert np.array_equal(a, b), "repeat scoring is not bit-identical"


def test_a_model_left_in_train_mode_is_detectable():
    """The guard above is only meaningful if train mode actually changes the answer."""
    models = build_models(n=1)
    hitter, context = sample_batch(rows=64)
    points = random_points()
    evaluated = query.expected_woba(models, hitter, context, points, N_BINS)[2]
    models[0].train()
    torch.manual_seed(0)
    trained = query.expected_woba(models, hitter, context, points, N_BINS)[2]
    assert not np.array_equal(evaluated, trained), \
        "dropout is inactive, so the eval-mode gate above proves nothing"


# --- gate 6: split boundaries and pool hygiene ---------------------------------------

def synthetic_frame(rows=400, seed=1):
    rng = np.random.default_rng(seed)
    descriptions = rng.choice(["ball", "called_strike", "foul", "hit_into_play",
                               "swinging_strike", "hit_by_pitch", "intent_ball"], rows)
    return pd.DataFrame({
        "batter": rng.integers(1, 20, rows), "season": rng.choice([2015, 2024], rows),
        "pitcher": rng.integers(1, 6, rows), "description": descriptions,
        "zone": rng.integers(1, 15, rows), "balls": rng.integers(0, 4, rows),
        "strikes": rng.integers(0, 3, rows), "stand": rng.choice(["L", "R"], rows),
        "p_throws": rng.choice(["L", "R"], rows),
        "plate_x": rng.normal(0, 0.8, rows), "plate_z": rng.normal(2.4, 0.7, rows),
        "game_pk": np.arange(rows), "at_bat_number": np.arange(rows)})


def test_repertoire_excludes_intentional_balls():
    """
    IBB sits outside the wOBA denominator, so a simulated intentional ball would produce
    a plate appearance the metric never counts and pred_woba would need renormalising.
    """
    frame = synthetic_frame()
    train_mask = (frame["season"] == 2015).to_numpy()
    repertoire = qt.build_repertoire(frame, train_mask)
    kept = frame.iloc[repertoire.order]
    assert not kept["description"].isin(qt.EXCLUDED_DESCRIPTIONS).any()


def test_repertoire_holds_only_train_rows():
    frame = synthetic_frame()
    train_mask = (frame["season"] == 2015).to_numpy()
    repertoire = qt.build_repertoire(frame, train_mask)
    assert (frame.iloc[repertoire.order]["season"] == 2015).all(), \
        "the repertoire pool leaked an eval-season pitch"


def test_repertoire_cell_matches_the_key_it_was_filed_under():
    """
    Every sampled row must already carry the stand and count of the cell requested, which
    is the property that lets D.5 resample whole rows without overwriting any field.
    """
    frame = synthetic_frame(rows=4000, seed=9)
    train_mask = (frame["season"] == 2015).to_numpy()
    repertoire = qt.build_repertoire(frame, train_mask)
    for slot in range(repertoire.n_pitchers):
        for stand_slot in (0, 1):
            for balls in range(4):
                for strikes in range(3):
                    rows = repertoire.rows(slot, stand_slot, balls, strikes)
                    if len(rows) == 0:
                        continue
                    subset = frame.iloc[rows]
                    assert (subset["balls"] == balls).all()
                    assert (subset["strikes"] == strikes).all()
                    assert (subset["stand"] == ("R" if stand_slot else "L")).all()
                    assert subset["pitcher"].nunique() == 1


def test_take_surfaces_are_fit_on_train_rows_only():
    frame = synthetic_frame(rows=6000, seed=4)
    train_mask = (frame["season"] == 2015).to_numpy()
    surfaces, _ = qt.fit_take_surfaces(frame, train_mask)
    eval_only, _ = qt.fit_take_surfaces(frame, ~train_mask)
    key = ("R", "R")
    assert not np.allclose(surfaces[key], eval_only[key]), \
        "the take surface ignores its train mask"
    for surface in surfaces.values():
        assert np.allclose(surface.sum(axis=-1), 1.0)


def test_contact_split_normalises_over_its_three_classes():
    frame = synthetic_frame(rows=6000, seed=6)
    train_mask = (frame["season"] == 2015).to_numpy()
    split = qt.fit_contact_split(frame, train_mask)
    assert split.shape == (4, 3, 2, 3)
    assert np.allclose(split.sum(axis=-1), 1.0)


def test_stand_lookup_is_keyed_on_the_pitcher_hand():
    """
    A switch hitter stands on a different side depending on the pitcher, so a per-batter
    lookup would collapse him onto whichever side he faced more often.
    """
    pa = pd.DataFrame({"batter": [1, 1, 1, 1, 2, 2], "season": 2024,
                       "p_throws": ["L", "L", "R", "R", "R", "R"],
                       "stand": ["R", "R", "L", "L", "R", "R"]})
    lookup = query._stand_lookup(pa, 2024)
    assert lookup[(1, "L")] == "R"
    assert lookup[(1, "R")] == "L"
    assert lookup[(2, "R")] == "R"


# --- gate 7: the claim-1 comparison table -------------------------------------------

def synthetic_eval_frame(predictions, seed=0):
    """
    A minimal build_eval_frame output: what the paired bootstraps actually read.
    Two rows per batter, mirroring the real frame, so batter clustering has something
    to cluster.
    """
    rng = np.random.default_rng(seed)
    n = len(predictions)
    batters = np.repeat(np.arange(n // 2), 2)[:n]
    return pd.DataFrame({
        "batter": batters, "season": 2024,
        "p_throws": np.tile(["L", "R"], n // 2 + 1)[:n],
        "woba": rng.normal(0.32, 0.05, n),
        "denominator": rng.integers(20, 600, n).astype(float),
        "stratum": np.where(np.arange(n) < n // 3, "low",
                            np.where(np.arange(n) < 2 * n // 3, "medium", "high")),
        "pred_woba": predictions,
    })


def test_compare_returns_one_row_per_opponent_and_stratum():
    """
    The paired helpers already slice by stratum. Slicing again before calling them would
    resample inside a stratum and silently narrow every interval — a tighter, wrong
    answer that nothing downstream would flag.
    """
    from src.analysis import d5_report

    rng = np.random.default_rng(1)
    n = 120
    truth = rng.normal(0.32, 0.05, n)
    frames = {
        "phase_d_v1": synthetic_eval_frame(truth + rng.normal(0, 0.01, n)),
        "c3_gbm_full": synthetic_eval_frame(truth + rng.normal(0, 0.03, n), seed=0),
        "c2_bivariate": synthetic_eval_frame(truth + rng.normal(0, 0.04, n), seed=0),
    }
    table = d5_report.compare(frames, name="phase_d_v1", seed=0)

    assert set(table["opponent"]) == set(d5_report.GATE_OPPONENTS)
    for column in ("stratum", "rmse_difference", "rank_difference",
                   "rmse_favours_phase_d", "rank_favours_phase_d",
                   "ci_low_rmse", "ci_high_rmse", "ci_low_rank", "ci_high_rank"):
        assert column in table.columns, f"compare() dropped {column}"
    assert (table.groupby("opponent")["stratum"].nunique() == 4).all(), \
        "expected low/medium/high/all per opponent"


def test_compare_reads_the_interval_not_the_point_estimate():
    """
    The pre-registered gate is an interval excluding zero (2026-07-30). A table that
    flagged a favourable point estimate would pass models the gate rejects.
    """
    from src.analysis import d5_report

    rng = np.random.default_rng(2)
    n = 120
    truth = rng.normal(0.32, 0.05, n)
    # a model that is barely better: the point estimate favours it, the interval will not
    frames = {
        "phase_d_v1": synthetic_eval_frame(truth + rng.normal(0, 0.0299, n)),
        "c3_gbm_full": synthetic_eval_frame(truth + rng.normal(0, 0.03, n), seed=0),
        "c2_bivariate": synthetic_eval_frame(truth + rng.normal(0, 0.03, n), seed=0),
    }
    table = d5_report.compare(frames, name="phase_d_v1", seed=0)
    flagged = table["rmse_favours_phase_d"]
    assert (flagged == (table["ci_high_rmse"] < 0)).all(), \
        "the RMSE verdict is not reading the upper interval bound"
    assert (table["rank_favours_phase_d"] == (table["ci_low_rank"] > 0)).all(), \
        "the ordering verdict is not reading the lower interval bound"


# --- D5-R15: the in-play share V is not fit on ----------------------------------------

def outcome_fixture():
    """
    Ten balls in play: seven fully binned, three missing a spray bin. The three that `V` cannot
    be fit on all made outs, so the population hit worse than the retained subset -- which is the
    asymmetry that makes valuing every ball through `V` run high.
    """
    n = 10
    frame = pd.DataFrame({
        "game_pk": 1, "at_bat_number": np.arange(n), "season": 2016,
        "description": ["hit_into_play"] * n,
    })
    bins = {"ev": np.zeros(n, dtype="int64"), "la": np.zeros(n, dtype="int64"),
            "spray": np.array([0] * 7 + [qt.MASKED] * 3, dtype="int64")}
    pa_df = pd.DataFrame({
        "game_pk": 1, "at_bat_number": np.arange(n),
        # retained: 4 homers, 3 outs. excluded: 3 outs.
        "woba_category": ["HR"] * 4 + ["OUT"] * 3 + ["OUT"] * 3,
    })
    return frame, bins, pa_df, np.ones(n, dtype=bool)


def test_the_unmeasured_share_and_its_outcomes_are_measured_not_assumed():
    frame, bins, pa_df, train_mask = outcome_fixture()
    _, n_joined, unmeasured = qt.fit_outcome_table(frame, bins, train_mask, pa_df, n_bins=1)

    assert n_joined == 7
    assert unmeasured["n_unmeasured"] == 3
    assert unmeasured["measured_share"] == pytest.approx(0.7)
    # all three excluded balls made outs, so the distribution is entirely on class OUT
    assert unmeasured["categories"][0] == pytest.approx(1.0)
    assert sum(unmeasured["categories"]) == pytest.approx(1.0)


def test_the_unmeasured_value_uses_the_same_season_weights_as_v():
    """
    A wOBA constant here would mix train-season weights into an eval-season number. Expressing
    the unmeasured share as a CATEGORY distribution and contracting it with the same weight
    vector `woba_points_table` uses is what rules that out.
    """
    weights = {"wBB": 0.69, "wHBP": 0.72, "w1B": 0.88, "w2B": 1.25, "w3B": 1.6, "wHR": 2.05}
    tables = {"unmeasured": {"measured_share": 0.9,
                             "categories": [0.5, 0.3, 0.1, 0.0, 0.1]}}
    expected = 0.5 * 0.0 + 0.3 * 0.88 + 0.1 * 1.25 + 0.0 * 1.6 + 0.1 * 2.05
    assert query.unmeasured_points(tables, weights) == pytest.approx(expected)

    # and the same contraction as V's, so the two terms cannot drift into different units
    table = np.array([[[[0.5, 0.3, 0.1, 0.0, 0.1]]]])
    assert float(qt.woba_points_table(table, weights)[0, 0, 0]) == pytest.approx(expected)


def test_disabling_the_split_is_algebraically_inert_not_approximately_so():
    """
    The pre-fix number has to be reproducible EXACTLY, or the D5-R15 delta is not measurable.
    A share of 1.0 makes the second term vanish rather than merely shrink.
    """
    tables = {"unmeasured": {"measured_share": 0.9635, "categories": [1, 0, 0, 0, 0]}}
    weights = {"wBB": 0.69, "wHBP": 0.72, "w1B": 0.88, "w2B": 1.25, "w3B": 1.6, "wHR": 2.05}
    assert query._unmeasured_terms(tables, weights, enabled=False) == (1.0, 0.0)
    share, value = query._unmeasured_terms(tables, weights, enabled=True)
    assert share == pytest.approx(0.9635)

    quality = np.array([0.42, 0.55])
    unfixed = 1.0 * quality + (1.0 - 1.0) * 0.0
    assert np.array_equal(unfixed, quality), "the disabled branch changed the value"
    fixed = share * quality + (1.0 - share) * value
    assert (fixed < quality).all(), "the unmeasured share hit worse, so value must fall"


def test_a_table_without_the_unmeasured_key_falls_back_to_inert():
    """Diagnostics built before D5-R15 must keep scoring rather than raise."""
    assert query._unmeasured_terms({}, {"wHR": 2.05}, enabled=True) == (1.0, 0.0)
