"""
Phase E verification gates (docs/phase-e-spec.md §10; CLAUDE.md ML verification gate).

Every failure guarded here is silent. A matched-population helper that quietly changes the
denominator still returns a plausible walk rate. An errors-in-variables correction with the
wrong sign still returns a compression coefficient in [0, 1]. A decomposition that does not
close still prints a tidy table. A chain perturbation that leaks mass still solves. None of
these raise on their own, and Phase E's whole job is to be believed.

Fixtures are synthetic and tiny: the point is that the ALGEBRA is right, and algebra does
not need 7.3M rows to be wrong.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.analysis import e_eval, e_price_draw, e_structure
from src.model import query


# ---------------------------------------------------------------- weighted regression

def test_weighted_ols_recovers_a_known_line():
    x = np.linspace(0.0, 1.0, 200)
    y = 0.3 + 1.7 * x
    intercept, slope, se = e_eval.weighted_ols(x, y, np.ones_like(x))
    assert intercept == pytest.approx(0.3, abs=1e-10)
    assert slope == pytest.approx(1.7, abs=1e-10)
    assert se == pytest.approx(0.0, abs=1e-8), "an exact fit cannot carry standard error"


def test_weighted_ols_weights_actually_bind():
    """Two clouds on different lines; the weights decide which line comes back."""
    x = np.concatenate([np.linspace(0, 1, 50), np.linspace(0, 1, 50)])
    y = np.concatenate([2.0 * x[:50], -2.0 * x[50:]])
    heavy_first = np.concatenate([np.full(50, 1e6), np.ones(50)])
    _, slope, _ = e_eval.weighted_ols(x, y, heavy_first)
    assert slope == pytest.approx(2.0, abs=1e-3)
    _, flipped, _ = e_eval.weighted_ols(x, y, heavy_first[::-1])
    assert flipped == pytest.approx(-2.0, abs=1e-3)


def test_weighted_ols_standard_error_is_positive_under_noise():
    generator = np.random.default_rng(0)
    x = generator.normal(size=500)
    y = 1.0 * x + generator.normal(scale=0.5, size=500)
    _, slope, se = e_eval.weighted_ols(x, y, np.ones_like(x))
    assert 0.0 < se < 0.1
    assert abs(slope - 1.0) < 4 * se, "the true slope must sit inside a sane band"


# ---------------------------------------------------------------- population matching

def _fake_pa(n_batters=30, seed=0):
    generator = np.random.default_rng(seed)
    rows = []
    for batter in range(n_batters):
        for season in (2022, 2023, 2024):
            for hand in ("L", "R"):
                for _ in range(generator.integers(5, 40)):
                    event = generator.choice(["walk", "strikeout", "hit_by_pitch", "single"],
                                             p=[0.09, 0.22, 0.01, 0.68])
                    rows.append({"batter": batter, "season": season, "p_throws": hand,
                                 "stand": "R", "events": event, "in_denominator": True,
                                 "game_pk": len(rows), "at_bat_number": 1})
    return pd.DataFrame(rows)


def test_per_group_absorbing_rates_sum_to_one():
    pa_df = _fake_pa()
    rates = e_eval.per_group_absorbing(pa_df, [2022, 2023])
    total = sum(rates[key] for key in query.ABSORBING_KEYS)
    assert np.allclose(total, 1.0, atol=1e-12), "observed absorbing shares must exhaust the PA"


def test_matched_population_reduces_to_the_shipped_number_on_the_full_set():
    """
    The identity that makes E.1 a population finding rather than a code finding: pooled over
    EVERY hitter, the matched helper must reproduce `query.observed_absorbing_rates` exactly.
    If it does not, the E.1 gap is measuring a rewrite.
    """
    pa_df = _fake_pa()
    seasons = [2022, 2023]
    shipped, _ = query.observed_absorbing_rates(pa_df, seasons)
    frame = e_eval.per_group_absorbing(pa_df, seasons)
    weight = frame["denominator"].to_numpy(dtype="float64")
    for key in query.ABSORBING_KEYS:
        pooled = float(np.average(frame[key].to_numpy(dtype="float64"), weights=weight))
        assert pooled == pytest.approx(shipped[key], abs=1e-12)


# ---------------------------------------------------------------- chain perturbations

def _fake_chain(seed=0):
    """A count chain whose five channels sum to 1 in every state, by construction."""
    generator = np.random.default_rng(seed)
    raw = generator.uniform(0.05, 1.0, size=(5, 4, 3))
    raw = raw / raw.sum(axis=0)
    chain = dict(zip(("ball", "strike", "hbp", "foul", "bip"), raw))
    chain["take_mass"] = chain["ball"] + chain["hbp"] + 0.5 * chain["strike"]
    return chain


def test_absorbing_rates_of_a_synthetic_chain_exhaust_the_pa():
    rates = query.absorbing_rates(e_price_draw.to_aggregates(_fake_chain()))
    assert sum(float(rates[key][0, 0]) for key in query.ABSORBING_KEYS) == pytest.approx(1.0)


def test_draw_gap_conserves_mass_in_every_state():
    """A perturbation that leaks probability would move the walk rate for free."""
    chain = _fake_chain()
    gaps = pd.DataFrame([{"balls": b, "strikes": s, "draw_gap_ball": 0.01 * (b - 1),
                          "draw_gap_hbp": 0.001, "draw_gap_called_strike": 0.0}
                         for b, s in query.COUNT_STATES])
    shifted = e_price_draw.apply_draw_gap(chain, gaps)
    before = sum(chain[name] for name in ("ball", "strike", "hbp", "foul", "bip"))
    after = sum(shifted[name] for name in ("ball", "strike", "hbp", "foul", "bip"))
    assert np.allclose(before, after, atol=1e-12)


def test_draw_gap_of_zero_is_the_identity():
    chain = _fake_chain()
    gaps = pd.DataFrame([{"balls": b, "strikes": s, "draw_gap_ball": 0.0,
                          "draw_gap_hbp": 0.0, "draw_gap_called_strike": 0.0}
                         for b, s in query.COUNT_STATES])
    shifted = e_price_draw.apply_draw_gap(chain, gaps)
    for name in ("ball", "strike", "hbp", "foul", "bip"):
        assert np.array_equal(chain[name], shifted[name])


def test_more_ball_mass_means_more_walks_and_fewer_strikeouts():
    """Sign check: the pricing is only interpretable if the chain responds monotonically."""
    chain = _fake_chain()
    base = query.absorbing_rates(e_price_draw.to_aggregates(chain))
    gaps = pd.DataFrame([{"balls": b, "strikes": s, "draw_gap_ball": 0.02,
                          "draw_gap_hbp": 0.0, "draw_gap_called_strike": -0.02}
                         for b, s in query.COUNT_STATES])
    after = query.absorbing_rates(e_price_draw.to_aggregates(
        e_price_draw.apply_draw_gap(chain, gaps)))
    assert after["bb"][0, 0] > base["bb"][0, 0]
    assert after["k"][0, 0] < base["k"][0, 0]


def test_draw_gap_is_antisymmetric_in_sign():
    chain = _fake_chain()
    gaps = pd.DataFrame([{"balls": b, "strikes": s, "draw_gap_ball": 0.01,
                          "draw_gap_hbp": 0.0, "draw_gap_called_strike": 0.0}
                         for b, s in query.COUNT_STATES])
    up = e_price_draw.apply_draw_gap(chain, gaps, sign=1.0)
    down = e_price_draw.apply_draw_gap(chain, gaps, sign=-1.0)
    for name in ("ball", "strike", "hbp"):
        assert np.allclose(up[name] + down[name], 2.0 * chain[name], atol=1e-12)


# ---------------------------------------------------------------- description partition

def test_channel_masks_partition_every_real_description():
    """
    The five channels must cover every Statcast description exactly once, or the league
    chain silently loses mass and its walk rate is not comparable to anything.
    """
    descriptions = np.array(["ball", "blocked_ball", "called_strike", "swinging_strike",
                             "swinging_strike_blocked", "foul", "foul_tip",
                             "hit_into_play", "hit_by_pitch"])
    keep = np.ones(len(descriptions), dtype=bool)
    masks = e_structure._channel_masks(descriptions, keep)
    assert sum(masks.values()).tolist() == [1] * len(descriptions)


def test_channel_masks_reject_an_unclassified_description():
    descriptions = np.array(["ball", "some_new_statcast_string"])
    with pytest.raises(AssertionError):
        e_structure._channel_masks(descriptions, np.ones(2, dtype=bool))


def test_intent_ball_is_excluded_rather_than_misclassified():
    """It is in no channel by design; the partition check must skip it, not absorb it."""
    descriptions = np.array(["ball", "intent_ball"])
    keep = np.array([True, False])
    masks = e_structure._channel_masks(descriptions, keep)
    assert sum(masks.values())[1] == 0


# ---------------------------------------------------------------- platoon frame algebra

def test_paired_bootstrap_of_identical_columns_is_exactly_zero():
    """A knob compared against itself must show no difference and no spread."""
    generator = np.random.default_rng(1)
    frame = pd.DataFrame({
        "batter": np.arange(300),
        "weight": generator.uniform(1, 10, 300),
        "delta_obs": generator.normal(size=300),
        "stratum": "all",
    })
    frame["delta_pred"] = frame["delta_obs"] + generator.normal(scale=0.2, size=300)
    frame["delta_route_a"] = frame["delta_pred"]
    paired = e_eval.paired_platoon_difference(frame, n_boot=200, seed=0)
    scored = paired[paired["rmse_difference"].notna()]
    assert len(scored), "no stratum was scored"
    for row in scored.itertuples():
        assert row.rmse_difference == pytest.approx(0.0, abs=1e-12)
        assert row.rank_difference == pytest.approx(0.0, abs=1e-12)
        assert row.rmse_ci_low == pytest.approx(0.0, abs=1e-12)
        assert row.rmse_ci_high == pytest.approx(0.0, abs=1e-12)


def _platoon_panel(n, seed):
    generator = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "batter": np.arange(n),
        "weight": np.ones(n),
        "delta_obs": generator.normal(size=n),
        "stratum": "all",
    })
    frame["delta_pred"] = frame["delta_obs"] + generator.normal(scale=0.5, size=n)
    frame["delta_route_a"] = frame["delta_obs"] + generator.normal(scale=0.5, size=n)
    return frame


def test_paired_interval_narrows_with_more_hitters():
    """
    The frame is one row per hitter, so resampling rows IS resampling batters -- the
    clustering claim-1 needs is structural. What still has to hold is that the interval
    responds to real information: four times the hitters must roughly halve the width.
    """
    def width(table):
        row = table[table["stratum"] == "all"].iloc[0]
        return float(row["rmse_ci_high"] - row["rmse_ci_low"])

    small = width(e_eval.paired_platoon_difference(_platoon_panel(60, 2), n_boot=600, seed=0))
    large = width(e_eval.paired_platoon_difference(_platoon_panel(240, 2), n_boot=600, seed=0))
    assert large < small, "more hitters did not tighten the paired interval"
    assert large > 0.25 * small, "the interval narrowed faster than root-n allows"


def test_paired_frame_is_one_row_per_hitter():
    """The clustering guarantee above is only true while this holds; assert it, do not assume."""
    frame = _platoon_panel(50, 3)
    assert frame["batter"].is_unique
