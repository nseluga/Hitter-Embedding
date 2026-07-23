"""
Verification gates for the Phase B.2 GBM screen (ml-engineer blocking gates).

Fast synthetic checks: the feature grouping partitions the 48 columns, permutation
importance separates a signal group from noise, a fitted head beats its base rate,
label-shuffle collapses signal to chance, and the screen is deterministic under a
fixed seed. The heavy real-data gates (split-boundary, decode, base-rate on the
7.35M table) run in b2_screen.main().
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis import b2_screen as b2
from src.features import context_features as cf


def synthetic_context(n=400, seed=0):
    """A tiny frame with every column the vectorizer consumes, for fitting params."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "release_speed": rng.normal(93, 3, n),
        "plate_x": rng.normal(0, 1, n), "plate_z": rng.normal(2.5, 0.5, n),
        "pfx_x": rng.normal(0, 0.5, n), "pfx_z": rng.normal(1, 0.5, n),
        "effective_speed": rng.normal(93, 3, n),
        "release_spin_rate": rng.normal(2200, 200, n),
        "release_extension": rng.normal(6.5, 0.3, n),
        "release_pos_x": rng.normal(-1, 1, n), "release_pos_y": rng.normal(54, 1, n),
        "release_pos_z": rng.normal(6, 0.3, n),
        "spin_axis": rng.uniform(0, 360, n),
        "pitch_type": rng.choice(["FF", "SL", "CH"], n),
        "p_throws": rng.choice(["L", "R"], n),
        "stand": rng.choice(["L", "R"], n),
        "balls": rng.integers(0, 4, n), "strikes": rng.integers(0, 3, n),
    })
    return df


def fitted_params():
    df = synthetic_context()
    return cf.fit(df)


# ---- gate: feature grouping partitions the context vector exactly ----

def test_feature_groups_partition_all_columns():
    params = fitted_params()
    groups = b2.feature_groups(params)
    covered = sorted(i for cols in groups.values() for i in cols)
    assert covered == list(range(len(cf.feature_names(params))))


def test_optional_feature_travels_with_its_missing_flag():
    params = fitted_params()
    names = cf.feature_names(params)
    groups = b2.feature_groups(params)
    assert names.index("release_spin_rate_missing") in groups["release_spin_rate"]


# ---- gate: head nesting honors the §1.2 factorization ----

def test_head_target_nesting():
    df = pd.DataFrame({
        "swing": pd.array([1, 1, 0, 1], dtype="Int8"),
        "contact": pd.array([1, 0, pd.NA, 1], dtype="Int8"),
        "ev": [95.0, np.nan, np.nan, 80.0],
        "la": [10.0, np.nan, np.nan, 20.0],
        "spray": [15.0, np.nan, np.nan, np.nan],  # one in-play ball clip-nulled
    })
    swing_mask, y_swing = b2.head_target(df, "swing")
    assert swing_mask.sum() == 4 and set(y_swing) == {0.0, 1.0}
    whiff_mask, y_whiff = b2.head_target(df, "whiff")
    assert whiff_mask.sum() == 3                      # swings only
    assert y_whiff.tolist() == [0.0, 1.0, 0.0]        # row 1 is the miss
    spray_mask, y_spray = b2.head_target(df, "spray")
    assert spray_mask.sum() == 1                      # clip-nulled row excluded


# ---- gate: permutation importance separates signal from noise ----

class _Col0Model:
    """Prediction depends only on column 0, so only its group should matter."""
    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-4 * X[:, 0]))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return X[:, 0]


def test_permutation_importance_isolates_the_signal_group():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (2000, 4)).astype("float32")
    p = 1 / (1 + np.exp(-4 * X[:, 0]))
    y = (rng.uniform(size=2000) < p).astype("float32")
    groups = {"sig": [0], "noise": [1, 2, 3]}
    _, imp = b2.permutation_importance(_Col0Model(), X, y, groups, "clf", seed=0)
    assert imp["sig"][0] > 0.05
    assert abs(imp["noise"][0]) < imp["sig"][0] / 5


def test_permutation_importance_is_deterministic():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (1000, 4)).astype("float32")
    y = (X[:, 0] > 0).astype("float32")
    groups = {"sig": [0], "noise": [1, 2, 3]}
    a = b2.permutation_importance(_Col0Model(), X, y, groups, "clf", seed=7)
    b = b2.permutation_importance(_Col0Model(), X, y, groups, "clf", seed=7)
    assert a == b


# ---- gate: a fitted head beats base rate; label-shuffle collapses it ----

def test_fitted_head_beats_base_rate_and_shuffle_collapses():
    rng = np.random.default_rng(0)
    n = 6000
    X = rng.normal(0, 1, (n, 5)).astype("float32")
    p = 1 / (1 + np.exp(-3 * X[:, 0]))
    y = (rng.uniform(size=n) < p).astype("float32")
    tr, va = slice(0, 4000), slice(4000, n)

    model = b2.fit_head(X[tr], y[tr], X[va], y[va], "clf", seed=0)
    val = b2._metric(model, X[va], y[va], "clf")
    base = b2.base_rate_metric(y[va], "clf")
    assert val < base                                 # loss-scale sanity

    y_shuf = rng.permutation(y)
    shuf_model = b2.fit_head(X[tr], y_shuf[tr], X[va], y_shuf[va], "clf", seed=0)
    shuf_val = b2._metric(shuf_model, X[va], y_shuf[va], "clf")
    # signal gone: shuffled labels give no real edge over the base rate...
    assert shuf_val >= b2.base_rate_metric(y_shuf[va], "clf") - 0.02
    assert shuf_val > val                             # ...and are far worse than the true model
