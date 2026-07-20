"""
Unit tests for the context vectorizer. Synthetic fixtures only, no snapshot
dependency. These are the ml-engineer verification gates for a data-pipeline
change: shape assertions, the split-boundary/leakage gate (stats fit on train
only), the decode/eyeball gate, and determinism, plus encoding correctness.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import context_features as cf

PITCH_TYPES = ["FF", "SL", "CH", "CU", "SI"]


def make_df(n, season, rng, pitch_types=PITCH_TYPES, speed_center=90.0, spin_null_frac=0.0):
    """A synthetic pitch frame carrying every column the vectorizer reads."""
    frame = pd.DataFrame({
        "season": season,
        "release_speed": rng.normal(speed_center, 5.0, n),
        "plate_x": rng.normal(0.0, 0.8, n),
        "plate_z": rng.normal(2.5, 0.6, n),
        "pfx_x": rng.normal(0.0, 0.7, n),
        "pfx_z": rng.normal(1.0, 0.6, n),
        "effective_speed": rng.normal(speed_center, 5.0, n),
        "release_spin_rate": rng.normal(2200, 300, n),
        "release_extension": rng.normal(6.2, 0.3, n),
        "release_pos_x": rng.normal(-1.0, 0.5, n),
        "release_pos_y": rng.normal(54.0, 0.5, n),
        "release_pos_z": rng.normal(5.8, 0.3, n),
        "spin_axis": rng.uniform(0, 360, n),
        "pitch_type": rng.choice(pitch_types, n),
        "p_throws": rng.choice(["R", "L"], n),
        "stand": rng.choice(["R", "L"], n),
        "balls": rng.integers(0, 4, n),
        "strikes": rng.integers(0, 3, n),
    })
    if spin_null_frac > 0:
        mask = rng.random(n) < spin_null_frac
        frame.loc[mask, "spin_axis"] = np.nan
    return frame


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def train_df(rng):
    return make_df(2000, 2020, rng)


def test_feature_names_match_transform_order(train_df):
    params = cf.fit(train_df)
    X, names = cf.transform(train_df, params)
    assert names == cf.feature_names(params)
    assert X.shape[1] == len(names)


def test_shape_dtype_and_finite(train_df):
    params = cf.fit(train_df)
    X, names = cf.transform(train_df, params)
    assert X.shape == (len(train_df), len(names))
    assert X.dtype == np.float32
    assert np.isfinite(X).all()


def test_stats_fit_on_train_only(rng):
    # train speeds ~90, val speeds ~300; a leaked fit would pull the mean upward
    train = make_df(3000, 2023, rng, speed_center=90.0)
    val = make_df(3000, 2024, rng, speed_center=300.0)
    df = pd.concat([train, val], ignore_index=True)
    params = cf.fit_on_train(df)
    assert params.fit_seasons[-1] == 2023 and 2024 not in params.fit_seasons
    assert params.means["release_speed"] == pytest.approx(train["release_speed"].mean(), rel=1e-6)
    assert abs(params.means["release_speed"] - 90.0) < 5.0


def test_val_rows_use_train_stats(rng):
    train = make_df(3000, 2023, rng, speed_center=90.0)
    val = make_df(5, 2024, rng, speed_center=300.0)
    df = pd.concat([train, val], ignore_index=True)
    params = cf.fit_on_train(df)
    X, names = cf.transform(df, params)
    idx = names.index("release_speed")
    val_row = df.index[df["season"] == 2024][0]
    expected = (df.loc[val_row, "release_speed"] - params.means["release_speed"]) / params.stds["release_speed"]
    assert X[val_row, idx] == pytest.approx(expected, rel=1e-5)


def test_standardized_train_is_zero_mean_unit_std(train_df):
    params = cf.fit(train_df)
    X, names = cf.transform(train_df, params)
    for col in cf.CORE_CONTINUOUS:
        column = X[:, names.index(col)]
        assert column.mean() == pytest.approx(0.0, abs=1e-4)
        assert column.std() == pytest.approx(1.0, abs=1e-4)


def test_unknown_category_encodes_to_all_zero_block(train_df, rng):
    params = cf.fit(train_df)
    # a pitch type never seen in training
    unseen = make_df(10, 2020, rng, pitch_types=["ZZ"])
    X, names = cf.transform(unseen, params)
    pitch_cols = [i for i, n in enumerate(names) if n.startswith("pitch_type=")]
    assert (X[:, pitch_cols] == 0.0).all()
    # dimensionality is unchanged; no new column was created for ZZ
    assert not any(n == "pitch_type=ZZ" for n in names)


def test_missing_optional_is_filled_and_flagged(rng):
    train = make_df(2000, 2020, rng)
    params = cf.fit(train)
    row = make_df(1, 2020, rng).copy()
    row.loc[0, "release_spin_rate"] = np.nan
    X, names = cf.transform(row, params)
    assert X[0, names.index("release_spin_rate")] == 0.0
    assert X[0, names.index("release_spin_rate_missing")] == 1.0
    # a present optional field flags 0
    assert X[0, names.index("effective_speed_missing")] == 0.0


def test_spin_axis_sincos_and_circularity(rng):
    train = make_df(2000, 2020, rng)
    params = cf.fit(train)
    probe = make_df(3, 2020, rng)
    probe.loc[0, "spin_axis"] = 90.0
    probe.loc[1, "spin_axis"] = 0.0
    probe.loc[2, "spin_axis"] = 360.0
    X, names = cf.transform(probe, params)
    s, c = names.index("spin_axis_sin"), names.index("spin_axis_cos")
    assert X[0, s] == pytest.approx(1.0, abs=1e-5) and X[0, c] == pytest.approx(0.0, abs=1e-5)
    # 0 and 360 degrees are the same direction and must encode identically
    assert X[1, s] == pytest.approx(X[2, s], abs=1e-5)
    assert X[1, c] == pytest.approx(X[2, c], abs=1e-5)


def test_spin_axis_missing_zeroed_and_flagged(rng):
    train = make_df(2000, 2020, rng)
    params = cf.fit(train)
    row = make_df(1, 2020, rng)
    row.loc[0, "spin_axis"] = np.nan
    X, names = cf.transform(row, params)
    assert X[0, names.index("spin_axis_sin")] == 0.0
    assert X[0, names.index("spin_axis_cos")] == 0.0
    assert X[0, names.index("spin_axis_missing")] == 1.0


def test_transform_is_deterministic(train_df):
    params = cf.fit(train_df)
    X1, _ = cf.transform(train_df, params)
    X2, _ = cf.transform(train_df, params)
    assert np.array_equal(X1, X2)


def test_decode_recovers_category_and_value(train_df):
    params = cf.fit(train_df)
    X, names = cf.transform(train_df, params)
    decoded = cf.decode_sample(train_df, X, names, params, n=20)
    # the activated one-hot must equal the raw category for every sampled row
    assert (decoded["pitch_type_decoded"] == decoded["pitch_type_raw"]).all()
    assert (decoded["balls_decoded"].astype(int) == decoded["balls_raw"]).all()
    # de-standardized release_speed round-trips to the raw value
    assert np.allclose(decoded["release_speed_roundtrip"], decoded["release_speed_raw"], atol=0.01)


def test_excluded_columns_never_appear(train_df):
    params = cf.fit(train_df)
    names = cf.feature_names(params)
    for excluded in cf.EXCLUDED:
        assert not any(excluded in name for name in names), f"{excluded} leaked into the context vector"


def test_categorical_vocab_is_train_only(rng):
    train = make_df(2000, 2023, rng, pitch_types=["FF", "SL"])
    val = make_df(2000, 2024, rng, pitch_types=["FF", "SL", "KN"])
    df = pd.concat([train, val], ignore_index=True)
    params = cf.fit_on_train(df)
    assert "KN" not in params.categories["pitch_type"]
    assert set(params.categories["pitch_type"]) == {"FF", "SL"}
