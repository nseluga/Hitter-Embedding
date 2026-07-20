"""
Vectorize the per-pitch context `c` for the conditional model.

This module turns the labeled pitch table into the numeric context vector that
both the GBM feature screen (Phase B step 2) and, later, the DL context tower
(§2.1) consume. It produces `c` ONLY: the pitch context a query is posed
against (pitch characteristics, location, release, handedness, count). Hitter
identity `h` is deliberately absent, it is the hitter tower's job; pitcher
identity is absent too, it is the deferred pitcher-ID residual (§2.3).

Leakage discipline (frozen-split rule, §2.2): every fitted statistic, the
continuous mean/std and the categorical vocabulary, is learned on TRAIN seasons
only and then applied unchanged to val/test. `fit_on_train` bakes that boundary
in so the safe path is the default path. No feature is a function of any row
other than its own, so "no future-dated features" holds by construction, the
trailing-window history encoder that would introduce that risk is v2 (§2.3).

Missingness is explicit, never imputed silently: an absent optional reading is
filled to the train mean (0 after standardization) AND carries a `<field>_missing`
flag recomputed from the base field, so the flag can never drift from the fill.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

# --- candidate context registry (the pitch context `c`, §2.1) ---

# always present after cleaning (validate_core_context drops nulls); standardized
CORE_CONTINUOUS = ["release_speed", "plate_x", "plate_z", "pfx_x", "pfx_z"]
# secondary context; standardized, may be missing, always carries a flag
OPTIONAL_CONTINUOUS = ["effective_speed", "release_spin_rate", "release_extension",
                       "release_pos_x", "release_pos_y", "release_pos_z"]
# circular in [0, 360] degrees; encoded as sin/cos so the 0/360 wraparound closes
CIRCULAR = ["spin_axis"]
# categoricals one-hot encoded against a train-only vocabulary
CATEGORICAL = ["pitch_type", "p_throws", "stand", "balls", "strikes"]

# continuous features that get a (value - mean) / std transform
STANDARDIZED = CORE_CONTINUOUS + OPTIONAL_CONTINUOUS
# fields that get an explicit missingness flag (everything that can be null)
FLAG_SOURCES = OPTIONAL_CONTINUOUS + CIRCULAR

# columns deliberately kept OUT of the context vector, recorded so the choice is
# auditable rather than a silent omission
EXCLUDED = {
    "batter": "hitter identity -> hitter tower embedding, not context",
    "pitcher": "pitcher identity -> deferred pitcher-ID residual (§2.3), not v1 context",
    "zone": "deterministic coarse function of plate_x/plate_z, already included continuously",
    "bat_speed": "bat-tracking, excluded from v1 (2026-07-20 decision)",
    "swing_length": "bat-tracking, excluded from v1 (2026-07-20 decision)",
    "attack_angle": "bat-tracking, excluded from v1 (2026-07-20 decision)",
    "attack_direction": "bat-tracking, excluded from v1 (2026-07-20 decision)",
    "swing_path_tilt": "bat-tracking, excluded from v1 (2026-07-20 decision)",
}

# a standard deviation at or below this is treated as a constant feature; using 1.0
# then leaves the centered value untouched instead of dividing by ~0
STD_FLOOR = 1e-8


@dataclass
class VectorizerParams:
    """
    Fitted state of the context vectorizer, learned on train seasons only.
    means/stds: per standardized feature (optional-field stats over non-missing rows).
    categories: per categorical column, the sorted train vocabulary as strings.
    fit_seasons: provenance, which seasons the state was fit on.
    """
    means: dict
    stds: dict
    categories: dict
    fit_seasons: list


def fit(df):
    """
    Learn standardization stats and categorical vocabularies from `df`.
    Caller is responsible for passing train rows only; `fit_on_train` enforces it.
    Continuous stats for optional fields are computed over present values only.
    """
    means, stds = {}, {}
    for col in STANDARDIZED:
        values = df[col].to_numpy(dtype="float64")
        values = values[~np.isnan(values)]
        mean = float(values.mean())
        std = float(values.std())
        means[col] = mean
        stds[col] = std if std > STD_FLOOR else 1.0

    categories = {}
    for col in CATEGORICAL:
        categories[col] = sorted(str(v) for v in df[col].dropna().unique())

    return VectorizerParams(means=means, stds=stds, categories=categories, fit_seasons=[])


def fit_on_train(df, config=None):
    """
    Fit the vectorizer on the frozen train seasons only (leakage boundary).
    Selects train seasons from the split config, never val/test, and records them.
    """
    from src.config.splits import load_splits

    config = config or load_splits()
    train_seasons = sorted(config["split"]["train"])
    train = df[df["season"].isin(train_seasons)]
    params = fit(train)
    params.fit_seasons = train_seasons
    return params


def feature_names(params):
    """
    The ordered feature names produced by `transform`, computable from the fitted
    params alone. Consumers use this to align columns without touching the data.
    """
    names = list(STANDARDIZED)
    for col in CIRCULAR:
        names += [f"{col}_sin", f"{col}_cos"]
    names += [f"{field}_missing" for field in FLAG_SOURCES]
    for col in CATEGORICAL:
        names += [f"{col}={cat}" for cat in params.categories[col]]
    return names


def _standardize(series, mean, std):
    """Center/scale to train stats; missing values land at 0 (the train mean)."""
    values = series.to_numpy(dtype="float64")
    missing = np.isnan(values)
    standardized = (values - mean) / std
    standardized[missing] = 0.0
    return standardized


def _spin_sincos(series):
    """Encode a circular degree feature as (sin, cos); missing -> (0, 0)."""
    degrees = series.to_numpy(dtype="float64")
    missing = np.isnan(degrees)
    radians = np.deg2rad(degrees)
    sin = np.sin(radians)
    cos = np.cos(radians)
    sin[missing] = 0.0
    cos[missing] = 0.0
    return sin, cos


def _missing_flag(df, field):
    """Recompute the missingness flag from the base field so it matches the fill."""
    return df[field].isna().to_numpy(dtype="float64")


def _onehot(series, category):
    """1.0 where the (stringified) value equals `category`, else 0.0; NA -> 0.0."""
    return (series.astype("string") == category).fillna(False).to_numpy(dtype="float64")


def transform(df, params):
    """
    Vectorize `df` into the context matrix using fitted `params`.
    Returns (X, names): X is float32 of shape (len(df), len(names)); names is the
    column order from `feature_names`. Categories unseen in training encode to an
    all-zero block (implicit "other"), never a new column.
    """
    columns = []

    for col in STANDARDIZED:
        columns.append(_standardize(df[col], params.means[col], params.stds[col]))
    for col in CIRCULAR:
        sin, cos = _spin_sincos(df[col])
        columns.append(sin)
        columns.append(cos)
    for field in FLAG_SOURCES:
        columns.append(_missing_flag(df, field))
    for col in CATEGORICAL:
        for cat in params.categories[col]:
            columns.append(_onehot(df[col], cat))

    names = feature_names(params)
    assert len(columns) == len(names), f"column/name mismatch: {len(columns)} vs {len(names)}"
    X = np.column_stack(columns).astype("float32")
    assert np.isfinite(X).all(), "non-finite value in context matrix"
    return X, names


def inverse_standardize(value, mean, std):
    """Map a standardized value back to raw units (for the decode/eyeball gate)."""
    return value * std + mean


def decode_sample(df, X, names, params, n=8, seed=0):
    """
    Human-readable comparison of raw context against its encoding, for the eyeball
    gate. For each sampled row: the raw categorical values beside the category the
    one-hot block activates, and raw release_speed beside its de-standardized value.
    """
    name_index = {name: i for i, name in enumerate(names)}
    sample = df.sample(min(n, len(df)), random_state=seed)
    rows = []
    for pos, (_, row) in zip(sample.index, sample.iterrows()):
        x = X[df.index.get_loc(pos)]
        decoded = {}
        for col in CATEGORICAL:
            active = [cat for cat in params.categories[col] if x[name_index[f"{col}={cat}"]] == 1.0]
            decoded[f"{col}_raw"] = row[col]
            decoded[f"{col}_decoded"] = active[0] if active else "<other>"
        speed_z = x[name_index["release_speed"]]
        decoded["release_speed_raw"] = round(float(row["release_speed"]), 2)
        decoded["release_speed_roundtrip"] = round(
            float(inverse_standardize(speed_z, params.means["release_speed"], params.stds["release_speed"])), 2
        )
        rows.append(decoded)
    return pd.DataFrame(rows)


def save_params(params, path):
    """Serialize fitted params to readable JSON (reproducibility standard)."""
    Path(path).write_text(json.dumps(asdict(params), indent=2))


def load_params(path):
    """Load fitted params from JSON."""
    return VectorizerParams(**json.loads(Path(path).read_text()))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fit the context vectorizer on train and run its verification gates on real data.")
    parser.add_argument("--in-path", default="data/processed/pitch_events_labeled.parquet")
    parser.add_argument("--out-path", default="src/features/context_vectorizer_params.json")
    parser.add_argument("--sample", type=int, default=200_000, help="rows to transform for the gate report")
    args = parser.parse_args()

    columns = list(dict.fromkeys(
        STANDARDIZED + CIRCULAR + CATEGORICAL + ["season"]
    ))
    df = pd.read_parquet(args.in_path, columns=columns)

    params = fit_on_train(df)
    save_params(params, args.out_path)
    names = feature_names(params)

    print(f"fit seasons              {params.fit_seasons}")
    print(f"context vector dim        {len(names)}")
    for col in CATEGORICAL:
        print(f"vocab {col:12s}        {params.categories[col]}")

    sample = df.sample(min(args.sample, len(df)), random_state=0).reset_index(drop=True)
    X, names = transform(sample, params)
    print(f"sample matrix shape       {X.shape}")
    print(f"non-finite values         {int((~np.isfinite(X)).sum())}")
    print("\ndecode sample (eyeball gate):")
    print(decode_sample(sample, X, names, params).to_string(index=False))
    print(f"\nwrote fitted params to {args.out_path}")


if __name__ == "__main__":
    main()
