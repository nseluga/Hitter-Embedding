"""
D.0 measurements — three facts that must be known before Phase D's dataset exists.

1. REDUNDANCY of B.2's six flagged context features against the ten it kept.
   B.2 flagged the six on out-of-sample permutation importance, then declined to
   drop them because permutation importance is deflated by collinearity and the
   six ARE the collinear group. This measures the collinearity directly: regress
   each flagged feature on the kept ten and read the R^2.

   The test is deliberately ASYMMETRIC. A high R^2 means the kept features already
   determine the flagged one, so it can be dropped on a measurement rather than on
   an argument. A low R^2 proves nothing — OLS understates what a nonlinear trunk
   can reconstruct, so a feature that survives here has NOT been shown to carry
   information. Surviving features go to the D.8 block ablation; that ablation, not
   this file, is what admits a feature to the model under the §4 rule.

2. COLD-START SHARE on the 2025 report frame against the 2024 selection frame.
   Under the 2026-08-01 selection-frame decision the model trains through 2023 and
   reports on 2025, so every hitter debuting in 2024 or 2025 routes to the reserved
   untrained embedding row (2026-07-30). That population was 43% of the low stratum
   on the 2024 frame; the gap to the report frame is now two seasons instead of one,
   and the size of the increase is what decides whether the zero-row treatment is
   still tenable.

3. FRAME STRUCTURE on 2025 — rows and hitters per exposure stratum. This is the
   resolving power Phase D will be graded with, and it is knowable in advance
   because strata are assigned on PRIOR exposure. No model is scored here: the
   predictions passed to the frame builder are a fixed baseline used only to
   obtain the row set, and row counts and strata do not depend on them.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import c1_trailing as c1
from src.analysis import claim1_eval as evaluation
from src.config.splits import load_splits

OUT_DIR = Path("results/phase_d")

# The ten groups B.2 kept, as they enter the context tower. Every flagged feature
# is regressed on exactly this set — nothing else is available to make it redundant.
KEPT_CONTINUOUS = ["plate_x", "plate_z", "pfx_x", "pfx_z", "release_speed", "release_pos_x"]
KEPT_CATEGORICAL = ["pitch_type", "stand", "p_throws", "balls", "strikes"]

# The six B.2 left UNDECIDED. spin_axis is circular, so it is tested as its sine and
# cosine components; both must be recoverable for the angle itself to be redundant.
FLAGGED_LINEAR = ["effective_speed", "release_extension", "release_pos_y",
                  "release_pos_z", "release_spin_rate"]
FLAGGED_CIRCULAR = "spin_axis"

REDUNDANCY_SAMPLE = 1_000_000   # rows drawn from TRAIN seasons; R^2 is stable well below this
REDUNDANCY_SEED = 0

# a flagged feature at or above this is determined by the kept ten and drops here
REDUNDANT_R2 = 0.99


def load_pitches(path, columns):
    return pd.read_parquet(path, columns=columns)


def design_matrix(df):
    """
    Build the predictor matrix from B.2's ten kept feature groups.
    df: pitch rows. Returns a float array with an intercept column appended.
    """
    blocks = [df[KEPT_CONTINUOUS].to_numpy(dtype=float)]
    for col in KEPT_CATEGORICAL:
        dummies = pd.get_dummies(df[col].astype("category"), drop_first=True)
        blocks.append(dummies.to_numpy(dtype=float))
    blocks.append(np.ones((len(df), 1)))
    return np.hstack(blocks)


def _r2(predictors, target):
    """Ordinary least-squares R^2 of target on predictors. Both must be finite."""
    coefficients, *_ = np.linalg.lstsq(predictors, target, rcond=None)
    residual = target - predictors @ coefficients
    total = target - target.mean()
    return float(1.0 - residual @ residual / (total @ total))


def redundancy_r2(pitch_df, train_seasons, extra_predictors=(),
                  sample_n=REDUNDANCY_SAMPLE, seed=REDUNDANCY_SEED):
    """
    R^2 of each B.2-flagged feature regressed on the ten kept features.
    pitch_df: labeled pitch table; train_seasons: the frozen train seasons;
    extra_predictors: further columns to condition on, for the case where one
    flagged feature is only redundant GIVEN another (perceived velocity is
    velocity plus extension, so effective_speed cannot be tested without it).
    Returns one row per flagged feature with its R^2 and the rows it was fit on.
    """
    train = pitch_df[pitch_df["season"].isin(train_seasons)]
    assert len(train) > 0, "no train-season rows in the pitch table"
    if len(train) > sample_n:
        train = train.sample(n=sample_n, random_state=seed)

    extra_predictors = list(extra_predictors)
    targets = {name: train[name].to_numpy(dtype=float)
               for name in FLAGGED_LINEAR if name not in extra_predictors}
    if FLAGGED_CIRCULAR not in extra_predictors:
        radians = np.deg2rad(train[FLAGGED_CIRCULAR].to_numpy(dtype=float))
        targets[f"{FLAGGED_CIRCULAR}_sin"] = np.sin(radians)
        targets[f"{FLAGGED_CIRCULAR}_cos"] = np.cos(radians)

    rows = []
    for name, target in targets.items():
        # the flagged features are optional context, so each regression uses the
        # rows where its own target and every predictor are present
        present = train[KEPT_CONTINUOUS + extra_predictors].notna().all(axis=1).to_numpy()
        usable = np.isfinite(target) & present
        subset = train.loc[usable]
        predictors = design_matrix(subset)
        if extra_predictors:
            predictors = np.hstack([predictors, subset[extra_predictors].to_numpy(dtype=float)])
        r2 = _r2(predictors, target[usable])
        rows.append({"feature": name, "r2": r2, "n_rows": int(usable.sum()),
                     "verdict": "redundant" if r2 >= REDUNDANT_R2 else "survives to D.8"})
    return pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)


def frame_structure(pa_df, pitch_df, eval_season, vocabulary_seasons):
    """
    Per-stratum row counts and cold-start share for one evaluation frame.
    A hitter is cold-start if he never appears in the modeling pitch table across
    vocabulary_seasons, which is exactly the seasons the hitter embedding table is
    built from. Returns one row per exposure stratum plus an "all" row.
    """
    predictions = c1.predict(pa_df, eval_season, variant="bucketed")
    frame, _ = evaluation.build_eval_frame(pa_df, predictions, eval_season)

    vocabulary = set(pitch_df.loc[pitch_df["season"].isin(vocabulary_seasons), "batter"].unique())
    frame = frame.assign(cold_start=~frame["batter"].isin(vocabulary))

    def summarize(part, label):
        return {"stratum": label, "rows": len(part),
                "hitters": int(part["batter"].nunique()),
                "cold_start_rows": int(part["cold_start"].sum()),
                "cold_start_share": float(part["cold_start"].mean())}

    rows = [summarize(part, str(label)) for label, part in frame.groupby("stratum", observed=True)]
    rows.append(summarize(frame, "all"))
    return pd.DataFrame(rows)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run the Phase D.0 measurements.")
    parser.add_argument("--eval-targets", default="data/processed/eval_targets_pa.parquet")
    parser.add_argument("--pitch-events", default="data/processed/pitch_events_labeled.parquet")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    train_seasons = load_splits()["split"]["train"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    redundancy_columns = (["season", FLAGGED_CIRCULAR] + FLAGGED_LINEAR
                          + KEPT_CONTINUOUS + KEPT_CATEGORICAL)
    pitch_df = load_pitches(args.pitch_events, redundancy_columns)
    redundancy = redundancy_r2(pitch_df, train_seasons)
    # perceived velocity is velocity adjusted for extension, so effective_speed is
    # only testable for redundancy once extension is available to predict it
    conditional = redundancy_r2(pitch_df, train_seasons, extra_predictors=["release_extension"])
    conditional = conditional.assign(given="release_extension")
    redundancy = pd.concat([redundancy.assign(given="kept ten only"), conditional])
    redundancy.to_csv(out_dir / "d0_redundancy_r2.csv", index=False)
    print("redundancy of B.2's flagged features against its kept ten")
    print(redundancy.to_string(index=False, float_format="%.4f"))

    pa_df = pd.read_parquet(args.eval_targets)
    batter_seasons = load_pitches(args.pitch_events, ["batter", "season"])
    # 2024 is the selection frame, 2025 the report frame. The third row is 2025 scored
    # by a model whose vocabulary also covers 2024 -- the cold-start cost of skipping
    # the refit, which is what the 2026-08-01 selection-frame decision turns on.
    cases = [(2024, train_seasons), (2025, train_seasons),
             (2025, train_seasons + [2024])]
    structures = []
    for eval_season, vocabulary_seasons in cases:
        structure = frame_structure(pa_df, batter_seasons, eval_season, vocabulary_seasons)
        label = f"{eval_season} (trained through {max(vocabulary_seasons)})"
        structures.append(structure.assign(case=label))
        print(f"\nframe structure, eval season {label}")
        print(structure.to_string(index=False, float_format="%.4f"))
    pd.concat(structures).to_csv(out_dir / "d0_frame_structure.csv", index=False)


if __name__ == "__main__":
    main()
