"""
Held-out version of the V.4 level query check (see model_visualization_heads.py).

The in-sample result (results/model_visualization/level_query.json) correlates the
model's level query against `woba_level`, which is computed from the same 2015-2023
pitches the embedding was trained on -- an in-sample comparison that cannot be
defended in review. This module reuses the exact same per-hitter level_query values
(results/model_visualization/level_query.csv, produced by build_level_query) and
correlates them against genuine held-out 2024 wOBA instead. Nothing here retrains
the model, touches src/model/, the loss, the data pipeline, MIN_EVAL_PA, or 2025
predictions -- it is read-only inference and analysis over an existing CSV plus
data/processed/eval_targets_pa.parquet.

Run: python -m src.analysis.level_query_heldout --out-dir results/paper_figures
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.claim1_eval import MIN_EVAL_PA
from src.data.eval_targets import aggregate, drop_pitcher_batters

LEVEL_QUERY_CSV = "results/model_visualization/level_query.csv"
EVAL_TARGETS_PARQUET = "data/processed/eval_targets_pa.parquet"
SUPERSEDES_PATH = "results/model_visualization/level_query.json"
HELDOUT_SEASON = 2024
BOOT_SEED = 7
N_BOOT = 1000


def ols_residualise(y, x):
    """Residuals of y after regressing on x plus an intercept (pairwise-complete)."""
    design = np.column_stack([np.ones_like(x), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coefficients


def paired_pearson(a, b, confound=None):
    """
    Pearson r on pairwise-complete rows of a and b. If `confound` is given, both a and b
    are residualised on it first (partial correlation), still pairwise-complete.
    Returns (r, n).
    """
    mask = np.isfinite(a) & np.isfinite(b)
    if confound is not None:
        mask &= np.isfinite(confound)
    a, b = a[mask], b[mask]
    n = mask.sum()
    if n < 3:
        return float("nan"), int(n)
    if confound is not None:
        c = confound[mask]
        a, b = ols_residualise(a, c), ols_residualise(b, c)
    r = float(np.corrcoef(a, b)[0, 1])
    return r, int(n)


def bootstrap_ci_partial(a, b, confound, n_boot=N_BOOT, seed=BOOT_SEED):
    """95% percentile bootstrap CI on the partial Pearson r, resampling hitters (rows)."""
    mask = np.isfinite(a) & np.isfinite(b) & np.isfinite(confound)
    a, b, confound = a[mask], b[mask], confound[mask]
    rng = np.random.default_rng(seed)
    n = len(a)
    point, _ = paired_pearson(a, b, confound)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[i], _ = paired_pearson(a[idx], b[idx], confound[idx])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi), int(n)


def load_heldout_woba(eval_targets_path=EVAL_TARGETS_PARQUET, season=HELDOUT_SEASON,
                       min_pa=MIN_EVAL_PA):
    """
    Per-hitter 2024 wOBA and PA count, same aggregation method (`aggregate`) and
    pitcher-batter exclusion (`drop_pitcher_batters`) the rest of the eval pipeline
    uses, filtered to that one season and to hitters with at least `min_pa` PA.
    """
    pa_df = pd.read_parquet(eval_targets_path)
    pa_df = drop_pitcher_batters(pa_df)
    season_pa = pa_df[pa_df["season"] == season]
    woba = aggregate(season_pa, by=("batter",))[["batter", "pa", "woba"]] \
        .rename(columns={"woba": "woba_2024", "pa": "pa_2024"})
    return woba[woba["pa_2024"] >= min_pa]


def build_heldout_frame(level_query_csv=LEVEL_QUERY_CSV, eval_targets_path=EVAL_TARGETS_PARQUET):
    """
    Joins the existing per-hitter level_query output to held-out 2024 wOBA on batter.
    Returns (merged, n_before, n_after) so the caller can report join loss.
    """
    level_query = pd.read_csv(level_query_csv)
    heldout = load_heldout_woba(eval_targets_path)
    n_before = len(level_query)
    merged = level_query.merge(heldout, on="batter", how="inner")
    return merged, n_before, len(merged)


def heldout_level_query_stats(merged):
    """Per-stratum + pooled Pearson (raw, partialled on log_prior_pa) with bootstrap CI."""
    out = {}
    confound_all = merged["log_prior_pa"].values.astype("float64")
    a_all = merged["level_query"].values.astype("float64")
    b_all = merged["woba_2024"].values.astype("float64")

    def _cell(a, b, confound):
        r_raw, _ = paired_pearson(a, b)
        r_partial, lo, hi, n = bootstrap_ci_partial(a, b, confound)
        return {"n": n, "r_raw": r_raw, "r_partial": r_partial, "ci_low": lo, "ci_high": hi}

    out["pooled"] = _cell(a_all, b_all, confound_all)
    for stratum, group in merged.groupby("stratum"):
        out[str(stratum)] = _cell(group["level_query"].values.astype("float64"),
                                  group["woba_2024"].values.astype("float64"),
                                  group["log_prior_pa"].values.astype("float64"))
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Held-out (2024 wOBA) version of the V.4 level query check.")
    parser.add_argument("--out-dir", default="results/paper_figures")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged, n_before, n_after = build_heldout_frame()
    n_lost = n_before - n_after
    print(f"level_query hitters: {n_before}, joined to 2024 wOBA (pa >= {MIN_EVAL_PA}): "
          f"{n_after}, lost: {n_lost}")

    merged.to_csv(out_dir / "level_query_heldout.csv", index=False)

    stats = heldout_level_query_stats(merged)
    stats["supersedes"] = SUPERSEDES_PATH
    stats["note"] = ("target changed from in-sample 2015-2023 woba_level to held-out "
                     f"{HELDOUT_SEASON} wOBA (min {MIN_EVAL_PA} PA); level_query values "
                     "themselves are reused unchanged from " + LEVEL_QUERY_CSV)
    stats["n_hitters_before_join"] = n_before
    stats["n_hitters_lost_in_join"] = n_lost
    (out_dir / "level_query_heldout.json").write_text(json.dumps(stats, indent=2))

    print(f"wrote {out_dir}/level_query_heldout.json, level_query_heldout.csv")


if __name__ == "__main__":
    main()
