"""
Build the claim-1 evaluation target: side-specific wOBA from a complete source.

This is the ground truth every model is graded against, so it is built from the
COMPLETE regular-season outcome record, NOT the modeling table. The modeling
filters (position-player pitchers, non-competitive pitches, mistracks, tracking
validity) deliberately sharpen the training set; applying them here would bias
the target. Per the two-table principle (decision log 2026-07-15) filters apply
only to the modeling table, never to eval targets, so this reads the raw
snapshot with regular-season the only restriction.

wOBA (FanGraphs formula, src/config/woba_weights.json):
  wOBA = (wBB*uBB + wHBP*HBP + w1B*1B + w2B*2B + w3B*3B + wHR*HR)
         / (AB + BB - IBB + SF + HBP)
Weights are season-specific. Intentional walks (events == "intent_walk") are a
distinct outcome: not credited in the numerator and removed from the denominator.

The primitive is a PA-level table: one row per completed plate appearance with
its numerator contribution (`woba_points`) and an `in_denominator` flag, so the
wOBA of ANY set of PAs is sum(woba_points) / sum(in_denominator). Stabilization
resamples PAs within a hitter; `aggregate` rolls up to the claim-1 target table.

Validation: the computed per-season LEAGUE wOBA must reproduce the published
league wOBA (the reconciliation guard). If the weights or event mapping are
wrong, the identity fails loud rather than biasing every target silently.
"""

import json
from pathlib import Path

import pandas as pd

DEFAULT_WEIGHTS_PATH = Path(__file__).parents[1] / "config" / "woba_weights.json"

# raw columns needed to identify a PA and its outcome/handedness
PA_COLUMNS = ["game_pk", "at_bat_number", "batter", "pitcher", "p_throws", "stand", "events", "game_type"]

# non-batting terminal rows dropped before mapping: truncated_pa is a PA ended by a
# baserunning event (e.g. caught stealing to end the inning), game_advisory is an
# administrative note. Neither is a completed batting outcome.
NON_BATTING_EVENTS = {"truncated_pa", "game_advisory"}

# every completed-PA `events` value -> a wOBA outcome category. A value outside this
# map is schema drift and must not be silently miscategorized (fail loud).
EVENT_TO_CATEGORY = {
    "single": "1B", "double": "2B", "triple": "3B", "home_run": "HR",
    "walk": "uBB", "intent_walk": "IBB", "hit_by_pitch": "HBP",
    "sac_fly": "SF", "sac_fly_double_play": "SF",
    "sac_bunt": "SH", "sac_bunt_double_play": "SH",
    "catcher_interf": "INT",
    "field_out": "OUT", "strikeout": "OUT", "force_out": "OUT",
    "grounded_into_double_play": "OUT", "field_error": "OUT", "double_play": "OUT",
    "fielders_choice": "OUT", "fielders_choice_out": "OUT",
    "strikeout_double_play": "OUT", "triple_play": "OUT",
}

# category -> the weight key that scores it; categories absent here score 0
CATEGORY_WEIGHT_KEY = {"1B": "w1B", "2B": "w2B", "3B": "w3B", "HR": "wHR", "uBB": "wBB", "HBP": "wHBP"}

# categories NOT in the wOBA denominator: intentional walks, sac bunts, interference
NON_DENOMINATOR_CATEGORIES = {"IBB", "SH", "INT"}

# loud guard: computed league wOBA must land within this of the published value
LEAGUE_WOBA_TOLERANCE = 0.005


def load_weights(path=DEFAULT_WEIGHTS_PATH):
    """Read the season wOBA weights; returns the {season_str: {weight: value}} map."""
    return json.loads(Path(path).read_text())["weights"]


def load_pa_terminals(snapshot_dir, seasons=None):
    """
    Read the raw snapshot and return one row per completed regular-season PA.
    Keeps the terminal pitch of each PA (`events` non-null), drops truncated PAs,
    and adds a `season` column. No modeling filters are applied.
    """
    import pyarrow.parquet as pq

    snapshot_dir = Path(snapshot_dir)
    frames = []
    for path in sorted(snapshot_dir.glob("season=*.parquet")):
        season = int(path.stem.split("=")[1])
        if seasons is not None and season not in seasons:
            continue
        available = set(pq.ParquetFile(path).schema.names)
        frame = pd.read_parquet(path, columns=sorted(set(PA_COLUMNS) & available))
        frame = frame[frame["game_type"] == "R"]
        frame = frame[frame["events"].notna() & ~frame["events"].isin(NON_BATTING_EVENTS)]
        frame["season"] = season
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def categorize(df):
    """Map each PA's `events` to a wOBA category; raise on any unmapped event value."""
    unknown = set(df["events"].unique()) - set(EVENT_TO_CATEGORY)
    if unknown:
        raise ValueError(f"unknown events, cannot categorize for wOBA: {sorted(unknown)}")
    df = df.copy()
    df["woba_category"] = df["events"].map(EVENT_TO_CATEGORY)
    return df


def add_woba_points(df, weights):
    """
    Add `woba_points` (season-weighted numerator contribution) and `in_denominator`.
    Points are 0 except for the six credited categories; the denominator excludes
    intentional walks, sac bunts, and interference.
    """
    df = df.copy()
    season_str = df["season"].astype(str)
    points = pd.Series(0.0, index=df.index)
    for category, weight_key in CATEGORY_WEIGHT_KEY.items():
        mask = df["woba_category"] == category
        points[mask] = season_str[mask].map(lambda s: weights[s][weight_key])
    df["woba_points"] = points
    df["in_denominator"] = ~df["woba_category"].isin(NON_DENOMINATOR_CATEGORIES)
    return df


def build(snapshot_dir, seasons=None, weights=None):
    """Run the full PA-level pipeline and return the eval-target primitive table."""
    weights = weights or load_weights()
    df = load_pa_terminals(snapshot_dir, seasons)
    df = categorize(df)
    df = add_woba_points(df, weights)
    return df


def aggregate(pa_df, by=("batter", "season", "p_throws")):
    """
    Roll the PA-level table up to wOBA per group (default: hitter x season x pitcher hand).
    wOBA = sum(woba_points) / sum(in_denominator); pa is the group's completed-PA count.
    """
    by = list(by)
    grouped = pa_df.groupby(by, sort=True)
    out = grouped.agg(pa=("woba_points", "size"),
                      denominator=("in_denominator", "sum"),
                      woba_points=("woba_points", "sum")).reset_index()
    out["woba"] = out["woba_points"] / out["denominator"]
    return out


def reconcile(pa_df, weights):
    """
    Per-season reconciliation report and the league-wOBA guard. Computed league
    wOBA must match the published value within tolerance, else a weight or mapping
    bug is poisoning every target and we fail loud.
    """
    report = {}
    for season, sub in pa_df.groupby("season"):
        season_str = str(int(season))
        denom = int(sub["in_denominator"].sum())
        computed = float(sub["woba_points"].sum() / denom)
        published = weights[season_str]["league_woba"]
        report[season_str] = {
            "n_pa": int(len(sub)),
            "denominator": denom,
            "n_ibb": int((sub["woba_category"] == "IBB").sum()),
            "computed_league_woba": round(computed, 4),
            "published_league_woba": published,
            "diff": round(computed - published, 4),
        }
        assert abs(computed - published) <= LEAGUE_WOBA_TOLERANCE, (
            f"{season_str}: league wOBA {computed:.4f} off published {published:.4f} "
            f"by {computed - published:+.4f} (> {LEAGUE_WOBA_TOLERANCE})"
        )
    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build the side-specific wOBA eval-target table from the raw snapshot.")
    parser.add_argument("--snapshot-dir", default="data/raw/statcast/snapshot_2026-07-14")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--seasons", type=int, nargs="+", default=None)
    args = parser.parse_args()

    weights = load_weights()
    pa_df = build(args.snapshot_dir, args.seasons, weights)
    report = reconcile(pa_df, weights)

    print(f"{'season':7s} {'n_pa':>8s} {'denom':>8s} {'IBB':>5s} {'computed':>9s} {'published':>10s} {'diff':>7s}")
    for season, row in report.items():
        print(f"{season:7s} {row['n_pa']:>8d} {row['denominator']:>8d} {row['n_ibb']:>5d} "
              f"{row['computed_league_woba']:>9.4f} {row['published_league_woba']:>10.3f} {row['diff']:>+7.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_targets_pa.parquet"
    pa_df.to_parquet(out_path, index=False)
    (out_dir / "eval_targets_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {len(pa_df)} PA rows to {out_path}")


if __name__ == "__main__":
    main()
