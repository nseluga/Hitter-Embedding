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


# Mirror of clean.py's position-player-pitcher rule, in the batting direction. Real
# pitchers face hundreds of batters a season; a pitcher who ALSO takes a full slate of
# PA is a two-way player (Ohtani) and is a genuine hitter, so both conditions are
# required. Thresholds match clean.py's for consistency.
PITCHER_MIN_BATTERS_FACED = 50
TWO_WAY_MIN_PA = 50


def primarily_pitchers(pa_df):
    """
    Batter ids that are pitchers taking their own turn at bat, per season.

    The eval-target table is built from the COMPLETE source on purpose, so it contains
    every PA including NL pitchers batting before the 2022 universal DH — roughly half
    the distinct batters in 2015-2021. That is correct for the target itself (ground
    truth must not be filtered), but any quantity describing HITTER talent — a prior,
    a stabilization point, a league average a projection shrinks toward — must exclude
    them, or a population of ~.15-wOBA non-hitters contaminates it.

    Returns a set of (season, batter) pairs.
    """
    out = set()
    for season, frame in pa_df.groupby("season"):
        pa_key = frame[["batter", "game_pk", "at_bat_number"]].drop_duplicates()
        batted = pa_key.groupby("batter").size()
        faced = frame[["pitcher", "game_pk", "at_bat_number"]].drop_duplicates().groupby("pitcher").size()

        real_pitchers = set(faced[faced >= PITCHER_MIN_BATTERS_FACED].index)
        two_way = set(batted[batted >= TWO_WAY_MIN_PA].index)
        # only ids that actually took a PA are meaningful here — a pitcher who never
        # batted needs no exclusion and would just clutter the set
        out.update((season, batter)
                   for batter in (real_pitchers & set(batted.index)) - two_way)
    return out


def drop_pitcher_batters(pa_df):
    """
    Remove pitchers' own plate appearances. Use for every hitter-talent quantity;
    never for the evaluation target itself.
    """
    excluded = primarily_pitchers(pa_df)
    if not excluded:
        return pa_df
    keys = pd.MultiIndex.from_arrays([pa_df["season"], pa_df["batter"]])
    mask = ~keys.isin(excluded)
    return pa_df[mask]


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
