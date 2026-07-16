"""
Clean the frozen Statcast snapshot into a modeling pitch table.

This produces the table the conditional model TRAINS on. It is deliberately
filtered (regular season, competitive pitches only, valid tracking) in ways
that would bias the claim-1 evaluation targets if applied to them. Evaluation
targets must be aggregated from a complete regular-season outcome source, NOT
from this table. See `README` / decision log for the two-table principle.

Every filter is evidence-backed by notebooks/01_statcast_profiling.ipynb:
- regular season only (game_type == "R")
- position-player pitchers removed (primarily-a-batter AND faced few batters);
  velocity is deliberately NOT used, it misclassifies hard-armed position players
- non-competitive pitches removed (pitchouts, automatic balls/strikes, bunts);
  intentional balls are KEPT, they are real takes and real walks
- rows missing or out-of-bounds on any CORE context field are dropped; optional
  context (spin) missingness is kept and flagged, never imputed
- deprecated/empty columns dropped; exact duplicate pitches deduped
"""

import json
from pathlib import Path

import pandas as pd

# a player with at least this many batting PAs in a season is primarily a batter
BATTER_PA_THRESHOLD = 50
# a primarily-a-batter who faced fewer than this many batters was mopping up, not pitching
POSITION_PLAYER_MAX_BATTERS_FACED = 50

# pitch identity: same three values means the same physical pitch
KEY_FIELDS = ["game_pk", "at_bat_number", "pitch_number"]

# context the conditional query cannot be posed without; a null here is unusable
CORE_CONTEXT_FIELDS = ["pitch_type", "release_speed", "plate_x", "plate_z", "pfx_x", "pfx_z"]
# secondary/structurally-absent context; kept with a missingness indicator, never imputed
OPTIONAL_CONTEXT_FIELDS = ["release_spin_rate", "spin_axis", "effective_speed",
                           "release_extension", "release_pos_x", "release_pos_y", "release_pos_z"]

# empty in every season of the snapshot (verified) plus the superseded legacy age fields
DEPRECATED_COLUMNS = ["spin_dir", "spin_rate_deprecated", "break_angle_deprecated",
                      "break_length_deprecated", "tfs_deprecated", "tfs_zulu_deprecated",
                      "age_bat_legacy", "age_pit_legacy"]

# non-competitive events removed from the modeling table (intentional balls are NOT here)
NONCOMPETITIVE_DESCRIPTIONS = {"pitchout", "foul_pitchout", "swinging_pitchout",
                               "automatic_ball", "automatic_strike",
                               "foul_bunt", "missed_bunt", "bunt_foul_tip"}

# physically impossible readings beyond these are tracking errors, not real pitches
RELEASE_SPEED_BOUNDS = (30.0, 110.0)
PLATE_X_ABS_MAX = 6.0
PLATE_Z_BOUNDS = (-4.0, 10.0)

# columns retained in the processed modeling table
RETAIN_COLUMNS = sorted(set(
    KEY_FIELDS + CORE_CONTEXT_FIELDS + OPTIONAL_CONTEXT_FIELDS + [
        "game_type", "game_date", "game_year", "season", "batter", "pitcher",
        "description", "des", "events", "balls", "strikes", "p_throws", "stand", "zone",
        "launch_speed", "launch_angle", "hc_x", "hc_y", "hit_distance_sc",
        "bat_speed", "swing_length", "attack_angle", "attack_direction", "swing_path_tilt",
    ]
))


def load_snapshot(snapshot_dir, seasons=None):
    """
    Read the per-season parquet files of a snapshot into one DataFrame.
    snapshot_dir: Path to a snapshot_<date> directory.
    seasons: optional list of ints to restrict which seasons are loaded.
    Returns the concatenated raw frame with a `season` column added.
    """
    import pyarrow.parquet as pq

    snapshot_dir = Path(snapshot_dir)
    wanted = {c for c in RETAIN_COLUMNS if c != "season"}
    frames = []
    for path in sorted(snapshot_dir.glob("season=*.parquet")):
        season = int(path.stem.split("=")[1])
        if seasons is not None and season not in seasons:
            continue
        available = set(pq.ParquetFile(path).schema.names)
        frame = pd.read_parquet(path, columns=sorted(wanted & available))
        frame["season"] = season
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def filter_regular_season(df):
    return df[df["game_type"] == "R"]


def drop_deprecated_columns(df):
    return df.drop(columns=[c for c in DEPRECATED_COLUMNS if c in df.columns])


def deduplicate(df):
    return df.drop_duplicates(subset=KEY_FIELDS)


def filter_position_player_pitchers(df):
    """
    Remove pitches thrown by position players who were mopping up.
    A pitch is dropped when its pitcher, that season, both batted at least
    BATTER_PA_THRESHOLD times and faced fewer than POSITION_PLAYER_MAX_BATTERS_FACED
    batters. Real pitchers face hundreds; two-way players face hundreds; only
    position players clear both conditions.
    """
    keep_masks = []
    for _, frame in df.groupby("season"):
        batting = frame[["batter", "game_pk", "at_bat_number"]].drop_duplicates().groupby("batter").size()
        faced = frame[["pitcher", "game_pk", "at_bat_number"]].drop_duplicates().groupby("pitcher").size()
        primarily_batter = set(batting[batting >= BATTER_PA_THRESHOLD].index)
        few_faced = set(faced[faced < POSITION_PLAYER_MAX_BATTERS_FACED].index)
        position_players = primarily_batter & few_faced
        keep_masks.append(~frame["pitcher"].isin(position_players))
    keep = pd.concat(keep_masks).reindex(df.index)
    return df[keep]


def filter_noncompetitive(df):
    """Remove pitchouts, automatic balls/strikes, and bunts; keep intentional balls."""
    labeled = df["description"].isin(NONCOMPETITIVE_DESCRIPTIONS)
    # in-play bunts are not flagged in `description`; catch them from the play text
    in_play_bunt = df["des"].fillna("").str.contains("bunt", case=False)
    return df[~(labeled | in_play_bunt)]


def validate_core_context(df):
    """Drop rows missing a core context field or holding a physically impossible reading."""
    valid = df[CORE_CONTEXT_FIELDS].notna().all(axis=1)
    valid &= df["release_speed"].between(*RELEASE_SPEED_BOUNDS)
    valid &= df["plate_x"].abs() <= PLATE_X_ABS_MAX
    valid &= df["plate_z"].between(*PLATE_Z_BOUNDS)
    return df[valid]


def add_missingness_indicators(df):
    """Add a boolean `<field>_missing` column for each optional context field, keeping the field itself."""
    df = df.copy()
    for field in OPTIONAL_CONTEXT_FIELDS:
        if field in df.columns:
            df[f"{field}_missing"] = df[field].isna()
    return df


def sort_pitches(df):
    """Order deterministically by time then pitch key, as the walk-forward split and history encoder require."""
    return df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)


def clean(snapshot_dir, seasons=None):
    """
    Run the full cleaning pipeline and return (modeling_table, report).
    report is a list of (stage, rows_remaining) recording attrition at each step.
    """
    df = load_snapshot(snapshot_dir, seasons)
    df = df[[c for c in RETAIN_COLUMNS if c in df.columns]]

    report = [("loaded", len(df))]
    for name, step in [
        ("regular_season", filter_regular_season),
        ("drop_deprecated", drop_deprecated_columns),
        ("deduplicate", deduplicate),
        ("position_player_pitchers", filter_position_player_pitchers),
        ("noncompetitive", filter_noncompetitive),
        ("core_context", validate_core_context),
    ]:
        df = step(df)
        report.append((name, len(df)))

    df = add_missingness_indicators(df)
    df = sort_pitches(df)
    report.append(("final", len(df)))
    return df, report


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Clean a Statcast snapshot into a modeling pitch table.")
    parser.add_argument("--snapshot-dir", default="data/raw/statcast/snapshot_2026-07-14")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--seasons", type=int, nargs="+", default=None)
    args = parser.parse_args()

    modeling_table, report = clean(args.snapshot_dir, args.seasons)

    for stage, rows in report:
        print(f"{stage:26s} {rows:>9}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pitch_events.parquet"
    modeling_table.to_parquet(out_path, index=False)
    (out_dir / "clean_report.json").write_text(json.dumps(dict(report), indent=2))
    print(f"wrote {len(modeling_table)} rows to {out_path}")


if __name__ == "__main__":
    main()
