"""
Pull pitch-level Statcast data and freeze it to a versioned parquet snapshot.

Statcast revises past data retroactively, so every pull is stamped with the
date it was run: re-pulling never overwrites an earlier snapshot, it creates a
new one. The snapshot directory and its manifest.json are what ship with the
public repo for reproducibility.

Layout produced:
    data/raw/statcast/snapshot_<YYYY-MM-DD>/season=<YYYY>.parquet   (all columns)
    data/raw/statcast/snapshot_<YYYY-MM-DD>/manifest.json           (committed)

Usage:
    python -m src.data.pull_statcast                      # all seasons, today's date
    python -m src.data.pull_statcast --seasons 2023 2024  # subset
    python -m src.data.pull_statcast --snapshot-date 2026-07-14
"""

import argparse
import json
import time
from datetime import date
from pathlib import Path

import pandas as pd
from pybaseball import statcast
from pybaseball import __version__ as pybaseball_version

# Full era targeted by the architecture plan (§5.10).
DEFAULT_SEASONS = list(range(2015, 2026))

# Regular season + postseason window. Spring training (game_type "S") is largely
# excluded by starting mid-March; the explicit game_type filter is applied later,
# in the Phase A processing step, not here — raw stays as complete as pulled.
SEASON_START = "03-15"
SEASON_END = "11-15"

# Retry a failed season pull this many times before giving up (network hiccups
# occasionally drop days mid-pull).
MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 30


def season_parquet_path(snapshot_dir: Path, season: int) -> Path:
    """Path to one season's parquet file inside a snapshot directory."""
    return snapshot_dir / f"season={season}.parquet"


def pull_one_season(season: int) -> pd.DataFrame:
    """
    Pull every pitch in one season's regular-season/postseason window.
    Retries the whole-season request on failure. Returns the raw DataFrame
    with all Statcast columns retained (no pruning — dropped columns can't be
    recovered without a re-pull).
    """
    start = f"{season}-{SEASON_START}"
    end = f"{season}-{SEASON_END}"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return statcast(start_dt=start, end_dt=end)
        except Exception as error:
            if attempt == MAX_RETRIES:
                raise
            print(f"season {season}: attempt {attempt} failed ({error}); retrying in {RETRY_WAIT_SECONDS}s")
            time.sleep(RETRY_WAIT_SECONDS)


def season_summary(season: int, df: pd.DataFrame) -> dict:
    """One season's entry for the manifest: counts and coverage for a reproducibility check."""
    return {
        "season": season,
        "rows": int(len(df)),
        "games": int(df["game_pk"].nunique()),
        "days_with_data": int(df["game_date"].nunique()),
        "start_date": str(df["game_date"].min()),
        "end_date": str(df["game_date"].max()),
    }


def write_manifest(snapshot_dir: Path, snapshot_date: str, seasons_meta: list[dict], columns: list[str]) -> None:
    """Write manifest.json — the committed, checkable record of what this snapshot contains."""
    manifest = {
        "snapshot_date": snapshot_date,
        "pybaseball_version": pybaseball_version,
        "pull_window": {"start": SEASON_START, "end": SEASON_END},
        "seasons": seasons_meta,
        "n_columns": len(columns),
        "columns": sorted(columns),
    }
    manifest_path = snapshot_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote manifest: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull Statcast into a versioned parquet snapshot.")
    parser.add_argument("--snapshot-date", default=date.today().isoformat(), help="Snapshot stamp (default: today).")
    parser.add_argument("--seasons", type=int, nargs="+", default=DEFAULT_SEASONS, help="Seasons to pull.")
    parser.add_argument("--out-dir", default="data/raw/statcast", help="Root output directory.")
    args = parser.parse_args()

    snapshot_dir = Path(args.out_dir) / f"snapshot_{args.snapshot_date}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    seasons_meta: list[dict] = []
    columns: list[str] = []

    for season in args.seasons:
        out_path = season_parquet_path(snapshot_dir, season)

        # Resume support: a season already on disk is left untouched so an
        # interrupted multi-hour pull can be restarted without re-fetching.
        if out_path.exists():
            print(f"season {season}: already present, reloading summary from {out_path}")
            df = pd.read_parquet(out_path)
        else:
            print(f"season {season}: pulling {season}-{SEASON_START} to {season}-{SEASON_END}")
            df = pull_one_season(season)
            df.to_parquet(out_path, index=False)
            print(f"season {season}: wrote {len(df)} rows to {out_path}")

        seasons_meta.append(season_summary(season, df))
        columns = list(df.columns)

    write_manifest(snapshot_dir, args.snapshot_date, seasons_meta, columns)
    total_rows = sum(meta["rows"] for meta in seasons_meta)
    print(f"snapshot complete: {len(seasons_meta)} seasons, {total_rows} total rows")


if __name__ == "__main__":
    main()
