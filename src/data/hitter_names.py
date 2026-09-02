"""
Build the hitter names table: MLBAM id, embedding row, display name, and
handedness for every row in the Phase D5 vocabulary.

Names come from pybaseball's MLBAM lookup (one batched network call). Handedness
is the majority `stand` value observed for that batter in the training-season
pitch table (2015-2023 only, per the frozen split); 2024-2025 is excluded so this
never leaks eval-season information. Row 0 is the reserved cold-start row and
carries no id or handedness.
"""

import json
import logging
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST_PATH = "data/processed/phase_d5/manifest.json"
DEFAULT_PITCH_TABLE_PATH = "data/processed/pitch_events_labeled.parquet"
STAND_TRAIN_SEASONS = range(2015, 2024)  # 2015..2023 inclusive, training seasons only

COLD_START_INDEX = 0
COLD_START_BATTER = -1
COLD_START_NAME = "COLD_START"


def load_vocabulary(manifest_path):
    """Return {mlbam_id: embedding_index} for all non-reserved rows in the manifest."""
    manifest = json.loads(Path(manifest_path).read_text())
    return {int(mlbam_id): index for mlbam_id, index in manifest["vocabulary"].items()}


def lookup_names(mlbam_ids, lookup_fn):
    """
    Resolve mlbam_ids to "First Last" names via lookup_fn (pybaseball.playerid_reverse_lookup
    signature: ids, key_type='mlbam' -> DataFrame with key_mlbam, name_first, name_last).
    One batched call, one retry on failure. Returns {mlbam_id: name}; unresolved ids are
    omitted, and their count is logged.
    """
    try:
        result = lookup_fn(list(mlbam_ids), key_type="mlbam")
    except Exception:
        logger.warning("player id lookup failed, retrying once")
        result = lookup_fn(list(mlbam_ids), key_type="mlbam")

    names = {}
    for row in result.itertuples():
        first, last = getattr(row, "name_first", None), getattr(row, "name_last", None)
        if first and last:
            names[int(row.key_mlbam)] = f"{first} {last}".title()

    n_unresolved = len(set(mlbam_ids) - set(names))
    if n_unresolved:
        logger.warning("could not resolve names for %d of %d ids", n_unresolved, len(mlbam_ids))
    return names


def majority_stand(pitch_table_path, batter_ids):
    """
    Return {batter_id: 'L' or 'R'}, the most frequent `stand` value per batter over
    STAND_TRAIN_SEASONS. Reads only the batter, stand, season columns.
    """
    df = pd.read_parquet(pitch_table_path, columns=["batter", "stand", "season"])
    df = df[df["season"].isin(STAND_TRAIN_SEASONS) & df["batter"].isin(batter_ids)]
    return df.groupby("batter")["stand"].agg(lambda s: s.value_counts().idxmax()).to_dict()


def build_frame(vocabulary, names, stands):
    """
    Assemble the output frame from a vocabulary map, a name lookup, and a stand lookup.
    Includes the reserved cold-start row at index 0. Sorted by embedding_index.
    """
    rows = [{
        "batter": COLD_START_BATTER,
        "embedding_index": COLD_START_INDEX,
        "name": COLD_START_NAME,
        "stand": "",
    }]
    for mlbam_id, index in vocabulary.items():
        rows.append({
            "batter": mlbam_id,
            "embedding_index": index,
            "name": names.get(mlbam_id, ""),
            "stand": stands.get(mlbam_id, ""),
        })
    return pd.DataFrame(rows).sort_values("embedding_index").reset_index(drop=True)


def main():
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Build the hitter names table from the Phase D5 vocabulary.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--pitch-table", default=DEFAULT_PITCH_TABLE_PATH)
    parser.add_argument("--out", default="data/processed/hitter_names.csv")
    args = parser.parse_args()

    from pybaseball import playerid_reverse_lookup

    vocabulary = load_vocabulary(args.manifest)
    names = lookup_names(vocabulary.keys(), playerid_reverse_lookup)
    stands = majority_stand(args.pitch_table, list(vocabulary.keys()))
    frame = build_frame(vocabulary, names, stands)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    n_unresolved = (frame["embedding_index"] != COLD_START_INDEX) & (frame["name"] == "")
    print(f"rows written: {len(frame)}")
    print(f"unresolved names: {int(n_unresolved.sum())}")


if __name__ == "__main__":
    main()
