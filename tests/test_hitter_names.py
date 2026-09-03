"""
Unit tests for the hitter names table build. build_frame is pure (no network, no
parquet read), so it is tested against synthetic vocabulary/name/stand maps. The
integration check hits the committed CSV and is skipped if it has not been built.
"""

from pathlib import Path

import pandas as pd

from src.data import hitter_names as hn

ANCHOR_NAMES = {
    545361: "Mike Trout",
    665742: "Juan Soto",
    592626: "Joc Pederson",
    664761: "Alec Bohm",
    656941: "Kyle Schwarber",
}


def make_vocabulary(n=1762):
    return {100000 + i: i + 1 for i in range(n)}


def test_build_frame_row_count_and_cold_start():
    vocabulary = make_vocabulary()
    names = {mlbam_id: "Some Player" for mlbam_id in vocabulary}
    stands = {mlbam_id: "L" for mlbam_id in vocabulary}
    frame = hn.build_frame(vocabulary, names, stands)

    assert len(frame) == 1763
    cold_start = frame.iloc[0]
    assert cold_start["embedding_index"] == 0
    assert cold_start["batter"] == -1
    assert cold_start["name"] == "COLD_START"
    assert cold_start["stand"] == ""


def test_build_frame_ids_round_trip_and_sorted():
    vocabulary = make_vocabulary(50)
    names = {mlbam_id: "Some Player" for mlbam_id in vocabulary}
    stands = {mlbam_id: "R" for mlbam_id in vocabulary}
    frame = hn.build_frame(vocabulary, names, stands)

    assert list(frame["embedding_index"]) == sorted(frame["embedding_index"])
    non_cold_start = frame[frame["embedding_index"] != 0]
    for _, row in non_cold_start.iterrows():
        assert vocabulary[row["batter"]] == row["embedding_index"]


def test_build_frame_stand_is_l_or_r():
    vocabulary = make_vocabulary(50)
    names = {mlbam_id: "Some Player" for mlbam_id in vocabulary}
    stands = {mlbam_id: ("L" if i % 2 else "R") for i, mlbam_id in enumerate(vocabulary)}
    frame = hn.build_frame(vocabulary, names, stands)

    non_cold_start = frame[frame["embedding_index"] != 0]
    assert set(non_cold_start["stand"]) <= {"L", "R"}


def test_build_frame_unresolved_name_is_blank():
    vocabulary = make_vocabulary(3)
    frame = hn.build_frame(vocabulary, names={}, stands={})

    non_cold_start = frame[frame["embedding_index"] != 0]
    assert (non_cold_start["name"] == "").all()


def test_lookup_names_retries_once_on_failure():
    calls = {"n": 0}

    def flaky_lookup(ids, key_type):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("simulated network failure")
        return pd.DataFrame({
            "key_mlbam": ids,
            "name_first": ["mike"] * len(ids),
            "name_last": ["trout"] * len(ids),
        })

    names = hn.lookup_names([545361], flaky_lookup)
    assert calls["n"] == 2
    assert names[545361] == "Mike Trout"


def test_anchor_ids_resolve_in_committed_csv():
    csv_path = Path("data/processed/hitter_names.csv")
    if not csv_path.exists():
        return  # integration check only; skip until the CSV is built

    frame = pd.read_csv(csv_path)
    by_batter = frame.set_index("batter")["name"].to_dict()
    for mlbam_id, expected_name in ANCHOR_NAMES.items():
        assert by_batter.get(mlbam_id) == expected_name
