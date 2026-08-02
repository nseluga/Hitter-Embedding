"""
Verification gates for the Phase D pitch tensors (ml-engineer blocking gates).

Every failure this file guards against is silent. A vocabulary built on all seasons
still trains; bin edges fit on the eval season still produce well-formed bins; a
label mask off by one nesting level still yields a loss that decreases. Nothing
crashes, and the resulting model is wrong in exactly the population claim 1 is about.

Gates: the three train-only quantities each refuse data outside their fit seasons,
unseen batters reach the reserved row and only the reserved row, the §1.2 label
nesting holds in both directions, bins carry roughly equal train mass, and feature
selection removes exactly the columns it names.
"""

import numpy as np
import pandas as pd
import pytest

from src.data import model_dataset as md
from src.features import context_features


SPLIT_CONFIG = {
    "seasons": [2015, 2020, 2023, 2024, 2025],
    "split": {"train": [2015, 2020, 2023], "val": [2024], "test": [2025]},
}


def synthetic_pitches(n=3000, seed=0, seasons=(2015, 2020, 2023, 2024)):
    """
    Labelled pitch rows with the §1.2 nesting built in: contact only on swings,
    quality only on balls in play. Batter ids are deliberately season-dependent so
    the vocabulary gates have someone to exclude.
    """
    rng = np.random.default_rng(seed)
    season = rng.choice(seasons, size=n)
    # batters 0-19 appear everywhere; 900+ only in the latest season, so they are
    # unseen whenever that season is held out of train
    batter = np.where(season == max(seasons),
                      rng.choice(list(range(20)) + [900, 901, 902], size=n),
                      rng.integers(0, 20, size=n))
    swing = rng.integers(0, 2, size=n)
    contact = np.where(swing == 1, rng.integers(0, 2, size=n), np.nan)
    in_play = (contact == 1)

    frame = pd.DataFrame({
        "batter": batter, "season": season, "swing": swing.astype("int8"),
        "contact": pd.array(contact, dtype="Int8"),
        "ev": np.where(in_play, rng.normal(88, 14, n), np.nan),
        "la": np.where(in_play, rng.normal(12, 25, n), np.nan),
        "spray": np.where(in_play, rng.normal(0, 28, n), np.nan),
        "release_speed": rng.normal(93, 3, n), "plate_x": rng.normal(0, 0.8, n),
        "plate_z": rng.normal(2.4, 0.7, n), "pfx_x": rng.normal(0, 0.7, n),
        "pfx_z": rng.normal(1.0, 0.6, n),
        "effective_speed": rng.normal(93, 3, n), "release_spin_rate": rng.normal(2200, 300, n),
        "release_extension": rng.normal(6.5, 0.4, n), "release_pos_x": rng.normal(-1.5, 1.5, n),
        "release_pos_y": rng.normal(54, 1, n), "release_pos_z": rng.normal(5.8, 0.4, n),
        "spin_axis": rng.uniform(0, 360, n),
        "pitch_type": rng.choice(["FF", "SL", "CH"], n), "p_throws": rng.choice(["L", "R"], n),
        "stand": rng.choice(["L", "R"], n), "balls": rng.integers(0, 4, n),
        "strikes": rng.integers(0, 3, n),
    })
    return frame


def fitted_params(frame):
    params = context_features.fit(frame[frame["season"].isin(SPLIT_CONFIG["split"]["train"])])
    params.fit_seasons = SPLIT_CONFIG["split"]["train"]
    return params


def built(frame=None, **kwargs):
    frame = synthetic_pitches() if frame is None else frame
    return md.build(frame, params=fitted_params(frame), config=SPLIT_CONFIG, **kwargs)


# ---- gate: pitchers taking their own at-bats never reach the embedding table ----

def test_pitcher_at_bats_are_dropped_from_rows_and_vocabulary():
    """
    A pitcher-batter must leave no trace: no pitches, no embedding row, and no
    reserved-row rerouting either — he is absent, not generic. Trained rows for a
    population `claim1_eval` drops are parameters that can never be queried.
    """
    frame = synthetic_pitches()
    # make batters 5 and 6 pitcher-batters in every season they appear
    excluded = {(int(season), batter)
                for batter in (5, 6)
                for season in frame.loc[frame["batter"] == batter, "season"].unique()}

    arrays, manifest = built(frame, excluded_batters=excluded)
    kept = md.drop_pitcher_at_bats(frame, excluded)

    assert not kept["batter"].isin([5, 6]).any()
    assert manifest["n_pitches"] == len(kept)
    assert manifest["n_pitcher_at_bat_pitches_dropped"] == len(frame) - len(kept)
    assert manifest["n_hitters"] == 18   # 20 synthetic hitters less the two pitchers
    assert len(arrays["hitter"]) == len(kept)


def test_dropping_pitcher_at_bats_is_a_no_op_when_nothing_is_excluded():
    frame = synthetic_pitches()
    assert len(md.drop_pitcher_at_bats(frame, set())) == len(frame)
    _, manifest = built(frame)
    assert manifest["n_pitcher_at_bat_pitches_dropped"] == 0


def test_exclusion_is_per_season_not_per_batter():
    """
    A two-way player is a pitcher in one season and a hitter in another, so the key
    is (season, batter). Excluding by batter id alone would delete his hitting career.
    """
    frame = synthetic_pitches()
    one_season = {(2015, 5)}
    kept = md.drop_pitcher_at_bats(frame, one_season)
    assert not ((kept["batter"] == 5) & (kept["season"] == 2015)).any()
    assert ((kept["batter"] == 5) & (kept["season"] == 2020)).any()


# ---- gate: the vocabulary is train-only, and unseen batters reach the reserved row ----

def test_vocabulary_excludes_batters_who_appear_only_outside_train():
    frame = synthetic_pitches()
    vocabulary = md.build_vocabulary(frame, SPLIT_CONFIG["split"]["train"])
    assert set(vocabulary) == set(range(20))
    for late_only in (900, 901, 902):
        assert late_only not in vocabulary


def test_unseen_batters_route_to_the_reserved_row_and_nobody_else_does():
    frame = synthetic_pitches()
    arrays, manifest = built(frame)
    reserved = arrays["hitter"] == md.RESERVED_HITTER_INDEX
    assert frame.loc[reserved, "batter"].isin([900, 901, 902]).all()
    assert not frame.loc[~reserved, "batter"].isin([900, 901, 902]).any()
    # the reserved index must not collide with a real hitter's row
    assert arrays["hitter"][~reserved].min() > md.RESERVED_HITTER_INDEX
    assert manifest["n_hitters"] == 20


def test_indices_are_stable_across_rebuilds():
    """A reshuffled table must produce the same batter -> index map, or seeds mean nothing."""
    frame = synthetic_pitches()
    shuffled = frame.sample(frac=1.0, random_state=3).reset_index(drop=True)
    assert (md.build_vocabulary(frame, SPLIT_CONFIG["split"]["train"])
            == md.build_vocabulary(shuffled, SPLIT_CONFIG["split"]["train"]))


# ---- gate: bin edges are train-only and carry equal mass ----

def test_bin_edges_ignore_seasons_outside_train():
    """
    Shifting the held-out season's EV far off-scale must not move a single edge.
    This is the check that would catch edges fit on the whole table.
    """
    frame = synthetic_pitches()
    contaminated = frame.copy()
    outside = ~contaminated["season"].isin(SPLIT_CONFIG["split"]["train"])
    contaminated.loc[outside, "ev"] = contaminated.loc[outside, "ev"] + 60.0

    base = md.fit_bin_edges(frame, SPLIT_CONFIG["split"]["train"])
    after = md.fit_bin_edges(contaminated, SPLIT_CONFIG["split"]["train"])
    assert base["ev"] == after["ev"]


def test_bins_carry_roughly_equal_train_mass():
    """Quantile edges, so each bin should hold about 1/n of labelled train values."""
    frame = synthetic_pitches(n=8000)
    n_bins = 10
    edges = md.fit_bin_edges(frame, SPLIT_CONFIG["split"]["train"], n_bins=n_bins)
    train = frame[frame["season"].isin(SPLIT_CONFIG["split"]["train"])]
    values = train["ev"].to_numpy()
    assigned = md.assign_bins(values[np.isfinite(values)], edges["ev"])
    counts = np.bincount(assigned, minlength=n_bins)
    assert len(counts) == n_bins
    assert counts.min() > 0.5 * counts.mean()


def test_build_refuses_to_fit_anything_on_the_test_season():
    with pytest.raises(AssertionError, match="frozen test season"):
        built(train_seasons=[2015, 2020, 2023, 2025])


# ---- gate: the §1.2 label nesting, in both directions ----

def test_label_masks_match_the_factorization_exactly():
    frame = synthetic_pitches()
    arrays, _ = built(frame)
    swing = frame["swing"].to_numpy()
    in_play = (frame["contact"] == 1).fillna(False).to_numpy(dtype=bool)

    # contact is present on every swing and absent on every take
    assert (arrays["contact"][swing == 1] != md.MASKED).all()
    assert (arrays["contact"][swing == 0] == md.MASKED).all()
    # quality is present on every ball in play and absent everywhere else
    for dimension in md.QUALITY_DIMENSIONS:
        assert (arrays[dimension][in_play] != md.MASKED).all()
        assert (arrays[dimension][~in_play] == md.MASKED).all()


def test_a_broken_nesting_is_caught_rather_than_encoded():
    """Contact set on a take must raise, not quietly become a trainable label."""
    frame = synthetic_pitches()
    take = frame.index[frame["swing"] == 0][0]
    frame.loc[take, "contact"] = 1
    with pytest.raises(AssertionError, match="nesting"):
        built(frame)


def test_masked_label_is_not_a_valid_class_index():
    """MASKED must be distinguishable from bin 0, which is a real class."""
    frame = synthetic_pitches()
    arrays, manifest = built(frame)
    for dimension in md.QUALITY_DIMENSIONS:
        present = arrays[dimension][arrays[dimension] != md.MASKED]
        assert present.min() >= 0 and present.max() < manifest["n_quality_bins"]
    assert md.MASKED < 0


# ---- gate: feature selection removes exactly what it names ----

def test_effective_speed_is_dropped_and_the_block_is_toggleable():
    frame = synthetic_pitches()
    with_block, manifest = built(frame)
    without_block, lean = built(frame, include_block=False)

    assert not any(name.startswith("effective_speed") for name in manifest["context_columns"])
    for field in md.ABLATION_BLOCK:
        assert any(name.startswith(field) for name in manifest["context_columns"])
        assert not any(name.startswith(field) for name in lean["context_columns"])
    assert without_block["context"].shape[1] < with_block["context"].shape[1]
    assert with_block["context"].shape[0] == without_block["context"].shape[0]


def test_selection_keeps_the_vectorizer_column_order():
    """Selection subsets columns; it must never reorder them, or names stop matching."""
    frame = synthetic_pitches()
    names = context_features.feature_names(fitted_params(frame))
    indices, kept = md.select_columns(names)
    assert kept == [names[i] for i in indices]
    assert list(indices) == sorted(indices)


def test_round_trip_through_disk_preserves_every_array(tmp_path):
    frame = synthetic_pitches()
    arrays, manifest = built(frame)
    reloaded, reloaded_manifest = md.load(md.save(arrays, manifest, tmp_path / "d1"))
    assert reloaded_manifest == manifest
    assert set(reloaded) == set(arrays)
    for name, array in arrays.items():
        assert np.array_equal(np.asarray(reloaded[name]), array)
