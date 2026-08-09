"""
Tensors and batching for the Phase D training loop (docs/phase-d-spec.md §5).

Three failures here are silent, and each one produces a number rather than an error.

1. MEMORY LAYOUT. `load` reads the arrays into RAM rather than memmapping them. The
   built table is ~1.7 GB against 8.6 GB, and a memmapped gather of 8,192 rows
   measured 8.74 s against 0.033 s resident -- an epoch of pure I/O either way,
   the difference being whether it is minutes or hours. `torch.from_numpy` then
   shares that memory instead of copying it, so nothing is held twice.

2. SPLIT SLICING. Splits are index arrays into the full table, never sliced copies:
   slicing the train seasons would duplicate 1.08 GB of context beside the original.
   Season membership comes from `src.config.splits`, which is the one definition of
   the frozen walk-forward; a second one here could drift from it without failing.

3. THE TEST SPLIT. `split_indices` returns test indices because D.6's final run
   needs them. Nothing in this module refuses to hand them over -- the guard is
   `assert_not_test_season` at the CLI boundary, mirroring `c_report.py`. A caller
   that skips that guard scores against 2025 and gets a perfectly plausible number.
"""

import numpy as np
import torch

from src.config.splits import load_splits, season_split_map
from src.data.model_dataset import load as load_arrays
from src.model.v1 import FACTORS, LEGACY_FACTORS

REQUIRED_NAMES = ("context", "hitter", "season") + LEGACY_FACTORS
# `split` postdates the datasets the first 39 runs were trained on, so it is loaded when
# present and absent otherwise rather than making every old artifact unreadable
OPTIONAL_NAMES = tuple(name for name in FACTORS if name not in LEGACY_FACTORS)
TENSOR_NAMES = REQUIRED_NAMES + OPTIONAL_NAMES


def load_tensors(data_dir):
    """
    Read the built dataset into resident torch tensors sharing numpy's memory.
    Returns (tensors, manifest). Contiguity is asserted rather than fixed: the
    fix belongs in `model_dataset.save`, which the 2015-2024 refit also runs.
    """
    arrays, manifest = load_arrays(data_dir, mmap_mode=None)
    missing = sorted(set(REQUIRED_NAMES) - set(arrays))
    assert not missing, f"built dataset is missing {missing}"

    tensors = {}
    for name in (n for n in TENSOR_NAMES if n in arrays):
        array = arrays[name]
        assert array.flags["C_CONTIGUOUS"], \
            f"{name}.npy is not row-major; rebuild through model_dataset.save"
        assert len(array) == manifest["n_pitches"], f"{name} disagrees with the manifest"
        tensors[name] = torch.from_numpy(array)

    frozen_test = set(load_splits()["split"]["test"])
    assert not set(manifest["train_seasons"]) & frozen_test, \
        "the built dataset was fit on the frozen test season"
    return tensors, manifest


def split_indices(season, config=None):
    """
    Row indices per split, from the frozen walk-forward season map.
    season: the int64 season tensor. Returns {split_name: int64 index tensor}.
    Asserts the splits partition the table -- every row in exactly one.
    """
    config = config or load_splits()
    seasons = season.numpy()
    mapping = season_split_map(config)
    unknown = sorted(set(np.unique(seasons).tolist()) - set(mapping))
    assert not unknown, f"seasons absent from the frozen split config: {unknown}"

    indices, assigned = {}, np.zeros(len(seasons), dtype=np.int8)
    for name, split_seasons in config["split"].items():
        mask = np.isin(seasons, split_seasons)
        assigned += mask
        indices[name] = torch.from_numpy(np.flatnonzero(mask).astype(np.int64))
    # one counter, both failures: a row in two splits lands on 2, a row in none on 0
    assert bool((assigned == 1).all()), "the splits do not partition the table"
    return indices


def batches(indices, batch_size, generator=None, shuffle=True):
    """
    Yield index tensors of at most batch_size rows, drawn from `indices`.
    generator: a seeded `torch.Generator`, so an epoch's order is reproducible.
    No DataLoader, so there are no worker seeds to get wrong.
    """
    order = torch.randperm(len(indices), generator=generator) if shuffle \
        else torch.arange(len(indices))
    for start in range(0, len(indices), batch_size):
        yield indices[order[start:start + batch_size]]


def gather(tensors, index, device=None):
    """
    Materialise one batch. Returns (hitter, context, labels).
    labels carries all five factors with MASKED intact; the model's teacher forcing
    reads the observed EV and LA bins straight out of it.
    """
    hitter = tensors["hitter"][index]
    context = tensors["context"][index]
    labels = {name: tensors[name][index] for name in FACTORS if name in tensors}
    if device is not None:
        hitter, context = hitter.to(device), context.to(device)
        labels = {name: value.to(device) for name, value in labels.items()}
    return hitter, context, labels
