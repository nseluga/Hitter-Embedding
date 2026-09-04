"""
Load and enforce the frozen walk-forward split.

The split is defined once in `split_config.json` and never revisited (architecture
doc §2.2): strict walk-forward by season, train on seasons <= t, validate on t+1,
test on t+2. Random splits are forbidden. Validation runs on load so a config that
violates the walk-forward shape fails loud rather than silently mis-splitting a run.
"""

import json
from pathlib import Path

DEFAULT_SPLIT_CONFIG = Path(__file__).parent / "split_config.json"


def load_splits(path=DEFAULT_SPLIT_CONFIG):
    """Read the frozen split config, validate its shape, and return it as a dict."""
    config = json.loads(Path(path).read_text())
    validate_split_config(config)
    return config


def validate_split_config(config):
    """
    Assert the split is a well-formed contiguous walk-forward:
    the three sets partition `seasons` with no overlap, train is a contiguous
    block, and val/test are exactly the next two seasons after train.

    A config with `final_run: true` is the one exception (2026-09-04 decision log): the
    final refit absorbs the validation season into train and keeps the SAME held-out test
    season, so `val` is empty and test is one season after train rather than two. It buys
    that by giving up early stopping, which is why such a config is only usable with a fixed
    step budget (`train.py --step-budget`). It cannot move the test season: a config that
    tests on anything but the frozen test season is rejected here, not downstream.
    """
    train, val, test = config["split"]["train"], config["split"]["val"], config["split"]["test"]
    assigned = train + val + test

    assert len(assigned) == len(set(assigned)), "a season appears in more than one split"
    assert set(assigned) == set(config["seasons"]), "split does not partition the configured seasons"
    assert train == list(range(min(train), max(train) + 1)), "train seasons must be contiguous"
    if config.get("final_run"):
        assert val == [], f"a final_run split has no validation season, got {val}"
        assert test == [max(train) + 1], f"test must be the season after train, got {test}"
        assert not config.get("frozen"), "the frozen split is not a final_run split"
        frozen_test = json.loads(DEFAULT_SPLIT_CONFIG.read_text())["split"]["test"]
        assert test == frozen_test, (
            f"a final_run split must keep the frozen test season {frozen_test}, got {test}")
    else:
        assert val == [max(train) + 1], f"val must be the single season after train, got {val}"
        assert test == [max(train) + 2], f"test must be two seasons after train, got {test}"


def validate_against_data(df, config=None):
    """Assert the labeled table's seasons match the config exactly: none missing, none unassigned."""
    config = config or load_splits()
    present = set(int(s) for s in df["season"].unique())
    configured = set(config["seasons"])
    assert configured <= present, f"configured seasons missing from data: {sorted(configured - present)}"
    assert present <= configured, f"data has seasons absent from the split config: {sorted(present - configured)}"


def season_split_map(config=None):
    """Return a {season: split_name} mapping for assigning rows to train/val/test."""
    config = config or load_splits()
    return {season: name for name, seasons in config["split"].items() for season in seasons}
