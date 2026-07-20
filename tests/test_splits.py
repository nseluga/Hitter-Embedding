"""Unit tests for the frozen walk-forward split config and its guards."""

import pandas as pd
import pytest

from src.config import splits


def valid_config():
    """A well-formed contiguous walk-forward config."""
    return {
        "seasons": [2015, 2016, 2017, 2018],
        "split": {"train": [2015, 2016], "val": [2017], "test": [2018]},
    }


def test_frozen_config_on_disk_is_valid():
    # the real committed config loads and passes every shape check
    config = splits.load_splits()
    assert config["frozen"] is True
    assert config["split"]["train"][-1] + 1 == config["split"]["val"][0]
    assert config["split"]["val"][0] + 1 == config["split"]["test"][0]


def test_valid_config_passes():
    splits.validate_split_config(valid_config())


def test_overlapping_split_rejected():
    config = valid_config()
    config["split"]["val"] = [2016]  # 2016 already in train
    with pytest.raises(AssertionError, match="more than one split"):
        splits.validate_split_config(config)


def test_incomplete_partition_rejected():
    config = valid_config()
    config["seasons"].append(2019)  # 2019 configured but assigned to no split
    with pytest.raises(AssertionError, match="partition"):
        splits.validate_split_config(config)


def test_non_contiguous_val_rejected():
    # this is exactly the val=t / test=t+2 gap the README typo would have produced
    config = {
        "seasons": [2015, 2016, 2017, 2018],
        "split": {"train": [2015, 2016], "val": [2018], "test": [2017]},
    }
    with pytest.raises(AssertionError):
        splits.validate_split_config(config)


def test_season_split_map_assigns_every_season():
    mapping = splits.season_split_map(valid_config())
    assert mapping == {2015: "train", 2016: "train", 2017: "val", 2018: "test"}


def test_validate_against_data_catches_unassigned_season():
    df = pd.DataFrame({"season": [2015, 2016, 2017, 2018, 2099]})  # 2099 not in config
    with pytest.raises(AssertionError, match="absent from the split config"):
        splits.validate_against_data(df, valid_config())


def test_validate_against_data_catches_missing_season():
    df = pd.DataFrame({"season": [2015, 2016]})  # 2017, 2018 configured but absent
    with pytest.raises(AssertionError, match="missing from data"):
        splits.validate_against_data(df, valid_config())
