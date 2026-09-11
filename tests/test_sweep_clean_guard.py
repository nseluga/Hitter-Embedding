"""The clean stage refuses a data dir that was not built --career-pitchers (sweep.launch)."""
import json
from types import SimpleNamespace

import pytest

from src.model import sweep


def _args(tmp_path):
    return SimpleNamespace(data_dir=str(tmp_path), device="cpu", train_args=[])


def test_clean_launch_refuses_a_build_without_the_career_pitcher_exclusion(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"n_hitters": 3}))
    with pytest.raises(AssertionError, match="career-pitchers"):
        sweep.launch("clean", "clean_base", [], 0, _args(tmp_path))


def test_clean_launch_refuses_a_missing_manifest(tmp_path):
    with pytest.raises(AssertionError, match="career-pitchers"):
        sweep.launch("clean", "clean_base", [], 0, _args(tmp_path))
