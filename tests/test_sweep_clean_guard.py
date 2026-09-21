"""The clean stage refuses a data dir that was not built --career-pitchers (sweep.launch)."""
import json
from types import SimpleNamespace
from unittest import mock

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


def test_clean_launch_resolves_the_arms_own_data_dir_override(tmp_path):
    # the bug this guards against: CLEAN_BASE's own --data-dir wins over the sweep-level
    # default (same resolution `knobs` documents), so the guard must check THAT path.
    wrong = tmp_path / "wrong"
    right = tmp_path / "right"
    wrong.mkdir()
    right.mkdir()
    (wrong / "manifest.json").write_text(json.dumps({"n_hitters": 3}))  # no exclusion flag
    (right / "manifest.json").write_text(json.dumps({"career_pitchers_excluded": True}))
    args = SimpleNamespace(data_dir=str(wrong), device="cpu", train_args=[])
    with pytest.raises(AssertionError, match="career-pitchers"):
        sweep.launch("clean", "clean_base", [], 0, args)  # no override -> checks the wrong dir
    # with the arm's own override present, the guard follows it and lets the run proceed
    # (we stub subprocess.run -- this test is only about which manifest the guard reads)
    with mock.patch("src.model.sweep.subprocess.run") as run:
        run.return_value = SimpleNamespace(returncode=0)
        sweep.launch("clean", "clean_base", ["--data-dir", str(right)], 0, args)
    run.assert_called_once()
