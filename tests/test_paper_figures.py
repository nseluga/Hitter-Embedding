"""Unit tests for src/analysis/paper_figures.py."""

import pandas as pd
import pytest

from src.analysis import paper_figures as pf

N_STAR = 226.03631271852322
CEILING_RANK_CORR = 0.7420469557106563


def test_reliability_is_monotone_increasing_in_n():
    plate_appearances = [1, 10, 50, 226, 500, 1200]
    reliabilities = [pf.reliability(n, N_STAR) for n in plate_appearances]
    assert reliabilities == sorted(reliabilities)


def test_reliability_equals_one_half_at_n_star():
    assert pf.reliability(N_STAR, N_STAR) == pytest.approx(0.5)


def test_ceiling_is_sqrt_of_the_same_reliability_curve():
    """One n* governs both curves; the ceiling is never independently calibrated."""
    for n in [10, 50, 226, 500, 1200]:
        assert pf.rank_correlation_ceiling(n, N_STAR) == pytest.approx(
            pf.reliability(n, N_STAR) ** 0.5)


def test_evaluated_population_lands_on_the_ceiling_curve():
    """
    The reported ceiling 0.7420 must be reproduced by the plotted curve at the
    exposure the evaluated population actually sits at, not at n*.
    """
    achieved_pa = pf.exposure_at_reliability(CEILING_RANK_CORR**2, N_STAR)
    assert achieved_pa == pytest.approx(277.0, abs=0.5)
    assert pf.rank_correlation_ceiling(achieved_pa, N_STAR) == pytest.approx(
        CEILING_RANK_CORR, abs=1e-6)
    assert pf.rank_correlation_ceiling(N_STAR, N_STAR) == pytest.approx(0.7071, abs=1e-4)


def test_min_pa_sweep_has_both_floors_and_expected_strata_for_gbm_full():
    root = pf.repo_root()
    sweep = pf.load_min_pa_sweep(root, "gbm_full", [10, 50])
    assert set(sweep["min_eval_pa"].unique()) == {10, 50}
    assert set(sweep["stratum"].unique()) >= {"low", "medium", "high"}


def test_calibration_parser_returns_four_strata_matching_file():
    root = pf.repo_root()
    calibration = pf.load_calibration(root)
    file_calibration = pd.read_csv(root / pf.CALIBRATION_CSV)
    assert set(calibration["stratum"]) == {"low", "medium", "high", "all"}
    for stratum_name in ["low", "medium", "high", "all"]:
        parsed_slope = calibration.loc[calibration["stratum"] == stratum_name, "slope"].iloc[0]
        file_slope = file_calibration.loc[file_calibration["stratum"] == stratum_name, "slope"].iloc[0]
        assert parsed_slope == pytest.approx(file_slope)


def test_smoke_renders_every_figure(tmp_path):
    root = pf.repo_root()
    for name in pf.FIGURE_NAMES:
        written_path = pf.render_figure(name, root, tmp_path, smoke=True)
        if written_path is None:
            # figure 6 is allowed to skip if per-anchor L/R columns are absent
            continue
        assert written_path.exists()
        assert written_path.stat().st_size > 0


def test_gate_encoding_matches_the_files_own_verdict_columns():
    """
    The figure fills a marker when the CI excludes zero in the model's favour.
    That rule must reproduce the sweep file's own favours_model_v1 verdicts, or
    the figure and the registered result disagree.
    """
    root = pf.repo_root()
    sweep = pf.load_min_pa_sweep(root, "gbm_full", [10, 50])
    sweep = sweep[sweep["stratum"].isin(["low", "medium", "high"])]
    assert len(sweep) == 6
    assert ((sweep["ci_low_rank"] > 0) == sweep["rank_favours_model_v1"]).all()
    assert ((sweep["ci_high_rmse"] < 0) == sweep["rmse_favours_model_v1"]).all()
