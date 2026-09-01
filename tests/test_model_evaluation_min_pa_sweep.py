"""
E.11b's sweep bookkeeping (src/analysis/model_evaluation_min_pa_sweep.py).

`sweep` itself computes nothing statistical -- it rebuilds the eval frame at each threshold
and then collapses `model_v1_ablation_report.compare`'s rows into a per-threshold verdict. That collapse is
where the silent failures live: reading the rank margin off the wrong OPPONENT's row (both
are present and both look like margins), taking the ordering gate from one opponent instead
of both, or declaring the verdict stable because the set it de-duplicates lost the flip.

So the paired comparison is stubbed with a scripted table whose every value is chosen by
hand, and the eval frames underneath it are real -- the censoring is the one thing the sweep
is actually varying.
"""

import pandas as pd
import pytest

from src.analysis import claim1_eval as evaluation, model_v1_ablation_report, model_evaluation_min_pa_sweep

LABEL = "min_pa_sweep_rebuild_baseline"
OPPONENTS = ("gbm_full", "eb_bivariate")


def _pa_table(rows):
    """PA-level eval targets from (batter, season, hand, n_pa, woba) tuples."""
    records = []
    pa_id = 0
    for batter, season, hand, n_pa, woba in rows:
        for _ in range(n_pa):
            pa_id += 1
            records.append({"batter": batter, "season": season, "p_throws": hand,
                            "woba_points": woba, "in_denominator": True,
                            "pitcher": 900001, "game_pk": pa_id, "at_bat_number": 1})
    return pd.DataFrame(records)


@pytest.fixture
def case():
    """
    Four hitters whose 2024 exposure straddles the swept thresholds (10, 25, 50):
    100, 60, 30 and 12 PA, so the scored group count is 4, then 3, then 2.
    """
    pa_df = _pa_table([
        (1, 2023, "L", 400, 0.320), (1, 2024, "L", 100, 0.310),
        (2, 2023, "L", 300, 0.300), (2, 2024, "L", 60, 0.290),
        (3, 2023, "L", 200, 0.340), (3, 2024, "L", 30, 0.350),
        (4, 2023, "L", 100, 0.280), (4, 2024, "L", 12, 0.270),
    ])
    predictions = {
        name: pd.DataFrame({"batter": [1, 2, 3, 4], "season": 2024, "p_throws": "L",
                            "pred_woba": [0.310, 0.300, 0.330, 0.290]})
        for name in (LABEL,) + OPPONENTS
    }
    return pa_df, predictions


def _stub_compare(monkeypatch, script):
    """
    Replace the paired bootstrap with a table the test wrote. `script(n_hitters, opponent)`
    returns (rmse_favours, rank_favours, rank_difference) for the low stratum.
    """
    def fake_compare(frames, name=LABEL, seed=0):
        n_hitters = len(frames[name])
        rows = []
        for opponent in OPPONENTS:
            rmse_ok, rank_ok, margin = script(n_hitters, opponent)
            for stratum in ("low", "high"):
                rows.append({"opponent": opponent, "stratum": stratum,
                             "n_hitters": n_hitters, "n_batters": n_hitters,
                             "rmse_favours_model_v1": rmse_ok, "rank_favours_model_v1": rank_ok,
                             "rank_difference": margin, "ci_low_rank": margin - 0.01,
                             "ci_high_rank": margin + 0.01})
        return pd.DataFrame(rows)

    monkeypatch.setattr(model_v1_ablation_report, "compare", fake_compare)


def test_sweep_rebuilds_the_frame_at_every_threshold(case, monkeypatch):
    """
    The thresholds are the committed (10, 25, 50), and the group counts the fixture implies
    are 4, 3 and 2 -- predictions never change, only the censoring does.
    """
    _stub_compare(monkeypatch, lambda n, opponent: (True, True, 0.05))
    pa_df, predictions = case
    table, verdict = model_evaluation_min_pa_sweep.sweep(pa_df, predictions, 2024, LABEL)

    assert verdict["thresholds"] == list(evaluation.MIN_EVAL_PA_SENSITIVITY) == [10, 25, 50]
    assert table.columns[0] == "min_eval_pa"
    assert table["min_eval_pa"].tolist() == [10] * 4 + [25] * 4 + [50] * 4   # 2 opponents x 2 strata
    assert [verdict["by_threshold"][str(t)]["low"]["n_hitters"] for t in (10, 25, 50)] == [4, 3, 2]
    assert verdict["stability"]["low_stratum_n_hitters"] == [4, 3, 2]


def test_the_rank_margin_is_read_off_the_gbm_full_row(case, monkeypatch):
    """Both opponents carry a rank_difference; only C.3-full's is the reported margin."""
    margins = {"gbm_full": 0.07, "eb_bivariate": -0.42}
    _stub_compare(monkeypatch, lambda n, opponent: (True, True, margins[opponent]))
    pa_df, predictions = case
    _, verdict = model_evaluation_min_pa_sweep.sweep(pa_df, predictions, 2024, LABEL)

    low = verdict["by_threshold"]["10"]["low"]
    assert low["rank_difference_vs_gbm_full"] == pytest.approx(0.07)
    assert low["rank_ci_low_vs_gbm_full"] == pytest.approx(0.06)     # margin - 0.01
    assert low["rank_ci_high_vs_gbm_full"] == pytest.approx(0.08)    # margin + 0.01
    assert verdict["stability"]["low_stratum_rank_difference"] == [0.07, 0.07, 0.07]


def test_the_ordering_gate_needs_both_opponents_and_the_rmse_gate_needs_only_gbm(case, monkeypatch):
    """C.3-full passes both; C.2 fails the ordering gate. RMSE is a C.3-full-only gate, so it
    still passes, while the ordering gate must fail on the AND across the two opponents."""
    _stub_compare(monkeypatch,
                  lambda n, opponent: (True, opponent == "gbm_full", 0.05))
    pa_df, predictions = case
    _, verdict = model_evaluation_min_pa_sweep.sweep(pa_df, predictions, 2024, LABEL)

    low = verdict["by_threshold"]["25"]["low"]
    assert low["rmse_gate_vs_gbm_full"] is True
    assert low["ordering_gate_vs_both"] is False


def test_a_verdict_that_flips_with_the_threshold_is_reported_unstable(case, monkeypatch):
    """
    The gate holds at 4 and 3 scored groups and fails at 2. Both gates are stable when the
    verdict never moves and both must go false the moment one threshold disagrees.
    """
    pa_df, predictions = case
    _stub_compare(monkeypatch, lambda n, opponent: (True, True, 0.05))
    _, steady = model_evaluation_min_pa_sweep.sweep(pa_df, predictions, 2024, LABEL)
    assert steady["stability"]["rmse_gate_stable"] is True
    assert steady["stability"]["ordering_gate_stable"] is True

    _stub_compare(monkeypatch, lambda n, opponent: (n > 2, n > 2, 0.05))
    _, flipped = model_evaluation_min_pa_sweep.sweep(pa_df, predictions, 2024, LABEL)
    assert flipped["stability"]["rmse_gate_stable"] is False
    assert flipped["stability"]["ordering_gate_stable"] is False
    assert flipped["by_threshold"]["10"]["low"]["rmse_gate_vs_gbm_full"] is True
    assert flipped["by_threshold"]["50"]["low"]["rmse_gate_vs_gbm_full"] is False


def test_a_missing_scored_arm_fails_loud(case, monkeypatch):
    """The label names the arm being swept; a typo must stop the sweep, not score an opponent."""
    _stub_compare(monkeypatch, lambda n, opponent: (True, True, 0.05))
    pa_df, predictions = case
    with pytest.raises(AssertionError):
        model_evaluation_min_pa_sweep.sweep(pa_df, predictions, 2024, "not_an_arm")
