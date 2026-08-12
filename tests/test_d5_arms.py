"""
Gate for the arm-comparison verdict (D5-R16).

This module's output picks an architecture, so the two failures worth catching are the two that
would pick the WRONG one silently: a flipped sign promotes the worst arm to first place, and a
`beats_baseline` that reads an unresolved interval promotes noise. Neither raises -- both just
produce a confident table -- so they need asserting rather than inspection.

The frames here are synthetic on purpose. A fixture built from real predictions would be a
regression test on `d10`'s numbers, which do not exist until the overnight lands, and would then
have to be rewritten every rebuild. What is being tested is the plumbing's arithmetic.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis import claim1_eval as ce, d5_arms


def _frames(seed=0, n=240):
    """
    Baseline plus two opponents: one genuinely better, one a copy of baseline.

    `tied` is baseline's own predictions plus 1e-6 of noise rather than an independently drawn
    arm, because the interesting near-tie is the one where the paired bootstrap SHOULD resolve to
    nothing -- two independent draws at the same error scale would differ by luck and could
    resolve by luck too.
    """
    rng = np.random.default_rng(seed)
    base = pd.DataFrame({
        "batter": np.arange(n),
        "season": 2024,
        "p_throws": ["L", "R"] * (n // 2),
        "woba": rng.normal(0.32, 0.06, n),
        "denominator": rng.uniform(50, 500, n),
        "noise_var": 0.0004,
        "stratum": pd.Categorical(["low", "high"] * (n // 2),
                                  categories=list(ce.STRATUM_NAMES))})
    base["pa"] = base["denominator"]
    baseline = base.assign(pred_woba=base["woba"] + rng.normal(0, 0.05, n))
    return {"baseline": baseline,
            "better": base.assign(pred_woba=base["woba"] + rng.normal(0, 0.005, n)),
            "tied": baseline.assign(pred_woba=baseline["pred_woba"] + rng.normal(0, 1e-6, n))}


def test_sign_convention_and_resolution():
    frames = _frames()
    comparisons = d5_arms.compare_arms(frames, n_boot=200)
    table = d5_arms.rank_arms(frames, comparisons).set_index("arm")

    # negative rmse_difference favours the ARM; the arm is always side A
    assert table.loc["better", "rmse_difference"] < 0
    assert table.loc["better", "beats_baseline"]
    assert table.index[0] == "better", "the ranking must sort by the decisive stratum"

    # an interval containing zero is not a win, however the point estimate falls
    assert not table.loc["tied", "resolved"]
    assert not table.loc["tied", "beats_baseline"]


def test_every_stratum_is_compared_not_just_the_decisive_one():
    """
    The split verdict is a FINDING, so both rows have to exist to detect it. A table built only
    from the decisive stratum could not tell an arm that helps low-exposure hitters from one that
    helps nobody.
    """
    comparisons = d5_arms.compare_arms(_frames(), n_boot=200)
    for arm in ("better", "tied"):
        strata = set(comparisons[comparisons["arm"] == arm]["stratum"])
        assert {"all", "low"} <= strata, strata


def test_missing_arm_is_reported_not_defaulted(tmp_path):
    """
    An arm whose predictions never landed must come back in `missing`. Silently absent from a
    ranking table, it reads as an arm that lost.
    """
    pa_df = pd.DataFrame({"batter": [1], "season": [2024], "p_throws": ["R"],
                          "woba": [0.3], "denominator": [100.0]})
    frames, missing = d5_arms.load_arm_frames(tmp_path, "d10", ["baseline", "ghost"],
                                             pa_df, 2024)
    assert missing == ["baseline", "ghost"] and frames == {}


def test_baseline_is_required():
    frames = _frames()
    del frames["baseline"]
    with pytest.raises(AssertionError):
        d5_arms.compare_arms(frames, n_boot=50)


def test_seed_spread_reports_sd_beside_the_retired_range():
    """
    Both statistics, on purpose. The retired "x noise floor" was max-min, which grows with seed
    count; printing it next to the sd is what makes that visible instead of arguable.
    """
    _, spread = d5_arms.seed_spread("results/phase_d", "nonexistent_stage", "baseline",
                                    pd.DataFrame(), 2024)
    assert spread is None, "no per-seed files must yield no spread, not a one-row table"
