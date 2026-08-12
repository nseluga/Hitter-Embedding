"""
D5-R12 as a standing diagnostic: the platoon asymmetry is emergent, and must stay emergent.

There is no per-hand parameter anywhere in the model. A hitter is one embedding; the only
thing that differs between "vs LHP" and "vs RHP" is the pitcher panel the composition routes
that embedding through. So the textbook asymmetry -- left-handed hitters worse against
left-handed pitchers, right-handed hitters better -- is not fitted, it falls out. A change to
the panel weighting, the handedness shares, or the count chain can quietly destroy it while
every log-loss number stays fine, which is why it gets a test rather than a paragraph.

This reads the shipped arm's composition output. It is a diagnostic over a real artefact, not
a unit test of a function, so it SKIPS when the artefact is absent (a fresh clone, or before an
overnight has produced one) rather than failing. `pytest -m ""` still collects it.

Reproducibility note: the population filter reproduces the review's n = 308 exactly. The
cohort means land within ~0.001 of the review's figures (LHB predicted -0.0310 against
observed -0.0283, RHB +0.0147 against +0.0160); the review did not record how it classified
switch hitters, so the bands below are set wide enough to hold under any of the defensible
choices and still fail if the asymmetry itself goes.
"""

import numpy as np
import pandas as pd
import pytest

from src.data import eval_targets as et

# the shipped arm, named rather than globbed: a glob would silently start testing whatever
# composition ran last, which is how a standing diagnostic stops standing for anything
PREDICTIONS = "results/phase_d/d5_predictions_d9_baseline_head.csv"
EVAL_TARGETS = "data/processed/eval_targets_pa.parquet"

# review's cohort definition -- side-specific PA in the evaluated season
MIN_PA_VS_LHP = 50
MIN_PA_VS_RHP = 150
EXPECTED_N = 308

# D5-R12's recorded figures, with a band that tolerates the switch-hitter ambiguity
LHB_SPLIT = -0.0310
RHB_SPLIT = 0.0147
SPLIT_TOLERANCE = 0.005
MIN_SPLIT_CORRELATION = 0.30  # recorded +0.41


@pytest.fixture(scope="module")
def splits_frame():
    """One row per qualifying hitter: observed and predicted (vs LHP - vs RHP), plus stand."""
    for path in (PREDICTIONS, EVAL_TARGETS):
        if not pd.io.common.file_exists(path):
            pytest.skip(f"{path} absent -- run the composition before this diagnostic")

    predictions = pd.read_csv(PREDICTIONS)
    season = int(predictions["season"].iloc[0])
    pa_df = et.drop_pitcher_batters(pd.read_parquet(EVAL_TARGETS))

    actuals = et.aggregate(pa_df)
    actuals = actuals[actuals["season"] == season]
    merged = actuals.merge(predictions, on=["batter", "season", "p_throws"])

    wide = merged.pivot(index="batter", columns="p_throws",
                        values=["woba", "pred_woba", "denominator"])
    qualified = ((wide[("denominator", "L")] >= MIN_PA_VS_LHP)
                 & (wide[("denominator", "R")] >= MIN_PA_VS_RHP))
    wide = wide[qualified]

    # a switch hitter has two stands; take the one carried by their first PA of the season.
    # Which convention is used moves the cohort means by ~0.001, well inside the band.
    season_pa = pa_df[pa_df["season"] == season]
    stand = season_pa.drop_duplicates("batter").set_index("batter")["stand"]

    return pd.DataFrame({
        "observed": wide[("woba", "L")] - wide[("woba", "R")],
        "predicted": wide[("pred_woba", "L")] - wide[("pred_woba", "R")],
        "stand": stand.reindex(wide.index),
    })


def test_cohort_reproduces(splits_frame):
    """The filter defines the population; if it drifts, every number below means something else."""
    assert len(splits_frame) == EXPECTED_N, \
        f"cohort moved: {len(splits_frame)} hitters, D5-R12 recorded {EXPECTED_N}"
    assert splits_frame["stand"].isin(["L", "R"]).all(), "a hitter has no batting side"


@pytest.mark.parametrize("stand,expected", [("L", LHB_SPLIT), ("R", RHB_SPLIT)])
def test_predicted_platoon_asymmetry(splits_frame, stand, expected):
    """
    The asymmetry itself. Sign first -- a sign flip is the failure that matters, and it would
    survive any tolerance stated as a percentage of a number this small.
    """
    cohort = splits_frame[splits_frame["stand"] == stand]
    predicted = cohort["predicted"].mean()
    observed = cohort["observed"].mean()

    assert np.sign(predicted) == np.sign(expected), \
        f"{stand}HB predicted split {predicted:+.4f}, D5-R12 recorded {expected:+.4f}"
    assert np.sign(predicted) == np.sign(observed), \
        f"{stand}HB predicted {predicted:+.4f} disagrees in sign with observed {observed:+.4f}"
    assert abs(predicted - expected) < SPLIT_TOLERANCE, \
        f"{stand}HB predicted split {predicted:+.4f} against recorded {expected:+.4f}"


def test_asymmetry_is_asymmetric(splits_frame):
    """
    Not a restatement of the two tests above. Both cohorts could shift together -- a level
    change dressed as a platoon effect. What D5-R12 found is that the two sides move in
    OPPOSITE directions, which only a hand-specific pitcher panel can produce.
    """
    means = splits_frame.groupby("stand")["predicted"].mean()
    assert means["L"] < 0 < means["R"], f"cohorts do not straddle zero: {dict(means)}"


def test_individual_splits_correlate(splits_frame):
    """
    Cohort means can come out right while every individual prediction is noise. The +0.41
    individual correlation is the claim that the embedding carries hitter-specific platoon
    skill, which is the part worth protecting.
    """
    r = float(np.corrcoef(splits_frame["observed"], splits_frame["predicted"])[0, 1])
    assert r > MIN_SPLIT_CORRELATION, \
        f"individual split correlation {r:.4f}, D5-R12 recorded +0.41"
