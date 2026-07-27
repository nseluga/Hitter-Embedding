"""
The claim-1 metric — the frozen scoring function every model is graded by (§5.2).

This module is not a model. It is the referee. Every Phase C baseline (bucketed
trailing averages, empirical-Bayes platoon regression, XGBoost context-interaction)
and the Phase D conditional-query model emit the same object: a predicted
side-specific wOBA for each (hitter, held-out season, pitcher hand). This module
turns that into the numbers the thesis is judged on, identically for all of them.

It is written BEFORE any baseline produces a number, deliberately: a yardstick
authored by someone who already knows whose result it will flatter is not a
yardstick. Boundaries and metric definitions here are frozen at commit time and
are not to be re-tuned after seeing a model's score (pre-registration).

What it computes (architecture §5.2):
  - PA-weighted RMSE — calibration. Weighted because a hitter with 400 held-out PA
    has a far less noisy observed wOBA than one with 20; unweighted, the noisiest
    observations dominate the error and the metric mostly measures luck.
  - Rank correlation (Spearman) — ordering. Layer 2 consumes the ORDER (which
    hitters to acquire), not the level. A model can be badly calibrated yet rank
    correctly and still be useful downstream, so both are reported and they are
    allowed to disagree. Rank correlation is unweighted by construction (ranks
    carry no PA), so it is the more noise-exposed of the two — stated, not hidden.
  - Both, stratified by PRIOR side-specific exposure. This is the headline, not a
    breakdown: manifest frozen rule #1 defines beating the baselines only on
    high-exposure veterans as a NULL RESULT.

Leakage boundary (the one that matters here). "Prior exposure" is PA against that
hand in seasons STRICTLY BEFORE the evaluated season — the information a forecaster
would actually have. Stratifying on held-out-season PA would condition on how much
the manager chose to play him AFTER the projection was made, which is exactly the
deployment/selection bias §5.5 warns about, and would leak the outcome's own
sample size into the strata definition. `prior_exposure` asserts this.

Noise-floor deconvolution (companion to the frozen metric, not a replacement). The
held-out "actual" side-specific wOBA is not truth — it is itself a small-sample
measurement. Errors add in quadrature:

    observed RMSE^2 = model error^2 + target sampling noise^2

so every model, however good, pays the same irreducible noise tax and RMSE is
trapped in a narrow band. On the 2024 frame that tax is ~60-70% of the mean squared
error in every stratum (low stratum: floor 0.0465 of an observed 0.0591), which
compresses real differences into what looks like rounding. `score` therefore reports
`noise_floor` and the deconvolved `model_rmse` alongside the raw PA-weighted RMSE.
The raw metric is unchanged and remains the frozen §5.2 number; the extra columns
only make it legible.

The floor is per stratum, not global — it is set by how many PA sit in THAT
stratum's answer key (low ~96 PA -> 0.0465; high ~251 PA -> 0.0330). The
deconvolution assumes model error and target sampling noise are independent, which
holds because no model sees the held-out season. It is an estimate and is labeled
as one wherever reported.

Inputs come from the eval-target table (src/data/eval_targets.py), which is built
from the COMPLETE source, never the filtered modeling table (two-table principle,
decision log 2026-07-15).
"""

import numpy as np
import pandas as pd

from src.config.splits import load_splits

# Stratum boundaries in prior side-specific PA. Derived from the B.1 result, not
# chosen by round-number convenience: side-specific wOBA vs LHP stabilizes at
# n* ~= 226 PA (variance-components estimate on TRAIN seasons, hitters only). Below
# n*/2 the observed split is noise-dominated — the regime the small-sample thesis
# exists to serve. Above 2*n* the hitter's own observed split is trustworthy on its
# own and a model has little to add.
#
# The vs-LHP figure anchors the cuts (rather than vs-RHP's 254) because the scarce
# side is the one the thesis is graded on. Revised 2026-07-27 from 190: the earlier
# value was estimated on a population that still included pitchers taking their own
# at-bats, whose ~.15 wOBA inflated between-hitter signal variance and so biased n*
# downward. Hitters-only CI [199, 264].
STABILIZATION_N_STAR = 226
STRATUM_BOUNDARIES = (STABILIZATION_N_STAR // 2, STABILIZATION_N_STAR * 2)  # (113, 452)
STRATUM_NAMES = ("low", "medium", "high")

# A hitter with a handful of held-out PA has an essentially random observed wOBA;
# scoring against it measures the target's noise, not the model. Hitters below this
# are dropped from scoring and the drop count is REPORTED (never silent — §6).
MIN_EVAL_PA = 25

KEY = ["batter", "season", "p_throws"]


def prior_exposure(pa_df, eval_season):
    """
    Side-specific PA accumulated strictly BEFORE eval_season — the exposure a
    forecaster would have had at projection time.
    pa_df: the PA-level eval-target table; eval_season: the season being projected.
    Returns one row per (batter, p_throws) with prior_pa.
    """
    prior = pa_df[pa_df["season"] < eval_season]
    assert len(prior) > 0, f"no seasons before {eval_season} in the PA table"
    assert prior["season"].max() < eval_season, "prior exposure leaked the evaluated season"

    out = (
        prior.groupby(["batter", "p_throws"], as_index=False)["in_denominator"]
        .sum()
        .rename(columns={"in_denominator": "prior_pa"})
    )
    out["prior_pa"] = out["prior_pa"].astype(float)
    return out


def sampling_noise(pa_df, eval_season):
    """
    Per-group sampling variance of the OBSERVED wOBA — the irreducible target noise.
    A group's observed wOBA is a mean over its PA, so its sampling variance is the
    within-group variance of wOBA points divided by the PA count. This is error the
    answer key contributes, not error the model made; no model can reduce it.
    Returns one row per (batter, p_throws) with noise_var.
    """
    scored = pa_df[(pa_df["season"] == eval_season) & pa_df["in_denominator"]]
    assert len(scored) > 0, f"no scorable PA in season {eval_season}"

    grouped = scored.groupby(["batter", "p_throws"])["woba_points"]
    out = pd.DataFrame({"n": grouped.size(), "within_var": grouped.var(ddof=1)}).reset_index()
    out["noise_var"] = out["within_var"] / out["n"]
    return out[["batter", "p_throws", "noise_var"]]


def assign_stratum(prior_pa, boundaries=STRATUM_BOUNDARIES):
    """
    Bucket prior side-specific PA into the low/medium/high exposure strata.
    prior_pa: a Series of PA counts. Returns a Series of stratum labels.
    Boundaries are half-open: low < b0 <= medium < b1 <= high.
    """
    low_cut, high_cut = boundaries
    assert low_cut < high_cut, "stratum boundaries must be increasing"
    edges = [-np.inf, low_cut, high_cut, np.inf]
    return pd.cut(prior_pa, bins=edges, labels=list(STRATUM_NAMES), right=False)


def build_eval_frame(pa_df, predictions, eval_season, min_eval_pa=MIN_EVAL_PA,
                     boundaries=STRATUM_BOUNDARIES):
    """
    Join model predictions to held-out actuals and attach exposure strata.
    pa_df: PA-level eval targets; predictions: frame with batter/season/p_throws/pred_woba.
    Returns (eval_frame, coverage) where coverage records what was dropped and why.
    """
    from src.data.eval_targets import aggregate, drop_pitcher_batters

    # claim 1 projects HITTERS; pitchers taking their own turn at bat are not the
    # population. A no-op on DH-era eval seasons, correct on any earlier one.
    pa_df = drop_pitcher_batters(pa_df)

    assert set(KEY + ["pred_woba"]).issubset(predictions.columns), \
        f"predictions must have {KEY + ['pred_woba']}, got {list(predictions.columns)}"
    assert predictions["season"].nunique() == 1 and predictions["season"].iloc[0] == eval_season, \
        "predictions must cover exactly the evaluated season"
    assert not predictions.duplicated(KEY).any(), "predictions contain duplicate (batter, season, hand) rows"

    actuals = aggregate(pa_df, by=tuple(KEY))
    actuals = actuals[actuals["season"] == eval_season]
    n_actual_groups = len(actuals)

    # low-PA held-out groups measure target noise, not model skill
    scorable = actuals[actuals["pa"] >= min_eval_pa]

    frame = scorable.merge(predictions[KEY + ["pred_woba"]], on=KEY, how="inner")
    n_unpredicted = len(scorable) - len(frame)

    frame = frame.merge(sampling_noise(pa_df, eval_season), on=["batter", "p_throws"], how="left")
    frame = frame.merge(prior_exposure(pa_df, eval_season), on=["batter", "p_throws"], how="left")
    # a hitter with no prior-season PA vs that hand has zero exposure, not missing
    frame["prior_pa"] = frame["prior_pa"].fillna(0.0)
    frame["stratum"] = assign_stratum(frame["prior_pa"], boundaries)

    assert frame["stratum"].notna().all(), "every scored row must land in exactly one stratum"
    assert frame["pred_woba"].notna().all(), "predictions contain nulls"
    assert np.isfinite(frame["pred_woba"]).all(), "predictions contain non-finite values"

    coverage = {
        "eval_season": int(eval_season),
        "actual_groups": int(n_actual_groups),
        "dropped_below_min_eval_pa": int(n_actual_groups - len(scorable)),
        "min_eval_pa": int(min_eval_pa),
        "dropped_no_prediction": int(n_unpredicted),
        "scored_groups": int(len(frame)),
    }
    return frame, coverage


def pa_weighted_rmse(actual, predicted, pa):
    """Root mean squared error weighting each hitter-hand group by its held-out PA."""
    actual, predicted, pa = np.asarray(actual), np.asarray(predicted), np.asarray(pa, dtype=float)
    assert actual.shape == predicted.shape == pa.shape, \
        f"shape mismatch: {actual.shape} {predicted.shape} {pa.shape}"
    assert pa.sum() > 0, "total PA weight is zero"
    return float(np.sqrt(np.sum(pa * (predicted - actual) ** 2) / pa.sum()))


def rank_correlation(actual, predicted):
    """Spearman rank correlation between predicted and observed wOBA. Unweighted."""
    actual, predicted = pd.Series(np.asarray(actual)), pd.Series(np.asarray(predicted))
    assert len(actual) == len(predicted), "shape mismatch"
    if len(actual) < 3 or actual.nunique() < 2 or predicted.nunique() < 2:
        return float("nan")
    return float(actual.corr(predicted, method="spearman"))


def score(eval_frame):
    """
    Compute the claim-1 metrics overall and within each exposure stratum.
    eval_frame: output of build_eval_frame. Returns one row per stratum plus "all",
    with the low-exposure row being the headline the thesis is judged on.
    """
    rows = []
    for name in list(STRATUM_NAMES) + ["all"]:
        part = eval_frame if name == "all" else eval_frame[eval_frame["stratum"] == name]
        if len(part) == 0:
            rows.append({"stratum": name, "n_hitters": 0, "pa": 0.0,
                         "pa_weighted_rmse": float("nan"), "rank_corr": float("nan")})
            continue
        rmse = pa_weighted_rmse(part["woba"], part["pred_woba"], part["pa"])
        floor = noise_floor(part)
        rows.append({
            "stratum": name,
            "n_hitters": len(part),
            "pa": float(part["pa"].sum()),
            "pa_weighted_rmse": rmse,
            "noise_floor": floor,
            "model_rmse": deconvolve(rmse, floor),
            "rank_corr": rank_correlation(part["woba"], part["pred_woba"]),
        })
    return pd.DataFrame(rows)


def noise_floor(part):
    """PA-weighted irreducible RMSE contributed by noise in the held-out target."""
    weight = part["pa"].astype(float)
    return float(np.sqrt((weight * part["noise_var"]).sum() / weight.sum()))


def deconvolve(rmse, floor):
    """
    Model error with the target's sampling noise removed: sqrt(RMSE^2 - floor^2).
    Clamped at zero — a model scoring below its floor means the floor estimate is
    slightly high, not that the model is better than perfect.
    """
    return float(np.sqrt(max(0.0, rmse**2 - floor**2)))


# ponytail: model-vs-model claims currently compare two absolute RMSEs, each inflated
# by the same target noise. The powerful form is a PAIRED loss differential —
# per-hitter squared-error differences, where the shared noise largely cancels —
# bootstrapped over hitters within stratum for a CI (Diebold-Mariano). Deferred to the
# first genuine head-to-head (C.2 vs C.1) since it needs two real models to be useful.
def skill_score(model_rmse, reference_model_rmse):
    """
    Share of the reference's true error the model eliminates, in variance terms:
    1 - (model/reference)^2. Reference is normally the no-information baseline, whose
    deconvolved error equals the full spread of true talent, so this reads as the
    fraction of between-hitter talent variance the model captures. 0 = knows nothing,
    1 = perfect. Both inputs must already be deconvolved.
    """
    assert reference_model_rmse > 0, "reference model error must be positive"
    return float(1.0 - (model_rmse / reference_model_rmse) ** 2)


def evaluate(pa_df, predictions, eval_season, min_eval_pa=MIN_EVAL_PA,
             boundaries=STRATUM_BOUNDARIES):
    """
    End-to-end: join predictions to actuals, stratify, and score.
    Returns (metrics_table, coverage_dict). Coverage must be reported alongside
    the metrics — a scored subset presented alone reads as full coverage.
    """
    frame, coverage = build_eval_frame(pa_df, predictions, eval_season, min_eval_pa, boundaries)
    return score(frame), coverage


def assert_not_test_season(eval_season, config=None):
    """Guard: refuse to score against the frozen test season outside a final run."""
    config = config or load_splits()
    test_seasons = config["split"]["test"]
    assert eval_season not in test_seasons, (
        f"season {eval_season} is the frozen TEST season {test_seasons}. Test is read once, "
        "for the final reported result — never for model selection or iteration."
    )
