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

WHICH PA. Two counts exist per group and they are not interchangeable: `pa`, every
completed plate appearance, and `denominator`, the wOBA denominator, which excludes
intentional walks, sac bunts, and interference (~0.8% of PA). wOBA is
numerator/denominator, so the DENOMINATOR is what sets how precisely a group's wOBA
is measured — Var(observed wOBA) = within-group variance / denominator. Everything
that expresses "how much this group's observation is worth" therefore uses the
denominator: the min-PA filter, the RMSE weight, the noise-floor weight, the paired
bootstrap, prior exposure, and the stratum boundaries (whose n* was itself estimated
in denominator units). `pa` is retained and reported, but only as description.

Until 2026-07-29 the filter and the RMSE weight used total `pa` while the noise floor
and the strata used the denominator. The effect was ~1% and second-order, but it was
systematic rather than noise — intentional walks accrue to the best hitters, sac bunts
to the weakest — so it slightly over-weighted the top of the distribution relative to
the precision of what was being weighted. C.1, C.2 and C.3 were already denominator-
based internally, so this also removed a units boundary between the referee and the
models it grades.
  - Rank correlation (Spearman) — ordering. Layer 2 consumes the ORDER (which
    hitters to acquire), not the level. A model can be badly calibrated yet rank
    correctly and still be useful downstream, so both are reported and they are
    allowed to disagree. Reported in two forms. The DENOMINATOR-WEIGHTED
    coefficient is the headline: an 11-PA group's observed wOBA is close to a coin
    flip, and unweighted it casts the same vote as a 500-PA group's, so the
    unweighted statistic is dominated by the answer key's noise in exactly the
    stratum the thesis is about. The unweighted coefficient is retained beside it
    because it is what §5.2 names and what every Phase C number was first reported
    under. Either way a claim that two models ORDER differently goes through
    `paired_rank_difference`, never through two bare numbers.

    Until 2026-08-01 only the unweighted form existed, on the stated grounds that
    rank correlation is "unweighted by construction (ranks carry no PA)". Ranks
    carry no PA, but the correlation computed OVER those ranks takes weights like
    any other, so the weighted form was never rejected — it was assumed
    unavailable. See `weighted_rank_correlation` for the estimator and its source.
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
#
# This filter is NOT free and is not a neutral hygiene step. It conditions on
# eval-season playing time, which is decided AFTER the projection is made and partly
# BY how the hitter performed — the same deployment/selection bias documented above as
# the reason not to stratify on held-out PA. Even at 10 it removes 18.3% of the low
# stratum against 2.4% of the high — it trims the headline stratum's worst performers
# and so flatters every model's level, and no threshold makes that go away (a hitter
# who never played cannot be scored). At 25 those figures were 36.6% and 6.2%. It is
# therefore accompanied by a reported sensitivity: `c_report.py` re-scores the headline
# comparison across MIN_EVAL_PA_SENSITIVITY and commits the result
# (results/phase_c/c_min_eval_pa_sensitivity.csv), so a conclusion that depends on the
# cut is visible rather than assumed away.
#
# Set to 10, revised from 25 on 2026-07-29 (Nate's call, decision log). The reason is
# POPULATION COVERAGE, not the score: at 25 the filter removed 36.6% of the low
# stratum, and hitters with little playing time are the object of study, so a frame
# that discards a third of them measures a different population than the one claim 1
# is about. At 10 the low-stratum drop is 18.3%. In side-specific terms 10 PA is a
# median ~8 games of exposure against that hand (1.6 PA/game vs LHP, 2.0 vs RHP), not
# 2-3 — these are per-hand counts, not whole-season ones.
#
# Stated plainly because it cannot be inferred later: C.3's margin over C.2 is LARGER
# at 10 than at 25 (-0.0031 vs -0.0026), and that was known before the change. The cut
# was not chosen for it. The guard against that hazard is that 25 and 50 remain in the
# committed sweep, so the claim is shown to hold across a 3x swing in censoring rather
# than resting on the chosen frame.
MIN_EVAL_PA = 10
MIN_EVAL_PA_SENSITIVITY = (10, 25, 50)

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

    # Strata are attached BEFORE the min-PA filter, not after, so the coverage report
    # can say which strata the dropped groups came from. The aggregate drop count on
    # its own is misleading here: the filter is far from uniform across strata (it
    # removes ~37% of the low stratum against ~6% of the high), and low is the
    # stratum the thesis is graded on. Stratification uses PRIOR exposure, so
    # computing it for dropped groups leaks nothing.
    actuals = actuals.merge(prior_exposure(pa_df, eval_season), on=["batter", "p_throws"],
                            how="left")
    # a hitter with no prior-season PA vs that hand has zero exposure, not missing
    actuals["prior_pa"] = actuals["prior_pa"].fillna(0.0)
    actuals["stratum"] = assign_stratum(actuals["prior_pa"], boundaries)

    # low-PA held-out groups measure target noise, not model skill
    scorable = actuals[actuals["denominator"] >= min_eval_pa]
    dropped = actuals[actuals["denominator"] < min_eval_pa]

    frame = scorable.merge(predictions[KEY + ["pred_woba"]], on=KEY, how="inner")
    n_unpredicted = len(scorable) - len(frame)

    frame = frame.merge(sampling_noise(pa_df, eval_season), on=["batter", "p_throws"], how="left")

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
        "by_stratum": stratum_coverage(scorable, dropped),
    }
    return frame, coverage


def stratum_coverage(scorable, dropped):
    """
    Per-stratum accounting for the min-PA filter: how many groups it removed, and how
    the removed hitters actually hit.

    This exists because the filter censors on a POST-projection quantity — how much
    the manager chose to play him in the evaluated season — and managers bench hitters
    who are going badly. So the drop is concentrated in the low-exposure stratum and
    removes its worst performers, which raises every stratum's observed mean and
    therefore UNDERSTATES the over-prediction bias every shrink-to-league-average
    baseline carries. Reporting one aggregate drop count hides all of that (§6:
    a coverage shortcut is stated where the result is stated).
    """
    out = {}
    for name in STRATUM_NAMES:
        kept, cut = scorable[scorable["stratum"] == name], dropped[dropped["stratum"] == name]
        total = len(kept) + len(cut)
        out[name] = {
            "scored": int(len(kept)),
            "dropped": int(len(cut)),
            "share_dropped": float(len(cut) / total) if total else float("nan"),
            "mean_woba_scored": float(kept["woba"].mean()) if len(kept) else float("nan"),
            "mean_woba_dropped": float(cut["woba"].mean()) if len(cut) else float("nan"),
        }
    return out


def pa_weighted_rmse(actual, predicted, pa):
    """
    Root mean squared error weighting each hitter-hand group by its held-out PA.
    Callers pass the wOBA DENOMINATOR, not total PA — that is the count wOBA is a
    mean over, so it is the one that sets the observation's precision (see the
    module docstring's "WHICH PA").
    """
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


def weighted_ranks(values, weights):
    """
    Weighted ranks per wCorr: a value's rank is the total weight strictly below it,
    plus a tie term. values/weights: equal-length arrays. Returns one rank per row.

    rank_j = a_j + b_j, where a_j = sum of weights of all smaller values and
    b_j = ((m+1)/2) * mean weight of the m units tied at this value. With every
    weight equal to 1 this collapses to the ordinary mid-rank, which is the
    property the equal-weights test asserts.

    Reading the scale: a rank is CUMULATIVE WEIGHT — the PA below a group plus its
    own — rather than a position in a roster list, so the coefficient responds to how
    the model orders plate appearances rather than row labels. Treat that as the
    analogy it is, not an identity. Replicating each row w times and taking an
    ordinary Spearman gives a slightly different number (~1e-2 on 12-row cases),
    because that convention mid-ranks the w copies while wCorr places a unit at the
    top of its own weight block, as the unweighted convention does. Neither is
    biased: under independent inputs this estimator centres on zero at weight
    spreads from equal through 10000:1, at n = 30 and n = 380.
    """
    frame = pd.DataFrame({"value": np.asarray(values), "weight": np.asarray(weights, dtype=float)})
    assert (frame["weight"] > 0).all(), "weighted ranks need strictly positive weights"
    by_value = frame.groupby("value")["weight"]
    # groupby sorts by key, so shifting the group totals down and accumulating gives
    # each distinct value the total weight lying strictly below it
    below = by_value.sum().shift(1).fillna(0.0).cumsum()
    tie_term = (by_value.size() + 1) / 2 * by_value.mean()
    return frame["value"].map(below + tie_term).to_numpy()


def weighted_pearson(x, y, weights):
    """Pearson correlation with observation weights, using weighted means and moments."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    weight = np.asarray(weights, dtype=float) / np.sum(weights)
    x_centered, y_centered = x - np.sum(weight * x), y - np.sum(weight * y)
    spread = np.sum(weight * x_centered ** 2) * np.sum(weight * y_centered ** 2)
    if spread <= 0:
        return float("nan")
    return float(np.sum(weight * x_centered * y_centered) / np.sqrt(spread))


def weighted_rank_correlation(actual, predicted, weights):
    """
    Denominator-weighted Spearman correlation — the ordering metric the Phase D gate
    reads. Callers pass the wOBA denominator, per the module docstring's "WHICH PA".

    Both steps are weighted: ranks by `weighted_ranks`, then a weighted Pearson over
    those ranks. This is wCorr's construction, adopted rather than invented because
    it is the only documented implementation and it reduces exactly to the unweighted
    Spearman when all weights are equal. wCorr's own weights are inverse-probability
    sampling weights and ours are precision weights, so the ESTIMATOR transfers but
    its consistency results, which are stated for the sampling interpretation, do not.

    Reference: Bailey, Emad, Zhang & Xie, "wCorr Formulas" (2023) — which also states
    outright that no commonly accepted weighted Spearman coefficient exists, Stata
    does not estimate one, and SAS does not document its method.
    """
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    weights = np.asarray(weights, dtype=float)
    assert actual.shape == predicted.shape == weights.shape, \
        f"shape mismatch: {actual.shape} {predicted.shape} {weights.shape}"
    if len(actual) < 3 or len(np.unique(actual)) < 2 or len(np.unique(predicted)) < 2:
        return float("nan")
    return weighted_pearson(weighted_ranks(actual, weights),
                            weighted_ranks(predicted, weights), weights)


# Which answer key a score is measured against (D5-R8, settled Q13). Realized wOBA stays
# PRIMARY and is the default: xwOBA as the primary target would change the claim from runs to a
# latent quality measure, re-score every frozen Phase C number, and -- decisively -- make the
# answer key another model's output rather than ground truth, reducing the task to predicting
# next-season batted-ball quality. xwOBA is a SECOND target, scored beside the first, because
# the gap between the two decomposes error into "wrong about batted-ball quality" versus
# "could not have known".
TARGETS = {"woba": ("woba", "denominator"),
           "xwoba": ("xwoba", "xwoba_denominator")}


def score(eval_frame, target="woba"):
    """
    Compute the claim-1 metrics overall and within each exposure stratum.
    eval_frame: output of build_eval_frame. Returns one row per stratum plus "all",
    with the low-exposure row being the headline the thesis is judged on.
    target: "woba" (primary) or "xwoba" (the second target, D5-R8).
    """
    assert target in TARGETS, f"unknown target {target!r}"
    actual_column, weight_column = TARGETS[target]
    assert actual_column in eval_frame.columns, (
        f"the eval frame carries no {actual_column!r} column -- rebuild the PA table with "
        f"eval_targets.py")

    rows = []
    for name in list(STRATUM_NAMES) + ["all"]:
        part = eval_frame if name == "all" else eval_frame[eval_frame["stratum"] == name]
        # a group Statcast never estimated has no second target; it leaves the xwOBA scoring
        # set rather than being imputed into it
        part = part[part[actual_column].notna() & (part[weight_column] > 0)]
        if len(part) == 0:
            # `target` must be carried even on an empty row: without it a caller filtering the
            # table by target silently drops the empty strata, and "this stratum had nobody with
            # a second target" is exactly the finding worth keeping
            rows.append({"stratum": name, "target": target, "n_hitters": 0, "pa": 0.0,
                         "pa_weighted_rmse": float("nan"), "rank_corr": float("nan"),
                         "rank_corr_weighted": float("nan")})
            continue
        actual, weight = part[actual_column], part[weight_column]
        rmse = pa_weighted_rmse(actual, part["pred_woba"], weight)
        # the noise floor is the sampling variance of REALIZED wOBA; it does not describe
        # xwOBA's own error, so it is not reported against the second target
        floor = noise_floor(part) if target == "woba" else float("nan")
        rows.append({
            "stratum": name,
            "target": target,
            "n_hitters": len(part),
            "pa": float(part["pa"].sum()),
            "scoring_pa": float(weight.sum()),
            "pa_weighted_rmse": rmse,
            "noise_floor": floor,
            "model_rmse": deconvolve(rmse, floor) if target == "woba" else float("nan"),
            "rank_corr": rank_correlation(actual, part["pred_woba"]),
            "rank_corr_weighted": weighted_rank_correlation(actual, part["pred_woba"], weight),
        })
    return pd.DataFrame(rows)


def achievable_floor(eval_frame):
    """
    PA-weighted RMSE between realized wOBA and xwOBA, per stratum (D5-R8, settled Q13).

    An approximate FLOOR on what any expected-value model can score against a luck-contaminated
    target. The composition's prediction is already an expected value: `V` is a league-average
    outcome table with no channel for "this one found a hole", so the model structurally cannot
    represent the part of realized wOBA that is fielder placement and sequencing. Grading it
    against realized wOBA charges it for variance it cannot express, and this is the size of
    that charge. If the floor sits close to the model's RMSE, the D.5 null is largely answer-key
    noise rather than a modeling failure.

    Approximate in BOTH directions, so it is a reference line and is never subtracted from a
    score: xwOBA uses ACTUAL strikeouts and walks where the model predicts them, which makes
    this too low, and xwOBA's own (exit velocity, launch angle) map is imperfect, which makes it
    too high.
    """
    assert "xwoba" in eval_frame.columns, (
        "the eval frame carries no 'xwoba' column -- rebuild the PA table with eval_targets.py")
    rows = []
    for name in list(STRATUM_NAMES) + ["all"]:
        part = eval_frame if name == "all" else eval_frame[eval_frame["stratum"] == name]
        part = part[part["xwoba"].notna() & (part["xwoba_denominator"] > 0)]
        if len(part) == 0:
            rows.append({"stratum": name, "n_hitters": 0,
                         "achievable_floor": float("nan"), "xwoba_bias": float("nan")})
            continue
        weight = part["denominator"].astype(float)
        rows.append({
            "stratum": name,
            "n_hitters": len(part),
            "achievable_floor": pa_weighted_rmse(part["woba"], part["xwoba"], weight),
            # a nonzero mean gap is the denominator difference plus xwOBA's own calibration,
            # reported so the floor is not read as a pure noise term
            "xwoba_bias": float(np.average(part["xwoba"] - part["woba"], weights=weight)),
        })
    return pd.DataFrame(rows)


def noise_floor(part):
    """PA-weighted irreducible RMSE contributed by noise in the held-out target."""
    weight = part["denominator"].astype(float)
    return float(np.sqrt((weight * part["noise_var"]).sum() / weight.sum()))


def deconvolve(rmse, floor):
    """
    Model error with the target's sampling noise removed: sqrt(RMSE^2 - floor^2).
    Clamped at zero — a model scoring below its floor means the floor estimate is
    slightly high, not that the model is better than perfect.
    """
    return float(np.sqrt(max(0.0, rmse**2 - floor**2)))


def batter_clusters(part):
    """
    Row indices grouped by batter — the independent unit for resampling.

    A hitter contributes up to TWO rows to the eval frame (vs LHP and vs RHP), and
    their errors are not independent: both are driven by the same underlying talent,
    the same season of health and aging, the same home park. Resampling rows would
    count a batter's two rows as two independent draws and understate every interval.
    On the 2024 frame the inflation is real but modest — the low stratum's 295 rows
    come from 205 batters, the full frame's 1015 from 563.

    Returns a list of index arrays, one per batter, for `_resample`.
    """
    codes = pd.factorize(np.asarray(part["batter"]))[0]
    order = np.argsort(codes, kind="stable")
    return np.split(order, np.flatnonzero(np.diff(codes[order])) + 1)


def _resample(clusters, rng):
    """One bootstrap replicate: draw batters with replacement, keep all their rows."""
    picks = rng.integers(0, len(clusters), len(clusters))
    return np.concatenate([clusters[i] for i in picks])


def _paired_setup(frame_a, frame_b, seed):
    """Shared preamble for the paired comparisons: merge, verify alignment, seed."""
    merged = frame_a.merge(frame_b[KEY + ["pred_woba"]], on=KEY, suffixes=("_a", "_b"))
    assert len(merged) == len(frame_a) == len(frame_b), \
        "paired comparison requires both models to score exactly the same groups"
    return merged, np.random.default_rng(seed)


def paired_rmse_difference(frame_a, frame_b, n_boot=2000, seed=0, ci=(2.5, 97.5)):
    """
    Head-to-head CALIBRATION comparison as a PAIRED bootstrap (Diebold-Mariano in
    spirit).

    Comparing two absolute RMSEs is weak here: each is ~60-70% the SAME target noise,
    so a real improvement reads as rounding and neither number carries an interval.
    Scoring both models on the same resampled hitters cancels the shared noise, so the
    difference is far better resolved than either level.

    frame_a, frame_b: build_eval_frame outputs for the two models. Returns one row per
    stratum with rmse_a, rmse_b, their difference (negative favours A), and a
    percentile CI. Resamples BATTERS, the independent unit (see `batter_clusters`).
    """
    merged, rng = _paired_setup(frame_a, frame_b, seed)

    rows = []
    for name in list(STRATUM_NAMES) + ["all"]:
        part = merged if name == "all" else merged[merged["stratum"] == name]
        if len(part) < 3:
            continue
        actual, pa = part["woba"].to_numpy(), part["denominator"].to_numpy(dtype=float)
        error_a = (part["pred_woba_a"].to_numpy() - actual) ** 2
        error_b = (part["pred_woba_b"].to_numpy() - actual) ** 2
        weighted = lambda err, idx: float(np.sqrt(np.sum(pa[idx] * err[idx]) / pa[idx].sum()))

        clusters = batter_clusters(part)
        full = np.arange(len(part))
        draws = [weighted(error_a, i) - weighted(error_b, i)
                 for i in (_resample(clusters, rng) for _ in range(n_boot))]
        rows.append({
            "stratum": name,
            "n_hitters": len(part),
            "n_batters": len(clusters),
            "rmse_a": weighted(error_a, full),
            "rmse_b": weighted(error_b, full),
            "rmse_difference": weighted(error_a, full) - weighted(error_b, full),
            "ci_low": float(np.percentile(draws, ci[0])),
            "ci_high": float(np.percentile(draws, ci[1])),
            "favours_a_share": float(np.mean(np.asarray(draws) < 0)),
        })
    return pd.DataFrame(rows)


def paired_rank_difference(frame_a, frame_b, n_boot=2000, seed=0, ci=(2.5, 97.5),
                           weighted=True):
    """
    Head-to-head ORDERING comparison — the rank counterpart of
    paired_rmse_difference, and required for exactly the same reason.

    Two bare rank correlations are the weakest comparison in this project, not the
    strongest: both models are ranked against the SAME noisy answer key. Quoting
    "0.169 vs 0.102" as a disagreement between the two frozen metrics is an inference
    about a difference, and a difference needs an interval — the same standard
    `paired_rmse_difference` was written to enforce on the other metric. Without this,
    the ordering claim was the one number in Phase C held to a lower evidentiary bar
    than the rest.

    `weighted` selects the coefficient. The default is the denominator-weighted
    Spearman, which the Phase D ordering gate reads; pass False for the unweighted
    §5.2 statistic, which is what Phase C first reported. Weighting removes one of
    the two noise sources — a low-denominator group no longer votes as loudly as a
    well-measured one — but not the other, since the target itself stays noisy.

    SIGN CONVENTION, deliberately opposite to the RMSE version because higher rank
    correlation is better: `rank_difference` = rank_a - rank_b, so POSITIVE favours A.
    `favours_a_share` keeps its meaning across both functions — the share of
    resamples in which A is the better model.

    Degenerate resamples (a replicate with fewer than 3 distinct predictions) yield
    NaN and are dropped; `n_draws` reports how many survived so a stratum that is
    mostly degenerate cannot masquerade as a tight interval.
    """
    merged, rng = _paired_setup(frame_a, frame_b, seed)

    rows = []
    for name in list(STRATUM_NAMES) + ["all"]:
        part = merged if name == "all" else merged[merged["stratum"] == name]
        if len(part) < 3:
            continue
        actual = part["woba"].to_numpy()
        pred_a, pred_b = part["pred_woba_a"].to_numpy(), part["pred_woba_b"].to_numpy()
        weight = part["denominator"].to_numpy(dtype=float)

        def coefficient(predicted, rows=slice(None)):
            if weighted:
                return weighted_rank_correlation(actual[rows], predicted[rows], weight[rows])
            return rank_correlation(actual[rows], predicted[rows])

        clusters = batter_clusters(part)
        draws = np.array([
            coefficient(pred_a, i) - coefficient(pred_b, i)
            for i in (_resample(clusters, rng) for _ in range(n_boot))
        ])
        draws = draws[np.isfinite(draws)]
        assert len(draws) > n_boot // 2, \
            f"stratum {name}: {n_boot - len(draws)} of {n_boot} resamples were degenerate"

        rank_a = coefficient(pred_a)
        rank_b = coefficient(pred_b)
        rows.append({
            "stratum": name,
            "n_hitters": len(part),
            "n_batters": len(clusters),
            "n_draws": len(draws),
            "weighted": bool(weighted),
            "rank_a": rank_a,
            "rank_b": rank_b,
            "rank_difference": rank_a - rank_b,
            "ci_low": float(np.percentile(draws, ci[0])),
            "ci_high": float(np.percentile(draws, ci[1])),
            "favours_a_share": float(np.mean(draws > 0)),
        })
    return pd.DataFrame(rows)


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


def assert_not_test_season(eval_season, config=None, final_run=False):
    """
    Guard: refuse to score against the frozen test season outside a final run.

    `final_run` is the deliberate escape, not a bypass. The 2026-08-01 selection
    frame puts Phase D's headline on the test season, so this call must be reachable
    exactly once — with the flag set explicitly by a caller that means it. Deleting
    the guard instead would leave nothing between an exploratory re-run and spending
    the season, which is the failure it exists to prevent.
    """
    config = config or load_splits()
    test_seasons = config["split"]["test"]
    if final_run:
        return
    assert eval_season not in test_seasons, (
        f"season {eval_season} is the frozen TEST season {test_seasons}. Test is read once, "
        "for the final reported result — never for model selection or iteration. "
        "A genuine final run passes final_run=True (--final-run)."
    )
