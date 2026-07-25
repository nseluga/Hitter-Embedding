"""
Phase C.1 — bucketed side-specific trailing averages (architecture §3 Phase C.1).

The naive baseline: project a hitter's side-specific wOBA from what he actually
did against that hand in recent seasons. Its job is not to be good. It is the
floor — the answer to "did you check whether his own past numbers already work?"
If the Phase D model cannot beat this, nothing downstream matters.

TWO variants are built and both are reported, because the pair decomposes a
quantity we want and a single variant hides it:

  raw      — predict his trailing side-specific wOBA, unshrunk. Falls back to the
             side-specific league average only when he has ZERO prior PA vs that
             hand. Deliberately naive: a .450 wOBA in 30 PA is predicted as .450.
  bucketed — the practitioner's version. Blend his trailing wOBA toward the
             side-specific league average by a fixed weight per PA bucket, a crude
             step-function stand-in for regression to the mean.

  raw -> bucketed      measures the value of shrinkage AT ALL
  bucketed -> C.2 (EB) measures the value of doing shrinkage properly
  C.2 -> Phase D       measures the value of cross-hitter structure (the thesis)

The bucket weights below are deliberately crude round numbers. That is the point:
C.1 is the hand-rolled version, and C.2 replaces the step function with the
principled continuous weight n/(n + n*). Making C.1 clever would collapse the two
baselines into the same model and destroy the decomposition.

Window: the 3 seasons preceding the evaluated season. Decided on measurement, not
convention — on the 2024 eval frame, a 3-season window costs the low-exposure
stratum essentially nothing (median prior PA 10.5 -> 10.0, same 117 hitters at zero)
because low-exposure hitters are young or newly arrived and their whole career is
already inside the window. It only trims veterans, who have ample sample either way,
so the recency/aging benefit comes free.

Leakage discipline: every input is drawn from seasons strictly before the evaluated
season, including the league average the predictions shrink toward.
"""

import pandas as pd

from src.data.eval_targets import aggregate

TRAILING_SEASONS = 3

# (min_pa, weight_on_his_own_trailing_wOBA). Crude by design — see module docstring.
# Below 50 PA the observed side-specific wOBA is nearly pure noise (B.1 put the
# stabilization point at ~190 PA), so the practitioner throws it away entirely.
PA_BUCKETS = ((0, 0.0), (50, 0.5), (200, 1.0))

KEY = ["batter", "season", "p_throws"]


def trailing_window(pa_df, eval_season, n_seasons=TRAILING_SEASONS):
    """
    The PA rows a forecaster could see: the n_seasons immediately before eval_season.
    Returns the filtered PA-level frame.
    """
    first = eval_season - n_seasons
    window = pa_df[(pa_df["season"] < eval_season) & (pa_df["season"] >= first)]
    assert len(window) > 0, f"no PA in seasons [{first}, {eval_season})"
    assert window["season"].max() < eval_season, "trailing window leaked the evaluated season"
    return window


def league_average(window):
    """Side-specific league wOBA over the window; the value predictions shrink toward."""
    grouped = window.groupby("p_throws")
    league = grouped["woba_points"].sum() / grouped["in_denominator"].sum()
    return league.to_dict()


def trailing_woba(window):
    """Each hitter's observed wOBA and PA against each pitcher hand over the window."""
    out = aggregate(window, by=("batter", "p_throws"))
    return out[["batter", "p_throws", "woba", "pa"]].rename(
        columns={"woba": "trailing_woba", "pa": "trailing_pa"}
    )


def bucket_weight(trailing_pa, buckets=PA_BUCKETS):
    """
    Weight placed on the hitter's own trailing wOBA, by PA bucket.
    trailing_pa: a Series of PA counts. Returns a Series of weights in [0, 1].
    """
    weight = pd.Series(0.0, index=trailing_pa.index)
    for min_pa, w in sorted(buckets):
        weight[trailing_pa >= min_pa] = w
    return weight


def predict(pa_df, eval_season, variant="bucketed", n_seasons=TRAILING_SEASONS,
            buckets=PA_BUCKETS):
    """
    Project side-specific wOBA for every hitter-hand active in eval_season.
    variant: "raw" (unshrunk, league fallback only at zero PA) or "bucketed"
    (blend toward the side-specific league average by PA bucket).
    Returns a frame with batter/season/p_throws/pred_woba, ready for claim1_eval.
    """
    assert variant in {"raw", "bucketed"}, f"unknown variant {variant!r}"

    window = trailing_window(pa_df, eval_season, n_seasons)
    league = league_average(window)
    trailing = trailing_woba(window)

    # the groups needing a projection are those active in the evaluated season
    targets = (
        pa_df[pa_df["season"] == eval_season][["batter", "p_throws"]]
        .drop_duplicates()
        .merge(trailing, on=["batter", "p_throws"], how="left")
    )
    targets["trailing_pa"] = targets["trailing_pa"].fillna(0.0)
    targets["league_woba"] = targets["p_throws"].map(league)
    assert targets["league_woba"].notna().all(), "a pitcher hand has no league average in the window"

    if variant == "raw":
        # trust his number entirely; league average only where he has no history
        weight = (targets["trailing_pa"] > 0).astype(float)
    else:
        weight = bucket_weight(targets["trailing_pa"], buckets)

    own = targets["trailing_woba"].fillna(0.0)  # masked by weight 0 wherever absent
    targets["pred_woba"] = weight * own + (1.0 - weight) * targets["league_woba"]

    assert targets["pred_woba"].notna().all(), "predictions contain nulls"
    out = targets[["batter", "p_throws", "pred_woba"]].copy()
    out["season"] = eval_season
    return out[KEY + ["pred_woba"]]
