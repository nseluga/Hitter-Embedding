"""
Phase E — evaluation (docs/phase-e-spec.md).

Two halves, run in this order and never merged:

  VALIDATE the instrument   E.1 matched population, E.2 contamination,
                            E.3 level closure, E.4 calibration
  EVALUATE effectiveness    E.5 platoon differential against an explicit Route A

Everything here is READ-ONLY with respect to the model. It consumes the prediction
tables Phase D.5 already wrote and the PA-level eval-target table. It trains nothing,
scores no new season, and touches no architecture, loss, or pipeline code, so the
CLAUDE.md ML verification gate's training-run items do not apply (spec §10).

E.6 (swing-head calibration on real pitches) needs checkpoints and a forward pass and
lives in `e_swing.py`.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import claim1_eval
from src.data import eval_targets
from src.model import query

DEFAULT_OUT_DIR = "results/phase_e"
DEFAULT_ARM = "d10_baseline"


# ---------------------------------------------------------------- shared helpers

def trained_batters(manifest):
    """
    The batter ids holding a real embedding row. Row 0 is reserved for cold start and is
    one shared vector, so a hitter mapped to it is not a trained hitter (query.py:375-407).
    """
    return {int(batter) for batter, row in manifest["vocabulary"].items() if int(row) != 0}


def per_group_absorbing(pa_df, seasons):
    """
    Observed absorbing-state shares per (batter, p_throws) over `seasons`, restricted to
    the wOBA denominator, plus that group's denominator count.

    Uses `query.ABSORBING_EVENTS` and takes `bip` as the remainder, exactly as
    `query.observed_absorbing_rates` does, so a new Statcast event string cannot fall out
    of one side of a comparison and not the other.
    """
    window = pa_df[np.isin(pa_df["season"].to_numpy(), seasons)
                   & pa_df["in_denominator"].to_numpy()]
    assert len(window), f"no plate appearances in the denominator for seasons {seasons}"
    events = window["events"].to_numpy()
    out = window[["batter", "p_throws"]].copy()
    named = np.zeros(len(window), dtype=bool)
    for key in ("bb", "k", "hbp"):
        flag = np.isin(events, query.ABSORBING_EVENTS[key])
        out[key] = flag.astype("float64")
        named |= flag
    out["bip"] = (~named).astype("float64")
    grouped = out.groupby(["batter", "p_throws"], as_index=False).agg(
        **{key: (key, "sum") for key in query.ABSORBING_KEYS}, denominator=("bb", "size"))
    for key in query.ABSORBING_KEYS:
        grouped[key] = grouped[key] / grouped["denominator"]
    return grouped


def weighted_average_rates(frame, weight_column, columns):
    """PA-weighted mean of each named column. Returns a plain dict."""
    weight = frame[weight_column].to_numpy(dtype="float64")
    assert weight.sum() > 0, "total weight is zero"
    return {column: float(np.average(frame[column].to_numpy(dtype="float64"), weights=weight))
            for column in columns}


def weighted_ols(x, y, weight):
    """
    Weighted least squares of y on x with an intercept. Returns (intercept, slope, se_slope).

    The standard error is the heteroskedasticity-robust (HC0) sandwich form, because the
    residual variance here is structurally non-constant: a group's observed rate is a mean
    over its own PA count, so a 10-PA group and a 600-PA group do not carry the same noise
    and a homoskedastic SE would understate every interval on the sparse end.
    """
    x, y, weight = (np.asarray(v, dtype="float64") for v in (x, y, weight))
    assert x.shape == y.shape == weight.shape, "shape mismatch"
    design = np.column_stack([np.ones_like(x), x])
    root = np.sqrt(weight)
    beta, *_ = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)
    residual = y - design @ beta
    # HC0 with weights: (X'WX)^-1 . X'W diag(e^2) W X . (X'WX)^-1
    bread = np.linalg.inv(design.T @ (design * weight[:, None]))
    meat = design.T @ (design * (weight ** 2 * residual ** 2)[:, None])
    cov = bread @ meat @ bread
    return float(beta[0]), float(beta[1]), float(np.sqrt(max(cov[1, 1], 0.0)))


# ---------------------------------------------------------------- E.1

def matched_population(predictions, pa_df, manifest, eval_season):
    """
    E.1 (spec §3). The §8 fidelity check's two sides count different populations: the
    modelled side is trained hitters weighted by their eval-season PA, the observed side is
    every plate appearance in the train window. Selection into "has a trained row" is
    selection on exposure, so the comparison carries a bias of known sign.

    Returns the 2x2 of {train window, eval window} x {all hitters, matched} for all four
    absorbing rates, where MATCHED means: the same trained-hitter set the modelled side
    scores, per-hitter rates, weighted by the same eval-season denominators.

    `pooled_all` over the train window reproduces `query.observed_absorbing_rates` exactly.
    That identity is the proof this measures a population difference and not a code
    difference, and it is asserted here rather than left to a test.
    """
    train_seasons = manifest["train_seasons"]
    trained = trained_batters(manifest)

    # the weights the modelled side actually uses, lifted from query.league_fidelity
    counts = (pa_df[(pa_df["season"] == eval_season) & pa_df["in_denominator"]]
              .groupby(["batter", "p_throws"]).size().rename("pa").reset_index())
    scored = predictions.merge(counts, on=["batter", "p_throws"], how="inner")
    scored = scored[scored["batter"].isin(trained)]
    assert len(scored), "no trained hitter survived the plate-appearance join"
    modelled = weighted_average_rates(scored, "pa",
                                      [f"rate_{key}" for key in query.ABSORBING_KEYS])
    modelled = {key: modelled[f"rate_{key}"] for key in query.ABSORBING_KEYS}

    rows = []
    for window_name, seasons in (("train", train_seasons), ("eval", [eval_season])):
        pooled, n_pooled = query.observed_absorbing_rates(pa_df, seasons)
        rows.append({"window": window_name, "population": "pooled_all", "n_pa": n_pooled,
                     "n_groups": np.nan, **pooled})

        groups = per_group_absorbing(pa_df, seasons)
        matched = groups.merge(scored[["batter", "p_throws", "pa"]],
                               on=["batter", "p_throws"], how="inner")
        assert len(matched), f"no matched group in the {window_name} window"
        rates = weighted_average_rates(matched, "pa", list(query.ABSORBING_KEYS))
        rows.append({"window": window_name, "population": "matched",
                     "n_pa": float(matched["denominator"].sum()),
                     "n_groups": int(len(matched)), **rates})

    table = pd.DataFrame(rows)

    # the identity gate: the pooled train-window row IS the number the shipped check uses
    reference, _ = query.observed_absorbing_rates(pa_df, train_seasons)
    pooled_train = table[(table["window"] == "train")
                         & (table["population"] == "pooled_all")].iloc[0]
    for key in query.ABSORBING_KEYS:
        assert abs(pooled_train[key] - reference[key]) < 1e-12, \
            f"E.1's pooled path does not reproduce observed_absorbing_rates on {key}"

    verdict = {}
    for key in query.ABSORBING_KEYS:
        entry = {"modelled": modelled[key], "tolerance": query.FIDELITY_TOLERANCE[key]}
        for _, row in table.iterrows():
            name = f"{row['window']}_{row['population']}"
            entry[name] = float(row[key])
            entry[f"relative_error_vs_{name}"] = modelled[key] / float(row[key]) - 1.0
            entry[f"pass_vs_{name}"] = bool(
                abs(entry[f"relative_error_vs_{name}"]) <= query.FIDELITY_TOLERANCE[key])
        verdict[key] = entry
    return table, verdict


# ---------------------------------------------------------------- E.2

def contamination(predictions, pa_df, manifest, eval_season, key="bb"):
    """
    E.2 (spec §4). Whether the composition's rate bias is a uniform level shift or scales
    with the hitter's own rate. A level shift leaves every rank untouched and cannot touch
    claim-1's headline; a slope compresses the spread and does.

    Regresses err = modelled - observed on the group's own OBSERVED rate, PA-weighted.

    The naive slope of that regression is biased downward by construction and the size of
    the bias is not small. `observed` is a mean over the group's own PA, so it carries
    sampling noise, and that same noise sits inside `err` with a minus sign. Writing
    o = t + e and m = a + b*t,

        slope_naive = [(b - 1) var(t) - var(e)] / [var(t) + var(e)]

    which is negative whenever var(e) > 0 even at b = 1, i.e. even when the model has no
    compression at all. Reporting the naive slope alone would manufacture the finding.

    var(e) is estimated per group from the binomial form p(1-p)/n and averaged under the
    same weights; var(t) is var(o) - var(e). The corrected compression coefficient is

        b = 1 + [slope_naive * var(o) + var(e)] / var(t)

    and b is the number the spec's decision rests on. Both are returned.
    """
    trained = trained_batters(manifest)
    observed = per_group_absorbing(pa_df, [eval_season])
    frame = predictions.merge(observed, on=["batter", "p_throws"], how="inner")
    frame = frame[frame["batter"].isin(trained)].copy()
    assert len(frame), "no trained hitter survived the observed-rate join"

    frame["err"] = frame[f"rate_{key}"] - frame[key]
    weight = frame["denominator"].to_numpy(dtype="float64")
    o = frame[key].to_numpy(dtype="float64")

    intercept, slope, se = weighted_ols(o, frame["err"].to_numpy(), weight)

    var_o = float(np.average((o - np.average(o, weights=weight)) ** 2, weights=weight))
    var_e = float(np.average(o * (1.0 - o) / np.maximum(weight, 1.0), weights=weight))
    var_t = var_o - var_e
    corrected = 1.0 + (slope * var_o + var_e) / var_t if var_t > 0 else float("nan")
    # the slope a model with NO compression would still show, given this much target noise
    null_slope = -var_e / var_o if var_o > 0 else float("nan")

    return {
        "state": key,
        "n_groups": int(len(frame)),
        "n_pa": float(weight.sum()),
        "mean_err": float(np.average(frame["err"], weights=weight)),
        "slope_naive": slope,
        "slope_se": se,
        "slope_z": slope / se if se > 0 else float("nan"),
        "intercept": intercept,
        "null_slope_from_target_noise": null_slope,
        "slope_excess_over_null": slope - null_slope,
        "var_observed": var_o,
        "var_target_noise": var_e,
        "var_true": var_t,
        "compression_b": corrected,
        "compression_shrinks_spread": bool(corrected < 1.0) if var_t > 0 else None,
    }


# ---------------------------------------------------------------- E.3

def level_closure(predictions, pa_df, manifest, eval_season):
    """
    E.3 (spec §5). Does the absorbing-rate error account for the composition's wOBA level
    bias? The chain is solved exactly, so the answer is exact arithmetic and not a fit.

    A plate appearance's wOBA is  sum_s rate_s * value_s  over the four absorbing states.
    Walks are paid wBB, hit-by-pitch wHBP, strikeouts zero, and balls in play whatever the
    remaining points imply, which is how `value_bip` is recovered on each side rather than
    assumed. The gap then splits Oaxaca-style,

        pred - obs = sum_s (rate_m - rate_o) * value_o   <- the RATE effect
                   + sum_s  rate_m * (value_m - value_o) <- the VALUE effect

    with the rate effect being everything the four absorbing rates can express and the
    value effect being everything they cannot, which is `V`, the contact-quality table.
    Reported over the matched population (E.1), so the level gap and the rate errors are
    measured on the same hitters.
    """
    weights = eval_targets.load_weights()[str(eval_season)]
    trained = trained_batters(manifest)

    actual = eval_targets.aggregate(eval_targets.drop_pitcher_batters(pa_df),
                                    by=tuple(claim1_eval.KEY))
    actual = actual[actual["season"] == eval_season]
    observed = per_group_absorbing(pa_df, [eval_season])

    frame = (predictions.merge(actual[claim1_eval.KEY + ["woba", "denominator"]],
                               on=claim1_eval.KEY, how="inner")
             .merge(observed.drop(columns=["denominator"]), on=["batter", "p_throws"],
                    how="inner"))
    frame = frame[frame["batter"].isin(trained)]
    assert len(frame), "no trained hitter survived the level-closure join"

    weight = frame["denominator"].to_numpy(dtype="float64")
    avg = lambda column: float(np.average(np.asarray(column, dtype="float64"), weights=weight))

    rate_m = {key: avg(frame[f"rate_{key}"]) for key in query.ABSORBING_KEYS}
    rate_o = {key: avg(frame[key]) for key in query.ABSORBING_KEYS}
    woba_m, woba_o = avg(frame["pred_woba"]), avg(frame["woba"])

    fixed = {"bb": weights["wBB"], "hbp": weights["wHBP"], "k": 0.0}
    residual_m = woba_m - sum(fixed[key] * rate_m[key] for key in fixed)
    residual_o = woba_o - sum(fixed[key] * rate_o[key] for key in fixed)
    value_m = {**fixed, "bip": residual_m / rate_m["bip"]}
    value_o = {**fixed, "bip": residual_o / rate_o["bip"]}

    rate_effect = {key: (rate_m[key] - rate_o[key]) * value_o[key]
                   for key in query.ABSORBING_KEYS}
    value_effect = {key: rate_m[key] * (value_m[key] - value_o[key])
                    for key in query.ABSORBING_KEYS}
    total = woba_m - woba_o
    closed = sum(rate_effect.values()) + sum(value_effect.values())
    assert abs(closed - total) < 1e-9, \
        f"the decomposition does not close: {closed} against {total}"

    return {
        "n_groups": int(len(frame)), "n_pa": float(weight.sum()),
        "pred_woba": woba_m, "observed_woba": woba_o, "level_bias": total,
        "rates_modelled": rate_m, "rates_observed": rate_o,
        "value_modelled": value_m, "value_observed": value_o,
        "rate_effect": rate_effect, "value_effect": value_effect,
        "rate_effect_total": sum(rate_effect.values()),
        "value_effect_total": sum(value_effect.values()),
        "share_explained_by_rates": sum(rate_effect.values()) / total if total else float("nan"),
    }


# ---------------------------------------------------------------- E.4

def calibration(eval_frame, n_bins=10):
    """
    E.4 (spec §6). UNSCORED readout: PA-weighted regression of realized wOBA on pred_woba,
    overall and per exposure stratum, plus a decile reliability table.

    Unscored is not a hedge. The scorer IS the model here, so a knob tuned to move this
    number would be tuned on the metric it is meant to validate (phase-d5-spec.md §9).
    A slope below 1 means the predicted spread is too narrow for the realized spread; it is
    reported, and nothing is adjusted to fix it.
    """
    rows, reliability = [], []
    for name in list(claim1_eval.STRATUM_NAMES) + ["all"]:
        part = eval_frame if name == "all" else eval_frame[eval_frame["stratum"] == name]
        if len(part) < 3:
            rows.append({"stratum": name, "n_hitters": int(len(part)),
                         "intercept": float("nan"), "slope": float("nan"),
                         "slope_se": float("nan")})
            continue
        weight = part["denominator"].to_numpy(dtype="float64")
        intercept, slope, se = weighted_ols(part["pred_woba"], part["woba"], weight)
        rows.append({"stratum": name, "n_hitters": int(len(part)),
                     "pa": float(weight.sum()),
                     "mean_pred": float(np.average(part["pred_woba"], weights=weight)),
                     "mean_obs": float(np.average(part["woba"], weights=weight)),
                     "intercept": intercept, "slope": slope, "slope_se": se,
                     "slope_z_vs_one": (slope - 1.0) / se if se > 0 else float("nan")})

        if name == "all":
            # weighted quantile bins: equal PA per bin, not equal hitter count, so a bin's
            # mean is estimated on comparable precision across the range
            order = np.argsort(part["pred_woba"].to_numpy())
            ordered = part.iloc[order]
            cumulative = np.cumsum(ordered["denominator"].to_numpy(dtype="float64"))
            edges = cumulative[-1] * np.arange(1, n_bins) / n_bins
            groups = np.searchsorted(edges, cumulative)
            for index in range(n_bins):
                bucket = ordered[groups == index]
                if not len(bucket):
                    continue
                bucket_weight = bucket["denominator"].to_numpy(dtype="float64")
                reliability.append({
                    "bin": index, "n_hitters": int(len(bucket)),
                    "pa": float(bucket_weight.sum()),
                    "mean_pred": float(np.average(bucket["pred_woba"], weights=bucket_weight)),
                    "mean_obs": float(np.average(bucket["woba"], weights=bucket_weight)),
                    "gap": float(np.average(bucket["pred_woba"], weights=bucket_weight)
                                 - np.average(bucket["woba"], weights=bucket_weight))})
    return pd.DataFrame(rows), pd.DataFrame(reliability)


# ---------------------------------------------------------------- E.5

def platoon_frame(pa_df, predictions, eval_season, min_eval_pa=claim1_eval.MIN_EVAL_PA):
    """
    One row per hitter who is scorable against BOTH hands: the model's platoon differential
    and the realized one, with the Route A reference attached.

    Route A is the explicit no-platoon-knowledge model: every hitter is given the
    league-average differential for his stand, fit on TRAIN seasons only. It is the opponent
    because side-specific wOBA can be scored well without any hitter-specific platoon
    knowledge at all — learn overall quality, apply the league split — and claim-1 cannot
    separate that from the real thing.

    Weight is the harmonic mean of the two sides' denominators, doubled so it stays on a
    plate-appearance scale: a difference is only as precise as its scarcer side, and the
    arithmetic mean would let a 600/10 split masquerade as a well-measured hitter.
    Stratum is likewise assigned on the MINIMUM of the two sides' prior exposure (spec §7).
    """
    pa_df = eval_targets.drop_pitcher_batters(pa_df)
    actual = eval_targets.aggregate(pa_df, by=tuple(claim1_eval.KEY))
    actual = actual[actual["season"] == eval_season]
    actual = actual[actual["denominator"] >= min_eval_pa]
    prior = claim1_eval.prior_exposure(pa_df, eval_season)

    frame = (actual.merge(predictions[claim1_eval.KEY + ["pred_woba"]],
                          on=claim1_eval.KEY, how="inner")
             .merge(prior, on=["batter", "p_throws"], how="left"))
    frame["prior_pa"] = frame["prior_pa"].fillna(0.0)

    wide = frame.pivot(index="batter", columns="p_throws",
                       values=["woba", "pred_woba", "denominator", "prior_pa"])
    wide = wide.dropna()
    assert len(wide), "no hitter is scorable against both hands"

    stand = (pa_df[pa_df["season"] == eval_season].groupby("batter")["stand"]
             .agg(lambda values: values.mode().iat[0]))

    out = pd.DataFrame({
        "batter": wide.index,
        "delta_obs": wide[("woba", "L")] - wide[("woba", "R")],
        "delta_pred": wide[("pred_woba", "L")] - wide[("pred_woba", "R")],
        "denom_L": wide[("denominator", "L")], "denom_R": wide[("denominator", "R")],
        "prior_pa": wide[("prior_pa", "L")].combine(wide[("prior_pa", "R")], min),
    }).reset_index(drop=True)
    out["stand"] = out["batter"].map(stand)
    assert out["stand"].notna().all(), "a scored hitter has no stand"
    out["weight"] = 2.0 / (1.0 / out["denom_L"] + 1.0 / out["denom_R"])
    out["stratum"] = claim1_eval.assign_stratum(out["prior_pa"])

    # Route A: the league differential per stand, fit on TRAIN seasons only. Nothing
    # hitter-specific enters, which is the point -- it is the zero-platoon-knowledge model.
    train = pa_df[pa_df["season"] < eval_season]
    league = eval_targets.aggregate(train, by=("stand", "p_throws"))
    league = league.set_index(["stand", "p_throws"])["woba"]
    route_a = {side: float(league.loc[(side, "L")] - league.loc[(side, "R")])
               for side in out["stand"].unique() if (side, "L") in league.index}
    out["delta_route_a"] = out["stand"].map(route_a)
    assert out["delta_route_a"].notna().all(), "a stand has no Route A differential"
    return out, route_a


def score_platoon(frame, prediction_column):
    """PA-weighted RMSE and weighted rank correlation of a differential, per stratum."""
    rows = []
    for name in list(claim1_eval.STRATUM_NAMES) + ["all"]:
        part = frame if name == "all" else frame[frame["stratum"] == name]
        if len(part) < 3:
            rows.append({"stratum": name, "model": prediction_column,
                         "n_hitters": int(len(part)), "rmse": float("nan"),
                         "rank_corr_weighted": float("nan")})
            continue
        weight = part["weight"].to_numpy(dtype="float64")
        rows.append({
            "stratum": name, "model": prediction_column, "n_hitters": int(len(part)),
            "weight": float(weight.sum()),
            "mean_obs": float(np.average(part["delta_obs"], weights=weight)),
            "mean_pred": float(np.average(part[prediction_column], weights=weight)),
            "sd_obs": float(np.sqrt(np.cov(part["delta_obs"], aweights=weight))),
            "sd_pred": float(np.sqrt(np.cov(part[prediction_column], aweights=weight)))
                       if part[prediction_column].nunique() > 1 else 0.0,
            "rmse": claim1_eval.pa_weighted_rmse(part["delta_obs"],
                                                 part[prediction_column], weight),
            "rank_corr_weighted": claim1_eval.weighted_rank_correlation(
                part["delta_obs"], part[prediction_column], weight),
        })
    return pd.DataFrame(rows)


def platoon_decomposition(frame):
    """
    How much of the model's predicted platoon differential is just the stand effect.

    DESCRIPTIVE, not a test. Route A is a two-valued model: one differential for left-handed
    hitters and one for right-handed. So any predictor that only knows a hitter's stand can
    reproduce it exactly, and a rank correlation against the realized differential will look
    respectable purely from the stand split. The question the claim actually needs is whether
    the model says anything ABOUT ONE HITTER that the stand does not already say.

    Splits the weighted variance of each differential into a between-stand part and a
    within-stand part. Route A's within-stand variance is zero by construction and is
    reported as the reference the model's number is read against.

    The within-stand rank correlation is also returned and is flagged POST-HOC: it was added
    after E.5's pre-registered table was read, so it is a description of where the model's
    signal sits and is not evidence for or against the pre-registered claim.
    """
    weight = frame["weight"].to_numpy(dtype="float64")
    out = {}
    for column in ("delta_obs", "delta_pred", "delta_route_a"):
        values = frame[column].to_numpy(dtype="float64")
        grand = np.average(values, weights=weight)
        between = 0.0
        residual = values.copy()
        for side in frame["stand"].unique():
            mask = (frame["stand"] == side).to_numpy()
            cell = np.average(values[mask], weights=weight[mask])
            between += weight[mask].sum() / weight.sum() * (cell - grand) ** 2
            residual[mask] = values[mask] - cell
        total = float(np.average((values - grand) ** 2, weights=weight))
        out[column] = {"variance": total, "between_stand": float(between),
                       "within_stand": float(total - between),
                       "share_between_stand": float(between / total) if total else float("nan")}
        frame = frame.assign(**{f"resid_{column}": residual})

    out["post_hoc_within_stand_rank_corr"] = float(claim1_eval.weighted_rank_correlation(
        frame["resid_delta_obs"], frame["resid_delta_pred"], weight))
    out["post_hoc_note"] = ("added after the pre-registered E.5 table was read; descriptive, "
                            "not evidence for the adoption rule")
    return out


def paired_platoon_difference(frame, n_boot=2000, seed=0, ci=(2.5, 97.5)):
    """
    Model minus Route A on both metrics, per stratum, by bootstrap over HITTERS.

    One row per hitter here, so the batter clustering claim-1 needs is automatic: resampling
    rows and resampling batters are the same operation on this frame. Negative RMSE
    difference and positive rank difference both favour the model.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for name in list(claim1_eval.STRATUM_NAMES) + ["all"]:
        part = frame if name == "all" else frame[frame["stratum"] == name]
        if len(part) < 3:
            rows.append({"stratum": name, "n_hitters": int(len(part))})
            continue
        weight = part["weight"].to_numpy(dtype="float64")
        obs = part["delta_obs"].to_numpy()
        model, route = part["delta_pred"].to_numpy(), part["delta_route_a"].to_numpy()

        def metrics(index):
            w, a = weight[index], obs[index]
            return (claim1_eval.pa_weighted_rmse(a, model[index], w)
                    - claim1_eval.pa_weighted_rmse(a, route[index], w),
                    claim1_eval.weighted_rank_correlation(a, model[index], w)
                    - claim1_eval.weighted_rank_correlation(a, route[index], w))

        base_rmse, base_rank = metrics(np.arange(len(part)))
        draws = np.array([metrics(rng.integers(0, len(part), len(part)))
                          for _ in range(n_boot)])
        low_r, high_r = np.percentile(draws[:, 0], ci)
        low_k, high_k = np.percentile(draws[:, 1], ci)
        rows.append({
            "stratum": name, "n_hitters": int(len(part)), "n_boot": n_boot,
            "rmse_difference": base_rmse, "rmse_ci_low": float(low_r),
            "rmse_ci_high": float(high_r),
            "rmse_favours_model": bool(high_r < 0),
            "rank_difference": base_rank, "rank_ci_low": float(low_k),
            "rank_ci_high": float(high_k),
            "rank_favours_model": bool(low_k > 0),
            "rank_se": float(draws[:, 1].std(ddof=1)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- driver

def main():
    parser = argparse.ArgumentParser(description="Phase E — validation and evaluation.")
    parser.add_argument("--arm", default=DEFAULT_ARM)
    parser.add_argument("--eval-season", type=int, default=2024)
    parser.add_argument("--final-run", action="store_true")
    parser.add_argument("--predictions", default=None,
                        help="defaults to results/phase_d/d5_predictions_<arm>.csv")
    parser.add_argument("--data-dir", default="data/processed/phase_d")
    parser.add_argument("--eval-targets", default="data/processed/eval_targets_pa.parquet")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    claim1_eval.assert_not_test_season(args.eval_season, final_run=args.final_run)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = args.predictions or f"results/phase_d/d5_predictions_{args.arm}.csv"
    predictions = pd.read_csv(path)
    pa_df = pd.read_parquet(args.eval_targets)
    # the manifest alone -- nothing here needs the 7.3M-row tensors, and loading them
    # costs ~1.3 GB on a machine that has 8
    manifest = json.loads((Path(args.data_dir) / "manifest.json").read_text())

    report = {"arm": args.arm, "eval_season": args.eval_season, "predictions": path}

    print("E.1 — matched-population fidelity control")
    table, verdict = matched_population(predictions, pa_df, manifest, args.eval_season)
    table.to_csv(out_dir / "e1_matched_population.csv", index=False)
    report["e1_matched_population"] = verdict

    print("E.2 — contamination test")
    report["e2_contamination"] = {key: contamination(predictions, pa_df, manifest,
                                                     args.eval_season, key=key)
                                  for key in query.ABSORBING_KEYS}

    print("E.3 — level-bias closure")
    report["e3_level_closure"] = level_closure(predictions, pa_df, manifest,
                                               args.eval_season)

    print("E.4 — calibration")
    eval_frame, coverage = claim1_eval.build_eval_frame(pa_df, predictions, args.eval_season)
    calib, reliability = calibration(eval_frame)
    calib.to_csv(out_dir / "e4_calibration.csv", index=False)
    reliability.to_csv(out_dir / "e4_reliability.csv", index=False)
    report["e4_calibration"] = {"coverage": coverage,
                                "by_stratum": calib.to_dict(orient="records")}

    print("E.5 — platoon differential against Route A")
    frame, route_a = platoon_frame(pa_df, predictions, args.eval_season)
    frame.to_csv(out_dir / "e5_platoon_frame.csv", index=False)
    scores = pd.concat([score_platoon(frame, "delta_pred"),
                        score_platoon(frame, "delta_route_a")], ignore_index=True)
    scores.to_csv(out_dir / "e5_platoon_scores.csv", index=False)
    paired = paired_platoon_difference(frame, n_boot=args.n_boot, seed=args.seed)
    paired.to_csv(out_dir / "e5_platoon_paired.csv", index=False)
    report["e5_platoon"] = {"route_a_by_stand": route_a,
                            "decomposition": platoon_decomposition(frame),
                            "n_hitters": int(len(frame)),
                            "scores": scores.to_dict(orient="records"),
                            "paired": paired.to_dict(orient="records")}

    (out_dir / "e_report.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"wrote {out_dir / 'e_report.json'}")


if __name__ == "__main__":
    main()
