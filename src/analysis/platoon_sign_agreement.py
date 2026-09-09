"""
Within-stand sign agreement for the platoon delta prediction.

The platoon frame carries one predicted platoon delta per hitter
(`delta_pred`) against the observed value (`delta_obs`). The naive check --
does `sign(delta_pred) == sign(delta_obs)` -- looks like a measure of
whether the model has learned something about a specific hitter. Mostly it
is not. Left-handed hitters have negative platoon deltas on average and
right-handed hitters have positive ones, so any predictor that has merely
learned "which hand does this guy bat with" already gets most of the raw
sign agreement for free, without knowing anything about the hitter beyond
his listed stand.

To separate "the model knows the hitter" from "the model knows his
handedness," this module centers both the observed and predicted deltas on
their own stand-group mean before comparing signs. A hitter's centered
value asks a narrower question: is he more or less extreme than a typical
hitter of his own stand? Sign agreement on the centered values can only be
explained by within-stand information -- it is unaffected by whichever
direction that stand skews as a whole.

The comparison point for the within-stand rate is not a flat 50%: the
centered predictions are themselves skewed within a stand (e.g. in the low
stratum, stand L, only 28.3% of `centered_pred` are positive against 45.7%
of `centered_obs`), and that marginal skew alone shifts the agreement rate
an independent predictor would produce away from 50%. Each cell below is
therefore also checked against an analytic independence rate and a
within-stand permutation null, not just against chance. The low stratum
additionally has many rows sharing one embedding row (cold-start hitters),
so its `delta_pred` is tied across hitters and carries no hitter-specific
platoon signal for those rows -- its rate is partly driven by a constant
predictor rather than genuine per-hitter skill.

The low stratum's clearance is not a constant-predictor artifact: splitting it
into `low_untied` and `low_tied` (rows whose `delta_pred` is unique vs.
duplicated across the frame) shows the untied rows score higher than the tied
ones, so the tied, no-signal rows are not the ones carrying the effect.

Two baselines quantify how much of the raw number was handedness:

- The handedness-only baseline scores a predictor that has learned nothing
  but the sign of the stand-group effect: for every hitter it asks whether
  his raw deviation from the overall mean carries the same sign as his
  stand-group's deviation from the overall mean. It has no within-stand
  information at all, so its within-stand agreement is undefined (its
  centered prediction is identically zero for everyone); what is reported
  here is its raw agreement, which is exactly the ceiling a "know the hand,
  nothing else" predictor could reach.
- `delta_route_a`, inspected below, turns out to be a second baseline of
  exactly that handedness-only shape: it takes only two values in the whole
  545-row frame, one per stand, with zero within-stand variation. It is not
  a per-hitter predicted delta, so the within-stand statistic is not run on
  it (every centered value would be zero, which duplicates the
  handedness-only baseline rather than adding information). This is
  recorded in the output rather than silently skipped.

Run: python -m src.analysis.platoon_sign_agreement
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_OUT_DIR = "results/paper_figures"
DEFAULT_FRAME_PATH = "results/model_evaluation/platoon_frame.csv"
STRATA = ("low", "medium", "high")
WILSON_Z_95 = 1.959963984540054


def load_platoon_frame(frame_path):
    """Read the platoon frame and confirm the columns this module depends on."""
    frame = pd.read_csv(frame_path)
    required_columns = {"batter", "delta_obs", "delta_pred", "stand", "stratum",
                        "delta_route_a"}
    missing_columns = required_columns - set(frame.columns)
    assert not missing_columns, f"platoon frame is missing columns: {missing_columns}"
    return frame


def route_a_is_a_predicted_delta(frame):
    """
    True only if `delta_route_a` carries per-hitter, within-stand variation.

    In the 2024 arm frame it does not: it takes exactly one value per stand
    (a stand-group constant), so it is a restatement of the handedness-only
    baseline rather than an independent per-hitter prediction.
    """
    within_stand_unique_counts = frame.groupby("stand")["delta_route_a"].nunique()
    return bool((within_stand_unique_counts > 1).any())


def wilson_interval(successes, n, z=WILSON_Z_95):
    """Wilson score 95% confidence interval on a binomial rate."""
    if n == 0:
        return (float("nan"), float("nan"))
    proportion = successes / n
    denominator = 1 + z ** 2 / n
    center = proportion + z ** 2 / (2 * n)
    adjustment = z * np.sqrt(proportion * (1 - proportion) / n + z ** 2 / (4 * n ** 2))
    return ((center - adjustment) / denominator, (center + adjustment) / denominator)


def rate_summary(agreement_flags):
    """Rate, count, and Wilson CI for a boolean array of hitter-level agreements."""
    agreement_flags = np.asarray(agreement_flags, dtype=bool)
    n = int(len(agreement_flags))
    successes = int(agreement_flags.sum())
    rate = successes / n if n else float("nan")
    low, high = wilson_interval(successes, n)
    return {"rate": rate, "n": n, "successes": successes, "ci95": [low, high]}


def add_centered_columns(frame):
    """
    Add `centered_obs` and `centered_pred`: each hitter's delta minus his own
    stand-group mean of that same column. This is the transform that removes
    the handedness effect before signs are compared.
    """
    frame = frame.copy()
    frame["centered_obs"] = frame["delta_obs"] - frame.groupby("stand")["delta_obs"].transform("mean")
    frame["centered_pred"] = frame["delta_pred"] - frame.groupby("stand")["delta_pred"].transform("mean")
    frame["agree"] = np.sign(frame["centered_obs"]) == np.sign(frame["centered_pred"])
    return frame


def handedness_only_agreement(frame):
    """
    Raw sign agreement of a predictor that only knows the hitter's stand: does
    his deviation from the overall mean match the sign of his stand-group's
    deviation from the overall mean.
    """
    overall_mean_obs = frame["delta_obs"].mean()
    stand_group_mean_obs = frame.groupby("stand")["delta_obs"].transform("mean")
    hitter_sign = np.sign(frame["delta_obs"] - overall_mean_obs)
    stand_sign = np.sign(stand_group_mean_obs - overall_mean_obs)
    return hitter_sign == stand_sign


def raw_model_agreement(frame):
    """Raw (uncentered) sign agreement of the model: sign(delta_obs) == sign(delta_pred)."""
    return np.sign(frame["delta_obs"]) == np.sign(frame["delta_pred"])


def independence_rate(centered_pred, centered_obs):
    """
    Chance agreement rate implied by each side's own marginal skew, under
    independence: P(both positive) + P(both negative). This is the correct
    null for within-stand sign agreement when centered_pred and/or
    centered_obs are not split 50/50 within a stand -- a flat 0.5 line
    overstates or understates the null depending on which way the skew runs.
    """
    centered_pred = np.asarray(centered_pred)
    centered_obs = np.asarray(centered_obs)
    if len(centered_pred) == 0:
        return float("nan")
    p_pred_pos = float((centered_pred > 0).mean())
    p_obs_pos = float((centered_obs > 0).mean())
    return p_pred_pos * p_obs_pos + (1 - p_pred_pos) * (1 - p_obs_pos)


def rate_and_independence(sub_frame):
    """rate_summary plus independence_rate for one cell of the centered frame."""
    cell = rate_summary(sub_frame["agree"])
    cell["independence_rate"] = independence_rate(sub_frame["centered_pred"], sub_frame["centered_obs"])
    return cell


def permutation_null_rates(frame, n_reps=2000, seed=7, extra_masks=None):
    """
    Shuffle centered_pred within each stand group (preserving stand
    structure) across the FULL frame, recompute the within-stand agreement
    rate, repeat n_reps times. Returns (overall_draws, {stratum: draws},
    {mask_name: draws}), each an array of n_reps agreement rates under the
    within-stand-shuffle null. `extra_masks` (name -> boolean array over the
    full frame) lets a caller score arbitrary row subsets -- e.g. tied vs
    untied rows within a stratum -- from the SAME full-frame shuffle, rather
    than shuffling within the subset alone.
    """
    rng = np.random.default_rng(seed)
    stand = frame["stand"].to_numpy()
    stratum = frame["stratum"].to_numpy()
    obs_sign = np.sign(frame["centered_obs"].to_numpy())
    pred = frame["centered_pred"].to_numpy()
    group_idx = {s: np.where(stand == s)[0] for s in np.unique(stand)}
    extra_masks = extra_masks or {}

    overall_draws = np.empty(n_reps)
    stratum_draws = {s: np.empty(n_reps) for s in STRATA}
    extra_draws = {name: np.empty(n_reps) for name in extra_masks}
    permuted = pred.copy()
    for rep in range(n_reps):
        for idx in group_idx.values():
            permuted[idx] = rng.permutation(pred[idx])
        agree = np.sign(permuted) == obs_sign
        overall_draws[rep] = agree.mean()
        for stratum_name in STRATA:
            stratum_draws[stratum_name][rep] = agree[stratum == stratum_name].mean()
        for name, mask in extra_masks.items():
            extra_draws[name][rep] = agree[mask].mean()
    return overall_draws, stratum_draws, extra_draws


def null_summary(draws, observed_rate):
    """null_mean, null_ci95, one-sided permutation p-value, excess_over_null."""
    draws = np.asarray(draws)
    n = len(draws)
    null_mean = float(draws.mean())
    ci95 = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
    p_value = float((np.sum(draws >= observed_rate) + 1) / (n + 1))
    return {
        "null_mean": null_mean,
        "null_ci95": ci95,
        "p_value": p_value,
        "excess_over_null": observed_rate - null_mean,
    }


def summarize_by_stratum(agreement_flags, stratum):
    """{"overall": rate_summary, "low": ..., "medium": ..., "high": ...}."""
    result = {"overall": rate_summary(agreement_flags)}
    for stratum_name in STRATA:
        mask = (stratum == stratum_name).to_numpy()
        result[stratum_name] = rate_summary(np.asarray(agreement_flags)[mask])
    return result


def compute_sign_agreement(frame):
    """
    Run the full statistic on a loaded platoon frame.

    Returns (per_hitter frame, summary dict). `per_hitter` is written to CSV
    as-is; `summary` is written to JSON as-is.
    """
    frame = add_centered_columns(frame)

    model_within_stand = {"overall": rate_and_independence(frame)}
    for stratum_name in STRATA:
        model_within_stand[stratum_name] = rate_and_independence(
            frame.loc[frame["stratum"] == stratum_name])
    model_within_stand["per_stand"] = {
        stand: rate_and_independence(frame.loc[frame["stand"] == stand])
        for stand in sorted(frame["stand"].unique())
    }
    model_within_stand["per_stand_stratum"] = {
        stand: {
            stratum_name: rate_and_independence(
                frame.loc[(frame["stand"] == stand) & (frame["stratum"] == stratum_name)])
            for stratum_name in STRATA
        }
        for stand in sorted(frame["stand"].unique())
    }

    # A row is "tied" when its delta_pred value occurs more than once in the
    # frame (cold-start hitters sharing one embedding row -- no
    # hitter-specific platoon signal). Tied/untied split of the low stratum
    # isolates whether its clearance is a constant-predictor artifact.
    tied_mask = frame["delta_pred"].duplicated(keep=False).to_numpy()
    low_mask = (frame["stratum"] == "low").to_numpy()
    extra_masks = {
        "low_untied": low_mask & ~tied_mask,
        "low_tied": low_mask & tied_mask,
        "overall_untied": ~tied_mask,
    }

    overall_draws, stratum_draws, extra_draws = permutation_null_rates(
        frame, extra_masks=extra_masks)
    model_within_stand["overall"]["permutation_null"] = null_summary(
        overall_draws, model_within_stand["overall"]["rate"])
    for stratum_name in STRATA:
        model_within_stand[stratum_name]["permutation_null"] = null_summary(
            stratum_draws[stratum_name], model_within_stand[stratum_name]["rate"])

    for name, mask in extra_masks.items():
        cell = rate_and_independence(frame.loc[mask])
        cell["permutation_null"] = null_summary(extra_draws[name], cell["rate"])
        model_within_stand[name] = cell

    n_tied_predictions = {
        stratum_name: int(
            frame.loc[frame["stratum"] == stratum_name, "delta_pred"]
            .duplicated(keep=False).sum())
        for stratum_name in STRATA
    }

    handedness_flags = handedness_only_agreement(frame)
    handedness_baseline = summarize_by_stratum(handedness_flags, frame["stratum"])

    raw_flags = raw_model_agreement(frame)
    raw_model = summarize_by_stratum(raw_flags, frame["stratum"])

    route_a_used = route_a_is_a_predicted_delta(frame)
    route_a_section = {"used": route_a_used}
    if route_a_used:
        route_a_frame = add_centered_columns(
            frame.assign(delta_pred=frame["delta_route_a"]))
        route_a_section["within_stand"] = summarize_by_stratum(
            route_a_frame["agree"], route_a_frame["stratum"])
    else:
        route_a_section["reason"] = (
            "delta_route_a takes exactly one value per stand (a stand-group "
            "constant, not a per-hitter prediction); running the within-stand "
            "statistic on it would produce centered_pred == 0 for every "
            "hitter, which is the handedness-only baseline again, not a new "
            "comparator.")

    summary = {
        "n_hitters": int(len(frame)),
        "model_within_stand": model_within_stand,
        "handedness_only_baseline": handedness_baseline,
        "raw_model_agreement": raw_model,
        "route_a_baseline": route_a_section,
        "n_tied_predictions": n_tied_predictions,
    }

    per_hitter = frame[["batter", "stand", "stratum", "centered_obs",
                        "centered_pred", "agree"]].copy()
    return per_hitter, summary


def fig_sign_agreement(summary, path):
    """
    Two panels, each comparing quantities measured on the SAME scale, because
    the model's raw rate and its within-stand rate have different denominators
    and a single axis holding both invites the reading that a handedness rule
    beats the model.

    Left, raw scale: the model's raw agreement against the handedness-only
    baseline -- the honest like-for-like comparison, which shows raw agreement
    is almost entirely batter stand. Right, within-stand scale: the same model
    with the handedness component removed, against its own permutation null,
    which is where its added signal is visible. A flat 50% line is not the
    comparator on either panel; within a stand the centered predictions are
    skewed, so independence alone does not sit at 0.5.
    """
    groups = ["overall"] + list(STRATA)

    def error_bars(rates, ci_bounds):
        lower = [max(0.0, rate - ci[0]) for rate, ci in zip(rates, ci_bounds)]
        upper = [max(0.0, ci[1] - rate) for rate, ci in zip(rates, ci_bounds)]
        return [lower, upper]

    def pull(block, field="rate"):
        return [summary[block][g][field] for g in groups]

    raw_rates = pull("raw_model_agreement")
    raw_ci = pull("raw_model_agreement", "ci95")
    base_rates = pull("handedness_only_baseline")
    base_ci = pull("handedness_only_baseline", "ci95")
    within_rates = pull("model_within_stand")
    within_ci = pull("model_within_stand", "ci95")
    null_means = [summary["model_within_stand"][g]["permutation_null"]["null_mean"]
                  for g in groups]
    null_ci = [summary["model_within_stand"][g]["permutation_null"]["null_ci95"]
               for g in groups]

    x = np.arange(len(groups))
    width = 0.36
    figure, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=True)

    left.bar(x - width / 2, raw_rates, width, yerr=error_bars(raw_rates, raw_ci),
             capsize=4, color="#4c8dff", label="model (raw)")
    left.bar(x + width / 2, base_rates, width, yerr=error_bars(base_rates, base_ci),
             capsize=4, color="#b0b0b0", label="handedness-only baseline (raw)")
    left.set_title("Raw scale: the model against a handedness rule\n"
                   "raw agreement is almost all batter stand")
    left.set_ylabel("sign agreement rate")

    right.bar(x - width / 2, within_rates, width,
              yerr=error_bars(within_rates, within_ci), capsize=4,
              color="#4c8dff", label="model (within-stand)")
    right.bar(x + width / 2, null_means, width,
              yerr=error_bars(null_means, null_ci), capsize=4,
              color="#e0a03c", label="permutation null (within-stand shuffle)")
    right.set_title("Within-stand scale: handedness removed\n"
                    "the model against its own permutation null")

    for axis in (left, right):
        axis.axhline(0.5, color="#444444", linestyle="--", linewidth=0.75,
                     alpha=0.4, label="50% (orientation only)")
        axis.set_xticks(x)
        axis.set_xticklabels(groups)
        axis.set_ylim(0, 1)
        axis.legend(loc="upper right", frameon=False, fontsize=8)

    figure.suptitle("Platoon delta sign agreement, compared only within a scale",
                    fontsize=12)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Within-stand sign agreement of the platoon delta model.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--frame", default=DEFAULT_FRAME_PATH)
    arguments = parser.parse_args()

    out_dir = Path(arguments.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = load_platoon_frame(arguments.frame)
    per_hitter, summary = compute_sign_agreement(frame)

    per_hitter.to_csv(out_dir / "platoon_sign_agreement.csv", index=False)
    (out_dir / "platoon_sign_agreement.json").write_text(json.dumps(summary, indent=2))
    fig_sign_agreement(summary, out_dir / "fig_platoon_sign_agreement.png")

    model_overall = summary["model_within_stand"]["overall"]
    baseline_overall = summary["handedness_only_baseline"]["overall"]
    raw_overall = summary["raw_model_agreement"]["overall"]
    print(f"n hitters: {summary['n_hitters']}")

    def print_cell(name, cell):
        null = cell["permutation_null"]
        print(f"  {name}: observed={cell['rate']:.4f} independence={cell['independence_rate']:.4f} "
              f"null_mean={null['null_mean']:.4f} null_ci95=[{null['null_ci95'][0]:.4f}, "
              f"{null['null_ci95'][1]:.4f}] excess={null['excess_over_null']:+.4f} "
              f"p={null['p_value']:.4f} (n={cell['n']})")

    print_cell("overall", model_overall)
    for stratum_name in STRATA:
        cell = summary["model_within_stand"][stratum_name]
        print_cell(stratum_name, cell)
        print(f"    n_tied_predictions: {summary['n_tied_predictions'][stratum_name]}")
    for name in ("low_untied", "low_tied", "overall_untied"):
        print_cell(name, summary["model_within_stand"][name])
    print(f"handedness-only baseline (raw) rate: {baseline_overall['rate']:.4f} "
          f"[{baseline_overall['ci95'][0]:.4f}, {baseline_overall['ci95'][1]:.4f}]")
    print(f"raw (uncentered) model agreement rate: {raw_overall['rate']:.4f} "
          f"[{raw_overall['ci95'][0]:.4f}, {raw_overall['ci95'][1]:.4f}]")
    print(f"delta_route_a used as comparator: {summary['route_a_baseline']['used']}")
    print(f"wrote {out_dir}/platoon_sign_agreement.csv, "
          f"platoon_sign_agreement.json, fig_platoon_sign_agreement.png")


if __name__ == "__main__":
    main()
