"""
Case studies graded against 2025: for the five Phase V anchors and the hitters on
whom the model and eb_bivariate disagreed most in 2024 (V.10), compare each
estimator's 2025 platoon differential (predicted L-R wOBA) with the observed 2025
split.

Separate from V.10 on purpose: V.10 stays descriptive ("on whom", pre-registered);
this module is the follow-up that asks who was closer. It reads 2025 outcomes only
through `results/model_evaluation_final_clean/platoon_frame.csv` (the claim-1 eval
already unsealed 2025 for these hitters) and recomputes EB at eval_season=2025 from
the trailing window, which ends at 2024. Phase M is not touched.

Per-hitter grades are noisy: a 2025 split over ~150 PA a side has an SE near 0.05
wOBA, the same size as most model-EB gaps. `se_obs` is reported for every row so
each grade can be read against it.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from src.analysis.model_visualization_disagreement import eb_delta_frame
from src.analysis.model_visualization_stats import ANCHORS

EVAL_SEASON = 2025
TOP_N = 20
Z_NOISE = 1.96


def grade(frame, woba_sd):
    """
    frame: batter, delta_obs, delta_pred, delta_eb, denom_L, denom_R.
    Adds abs errors, who was closer, sign checks and the SE of the observed split.
    """
    # delta_* are predicted (or observed) L-R wOBA differentials; error = absolute miss vs the observed split
    out = frame.copy()
    out["err_model"] = (out["delta_pred"] - out["delta_obs"]).abs()
    out["err_eb"] = (out["delta_eb"] - out["delta_obs"]).abs()
    out["closer"] = np.where(out["err_model"] < out["err_eb"], "model",
                             np.where(out["err_eb"] < out["err_model"], "eb", "tie"))
    # sign check: did the estimator get the direction of the platoon split right
    out["model_sign_right"] = np.sign(out["delta_pred"]) == np.sign(out["delta_obs"])
    out["eb_sign_right"] = np.sign(out["delta_eb"]) == np.sign(out["delta_obs"])
    # SE of a difference of two independent side means: sd * sqrt(1/n_L + 1/n_R)
    out["se_obs"] = woba_sd * np.sqrt(1 / out["denom_L"] + 1 / out["denom_R"])
    # who is closer depends only on which side of the two predictions' midpoint the
    # observed split lands; within ~2 SE of the midpoint, noise alone could flip it
    midpoint = (out["delta_pred"] + out["delta_eb"]) / 2
    out["z_midpoint"] = (out["delta_obs"] - midpoint).abs() / out["se_obs"]
    out["within_noise"] = out["z_midpoint"] < Z_NOISE
    return out


def tally(graded):
    """Counts of who was closer, plus PA-weighted mean absolute error per estimator."""
    # weight: per-hitter weight column carried in from the platoon frame; must be positive
    w = graded["weight"]
    assert np.isfinite(w).all() and (w > 0).all(), "weights must be finite and positive"
    return {"n": int(len(graded)),
            "model_closer": int((graded["closer"] == "model").sum()),
            "eb_closer": int((graded["closer"] == "eb").sum()),
            "within_noise": int(graded["within_noise"].sum()),
            "model_sign_right": int(graded["model_sign_right"].sum()),
            "eb_sign_right": int(graded["eb_sign_right"].sum()),
            "wmae_model": float(np.average(graded["err_model"], weights=w)),
            "wmae_eb": float(np.average(graded["err_eb"], weights=w))}


def bootstrap(graded, draws=2000, seed=0):
    """
    Hitter-level bootstrap of the two headline numbers: the PA-weighted MAE difference
    (model minus EB, negative favours the model) and the share of hitters the model was
    closer on. Hitters are resampled with replacement; intervals are 2.5/97.5 percentiles.
    """
    rng = np.random.default_rng(seed)
    w = graded["weight"].to_numpy()
    diff = (graded["err_model"] - graded["err_eb"]).to_numpy()
    closer = (graded["closer"] == "model").to_numpy()
    idx = rng.integers(0, len(graded), size=(draws, len(graded)))
    wmae = (w[idx] * diff[idx]).sum(axis=1) / w[idx].sum(axis=1)
    share = closer[idx].mean(axis=1)
    return {"n": int(len(graded)),
            "wmae_diff": float(np.average(diff, weights=w)),
            "ci": [float(x) for x in np.percentile(wmae, [2.5, 97.5])],
            "share_model_closer": float(closer.mean()),
            "share_ci": [float(x) for x in np.percentile(share, [2.5, 97.5])],
            "draws": draws, "seed": seed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platoon-frame", default="results/model_evaluation_final_clean/platoon_frame.csv")
    parser.add_argument("--disagreement", default="results/v_chain_clean/disagreement.csv")
    parser.add_argument("--names", default="data/processed/hitter_names_clean.csv")
    parser.add_argument("--out-dir", default="results/case_studies_2025")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # observed 2025 splits and the model's 2025 predictions, from the claim-1 eval output
    platoon = pd.read_csv(args.platoon_frame)
    pa_df = pd.read_parquet("data/processed/eval_targets_pa.parquet")
    eb = eb_delta_frame(pa_df, eval_season=EVAL_SEASON)
    # per-PA wOBA standard deviation in 2025, the noise scale used for each hitter's se_obs
    season = pa_df[(pa_df["season"] == EVAL_SEASON) & pa_df["in_denominator"]]
    woba_sd = float(season["woba_points"].std(ddof=1))

    # one row per hitter: observed + model + EB differentials, with names; then grade
    names = pd.read_csv(args.names)[["batter", "name"]]
    frame = platoon.merge(eb, on="batter", how="inner").merge(names, on="batter", how="left")
    graded = grade(frame, woba_sd)

    # the TOP_N hitters with the largest absolute model-vs-EB rank gap in 2024 (V.10)
    disagreement = pd.read_csv(args.disagreement)
    top = disagreement.assign(abs_gap=disagreement["rank_gap"].abs()).nlargest(TOP_N, "abs_gap")
    top = top[["batter", "rank_gap"]].rename(columns={"rank_gap": "rank_gap_2024"})

    # the five Phase V anchor hitters, and any that did not play in the 2025 frame
    anchors = graded[graded["batter"].isin(ANCHORS)].assign(group="anchor")
    missing = sorted(set(ANCHORS) - set(anchors["batter"]))
    disagree = graded.merge(top, on="batter").assign(group="disagreement_top")
    # selected on playing in 2025: name the 2024 hitters that drop out
    absent = top[~top["batter"].isin(disagree["batter"])].merge(
        disagreement[["batter", "name"]], on="batter")

    # output column order for the per-hitter CSV
    cols = ["group", "batter", "name", "stand", "stratum", "prior_pa", "denom_L", "denom_R",
            "weight", "delta_obs", "se_obs", "delta_pred", "delta_eb", "err_model",
            "err_eb", "closer", "z_midpoint", "within_noise", "model_sign_right", "eb_sign_right", "rank_gap_2024"]
    table = pd.concat([anchors, disagree], ignore_index=True).reindex(columns=cols)
    table.to_csv(f"{args.out_dir}/case_studies_2025.csv", index=False)

    # tallies per group plus the hitters who dropped out of 2025
    summary = {"woba_sd_per_pa": woba_sd,
               "anchors_missing_from_2025_frame": [ANCHORS[b] for b in missing],
               "anchors": tally(anchors) if len(anchors) else None,
               "disagreement_top": tally(disagree) if len(disagree) else None,
               "all_hitters": tally(graded),
               "disagreement_top_found_in_2025": int(len(disagree)),
               "disagreement_top_absent_in_2025": [
                   {"name": r.name, "rank_gap_2024": int(r.rank_gap_2024)} for r in absent.itertuples()],
               "note": "anchor sign counts here are 2025 forecasts vs the 2025 split; the "
                       "4/5 in test_paper_figures is V.9 surfaces vs the training-window history"}
    with open(f"{args.out_dir}/case_studies_2025.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(f"{args.out_dir}/all_hitters_bootstrap.json", "w") as f:
        json.dump(bootstrap(graded), f, indent=1)

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(table.round(3).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
