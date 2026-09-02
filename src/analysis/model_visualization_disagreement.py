"""
Figure V.10 — where the trained model and the C.2 empirical-Bayes baseline
(`eb_bivariate`) disagree about a hitter's platoon differential (predicted L-R wOBA).
Both are estimators of the same unobserved quantity; this module ranks hitters by
each estimator within their batting stand and reports the rank gap, per exposure
stratum and pooled.

Descriptive only, per phase-v-spec.md §0.5/§2 V.10: this is "on whom the two
estimators disagree," never "who is right." No accuracy, RMSE, or correlation
against an observed outcome is computed anywhere in this module, and `delta_obs`
from `platoon_frame.csv` is never read. 2025 is never read: the EB baseline is
recomputed at `eval_season=2024` only, and every frame this module builds itself
is asserted to carry no season past 2024.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.analysis import baseline_ladder_bivariate_eb

DEFAULT_OUT_DIR = "results/model_visualization"
EVAL_SEASON = 2024
TOP_N_LABEL = 10
TOP_N_TABLE = 20
MAX_JOIN_LOSS = 0.20


def eb_delta_frame(pa_df, eval_season=EVAL_SEASON):
    """
    The baseline's platoon differential: posterior E[wOBA|LHP] - E[wOBA|RHP] per
    batter, from C.2's own scorer.

    `model_evaluation_probe_coverage.eb_posterior_split` was written to wrap this
    exact call but references an undefined name (`eb_bivariate_eb` is never
    imported there — NameError on any call) and cannot be used as spec'd. Rather
    than edit that file, this reproduces its documented logic directly against
    `baseline_ladder_bivariate_eb.predict`, the module it names as its source.

    Returns one row per batter with `delta_eb`.
    """
    predictions = baseline_ladder_bivariate_eb.predict(pa_df, eval_season)
    assert set(predictions["p_throws"]) <= {"L", "R"}, "unexpected pitcher hand in EB output"
    assert (predictions["season"] <= 2024).all(), "EB predictions leaked a post-2024 season"
    wide = predictions.pivot(index="batter", columns="p_throws", values="pred_woba")
    # a hitter projected against only one hand has no split to compare
    wide = wide.dropna(subset=["L", "R"])
    out = pd.DataFrame({"batter": wide.index.astype(int).to_numpy(),
                        "delta_eb": (wide["L"] - wide["R"]).to_numpy()})
    assert np.isfinite(out["delta_eb"]).all(), "EB platoon differential is non-finite"
    return out.reset_index(drop=True)


def _rank_most_positive_first(series):
    # rank 1 = most favours LHP (largest L-R differential)
    return series.rank(ascending=False, method="min").astype(int)


def build_disagreement(platoon_frame, eb_frame, names):
    """
    Inner-join the model's and the baseline's platoon differentials on batter,
    attach names, and rank each estimator within stand (and pooled).

    platoon_frame: batter,delta_pred,stand,stratum,prior_pa (delta_obs ignored).
    eb_frame: batter,delta_eb (see `eb_delta_frame`).
    names: batter,name.
    Returns (disagreement_df, join_counts).
    """
    n_before = len(platoon_frame)
    merged = platoon_frame[["batter", "delta_pred", "stand", "stratum", "prior_pa"]].merge(
        eb_frame, on="batter", how="inner")
    n_after = len(merged)
    loss_frac = 1.0 - n_after / n_before
    print(f"join: {n_before} platoon_frame rows -> {n_after} matched to eb_bivariate "
          f"({n_before - n_after} dropped, {loss_frac:.1%})")
    assert loss_frac < MAX_JOIN_LOSS, f"join dropped {loss_frac:.1%} of platoon_frame rows"

    merged = merged.merge(names[["batter", "name"]], on="batter", how="left")

    merged["rank_model"] = merged.groupby("stand")["delta_pred"].transform(_rank_most_positive_first)
    merged["rank_eb"] = merged.groupby("stand")["delta_eb"].transform(_rank_most_positive_first)
    merged["rank_gap"] = merged["rank_model"] - merged["rank_eb"]
    merged["rank_model_pooled"] = _rank_most_positive_first(merged["delta_pred"])
    merged["rank_eb_pooled"] = _rank_most_positive_first(merged["delta_eb"])

    cols = ["batter", "name", "stand", "stratum", "prior_pa", "delta_pred", "delta_eb",
            "rank_model", "rank_eb", "rank_gap", "rank_model_pooled", "rank_eb_pooled"]
    return merged[cols].reset_index(drop=True), {"n_before": n_before, "n_after": n_after,
                                                  "n_dropped": n_before - n_after,
                                                  "loss_frac": loss_frac}


def summarize(disagreement):
    """Per stratum + pooled: n_hitters, Spearman(delta_pred, delta_eb), SD of rank_gap."""
    def one(df):
        rho, _ = stats.spearmanr(df["delta_pred"], df["delta_eb"])
        return {"n_hitters": int(len(df)), "spearman": float(rho),
                "rank_gap_sd": float(df["rank_gap"].std(ddof=1))}

    out = {stratum: one(group) for stratum, group in disagreement.groupby("stratum")}
    out["pooled"] = one(disagreement)
    return out


def top_disagreements(disagreement, n=TOP_N_TABLE):
    """Top-n hitters by |rank_gap|, pooled across strata, largest disagreement first."""
    ranked = disagreement.assign(abs_rank_gap=disagreement["rank_gap"].abs()).sort_values(
        "abs_rank_gap", ascending=False).head(n)
    cols = ["name", "stand", "stratum", "prior_pa", "delta_pred", "delta_eb",
            "rank_model", "rank_eb", "rank_gap"]
    return ranked[cols].reset_index(drop=True)


def fig_disagreement(disagreement, summary_by_stratum, path):
    """Four scatter panels (low, medium, high, pooled) of rank_eb vs rank_model with the
    top-10 |rank_gap| hitters labelled by name and a diagonal marking perfect agreement."""
    panels = [("low", "low exposure"), ("medium", "medium exposure"),
              ("high", "high exposure"), (None, "pooled")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, (stratum, label) in zip(axes.flat, panels):
        subset = disagreement if stratum is None else disagreement[disagreement["stratum"] == stratum]
        key = "pooled" if stratum is None else stratum
        stat = summary_by_stratum[key]

        ax.scatter(subset["rank_eb"], subset["rank_model"], s=14, alpha=0.5, color="#4c8dff")
        limit = max(subset["rank_eb"].max(), subset["rank_model"].max())
        ax.plot([1, limit], [1, limit], color="#999999", linestyle="--", linewidth=1)

        top = subset.assign(abs_rank_gap=subset["rank_gap"].abs()).nlargest(TOP_N_LABEL, "abs_rank_gap")
        for _, row in top.iterrows():
            ax.annotate(row["name"], (row["rank_eb"], row["rank_model"]), fontsize=7,
                       xytext=(3, 3), textcoords="offset points")

        ax.set_xlabel("rank, eb_bivariate (1 = favours LHP most)")
        ax.set_ylabel("rank, model")
        ax.set_title(f"{label}: n={stat['n_hitters']}, spearman={stat['spearman']:.3f}")
    fig.suptitle("V.10 disagreement: on whom the model and eb_bivariate rank the platoon differential differently")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    platoon_frame = pd.read_csv("results/model_evaluation/platoon_frame.csv")
    pa_df = pd.read_parquet("data/processed/eval_targets_pa.parquet")
    eb_frame = eb_delta_frame(pa_df, eval_season=EVAL_SEASON)

    names_path = "data/processed/hitter_names.csv"
    if os.path.exists(names_path):
        names = pd.read_csv(names_path)
    else:
        print(f"{names_path} not found yet; writing names as NaN")
        names = pd.DataFrame({"batter": platoon_frame["batter"].unique(), "name": np.nan})

    disagreement, join_counts = build_disagreement(platoon_frame, eb_frame, names)
    summary = summarize(disagreement)
    top = top_disagreements(disagreement)

    disagreement.to_csv(f"{args.out_dir}/disagreement.csv", index=False)
    with open(f"{args.out_dir}/disagreement_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    top.to_csv(f"{args.out_dir}/disagreement_top.csv", index=False)
    fig_disagreement(disagreement, summary, f"{args.out_dir}/fig_disagreement.png")

    print(f"join counts: {join_counts}")
    print("summary:", json.dumps(summary, indent=2))
    print(f"wrote disagreement.csv, disagreement_summary.json, disagreement_top.csv, "
          f"fig_disagreement.png to {args.out_dir}/")


if __name__ == "__main__":
    main()
