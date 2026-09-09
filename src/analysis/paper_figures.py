"""
Six paper figures rendered from already-committed result artifacts.

No model inference and no training happen here. Every number plotted is either
read directly from a committed CSV/JSON, or is plotting-only arithmetic (curve
shapes, offsets, sort order) applied to those numbers.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Repo-relative locations of the read-only input artifacts.
STABILIZATION_PANEL_CSV = "results/feature_screening/stabilization_panel.csv"
LEVEL_CEILING_JSON = "results/measurement_ceiling/level_ceiling_level_ceiling.json"
MIN_PA_SWEEP_CSV = "results/model_evaluation_final/min_pa_sweep_b_min_pa_sweep.csv"
CALIBRATION_CSV = "results/model_evaluation_final/calibration.csv"
CALIBRATION_RELIABILITY_CSV = "results/model_evaluation_final/calibration_reliability.csv"
REPLAY_SCHEDULE_JSON = "results/model_v1/replay_schedule.json"
SEED_STABILITY_JSON = "results/model_visualization/seed_stability_corrected_null.json"
DIMENSION_USAGE_JSON = "results/model_visualization/dimension_usage.json"
EXPOSURE_LOADINGS_JSON = "results/model_visualization/exposure_loadings.json"
LEVEL_QUERY_JSON = "results/model_visualization/level_query.json"
SURFACES_SUMMARY_CSV = "results/model_visualization/surfaces_summary.csv"
HITTER_STATS_CSV = "results/model_visualization/hitter_stats.csv"
HITTER_NAMES_CSV = "data/processed/hitter_names.csv"

# Frozen numbers used by fig 4 tile (a), the replay gate, sourced from the
# sealed refit reported in the research manifest (not present as a standalone
# artifact, so they are recorded here as named constants rather than a magic
# number inline).
REPLAY_GATE_ARM_MEAN = 1.02386
REPLAY_GATE_ARM_BAND = 0.00018
REPLAY_GATE_REFIT_ACHIEVED = 1.02393

DEFAULT_OUT_DIR = "results/paper_figures"
SMOKE_MAX_PA = 200
FULL_MAX_PA = 1200

FIGURE_NAMES = [
    "reliability_curve",
    "claim1_2025_strata",
    "calibration_2025",
    "tuned_well",
    "level_query_strata",
    "platoon_direction_summary",
]


def repo_root() -> Path:
    """Returns the repo root, three levels up from this file (src/analysis/paper_figures.py)."""
    return Path(__file__).resolve().parents[2]


def reliability(n, n_star):
    """Split-half reliability n / (n + n_star) for exposure n and stabilization point n_star."""
    return n / (n + n_star)


def rank_correlation_ceiling(n, n_star):
    """
    Attenuation ceiling on rank correlation: the square root of reliability.

    A rank correlation between two noisy measurements is attenuated by the square
    root of each one's reliability, so sqrt(reliability) is the highest correlation
    any model could score against a target measured this noisily.
    """
    return np.sqrt(reliability(n, n_star))


def exposure_at_reliability(target_reliability, n_star):
    """
    The exposure n at which reliability(n, n_star) equals target_reliability.

    Inverts n / (n + n*). The measurement-ceiling result reports the reliability of
    the evaluated hitter population, not a PA count, so this is how that result is
    placed on an exposure axis. It is NOT n*: reliability at n* is 0.5 by
    construction, while the evaluated population sits at 0.5506.
    """
    return n_star * target_reliability / (1 - target_reliability)


def load_stabilization_row(root, metric_name):
    """Returns the stabilization_panel.csv row for the given metric name as a dict."""
    panel = pd.read_csv(root / STABILIZATION_PANEL_CSV)
    row = panel[panel["metric"] == metric_name]
    if row.empty:
        raise ValueError(f"metric {metric_name!r} not found in {STABILIZATION_PANEL_CSV}")
    return row.iloc[0].to_dict()


def load_min_pa_sweep(root, opponent, min_eval_pa_values):
    """Returns min_pa_sweep rows for the given opponent restricted to the given min_eval_pa floors."""
    sweep = pd.read_csv(root / MIN_PA_SWEEP_CSV)
    filtered = sweep[(sweep["opponent"] == opponent) & (sweep["min_eval_pa"].isin(min_eval_pa_values))]
    return filtered.reset_index(drop=True)


def load_calibration(root):
    """Returns the calibration.csv rows for the four strata (low, medium, high, all)."""
    return pd.read_csv(root / CALIBRATION_CSV)


def fig_reliability_curve(root, out_dir, max_pa=FULL_MAX_PA):
    """Figure 1: reliability(n) and its rank-correlation ceiling over exposure n, with the frozen markers."""
    stabilization_row = load_stabilization_row(root, "wOBA vs LHP")
    n_star = stabilization_row["vc_n50"]
    n_star_low = stabilization_row["vc_n50_lo"]
    n_star_high = stabilization_row["vc_n50_hi"]

    with open(root / LEVEL_CEILING_JSON) as f:
        level_ceiling = json.load(f)["intersection"]
    ceiling_rank_corr = level_ceiling["ceiling_rank_corr"]
    achieved_rank_corr = level_ceiling["achieved_rank_corr_weighted"]
    fraction_of_ceiling = level_ceiling["fraction_of_ceiling"]

    exposure_stratum_edges = [113, 452]
    evaluation_floors = [10, 50]
    achieved_reliability = ceiling_rank_corr**2
    achieved_pa = exposure_at_reliability(achieved_reliability, n_star)

    plate_appearances = np.linspace(1, max_pa, 2000)
    reliability_curve = reliability(plate_appearances, n_star)
    ceiling_curve = rank_correlation_ceiling(plate_appearances, n_star)

    figure, axis = plt.subplots(figsize=(8, 5.5))
    axis.fill_between(
        np.linspace(n_star_low, n_star_high, 2),
        0,
        1,
        color="tab:blue",
        alpha=0.08,
        label=f"n* 95% CI [{n_star_low:.0f}, {n_star_high:.0f}] PA",
    )
    axis.plot(plate_appearances, reliability_curve, label="split-half reliability  n / (n + n*)", color="tab:blue")
    axis.plot(plate_appearances, ceiling_curve, label="rank-correlation ceiling  sqrt(reliability)", color="tab:orange")

    axis.axhline(ceiling_rank_corr, color="tab:orange", linestyle="--", linewidth=1)
    axis.annotate(
        f"ceiling {ceiling_rank_corr:.4f}",
        xy=(max_pa * 0.98, ceiling_rank_corr),
        ha="right",
        va="bottom",
        fontsize=9,
        color="tab:orange",
    )
    axis.scatter([achieved_pa], [ceiling_rank_corr], facecolors="none",
                 edgecolors="tab:orange", zorder=5, s=60)
    axis.scatter([achieved_pa], [achieved_rank_corr], color="black", zorder=5)
    axis.annotate(
        f"achieved {achieved_rank_corr:.4f}\n({fraction_of_ceiling:.1%} of ceiling)",
        xy=(achieved_pa, achieved_rank_corr),
        xytext=(achieved_pa + max_pa * 0.05, achieved_rank_corr - 0.12),
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "black"},
    )

    for edge_pa in exposure_stratum_edges:
        axis.axvline(edge_pa, color="gray", linestyle=":", linewidth=1)
        axis.text(edge_pa, 0.96, f"stratum edge n={edge_pa}", ha="left", va="top",
                  fontsize=8, color="gray", rotation=90,
                  transform=axis.get_xaxis_transform())

    # The two floors sit close together on a 1200-PA axis, so their labels are
    # staggered vertically rather than centred on the line.
    for floor_pa, label_y in zip(evaluation_floors, [0.16, 0.08]):
        floor_reliability = reliability(floor_pa, n_star)
        axis.axvline(floor_pa, color="firebrick", linestyle=":", linewidth=1)
        axis.annotate(
            f"eval floor n={floor_pa}  ({floor_reliability:.1%} reliable)",
            xy=(floor_pa, label_y), xycoords=axis.get_xaxis_transform(),
            xytext=(max_pa * 0.10, label_y), textcoords=axis.get_xaxis_transform(),
            ha="left", va="center", fontsize=8, color="firebrick",
            arrowprops={"arrowstyle": "->", "color": "firebrick", "linewidth": 0.7},
        )

    axis.set_xlabel("plate appearances vs LHP (n)")
    axis.set_ylabel("reliability / rank-correlation ceiling")
    axis.set_xlim(0, max_pa)
    axis.set_ylim(0, 1.05)
    axis.set_title("Reliability and its rank-correlation ceiling vs plate-appearance exposure",
                   pad=12)
    axis.legend(loc="lower right", fontsize=8)
    figure.tight_layout()

    output_path = out_dir / "fig_reliability_curve.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def fig_claim1_2025_strata(root, out_dir):
    """Figure 2: rank_difference and rmse_difference vs gbm_full by stratum, at both evaluation floors."""
    floors = [10, 50]
    strata = ["low", "medium", "high"]
    floor_offset = 0.15

    sweep = load_min_pa_sweep(root, "gbm_full", floors)
    sweep = sweep[sweep["stratum"].isin(strata)]

    figure, (axis_rank, axis_rmse) = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    stratum_positions = {stratum: i for i, stratum in enumerate(strata)}
    floor_colors = {10: "tab:blue", 50: "tab:orange"}

    for floor_pa in floors:
        floor_rows = sweep[sweep["min_eval_pa"] == floor_pa]
        y_positions = [stratum_positions[s] + (floor_offset if floor_pa == floors[-1] else -floor_offset) for s in floor_rows["stratum"]]

        rank_diff = floor_rows["rank_difference"].to_numpy()
        rank_lo = rank_diff - floor_rows["ci_low_rank"].to_numpy()
        rank_hi = floor_rows["ci_high_rank"].to_numpy() - rank_diff
        gate_pass = (floor_rows["ci_low_rank"].to_numpy() > 0)
        axis_rank.errorbar(
            rank_diff, y_positions, xerr=[rank_lo, rank_hi],
            fmt="none", color=floor_colors[floor_pa], label=f"min PA = {floor_pa}",
            ecolor=floor_colors[floor_pa], capsize=3,
        )
        for x, y, passed in zip(rank_diff, y_positions, gate_pass):
            axis_rank.scatter([x], [y], marker="o", facecolor=floor_colors[floor_pa] if passed else "none", edgecolor=floor_colors[floor_pa], zorder=5)

        rmse_diff = floor_rows["rmse_difference"].to_numpy()
        rmse_lo = rmse_diff - floor_rows["ci_low_rmse"].to_numpy()
        rmse_hi = floor_rows["ci_high_rmse"].to_numpy() - rmse_diff
        rmse_gate_pass = (floor_rows["ci_high_rmse"].to_numpy() < 0)
        axis_rmse.errorbar(
            rmse_diff, y_positions, xerr=[rmse_lo, rmse_hi],
            fmt="none", color=floor_colors[floor_pa], label=f"min PA = {floor_pa}",
            ecolor=floor_colors[floor_pa], capsize=3,
        )
        for x, y, passed in zip(rmse_diff, y_positions, rmse_gate_pass):
            axis_rmse.scatter([x], [y], marker="o", facecolor=floor_colors[floor_pa] if passed else "none", edgecolor=floor_colors[floor_pa], zorder=5)

    axis_rank.axvline(0, color="black", linewidth=0.8)
    axis_rank.set_yticks(list(stratum_positions.values()))
    axis_rank.set_yticklabels(list(stratum_positions.keys()))
    axis_rank.set_xlabel("rank correlation difference vs gbm_full\n(model minus opponent, positive favours model)")
    axis_rank.set_title("Panel A: rank correlation, 2025")
    axis_rank.legend(fontsize=8)

    axis_rmse.axvline(0, color="black", linewidth=0.8)
    axis_rmse.set_xlabel("RMSE difference vs gbm_full\n(model minus opponent, negative favours model)")
    axis_rmse.set_title("Panel B: RMSE, 2025")
    axis_rmse.legend(fontsize=8)

    figure.suptitle("Claim 1: model vs gbm_full by exposure stratum, filled marker = CI excludes zero in model's favour")
    figure.tight_layout()

    output_path = out_dir / "fig_claim1_2025_strata.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def fig_calibration_2025(root, out_dir):
    """Figure 3: fitted calibration line per stratum, identity line, and binned overlay on the all panel."""
    calibration = load_calibration(root)
    reliability_bins = pd.read_csv(root / CALIBRATION_RELIABILITY_CSV)

    panel_order = ["low", "medium", "high", "all"]
    figure, axes = plt.subplots(1, 4, figsize=(15, 4), sharex=True, sharey=True)

    # No hitter is predicted outside roughly 0.26-0.37 wOBA, so a fitted line drawn
    # across the full axis is extrapolation. It is drawn faint outside the observed
    # span so the eye does not read the low panel's divergence off invented range.
    observed_low = reliability_bins["mean_pred"].min()
    observed_high = reliability_bins["mean_pred"].max()

    for axis, stratum_name in zip(axes, panel_order):
        row = calibration[calibration["stratum"] == stratum_name].iloc[0]
        predicted_range = np.linspace(0.15, 0.45, 200)
        observed_range = np.linspace(observed_low, observed_high, 50)
        fitted_line = row["intercept"] + row["slope"] * predicted_range

        axis.axvspan(observed_low, observed_high, color="tab:blue", alpha=0.05)
        axis.plot(predicted_range, predicted_range, color="gray", linestyle="--", linewidth=1, label="identity")
        axis.plot(predicted_range, fitted_line, color="tab:blue", alpha=0.25, linewidth=1)
        axis.plot(observed_range, row["intercept"] + row["slope"] * observed_range,
                  color="tab:blue", linewidth=2, label="fitted (observed range)")

        if stratum_name == "all":
            axis.scatter(
                reliability_bins["mean_pred"], reliability_bins["mean_obs"],
                color="tab:red", s=20, zorder=5, label="PA-weighted quantile bins",
            )

        axis.set_title(stratum_name)
        axis.set_xlabel("predicted wOBA")
        axis.annotate(
            f"slope {row['slope']:.3f} +/- {row['slope_se']:.3f}\nn={row['n_hitters']}",
            xy=(0.03, 0.95), xycoords="axes fraction", va="top", fontsize=8,
        )
        axis.legend(fontsize=7, loc="lower right")

    axes[0].set_ylabel("observed wOBA")
    figure.suptitle("Per-hitter side-specific calibration, 2025 sealed evaluation")
    figure.tight_layout()

    output_path = out_dir / "fig_calibration_2025.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def fig_tuned_well(root, out_dir):
    """
    Appendix table: the four build diagnostics, as numbers rather than as four
    independently scaled tiles.

    The 2x2 tile grid this replaces auto-scaled each panel to its own range, so a
    0.00007 replay difference and a 0.82 stability margin drew the same size mark.
    Two of the four rows are gate passes and read as numbers; the other two —
    effective rank and the norm-exposure correlation — are results that bound how
    the embedding may be projected, and they are quoted in the representation
    section's text as well as here.
    """
    with open(root / REPLAY_SCHEDULE_JSON) as f:
        replay_schedule = json.load(f)
    with open(root / SEED_STABILITY_JSON) as f:
        seed_stability = json.load(f)
    with open(root / DIMENSION_USAGE_JSON) as f:
        dimension_usage = json.load(f)
    with open(root / EXPOSURE_LOADINGS_JSON) as f:
        exposure_loadings = json.load(f)

    stability_low, stability_high = seed_stability["diff_ci95"]
    norm_r = exposure_loadings["norm_r"]
    norm_low, norm_high = exposure_loadings["norm_r_ci95"]
    effective_rank = dimension_usage["effective_rank"]
    n_dims = dimension_usage["n_dims"]

    rows = [
        ["replay gate",
         f"{REPLAY_GATE_REFIT_ACHIEVED:.5f}",
         f"arm {REPLAY_GATE_ARM_MEAN:.5f} +/- {REPLAY_GATE_ARM_BAND:.5f}",
         f"step budget {replay_schedule['step_budget']}, arm {replay_schedule['arm']}"],
        ["seed stability",
         f"{seed_stability['real_mean']:.4f}",
         f"corrected null {seed_stability['corrected_null_mean']:.4f}",
         f"difference 95% CI [{stability_low:.4f}, {stability_high:.4f}]"],
        ["effective rank",
         f"{effective_rank:.1f}",
         f"of {n_dims} dimensions",
         "a two-dimensional projection discards almost all of it"],
        ["norm vs log prior PA",
         f"r = {norm_r:.3f}",
         f"95% CI [{norm_low:.3f}, {norm_high:.3f}]",
         f"PC1 vs the same exposure: r = {exposure_loadings['pc1_abs_r']:.3f}"],
    ]

    figure, axis = plt.subplots(figsize=(11, 2.6))
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=["diagnostic", "value", "reference", "note"],
        cellLoc="left",
        colLoc="left",
        loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    for column, width in enumerate([0.18, 0.14, 0.24, 0.44]):
        for row in range(len(rows) + 1):
            table[row, column].set_width(width)
    for column in range(4):
        table[0, column].set_text_props(weight="bold")
    axis.set_title("Appendix: build and embedding diagnostics", loc="left")

    figure.tight_layout()
    output_path = out_dir / "fig_tuned_well.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def fig_level_query_strata(root, out_dir):
    """Figure 5: partial correlation with its CI per stratum plus pooled, from level_query.json."""
    with open(root / LEVEL_QUERY_JSON) as f:
        level_query = json.load(f)

    row_order = ["pooled", "low", "medium", "high"]
    r_partial = [level_query[name]["r_partial"] for name in row_order]
    ci_low = [level_query[name]["ci_low"] for name in row_order]
    ci_high = [level_query[name]["ci_high"] for name in row_order]

    figure, axis = plt.subplots(figsize=(7, 4))
    y_positions = np.arange(len(row_order))
    error_low = [r - lo for r, lo in zip(r_partial, ci_low)]
    error_high = [hi - r for hi, r in zip(ci_high, r_partial)]
    axis.errorbar(r_partial, y_positions, xerr=[error_low, error_high], fmt="o", color="tab:blue", capsize=4)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_yticks(y_positions)
    axis.set_yticklabels(row_order)
    axis.set_xlabel("partial correlation r (descriptive, in-sample, not a held-out test)")
    axis.set_title("Level query: partial correlation by exposure stratum")
    figure.tight_layout()

    output_path = out_dir / "fig_level_query_strata.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


PLATOON_BARELY_USED_THRESHOLD = 0.005


def load_platoon_lr_gaps(root):
    """
    Per-anchor model platoon gap, with the league context offset removed, joined to
    each anchor's observed platoon split.

    surfaces_summary.csv holds one row per anchor x p_throws x quantity; quantity "q"
    is the anchor's expected wOBA over that hand's context pool. The L and R pools are
    different sets of pitches, so the raw q(L) - q(R) carries a league-wide offset that
    is identical for every anchor (+0.0035 here) and is a property of the pools, not of
    any hitter. Subtracting the same gap computed on league_value removes it, which is
    why the adjusted gap and not the raw gap is the hitter's platoon signal.
    """
    surfaces = pd.read_csv(root / SURFACES_SUMMARY_CSV)
    expected_woba_rows = surfaces[surfaces["quantity"] == "q"]
    pivoted = expected_woba_rows.pivot(index="anchor_name", columns="p_throws",
                                       values=["anchor_value", "league_value"])
    if ("anchor_value", "L") not in pivoted.columns or ("anchor_value", "R") not in pivoted.columns:
        return None
    raw_gap = pivoted[("anchor_value", "L")] - pivoted[("anchor_value", "R")]
    league_gap = pivoted[("league_value", "L")] - pivoted[("league_value", "R")]
    gaps = pd.DataFrame({"anchor_name": pivoted.index,
                         "raw_gap": raw_gap.to_numpy(),
                         "league_gap": league_gap.to_numpy(),
                         "model_gap": (raw_gap - league_gap).to_numpy()})

    hitter_stats = pd.read_csv(root / HITTER_STATS_CSV)[["batter", "stand", "obs_platoon_diff"]]
    names = pd.read_csv(root / HITTER_NAMES_CSV)[["batter", "name"]]
    observed = hitter_stats.merge(names, on="batter").rename(columns={"name": "anchor_name"})
    return gaps.merge(observed, on="anchor_name", how="left")


def fig_platoon_direction_summary(root, out_dir):
    """
    Figure 6: the model's platoon gap against the observed one, per anchor hitter.

    Plotting both makes the two facts visible at once: every anchor lands in the
    correct half of the plot, so the direction is right, and every anchor sits far
    below the identity line, so the size is heavily compressed.
    """
    gaps = load_platoon_lr_gaps(root)
    if gaps is None:
        print("skipped fig_platoon_direction_summary: no per-anchor L/R expected-wOBA columns found")
        return None

    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    axis.axhline(0, color="gray", linewidth=0.8)
    axis.axvline(0, color="gray", linewidth=0.8)

    span = [-0.05, 0.10]
    axis.plot(span, span, color="gray", linestyle="--", linewidth=1,
              label="identity (model matches observed)")
    axis.fill_between(span, [-PLATOON_BARELY_USED_THRESHOLD] * 2,
                      [PLATOON_BARELY_USED_THRESHOLD] * 2, color="black", alpha=0.06,
                      label=f"model gap below {PLATOON_BARELY_USED_THRESHOLD} (barely used)")

    axis.scatter(gaps["obs_platoon_diff"], gaps["model_gap"], color="tab:blue", zorder=5)
    # Anchors cluster near the same observed split, so labels alternate above and
    # below the point rather than all sitting to the upper right of each other.
    gaps = gaps.sort_values("obs_platoon_diff").reset_index(drop=True)
    for position, row in gaps.iterrows():
        label_offset = (8, 6) if position % 2 == 0 else (8, -14)
        axis.annotate(f"{row['anchor_name']} ({row['stand']}HH)",
                      xy=(row["obs_platoon_diff"], row["model_gap"]),
                      xytext=label_offset, textcoords="offset points", fontsize=8)

    correct_direction = int((np.sign(gaps["model_gap"]) == np.sign(gaps["obs_platoon_diff"])).sum())
    axis.set_xlim(*span)
    axis.set_ylim(-0.05, 0.10)
    axis.set_xlabel("observed platoon split, wOBA vs LHP minus vs RHP")
    axis.set_ylabel("model platoon gap, league context offset removed")
    axis.set_title(
        f"Direction right on {correct_direction}/{len(gaps)} anchors, magnitude compressed\n"
        "(points below the identity line under-state the real split)")
    axis.legend(fontsize=8, loc="upper left")
    figure.tight_layout()

    output_path = out_dir / "fig_platoon_direction_summary.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


FIGURE_FUNCTIONS = {
    "reliability_curve": fig_reliability_curve,
    "claim1_2025_strata": fig_claim1_2025_strata,
    "calibration_2025": fig_calibration_2025,
    "tuned_well": fig_tuned_well,
    "level_query_strata": fig_level_query_strata,
    "platoon_direction_summary": fig_platoon_direction_summary,
}


def render_figure(name, root, out_dir, smoke=False):
    """Renders one named figure into out_dir, returning the Path written or None if skipped."""
    if name == "reliability_curve":
        max_pa = SMOKE_MAX_PA if smoke else FULL_MAX_PA
        return fig_reliability_curve(root, out_dir, max_pa=max_pa)
    return FIGURE_FUNCTIONS[name](root, out_dir)


def main():
    parser = argparse.ArgumentParser(description="Render the six paper figures from committed result artifacts.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="directory to write figures into")
    parser.add_argument("--smoke", action="store_true", help="tiny run for tests, still writes real figures")
    parser.add_argument("--only", default=None, choices=FIGURE_NAMES, help="render a single figure by name")
    args = parser.parse_args()

    root = repo_root()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names_to_render = [args.only] if args.only else FIGURE_NAMES
    for name in names_to_render:
        written_path = render_figure(name, root, out_dir, smoke=args.smoke)
        if written_path is not None:
            print(f"wrote {written_path}")


if __name__ == "__main__":
    main()
