"""
Alternative visualizations of the hitter embedding, built because the honest
2-D PCA/t-SNE map (see embedding_structure.py) is a featureless blob: no
discrete clusters (silhouette maxes 0.0617 at k=2), and PCA retains only 18%
of 32-D variance.

Plots the two winning axes from the frozen axis screen
(results/embedding_structure/axis_screen.py — read that file for the exact
metric definitions, min-sample rules, and seed; not re-run here):

- ev_p90: per-hitter 90th-percentile exit velocity (power).
- whiff_brk_minus_fb: per-hitter (whiff rate on breaking balls) minus
  (whiff rate on fastballs).

For each axis: tercile the continuous metric, fit LDA on the frozen
embedding to predict the tercile, and use the discriminant score as the
plotted coordinate. Held-out 5-fold CV accuracy is the honest number, since
a supervised projection always looks good on the data it was fit to.

1. Supervised LDA projection — two panels sharing the SAME (ev_p90-LDA,
   whiff-LDA) axes, colored by each metric's own tercile.
2. Density-binned heatmap over the same plane, colored by mean ev_p90 (mph)
   per bin — a reading aid, not new evidence of structure.

whiff_brk_minus_fb is only defined for hitters with >= 20 breaking AND >= 20
fastball swings (axis_screen.MIN_SWINGS_PER_SLICE); both figures plot that
valid subset so the two panels/figures show identical points.

Reads frozen checkpoints and existing results only. No retraining, no 2025
data.

Run: PYTHONPATH=. .venv/bin/python -m src.analysis.embedding_alt_views --out-dir results/embedding_structure
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold

from src.analysis.embedding_structure import (
    DEFAULT_ARM, DEFAULT_CHECKPOINT_DIR, DEFAULT_OUT_DIR, HITTER_STATS_PATH,
    NAMES_PATH, BOOT_SEED, load_hitters, tercile_labels, unit_normalize,
)
from results.embedding_structure import axis_screen

PITCH_EVENTS_PATH = "data/processed/pitch_events_labeled.parquet"
# Same fastball/breaking Statcast pitch_type groups as axis_screen.py /
# baseline_ladder_gbm.PITCH_GROUPS (copied, not imported, to avoid
# baseline_ladder_gbm's xgboost import; axis_screen.py copies it for the
# same reason).
PITCH_GROUPS = {
    "fastball": {"FF", "SI", "FC", "FA"},
    "breaking": {"SL", "CU", "KC", "ST", "SV", "CS", "KN", "SC", "EP"},
}
AXIS_LABELS = {
    "ev_p90": "power (90th-pct exit velocity), LDA score",
    "whiff_brk_minus_fb": "breaking-ball minus fastball whiff, LDA score",
    "contact_rate": "contact rate, LDA score",
    "selectivity": "selectivity (zone swing minus chase), LDA score",
}
METRIC_TITLES = {
    "ev_p90": "Power tercile (ev_p90)",
    "contact_rate": "Contact tercile (contact_rate)",
    "selectivity": "Selectivity tercile (zone swing minus chase)",
    "whiff_brk_minus_fb": "Breaking-minus-fastball whiff tercile",
}
METRIC_UNITS = {
    "ev_p90": "mph",
    "contact_rate": "rate (0-1)",
    "selectivity": "rate (0-1)",
    "whiff_brk_minus_fb": "rate (0-1)",
}
N_HEATMAP_BINS = 12
MIN_HITTERS_PER_BIN = 5

# id -> (x_metric, y_metric, expected_x_cv, expected_y_cv). ev_p90 is the x
# axis throughout (frozen "power" axis); y varies. All three metrics besides
# ev_p90 come straight from axis_screen (either a MARGINAL_COLUMNS hitters
# column or a build_contrasts() entry) -- reused exactly, not re-derived.
AXIS_PAIRINGS = [
    ("power_contact", "ev_p90", "contact_rate", 0.800, 0.765),
    ("power_discipline", "ev_p90", "selectivity", 0.800, 0.650),
    ("power_spin", "ev_p90", "whiff_brk_minus_fb", 0.800, 0.633),
]
CV_TOLERANCE = 0.02


# --------------------------------------------------------------------- pitch-level feature
# NOTE: axis_screen.build_contrasts imports this function from this module
# (lazily, inside the function) — keep name/signature stable.

def whiff_brk_minus_fb_by_batter(pitch_events_path=PITCH_EVENTS_PATH):
    """Per-batter (whiff rate on breaking pitches) minus (whiff rate on
    fastballs), computed from swing-level pitch_events_labeled.parquet.
    whiff = swing & not contact. Descriptive-only feature (not a training
    target), so none of the leakage-window discipline in baseline_ladder_gbm
    applies. Returns a Series indexed by batter."""
    df = pd.read_parquet(pitch_events_path, columns=["batter", "pitch_type", "swing", "contact"])
    df = df[df["swing"] == 1]
    group = pd.Series(np.nan, index=df.index, dtype=object)
    for name, codes in PITCH_GROUPS.items():
        group[df["pitch_type"].isin(codes)] = name
    df = df.assign(pitch_group=group)
    df = df[df["pitch_group"].isin(PITCH_GROUPS)]
    df = df.assign(whiff=(df["contact"] == 0).astype(float))
    rates = df.groupby(["batter", "pitch_group"])["whiff"].mean().unstack("pitch_group")
    return (rates["breaking"] - rates["fastball"]).rename("whiff_brk_minus_fb")


# --------------------------------------------------------------------- supervised LDA axis

def lda_axis_cv(normalized, labels, seed=BOOT_SEED, n_splits=5):
    """K-fold held-out classification accuracy of an LDA fit per-fold (fit
    on train, scored on held-out test) — the honest number, since a
    supervised projection always looks good on the data it was fit to.
    Returns (model fit on ALL rows, cv_mean, cv_std)."""
    labels = np.asarray(labels)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for train_idx, test_idx in kf.split(normalized):
        model = LinearDiscriminantAnalysis()
        model.fit(normalized[train_idx], labels[train_idx])
        accs.append(model.score(normalized[test_idx], labels[test_idx]))
    full_model = LinearDiscriminantAnalysis().fit(normalized, labels)
    return full_model, float(np.mean(accs)), float(np.std(accs))


def _padded_lim(values, pad_frac=0.05):
    lo, hi = float(np.min(values)), float(np.max(values))
    pad = pad_frac * (hi - lo)
    return lo - pad, hi + pad


# --------------------------------------------------------------------- (1) supervised projection

def fig_supervised_projection(x, y, ev_tercile, whiff_tercile, ev_cv_mean, whiff_cv_mean, path):
    """Two panels, IDENTICAL (x, y) axes — x = ev_p90 LDA score, y =
    whiff_brk_minus_fb LDA score. Left colored by ev_p90 tercile, right by
    whiff_brk_minus_fb tercile."""
    xlim, ylim = _padded_lim(x), _padded_lim(y)
    palette = {0: "#d9d9d9", 1: "#8fb3ff", 2: "#1f4fd1"}
    tercile_names = ["low", "mid", "high"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)
    panels = [
        (axes[0], np.asarray(ev_tercile), ev_cv_mean, "Power tercile (ev_p90)"),
        (axes[1], np.asarray(whiff_tercile), whiff_cv_mean, "Breaking-minus-fastball whiff tercile"),
    ]
    for ax, tercile, cv_mean, title in panels:
        for t, color in palette.items():
            mask = tercile == t
            ax.scatter(x[mask], y[mask], s=8, alpha=0.6, color=color, label=tercile_names[t])
        ax.set_xlabel(AXIS_LABELS["ev_p90"])
        ax.set_ylabel(AXIS_LABELS["whiff_brk_minus_fb"])
        ax.set_title(f"{title}\nheld-out CV accuracy {cv_mean:.3f} (chance = 0.333)")
        ax.legend(fontsize=8, title="tercile")
        ax.set_xlim(xlim); ax.set_ylim(ylim)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------- (2) density-binned heatmap

def fig_similarity_heatmap(x, y, ev_values, path, n_bins=N_HEATMAP_BINS,
                            min_per_bin=MIN_HITTERS_PER_BIN):
    """2-D binned gradient over the SAME (ev_p90-LDA, whiff-LDA) plane as
    Figure 1 — no individual dots. Each cell colored by the mean ev_p90
    (mph) of hitters in it; cells with fewer than `min_per_bin` hitters are
    greyed out."""
    x, y, ev_values = np.asarray(x), np.asarray(y), np.asarray(ev_values)
    x_edges = np.linspace(x.min(), x.max(), n_bins + 1)
    y_edges = np.linspace(y.min(), y.max(), n_bins + 1)
    xi = np.clip(np.digitize(x, x_edges) - 1, 0, n_bins - 1)
    yi = np.clip(np.digitize(y, y_edges) - 1, 0, n_bins - 1)

    sums = np.zeros((n_bins, n_bins))
    counts = np.zeros((n_bins, n_bins))
    np.add.at(sums, (yi, xi), ev_values)
    np.add.at(counts, (yi, xi), 1)
    means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    means_masked = np.ma.masked_where(counts < min_per_bin, means)

    cmap = plt.get_cmap("viridis").with_extremes(bad="#d0d0d0")

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(means_masked, origin="lower", cmap=cmap, aspect="auto",
                    extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
    fig.colorbar(im, ax=ax, label="mean 90th-pct exit velocity (mph)")
    ax.set_xlabel(AXIS_LABELS["ev_p90"])
    ax.set_ylabel(AXIS_LABELS["whiff_brk_minus_fb"])
    ax.set_title(
        f"Same plane as Figure 1, density-binned ({n_bins}x{n_bins}) —\n"
        f"grey cells have fewer than {min_per_bin} hitters (n={len(x)} hitters)",
        fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------- (3) generalized axis pairing

def metric_series_by_batter(name, hitters, contrasts):
    """Per-batter Series for any axis_screen candidate: a hitters column for
    a MARGINAL_COLUMNS metric, or a build_contrasts() entry otherwise. Same
    lookup axis_screen.score_candidate uses -- reused, not re-derived."""
    if name in axis_screen.MARGINAL_COLUMNS:
        return hitters.set_index("batter")[name].dropna()
    return contrasts[name]


def run_pairing(pairing_id, x_metric, y_metric, expected_x_cv, expected_y_cv,
                 normalized, hitters, contrasts):
    """Full protocol for one (x_metric, y_metric) axis pair: frozen
    embedding, tercile labels, LDA, 5-fold shuffled CV (seed=BOOT_SEED),
    season <=2024 (inherited from hitters/contrasts). Raises if either CV
    accuracy drifts from its expected value by more than CV_TOLERANCE."""
    x_series = metric_series_by_batter(x_metric, hitters, contrasts)
    y_series = metric_series_by_batter(y_metric, hitters, contrasts)

    x_valid = hitters["batter"].isin(x_series.index).to_numpy()
    y_valid = hitters["batter"].isin(y_series.index).to_numpy()

    # LDA + CV for each axis is fit on ITS OWN valid population (same as the
    # original ev_p90-vs-whiff code: x fit on the full ev_p90 population,
    # y fit on the smaller whiff-valid population) -- not the intersection.
    x_vals_full = hitters.loc[x_valid, "batter"].map(x_series).to_numpy()
    x_tercile_full = tercile_labels(pd.Series(x_vals_full)).values
    x_model, x_cv_mean, x_cv_std = lda_axis_cv(normalized[x_valid], x_tercile_full, BOOT_SEED)

    y_vals_full = hitters.loc[y_valid, "batter"].map(y_series).to_numpy()
    y_tercile_full = tercile_labels(pd.Series(y_vals_full)).values
    y_model, y_cv_mean, y_cv_std = lda_axis_cv(normalized[y_valid], y_tercile_full, BOOT_SEED)

    if abs(x_cv_mean - expected_x_cv) > CV_TOLERANCE or abs(y_cv_mean - expected_y_cv) > CV_TOLERANCE:
        raise RuntimeError(
            f"[{pairing_id}] CV accuracy drifted from expectation: "
            f"{x_metric}={x_cv_mean:.4f} (expected {expected_x_cv}), "
            f"{y_metric}={y_cv_mean:.4f} (expected {expected_y_cv}). Stopping.")

    # Plot the population where BOTH axes are defined, and where woba_level
    # (the gradient panel's color) is also available.
    woba_series = metric_series_by_batter("woba_level", hitters, contrasts)
    plot_mask = x_valid & y_valid & hitters["batter"].isin(woba_series.index).to_numpy()

    x_plot_vals = hitters.loc[plot_mask, "batter"].map(x_series).to_numpy()
    y_plot_vals = hitters.loc[plot_mask, "batter"].map(y_series).to_numpy()
    x_tercile_plot = tercile_labels(pd.Series(x_plot_vals)).values
    y_tercile_plot = tercile_labels(pd.Series(y_plot_vals)).values
    woba_plot_vals = hitters.loc[plot_mask, "batter"].map(woba_series).to_numpy()

    x_proj = x_model.transform(normalized[plot_mask])[:, 0]
    y_proj = y_model.transform(normalized[plot_mask])[:, 0]

    x_span = (float(x_plot_vals.min()), float(x_plot_vals.max()))
    y_span = (float(y_plot_vals.min()), float(y_plot_vals.max()))

    definitions = axis_screen.DEFINITIONS
    summary = {
        "x": {
            "metric": x_metric, "definition": definitions[x_metric][0],
            "label": AXIS_LABELS[x_metric], "lda_cv_accuracy_mean": x_cv_mean,
            "lda_cv_accuracy_std": x_cv_std, "n": int(x_valid.sum()),
            "real_unit_span_plotted": x_span, "unit": METRIC_UNITS[x_metric],
        },
        "y": {
            "metric": y_metric, "definition": definitions[y_metric][0],
            "label": AXIS_LABELS[y_metric], "lda_cv_accuracy_mean": y_cv_mean,
            "lda_cv_accuracy_std": y_cv_std, "n": int(y_valid.sum()),
            "real_unit_span_plotted": y_span, "unit": METRIC_UNITS[y_metric],
        },
        "n_hitters_plotted": int(plot_mask.sum()),
    }
    return summary, x_proj, y_proj, x_tercile_plot, y_tercile_plot, woba_plot_vals, x_span, y_span


def run_all_pairings(normalized, hitters, out_dir):
    """Renders each AXIS_PAIRINGS entry as ONE 3-panel PNG (projection x
    tercile, projection y tercile, woba gradient) sharing identical x/y axes
    across all three panels."""
    figures_dir = Path(out_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    contrasts = axis_screen.build_contrasts(hitters)

    pairings_summary = {}
    for pairing_id, x_metric, y_metric, exp_x, exp_y in AXIS_PAIRINGS:
        (summary, x_proj, y_proj, x_tercile_plot, y_tercile_plot, woba_plot_vals,
         x_span, y_span) = run_pairing(
            pairing_id, x_metric, y_metric, exp_x, exp_y, normalized, hitters,
            contrasts)

        xlim, ylim = _padded_lim(x_proj), _padded_lim(y_proj)
        palette = {0: "#d9d9d9", 1: "#8fb3ff", 2: "#1f4fd1"}
        tercile_names = ["low", "mid", "high"]

        fig, axes = plt.subplots(1, 3, figsize=(19.5, 5.8))
        panels = [
            (axes[0], np.asarray(x_tercile_plot), summary["x"]["lda_cv_accuracy_mean"], METRIC_TITLES[x_metric]),
            (axes[1], np.asarray(y_tercile_plot), summary["y"]["lda_cv_accuracy_mean"], METRIC_TITLES[y_metric]),
        ]
        for ax, tercile, cv_mean, title in panels:
            for t, color in palette.items():
                mask = tercile == t
                ax.scatter(x_proj[mask], y_proj[mask], s=8, alpha=0.6, color=color, label=tercile_names[t])
            ax.set_xlabel(AXIS_LABELS[x_metric])
            ax.set_ylabel(AXIS_LABELS[y_metric])
            ax.set_title(f"{title}\nheld-out CV accuracy {cv_mean:.3f} (chance = 0.333)")
            ax.legend(fontsize=8, title="tercile")
            ax.set_xlim(xlim); ax.set_ylim(ylim)

        ax3 = axes[2]
        x_edges = np.linspace(x_proj.min(), x_proj.max(), N_HEATMAP_BINS + 1)
        y_edges = np.linspace(y_proj.min(), y_proj.max(), N_HEATMAP_BINS + 1)
        xi = np.clip(np.digitize(x_proj, x_edges) - 1, 0, N_HEATMAP_BINS - 1)
        yi = np.clip(np.digitize(y_proj, y_edges) - 1, 0, N_HEATMAP_BINS - 1)
        sums = np.zeros((N_HEATMAP_BINS, N_HEATMAP_BINS))
        counts = np.zeros((N_HEATMAP_BINS, N_HEATMAP_BINS))
        np.add.at(sums, (yi, xi), woba_plot_vals)
        np.add.at(counts, (yi, xi), 1)
        means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
        means_masked = np.ma.masked_where(counts < MIN_HITTERS_PER_BIN, means)
        cmap = plt.get_cmap("viridis").with_extremes(bad="#d0d0d0")
        im = ax3.imshow(means_masked, origin="lower", cmap=cmap, aspect="auto",
                         extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
        fig.colorbar(im, ax=ax3, label="mean observed wOBA (woba_level)")
        ax3.set_xlabel(AXIS_LABELS[x_metric])
        ax3.set_ylabel(AXIS_LABELS[y_metric])
        x_unit, y_unit = METRIC_UNITS[x_metric], METRIC_UNITS[y_metric]
        ax3.set_title(
            f"Density-binned wOBA gradient ({N_HEATMAP_BINS}x{N_HEATMAP_BINS}) -- "
            f"grey <{MIN_HITTERS_PER_BIN} hitters\n"
            f"x spans {x_metric} in [{x_span[0]:.2f}, {x_span[1]:.2f}] {x_unit}; "
            f"y spans {y_metric} in [{y_span[0]:.2f}, {y_span[1]:.2f}] {y_unit}",
            fontsize=8.5)
        ax3.set_xlim(xlim); ax3.set_ylim(ylim)

        fig.suptitle(f"{pairing_id}: {x_metric} vs {y_metric} (n={summary['n_hitters_plotted']})", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(figures_dir / f"fig_axes_{pairing_id}.png", dpi=130)
        plt.close(fig)

        pairings_summary[pairing_id] = summary

    return pairings_summary


# --------------------------------------------------------------------- pipeline

def run(checkpoint_dir, arm, hitter_stats_path, names_path, out_dir,
        pitch_events_path=PITCH_EVENTS_PATH):
    out_dir = Path(out_dir)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    embedding, hitters, _ = load_hitters(
        checkpoint_dir, arm, hitter_stats_path, names_path)
    normalized, _ = unit_normalize(embedding)  # positionally aligned to `hitters`

    # ev_p90 axis: defined for every hitter load_hitters keeps (required
    # scouting column), same population axis_screen scored (n matches its
    # ev_p90 candidate, lda_cv_mean ~0.800).
    ev_tercile_full = tercile_labels(hitters["ev_p90"]).values
    ev_model, ev_cv_mean, ev_cv_std = lda_axis_cv(normalized, ev_tercile_full, BOOT_SEED)

    # whiff_brk_minus_fb axis: only defined for hitters with >=20 breaking
    # AND >=20 fastball swings. Reuse axis_screen's own contrast builder
    # (not rewritten here) so the validity mask is byte-for-byte the one
    # that produced lda_cv_mean ~0.633 in the frozen screen.
    contrasts = axis_screen.build_contrasts(hitters)
    whiff_series = contrasts["whiff_brk_minus_fb"]
    valid_mask = hitters["batter"].isin(whiff_series.index).to_numpy()

    whiff_values_valid = hitters.loc[valid_mask, "batter"].map(whiff_series).to_numpy()
    whiff_tercile_valid = tercile_labels(pd.Series(whiff_values_valid)).values
    whiff_model, whiff_cv_mean, whiff_cv_std = lda_axis_cv(
        normalized[valid_mask], whiff_tercile_valid, BOOT_SEED)

    if abs(ev_cv_mean - 0.800) > 0.03 or abs(whiff_cv_mean - 0.633) > 0.03:
        raise RuntimeError(
            f"CV accuracy drifted from the axis screen's expectation: "
            f"ev_p90={ev_cv_mean:.3f} (expected ~0.800), "
            f"whiff_brk_minus_fb={whiff_cv_mean:.3f} (expected ~0.633). "
            f"Stopping instead of proceeding on a possibly-broken axis.")

    # Plot the SAME points (the whiff-valid subset) on both figures so the
    # two panels of Figure 1 and the Figure 2 heatmap share identical axes.
    x = ev_model.transform(normalized[valid_mask])[:, 0]
    y = whiff_model.transform(normalized[valid_mask])[:, 0]
    ev_tercile_plot = ev_tercile_full[valid_mask]
    ev_values_plot = hitters.loc[valid_mask, "ev_p90"].to_numpy()

    fig_supervised_projection(
        x, y, ev_tercile_plot, whiff_tercile_valid, ev_cv_mean, whiff_cv_mean,
        figures_dir / "fig_supervised_projection.png")
    fig_similarity_heatmap(x, y, ev_values_plot, figures_dir / "fig_similarity_heatmap.png")

    summary = {
        "n_hitters": int(len(hitters)),
        "n_hitters_plotted": int(valid_mask.sum()),
        "axes": {
            "x": {
                "metric": "ev_p90",
                "definition": "Per-hitter 90th-percentile exit velocity.",
                "label": AXIS_LABELS["ev_p90"],
                "lda_cv_accuracy_mean": ev_cv_mean,
                "lda_cv_accuracy_std": ev_cv_std,
                "n": int(len(hitters)),
            },
            "y": {
                "metric": "whiff_brk_minus_fb",
                "definition": "Whiff rate on breaking balls minus on fastballs, over swings.",
                "label": AXIS_LABELS["whiff_brk_minus_fb"],
                "lda_cv_accuracy_mean": whiff_cv_mean,
                "lda_cv_accuracy_std": whiff_cv_std,
                "n": int(valid_mask.sum()),
                "min_swings_per_pitch_group": axis_screen.MIN_SWINGS_PER_SLICE,
            },
        },
        "method": {
            "seed": BOOT_SEED,
            "n_splits": 5,
            "source_screen": "results/embedding_structure/axis_screen.py",
        },
        "fig_similarity_heatmap": {
            "type": "density-binned mean ev_p90 over the ev_p90-LDA x whiff-LDA plane",
            "n_bins": N_HEATMAP_BINS,
            "min_hitters_per_bin": MIN_HITTERS_PER_BIN,
        },
    }
    with open(out_dir / "embedding_alt_views.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--arm", default=DEFAULT_ARM)
    parser.add_argument("--hitter-stats", default=HITTER_STATS_PATH)
    parser.add_argument("--names", default=NAMES_PATH)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = run(args.checkpoint_dir, args.arm, args.hitter_stats, args.names, args.out_dir)
    print(f"ev_p90 LDA held-out CV acc: {summary['axes']['x']['lda_cv_accuracy_mean']:.4f}")
    print(f"whiff_brk_minus_fb LDA held-out CV acc: {summary['axes']['y']['lda_cv_accuracy_mean']:.4f}")
