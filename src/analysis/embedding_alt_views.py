"""
Alternative visualizations of the hitter embedding, built because the honest
2-D PCA/t-SNE map (see embedding_structure.py) is a featureless blob: no
discrete clusters (silhouette maxes 0.0617 at k=2), and PCA retains only 18%
of 32-D variance.

Plots four axis pairings, all scored against the frozen axis screen
(results/embedding_structure/axis_screen.py — read that file for the exact
metric definitions, min-sample rules, and seed; not re-run here):

- power_contact: ev_p90 (power) vs contact_rate.
- discipline: zone_swing_rate vs chase_rate.
- spin_vs_fastball: whiff rate on breaking balls vs whiff rate on fastballs
  (the two components axis_screen's whiff_brk_minus_fb already derives
  internally, exposed here as standalone metrics -- same classification,
  same >=20-swings-per-group validity mask).
- spin_vs_fastball_damage: ev_p90 on breaking-ball batted balls vs ev_p90 on
  fastball batted balls (the two components axis_screen's ev_brk_minus_fb
  already derives internally, exposed here as standalone metrics -- same
  classification, same >=15-batted-balls-per-group validity mask).

For each axis: tercile the continuous metric, fit LDA on the frozen
embedding to predict the tercile, and use the discriminant score as the
plotted coordinate. Held-out 5-fold CV accuracy is the honest number, since
a supervised projection always looks good on the data it was fit to.

Each pairing renders one 3-panel PNG, all three panels sharing identical
x/y axes:
1. scatter colored by the x-metric's own tercile.
2. scatter colored by the y-metric's own tercile.
3. density-binned gradient over the same plane, colored by mean woba_level
   (observed wOBA) -- a quantity on NEITHER axis, so the color carries real
   information rather than being true by construction.

On the discipline, spin_vs_fastball, and spin_vs_fastball_damage figures
(not power_contact, whose axes are in unlike units), panels 1-2 also draw
the y = x identity line: on those pairings the diagonal is the shared
"easy" component (overall aggressiveness / overall whiff rate / overall
power) and perpendicular distance from it is the per-hitter contrast
(selectivity / spin exposure / damage-vs-spin). The line is drawn in
LDA-score space -- it marks equal standing on the two axes, not equal raw
rates.

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
    "zone_swing_rate": "in-zone swing rate, LDA score",
    "chase_rate": "chase rate (out-of-zone swing), LDA score",
    "whiff_rate_brk": "whiff rate on breaking balls, LDA score",
    "whiff_rate_fb": "whiff rate on fastballs, LDA score",
    "ev_p90_brk": "90th-pct exit velocity vs breaking balls, LDA score",
    "ev_p90_fb": "90th-pct exit velocity vs fastballs, LDA score",
}
METRIC_TITLES = {
    "ev_p90": "Power tercile (ev_p90)",
    "contact_rate": "Contact tercile (contact_rate)",
    "selectivity": "Selectivity tercile (zone swing minus chase)",
    "whiff_brk_minus_fb": "Breaking-minus-fastball whiff tercile",
    "zone_swing_rate": "In-zone swing tercile (zone_swing_rate)",
    "chase_rate": "Chase tercile (chase_rate)",
    "whiff_rate_brk": "Breaking-ball whiff tercile (whiff_rate_brk)",
    "whiff_rate_fb": "Fastball whiff tercile (whiff_rate_fb)",
    "ev_p90_brk": "Breaking-ball exit-velocity tercile (ev_p90_brk)",
    "ev_p90_fb": "Fastball exit-velocity tercile (ev_p90_fb)",
}
METRIC_UNITS = {
    "ev_p90": "mph",
    "contact_rate": "rate (0-1)",
    "selectivity": "rate (0-1)",
    "whiff_brk_minus_fb": "rate (0-1)",
    "zone_swing_rate": "rate (0-1)",
    "chase_rate": "rate (0-1)",
    "whiff_rate_brk": "rate (0-1)",
    "whiff_rate_fb": "rate (0-1)",
    "ev_p90_brk": "mph",
    "ev_p90_fb": "mph",
}
N_HEATMAP_BINS = 12
MIN_HITTERS_PER_BIN = 5

# id -> (x_metric, y_metric, expected_x_cv, expected_y_cv). None expected
# means "unmeasured -- just report it" (spin_vs_fastball). x/y metrics come
# straight from axis_screen (a MARGINAL_COLUMNS hitters column, a
# build_contrasts() entry, or the local whiff_rate_brk/fb split below) --
# reused exactly, not re-derived.
AXIS_PAIRINGS = [
    ("power_contact", "ev_p90", "contact_rate", 0.800, 0.765),
    ("discipline", "zone_swing_rate", "chase_rate", 0.782, 0.752),
    ("spin_vs_fastball", "whiff_rate_brk", "whiff_rate_fb", None, None),
    ("spin_vs_fastball_damage", "ev_p90_brk", "ev_p90_fb", None, None),
]
# Pairings whose two axes are on comparable/aligned scales (both LDA scores
# of "aggressiveness"/"whiff" style metrics) -- these get the y = x identity
# line on their scatter panels. power_contact's axes are unlike units
# (power vs contact) so a diagonal there would be meaningless.
IDENTITY_LINE_PAIRINGS = {"discipline", "spin_vs_fastball", "spin_vs_fastball_damage"}
CV_TOLERANCE = 0.02

# Extra metric definitions beyond axis_screen.DEFINITIONS, for pairings that
# use whiff_rate_brk/whiff_rate_fb (not in axis_screen -- those two are
# exposed locally, see whiff_rate_components_by_batter below).
LOCAL_DEFINITIONS = {
    "whiff_rate_brk": (
        "Per-hitter whiff rate against breaking balls, over swings.",
        "Component of axis_screen's whiff_brk_minus_fb, exposed standalone; "
        "same PITCH_GROUPS classification and min-20-swings-per-family rule."),
    "whiff_rate_fb": (
        "Per-hitter whiff rate against fastballs, over swings.",
        "Component of axis_screen's whiff_brk_minus_fb, exposed standalone; "
        "same PITCH_GROUPS classification and min-20-swings-per-family rule."),
    "ev_p90_brk": (
        "Per-hitter 90th-percentile exit velocity on breaking-ball batted balls.",
        "Component of axis_screen's ev_brk_minus_fb, exposed standalone; "
        "same PITCH_GROUPS classification and min-15-batted-balls-per-family rule."),
    "ev_p90_fb": (
        "Per-hitter 90th-percentile exit velocity on fastball batted balls.",
        "Component of axis_screen's ev_brk_minus_fb, exposed standalone; "
        "same PITCH_GROUPS classification and min-15-batted-balls-per-family rule."),
}


# --------------------------------------------------------------------- pitch-level feature
# NOTE: axis_screen.build_contrasts imports this function from this module
# (lazily, inside the function) — keep name/signature stable.

def _whiff_rates_by_pitch_group(pitch_events_path):
    """Shared groundwork for whiff_brk_minus_fb_by_batter and
    whiff_rate_components_by_batter: per-batter x pitch_group whiff rate and
    swing count, using the frozen PITCH_GROUPS classification. whiff =
    swing & not contact."""
    df = pd.read_parquet(pitch_events_path, columns=["batter", "pitch_type", "swing", "contact"])
    df = df[df["swing"] == 1]
    group = pd.Series(np.nan, index=df.index, dtype=object)
    for name, codes in PITCH_GROUPS.items():
        group[df["pitch_type"].isin(codes)] = name
    df = df.assign(pitch_group=group)
    df = df[df["pitch_group"].isin(PITCH_GROUPS)]
    df = df.assign(whiff=(df["contact"] == 0).astype(float))
    counts = df.groupby(["batter", "pitch_group"]).size().unstack(fill_value=0)
    rates = df.groupby(["batter", "pitch_group"])["whiff"].mean().unstack("pitch_group")
    return rates, counts


def whiff_brk_minus_fb_by_batter(pitch_events_path=PITCH_EVENTS_PATH):
    """Per-batter (whiff rate on breaking pitches) minus (whiff rate on
    fastballs), computed from swing-level pitch_events_labeled.parquet.
    whiff = swing & not contact. Descriptive-only feature (not a training
    target), so none of the leakage-window discipline in baseline_ladder_gbm
    applies. Returns a Series indexed by batter."""
    rates, _ = _whiff_rates_by_pitch_group(pitch_events_path)
    return (rates["breaking"] - rates["fastball"]).rename("whiff_brk_minus_fb")


def whiff_rate_components_by_batter(pitch_events_path=PITCH_EVENTS_PATH):
    """whiff_rate_brk and whiff_rate_fb: the two component rates
    whiff_brk_minus_fb_by_batter already derives internally, exposed as
    standalone metrics -- same PITCH_GROUPS classification, and the same
    >=20-swings-per-group (axis_screen.MIN_SWINGS_PER_SLICE) validity mask
    axis_screen.build_contrasts applies to whiff_brk_minus_fb. Returns
    (brk_series, fb_series), both indexed by batter, restricted to hitters
    valid in BOTH groups."""
    rates, counts = _whiff_rates_by_pitch_group(pitch_events_path)
    valid = (counts.get("fastball", 0) >= axis_screen.MIN_SWINGS_PER_SLICE) & \
            (counts.get("breaking", 0) >= axis_screen.MIN_SWINGS_PER_SLICE)
    valid_idx = valid[valid].index
    brk = rates["breaking"].reindex(valid_idx).rename("whiff_rate_brk")
    fb = rates["fastball"].reindex(valid_idx).rename("whiff_rate_fb")
    return brk, fb


def ev_p90_components_by_batter(pitch_events_path=PITCH_EVENTS_PATH):
    """ev_p90_brk and ev_p90_fb: per-hitter 90th-percentile exit velocity on
    breaking-ball and fastball batted balls -- the two component levels
    axis_screen's ev_brk_minus_fb already derives internally (mask_brk /
    mask_fb over swings, batted balls only), exposed as standalone metrics.
    Same PITCH_GROUPS classification and the same
    >=15-batted-balls-per-group validity rule
    (axis_screen.MIN_BATTED_BALLS_PER_SLICE) axis_screen.build_contrasts
    applies to ev_brk_minus_fb. Returns (brk_series, fb_series), both
    indexed by batter, restricted to hitters valid in BOTH groups."""
    df = pd.read_parquet(pitch_events_path, columns=["batter", "pitch_type", "swing", "ev"])
    df = df[(df["swing"] == 1) & df["ev"].notna()]
    group = pd.Series(np.nan, index=df.index, dtype=object)
    for name, codes in PITCH_GROUPS.items():
        group[df["pitch_type"].isin(codes)] = name
    df = df.assign(pitch_group=group)
    df = df[df["pitch_group"].isin(PITCH_GROUPS)]
    counts = df.groupby(["batter", "pitch_group"]).size().unstack(fill_value=0)
    p90 = df.groupby(["batter", "pitch_group"])["ev"].quantile(0.9).unstack("pitch_group")
    valid = (counts.get("fastball", 0) >= axis_screen.MIN_BATTED_BALLS_PER_SLICE) & \
            (counts.get("breaking", 0) >= axis_screen.MIN_BATTED_BALLS_PER_SLICE)
    valid_idx = valid[valid].index
    brk = p90["breaking"].reindex(valid_idx).rename("ev_p90_brk")
    fb = p90["fastball"].reindex(valid_idx).rename("ev_p90_fb")
    return brk, fb


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

    # expected_*_cv of None means "unmeasured -- just report it", not a
    # frozen expectation to check against.
    x_drift = expected_x_cv is not None and abs(x_cv_mean - expected_x_cv) > CV_TOLERANCE
    y_drift = expected_y_cv is not None and abs(y_cv_mean - expected_y_cv) > CV_TOLERANCE
    if x_drift or y_drift:
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

    definitions = {**axis_screen.DEFINITIONS, **LOCAL_DEFINITIONS}
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
        # The two LDA axes are directions in one low-dimensional learned space,
        # so they share its dominant variance. Reporting the projected
        # correlation beside the hitter-level one keeps the figure's visible
        # tilt separable from the tilt the hitters actually have.
        "axis_correlation": {
            "lda_projected": float(np.corrcoef(x_proj, y_proj)[0, 1]),
            "raw_metric": float(np.corrcoef(x_plot_vals, y_plot_vals)[0, 1]),
        },
    }
    return summary, x_proj, y_proj, x_tercile_plot, y_tercile_plot, woba_plot_vals, x_span, y_span


def run_all_pairings(normalized, hitters, out_dir):
    """Renders each AXIS_PAIRINGS entry as ONE 3-panel PNG (projection x
    tercile, projection y tercile, woba gradient) sharing identical x/y axes
    across all three panels."""
    figures_dir = Path(out_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    contrasts = axis_screen.build_contrasts(hitters)
    whiff_brk, whiff_fb = whiff_rate_components_by_batter()
    contrasts["whiff_rate_brk"] = whiff_brk
    contrasts["whiff_rate_fb"] = whiff_fb
    ev_p90_brk, ev_p90_fb = ev_p90_components_by_batter()
    contrasts["ev_p90_brk"] = ev_p90_brk
    contrasts["ev_p90_fb"] = ev_p90_fb

    pairings_summary = {}
    for pairing_id, x_metric, y_metric, exp_x, exp_y in AXIS_PAIRINGS:
        (summary, x_proj, y_proj, x_tercile_plot, y_tercile_plot, woba_plot_vals,
         x_span, y_span) = run_pairing(
            pairing_id, x_metric, y_metric, exp_x, exp_y, normalized, hitters,
            contrasts)

        xlim, ylim = _padded_lim(x_proj), _padded_lim(y_proj)
        palette = {0: "#d9d9d9", 1: "#8fb3ff", 2: "#1f4fd1"}
        tercile_names = ["low", "mid", "high"]
        draw_identity = pairing_id in IDENTITY_LINE_PAIRINGS

        fig, axes = plt.subplots(1, 3, figsize=(19.5, 5.8))
        panels = [
            (axes[0], np.asarray(x_tercile_plot), summary["x"]["lda_cv_accuracy_mean"], METRIC_TITLES[x_metric]),
            (axes[1], np.asarray(y_tercile_plot), summary["y"]["lda_cv_accuracy_mean"], METRIC_TITLES[y_metric]),
        ]
        for ax, tercile, cv_mean, title in panels:
            for t, color in palette.items():
                mask = tercile == t
                ax.scatter(x_proj[mask], y_proj[mask], s=8, alpha=0.6, color=color,
                           label=tercile_names[t], zorder=2)
            if draw_identity:
                lo, hi = max(xlim[0], ylim[0]), min(xlim[1], ylim[1])
                ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="#999999",
                        zorder=1, label="y = x")
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
        if draw_identity:
            fig.text(0.5, 0.01,
                "Identity line drawn in LDA-score space: marks equal standing on the two axes, "
                "not equal raw rates. Perpendicular distance from it is the per-hitter contrast.",
                ha="center", fontsize=8, color="#555555")
            fig.tight_layout(rect=[0, 0.04, 1, 0.95])
        else:
            fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(figures_dir / f"fig_axes_{pairing_id}.png", dpi=130)
        plt.close(fig)

        pairings_summary[pairing_id] = summary

    return pairings_summary


# --------------------------------------------------------------------- pipeline

def run(checkpoint_dir, arm, hitter_stats_path, names_path, out_dir,
        pitch_events_path=PITCH_EVENTS_PATH):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    embedding, hitters, _ = load_hitters(
        checkpoint_dir, arm, hitter_stats_path, names_path)
    normalized, _ = unit_normalize(embedding)  # positionally aligned to `hitters`

    # NOTE: fig_supervised_projection.png / fig_similarity_heatmap.png
    # (the original ev_p90-vs-whiff_brk_minus_fb figure) are frozen outputs
    # of an earlier version of this pipeline -- left on disk untouched.
    # run() no longer regenerates them; see fig_supervised_projection() /
    # fig_similarity_heatmap() above, still exercised directly by tests.
    pairings_summary = run_all_pairings(normalized, hitters, out_dir)

    summary = {
        "pairings": pairings_summary,
        "method": {
            "seed": BOOT_SEED,
            "n_splits": 5,
            "chance": round(1 / 3, 3),
            "season_filter": "season <= 2024 (inherited from hitters/contrasts)",
            "source_screen": "results/embedding_structure/axis_screen.py",
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
    for pairing_id, s in summary["pairings"].items():
        print(f"{pairing_id}: {s['x']['metric']}={s['x']['lda_cv_accuracy_mean']:.4f}, "
              f"{s['y']['metric']}={s['y']['lda_cv_accuracy_mean']:.4f} "
              f"(n_plotted={s['n_hitters_plotted']})")
