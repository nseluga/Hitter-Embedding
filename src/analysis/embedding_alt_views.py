"""
Alternative visualizations of the hitter embedding, built because the honest
2-D PCA/t-SNE map (see embedding_structure.py) is a featureless blob: no
discrete clusters (silhouette maxes 0.0617 at k=2), and PCA retains only 18%
of 32-D variance. Local structure IS real (10-NN purity beats permutation
null for handedness/power/contact), so these three views try to show that
structure honestly instead of forcing it into an unsupervised 2-D scatter.

Rough drafts: real data, correct method, readable output, not polished.

1. Cluster-ordered cosine similarity heatmap on a high-exposure subset.
2. Archetype coordinates: cosine similarity to a small set of reproducibly
   chosen "pole" hitters.
3. Supervised projections (LDA, PLS) with held-out separation/R^2, contrasted
   with PCA's unsupervised 18%.

Every figure also reports whether known pitcher-batters (Colon, Bumgarner,
Greinke — identified by name match against data/processed/hitter_names.csv,
not a systematic pitcher classifier) sit at an extreme and could be driving
the visible pattern.

Reads frozen checkpoints and existing results only. No retraining.

Run: python -m src.analysis.embedding_alt_views --out-dir results/embedding_structure
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold

from src.analysis.embedding_structure import (
    DEFAULT_ARM, DEFAULT_CHECKPOINT_DIR, DEFAULT_OUT_DIR, HITTER_STATS_PATH,
    NAMES_PATH, BOOT_SEED, load_hitters, tercile_labels, unit_normalize,
)

PITCHER_BATTER_NAMES = ("Bartolo Col", "Madison Bumgarner", "Zack Greinke")
HEATMAP_SUBSET_N = 300
STAND_COLORS = {"L": "#4c8dff", "R": "#e0574a", "S": "#7a7a7a"}


def flag_pitcher_batters(hitters, names_by_batter):
    """True for hitters whose name matches a known pitcher-batter. Matched
    by name substring against the three players named in the task brief
    (Colon has an accent in the source file, hence the truncated match) —
    not a general pitcher classifier."""
    names = hitters["batter"].map(names_by_batter).fillna("")
    return names.str.contains("|".join(PITCHER_BATTER_NAMES), case=False, regex=True).values


# --------------------------------------------------------------------- (1) similarity heatmap

def similarity_heatmap_subset(normalized, hitters, n=HEATMAP_SUBSET_N):
    """Top-n hitters by log_prior_pa (exposure) — the readable-pixel-count
    rule stated in the brief. Returns (sub_normalized, sub_hitters)."""
    order = hitters["log_prior_pa"].to_numpy().argsort()[::-1][:n]
    order = np.sort(order)
    return normalized[order], hitters.iloc[order].reset_index(drop=True)


def cluster_order(sub_normalized, method="average"):
    """Leaf order from hierarchical clustering on cosine distance."""
    sim = cosine_similarity(sub_normalized)
    dist = np.clip(1 - sim, 0, None)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    z = linkage(condensed, method=method)
    return leaves_list(z), sim


def fig_similarity_heatmap(sim, order, sub_hitters, path):
    ordered_sim = sim[np.ix_(order, order)]
    ordered_hitters = sub_hitters.iloc[order].reset_index(drop=True)
    power_tercile = tercile_labels(sub_hitters["ev_p90"]).values[order]

    fig = plt.figure(figsize=(9.5, 9))
    grid = fig.add_gridspec(2, 2, width_ratios=[40, 1], height_ratios=[1, 40],
                             wspace=0.03, hspace=0.03)
    ax_top = fig.add_subplot(grid[0, 0])
    ax_main = fig.add_subplot(grid[1, 0])
    ax_right = fig.add_subplot(grid[1, 1])

    im = ax_main.imshow(ordered_sim, cmap="viridis", vmin=-0.2, vmax=1.0, aspect="auto")
    ax_main.set_xlabel("hitters, hierarchical-cluster order")
    ax_main.set_ylabel("hitters, same order")
    fig.colorbar(im, ax=ax_main, fraction=0.03, pad=0.06, label="cosine similarity")

    import matplotlib.colors as mcolors
    stand_rgb = np.array([mcolors.to_rgb(STAND_COLORS.get(s, "#000000"))
                           for s in ordered_hitters["stand"]])
    ax_top.imshow(stand_rgb[np.newaxis, :, :], aspect="auto")
    ax_top.set_xticks([]); ax_top.set_yticks([])
    ax_top.set_title(f"cluster-ordered cosine similarity, top {len(sub_hitters)} hitters by exposure "
                      f"(top strip: stand; right strip: power tercile)", fontsize=10)

    power_palette = ["#d9d9d9", "#8fb3ff", "#1f4fd1"]
    power_rgb = np.array([mcolors.to_rgb(power_palette[t]) for t in power_tercile])
    ax_right.imshow(power_rgb[:, np.newaxis, :], aspect="auto")
    ax_right.set_xticks([]); ax_right.set_yticks([])

    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------- (2) archetype coordinates

def pick_poles(hitters, min_stratum=("medium", "high")):
    """
    Reproducible pole-selection rule, restricted to hitters in the medium/high
    exposure strata (single-extreme values in the low stratum are noisy —
    see the exposure-confound note in embedding_structure.py):
      - power pole:   max ev_p90
      - contact pole: max contact_rate
      - platoon pole: min |obs_platoon_diff| (most even vs both hands, i.e.
        least platoon-split — "opposite-hand-friendly" read as hitters who
        don't lean on facing one hand)
    Returns dict of pole name -> row index into the full `hitters` frame.
    """
    reliable = hitters[hitters["stratum"].isin(min_stratum)]
    poles = {
        "power": reliable["ev_p90"].idxmax(),
        "contact": reliable["contact_rate"].idxmax(),
        "platoon_neutral": reliable["obs_platoon_diff"].abs().idxmin(),
    }
    return poles


def archetype_coordinates(normalized, poles):
    """Cosine similarity of every hitter to each pole vector, plus the
    pairwise pole-to-pole cosine (tells us if the poles are actually
    distinct directions or nearly the same vector)."""
    pole_idx = list(poles.values())
    pole_vecs = normalized[pole_idx]
    sims = cosine_similarity(normalized, pole_vecs)  # (n_hitters, n_poles)
    pole_pole = cosine_similarity(pole_vecs)
    return sims, pole_pole


def ternary_coords(sims):
    """Rescale each pole's similarity column to [0,1] across hitters, then
    normalize each row to sum to 1 for barycentric (ternary) coordinates.
    This is a display convenience, not a claim that similarities are already
    a simplex — cosine similarity can be negative and doesn't sum to 1."""
    rescaled = (sims - sims.min(axis=0)) / (sims.max(axis=0) - sims.min(axis=0) + 1e-12)
    return rescaled / rescaled.sum(axis=1, keepdims=True)


def fig_archetype_coords(bary, hitters, pole_names, path):
    # equilateral-triangle corners for a 3-pole ternary plot
    corners = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
    xy = bary @ corners

    fig, ax = plt.subplots(figsize=(7.5, 7))
    scatter = ax.scatter(xy[:, 0], xy[:, 1], c=hitters["woba_level"], cmap="viridis", s=10, alpha=0.65)
    fig.colorbar(scatter, ax=ax, fraction=0.04, pad=0.03, label="woba_level")
    for corner, name in zip(corners, pole_names):
        ax.scatter(*corner, marker="*", s=400, color="red", edgecolor="black", zorder=5)
        ax.annotate(name, corner, textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=10, weight="bold")
    tri = plt.Polygon(corners, fill=False, edgecolor="gray", linewidth=1)
    ax.add_patch(tri)
    ax.set_title("Archetype coordinates: cosine similarity to 3 poles, ternary layout")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------- (3) supervised projections

def lda_axis_cv(normalized, labels, seed=BOOT_SEED, n_splits=5):
    """K-fold held-out classification accuracy of an LDA fit per-fold (fit
    on train, scored on held-out test) — the honest number, since a
    supervised projection always looks good on the data it was fit to."""
    labels = np.asarray(labels)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for train_idx, test_idx in kf.split(normalized):
        model = LinearDiscriminantAnalysis()
        model.fit(normalized[train_idx], labels[train_idx])
        accs.append(model.score(normalized[test_idx], labels[test_idx]))
    full_model = LinearDiscriminantAnalysis().fit(normalized, labels)
    return full_model, float(np.mean(accs)), float(np.std(accs))


def pls_cv(normalized, target, n_components=2, seed=BOOT_SEED, n_splits=5):
    """K-fold held-out R^2 for a 2-component PLS regression."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    r2s = []
    for train_idx, test_idx in kf.split(normalized):
        model = PLSRegression(n_components=n_components)
        model.fit(normalized[train_idx], target[train_idx])
        r2s.append(model.score(normalized[test_idx], target[test_idx]))
    full_model = PLSRegression(n_components=n_components).fit(normalized, target)
    return full_model, float(np.mean(r2s)), float(np.std(r2s))


def fig_supervised_projection(normalized, hitters, path, seed=BOOT_SEED):
    stand_mask = hitters["stand"].isin(["L", "R"]).values
    stand_labels = hitters["stand"].values[stand_mask]
    power_tercile = tercile_labels(hitters["ev_p90"]).values

    stand_model, stand_cv_mean, stand_cv_std = lda_axis_cv(normalized[stand_mask], stand_labels, seed)
    power_model, power_cv_mean, power_cv_std = lda_axis_cv(normalized, power_tercile, seed)
    pls_model, pls_cv_mean, pls_cv_std = pls_cv(
        normalized, hitters["woba_level"].to_numpy(), n_components=2, seed=seed)

    stand_axis = stand_model.transform(normalized)[:, 0]
    power_axis = power_model.transform(normalized)[:, 0]
    pls_coords = pls_model.transform(normalized)

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.5))

    # Same LDA(hand) x LDA(power) points, shown twice with different color
    # schemes: colored by handedness (axes[0]) and by power tercile
    # (axes[1]). Coloring only by handedness hides the power result (0.813
    # held-out) inside a mixed blob, since power tercile isn't the coloring
    # variable there -- the whole point of this figure is contrasting an
    # axis that separates (power) against one that doesn't (handedness), so
    # both need their own readable coloring.
    ax = axes[0]
    for stand, color in STAND_COLORS.items():
        mask = hitters["stand"] == stand
        ax.scatter(stand_axis[mask.values], power_axis[mask.values], s=8, alpha=0.6, color=color, label=stand)
    ax.set_xlabel(f"LDA(handedness) axis  [5-fold held-out acc {stand_cv_mean:.3f}]")
    ax.set_ylabel(f"LDA(power tercile) axis 1  [5-fold held-out acc {power_cv_mean:.3f}]")
    ax.set_title("Colored by handedness (near-chance axis)")
    ax.legend(fontsize=8, title="stand")

    power_palette = {0: "#d9d9d9", 1: "#8fb3ff", 2: "#1f4fd1"}
    power_tercile_labels = pd.Series(power_tercile, index=hitters.index)
    ax = axes[1]
    for tercile, color in power_palette.items():
        mask = (power_tercile_labels == tercile).values
        ax.scatter(stand_axis[mask], power_axis[mask], s=8, alpha=0.6, color=color,
                   label=["low", "mid", "high"][tercile])
    ax.set_xlabel(f"LDA(handedness) axis  [5-fold held-out acc {stand_cv_mean:.3f}]")
    ax.set_ylabel(f"LDA(power tercile) axis 1  [5-fold held-out acc {power_cv_mean:.3f}]")
    ax.set_title("Colored by power tercile (separating axis)")
    ax.legend(fontsize=8, title="power tercile")

    ax = axes[2]
    scatter = ax.scatter(pls_coords[:, 0], pls_coords[:, 1], c=hitters["woba_level"], cmap="viridis", s=8, alpha=0.65)
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="woba_level")
    ax.set_xlabel("PLS component 1")
    ax.set_ylabel("PLS component 2")
    ax.set_title(f"PLS(woba_level), 2 comp  [5-fold held-out R^2 {pls_cv_mean:.3f}]")

    fig.suptitle("Supervised projections beat PCA's 18% by construction — held-out numbers are the honest check")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)

    return {
        "lda_handedness_cv_accuracy_mean": stand_cv_mean, "lda_handedness_cv_accuracy_std": stand_cv_std,
        "lda_power_tercile_cv_accuracy_mean": power_cv_mean, "lda_power_tercile_cv_accuracy_std": power_cv_std,
        "pls_woba_level_cv_r2_mean": pls_cv_mean, "pls_woba_level_cv_r2_std": pls_cv_std,
        "n_splits": 5, "seed": seed,
    }


# --------------------------------------------------------------------- pitcher-batter contamination check

def pitcher_batter_report(hitters, pitcher_mask, names_by_batter, **positions):
    """For each named coordinate array (e.g. heatmap_order=..., pls_comp1=...),
    report where the pitcher-batters land relative to the full distribution
    (percentile rank), so an extreme position is visible instead of silent."""
    matched_batters = hitters.loc[pitcher_mask, "batter"].tolist()
    report = {"n_pitcher_batters_found": int(pitcher_mask.sum()),
              "pitcher_batter_names": [names_by_batter.get(b, str(b)) for b in matched_batters]}
    for name, values in positions.items():
        values = np.asarray(values, dtype=float)
        ranks = pd.Series(values).rank(pct=True).to_numpy()
        report[name] = {"pitcher_batter_percentiles": ranks[pitcher_mask].round(3).tolist()}
    return report


# --------------------------------------------------------------------- pipeline

def run(checkpoint_dir, arm, hitter_stats_path, names_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    embedding, hitters, names_by_batter = load_hitters(
        checkpoint_dir, arm, hitter_stats_path, names_path)
    normalized, _ = unit_normalize(embedding)
    pitcher_mask = flag_pitcher_batters(hitters, names_by_batter)

    # (1) similarity heatmap
    sub_normalized, sub_hitters = similarity_heatmap_subset(normalized, hitters)
    order, sim = cluster_order(sub_normalized, method="average")
    fig_similarity_heatmap(sim, order, sub_hitters, out_dir / "fig_similarity_heatmap.png")
    sub_pitcher_mask = flag_pitcher_batters(sub_hitters, names_by_batter)
    heatmap_report = {
        "subset_rule": f"top {HEATMAP_SUBSET_N} hitters by log_prior_pa (exposure)",
        "linkage_method": "average", "distance": "cosine",
        "n_subset": int(len(sub_hitters)),
        "pitcher_batters_in_subset": pitcher_batter_report(
            sub_hitters, sub_pitcher_mask, names_by_batter,
            heatmap_leaf_position=np.argsort(order)) if sub_pitcher_mask.any() else
            {"n_pitcher_batters_found": 0},
    }

    # (2) archetype coordinates
    poles = pick_poles(hitters)
    sims, pole_pole = archetype_coordinates(normalized, poles)
    bary = ternary_coords(sims)
    pole_names = list(poles.keys())
    fig_archetype_coords(bary, hitters, pole_names, out_dir / "fig_archetype_coords.png")
    archetype_report = {
        "pole_selection_rule": "within stratum in {medium, high}: power=argmax ev_p90, "
                                "contact=argmax contact_rate, platoon_neutral=argmin |obs_platoon_diff|",
        "pole_hitters": {name: {"batter": int(hitters.loc[idx, "batter"]),
                                 "name": names_by_batter.get(int(hitters.loc[idx, "batter"]), "?")}
                          for name, idx in poles.items()},
        "pole_pole_cosine": {f"{a}_vs_{b}": float(pole_pole[i, j])
                              for i, a in enumerate(pole_names) for j, b in enumerate(pole_names) if i < j},
        "pitcher_batters": pitcher_batter_report(
            hitters, pitcher_mask, names_by_batter,
            cosine_to_power_pole=sims[:, pole_names.index("power")],
            cosine_to_contact_pole=sims[:, pole_names.index("contact")],
        ) if pitcher_mask.any() else {"n_pitcher_batters_found": 0},
    }
    max_pole_pole = max(archetype_report["pole_pole_cosine"].values())
    archetype_report["poles_degenerate"] = bool(max_pole_pole > 0.9)

    # (3) supervised projection
    sup_report = fig_supervised_projection(normalized, hitters, out_dir / "fig_supervised_projection.png")
    pls_model = PLSRegression(n_components=2).fit(normalized, hitters["woba_level"].to_numpy())
    pls_coords_full = pls_model.transform(normalized)
    sup_report["pitcher_batters"] = pitcher_batter_report(
        hitters, pitcher_mask, names_by_batter, pls_component_1=pls_coords_full[:, 0]) if pitcher_mask.any() else {
        "n_pitcher_batters_found": 0}

    summary = {
        "n_hitters": int(len(hitters)),
        "similarity_heatmap": heatmap_report,
        "archetype_coordinates": archetype_report,
        "supervised_projection": sup_report,
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
    print(f"pole-pole cosine (max pair): {max(summary['archetype_coordinates']['pole_pole_cosine'].values()):.4f}, "
          f"degenerate={summary['archetype_coordinates']['poles_degenerate']}")
    print(f"LDA(handedness) held-out acc: {summary['supervised_projection']['lda_handedness_cv_accuracy_mean']:.4f}")
    print(f"LDA(power tercile) held-out acc: {summary['supervised_projection']['lda_power_tercile_cv_accuracy_mean']:.4f}")
    print(f"PLS(woba_level) held-out R^2: {summary['supervised_projection']['pls_woba_level_cv_r2_mean']:.4f}")
