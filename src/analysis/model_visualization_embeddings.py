"""
Phase V — embedding visualization: V.11 (seed stability gate), V.1 (stand map),
V.2 (exposure map), V.6 (cold-start), V.7 (dimension usage).

Reads only `embedding.weight` from the frozen `d10_baseline_s{0..4}` checkpoints
(spec §0.4: no model, scorer, or loss changes). Seed 0 draws every figure; all
five seeds feed V.11 only (spec §0.2). Purpose is how the model learned, not how
it performed (spec §0.1) — nothing here grades against 2025 or a baseline.

Run: python -m src.analysis.model_visualization_embeddings --out-dir results/model_visualization
"""

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.analysis.model_evaluation_probe_coverage import load_seed_embeddings
from src.analysis.model_visualization_stats import ANCHORS

DEFAULT_OUT_DIR = "results/model_visualization"
DEFAULT_CHECKPOINT_DIR = "results/checkpoints"
DEFAULT_ARM = "d10_baseline"
LOG_DIR = "results/model_v1/logs"
LOG_ARM = "rebuild_baseline"  # the pre-rename slug the logs were written under (spec §0.2)
NAMES_PATH = "data/processed/hitter_names.csv"

SEEDS = (0, 1, 2, 3, 4)
EPOCH_TOL = 1e-5
ANCHOR_IDS = list(ANCHORS)  # Trout, Soto, Pederson, Bohm, Schwarber
N_BOOT = 1000
BOOT_SEED = 7
STAND_COLORS = {"L": "#4c8dff", "R": "#e0574a", "S": "#7a7a7a"}


# --------------------------------------------------------------------- shared

def bootstrap_ci(values, statistic, n_boot=N_BOOT, seed=BOOT_SEED):
    """
    Percentile bootstrap interval for `statistic(values)`, resampling rows of
    `values` (an array or tuple of same-length arrays) with replacement.
    Returns (point_estimate, lo, hi) at the 95% level.
    """
    rng = np.random.default_rng(seed)
    arrays = values if isinstance(values, tuple) else (values,)
    n = len(arrays[0])
    point = statistic(*arrays)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = statistic(*(a[idx] for a in arrays))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)


# --------------------------------------------------------------------- V.11 (a)

def check_provenance(checkpoint_dir, arm, out_dir):
    """
    Parses each seed's "best val loss X at epoch N" line from its training log
    and asserts it matches the checkpoint's saved epoch/val_loss (tolerance
    1e-5). A mismatch means the wrong checkpoint is being read and everything
    downstream is meaningless, so this is a hard assertion, not a soft check.
    """
    pattern = re.compile(r"best val loss ([\d.]+) at epoch (\d+)")
    embeddings = load_seed_embeddings(checkpoint_dir, arm, seeds=SEEDS)
    rows = []
    for seed in SEEDS:
        log_path = Path(LOG_DIR) / f"{LOG_ARM}_s{seed}.log"
        text = log_path.read_text()
        match = pattern.search(text)
        assert match, f"no 'best val loss' line found in {log_path}"
        log_val_loss, log_epoch = float(match.group(1)), int(match.group(2))

        ckpt_path = Path(checkpoint_dir) / f"{arm}_s{seed}.pt"
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt_epoch, ckpt_val_loss = int(ckpt["epoch"]), float(ckpt["val_loss"])

        assert ckpt_epoch == log_epoch, (
            f"seed {seed}: checkpoint epoch {ckpt_epoch} != log epoch {log_epoch}")
        assert abs(ckpt_val_loss - log_val_loss) <= EPOCH_TOL, (
            f"seed {seed}: checkpoint val_loss {ckpt_val_loss} != log val_loss "
            f"{log_val_loss} (tol {EPOCH_TOL})")
        rows.append({"seed": seed, "log_epoch": log_epoch, "log_val_loss": log_val_loss,
                     "ckpt_epoch": ckpt_epoch, "ckpt_val_loss": ckpt_val_loss, "match": True})

    provenance = pd.DataFrame(rows)
    provenance.to_csv(f"{out_dir}/seed_provenance.csv", index=False)
    return embeddings, provenance


# --------------------------------------------------------------------- V.11 (b)

def procrustes_align(source, target):
    """
    Rotates `source` onto `target` (both n x d, row-matched, mean-centered)
    with the orthogonal Procrustes solution. Returns the aligned source.
    """
    source_c = source - source.mean(axis=0)
    target_c = target - target.mean(axis=0)
    rotation, _ = orthogonal_procrustes(source_c, target_c)
    return source_c @ rotation


def row_cosine(a, b):
    """Row-wise cosine similarity between two same-shape matrices."""
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.sum(a_norm * b_norm, axis=1)


def seed_stability(embeddings, hitters, out_dir):
    """
    Aligns seeds 1-4 to seed 0 (rows 1..1762 only, row 0 is the shared
    cold-start vector and carries no per-hitter identity to align). For each
    seed, also computes a row-shuffled null: the same aligned rows with hitter
    identity permuted, so the null has the real alignment's geometry but no
    correspondence to seed 0's hitters.
    """
    rng = np.random.default_rng(BOOT_SEED)
    seed0_rows = embeddings[0][1:]
    real_cos = np.zeros((len(seed0_rows), 4))
    null_cos = np.zeros((len(seed0_rows), 4))
    seed0_centered = seed0_rows - seed0_rows.mean(axis=0)

    for j, seed in enumerate((1, 2, 3, 4)):
        aligned = procrustes_align(embeddings[seed][1:], seed0_rows)
        real_cos[:, j] = row_cosine(aligned, seed0_centered)
        shuffled = aligned[rng.permutation(len(aligned))]
        null_cos[:, j] = row_cosine(shuffled, seed0_centered)

    stability = hitters.copy()
    stability["mean_cosine_real"] = real_cos.mean(axis=1)
    stability["mean_cosine_null"] = null_cos.mean(axis=1)
    stability.to_csv(f"{out_dir}/seed_stability.csv", index=False)

    real_flat, null_flat = real_cos.mean(axis=1), null_cos.mean(axis=1)
    diff, lo, hi = bootstrap_ci((real_flat, null_flat),
                                lambda r, n: float(np.median(r) - np.median(n)))
    summary = {
        "median_cosine_real": float(np.median(real_flat)),
        "median_cosine_null": float(np.median(null_flat)),
        "median_diff": diff,
        "diff_ci95": [lo, hi],
        "n_boot": N_BOOT,
        "boot_seed": BOOT_SEED,
        "passes": bool(lo > 0 or hi < 0),
    }
    with open(f"{out_dir}/seed_stability_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return stability, summary


def fig_seed_stability(stability, embeddings, hitters, seed0_pca, path):
    fig = plt.figure(figsize=(13, 6))
    ax_hist = fig.add_subplot(2, 3, 1)
    ax_hist.hist(stability["mean_cosine_real"], bins=30, alpha=0.6, label="real", color="#4c8dff")
    ax_hist.hist(stability["mean_cosine_null"], bins=30, alpha=0.6, label="null (shuffled)", color="#999999")
    ax_hist.set_xlabel("mean cosine across seeds 1-4 vs seed 0")
    ax_hist.set_title("seed stability: real vs null")
    ax_hist.legend(fontsize=8)

    seed0_rows = embeddings[0][1:]
    for j, seed in enumerate(SEEDS):
        ax = fig.add_subplot(2, 3, j + 2)
        if seed == 0:
            coords = seed0_pca.transform(seed0_rows - seed0_rows.mean(axis=0))[:, :2]
        else:
            aligned = procrustes_align(embeddings[seed][1:], seed0_rows)
            coords = seed0_pca.transform(aligned)[:, :2]
        for stand, color in STAND_COLORS.items():
            mask = hitters["stand"].values == stand
            ax.scatter(coords[mask, 0], coords[mask, 1], s=4, alpha=0.5, color=color, label=stand)
        ax.set_title(f"seed {seed}")
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.legend(fontsize=7, markerscale=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------- coordinates

def compute_coordinates(embeddings, hitters, out_dir):
    """
    PCA (all 32 components) and 2-D t-SNE on seed 0's trained rows (1..1762,
    mean-centered). Cold-start row 0 is projected through the fitted PCA for
    V.6 but never fit on, since it is one shared vector for many hitters.
    """
    seed0_rows = embeddings[0][1:]
    centered = seed0_rows - seed0_rows.mean(axis=0)

    pca = PCA(n_components=32, random_state=BOOT_SEED)
    pcs = pca.fit_transform(centered)

    tsne = TSNE(n_components=2, perplexity=30, random_state=BOOT_SEED, init="pca")
    tsne_coords = tsne.fit_transform(centered)

    coords = hitters.copy()
    for k in range(5):
        coords[f"pc{k+1}"] = pcs[:, k]
    coords["tsne1"] = tsne_coords[:, 0]
    coords["tsne2"] = tsne_coords[:, 1]
    coords.to_csv(f"{out_dir}/embedding_coords.csv", index=False)

    variance = pd.DataFrame({
        "component": np.arange(1, 33),
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
    })
    variance.to_csv(f"{out_dir}/pca_explained_variance.csv", index=False)

    return coords, pca, seed0_rows


def label_anchors(ax, coords, hitters, x_col, y_col):
    anchors = coords[coords["batter"].isin(ANCHOR_IDS)]
    for _, row in anchors.iterrows():
        ax.scatter([row[x_col]], [row[y_col]], s=40, facecolors="none", edgecolors="black", linewidths=1.2)
        ax.annotate(row["name"], (row[x_col], row[y_col]), fontsize=8, xytext=(4, 4), textcoords="offset points")


# --------------------------------------------------------------------- V.1

def stand_separability(coords_2d, stand):
    """
    Out-of-fold 5-fold logistic regression accuracy of `stand` from 2-D
    coordinates, with a hitter-level bootstrap interval on accuracy.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=BOOT_SEED)
    oof_pred = cross_val_predict(LogisticRegression(max_iter=1000), coords_2d, stand, cv=cv)
    correct = (oof_pred == stand).astype(float).values
    accuracy, lo, hi = bootstrap_ci(correct, lambda c: float(np.mean(c)))
    return {"accuracy": accuracy, "accuracy_ci95": [lo, hi], "n_boot": N_BOOT,
            "boot_seed": BOOT_SEED, "passes": bool(lo > 0.5)}


def fig_stand_map(coords, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, (x_col, y_col, title) in zip(
        axes, [("pc1", "pc2", "PCA (pc1, pc2)"), ("tsne1", "tsne2", "t-SNE")]
    ):
        for stand, color in STAND_COLORS.items():
            mask = coords["stand"] == stand
            ax.scatter(coords.loc[mask, x_col], coords.loc[mask, y_col], s=6, alpha=0.5, color=color, label=stand)
        label_anchors(ax, coords, coords, x_col, y_col)
        ax.set_title(title)
        ax.set_xlabel(x_col); ax.set_ylabel(y_col)
    axes[0].legend(fontsize=8, markerscale=2)
    fig.suptitle("Hitter embedding coloured by batting side")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------- V.2 / V.6

def exposure_loadings(coords, norms, log_prior_pa):
    pc1_r, pc1_lo, pc1_hi = bootstrap_ci(
        (coords["pc1"].values, log_prior_pa.values),
        lambda p, e: float(abs(np.corrcoef(p, e)[0, 1])))
    norm_r, norm_lo, norm_hi = bootstrap_ci(
        (norms, log_prior_pa.values),
        lambda n, e: float(np.corrcoef(n, e)[0, 1]))
    return {
        "pc1_abs_r": pc1_r, "pc1_abs_r_ci95": [pc1_lo, pc1_hi],
        "pc1_note": "pc1 sign is arbitrary; passes uses the |r| interval excluding zero",
        "pc1_passes": bool(pc1_lo > 0),
        "norm_r": norm_r, "norm_r_ci95": [norm_lo, norm_hi],
        "norm_passes": bool(norm_lo > 0),
        "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
    }


def fig_exposure_map(coords, norms, log_prior_pa, cold_start_pc, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    hb = axes[0].hexbin(coords["pc1"], coords["pc2"], C=log_prior_pa,
                        reduce_C_function=np.mean, gridsize=35, cmap="viridis")
    axes[0].scatter([cold_start_pc[0]], [cold_start_pc[1]], marker="*", s=180, color="red",
                    edgecolors="black", label="cold-start row 0")
    axes[0].legend(fontsize=8)
    axes[0].set_title("PCA coloured by log prior PA")
    axes[0].set_xlabel("pc1"); axes[0].set_ylabel("pc2")
    fig.colorbar(hb, ax=axes[0], label="mean log prior PA per cell")

    order = np.argsort(log_prior_pa.values)
    x_sorted, y_sorted = log_prior_pa.values[order], norms[order]
    bins = np.linspace(x_sorted.min(), x_sorted.max(), 16)
    bin_idx = np.digitize(x_sorted, bins)
    binned_x, binned_y = [], []
    for b in range(1, len(bins)):
        mask = bin_idx == b
        if mask.sum() > 0:
            binned_x.append(x_sorted[mask].mean())
            binned_y.append(np.median(y_sorted[mask]))
    axes[1].scatter(log_prior_pa, norms, s=6, alpha=0.3, color="#4c8dff")
    axes[1].plot(binned_x, binned_y, color="#e0574a", linewidth=2, label="binned median")
    axes[1].set_xlabel("log prior PA"); axes[1].set_ylabel("embedding L2 norm")
    axes[1].set_title("norm vs exposure")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def cold_start_summary(embeddings, hitters):
    seed0_rows = embeddings[0][1:]
    row0 = embeddings[0][0]
    weights = hitters["log_prior_pa"].values
    weighted_centroid = np.average(seed0_rows, axis=0, weights=weights)
    unweighted_centroid = seed0_rows.mean(axis=0)

    out = {}
    for name, centroid in [("weighted", weighted_centroid), ("unweighted", unweighted_centroid)]:
        hitter_dists = np.linalg.norm(seed0_rows - centroid, axis=1)
        row0_dist = float(np.linalg.norm(row0 - centroid))
        z = (row0_dist - hitter_dists.mean()) / hitter_dists.std()
        out[name] = {"row0_distance": row0_dist, "hitter_dist_mean": float(hitter_dists.mean()),
                     "hitter_dist_std": float(hitter_dists.std()), "z_score": float(z)}
    return out


# --------------------------------------------------------------------- V.7

def dimension_usage(matrix):
    """
    Per-dim variance, PCA eigenvalues, effective rank (exp of the entropy of
    normalized eigenvalues), and participation ratio ((sum lambda)^2 / sum
    lambda^2). Both collapse to the raw dimension count under isotropic
    variance and to 1 when all variance sits on one direction.
    """
    centered = matrix - matrix.mean(axis=0)
    per_dim_variance = centered.var(axis=0, ddof=1)
    pca = PCA(n_components=min(matrix.shape), random_state=BOOT_SEED)
    pca.fit(centered)
    eigenvalues = pca.explained_variance_
    p = eigenvalues / eigenvalues.sum()
    p_nonzero = p[p > 1e-12]
    entropy = -np.sum(p_nonzero * np.log(p_nonzero))
    effective_rank = float(np.exp(entropy))
    participation_ratio = float(eigenvalues.sum() ** 2 / np.sum(eigenvalues ** 2))
    return per_dim_variance, eigenvalues, effective_rank, participation_ratio


# --------------------------------------------------------------------- hitter table

def load_hitters(hitter_stats_path, names_path, n_trained_rows):
    """
    Merges the exposure/stratum table (`hitter_stats.csv`, built in parallel)
    with the name table into one frame indexed by embedding_index 1..n. Fails
    loudly if either input is missing rather than silently skipping figures.
    """
    missing = [p for p in (hitter_stats_path, names_path) if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "model_visualization_embeddings needs both hitter tables; missing: "
            + ", ".join(missing) + ". Build against a synthetic stand-in for testing; "
            "re-run once these land."
        )
    stats = pd.read_csv(hitter_stats_path)
    names = pd.read_csv(names_path)[["batter", "embedding_index", "name", "stand"]]
    hitters = stats.merge(names, on=["batter", "embedding_index"], suffixes=("", "_names"))
    hitters = hitters.sort_values("embedding_index").reset_index(drop=True)

    expected = set(range(1, n_trained_rows))
    got = set(hitters["embedding_index"])
    assert got == expected, (
        f"hitter table embedding_index does not cover rows 1..{n_trained_rows - 1}: "
        f"missing {sorted(expected - got)[:5]}, extra {sorted(got - expected)[:5]}"
    )
    return hitters


# --------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed-index", type=int, default=0,
                        help="which seed's checkpoint drives the primary figures (V.1/V.2/V.6/V.7)")
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--arm", default=DEFAULT_ARM)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("V.11(a): parsing training logs, asserting against checkpoints")
    embeddings, provenance = check_provenance(args.checkpoint_dir, args.arm, args.out_dir)
    print(provenance.to_string(index=False))

    # seed 0 is always V.11's alignment reference regardless of --seed-index,
    # which only selects which seed drives the coordinate-based figures
    primary = embeddings[args.seed_index]
    hitters = load_hitters(f"{args.out_dir}/hitter_stats.csv", NAMES_PATH, primary.shape[0])

    print("V.11(b): Procrustes alignment and seed stability")
    stability, summary = seed_stability(embeddings, hitters, args.out_dir)
    print(f"median cosine real={summary['median_cosine_real']:.4f} "
          f"null={summary['median_cosine_null']:.4f} passes={summary['passes']}")
    if not summary["passes"]:
        print("WARNING: V.11 stability check failed; V.1-V.5 verdicts are seed-specific (spec table row V.11)")

    print("coordinates: PCA + t-SNE on the primary seed")
    coords, pca, seed0_rows = compute_coordinates({0: primary}, hitters, args.out_dir)
    fig_seed_stability(stability, embeddings, hitters, pca, f"{args.out_dir}/fig_seed_stability.png")

    print("V.1: stand map + separability")
    separability = {
        "pca": stand_separability(coords[["pc1", "pc2"]].values, coords["stand"]),
        "tsne": stand_separability(coords[["tsne1", "tsne2"]].values, coords["stand"]),
    }
    with open(f"{args.out_dir}/stand_separability.json", "w") as f:
        json.dump(separability, f, indent=2)
    fig_stand_map(coords, f"{args.out_dir}/fig_stand_map.png")

    print("V.2/V.6: exposure map, cold start")
    norms = np.linalg.norm(seed0_rows, axis=1)
    loadings = exposure_loadings(coords, norms, hitters["log_prior_pa"])
    with open(f"{args.out_dir}/exposure_loadings.json", "w") as f:
        json.dump(loadings, f, indent=2)

    cold_start = cold_start_summary(embeddings, hitters)
    with open(f"{args.out_dir}/cold_start.json", "w") as f:
        json.dump(cold_start, f, indent=2)

    row0_centered = primary[0] - seed0_rows.mean(axis=0)
    row0_pc = pca.transform(row0_centered.reshape(1, -1))[0, :2]
    fig_exposure_map(coords, norms, hitters["log_prior_pa"], row0_pc, f"{args.out_dir}/fig_exposure_map.png")

    print("V.7: dimension usage")
    per_dim_variance, eigenvalues, effective_rank, participation_ratio = dimension_usage(seed0_rows)
    dim_usage_report = {
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "n_dims": int(seed0_rows.shape[1]),
    }
    with open(f"{args.out_dir}/dimension_usage.json", "w") as f:
        json.dump(dim_usage_report, f, indent=2)

    print(f"wrote figures + CSV/JSON to {args.out_dir}/")


if __name__ == "__main__":
    main()
