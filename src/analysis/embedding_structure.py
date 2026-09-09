"""
Embedding structure: does the learned hitter space contain real discrete
groups, and is the map we draw of it honest about exposure?

Two known traps motivate this module. First, embedding vector length
correlates r=0.93 with log prior plate appearances, so an unnormalized 2-D
map is mostly an exposure map. Second, the embedding's effective rank is
about 27.6 of 32, so any 2-D projection discards most of the space and
apparent clusters can be projection artifact. This module (1) normalizes
away length before projecting, (2) clusters in the full 32-D space rather
than on 2-D coordinates, (3) tests neighbour purity against a permutation
null instead of eyeballing color patches, and (4) draws attribute-colored
maps so a real lefty-power group would show as co-located color across
panels.

Batter handedness, power, and contact tendency are CONTEXT features fed to
the model's scorer, never to the hitter embedding lookup (see
src/features/context_features.py). The embedding sees only batter index and
pitch outcomes, so any handedness or power structure found here is learned,
not circular.

Reads frozen checkpoints and existing results only. No retraining, no model
or pipeline changes.

Run: python -m src.analysis.embedding_structure --out-dir results/embedding_structure
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances

from src.analysis.model_visualization_arm_agreement import seed_mean_embedding
from src.data.eval_targets import (  # the pitcher rule has ONE definition, in the data layer
    CAREER_TWO_WAY_MIN_PA, PITCHER_MIN_BATTERS_FACED, career_pitcher_batters)

DEFAULT_OUT_DIR = "results/embedding_structure"
DEFAULT_CHECKPOINT_DIR = "results/checkpoints"
DEFAULT_ARM = "embedding_sgd_sgd_lr1"
HITTER_STATS_PATH = "results/model_visualization/hitter_stats.csv"
NAMES_PATH = "data/processed/hitter_names.csv"
EVAL_TARGETS_PA_PATH = "data/processed/eval_targets_pa.parquet"


SEEDS = (0, 1, 2, 3, 4)
BOOT_SEED = 7
N_PERMUTATIONS = 1000
K_RANGE = range(2, 11)
N_NEIGHBOURS = 10
SCOUTING_COLUMNS = [
    "ev_p90", "pull_rate", "contact_rate", "whiff_rate", "chase_rate",
    "woba_level", "obs_platoon_diff",
]
ATTRIBUTE_MAP_COLUMNS = [
    "stand", "ev_p90", "pull_rate", "contact_rate", "obs_platoon_diff", "woba_level",
]


# --------------------------------------------------------------------- (1) length normalization

def unit_normalize(matrix):
    """Divides each row by its L2 norm, projecting onto the unit sphere."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / norms, norms.ravel()


def norm_exposure_correlation(norms, log_prior_pa):
    return float(np.corrcoef(norms, log_prior_pa)[0, 1])


# --------------------------------------------------------------------- (0) career pitcher-batter filter

def career_pitcher_batter_ids(eval_targets_pa_path=EVAL_TARGETS_PA_PATH,
                              min_batters_faced=PITCHER_MIN_BATTERS_FACED,
                              career_two_way_min_pa=CAREER_TWO_WAY_MIN_PA):
    """
    Batter ids that are real pitchers contaminating the hitter population, at the CAREER
    level rather than the pipeline's per-season level. Thin reader over
    `eval_targets.career_pitcher_batters`, which owns the rule.

    Why a career rule is needed at all: `eval_targets.primarily_pitchers` uses
    TWO_WAY_MIN_PA=50, a PER-SEASON gate, so any batter with >=50 PA in one season is
    exempted as "two-way". That was tuned for Ohtani, but a pre-2022-DH NL starter batted
    50-97 PA/season and is therefore exempted in exactly the seasons he bats most --
    Bumgarner has 306 PA surviving `drop_pitcher_batters`. It also returns (season,
    batter) pairs, and the embedding population has one row per hitter with no season, so
    a season-keyed set cannot filter it.

    This is an ANALYSIS-layer filter. It does not touch the training tensors; the pipeline
    still builds with the season rule, and switching it over is the deferred retrain.

    Yields 1927 ids, 198 of them present in hitter_stats.csv (all `low` stratum).
    Trout, Judge and Ohtani are all retained -- none appears as a `pitcher` facing >=50
    batters in any season.
    """
    return career_pitcher_batters(pd.read_parquet(eval_targets_pa_path),
                                  min_batters_faced=min_batters_faced,
                                  career_two_way_min_pa=career_two_way_min_pa)


# --------------------------------------------------------------------- (2) cluster in 32-D

def silhouette_curve(normalized_matrix, k_range=K_RANGE, seed=BOOT_SEED):
    """
    K-means silhouette score for each k, fit on the full normalized
    32-D embedding (never on a 2-D projection). Returns a list of
    {k, silhouette_score} and the k with the highest score.
    """
    scores = []
    for k in k_range:
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(normalized_matrix)
        score = silhouette_score(normalized_matrix, labels)
        scores.append({"k": k, "silhouette_score": float(score)})
    best_k = max(scores, key=lambda row: row["silhouette_score"])["k"]
    return scores, best_k


def cluster_labels(normalized_matrix, k, seed=BOOT_SEED):
    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(normalized_matrix)


def cluster_profiles(labels, hitters, names_by_batter, n_examples=5):
    """
    Per-cluster n, mean and league-relative z-score of each scouting
    attribute, plus example hitter names. z-scores are relative to the
    full league (all hitters), so a cluster's row says how far its
    average member sits from a typical hitter.
    """
    df = hitters.copy()
    df["cluster"] = labels
    league_mean = df[SCOUTING_COLUMNS].mean()
    league_std = df[SCOUTING_COLUMNS].std(ddof=0)

    rows = []
    for cluster in sorted(df["cluster"].unique()):
        members = df[df["cluster"] == cluster]
        row = {"cluster": int(cluster), "n": int(len(members)),
               "stand_frac_L": float((members["stand"] == "L").mean())}
        for col in SCOUTING_COLUMNS:
            mean_val = members[col].mean()
            row[f"{col}_mean"] = float(mean_val)
            row[f"{col}_z"] = float((mean_val - league_mean[col]) / league_std[col])
        example_batters = members["batter"].head(n_examples).tolist()
        row["example_hitters"] = "; ".join(
            names_by_batter.get(b, str(b)) for b in example_batters)
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- (3) neighbour purity

def nearest_neighbour_indices(normalized_matrix, k=N_NEIGHBOURS):
    """Row indices of each hitter's k nearest neighbours by cosine distance, excluding itself."""
    distances = cosine_distances(normalized_matrix)
    np.fill_diagonal(distances, np.inf)
    return np.argsort(distances, axis=1)[:, :k]


def purity_fraction(neighbour_idx, labels):
    """Fraction of each hitter's neighbours sharing its own label value."""
    labels = np.asarray(labels)
    own = labels[:, None]
    neighbour_labels = labels[neighbour_idx]
    return (neighbour_labels == own).mean(axis=1)


def permutation_null(neighbour_idx, labels, n_permutations=N_PERMUTATIONS, seed=BOOT_SEED):
    """
    Null distribution of mean purity when the attribute is shuffled across
    hitters, holding the neighbour structure fixed. Returns (null_mean,
    ci_lo, ci_hi, null_draws) where the interval is the 95% percentile
    interval of the per-permutation mean purity.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    draws = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = labels[rng.permutation(len(labels))]
        draws[i] = purity_fraction(neighbour_idx, shuffled).mean()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(draws.mean()), float(lo), float(hi), draws


def purity_test(neighbour_idx, labels, n_permutations=N_PERMUTATIONS, seed=BOOT_SEED):
    observed = purity_fraction(neighbour_idx, labels).mean()
    null_mean, lo, hi, draws = permutation_null(neighbour_idx, labels, n_permutations, seed)
    p_value = (np.sum(draws >= observed) + 1) / (n_permutations + 1)
    return {"observed_purity": float(observed), "null_mean": null_mean,
            "null_ci95": [lo, hi], "p_value": float(p_value), "n_permutations": n_permutations}


def tercile_labels(values):
    """Assigns each value to tercile 0/1/2 (low/mid/high) by rank."""
    return pd.qcut(values, 3, labels=False, duplicates="drop")


def neighbour_purity_report(normalized_matrix, hitters):
    """
    Neighbour purity for handedness, power tercile (ev_p90), and contact
    tercile (contact_rate), overall and within each exposure stratum,
    each against its own permutation null.
    """
    neighbour_idx = nearest_neighbour_indices(normalized_matrix)
    power_tercile = tercile_labels(hitters["ev_p90"])
    contact_tercile = tercile_labels(hitters["contact_rate"])

    attribute_labels = {
        "stand": hitters["stand"].values,
        "power_tercile": power_tercile.values,
        "contact_tercile": contact_tercile.values,
    }

    rows = []
    for attribute_name, labels in attribute_labels.items():
        result = purity_test(neighbour_idx, labels)
        result.update({"attribute": attribute_name, "stratum": "all"})
        rows.append(result)
        for stratum in sorted(hitters["stratum"].unique()):
            mask = (hitters["stratum"] == stratum).values
            stratum_idx_map = {old: new for new, old in enumerate(np.where(mask)[0])}
            sub_matrix = normalized_matrix[mask]
            sub_neighbour_idx = nearest_neighbour_indices(sub_matrix)
            sub_labels = labels[mask]
            if len(np.unique(sub_labels)) < 2 or len(sub_labels) <= N_NEIGHBOURS:
                continue
            sub_result = purity_test(sub_neighbour_idx, sub_labels)
            sub_result.update({"attribute": attribute_name, "stratum": stratum})
            rows.append(sub_result)
    return pd.DataFrame(rows), neighbour_idx


# --------------------------------------------------------------------- projections

def project(normalized_matrix, seed=BOOT_SEED):
    pca = PCA(n_components=2, random_state=seed)
    pca_coords = pca.fit_transform(normalized_matrix)
    tsne_coords = TSNE(n_components=2, perplexity=30, random_state=seed, init="pca").fit_transform(normalized_matrix)
    variance_retained = float(pca.explained_variance_ratio_[:2].sum())
    return pca_coords, tsne_coords, variance_retained


# --------------------------------------------------------------------- figures

def fig_clusters(pca_coords, tsne_coords, labels, variance_retained, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, coords, title in zip(
        axes,
        [pca_coords, tsne_coords],
        [f"PCA (retains {variance_retained:.1%} of 32-D variance)", "t-SNE"],
    ):
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=8, alpha=0.6)
        ax.set_xlabel(f"{title.split(' ')[0].lower()} dim 1")
        ax.set_ylabel(f"{title.split(' ')[0].lower()} dim 2")
        ax.set_title(title)
        legend = ax.legend(*scatter.legend_elements(), title="cluster (fit in 32-D)", fontsize=7, loc="best")
        ax.add_artist(legend)
    fig.suptitle("Hitter embedding clusters: fit on normalized 32-D vectors, shown in 2-D")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_silhouette(scores, best_k, path):
    ks = [row["k"] for row in scores]
    values = [row["silhouette_score"] for row in scores]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(ks, values, marker="o")
    ax.axvline(best_k, color="red", linestyle="--", alpha=0.6, label=f"chosen k={best_k}")
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("silhouette score")
    ax.set_title("Silhouette score by k, k-means on normalized 32-D embedding")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_attribute_maps(pca_coords, hitters, variance_retained, path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, column in zip(axes.ravel(), ATTRIBUTE_MAP_COLUMNS):
        if column == "stand":
            for stand, color in {"L": "#4c8dff", "R": "#e0574a", "S": "#7a7a7a"}.items():
                mask = hitters["stand"] == stand
                ax.scatter(pca_coords[mask.values, 0], pca_coords[mask.values, 1],
                           s=6, alpha=0.5, color=color, label=stand)
            ax.legend(fontsize=7, title="stand")
        else:
            # Clip the color scale to the 2nd-98th percentile per panel so a
            # few outliers don't compress everyone else into uniform teal.
            # Presentation only -- doesn't change any conclusion (2-D still
            # retains only ~18% of the 32-D variance either way).
            vmin, vmax = np.percentile(hitters[column], [2, 98])
            scatter = ax.scatter(pca_coords[:, 0], pca_coords[:, 1], c=hitters[column],
                                 cmap="viridis", s=6, alpha=0.6, vmin=vmin, vmax=vmax)
            fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, extend="both")
        ax.set_xlabel("pca dim 1")
        ax.set_ylabel("pca dim 2")
        ax.set_title(column)
    fig.suptitle(f"Attribute-colored PCA maps (retains {variance_retained:.1%} of 32-D variance)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------- pipeline

def load_hitters(checkpoint_dir, arm, hitter_stats_path, names_path, seeds=SEEDS,
                  eval_targets_pa_path=EVAL_TARGETS_PA_PATH, exclude_pitcher_batters=True):
    """
    Loads the seed-mean 32-D embedding aligned to per-hitter attributes.
    Rows are ordered by embedding_index (1..n), matching seed_mean_embedding's
    output. Drops hitters missing a scouting attribute this module needs, and
    (by default) drops career pitcher-batters per career_pitcher_batter_ids —
    see that function's docstring for why the pipeline's own filter misses
    them.
    """
    hitters = pd.read_csv(hitter_stats_path).sort_values("embedding_index").reset_index(drop=True)
    names = pd.read_csv(names_path)
    names_by_batter = dict(zip(names["batter"], names["name"]))

    required = SCOUTING_COLUMNS + ["stand", "stratum", "log_prior_pa"]
    keep_mask = hitters[required].notna().all(axis=1)
    hitters = hitters[keep_mask].reset_index(drop=True)

    if exclude_pitcher_batters:
        pitcher_ids = career_pitcher_batter_ids(eval_targets_pa_path)
        hitters = hitters[~hitters["batter"].isin(pitcher_ids)].reset_index(drop=True)

    embedding = seed_mean_embedding(checkpoint_dir, arm, seeds=seeds)
    rows = hitters["embedding_index"].to_numpy() - 1
    embedding = embedding[rows]

    return embedding, hitters, names_by_batter


def run(checkpoint_dir, arm, hitter_stats_path, names_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    embedding, hitters, names_by_batter = load_hitters(
        checkpoint_dir, arm, hitter_stats_path, names_path)

    raw_norm_correlation = norm_exposure_correlation(
        np.linalg.norm(embedding, axis=1), hitters["log_prior_pa"].to_numpy())
    normalized, norms = unit_normalize(embedding)

    scores, best_k = silhouette_curve(normalized)
    labels = cluster_labels(normalized, best_k)
    profiles = cluster_profiles(labels, hitters, names_by_batter)
    profiles.to_csv(out_dir / "cluster_profiles.csv", index=False)

    purity, _ = neighbour_purity_report(normalized, hitters)
    purity.to_csv(out_dir / "neighbour_purity.csv", index=False)

    pca_coords, tsne_coords, variance_retained = project(normalized)
    # After normalization every row has norm 1.0 by construction (no longer
    # correlated with exposure, trivially). The meaningful post-normalization
    # check is whether the 2-D map's own coordinates still track exposure.
    pc1_exposure_correlation = norm_exposure_correlation(
        pca_coords[:, 0], hitters["log_prior_pa"].to_numpy())
    fig_clusters(pca_coords, tsne_coords, labels, variance_retained, out_dir / "fig_clusters_2d.png")
    fig_silhouette(scores, best_k, out_dir / "fig_silhouette.png")
    fig_attribute_maps(pca_coords, hitters, variance_retained, out_dir / "fig_attribute_maps.png")

    summary = {
        "n_hitters": int(len(hitters)),
        "norm_vs_log_prior_pa_before_normalization": raw_norm_correlation,
        "normalized_row_norm_min_max": [float(np.linalg.norm(normalized, axis=1).min()),
                                         float(np.linalg.norm(normalized, axis=1).max())],
        "pca_pc1_vs_log_prior_pa_after_normalization": pc1_exposure_correlation,
        "silhouette_curve": scores,
        "chosen_k": int(best_k),
        "pca_variance_retained_2d": variance_retained,
        "neighbour_purity": purity.to_dict(orient="records"),
    }
    with open(out_dir / "embedding_structure.json", "w") as f:
        json.dump(summary, f, indent=2)

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
    print(f"norm-exposure correlation before normalization: {summary['norm_vs_log_prior_pa_before_normalization']:.4f}")
    print(f"pca pc1-exposure correlation after normalization: {summary['pca_pc1_vs_log_prior_pa_after_normalization']:.4f}")
    print(f"chosen k: {summary['chosen_k']}")
