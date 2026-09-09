"""
Cross-arm agreement: does adding the 2024 season move the learned hitter space?

The paper's representation figures are drawn from the pre-registration arm
(`embedding_sgd_sgd_lr1`, trained 2015-2023, validated on 2024). The headline
claim-1 numbers come from the refit arm (`embedding_sgd_sgd_lr1_final`, trained
2015-2024, no validation season). A reader is owed evidence that the figures
describe the same space the refit occupies.

This module measures that directly. It takes each arm's seed-mean embedding,
matches rows by `batter` id rather than by row index, rotates one arm onto the
other, and reports the per-hitter cosine against a row-shuffled null.

The null shuffles the rows BEFORE the rotation, not after. This matters. An
orthogonal rotation fitted on 1700 rows in 32 dimensions can align two
completely unrelated spaces to a median cosine near 0.11, so a null that
reuses the real rotation credits that fitting to the model. Shuffling first
gives the null rotation the same freedom, and the reported gap is then the
part that only hitter identity can explain.

Matching by `batter` is not optional. The two builds have different hitter
populations (1762 against 1870), so a given row index names a different hitter
in each arm.

Run: python -m src.analysis.model_visualization_arm_agreement
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.model_evaluation_probe_coverage import load_seed_embeddings
from src.analysis.model_visualization_embeddings import (
    bootstrap_ci,
    procrustes_align,
    row_cosine,
)

DEFAULT_OUT_DIR = "results/paper_figures"
DEFAULT_CHECKPOINT_DIR = "results/checkpoints"

REFERENCE_ARM = "embedding_sgd_sgd_lr1"
REFERENCE_STATS = "results/model_visualization/hitter_stats.csv"
COMPARISON_ARM = "embedding_sgd_sgd_lr1_final"
COMPARISON_STATS = "results/model_visualization_final/hitter_stats.csv"

SEEDS = (0, 1, 2, 3, 4)
COLD_START_ROW = 0
N_BOOT = 1000
BOOT_SEED = 7
N_NULL_PERMUTATIONS = 20


def seed_mean_embedding(checkpoint_dir, arm, seeds=SEEDS):
    """
    One embedding matrix per arm, averaged over its seeds.

    Seeds 1-4 are rotated onto seed 0 before averaging, because separately
    seeded runs learn the same space only up to an orthogonal rotation. Row 0 is
    the reserved cold-start row and is dropped: it carries no hitter identity.
    Returns an array of shape (n_hitters, dim) whose row i is embedding index
    i + 1.
    """
    embeddings = load_seed_embeddings(checkpoint_dir, arm, seeds=seeds)
    reference_rows = embeddings[seeds[0]][COLD_START_ROW + 1:]
    aligned_stack = [reference_rows - reference_rows.mean(axis=0)]
    for seed in seeds[1:]:
        aligned_stack.append(
            procrustes_align(embeddings[seed][COLD_START_ROW + 1:], reference_rows))
    return np.mean(aligned_stack, axis=0)


def shared_batter_rows(reference_stats_path, comparison_stats_path):
    """
    The batters present in both builds, with each build's own row offset.

    Returns (batters, reference_rows, comparison_rows) where the row arrays
    index a seed-mean matrix from `seed_mean_embedding` (embedding index minus
    one, since row 0 was dropped). Order is by batter id so the two arms line up
    element for element.
    """
    reference = pd.read_csv(reference_stats_path)[["batter", "embedding_index"]]
    comparison = pd.read_csv(comparison_stats_path)[["batter", "embedding_index"]]
    merged = reference.merge(comparison, on="batter", suffixes=("_reference",
                                                                "_comparison"))
    merged = merged.sort_values("batter").reset_index(drop=True)
    assert len(merged) > 0, "the two builds share no batters; check the stats paths"
    return (merged["batter"].to_numpy(),
            merged["embedding_index_reference"].to_numpy() - 1,
            merged["embedding_index_comparison"].to_numpy() - 1)


def null_cosines(reference_matrix, comparison_matrix, rng):
    """
    One null draw: shuffle the comparison rows, then fit the rotation.

    Fitting after the shuffle is what makes this a null rather than a floor of
    zero. The rotation still gets to do its best, but on rows that no longer
    correspond to the same hitters.
    """
    shuffled = comparison_matrix[rng.permutation(len(comparison_matrix))]
    return row_cosine(procrustes_align(shuffled, reference_matrix),
                      reference_matrix - reference_matrix.mean(axis=0))


def arm_agreement(reference_matrix, comparison_matrix, batters,
                  n_boot=N_BOOT, seed=BOOT_SEED):
    """
    Per-hitter cosine between the two arms after rotating one onto the other,
    against a row-shuffled null.

    Both matrices must already be restricted to the shared batters, in the same
    order. The confidence interval pairs each hitter's real cosine with one null
    draw; `null_floor_mean` averages the null median over more permutations, so
    the reported floor does not rest on a single shuffle.

    Returns (per_hitter frame, summary dict).
    """
    rng = np.random.default_rng(seed)
    reference_centered = reference_matrix - reference_matrix.mean(axis=0)
    real_cosine = row_cosine(procrustes_align(comparison_matrix, reference_matrix),
                             reference_centered)
    null_cosine = null_cosines(reference_matrix, comparison_matrix, rng)
    repeated_null_medians = [
        float(np.median(null_cosines(reference_matrix, comparison_matrix, rng)))
        for _ in range(N_NULL_PERMUTATIONS - 1)]

    per_hitter = pd.DataFrame({
        "batter": batters,
        "cosine_real": real_cosine,
        "cosine_null": null_cosine,
    })

    difference, low, high = bootstrap_ci(
        (real_cosine, null_cosine),
        lambda real, null: float(np.median(real) - np.median(null)),
        n_boot=n_boot, seed=seed)

    summary = {
        "reference_arm": REFERENCE_ARM,
        "comparison_arm": COMPARISON_ARM,
        "n_shared_batters": int(len(batters)),
        "median_cosine_real": float(np.median(real_cosine)),
        "median_cosine_null": float(np.median(null_cosine)),
        "null_floor_mean": float(np.mean([float(np.median(null_cosine))]
                                         + repeated_null_medians)),
        "n_null_permutations": N_NULL_PERMUTATIONS,
        "median_diff": difference,
        "diff_ci95": [low, high],
        "n_boot": n_boot,
        "boot_seed": seed,
        "separates_from_null": bool(low > 0),
    }
    return per_hitter, summary


def fig_arm_agreement(per_hitter, summary, path):
    """Histogram of the real cosines against the shuffled null."""
    figure, axis = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(-1, 1, 61)
    axis.hist(per_hitter["cosine_null"], bins=bins, color="#7a7a7a", alpha=0.55,
              label="shuffled-then-aligned null")
    axis.hist(per_hitter["cosine_real"], bins=bins, color="#4c8dff", alpha=0.8,
              label="matched hitter")
    axis.axvline(summary["median_cosine_real"], color="#1f4fa3", linestyle="--",
                 linewidth=1.2)
    axis.set_xlabel("cosine between the two arms' seed-mean hitter vectors")
    axis.set_ylabel("hitters")
    axis.set_title(
        f"Adding 2024 preserves the hitter space\n"
        f"median cosine {summary['median_cosine_real']:.3f} over "
        f"{summary['n_shared_batters']} shared hitters")
    axis.legend(loc="upper left", frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Cross-arm agreement between the pre-registration arm and the refit.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--reference-arm", default=REFERENCE_ARM)
    parser.add_argument("--reference-stats", default=REFERENCE_STATS)
    parser.add_argument("--comparison-arm", default=COMPARISON_ARM)
    parser.add_argument("--comparison-stats", default=COMPARISON_STATS)
    arguments = parser.parse_args()

    out_dir = Path(arguments.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_matrix = seed_mean_embedding(arguments.checkpoint_dir,
                                           arguments.reference_arm)
    comparison_matrix = seed_mean_embedding(arguments.checkpoint_dir,
                                            arguments.comparison_arm)

    batters, reference_rows, comparison_rows = shared_batter_rows(
        arguments.reference_stats, arguments.comparison_stats)

    per_hitter, summary = arm_agreement(reference_matrix[reference_rows],
                                        comparison_matrix[comparison_rows],
                                        batters)

    per_hitter.to_csv(out_dir / "arm_agreement.csv", index=False)
    (out_dir / "arm_agreement.json").write_text(json.dumps(summary, indent=2))
    fig_arm_agreement(per_hitter, summary, out_dir / "fig_arm_agreement.png")

    print(f"shared batters: {summary['n_shared_batters']}")
    print(f"median cosine real: {summary['median_cosine_real']:.4f}")
    print(f"median cosine null: {summary['median_cosine_null']:.4f}")
    print(f"null floor over {summary['n_null_permutations']} permutations: "
          f"{summary['null_floor_mean']:.4f}")
    print(f"median difference: {summary['median_diff']:.4f} "
          f"[{summary['diff_ci95'][0]:.4f}, {summary['diff_ci95'][1]:.4f}]")
    print(f"wrote {out_dir}/arm_agreement.csv, arm_agreement.json, fig_arm_agreement.png")


if __name__ == "__main__":
    main()
