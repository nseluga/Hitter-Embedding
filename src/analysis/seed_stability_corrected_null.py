"""
Corrected-null sidecar for seed stability.

`model_visualization_embeddings.seed_stability` shuffles rows AFTER fitting
the Procrustes rotation onto seed 0. That leaves the fitted alignment intact
under the shuffle and reports a null near 0 — it doesn't give the null
rotation the same fitting freedom the real rotation gets. The correct null
shuffles rows BEFORE fitting, exactly as
`model_visualization_arm_agreement.null_cosines` does.

This is a NEW result, not a fix in place: it writes its own JSON and never
touches `seed_stability_summary.json` or anything else in
results/model_visualization/.

Run: python -m src.analysis.seed_stability_corrected_null
"""

import json
from pathlib import Path

import numpy as np

from src.analysis.model_evaluation_probe_coverage import load_seed_embeddings
from src.analysis.model_visualization_embeddings import (
    bootstrap_ci,
    procrustes_align,
    row_cosine,
)

CHECKPOINT_DIR = "results/checkpoints"
ARM = "embedding_sgd_sgd_lr1"
SEEDS = (0, 1, 2, 3, 4)
COLD_START_ROW = 0
N_BOOT = 1000
BOOT_SEED = 7
OUT_PATH = "results/model_visualization/seed_stability_corrected_null.json"


def corrected_null_cosine(source, target, rng):
    """Shuffle source rows, then fit the rotation onto target (the correct null)."""
    shuffled = source[rng.permutation(len(source))]
    return row_cosine(procrustes_align(shuffled, target), target - target.mean(axis=0))


def main():
    embeddings = load_seed_embeddings(CHECKPOINT_DIR, ARM, seeds=SEEDS)
    seed0_rows = embeddings[SEEDS[0]][COLD_START_ROW + 1:]
    seed0_centered = seed0_rows - seed0_rows.mean(axis=0)

    rng = np.random.default_rng(BOOT_SEED)
    per_seed = {}
    real_medians, null_medians = [], []
    for seed in SEEDS[1:]:
        rows = embeddings[seed][COLD_START_ROW + 1:]
        real_cos = row_cosine(procrustes_align(rows, seed0_rows), seed0_centered)
        null_cos = corrected_null_cosine(rows, seed0_rows, rng)
        real_median, null_median = float(np.median(real_cos)), float(np.median(null_cos))
        real_medians.append(real_median)
        null_medians.append(null_median)
        per_seed[str(seed)] = {
            "real_median_cosine": real_median,
            "corrected_null_median_cosine": null_median,
        }

    real_mean = float(np.mean(real_medians))
    null_mean = float(np.mean(null_medians))

    diff, lo, hi = bootstrap_ci(
        (np.array(real_medians), np.array(null_medians)),
        lambda r, n: float(np.median(r) - np.median(n)),
        n_boot=N_BOOT, seed=BOOT_SEED)

    summary = {
        "arm": ARM,
        "seeds": list(SEEDS),
        "reference_seed": SEEDS[0],
        "per_seed": per_seed,
        "real_median_per_seed": real_medians,
        "corrected_null_median_per_seed": null_medians,
        "real_mean": real_mean,
        "corrected_null_mean": null_mean,
        "margin": real_mean - null_mean,
        "median_diff": diff,
        "diff_ci95": [lo, hi],
        "n_boot": N_BOOT,
        "boot_seed": BOOT_SEED,
        "supersedes": "seed_stability_summary.json",
        "note": ("seed_stability_summary.json's null shuffles rows after fitting the "
                 "Procrustes rotation, which leaves the fitted alignment intact and "
                 "reports a null near zero; this sidecar shuffles rows before fitting "
                 "so the null rotation gets the same fitting freedom as the real one."),
    }

    out_path = Path(OUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"real per-seed: {[round(v, 4) for v in real_medians]} mean {real_mean:.4f}")
    print(f"corrected null per-seed: {[round(v, 4) for v in null_medians]} mean {null_mean:.4f}")
    print(f"margin: {summary['margin']:.4f}")
    print(f"median diff: {diff:.4f} [{lo:.4f}, {hi:.4f}]")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
