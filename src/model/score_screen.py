"""
Scores the RPS screen's checkpoints on 2024 under one shared objective (2026-08-02).

The screen is promote-only: log loss stays v1's objective, and RPS can only be
promoted to a claim-1 ablation, never adopted. Nothing here decides anything -- it
produces the number the 2026-08-02 entry's promotion condition is read against.

Two failures here are silent.

1. THE ARM'S OWN OBJECTIVE. `best_val_loss` in the ledger is each arm's own training
   objective, so the RPS arm's 0.934 and the log arm's 1.076 are in different units
   and their ordering means nothing. Every checkpoint is re-scored here under
   `train.CANONICAL`, which is the only column comparable across arms.

2. THE FROZEN TEST SEASON. A checkpoint records the season it validated on. That
   value goes to `assert_not_test_season` with no `final_run` escape, so a checkpoint
   trained against 2025 raises here rather than quietly reporting a number.

Each checkpoint was saved at its own arm's best epoch, so the number below is the
canonical score of the model that arm would actually ship -- the same convention the
later runs' `reference` column records.
"""

import argparse
import csv
from pathlib import Path

import torch

from src.analysis import claim1_eval as evaluation
from src.model import loader, train
from src.model.v1 import HitterEmbeddingV1

SCREEN_ARMS = ("log", "rps")
SCREEN_SEEDS = (0, 1)
OUTPUT_NAME = "screen_scores.csv"

# The log arm trains on the canonical objective, so re-scoring it must return the
# value already in the ledger. That identity is this module's only self-check: the
# checkpoints and the 1.7 GB table are both gitignored, so the test suite cannot
# reach them, and a scoring path that has drifted otherwise fails silently.
LEDGER_LOG_REFERENCE = {0: 1.07581, 1: 1.07582}


def load_checkpoint(path, tensors, manifest):
    """
    Rebuild the model a checkpoint was saved from, and return (model, saved dict).
    The architecture comes out of the checkpoint's own `args`, never off the filename:
    embedding dimension and the bilinear toggle both vary across arms, and a filename
    disagreeing with the weights would load a mis-shaped state dict or, worse, a
    correctly-shaped one belonging to a different arm.
    """
    # weights_only=False: the invfreq arm stores a real tensor inside its args dict
    saved = torch.load(path, map_location="cpu", weights_only=False)
    args = argparse.Namespace(**saved["args"])
    evaluation.assert_not_test_season(args.eval_season)

    model = HitterEmbeddingV1(manifest["n_hitters"], tensors["context"].shape[1],
                              embedding_dim=args.embedding_dim,
                              n_bins=manifest["n_quality_bins"],
                              bilinear=args.bilinear)
    model.load_state_dict(saved["model"])
    return model, saved, args


def score(checkpoint_dir=None, data_dir=None):
    """
    Score every screen checkpoint on its validation season under the canonical
    objective. Returns one dict per checkpoint. The built table is ~1.7 GB resident,
    so tensors are read once and shared across all four.
    """
    checkpoint_dir = Path(checkpoint_dir or train.CHECKPOINT_DIR)
    tensors, manifest = loader.load_tensors(data_dir or train.DATA_DIR)
    indices = loader.split_indices(tensors["season"])

    rows = []
    for arm in SCREEN_ARMS:
        for seed in SCREEN_SEEDS:
            path = checkpoint_dir / f"screen_{arm}_s{seed}.pt"
            assert path.exists(), f"missing screen checkpoint {path}"
            model, saved, args = load_checkpoint(path, tensors, manifest)
            reference = train.run_epoch(model, tensors, indices[args.eval_split],
                                        None, None, args,
                                        objective=train.CANONICAL)[0]
            rows.append({"arm": arm, "seed": seed, "epoch": saved["epoch"],
                         "own_objective": round(saved["val_loss"], 5),
                         "reference": round(reference, 5)})

    for row in rows:
        expected = LEDGER_LOG_REFERENCE.get(row["seed"]) if row["arm"] == "log" else None
        assert expected is None or row["reference"] == expected, \
            f"screen_log_s{row['seed']} scored {row['reference']}, ledger says {expected}: " \
            "the scoring path has drifted and no number here is trustworthy"
    return rows


def main():
    rows = score()
    out_path = Path(train.OUT_DIR) / OUTPUT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(f"screen_{row['arm']}_s{row['seed']} "
              f"own objective {row['own_objective']:.5f} "
              f"canonical {row['reference']:.5f}")
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
