"""Train vs test (2025) loss per row for the final refit checkpoints, same units as the loss curves.
Usage: .venv/bin/python -m src.analysis.final_loss_report [arm]   (default clean_dim64_final)

Scores each seed's checkpoint in eval mode (no dropout, fixed order) with train.py's own run_epoch and
CANONICAL objective (sum of the 5 factor NLLs per scored row):
  train_eval  all 2015-2024 rows (eval mode, so comparable to test; the logged train_loss_epoch is a
              running mean taken while the weights moved, so it reads slightly higher)
  test        all 2025 rows
  test_seen   2025 rows whose hitter has a trained embedding row (hitter index > 0)
Nothing here selects anything: the refit has a fixed step budget, no early stopping, no LR plateau.
Writes results/loss_curves/final_test_loss.csv, read by src/analysis/loss_curves.py for the test marker."""
import csv, pathlib, sys
from types import SimpleNamespace
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
from src.model import loader, train
from src.model.v1 import HitterEmbeddingV1
from src.config.splits import load_splits

ARM_DATA = {"clean_dim64_final": "data/processed/phase_d5_clean_final"}
FINAL_SPLIT = ROOT / "src/config/split_config_final_run.json"


def seen_rows(hitter, rows):
    """Subset of `rows` whose hitter has a trained embedding row (index 0 is the cold-start row)."""
    return rows[hitter[rows] > 0]


def main(arm):
    tensors, manifest = loader.load_tensors(str(ROOT / ARM_DATA[arm]))
    idx = loader.split_indices(tensors["season"], load_splits(FINAL_SPLIT))
    test = idx["test"]
    assert set(tensors["season"][test].unique().tolist()) == {2025}
    sets = {"train_eval": idx["train"], "test": test,
            "test_seen": seen_rows(tensors["hitter"], test)}
    out = ROOT / "results/loss_curves/final_test_loss.csv"
    rows = []
    for ck in sorted((ROOT / "results/checkpoints").glob(f"{arm}_s*.pt")):
        saved = torch.load(ck, map_location="cpu", weights_only=False)
        a = saved["args"]
        model = HitterEmbeddingV1(manifest["n_hitters"], tensors["context"].shape[1],
                                  embedding_dim=a["embedding_dim"], n_bins=manifest["n_quality_bins"],
                                  bilinear=a["bilinear"], split=a.get("split", False),
                                  spray=not a.get("no_spray", False))
        model.load_state_dict(saved["model"])
        args = SimpleNamespace(device="cpu", batch_size=a["batch_size"], **{
            k: a.get(k) for k in ("loss_rule", "loss_weighting", "contact_pos_weight")})
        row = {"run_name": ck.stem, "epoch": saved["epoch"], "steps": saved["steps"]}
        for name, ix in sets.items():
            row[name] = train.run_epoch(model, tensors, ix, None, None, args,
                                        objective=train.CANONICAL)[0]
            row[f"n_{name}"] = len(ix)
        print(row, flush=True)
        rows.append(row)
        with open(out, "w", newline="") as fh:  # rewritten per seed so a crash keeps what finished
            w = csv.DictWriter(fh, fieldnames=list(row)); w.writeheader(); w.writerows(rows)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "clean_dim64_final")
