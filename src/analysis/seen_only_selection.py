"""
Re-scores the `clean` selection stage on seen hitters only (decision log 2026-11-17).

The `clean` stage picked the reported build (`clean_dim64`) on `reference`: 2024 log loss
per scored pitch. That loss counted pitches to hitters who debuted in 2024, whom a model
trained on 2015-2023 can only send to the reserved untrained embedding row. This module
re-scores every saved `clean` checkpoint twice, on all 2024 pitches and on seen hitters'
pitches only, and writes a ledger the unchanged promotion rule can read:

    python -m src.analysis.seen_only_selection
    python -m src.analysis.hyperparameter_tuning_select --stage clean \
        --ledger results/hyperparameter_tuning/seen_only/sweep_log.csv \
        --out-dir results/hyperparameter_tuning/seen_only

No training happens here. What it cannot undo: each checkpoint was saved at the epoch the
ALL-pitch loss picked, so the stopping point itself is still the old rule's.
"""

import argparse
import csv
import json
from pathlib import Path

import torch

from src.analysis import hyperparameter_tuning_select as select
from src.model import loader, query, train

STAGE = "clean"
DATA_DIR = "data/processed/phase_d5_clean"
OUT_DIR = Path("results/hyperparameter_tuning/seen_only")


def main():
    tensors, manifest = loader.load_tensors(DATA_DIR)
    indices = loader.split_indices(tensors["season"])
    rows = [row for row in select.ledger_rows() if row["stage"] == STAGE]
    assert rows, f"no `{STAGE}` rows in the ledger"

    out, share = [], None
    for row in rows:
        path = train.CHECKPOINT_DIR / f"{STAGE}_{row['config']}_s{row['seed']}.pt"
        model = query.load_ensemble([path], manifest, tensors["context"].shape[1])[0]
        args = argparse.Namespace(**torch.load(path, map_location="cpu",
                                               weights_only=False)["args"])
        args.device = "cpu"
        val = indices[args.eval_split]
        seen = tensors["hitter"][val] != manifest["reserved_hitter_index"]
        share = {"val_pitches": int(len(val)), "unseen_pitches": int((~seen).sum()),
                 "unseen_share": float((~seen).float().mean())}

        def loss(index):
            return train.run_epoch(model, tensors, index, None, None, args,
                                   objective=train.CANONICAL)[0]

        everything, seen_only = loss(val), loss(val[seen])
        # the all-pitch score must reproduce the ledger, or the scoring path has drifted
        # and the seen-only number beside it means nothing
        assert round(everything, 5) == round(float(row["reference"]), 5), \
            f"{path.name}: re-scored {everything:.5f}, ledger says {row['reference']}"
        out.append({**row, "reference_all_pitches": row["reference"],
                    "reference": f"{seen_only:.5f}"})
        print(f"{row['config']:>18s} s{row['seed']}  all {everything:.5f}  "
              f"seen {seen_only:.5f}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "sweep_log.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    (OUT_DIR / "unseen_pitch_share.json").write_text(json.dumps(share, indent=2))
    print(f"\n{share}\nwrote {OUT_DIR}")




def plot():
    """Per-seed `reference`, all pitches against seen hitters only, one row per arm."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    ledger = pd.read_csv(OUT_DIR / "sweep_log.csv")
    arms = ledger.groupby("config")["reference"].mean().sort_values().index[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 3))
    for y, arm in enumerate(arms):
        rows = ledger[ledger["config"] == arm]
        ax.plot(rows["reference_all_pitches"], [y + 0.12] * len(rows), "o", color="0.6",
                label="all 2024 pitches" if y == 0 else None)
        ax.plot(rows["reference"], [y - 0.12] * len(rows), "o", color="C0",
                label="seen hitters only" if y == 0 else None)
    ax.set_yticks(range(len(arms)), arms)
    ax.set_xlabel("2024 log loss per pitch, one dot per seed (lower is better)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "reference_all_vs_seen.png", dpi=200)


if __name__ == "__main__":
    main()
    plot()
