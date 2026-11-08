"""Draft-3: V.8 neighbour lists rerun on the 5-seed aligned mean embedding (clean_clean_dim64)
instead of seed 0. Run: .venv/bin/python -m src.analysis.paper_seed_mean_neighbours"""
from pathlib import Path
import numpy as np
import pandas as pd
from src.analysis.model_visualization_arm_agreement import seed_mean_embedding
from src.analysis.model_visualization_anchors import build_neighbours_table, overlap_counts

OUT = Path("results/paper_draft3")
mean = seed_mean_embedding("results/checkpoints", "clean_clean_dim64")
embedding = np.vstack([np.zeros((1, mean.shape[1])), mean])  # row i = embedding index i
names = pd.read_csv("data/processed/hitter_names_clean.csv")
stats = pd.read_csv("results/v_chain_clean/hitter_stats.csv")
assert len(embedding) == len(names), (len(embedding), len(names))
OUT.mkdir(parents=True, exist_ok=True)
for tag, strata in (("all", None), ("high", {"high"})):
    t = build_neighbours_table(embedding, stats, names, strata=strata)
    t.to_csv(OUT / f"neighbours_seed_mean_{tag}.csv", index=False)
    ov = overlap_counts(t)
    print(f"== {tag}: embedding/stat overlap {sum(ov.values())}/{8 * len(ov)}")
    for a, g in t[t.method == "embedding"].groupby("anchor_name"):
        print(f"  {a}: " + ", ".join(g["name"]))
    for a, g in t[t.method == "stats"].groupby("anchor_name"):
        print(f"  [stats] {a}: " + ", ".join(g["name"]))
