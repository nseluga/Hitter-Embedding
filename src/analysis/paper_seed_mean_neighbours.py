"""Draft-3: V.8 neighbour lists rerun on the 5-seed aligned mean embedding (clean_clean_dim64)
instead of seed 0. Run: .venv/bin/python -m src.analysis.paper_seed_mean_neighbours"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from src.analysis.model_visualization_arm_agreement import seed_mean_embedding
from src.analysis.model_visualization_anchors import build_neighbours_table, overlap_counts

# Optional args: arm, hitter-stats csv, out dir. Defaults are the selection model (draft 3).
ARM, STATS, OUT = (sys.argv[1:] + ["clean_clean_dim64", "results/v_chain_clean/hitter_stats.csv",
                                   "results/paper_draft3"][len(sys.argv) - 1:])
OUT = Path(OUT)
# Average the 5 seeds' embeddings after aligning them, so the neighbour lists do not depend on one seed's arbitrary rotation.
mean = seed_mean_embedding("results/checkpoints", ARM)
# Prepend a zero row for the reserved cold-start index 0, so row i of `embedding` is embedding index i (what the names/stats tables key on).
embedding = np.vstack([np.zeros((1, mean.shape[1])), mean])  # row i = embedding index i
stats = pd.read_csv(STATS)
# embedding_index is a row position in one build's table, so take it from that build's stats and
# join names by batter id (the two name files together cover every hitter in either build).
all_names = pd.concat([pd.read_csv(f"data/processed/hitter_names{s}.csv") for s in ("_clean", "")])
names = stats[["batter", "embedding_index", "stand"]].merge(
    all_names.drop_duplicates("batter")[["batter", "name"]], on="batter", how="left")
assert len(embedding) == len(stats) + 1, (len(embedding), len(stats))
print(f"{names['name'].isna().sum()} hitters have no name on file; shown by MLB id")
names["name"] = names["name"].fillna("id " + names["batter"].astype(str))
OUT.mkdir(parents=True, exist_ok=True)
# Build the neighbour table twice: over all hitters, and restricted to the high-exposure stratum.
for tag, strata in (("all", None), ("high", {"high"})):
    t = build_neighbours_table(embedding, stats, names, strata=strata)
    t.to_csv(OUT / f"neighbours_seed_mean_{tag}.csv", index=False)
    # Count how many of each anchor's 8 embedding neighbours also appear in its stats-based neighbour list.
    ov = overlap_counts(t)
    print(f"== {tag}: embedding/stat overlap {sum(ov.values())}/{8 * len(ov)}")
    # Print each anchor's embedding neighbours, then its stats-based neighbours for comparison.
    for a, g in t[t.method == "embedding"].groupby("anchor_name"):
        print(f"  {a}: " + ", ".join(g["name"]))
    for a, g in t[t.method == "stats"].groupby("anchor_name"):
        print(f"  [stats] {a}: " + ", ".join(g["name"]))
