"""
Phase V.8 and V.9 — five named hitters as a lens on the trained model (docs/phase-v-spec.md).

V.8 shows, for five hitters fans already know (Trout, Soto, Pederson, Duvall, Schwarber), who
sits nearest them in two independent spaces: the model's 32-dim embedding (cosine similarity)
and hitters' observed rate stats (Euclidean distance on z-scored swing/contact/power/level
stats). Neither list is graded against the other or against an outcome; the overlap count is
reported as description, not a verdict on the embedding.

V.9 shows, for the same five hitters, the three separate outputs of the trained scorer's
`expected_woba` at a fixed pool of training-season pitches: p_swing (probability of a swing),
p_contact (probability of contact given a swing), and q = E[wOBA points | ball in play] (quality
of contact given a ball in play). They are never multiplied into one composite. Each is averaged
by pitch-type family x count and by plate location, against LHP and RHP, and reported as
anchor minus a 100-hitter league average scored on the identical pool -- so a cell says "this
anchor's model output differs from the trained-hitter average here" and nothing about whether
either number matches what the hitter actually did.
"""

import argparse
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data import eval_targets
from src.model import loader, query, query_tables as qt

DEFAULT_OUT_DIR = "results/model_visualization"
DEFAULT_CHECKPOINT = "results/checkpoints/d10_baseline_s0.pt"
DEFAULT_DATA_DIR = "data/processed/phase_d5"
DEFAULT_NAMES_CSV = "data/processed/hitter_names.csv"
DEFAULT_STATS_CSV = "results/model_visualization/hitter_stats.csv"
EVAL_SEASON_WEIGHTS = "2024"

ANCHORS = {
    545361: "Mike Trout",
    665742: "Juan Soto",
    592626: "Joc Pederson",
    594807: "Adam Duvall",
    656941: "Kyle Schwarber",
}

STAT_COLUMNS = ["swing_rate", "whiff_rate", "contact_rate", "chase_rate", "zone_swing_rate",
                "ev_mean", "ev_p90", "la_mean", "pull_rate", "woba_level"]
MIN_PRIOR_PA = 113  # spec's low-stratum cut; below this a hitter cannot be a stat neighbour
K_NEIGHBOURS = 8

# coarse pitch-type families; raw Statcast codes are too sparse per (count, hand) cell to use
# directly (spec says say which grouping was used)
PITCH_FAMILY = {
    "FF": "fastball", "FA": "fastball", "FC": "fastball", "SI": "fastball",
    "CU": "breaking", "CS": "breaking", "KC": "breaking", "KN": "breaking",
    "SC": "breaking", "SL": "breaking", "ST": "breaking", "SV": "breaking",
    "CH": "offspeed", "EP": "offspeed", "FO": "offspeed", "FS": "offspeed",
}

LOCATION_X_EDGES = np.linspace(-1.5, 1.5, 6)
LOCATION_Z_EDGES = np.linspace(1.0, 4.0, 6)
N_LOCATION_BINS = 5

QUANTITIES = ["p_swing", "p_contact", "q"]


# --- V.8: nearest neighbours -------------------------------------------------------------

def cosine_neighbours(embedding, anchor_row, k=K_NEIGHBOURS, exclude_rows=(0,)):
    """
    Rows nearest `anchor_row` by cosine similarity, descending.
    embedding: (n_rows, dim). exclude_rows: rows that can never be a neighbour (cold-start).
    Returns a DataFrame with columns rank, row, similarity, excluding the anchor itself.
    """
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    normed = embedding / np.clip(norms, 1e-12, None)
    similarity = normed @ normed[anchor_row]
    excluded = set(exclude_rows) | {anchor_row}
    order = [row for row in np.argsort(-similarity) if row not in excluded][:k]
    return pd.DataFrame({"rank": range(1, len(order) + 1), "row": order,
                         "similarity_or_distance": similarity[order]})


def zscore_candidates(stats_df, columns=STAT_COLUMNS, min_prior_pa=MIN_PRIOR_PA):
    """
    Candidate pool for the stat-neighbour search: drop rows with NaN in any used column
    and rows below the exposure floor, then z-score the remaining columns.
    Returns (filtered_df, z matrix aligned row-for-row with filtered_df).
    """
    eligible = stats_df["log_prior_pa"] >= math.log(min_prior_pa)
    complete = stats_df[columns].notna().all(axis=1)
    filtered = stats_df[eligible & complete].reset_index(drop=True)
    values = filtered[columns].to_numpy(dtype="float64")
    mean, std = values.mean(axis=0), values.std(axis=0)
    z = (values - mean) / np.clip(std, 1e-12, None)
    return filtered, z


def euclidean_neighbours(filtered_df, z, anchor_embedding_index, k=K_NEIGHBOURS):
    """
    Rows nearest the anchor by Euclidean distance in the z-scored stat space, ascending.
    Returns a DataFrame with rank, row (embedding_index), similarity_or_distance -- or an
    empty frame if the anchor itself failed the NaN/exposure filter.
    """
    match = filtered_df.index[filtered_df["embedding_index"] == anchor_embedding_index]
    if len(match) == 0:
        return pd.DataFrame(columns=["rank", "row", "similarity_or_distance"])
    anchor_pos = match[0]
    distance = np.linalg.norm(z - z[anchor_pos], axis=1)
    order = [i for i in np.argsort(distance) if i != anchor_pos][:k]
    return pd.DataFrame({"rank": range(1, len(order) + 1),
                         "row": filtered_df.loc[order, "embedding_index"].to_numpy(),
                         "similarity_or_distance": distance[order]})


def build_neighbours_table(embedding, stats_df, names_df, anchors=ANCHORS):
    """
    V.8's neighbours.csv: embedding and stat neighbours for every anchor, one row each.
    """
    names_by_row = names_df.set_index("embedding_index")
    stats_by_row = stats_df.set_index("embedding_index")
    filtered, z = zscore_candidates(stats_df)

    rows = []
    for batter, anchor_name in anchors.items():
        anchor_row = int(names_df.loc[names_df["batter"] == batter, "embedding_index"].iloc[0])
        for method, table in (("embedding", cosine_neighbours(embedding, anchor_row)),
                              ("stats", euclidean_neighbours(filtered, z, anchor_row))):
            for _, neighbour in table.iterrows():
                row = int(neighbour["row"])
                name_row = names_by_row.loc[row] if row in names_by_row.index else None
                stat_row = stats_by_row.loc[row] if row in stats_by_row.index else None
                rows.append({
                    "anchor": batter, "anchor_name": anchor_name, "method": method,
                    "rank": int(neighbour["rank"]),
                    "batter": int(name_row["batter"]) if name_row is not None else None,
                    "name": name_row["name"] if name_row is not None else "?",
                    "stand": name_row["stand"] if name_row is not None else None,
                    "stratum": stat_row["stratum"] if stat_row is not None else None,
                    "log_prior_pa": stat_row["log_prior_pa"] if stat_row is not None else None,
                    "woba_level": stat_row["woba_level"] if stat_row is not None else None,
                    "obs_platoon_diff": (stat_row["obs_platoon_diff"]
                                         if stat_row is not None else None),
                    "similarity_or_distance": float(neighbour["similarity_or_distance"]),
                })
    return pd.DataFrame(rows)


def overlap_counts(neighbours_table):
    """Per-anchor count of rows appearing in both the embedding and stats neighbour lists."""
    counts = {}
    for anchor, group in neighbours_table.groupby("anchor"):
        embed_ids = set(group.loc[group["method"] == "embedding", "batter"])
        stat_ids = set(group.loc[group["method"] == "stats", "batter"])
        counts[anchor] = len(embed_ids & stat_ids)
    return counts


def plot_neighbours(neighbours_table, anchors, out_path):
    """One row per anchor, two text-table panels: embedding neighbours vs stat neighbours."""
    fig, axes = plt.subplots(len(anchors), 2, figsize=(11, 2.1 * len(anchors)))
    for i, (batter, anchor_name) in enumerate(anchors.items()):
        for j, method in enumerate(["embedding", "stats"]):
            ax = axes[i, j]
            ax.axis("off")
            rows = neighbours_table[(neighbours_table["anchor"] == batter)
                                    & (neighbours_table["method"] == method)]
            lines = [f"{anchor_name} — {method}"]
            for _, r in rows.iterrows():
                platoon = "n/a" if pd.isna(r["obs_platoon_diff"]) else f"{r['obs_platoon_diff']:+.3f}"
                woba = "n/a" if pd.isna(r["woba_level"]) else f"{r['woba_level']:.3f}"
                lines.append(f"{r['rank']}. {r['name']} ({r['stand']}) wOBA={woba} platoon={platoon}")
            ax.text(0, 1, "\n".join(lines), va="top", ha="left", fontsize=8, family="monospace")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# --- V.9: conditional surfaces ------------------------------------------------------------

def decode_pitch_family(context, context_columns):
    """
    Coarse pitch-type family per row, decoded from the context tensor's one-hot
    `pitch_type=XX` columns (the aligned frame does not carry pitch_type directly).
    Returns an array of "fastball"/"breaking"/"offspeed"/"other".
    """
    onehot_cols = [(i, name.split("=")[1]) for i, name in enumerate(context_columns)
                  if name.startswith("pitch_type=")]
    indices = np.array([i for i, _ in onehot_cols])
    codes = np.array([code for _, code in onehot_cols])
    block = context[:, indices]
    winner = codes[np.argmax(block, axis=1)]
    return np.array([PITCH_FAMILY.get(code, "other") for code in winner])


def bin_index(values, edges):
    """
    Which of len(edges)-1 bins each value falls in, edges[0]..edges[-1] inclusive on
    both ends (a point exactly on the right edge lands in the last bin, not bin n).
    Values outside the range clip to the nearest edge bin.
    """
    n_bins = len(edges) - 1
    idx = np.digitize(values, edges[1:-1], right=False)
    return np.clip(idx, 0, n_bins - 1)


def sample_pool(frame, train_seasons, hand, n_pitches, seed=0):
    """Seeded sample of n_pitches training-season row indices with p_throws == hand."""
    eligible = np.flatnonzero(np.isin(frame["season"].to_numpy(), train_seasons)
                              & (frame["p_throws"].to_numpy() == hand))
    n = min(n_pitches, len(eligible))
    rng = np.random.default_rng(seed)
    return rng.choice(eligible, size=n, replace=False)


def sample_league_hitters(n_hitters, n_sample, seed=0):
    """Seeded sample of trained-hitter embedding rows (1..n_hitters), row 0 excluded."""
    rng = np.random.default_rng(seed)
    return rng.choice(np.arange(1, n_hitters + 1), size=min(n_sample, n_hitters), replace=False)


def query_quantities(models, kernels, points, n_bins, embedding_index, context_pool):
    """
    p_swing, p_contact, q for one hitter over one pool of pitch contexts.
    A thin call-through to query.expected_woba -- nothing about the model, scorer, or
    loss is touched here, only queried.
    """
    hitter = torch.full((context_pool.shape[0],), embedding_index, dtype=torch.long)
    swing, contact, quality, _ = query.expected_woba(models, hitter, context_pool,
                                                      points, n_bins, kernels)
    return swing, contact, quality


def project_runtime(seconds_per_unit_batch, unit_batch_size, total_units):
    """Linear projection: seconds per row-eval, times total row-evals."""
    return seconds_per_unit_batch / unit_batch_size * total_units


def aggregate_type_count(family, balls, strikes, values):
    """Mean of `values` (dict quantity -> array) grouped by (family, balls, strikes)."""
    df = pd.DataFrame({"family": family, "balls": balls, "strikes": strikes, **values})
    return df.groupby(["family", "balls", "strikes"]).agg(
        n_pitches=("family", "size"), **{q: (q, "mean") for q in values}
    ).reset_index()


def aggregate_location(x_bin, z_bin, values):
    """Mean of `values` grouped by (x_bin, z_bin)."""
    df = pd.DataFrame({"x_bin": x_bin, "z_bin": z_bin, **values})
    return df.groupby(["x_bin", "z_bin"]).agg(
        n_pitches=("x_bin", "size"), **{q: (q, "mean") for q in values}
    ).reset_index()


def cell_delta(anchor_cells, league_cells, key_columns):
    """
    Merge anchor and league per-cell means on `key_columns` and return anchor - league
    per quantity, long-format (one row per cell x quantity).
    """
    merged = anchor_cells.merge(league_cells, on=key_columns, suffixes=("_anchor", "_league"))
    rows = []
    for _, r in merged.iterrows():
        for q in QUANTITIES:
            rows.append({**{c: r[c] for c in key_columns}, "n_pitches": int(r["n_pitches_anchor"]),
                        "quantity": q, "anchor_value": float(r[f"{q}_anchor"]),
                        "league_value": float(r[f"{q}_league"]),
                        "delta": float(r[f"{q}_anchor"] - r[f"{q}_league"])})
    return pd.DataFrame(rows)


# --- orchestration --------------------------------------------------------------------

def run_v8(embedding, names_df, stats_df, out_dir):
    table = build_neighbours_table(embedding, stats_df, names_df)
    table.to_csv(out_dir / "neighbours.csv", index=False)
    plot_neighbours(table, ANCHORS, out_dir / "fig_neighbours.png")
    overlap = overlap_counts(table)
    for batter, name in ANCHORS.items():
        print(f"V.8 {name}: overlap between embedding and stat neighbour lists = "
              f"{overlap.get(batter, 0)}/{K_NEIGHBOURS}")
        for method in ("embedding", "stats"):
            rows = table[(table["anchor"] == batter) & (table["method"] == method)]
            print(f"  {method}:")
            for _, r in rows.iterrows():
                platoon = "n/a" if pd.isna(r["obs_platoon_diff"]) else f"{r['obs_platoon_diff']:+.3f}"
                woba = "n/a" if pd.isna(r["woba_level"]) else f"{r['woba_level']:.3f}"
                print(f"    {r['rank']}. {r['name']} ({r['stand']}) wOBA={woba} platoon={platoon}")
    return table


def run_v9(checkpoint, out_dir, data_dir, pitch_events, eval_targets_path, names_df,
           n_pitches=40000, n_league=100, budget_seconds=1200):
    tensors, manifest = loader.load_tensors(data_dir)
    train_seasons = manifest["train_seasons"]
    assert max(train_seasons) <= 2023, "2025 must never be read (spec §0.3)"

    frame = qt.align_pitch_frame(pitch_events, eval_targets_path, tensors["season"])
    pa_df = pd.read_parquet(eval_targets_path)
    tables = query.build_tables(frame, tensors, manifest, pa_df)
    weights = eval_targets.load_weights()[EVAL_SEASON_WEIGHTS]
    points = torch.from_numpy(qt.woba_points_table(tables["outcome"], weights)).float()
    n_bins = manifest["n_quality_bins"]

    models = query.load_ensemble([Path(checkpoint)], manifest, tensors["context"].shape[1])
    for model in models:
        assert not model.training, "queried model must be in eval mode"
    kernels = [query.spray_kernels(model, points, n_bins, tables["spray_mass"])
              for model in models]

    context_np = tensors["context"].numpy()
    family_all = decode_pitch_family(context_np, manifest["context_columns"])
    x_bin_all = bin_index(frame["plate_x"].to_numpy(), LOCATION_X_EDGES)
    z_bin_all = bin_index(frame["plate_z"].to_numpy(), LOCATION_Z_EDGES)
    balls_all = frame["balls"].to_numpy()
    strikes_all = frame["strikes"].to_numpy()

    hands = ["L", "R"]
    pools = {hand: sample_pool(frame, train_seasons, hand, n_pitches) for hand in hands}

    # runtime projection: time one anchor x one hand on 5000 pitches, project the full job
    benchmark_hand = hands[0]
    benchmark_idx = pools[benchmark_hand][:min(5000, len(pools[benchmark_hand]))]
    benchmark_context = tensors["context"][torch.as_tensor(benchmark_idx, dtype=torch.long)]
    first_anchor_row = int(names_df.loc[names_df["batter"] == next(iter(ANCHORS)),
                                        "embedding_index"].iloc[0])
    start = time.monotonic()
    query_quantities(models, kernels, points, n_bins, first_anchor_row, benchmark_context)
    benchmark_seconds = time.monotonic() - start
    benchmark_n = len(benchmark_idx)

    total_units = (len(ANCHORS) + n_league) * len(hands) * n_pitches
    projection = project_runtime(benchmark_seconds, benchmark_n, total_units)
    print(f"V.9 runtime projection: {projection / 60:.1f} min "
          f"({benchmark_seconds:.2f}s for {benchmark_n} pitches)")

    if projection > budget_seconds:
        n_league, n_pitches = 40, 20000
        pools = {hand: sample_pool(frame, train_seasons, hand, n_pitches) for hand in hands}
        total_units = (len(ANCHORS) + n_league) * len(hands) * n_pitches
        projection = project_runtime(benchmark_seconds, benchmark_n, total_units)
        print(f"V.9 runtime projection (shrunk pool): {projection / 60:.1f} min")
        if projection > budget_seconds:
            print(f"V.9 STOP: projected {projection / 60:.1f} min exceeds the {budget_seconds / 60:.0f} "
                  f"min budget even at the shrunk pool. Not running.")
            return None, None, None

    league_rows = sample_league_hitters(manifest["n_hitters"], n_league)
    started = time.monotonic()

    type_count_rows, location_rows, summary_rows = [], [], []
    for hand in hands:
        idx = pools[hand]
        context_pool = tensors["context"][torch.as_tensor(idx, dtype=torch.long)]
        family, x_bin, z_bin = family_all[idx], x_bin_all[idx], z_bin_all[idx]
        balls, strikes = balls_all[idx], strikes_all[idx]

        league_type_count, league_location, league_pool_means = [], [], {q: [] for q in QUANTITIES}
        for row in league_rows:
            swing, contact, q = query_quantities(models, kernels, points, n_bins,
                                                  int(row), context_pool)
            values = {"p_swing": swing, "p_contact": contact, "q": q}
            league_type_count.append(aggregate_type_count(family, balls, strikes, values))
            league_location.append(aggregate_location(x_bin, z_bin, values))
            for qty in QUANTITIES:
                league_pool_means[qty].append(values[qty].mean())
        league_tc = pd.concat(league_type_count).groupby(["family", "balls", "strikes"]).agg(
            n_pitches=("n_pitches", "sum"),
            **{qty: (qty, "mean") for qty in QUANTITIES}).reset_index()
        league_loc = pd.concat(league_location).groupby(["x_bin", "z_bin"]).agg(
            n_pitches=("n_pitches", "sum"),
            **{qty: (qty, "mean") for qty in QUANTITIES}).reset_index()
        league_summary = {qty: float(np.mean(vals)) for qty, vals in league_pool_means.items()}

        for batter, anchor_name in ANCHORS.items():
            anchor_row = int(names_df.loc[names_df["batter"] == batter, "embedding_index"].iloc[0])
            swing, contact, q = query_quantities(models, kernels, points, n_bins,
                                                  anchor_row, context_pool)
            values = {"p_swing": swing, "p_contact": contact, "q": q}
            anchor_tc = aggregate_type_count(family, balls, strikes, values)
            anchor_loc = aggregate_location(x_bin, z_bin, values)

            tc_delta = cell_delta(anchor_tc, league_tc, ["family", "balls", "strikes"])
            tc_delta.insert(0, "anchor_name", anchor_name)
            tc_delta.insert(0, "anchor", batter)
            tc_delta.insert(2, "p_throws", hand)
            type_count_rows.append(tc_delta.rename(columns={"family": "pitch_type"}))

            loc_delta = cell_delta(anchor_loc, league_loc, ["x_bin", "z_bin"])
            loc_delta.insert(0, "anchor_name", anchor_name)
            loc_delta.insert(0, "anchor", batter)
            loc_delta.insert(2, "p_throws", hand)
            location_rows.append(loc_delta)

            for qty in QUANTITIES:
                anchor_mean = float(values[qty].mean())
                summary_rows.append({"anchor": batter, "anchor_name": anchor_name,
                                    "p_throws": hand, "quantity": qty,
                                    "anchor_value": anchor_mean,
                                    "league_value": league_summary[qty],
                                    "delta": anchor_mean - league_summary[qty]})

    actual_seconds = time.monotonic() - started
    print(f"V.9 actual runtime: {actual_seconds / 60:.1f} min")

    type_count = pd.concat(type_count_rows, ignore_index=True)
    location = pd.concat(location_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    type_count.to_csv(out_dir / "surfaces_type_count.csv", index=False)
    location.to_csv(out_dir / "surfaces_location.csv", index=False)
    summary.to_csv(out_dir / "surfaces_summary.csv", index=False)
    plot_surfaces_type_count(type_count, out_dir / "fig_surfaces_type_count.png")
    plot_surfaces_location(location, out_dir / "fig_surfaces_location.png")
    return type_count, location, summary


def plot_surfaces_type_count(type_count, out_path):
    """Rows = anchors, columns = quantity x hand; each cell a pitch_type x count heatmap."""
    anchors = list(ANCHORS.values())
    hands = sorted(type_count["p_throws"].unique())
    cols = [(q, h) for q in QUANTITIES for h in hands]
    fig, axes = plt.subplots(len(anchors), len(cols),
                             figsize=(2.6 * len(cols), 2.2 * len(anchors)))
    vmax = np.abs(type_count["delta"]).max() or 1.0
    for i, anchor_name in enumerate(anchors):
        sub_anchor = type_count[type_count["anchor_name"] == anchor_name]
        for j, (qty, hand) in enumerate(cols):
            ax = axes[i, j]
            cell = sub_anchor[(sub_anchor["quantity"] == qty) & (sub_anchor["p_throws"] == hand)]
            pivot = cell.pivot_table(index="pitch_type", columns=["balls", "strikes"],
                                     values="delta")
            im = ax.imshow(pivot.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            if i == 0:
                ax.set_title(f"{qty} vs {hand}HP", fontsize=8)
            if j == 0:
                ax.set_ylabel(anchor_name, fontsize=8)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=6)
            ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_surfaces_location(location, out_path):
    """Rows = anchors, columns = quantity x hand; each cell a 5x5 plate-location heatmap."""
    anchors = list(ANCHORS.values())
    hands = sorted(location["p_throws"].unique())
    cols = [(q, h) for q in QUANTITIES for h in hands]
    fig, axes = plt.subplots(len(anchors), len(cols),
                             figsize=(2.2 * len(cols), 2.0 * len(anchors)))
    vmax = np.abs(location["delta"]).max() or 1.0
    for i, anchor_name in enumerate(anchors):
        sub_anchor = location[location["anchor_name"] == anchor_name]
        for j, (qty, hand) in enumerate(cols):
            ax = axes[i, j]
            cell = sub_anchor[(sub_anchor["quantity"] == qty) & (sub_anchor["p_throws"] == hand)]
            grid = np.full((N_LOCATION_BINS, N_LOCATION_BINS), np.nan)
            for _, r in cell.iterrows():
                grid[N_LOCATION_BINS - 1 - int(r["z_bin"]), int(r["x_bin"])] = r["delta"]
            ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax)  # catcher's view: high z on top
            if i == 0:
                ax.set_title(f"{qty} vs {hand}HP", fontsize=8)
            if j == 0:
                ax.set_ylabel(anchor_name, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Phase V.8/V.9 -- anchor hitters.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--pitch-events", default="data/processed/pitch_events_labeled.parquet")
    parser.add_argument("--eval-targets", default="data/processed/eval_targets_pa.parquet")
    parser.add_argument("--names-csv", default=DEFAULT_NAMES_CSV)
    parser.add_argument("--stats-csv", default=DEFAULT_STATS_CSV)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names_df = pd.read_csv(args.names_csv)
    stats_df = pd.read_csv(args.stats_csv)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    embedding = saved["model"]["embedding.weight"].numpy()

    run_v8(embedding, names_df, stats_df, out_dir)
    run_v9(args.checkpoint, out_dir, args.data_dir, args.pitch_events, args.eval_targets,
          names_df)


if __name__ == "__main__":
    main()
