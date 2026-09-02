"""
Phase V — head fingerprints (V.3) and the level query map (V.4), docs/phase-v-spec.md.

V.3 asks what each factor head (swing, contact, split, EV, LA, spray) has learned about
each hitter, independent of how the pieces later compose into a wOBA number. For every
trained hitter (embedding rows 1..1762) this module runs the frozen seed-0 trunk and
heads over one fixed reference set of 2,000 training-season pitches and averages each
head's output into a single per-hitter "marginal": swing and contact are mean predicted
probabilities; split_inplay is the mean softmax probability of the in-play class;
ev and la are the mean of the head's own predicted distribution dotted with the quality
bin centres (la marginalised over the model's own predicted EV bin, exactly as the head
conditions at inference); spray is a pull-tendency scalar (see `PULL_SCALAR_NOTE`). These
marginals are mapped onto the PCA embedding coordinates and correlated (raw and
partialled on log prior PA) against the observable-stat panel in `hitter_stats.csv`.

V.4 reads the committed 2024 ensemble query (`model_v1_predictions_rebuild_baseline.csv`),
averages each hitter's predicted wOBA over the sides they were queried on into one
"level query" number, and checks it against training wOBA level by exposure stratum.
Nothing here trains, scores a new season, or changes the model, scorer, or loss --
only `load_ensemble`, `_trunk`, `spray_kernels`, and the factor heads are called.

Run: python -m src.analysis.model_visualization_heads --out-dir results/model_visualization
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from src.model import loader
from src.model.query import load_ensemble, _trunk, spray_kernels
from src.model.v1 import SPLIT_CLASSES

DEFAULT_DATA_DIR = "data/processed/phase_d5"
DEFAULT_CHECKPOINT = "results/checkpoints/d10_baseline_s0.pt"
DEFAULT_OUT_DIR = "results/model_visualization"
MANIFEST_PATH = "data/processed/phase_d5/manifest.json"
NAMES_PATH = "data/processed/hitter_names.csv"
HITTER_STATS_PATH = "results/model_visualization/hitter_stats.csv"
EMBEDDING_COORDS_PATH = "results/model_visualization/embedding_coords.csv"
LEVEL_QUERY_PREDICTIONS_PATH = "results/model_v1/model_v1_predictions_rebuild_baseline.csv"

N_REFERENCE_PITCHES = 2000
REFERENCE_SEED = 0
HITTER_BATCH = 64
BOOT_SEED = 0
N_BOOT = 1000
ANCHOR_IDS = {545361: "Trout", 665742: "Soto", 592626: "Pederson",
              594807: "Duvall", 656941: "Schwarber"}
IN_PLAY_INDEX = SPLIT_CLASSES.index("in_play")

# spec §5 vars for the loadings heatmap, joined from hitter_stats.csv
STAT_VARIABLES = ["swing_rate", "whiff_rate", "contact_rate", "chase_rate",
                   "zone_swing_rate", "ev_mean", "ev_p90", "la_mean",
                   "bat_speed_mean", "pull_rate", "woba_level", "obs_platoon_diff"]
HEAD_MARGINALS = ["swing", "contact", "split_inplay", "ev", "la", "spray"]

# PULL_SCALAR_NOTE: the spray head's predicted quality-bin distribution p(spray | ev, la)
# uses the same bin edges as the `spray` label in src.analysis.model_visualization_stats,
# which is ALREADY mirrored by stand so a positive value means "toward the batter's pull
# side" for both L and R hitters. The chosen scalar is therefore the model's own predicted
# mean spray angle (bin-centre expectation, marginalised over its predicted EV/LA), with no
# further stand mirroring needed -- reusing `spray_kernels` with the bin centres in place of
# wOBA points is the same factored-sum machinery `_conditionals` already computes for R.
PULL_SCALAR_NOTE = ("spray marginal = model's predicted E[mirrored spray angle], marginalised "
                     "over its own predicted EV/LA bins; positive means pull for both hands "
                     "because the underlying bin edges are already stand-mirrored upstream")


def bin_centres(edges):
    """
    Midpoints of K-1 interior edges into K bin centres. The two outer (open) bins have no
    natural centre, so they extrapolate the adjacent gap outward -- documented per the task
    spec's "if only edges exist, use midpoints" instruction.
    """
    edges = np.asarray(edges, dtype="float64")
    interior = (edges[:-1] + edges[1:]) / 2.0
    first = edges[0] - (edges[1] - edges[0]) / 2.0
    last = edges[-1] + (edges[-1] - edges[-2]) / 2.0
    return np.concatenate([[first], interior, [last]])


def sample_reference_context(tensors, manifest, n=N_REFERENCE_PITCHES, seed=REFERENCE_SEED):
    """
    A fixed seeded sample of n pitch-row indices restricted to training seasons. Returns
    the sorted index array; deterministic under the seed so every head reuses the same set.
    """
    train_seasons = set(manifest["train_seasons"])
    assert max(train_seasons) <= 2023, "reference set must draw from training seasons only"
    seasons = tensors["season"].numpy()
    eligible = np.flatnonzero(np.isin(seasons, list(train_seasons)))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(eligible, size=n, replace=False)
    chosen.sort()
    return chosen


def head_marginals_for_hitters(model, hitter_ids, reference_context, centres, n_bins):
    """
    Averages the six per-pitch head outputs over `reference_context` for each id in
    `hitter_ids`. reference_context: (n, n_context) fixed pitch rows. Returns a dict of
    arrays, one entry per HEAD_MARGINALS name, each length len(hitter_ids).
    """
    n_ref = reference_context.shape[0]
    n_hit = len(hitter_ids)
    ev_centre = torch.from_numpy(centres["ev"]).float()
    la_centre = torch.from_numpy(centres["la"]).float()
    spray_centre_points = torch.from_numpy(
        np.broadcast_to(centres["spray"], (n_bins, n_bins, n_bins)).copy()).float()

    with torch.no_grad():
        weighted, kernel = spray_kernels(model, spray_centre_points, n_bins)
        hitter_tensor = torch.as_tensor(hitter_ids, dtype=torch.long).repeat_interleave(n_ref)
        context_tensor = reference_context.repeat(n_hit, 1)
        trunk = _trunk(model, hitter_tensor, context_tensor)
        hidden = trunk.shape[1]

        swing = torch.sigmoid(model.head_swing(trunk).squeeze(-1))
        contact = torch.sigmoid(model.head_contact(trunk).squeeze(-1))
        split_inplay = torch.softmax(model.head_split(trunk), dim=1)[:, IN_PLAY_INDEX]

        p_ev = torch.softmax(model.head_ev(trunk), dim=1)                       # (rows, K)
        base_la = trunk @ model.head_la.weight[:, :hidden].T + model.head_la.bias
        column_la = model.head_la.weight[:, hidden:].T                          # (K_e, K_l)
        p_la = torch.softmax(base_la.unsqueeze(1) + column_la.unsqueeze(0), dim=-1)  # (rows,K_e,K_l)

        base_spray = trunk @ model.head_spray.weight[:, :hidden].T + model.head_spray.bias
        scaled = torch.exp(base_spray - base_spray.amax(dim=1, keepdim=True))
        numerator = scaled @ weighted.T
        denominator = scaled @ kernel.T
        spray_expect = (numerator / denominator).reshape(-1, n_bins, n_bins)    # (rows,K_e,K_l)

        ev = p_ev @ ev_centre
        la_given_e = p_la @ la_centre                                           # (rows, K_e)
        la = (p_ev * la_given_e).sum(dim=1)
        spray = torch.einsum("me,mel,mel->m", p_ev, p_la, spray_expect)

        rows = {"swing": swing, "contact": contact, "split_inplay": split_inplay,
                "ev": ev, "la": la, "spray": spray}
        out = {name: value.reshape(n_hit, n_ref).mean(dim=1).numpy()
               for name, value in rows.items()}
    return out


def compute_head_marginals(model, hitter_ids, reference_context, manifest, batch=HITTER_BATCH):
    """Batches `head_marginals_for_hitters` over all hitters, HITTER_BATCH at a time."""
    n_bins = manifest["n_quality_bins"]
    centres = {axis: bin_centres(edges) for axis, edges in manifest["quality_bin_edges"].items()}
    chunks = []
    for start in range(0, len(hitter_ids), batch):
        ids = hitter_ids[start:start + batch]
        chunks.append(head_marginals_for_hitters(model, ids, reference_context, centres, n_bins))
    return {name: np.concatenate([chunk[name] for chunk in chunks]) for name in HEAD_MARGINALS}


def ols_residualise(y, x):
    """Residuals of y after regressing on x plus an intercept (pairwise-complete)."""
    design = np.column_stack([np.ones_like(x), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coefficients


def paired_pearson(a, b, confound=None):
    """
    Pearson r on pairwise-complete rows of a and b. If `confound` is given, both a and b
    are residualised on it first (partial correlation), still pairwise-complete.
    Returns (r, n).
    """
    mask = np.isfinite(a) & np.isfinite(b)
    if confound is not None:
        mask &= np.isfinite(confound)
    a, b = a[mask], b[mask]
    n = mask.sum()
    if n < 3:
        return float("nan"), int(n)
    if confound is not None:
        c = confound[mask]
        a, b = ols_residualise(a, c), ols_residualise(b, c)
    r = float(np.corrcoef(a, b)[0, 1])
    return r, int(n)


def bootstrap_ci_partial(a, b, confound, n_boot=N_BOOT, seed=BOOT_SEED):
    """95% percentile bootstrap CI on the partial Pearson r, resampling hitters (rows)."""
    mask = np.isfinite(a) & np.isfinite(b) & np.isfinite(confound)
    a, b, confound = a[mask], b[mask], confound[mask]
    rng = np.random.default_rng(seed)
    n = len(a)
    point, _ = paired_pearson(a, b, confound)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[i], _ = paired_pearson(a[idx], b[idx], confound[idx])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi), int(n)


# --------------------------------------------------------------------- V.3 maps + loadings

def get_or_compute_pca_coords(hitters, out_dir):
    """
    Reuses embedding_coords.csv (pc1/pc2/.../batter or embedding_index) if the sibling
    script has produced it; otherwise falls back to PCA on seed 0's embedding.weight[1:]
    and says so via the returned `fallback` flag.
    """
    path = Path(EMBEDDING_COORDS_PATH)
    if path.exists():
        coords = pd.read_csv(path)
        pc_cols = [c for c in coords.columns if c.startswith("pc")]
        assert len(pc_cols) >= 2, f"{path} has no pc columns"
        return coords, sorted(pc_cols, key=lambda c: int(c[2:])), False

    saved = torch.load(DEFAULT_CHECKPOINT, map_location="cpu", weights_only=False)
    embedding = saved["model"]["embedding.weight"].numpy().astype("float64")[1:]
    centered = embedding - embedding.mean(axis=0)
    pca = PCA(n_components=4, random_state=BOOT_SEED)
    pcs = pca.fit_transform(centered)
    coords = hitters[["batter", "embedding_index"]].copy()
    for k in range(4):
        coords[f"pc{k + 1}"] = pcs[:, k]
    return coords, [f"pc{k + 1}" for k in range(4)], True


def fig_head_fingerprints(coords, marginals_df, hitters, path):
    """2x3 small multiples: PCA(pc1,pc2) coloured by each head marginal."""
    merged = coords.merge(marginals_df, on=["batter", "embedding_index"]) \
                    .merge(hitters[["batter", "name"]], on="batter", how="left")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, marginal in zip(axes.flat, HEAD_MARGINALS):
        sc = ax.scatter(merged["pc1"], merged["pc2"], c=merged[marginal],
                        cmap="viridis", s=6, alpha=0.75)
        fig.colorbar(sc, ax=ax, label=marginal)
        anchors = merged[merged["batter"].isin(ANCHOR_IDS)]
        for _, row in anchors.iterrows():
            ax.scatter([row["pc1"]], [row["pc2"]], s=45, facecolors="none",
                      edgecolors="black", linewidths=1.2)
            ax.annotate(ANCHOR_IDS[row["batter"]], (row["pc1"], row["pc2"]),
                       fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.set_title(marginal)
        ax.set_xlabel("pc1"); ax.set_ylabel("pc2")
    fig.suptitle("Head marginals over the reference context set")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def build_loadings(coords, pc_cols, marginals_df, hitters):
    """Long-form (pc, variable, r_raw, r_partial, n) over the head marginals + stat panel."""
    merged = coords.merge(marginals_df, on=["batter", "embedding_index"]) \
                    .merge(hitters, on=["batter", "embedding_index"], suffixes=("", "_h"))
    variables = HEAD_MARGINALS + STAT_VARIABLES + ["log_prior_pa"]
    confound = merged["log_prior_pa"].values.astype("float64")
    rows = []
    for pc in pc_cols[:4]:
        pc_values = merged[pc].values.astype("float64")
        for variable in variables:
            values = merged[variable].values.astype("float64")
            r_raw, n = paired_pearson(pc_values, values)
            r_partial, _ = paired_pearson(pc_values, values, confound=confound)
            rows.append({"pc": pc, "variable": variable, "r_raw": r_raw,
                        "r_partial": r_partial, "n": n})
    return pd.DataFrame(rows), merged


def fig_loadings(loadings, pc_cols, path):
    pcs = pc_cols[:4]
    variables = HEAD_MARGINALS + STAT_VARIABLES + ["log_prior_pa"]
    raw = loadings.pivot(index="variable", columns="pc", values="r_raw").reindex(
        index=variables, columns=pcs)
    partial = loadings.pivot(index="variable", columns="pc", values="r_partial").reindex(
        index=variables, columns=pcs)
    fig, axes = plt.subplots(1, 2, figsize=(10, 8))
    for ax, frame, title in zip(axes, [raw, partial], ["raw", "partialled on log prior PA"]):
        im = ax.imshow(frame.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(pcs))); ax.set_xticklabels(pcs)
        ax.set_yticks(range(len(variables))); ax.set_yticklabels(variables, fontsize=8)
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                value = frame.values[i, j]
                if np.isfinite(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6)
        ax.set_title(title)
    fig.colorbar(im, ax=axes, label="Pearson r", shrink=0.6)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def preregistered_cells(merged):
    """The two spec V.3 expectation cells: contact-marginal vs contact_rate, EV vs ev_mean."""
    confound = merged["log_prior_pa"].values.astype("float64")
    cells = {"contact_vs_contact_rate": ("contact", "contact_rate"),
             "ev_vs_ev_mean": ("ev", "ev_mean")}
    out = {}
    for key, (marginal, stat) in cells.items():
        a = merged[marginal].values.astype("float64")
        b = merged[stat].values.astype("float64")
        r_raw, n_raw = paired_pearson(a, b)
        r_partial, lo, hi, n = bootstrap_ci_partial(a, b, confound)
        out[key] = {"n": n, "r_raw": r_raw, "r_partial": r_partial,
                   "ci_low": lo, "ci_high": hi,
                   "verdict": bool(r_partial > 0 and lo > 0)}
    return out


# --------------------------------------------------------------------- V.4

def build_level_query(hitters):
    predictions = pd.read_csv(LEVEL_QUERY_PREDICTIONS_PATH)
    level_query = predictions.groupby("batter")["pred_woba"].mean().rename("level_query")
    names = pd.read_csv(NAMES_PATH)[["batter", "embedding_index", "name", "stand"]]
    merged = (hitters[["batter", "stratum", "log_prior_pa", "woba_level"]]
              .merge(level_query, on="batter", how="inner")
              .merge(names, on="batter", how="left"))
    return merged[["batter", "name", "stand", "stratum", "log_prior_pa",
                  "level_query", "woba_level"]]


def fig_level_map(coords, level_query_df, hitters, path):
    merged = coords.merge(level_query_df, on="batter", how="inner")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, column, title in zip(axes, ["level_query", "woba_level"],
                                 ["model level query", "training wOBA level"]):
        lo, hi = np.nanpercentile(merged[column], [2, 98])
        sc = ax.scatter(merged["pc1"], merged["pc2"], c=merged[column],
                        cmap="viridis", vmin=lo, vmax=hi, s=6, alpha=0.75)
        fig.colorbar(sc, ax=ax, label=column)
        anchors = merged[merged["batter"].isin(ANCHOR_IDS)]
        for _, row in anchors.iterrows():
            ax.scatter([row["pc1"]], [row["pc2"]], s=45, facecolors="none",
                      edgecolors="black", linewidths=1.2)
            ax.annotate(ANCHOR_IDS[row["batter"]], (row["pc1"], row["pc2"]),
                       fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def level_query_stats(level_query_df):
    """Per-stratum + pooled Pearson (raw, partialled on log_prior_pa) with bootstrap CI."""
    out = {}
    confound_all = level_query_df["log_prior_pa"].values.astype("float64")
    a_all = level_query_df["level_query"].values.astype("float64")
    b_all = level_query_df["woba_level"].values.astype("float64")

    def _cell(a, b, confound):
        r_raw, _ = paired_pearson(a, b)
        r_partial, lo, hi, n = bootstrap_ci_partial(a, b, confound)
        return {"n": n, "r_raw": r_raw, "r_partial": r_partial, "ci_low": lo, "ci_high": hi}

    out["pooled"] = _cell(a_all, b_all, confound_all)
    for stratum, group in level_query_df.groupby("stratum"):
        cell = _cell(group["level_query"].values.astype("float64"),
                    group["woba_level"].values.astype("float64"),
                    group["log_prior_pa"].values.astype("float64"))
        if stratum == "high":
            cell["verdict"] = bool(cell["r_partial"] > 0 and cell["ci_low"] > 0)
        out[str(stratum)] = cell
    return out


# --------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description="Phase V.3/V.4 -- head fingerprints and level query map.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors, manifest = loader.load_tensors(args.data_dir)
    n_context = tensors["context"].shape[1]
    models = load_ensemble([args.checkpoint], manifest, n_context)
    model = models[0]
    assert not model.training, "model must be in eval mode"

    reference_idx = sample_reference_context(tensors, manifest)
    pd.DataFrame({"row_index": reference_idx}).to_csv(
        out_dir / "reference_context_index.csv", index=False)
    reference_context = tensors["context"][torch.from_numpy(reference_idx)].float()

    names = pd.read_csv(NAMES_PATH)
    hitters = names[names["embedding_index"] >= 1][["batter", "embedding_index"]].sort_values(
        "embedding_index").reset_index(drop=True)
    hitter_ids = hitters["embedding_index"].values

    started = time.monotonic()
    marginals = compute_head_marginals(model, hitter_ids, reference_context, manifest)
    print(f"head marginal computation: {time.monotonic() - started:.1f}s "
          f"({len(hitter_ids)} hitters x {len(reference_idx)} pitches)")

    marginals_df = hitters.copy()
    for name in HEAD_MARGINALS:
        marginals_df[name] = marginals[name]
    marginals_df.to_csv(out_dir / "head_marginals.csv", index=False)

    hitter_stats = pd.read_csv(HITTER_STATS_PATH)
    coords, pc_cols, fallback = get_or_compute_pca_coords(hitter_stats, out_dir)
    print(f"embedding_coords.csv {'was absent, computed PCA fallback' if fallback else 'found'}")

    fig_head_fingerprints(coords[["batter", "embedding_index", "pc1", "pc2"]],
                          marginals_df, names, out_dir / "fig_head_fingerprints.png")

    loadings, merged = build_loadings(coords, pc_cols, marginals_df, hitter_stats)
    loadings.to_csv(out_dir / "loadings.csv", index=False)
    fig_loadings(loadings, pc_cols, out_dir / "fig_loadings.png")

    cells = preregistered_cells(merged)
    (out_dir / "head_stat_loadings.json").write_text(json.dumps(cells, indent=2))

    level_query_df = build_level_query(hitter_stats)
    level_query_df.to_csv(out_dir / "level_query.csv", index=False)
    fig_level_map(coords[["batter", "pc1", "pc2"]], level_query_df, hitter_stats,
                 out_dir / "fig_level_map.png")
    level_stats = level_query_stats(level_query_df)
    (out_dir / "level_query.json").write_text(json.dumps(level_stats, indent=2))

    print(f"wrote {out_dir}/head_marginals.csv, loadings.csv, level_query.csv, "
          f"head_stat_loadings.json, level_query.json, fig_head_fingerprints.png, "
          f"fig_loadings.png, fig_level_map.png, reference_context_index.csv")
    print(f"preregistered cells: {json.dumps(cells)}")
    print(f"level query stats: {json.dumps(level_stats)}")


if __name__ == "__main__":
    main()
