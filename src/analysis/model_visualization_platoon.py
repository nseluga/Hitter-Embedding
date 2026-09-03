"""
Phase V — V.5, the platoon direction. Shows whether the embedding table encodes a
consistent left/right platoon-skill direction: a gradient direction found by perturbing
the trained embedding table and re-scoring through the real query path, set beside the
direction the E.14 ridge probe already found by decoding the empirical-Bayes split from
the same table.

(a) finite-differences the L-R query gap on the real query (`src/model/query.py::predict`,
not a torch proxy), one dimension at a time, with the pitcher panel held fixed (same
`n_pitchers=128, seed=0`) so panel noise cancels out of the difference. (b) projects every
trained hitter onto that direction and correlates it against each hitter's TRAINING-SEASON
observed L-R wOBA differential (never an eval-season number), partialled on exposure.
Nothing here grades against 2025 or promotes a cluster (spec §0.3, §0.5).

Two CLI stages: `gradient` (the ~2h finite-difference run) and `analyse` (fast, reads the
raw file `gradient` wrote). Run: python -m src.analysis.model_visualization_platoon gradient
--out-dir results/model_visualization; then ... analyse --raw <out>/platoon_gradient_raw.csv
--out-dir results/model_visualization
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
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis import baseline_ladder_bivariate_eb
from src.analysis.model_evaluation_probe_coverage import RIDGE_ALPHAS, load_seed_embeddings
from src.analysis.model_visualization_embeddings import bootstrap_ci
from src.analysis.model_visualization_stats import ANCHORS
from src.model import loader, query, query_tables as qt

DEFAULT_DATA_DIR = "data/processed/phase_d5"
PITCH_EVENTS = "data/processed/pitch_events_labeled.parquet"
EVAL_TARGETS = "data/processed/eval_targets_pa.parquet"
CHECKPOINT_DIR = "results/checkpoints"
ARM = "d10_baseline"
PLATOON_FRAME = "results/model_evaluation/platoon_frame.csv"
HITTER_STATS = "results/model_visualization/hitter_stats.csv"
NAMES_PATH = "data/processed/hitter_names.csv"
EVAL_SEASON = 2024

ANCHOR_IDS = list(ANCHORS)  # Trout, Soto, Pederson, Bohm, Schwarber
N_OTHER = 35
N_DIMS = 32
DEFAULT_N_PITCHERS = 128
BASE_PASS = "base"
LINEARITY_PASS = "dim0_2x"
N_BOOT = 1000
BOOT_SEED = 0


# --------------------------------------------------------------------- subset

def build_subset(platoon_frame_path, anchor_ids, n_other, seed):
    """
    The gradient hitter subset: the named anchors (dropping any absent from the L/R-scorable
    platoon frame) plus `n_other` more, drawn stratified over stand x stratum with
    `np.random.default_rng(seed)`. Returns (subset DataFrame, anchors present, anchors missing).
    """
    frame = pd.read_csv(platoon_frame_path)
    present = set(frame["batter"].astype(int))
    anchors_present = [b for b in anchor_ids if b in present]
    anchors_missing = [b for b in anchor_ids if b not in present]

    pool = frame[~frame["batter"].isin(anchor_ids)][["batter", "stand", "stratum"]].drop_duplicates("batter")
    cells = sorted(pool.groupby(["stand", "stratum"]).groups.keys())
    base, extra = divmod(n_other, len(cells))
    rng = np.random.default_rng(seed)
    picks = []
    for i, (stand, stratum) in enumerate(cells):
        n_cell = base + (1 if i < extra else 0)
        cell_pool = pool[(pool["stand"] == stand) & (pool["stratum"] == stratum)]["batter"].to_numpy()
        n_cell = min(n_cell, len(cell_pool))
        picks.extend(rng.choice(cell_pool, size=n_cell, replace=False).tolist())

    subset_ids = anchors_present + picks
    subset = frame[frame["batter"].isin(subset_ids)][["batter", "stand", "stratum"]].drop_duplicates("batter").copy()
    subset["is_anchor"] = subset["batter"].isin(anchor_ids)
    return subset.reset_index(drop=True), anchors_present, anchors_missing


# --------------------------------------------------------------------- gradient passes

def parse_dims(spec):
    """'0-31' or '3,7,9' -> a sorted list of int dims."""
    if spec is None:
        return list(range(N_DIMS))
    if "-" in spec and "," not in spec:
        lo, hi = (int(x) for x in spec.split("-"))
        return list(range(lo, hi + 1))
    return sorted(int(x) for x in spec.split(","))


def build_pass_list(dims):
    """Base pass, one pass per selected dim, plus the dim-0 linearity check if dim 0 is selected."""
    passes = [(BASE_PASS, None, 0.0)]
    passes += [(f"dim{d}", d, None) for d in dims]  # eps filled in by the caller
    if 0 in dims:
        passes.append((LINEARITY_PASS, 0, None))
    return passes


def perturb_embedding(model, dim, eps):
    """
    Add `eps` to embedding column `dim` for every TRAINED row (row 0, cold start, untouched).
    Returns the exact pre-perturbation tensor, to be restored via `restore_embedding`.
    """
    weight = model.embedding.weight
    saved = weight.detach().clone()
    with torch.no_grad():
        weight[1:, dim] += eps
    return saved


def restore_embedding(model, saved):
    """Undo `perturb_embedding` and assert the table is back bit-identical."""
    with torch.no_grad():
        model.embedding.weight.copy_(saved)
    assert torch.equal(model.embedding.weight, saved), "embedding restore did not round-trip"


def pending_passes(raw_path, all_pass_ids):
    """Pass ids from `all_pass_ids` not yet present in the raw CSV at `raw_path` (resume)."""
    path = Path(raw_path)
    if not path.exists():
        return list(all_pass_ids)
    done = set(pd.read_csv(path, usecols=["pass_id"])["pass_id"].unique())
    return [pass_id for pass_id in all_pass_ids if pass_id not in done]


def run_gradient(out_dir, n_pitchers, dims, passes_filter, resume, smoke):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "platoon_gradient_raw.csv"

    anchor_ids = ANCHOR_IDS[:2] if smoke else ANCHOR_IDS
    n_other = 0 if smoke else N_OTHER
    n_pitchers = 8 if smoke else n_pitchers
    dims = [0, 1] if smoke else dims

    subset, anchors_present, anchors_missing = build_subset(PLATOON_FRAME, anchor_ids, n_other, seed=0)
    if anchors_missing:
        print(f"anchors dropped, not in {PLATOON_FRAME}: {anchors_missing}")
    subset.to_csv(out_dir / "platoon_gradient_subset.csv", index=False)
    subset_ids = subset["batter"].tolist()
    print(f"subset: {len(subset_ids)} hitters ({len(anchors_present)} anchors)")

    print("loading tensors, pitch frame, and the seed-0 checkpoint")
    tensors, manifest = loader.load_tensors(DEFAULT_DATA_DIR)
    frame = qt.align_pitch_frame(PITCH_EVENTS, EVAL_TARGETS, tensors["season"])
    pa_df = pd.read_parquet(EVAL_TARGETS)
    tables = query.build_tables(frame, tensors, manifest, pa_df)
    models = query.load_ensemble([Path(CHECKPOINT_DIR) / f"{ARM}_s0.pt"], manifest,
                                 tensors["context"].shape[1])
    model = models[0]
    assert not model.training, "embedding.eval() was not applied by load_ensemble"

    base_weight = model.embedding.weight.detach()
    eps = float(0.25 * base_weight[1:].std(0).mean())
    print(f"eps = {eps:.6f}")

    all_passes = build_pass_list(dims)
    eps_of = {BASE_PASS: 0.0, LINEARITY_PASS: 2 * eps}
    all_passes = [(pass_id, dim, eps_of.get(pass_id, eps)) for pass_id, dim, _ in all_passes]
    if passes_filter is not None:
        all_passes = [p for p in all_passes if p[0] in passes_filter]

    todo_ids = pending_passes(raw_path, [p[0] for p in all_passes]) if resume else [p[0] for p in all_passes]
    todo = [p for p in all_passes if p[0] in todo_ids]
    if resume and len(todo) < len(all_passes):
        print(f"resume: {len(all_passes) - len(todo)} pass(es) already complete, skipping")

    for pass_id, dim, pass_eps in todo:
        saved = perturb_embedding(model, dim, pass_eps) if dim is not None and pass_eps != 0.0 else None
        started = time.monotonic()
        predictions, _ = query.predict(models, tensors, manifest, frame, tables, pa_df,
                                       EVAL_SEASON, n_pitchers=n_pitchers, seed=0,
                                       batters=subset_ids)
        elapsed = time.monotonic() - started
        if saved is not None:
            restore_embedding(model, saved)

        rows = predictions[["batter", "p_throws", "pred_woba"]].copy()
        rows.insert(0, "eps", pass_eps)
        rows.insert(0, "dim", -1 if dim is None else dim)
        rows.insert(0, "pass_id", pass_id)
        rows.to_csv(raw_path, mode="a", header=not raw_path.exists(), index=False)
        print(f"pass {pass_id}: {elapsed:.1f}s ({len(rows)} rows written)")

    return raw_path, eps


# --------------------------------------------------------------------- gradient/direction

def finite_difference_gradient(gap_pass, gap_base, eps):
    """The forward-difference slope: (gap at the perturbed table - gap at base) / eps."""
    return (gap_pass - gap_base) / eps


def gap_by_batter(raw, pass_id):
    """pred_woba(L) - pred_woba(R) per batter for one pass, as a batter-indexed Series."""
    rows = raw[raw["pass_id"] == pass_id]
    wide = rows.pivot(index="batter", columns="p_throws", values="pred_woba")
    return wide["L"] - wide["R"]


def compute_gradient_table(raw, dims=range(N_DIMS)):
    """
    Per-hitter, per-dim gradient g[k] = (gap_k - gap_0) / eps_k. Dims absent from `raw`
    come back all-NaN. Returns a DataFrame (batter, dim, gradient).
    """
    gap_base = gap_by_batter(raw, BASE_PASS)
    present_dims = set(raw.loc[raw["dim"] >= 0, "dim"].unique())
    rows = []
    for dim in dims:
        pass_id = f"dim{dim}"
        if dim not in present_dims:
            rows.append(pd.DataFrame({"batter": gap_base.index, "dim": dim, "gradient": np.nan}))
            continue
        eps = raw.loc[raw["pass_id"] == pass_id, "eps"].iloc[0]
        gap_k = gap_by_batter(raw, pass_id)
        gradient = finite_difference_gradient(gap_k, gap_base.reindex(gap_k.index), eps)
        rows.append(pd.DataFrame({"batter": gradient.index, "dim": dim, "gradient": gradient.to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def linearity_ratio(raw):
    """Dim-0 slope at 2*eps over the slope at eps, averaged over hitters. NaN if pass 33 is absent."""
    if LINEARITY_PASS not in set(raw["pass_id"]):
        return float("nan")
    gap_base = gap_by_batter(raw, BASE_PASS)
    eps = raw.loc[raw["pass_id"] == "dim0", "eps"].iloc[0]
    eps2 = raw.loc[raw["pass_id"] == LINEARITY_PASS, "eps"].iloc[0]
    slope_1x = finite_difference_gradient(gap_by_batter(raw, "dim0"), gap_base, eps)
    slope_2x = finite_difference_gradient(gap_by_batter(raw, LINEARITY_PASS), gap_base, eps2)
    return float(np.nanmean(slope_2x) / np.nanmean(slope_1x))


def ridge_direction(pa_df, manifest, embedding):
    """
    E.14's ridge weight vector, refit here because the committed
    `results/model_evaluation/coverage_probe.json` carries no weights. Reproduces
    `model_evaluation_probe_coverage.eb_posterior_split` / `probe_frame`'s logic inline
    rather than importing them: that module's `eb_posterior_split` raises NameError (it
    imports the EB scorer as `baseline_ladder_bivariate_eb` but calls `eb_bivariate_eb`),
    and this module does not modify existing files.

    features = seed-0 embedding.weight at each trained hitter's row; target = the
    empirical-Bayes posterior L-R split (`baseline_ladder_bivariate_eb.predict`). Fit on
    all rows, no CV split needed for a direction. Returns the coefficient converted back
    to raw-embedding units (coef / scaler.scale_).
    """
    predictions = baseline_ladder_bivariate_eb.predict(pa_df, EVAL_SEASON)
    assert set(predictions["p_throws"]) <= {"L", "R"}, "unexpected pitcher hand"
    wide = predictions.pivot(index="batter", columns="p_throws", values="pred_woba").dropna(subset=["L", "R"])
    target = pd.DataFrame({"batter": wide.index.astype(int).to_numpy(),
                           "true_split": (wide["L"] - wide["R"]).to_numpy()})

    vocabulary = {int(batter): int(row) for batter, row in manifest["vocabulary"].items()}
    target["row"] = target["batter"].map(vocabulary)
    trained = target[target["row"].notna()].copy()
    trained["row"] = trained["row"].astype(int)
    assert (trained["row"] > 0).all(), "a trained hitter mapped to the reserved row"

    features = embedding[trained["row"].to_numpy()]
    pipeline = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
    pipeline.fit(features, trained["true_split"].to_numpy())
    scaler, ridge = pipeline.named_steps["standardscaler"], pipeline.named_steps["ridgecv"]
    return ridge.coef_ / scaler.scale_


def cosine_finite(a, b):
    """Cosine similarity restricted to dims finite in both vectors. NaN if none are."""
    mask = np.isfinite(a) & np.isfinite(b)
    if not mask.any():
        return float("nan")
    av, bv = a[mask], b[mask]
    return float(np.dot(av, bv) / (np.linalg.norm(av) * np.linalg.norm(bv)))


def direction_analysis(raw, pa_df, manifest, embedding, out_dir):
    """(a): g_bar, g_bar_L/R, the ridge vector, and their bootstrapped cosine."""
    gradient_long = compute_gradient_table(raw)
    # stand comes from the subset file written alongside the raw CSV
    subset = pd.read_csv(out_dir / "platoon_gradient_subset.csv")
    gradient_long = gradient_long.merge(subset[["batter", "stand", "stratum"]], on="batter", how="left")

    matrix = gradient_long.pivot(index="batter", columns="dim", values="gradient").reindex(columns=range(N_DIMS))
    g_bar = np.nanmean(matrix.to_numpy(), axis=0)
    stand_of = subset.set_index("batter")["stand"]
    g_bar_l = np.nanmean(matrix.loc[matrix.index.isin(stand_of[stand_of == "L"].index)].to_numpy(), axis=0)
    g_bar_r = np.nanmean(matrix.loc[matrix.index.isin(stand_of[stand_of == "R"].index)].to_numpy(), axis=0)

    ridge_w = ridge_direction(pa_df, manifest, embedding)
    cosine, ci_lo, ci_hi = bootstrap_ci(
        matrix.to_numpy(), lambda gm: cosine_finite(np.nanmean(gm, axis=0), ridge_w),
        n_boot=N_BOOT, seed=BOOT_SEED)
    ratio = linearity_ratio(raw)

    direction = pd.DataFrame({"dim": range(N_DIMS), "g_bar": g_bar, "g_bar_L": g_bar_l,
                              "g_bar_R": g_bar_r, "ridge_w": ridge_w})
    direction.to_csv(out_dir / "platoon_direction.csv", index=False)

    gradient_out = gradient_long[["batter", "stand", "stratum", "dim", "gradient"]]
    gradient_out.to_csv(out_dir / "platoon_gradient.csv", index=False)

    plot_direction(g_bar, ridge_w, cosine, out_dir / "fig_platoon_direction.png")

    verdict = bool(cosine > 0 and ci_lo > 0)
    summary = {"cosine": cosine, "cosine_ci": [ci_lo, ci_hi], "verdict": verdict,
              "linearity_ratio_dim0": ratio, "n_subset_hitters": int(matrix.shape[0])}
    print(f"cosine(g_bar, ridge_w) = {cosine:.4f} [{ci_lo:.4f}, {ci_hi:.4f}], verdict={verdict}")
    print(f"linearity ratio (dim 0, 2eps slope / eps slope) = {ratio:.4f}")
    return g_bar, summary


def plot_direction(g_bar, ridge_w, cosine, out_path):
    def unit(v):
        mask = np.isfinite(v)
        out = np.zeros_like(v)
        norm = np.linalg.norm(v[mask])
        out[mask] = v[mask] / norm if norm > 0 else 0.0
        out[~mask] = np.nan
        return out

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), dpi=130, sharex=True)
    dims = np.arange(len(g_bar))
    axes[0].bar(dims, unit(g_bar), color="#4c8dff")
    axes[0].set_title(f"finite-difference gradient (unit norm), cosine = {cosine:.3f}")
    axes[1].bar(dims, unit(ridge_w), color="#e0574a")
    axes[1].set_title("E.14 ridge weight (unit norm)")
    axes[1].set_xlabel("embedding dim")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------------------------- projection

def partial_correlation(x, y, z):
    """Pearson r between x and y after removing their linear dependence on z."""
    x, y, z = (np.asarray(v, dtype="float64") for v in (x, y, z))

    def residual(a):
        slope, intercept = np.polyfit(z, a, 1)
        return a - (slope * z + intercept)

    return float(pearsonr(residual(x), residual(y))[0])


def load_projection_frame(embedding, g_bar):
    """
    Every trained hitter, projected onto g_bar and joined to training-season stats and
    names. `obs_platoon_diff` in hitter_stats.csv is oriented by stand (positive = better
    vs the opposite hand), not raw L-R: for a RHB it IS L-R, for a LHB it is the negative
    of L-R. Converted here to a common L-R sign before anything is correlated against the
    gradient direction, which is defined L-R. Switch-hitters ("S") have no fixed
    orientation and are dropped.
    """
    mask = np.isfinite(g_bar)
    unit = g_bar.copy()
    unit[mask] = g_bar[mask] / np.linalg.norm(g_bar[mask])
    unit[~mask] = 0.0

    stats = pd.read_csv(HITTER_STATS)
    names = pd.read_csv(NAMES_PATH)[["batter", "name"]]

    n_switch = int((stats["stand"] == "S").sum())
    if n_switch:
        print(f"dropping {n_switch} switch-hitter(s) (stand == 'S'), no fixed L-R orientation")
    stats = stats[stats["stand"] != "S"].copy()

    n_no_split = int(stats["obs_platoon_diff"].isna().sum())
    if n_no_split:
        # low-exposure hitters missing one side entirely have no L-R differential to correlate
        print(f"dropping {n_no_split} hitter(s) with no obs_platoon_diff (all in the low stratum)")
    stats = stats[stats["obs_platoon_diff"].notna()].copy()

    stats["obs_lr"] = np.where(stats["stand"] == "L", -stats["obs_platoon_diff"], stats["obs_platoon_diff"])
    stats["proj"] = embedding[stats["embedding_index"].to_numpy()] @ unit
    frame = stats.merge(names, on="batter", how="left")
    frame["is_anchor"] = frame["batter"].isin(ANCHOR_IDS)
    return frame


def projection_analysis(embedding, g_bar, out_dir):
    frame = load_projection_frame(embedding, g_bar)
    frame.to_csv(out_dir / "platoon_projection.csv", index=False)

    correlations = {}
    for stratum, group in list(frame.groupby("stratum")) + [("pooled", frame)]:
        raw_r = float(pearsonr(group["proj"], group["obs_lr"])[0])
        partial_r = partial_correlation(group["proj"], group["obs_lr"], group["log_prior_pa"])

        def statistic(proj, obs, exposure):
            return partial_correlation(proj, obs, exposure)

        point, lo, hi = bootstrap_ci(
            (group["proj"].to_numpy(), group["obs_lr"].to_numpy(), group["log_prior_pa"].to_numpy()),
            statistic, n_boot=N_BOOT, seed=BOOT_SEED)
        correlations[str(stratum)] = {"n": int(len(group)), "raw": raw_r, "partial": partial_r,
                                      "partial_ci": [lo, hi]}

    high = correlations["high"]
    projection_verdict = bool(high["partial"] > 0 and high["partial_ci"][0] > 0)
    print(f"projection, high stratum: partial r = {high['partial']:.4f} "
         f"[{high['partial_ci'][0]:.4f}, {high['partial_ci'][1]:.4f}], verdict={projection_verdict}")

    plot_projection(frame, out_dir / "fig_platoon_projection.png")
    return frame, correlations, projection_verdict


def plot_projection(frame, out_path):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), dpi=130, sharey=True)
    panels = [("low", frame[frame["stratum"] == "low"]),
             ("medium", frame[frame["stratum"] == "medium"]),
             ("high", frame[frame["stratum"] == "high"]),
             ("pooled", frame)]
    for ax, (title, group) in zip(axes, panels):
        ax.scatter(group["proj"], group["obs_lr"], s=14, alpha=0.5, color="#4c8dff")
        anchors = group[group["is_anchor"]]
        ax.scatter(anchors["proj"], anchors["obs_lr"], s=40, color="#e0574a", zorder=3)
        for _, row in anchors.iterrows():
            ax.annotate(row["name"], (row["proj"], row["obs_lr"]), fontsize=7)
        ax.set_title(title)
        ax.set_xlabel("projection on g_bar")
    axes[0].set_ylabel("observed L-R wOBA")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------------------------- spread

def spread_verdict(sd_low, sd_high, ci_low_of_diff):
    """(c)'s pass condition: the high stratum spreads out more than the low, and the CI
    on that gap excludes zero."""
    return bool(sd_low < sd_high and ci_low_of_diff > 0)


def spread_analysis(frame, out_dir):
    by_stratum = {s: g["proj"].to_numpy() for s, g in frame.groupby("stratum")}
    rng = np.random.default_rng(BOOT_SEED)

    def sd_ci(values):
        return bootstrap_ci(values, np.std, n_boot=N_BOOT, seed=BOOT_SEED)

    sds = {s: sd_ci(v) for s, v in by_stratum.items()}

    diffs = np.empty(N_BOOT)
    low, high = by_stratum["low"], by_stratum["high"]
    for b in range(N_BOOT):
        diffs[b] = np.std(rng.choice(high, len(high), replace=True)) - np.std(rng.choice(low, len(low), replace=True))
    diff_lo, diff_hi = np.percentile(diffs, [2.5, 97.5])

    verdict = spread_verdict(sds["low"][0], sds["high"][0], diff_lo)
    print(f"spread: SD low = {sds['low'][0]:.4f}, SD high = {sds['high'][0]:.4f}, "
         f"diff CI = [{diff_lo:.4f}, {diff_hi:.4f}], verdict={verdict}")

    plot_spread(by_stratum, out_dir / "fig_platoon_spread.png")
    return {"sd_by_stratum": {s: {"point": p, "ci": [lo, hi]} for s, (p, lo, hi) in sds.items()},
           "diff_high_minus_low_ci": [float(diff_lo), float(diff_hi)], "verdict": verdict}


def plot_spread(by_stratum, out_path):
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=130)
    order = ["low", "medium", "high"]
    ax.boxplot([by_stratum[s] for s in order if s in by_stratum],
              tick_labels=[s for s in order if s in by_stratum])
    ax.set_ylabel("projection on g_bar")
    ax.set_title("platoon-direction projection spread by exposure stratum")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------------------------- CLI

def run_analyse(raw_path, out_dir):
    out_dir = Path(out_dir)
    raw = pd.read_csv(raw_path)
    present_dims = sorted(int(d) for d in raw.loc[raw["dim"] >= 0, "dim"].unique())
    missing_dims = sorted(set(range(N_DIMS)) - set(present_dims))
    if missing_dims:
        print(f"dims absent from raw file, filled NaN: {missing_dims}")

    tensors, manifest = loader.load_tensors(DEFAULT_DATA_DIR)
    pa_df = pd.read_parquet(EVAL_TARGETS)
    embedding = load_seed_embeddings(CHECKPOINT_DIR, ARM, seeds=[0])[0]

    g_bar, direction_summary = direction_analysis(raw, pa_df, manifest, embedding, out_dir)
    frame, correlations, projection_verdict = projection_analysis(embedding, g_bar, out_dir)
    spread_summary = spread_analysis(frame, out_dir)

    (out_dir / "platoon_projection.json").write_text(json.dumps(
        {"correlations": correlations, "verdict_high_stratum": projection_verdict,
         "spread": spread_summary, "direction": direction_summary}, indent=2))
    print(f"wrote {out_dir / 'platoon_projection.json'}")


def main():
    parser = argparse.ArgumentParser(description="V.5 — the platoon direction.")
    sub = parser.add_subparsers(dest="stage", required=True)

    gradient = sub.add_parser("gradient")
    gradient.add_argument("--out-dir", required=True)
    gradient.add_argument("--n-pitchers", type=int, default=DEFAULT_N_PITCHERS)
    gradient.add_argument("--dims", default=None, help="e.g. '0-31' or '3,7,9'")
    gradient.add_argument("--passes", nargs="*", default=None,
                          help="pass ids to run, e.g. 'base' for the base-only timing run")
    gradient.add_argument("--resume", action="store_true")
    gradient.add_argument("--smoke", action="store_true")

    analyse = sub.add_parser("analyse")
    analyse.add_argument("--raw", required=True)
    analyse.add_argument("--out-dir", required=True)

    args = parser.parse_args()
    if args.stage == "gradient":
        dims = parse_dims(args.dims)
        run_gradient(args.out_dir, args.n_pitchers, dims, args.passes, args.resume, args.smoke)
    else:
        run_analyse(args.raw, args.out_dir)


if __name__ == "__main__":
    main()
