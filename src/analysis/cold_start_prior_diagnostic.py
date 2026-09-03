"""
Post-V item 2, step 1 -- the row-0 cold-start diagnostic.

Row 0 of the hitter embedding (`RESERVED_HITTER_INDEX`, src/data/model_dataset.py:61) is
the frozen origin every unseen hitter maps to (`nn.Embedding(..., padding_idx=0)`,
src/model/v1.py:91). Before touching the live scorer this asks one question: does the
level query (expected wOBA over a fixed reference context pool) separate when row 0 is
queried at the trained origin vs at the mean embedding vector of low-stratum (little
prior exposure) hitters, within each stand? If it doesn't separate, substituting a debut
prior into the scorer (deliverable B, `query.predict(..., cold_start_prior=...)`) has
nothing to substitute.

PER-SEED, NOT POOLED. Each seed's embedding table is its own trained space, so the
low-stratum mean vector is built and queried inside that seed's OWN model -- never
averaged with another seed's raw vector. What IS averaged across seeds is the resulting
scalar query OUTPUT (p_swing, p_contact, q) for each of (origin, L-mean, R-mean),
mirroring how the live ensemble already averages conditionals before composition
(`query.expected_woba`).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.analysis import claim1_eval
from src.analysis.model_visualization_anchors import query_quantities, sample_pool
from src.analysis.model_visualization_platoon import restore_embedding, set_row0
from src.data import eval_targets
from src.data.eval_targets import aggregate, drop_pitcher_batters
from src.data.model_dataset import RESERVED_HITTER_INDEX
from src.model import loader, query, query_tables as qt

DEFAULT_CHECKPOINT_DIR = "results/checkpoints"
DEFAULT_ARM = "d10_baseline"
DEFAULT_SEEDS = list(range(5))
DEFAULT_STATS_CSV = "results/model_visualization/hitter_stats.csv"
DEFAULT_OUT_DIR = "results/model_visualization"
DEFAULT_DATA_DIR = "data/processed/phase_d5"
DEFAULT_PITCH_EVENTS = "data/processed/pitch_events_labeled.parquet"
DEFAULT_EVAL_TARGETS = "data/processed/eval_targets_pa.parquet"
EVAL_SEASON = 2024
EVAL_SEASON_WEIGHTS = "2024"
N_PITCHES = 20000
N_BOOT = 500
BOOT_SEED = 0


def low_stratum_means(embeddings, stats):
    """
    Mean embedding row over LOW-stratum hitters, within each stand.
    `embeddings`: (n_hitters+1, D) for ONE model/seed -- never averaged across seeds,
    only the resulting query output is (see module docstring). `stats`: the
    hitter_stats.csv frame (batter, embedding_index, stand, stratum, ...).
    Returns {"L": vec, "R": vec}.
    """
    low = stats[stats["stratum"] == "low"]
    return {stand: embeddings[group["embedding_index"].to_numpy()].mean(axis=0)
            for stand, group in low.groupby("stand")}


def _low_stratum_rows(stats, stand):
    match = (stats["stratum"] == "low") & (stats["stand"] == stand)
    return stats.loc[match, "embedding_index"].to_numpy()


def level_query_at(model, kernel, points, n_bins, context_pool, vector):
    """
    p_swing, p_contact, q with row 0 of `model`'s embedding table temporarily set to
    `vector`, queried over `context_pool` (a single-model call through
    `query.expected_woba`). Restores row 0 afterward -- the checkpoint in memory is
    never left mutated, even if the query raises.
    """
    saved = set_row0(model, vector)
    try:
        swing, contact, q = query_quantities([model], [kernel], points, n_bins,
                                             RESERVED_HITTER_INDEX, context_pool)
    finally:
        restore_embedding(model, saved)
    return {"p_swing": float(swing.mean()), "p_contact": float(contact.mean()),
            "q": float(q.mean())}


def _seed_checkpoint(checkpoint_dir, arm, seed):
    return Path(checkpoint_dir) / f"{arm}_s{seed}.pt"


def _build_context_pool(frame, tensors, manifest, n_pitches, seed=0):
    """
    One reference pool of training-season pitches, held fixed across the origin/L-mean/
    R-mean queries so pool noise cancels in the differences between them. This is a
    level query, not the live per-(pitcher hand, batter stand) scorer, so pitches are
    drawn across both p_throws values rather than gated by hand.
    """
    train_seasons = manifest["train_seasons"]
    idx = np.concatenate([sample_pool(frame, train_seasons, hand, n_pitches // 2, seed)
                          for hand in ("L", "R")])
    return tensors["context"][torch.as_tensor(idx, dtype=torch.long)]


def cold_start_group_count(pa_df, eval_season=EVAL_SEASON):
    """
    Count of low-stratum, zero-prior-PA (batter, hand) groups in the claim-1 eval frame
    for `eval_season` -- the population deliverable B's substitution would actually
    touch. Reuses `claim1_eval.build_eval_frame` (the frozen eval-frame construction)
    rather than re-deriving the stratum/coverage logic. This diagnostic has no model
    predictions to merge, so a placeholder pred_woba is supplied for every scorable
    group (build_eval_frame's population/coverage logic doesn't read prediction
    values, only which groups survive the min-PA filter and the join).
    """
    hitters = drop_pitcher_batters(pa_df)
    actuals = aggregate(hitters, by=tuple(claim1_eval.KEY))
    actuals = actuals[actuals["season"] == eval_season]
    placeholder = actuals[claim1_eval.KEY].copy()
    placeholder["pred_woba"] = 0.0
    frame, coverage = claim1_eval.build_eval_frame(pa_df, placeholder, eval_season)
    zero_prior = frame[(frame["stratum"] == "low") & (frame["prior_pa"] == 0)]
    by_hand = zero_prior.groupby("p_throws").size().to_dict()
    return {"total": int(len(zero_prior)),
            "by_hand": {hand: int(n) for hand, n in by_hand.items()},
            "coverage": coverage}


def run_diagnostic(checkpoint_dir, arm, seeds, stats_csv, data_dir,
                   pitch_events=DEFAULT_PITCH_EVENTS, eval_targets_path=DEFAULT_EVAL_TARGETS,
                   n_pitches=N_PITCHES, n_boot=N_BOOT):
    tensors, manifest = loader.load_tensors(data_dir)
    frame = qt.align_pitch_frame(pitch_events, eval_targets_path, tensors["season"])
    pa_df = pd.read_parquet(eval_targets_path)
    tables = query.build_tables(frame, tensors, manifest, pa_df)
    weights = eval_targets.load_weights()[EVAL_SEASON_WEIGHTS]
    points = torch.from_numpy(qt.woba_points_table(tables["outcome"], weights)).float()
    n_bins = manifest["n_quality_bins"]

    stats = pd.read_csv(stats_csv)
    context_pool = _build_context_pool(frame, tensors, manifest, n_pitches)
    n_low = {stand: len(_low_stratum_rows(stats, stand)) for stand in ("L", "R")}

    per_seed = {"origin": [], "L": [], "R": []}
    boot_diffs = {"L": [], "R": []}
    rng = np.random.default_rng(BOOT_SEED)

    for path in (_seed_checkpoint(checkpoint_dir, arm, s) for s in seeds):
        models = query.load_ensemble([path], manifest, tensors["context"].shape[1])
        model = models[0]
        kernel = query.spray_kernels(model, points, n_bins, tables["spray_mass"])
        embeddings = model.embedding.weight.detach().numpy()

        means = low_stratum_means(embeddings, stats)
        origin_vec = embeddings[RESERVED_HITTER_INDEX]
        origin_query = level_query_at(model, kernel, points, n_bins, context_pool, origin_vec)
        per_seed["origin"].append(origin_query)

        for stand in ("L", "R"):
            per_seed[stand].append(
                level_query_at(model, kernel, points, n_bins, context_pool, means[stand]))
            rows = _low_stratum_rows(stats, stand)
            for _ in range(n_boot):
                draw = rng.choice(rows, size=len(rows), replace=True)
                vec = embeddings[draw].mean(axis=0)
                q = level_query_at(model, kernel, points, n_bins, context_pool, vec)["q"]
                boot_diffs[stand].append(q - origin_query["q"])

    def _avg(records, key):
        return float(np.mean([r[key] for r in records]))

    origin_summary = {key: _avg(per_seed["origin"], key) for key in ("p_swing", "p_contact", "q")}
    result = {
        "n_seeds": len(list(seeds)), "n_pitches": n_pitches,
        "origin": origin_summary,
        "cold_start_groups": cold_start_group_count(pa_df),
    }
    for stand in ("L", "R"):
        summary = {key: _avg(per_seed[stand], key) for key in ("p_swing", "p_contact", "q")}
        diffs = np.asarray(boot_diffs[stand])
        result[stand] = {
            "level_query": summary,
            "difference_from_origin": {k: summary[k] - origin_summary[k] for k in summary},
            "n_low_stratum_rows": n_low[stand],
            "bootstrap_q_difference_ci": {
                "n_draws": int(len(diffs)), "seed": BOOT_SEED,
                "low": float(np.percentile(diffs, 2.5)),
                "high": float(np.percentile(diffs, 97.5)),
                "mean": float(diffs.mean())},
        }
    return result


def main():
    parser = argparse.ArgumentParser(description="Post-V item 2, step 1 -- cold-start level query.")
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--arm", default=DEFAULT_ARM)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--stats-csv", default=DEFAULT_STATS_CSV)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--pitch-events", default=DEFAULT_PITCH_EVENTS)
    parser.add_argument("--eval-targets", default=DEFAULT_EVAL_TARGETS)
    parser.add_argument("--n-pitches", type=int, default=N_PITCHES)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_diagnostic(args.checkpoint_dir, args.arm, args.seeds, args.stats_csv,
                            args.data_dir, args.pitch_events, args.eval_targets,
                            args.n_pitches, args.n_boot)
    out_path = out_dir / "cold_start_level_query.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
