"""
Post-V item 2, step 2 -- does an explicit debut prior help the low-stratum cold-start
population, on the frozen claim-1 metric? This module doesn't gate on step 1's
diagnostic (`cold_start_prior_diagnostic.py`) itself -- it just builds and scores the
two variants each side needs, for whoever reads both reports together.

Builds four claim-1 predictions frames -- the model with/without `cold_start_prior`
(query.predict, deliverable B), the EB baseline with/without a matching `debut_mu`
(baseline_ladder_bivariate_eb.predict, deliverable C) -- restricts each pair to the
LOW-STRATUM, ZERO-PRIOR-PA cold-start groups, and runs the frozen paired bootstraps
(`claim1_eval.paired_rmse_difference`, `paired_rank_difference`) on that restriction.
A comparison with too few matched groups for the paired bootstrap to resolve is
reported as "not resolvable", never crashed on.

MATCHING PRIOR (documented here since it cannot be inferred from either predict()).
Deliverable B substitutes, per hitter STAND, the low-stratum mean EMBEDDING vector
(`cold_start_prior_diagnostic.low_stratum_means`). The EB baseline has no embedding
space to match into, so the matching quantity for it is the low-stratum, within-stand
mean of the OBSERVED training wOBA (`hitter_stats.csv`'s `woba_level`) -- the same
low-stratum population B averages, just averaged in wOBA points instead of embedding
coordinates. EB's `debut_mu` is a (mu_vs_L, mu_vs_R) pair keyed by `batter_type`
(pitcher hand, not stand), and hitter_stats.csv keeps no pitcher-hand split for the
low-stratum population, so the single stand-matched scalar is used for BOTH mu
components. `batter_type` "S" (switch hitters) has no single matching stand; it is
matched to the combined-stand low-stratum mean.
# ponytail: the S-hitter match and the shared mu_L=mu_R value are the simplest
# defensible choices given what hitter_stats.csv carries, not a principled platoon
# split -- upgrade by adding a pitcher-hand-split low-stratum wOBA column if this
# comparison needs to be tighter.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import baseline_ladder_bivariate_eb as eb
from src.analysis import claim1_eval
from src.analysis.cold_start_prior_diagnostic import low_stratum_means
from src.model import loader, query, query_tables as qt

DEFAULT_CHECKPOINT_DIR = "results/checkpoints"
DEFAULT_ARM = "embedding_sgd_sgd_lr1"
DEFAULT_SEEDS = list(range(5))
DEFAULT_STATS_CSV = "results/model_visualization/hitter_stats.csv"
DEFAULT_OUT_DIR = "results/model_visualization"
DEFAULT_DATA_DIR = "data/processed/phase_d5"
DEFAULT_PITCH_EVENTS = "data/processed/pitch_events_labeled.parquet"
DEFAULT_EVAL_TARGETS = "data/processed/eval_targets_pa.parquet"
EVAL_SEASON = 2024


def ensemble_cold_start_prior(models, stats):
    """
    `query.predict`'s `cold_start_prior` takes ONE vector for the whole ensemble; each
    seed's embedding space is its own, so the low-stratum mean is computed inside each
    seed's own space (`low_stratum_means`) and the per-seed vectors are element-averaged
    into the single vector the API needs.
    # ponytail: cross-seed vector averaging isn't principled (unlike averaging query
    # OUTPUTS, which is how the ensemble already composes) -- upgrade to a per-model
    # cold_start_prior (list matching len(models)) in query.predict if that matters.
    """
    means = [low_stratum_means(model.embedding.weight.detach().numpy(), stats)
             for model in models]
    return {stand: np.mean([m[stand] for m in means], axis=0) for stand in ("L", "R")}


def matching_debut_mu(stats):
    """
    Low-stratum, within-stand mean observed training wOBA (`woba_level`) -- the EB
    counterpart of `ensemble_cold_start_prior`'s embedding-space match. Returns
    {"L": value, "R": value, "S": value}; "S" is the combined-stand mean (see module
    docstring).
    """
    low = stats[stats["stratum"] == "low"]
    by_stand = low.groupby("stand")["woba_level"].mean().to_dict()
    combined = float(low["woba_level"].mean())
    return {"L": float(by_stand.get("L", combined)), "R": float(by_stand.get("R", combined)),
            "S": combined}


def eb_debut_mu(stats):
    """
    `debut_mu` for `baseline_ladder_bivariate_eb.predict`: batter_type -> (mu_L, mu_R),
    both components the matched stand value (see module docstring).
    """
    matched = matching_debut_mu(stats)
    return {batter_type: (matched[batter_type], matched[batter_type])
            for batter_type in ("L", "R", "S")}


def _cold_start_groups(eval_frame):
    """Low-stratum, zero-prior-PA (batter, season, hand) rows of a built eval frame."""
    return eval_frame[(eval_frame["stratum"] == "low") & (eval_frame["prior_pa"] == 0)]


def _align(frame_a, frame_b, key_cols=claim1_eval.KEY):
    """
    Restrict both frames to their common key rows, in identical order, so
    `paired_rmse_difference`/`paired_rank_difference` (which require frame_a and
    frame_b to cover exactly the same groups) can run on them.
    """
    common = (frame_a[key_cols].merge(frame_b[key_cols], on=key_cols)
              .drop_duplicates().sort_values(key_cols).reset_index(drop=True))
    part_a = common.merge(frame_a, on=key_cols).sort_values(key_cols).reset_index(drop=True)
    part_b = common.merge(frame_b, on=key_cols).sort_values(key_cols).reset_index(drop=True)
    return part_a, part_b


def _compare(label, frame_a, frame_b, n_boot, seed):
    cold_a, cold_b = _cold_start_groups(frame_a), _cold_start_groups(frame_b)
    part_a, part_b = _align(cold_a, cold_b)
    if len(part_a) < 3:
        return {"resolvable": False, "reason": f"only {len(part_a)} matched cold-start groups"}, None
    try:
        rmse = claim1_eval.paired_rmse_difference(part_a, part_b, n_boot=n_boot, seed=seed)
        rank = claim1_eval.paired_rank_difference(part_a, part_b, n_boot=n_boot, seed=seed)
    except AssertionError as exc:
        return {"resolvable": False, "reason": str(exc)}, None
    rmse = rmse.copy(); rmse.insert(0, "comparison", label); rmse["metric"] = "rmse_difference"
    rank = rank.copy(); rank.insert(0, "comparison", label); rank["metric"] = "rank_difference"
    return {"resolvable": True, "n_cold_start_groups": int(len(part_a))}, pd.concat(
        [rmse, rank], ignore_index=True)


def run_eval(checkpoint_dir, arm, seeds, stats_csv, data_dir, pitch_events, eval_targets_path,
            eval_season=EVAL_SEASON, n_boot=2000, seed=0):
    tensors, manifest = loader.load_tensors(data_dir)
    frame = qt.align_pitch_frame(pitch_events, eval_targets_path, tensors["season"])
    pa_df = pd.read_parquet(eval_targets_path)
    tables = query.build_tables(frame, tensors, manifest, pa_df)
    stats = pd.read_csv(stats_csv)

    checkpoints = [Path(checkpoint_dir) / f"{arm}_s{s}.pt" for s in seeds]
    models = query.load_ensemble(checkpoints, manifest, tensors["context"].shape[1])
    cold_start_prior = ensemble_cold_start_prior(models, stats)

    model_base, _ = query.predict(models, tensors, manifest, frame, tables, pa_df, eval_season)
    model_prior, _ = query.predict(models, tensors, manifest, frame, tables, pa_df, eval_season,
                                   cold_start_prior=cold_start_prior)

    eb_params = eb.fit(pa_df, eval_season)
    eb_base = eb.predict(pa_df, eval_season, params=eb_params)
    debut_mu = eb_debut_mu(stats)
    eb_prior = eb.predict(pa_df, eval_season, params=eb_params, debut_mu=debut_mu)

    frame_model_base, _ = claim1_eval.build_eval_frame(pa_df, model_base, eval_season)
    frame_model_prior, _ = claim1_eval.build_eval_frame(pa_df, model_prior, eval_season)
    frame_eb_base, _ = claim1_eval.build_eval_frame(pa_df, eb_base, eval_season)
    frame_eb_prior, _ = claim1_eval.build_eval_frame(pa_df, eb_prior, eval_season)

    summary = {}
    tables_out = []
    for label, frame_a, frame_b in (
        ("model_prior_vs_baseline", frame_model_base, frame_model_prior),
        ("eb_prior_vs_baseline", frame_eb_base, frame_eb_prior),
    ):
        summary[label], table = _compare(label, frame_a, frame_b, n_boot, seed)
        if table is not None:
            tables_out.append(table)

    csv_out = pd.concat(tables_out, ignore_index=True) if tables_out else pd.DataFrame(
        columns=["comparison", "stratum", "metric"])
    result = {
        "eval_season": int(eval_season), "n_boot": n_boot, "seed": seed,
        "cold_start_prior_vector_norm": {stand: float(np.linalg.norm(vec))
                                         for stand, vec in cold_start_prior.items()},
        "eb_debut_mu": {bt: list(mu) for bt, mu in debut_mu.items()},
        "comparisons": summary,
    }
    return result, csv_out


def main():
    parser = argparse.ArgumentParser(
        description="Post-V item 2, step 2 -- cold-start debut prior claim-1 evaluation.")
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--arm", default=DEFAULT_ARM)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--stats-csv", default=DEFAULT_STATS_CSV)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--pitch-events", default=DEFAULT_PITCH_EVENTS)
    parser.add_argument("--eval-targets", default=DEFAULT_EVAL_TARGETS)
    parser.add_argument("--eval-season", type=int, default=EVAL_SEASON)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result, csv_out = run_eval(args.checkpoint_dir, args.arm, args.seeds, args.stats_csv,
                               args.data_dir, args.pitch_events, args.eval_targets,
                               args.eval_season, args.n_boot, args.seed)
    csv_path = out_dir / "cold_start_prior_eval.csv"
    json_path = out_dir / "cold_start_prior_eval.json"
    csv_out.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
