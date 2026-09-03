"""
Post-V item 1 read-out for one training arm: does the embedding norm still fall with
exposure, and does the arm's `reference` clear the incumbent bar?

Per seed: OLS slope and Pearson r of row L2 norm vs `log_prior_pa` (hitter bootstrap CI),
plus the norm sd within each stratum (V.5c proxy: low <= high). Ledger bar: the arm's
`reference` mean must sit within MARGIN_SES standard errors of the incumbent, one-sided,
`se = floor_sd * sqrt(1/n_arm + 1/n_incumbent)` with `floor_sd` the rebuild-baseline
across-seed sd -- the same rule as `hyperparameter_tuning_select`, reused not copied.

Writes `<out_dir>/embedding_norm_check_<arm>.json`. Point it at the scratchpad during
the one-seed screen; only the five-seed run writes under results/.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.hyperparameter_tuning_select import (
    MARGIN_SES, ledger_rows, noise_floor, references)
from src.analysis.model_evaluation_probe_coverage import load_seed_embeddings
from src.analysis.model_visualization_embeddings import bootstrap_ci

HITTER_STATS = "results/model_visualization/hitter_stats.csv"
STRATA = ("low", "medium", "high")


def slope(x, y):
    return float(np.polyfit(x, y, 1)[0])


def pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def norm_exposure(weight, stats):
    """One seed: {slope, slope_ci95, r, r_ci95, spread: {stratum: norm sd}}.
    `stats` rows map `embedding_index` -> `log_prior_pa`, `stratum`; row 0 is never used."""
    rows = stats["embedding_index"].to_numpy()
    norms = np.linalg.norm(weight[rows], axis=1)
    x = stats["log_prior_pa"].to_numpy(dtype=float)
    s, s_lo, s_hi = bootstrap_ci((x, norms), slope)
    r, r_lo, r_hi = bootstrap_ci((x, norms), pearson)
    spread = {k: float(np.std(norms[stats["stratum"].to_numpy() == k], ddof=1))
              for k in STRATA}
    return {"slope": s, "slope_ci95": [s_lo, s_hi], "r": r, "r_ci95": [r_lo, r_hi],
            "spread": spread, "n": int(len(norms))}


def reference_bar(rows, stage, config, incumbent=("rebuild", "baseline")):
    """Arm vs incumbent on `reference`; pass if the arm is not worse by more than
    MARGIN_SES standard errors (one-sided). Advisory below two seeds."""
    floor_sd, _, _ = noise_floor(rows)
    arm = references(rows, stage, config)
    inc = references(rows, *incumbent)
    if not arm:
        return {"n_arm": 0, "passes": None, "note": "no completed runs for the arm"}
    se = floor_sd * math.sqrt(1 / len(arm) + 1 / len(inc))
    margin = float(np.mean(inc) - np.mean(arm))  # >0 arm is better
    return {"n_arm": len(arm), "n_incumbent": len(inc), "arm_mean": float(np.mean(arm)),
            "incumbent_mean": float(np.mean(inc)), "margin": margin, "se": se,
            "margin_in_ses": margin / se, "passes": bool(margin > -MARGIN_SES * se),
            "advisory": len(arm) < 2}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="embedding_sgd")
    p.add_argument("--config", required=True, help="e.g. sgd_lr1e-1")
    p.add_argument("--checkpoint-dir", default="results/checkpoints")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--stats-csv", default=HITTER_STATS)
    p.add_argument("--ledger", default="results/model_v1/sweep_log.csv")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    arm = f"{args.stage}_{args.config}"
    stats = pd.read_csv(args.stats_csv)[["embedding_index", "log_prior_pa", "stratum"]]
    per_seed = {s: norm_exposure(w, stats)
                for s, w in load_seed_embeddings(args.checkpoint_dir, arm, args.seeds).items()}
    bar = reference_bar(ledger_rows(args.ledger), args.stage, args.config)
    seeds = list(per_seed.values())
    out = {
        "arm": arm, "seeds": args.seeds, "per_seed": per_seed,
        "slope_mean": float(np.mean([d["slope"] for d in seeds])),
        "slope_passes": bool(all(d["slope"] >= 0 for d in seeds)),
        "spread_passes": bool(all(d["spread"]["low"] <= d["spread"]["high"] for d in seeds)),
        "reference_bar": bar,
    }
    out["passes"] = bool(out["slope_passes"] and out["spread_passes"] and bar["passes"])
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(args.out_dir) / f"embedding_norm_check_{arm}.json"
    path.write_text(json.dumps(out, indent=2))
    for s, d in per_seed.items():
        print(f"seed {s}: slope {d['slope']:+.4f} [{d['slope_ci95'][0]:+.4f}, {d['slope_ci95'][1]:+.4f}]"
              f"  r {d['r']:+.3f}  spread low/med/high "
              f"{d['spread']['low']:.3f}/{d['spread']['medium']:.3f}/{d['spread']['high']:.3f}")
    print(f"reference bar: {bar}")
    print(f"PASS" if out["passes"] else "FAIL", "->", path)


if __name__ == "__main__":
    main()
