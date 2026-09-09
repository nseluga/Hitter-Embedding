"""Paired pred_woba delta at the 2024 cold-start hitters: contaminated vs clean prior.

Same pitcher panel (seed 0) and same batter set on both sides, so the difference carries
no panel noise. Writes to a temp dir; touches no protected results path.
"""
import json, sys, numpy as np, pandas as pd
from pathlib import Path
from src.model import loader, query, query_tables as qt
from src.analysis import claim1_eval

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
batters = [int(b) for b in Path(sys.argv[2]).read_text().split()]
claim1_eval.assert_not_test_season(2024, final_run=False)

tensors, manifest = loader.load_tensors("data/processed/phase_d5")
frame = qt.align_pitch_frame("data/processed/pitch_events_labeled.parquet",
                             "data/processed/eval_targets_pa.parquet", tensors["season"])
pa_df = pd.read_parquet("data/processed/eval_targets_pa.parquet")
tables = query.build_tables(frame, tensors, manifest, pa_df)
paths = [Path("results/checkpoints") / f"embedding_sgd_sgd_lr1_s{s}.pt" for s in range(5)]
models = query.load_ensemble(paths, manifest, tensors["context"].shape[1])

out = {}
for name, excl in (("contaminated", False), ("clean", True)):
    prior = query.default_cold_start_prior(models, exclude_pitcher_batters=excl)
    preds, _ = query.predict(models, tensors, manifest, frame, tables, pa_df, 2024,
                             batters=batters, cold_start_prior=prior)
    preds.to_csv(OUT / f"pred_{name}.csv", index=False)
    out[name] = preds
    print(f"{name}: {len(preds)} rows", flush=True)

key = ["batter", "p_throws"]
m = out["contaminated"].merge(out["clean"], on=key, suffixes=("_dirty", "_clean"))
d = (m["pred_woba_clean"] - m["pred_woba_dirty"]).to_numpy()
rng = np.random.default_rng(0)
boot = np.array([rng.choice(d, size=len(d), replace=True).mean() for _ in range(10000)])
res = {"n_rows": int(len(d)), "n_batters": int(m["batter"].nunique()),
       "eval_season": 2024, "arm": "embedding_sgd_sgd_lr1", "seeds": list(range(5)),
       "delta_definition": "pred_woba(clean prior) - pred_woba(contaminated prior)",
       "mean": float(d.mean()), "sd": float(d.std(ddof=1)),
       "median": float(np.median(d)), "min": float(d.min()), "max": float(d.max()),
       "abs_mean": float(np.abs(d).mean()), "abs_p90": float(np.percentile(np.abs(d), 90)),
       "mean_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
       "n_boot": 10000, "boot_seed": 0}
(OUT / "prior_delta_2024.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
