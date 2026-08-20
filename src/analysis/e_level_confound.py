"""
E.12 — is the D.10 ablation table ranking arms by level bias rather than by skill?
(docs/phase-e-spec.md §12.2)

Phase O is specified to select within the approach using the D.10 ablation table. That table's
decisive column is PA-weighted RMSE. RMSE decomposes as bias^2 + variance, and every arm carries
a large shared level bias (+0.0139 by E.3). If arms differ in their MEAN PREDICTED wOBA by an
amount comparable to how much they differ in ranking skill, the column that looks like a skill
ranking is substantially a bias ranking, and Phase O would select on the wrong quantity.

This is NOT a proposal to debias. The +0.01771 level offset is never subtracted (spec §10), and
D.5-style knobs validate on composition fidelity, never on claim-1. This module reads existing
prediction tables and the shipped arms verdict. It trains nothing and edits no scorer.

FORK-OPENED, not pre-registered: the numbers underlying this test were read before it was
proposed (spec §12.0).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PHASE_D = Path("results/phase_d")
OUT_DIR = Path("results/phase_e")
# `invfreq` is degenerate and is reported but excluded from the correlations: its level moved
# -0.050 against a between-arm spread of 0.008, and its RMSE blew out from 0.0479 to 0.0607.
# Leaving it in would manufacture the correlation this module is testing for. Excluding it is
# stated here rather than done silently.
DEGENERATE = ("invfreq",)
SEEDS = tuple(f"baseline_s{i}" for i in range(5))


def arm_mean(arm):
    """Mean predicted wOBA over an arm's prediction table. Unweighted: the question is where the
    arm's output distribution sits, not what the PA-weighted population average is."""
    frame = pd.read_csv(PHASE_D / f"d5_predictions_d10_{arm}.csv")
    assert "pred_woba" in frame, f"{arm} has no pred_woba"
    return float(frame["pred_woba"].mean()), int(len(frame))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    verdict = json.loads((PHASE_D / "d5_arms_verdict_d10.json").read_text())
    rows = []
    for arm, scores in verdict["arms"].items():
        mean, n = arm_mean(arm)
        rows.append({"arm": arm, "mean_pred_woba": mean, "n_rows": n,
                     "rmse_decisive": scores["rmse_decisive"],
                     "rank_decisive": scores["rank_decisive"],
                     "rmse_low": scores["rmse_low"], "rank_low": scores["rank_low"],
                     "degenerate": arm in DEGENERATE})
    # the baseline arm is scored in the verdict under its own name; add it if the verdict omits it
    if "baseline" not in verdict["arms"]:
        mean, n = arm_mean("baseline")
        rows.append({"arm": "baseline", "mean_pred_woba": mean, "n_rows": n,
                     "rmse_decisive": np.nan, "rank_decisive": np.nan,
                     "rmse_low": np.nan, "rank_low": np.nan, "degenerate": False})
    table = pd.DataFrame(rows).sort_values("rmse_decisive").reset_index(drop=True)

    fit = table[~table["degenerate"] & table["rmse_decisive"].notna()]
    assert len(fit) >= 5, f"only {len(fit)} non-degenerate scored arms; correlation is not worth reading"

    out = {"n_arms_total": int(len(table)), "n_arms_in_correlation": int(len(fit)),
           "excluded_as_degenerate": list(DEGENERATE),
           "decisive_stratum": verdict["decisive_stratum"]}
    for metric, direction in (("rmse_decisive", "lower is better"),
                              ("rank_decisive", "higher is better"),
                              ("rmse_low", "lower is better"),
                              ("rank_low", "higher is better")):
        x = fit["mean_pred_woba"].to_numpy()
        y = fit[metric].to_numpy()
        pear = stats.pearsonr(x, y)
        spear = stats.spearmanr(x, y)
        out[metric] = {"direction": direction,
                       "pearson_r": float(pear[0]), "pearson_p": float(pear[1]),
                       "spearman_rho": float(spear[0]), "spearman_p": float(spear[1])}

    # SCALE CHECK. A between-arm correlation means nothing unless the between-arm spread of the
    # level is large relative to the spread the same architecture produces from seed noise alone.
    # If seeds move the level as much as arms do, the arms are not distinguishable on level and
    # the correlation is reading noise.
    seed_means = np.array([arm_mean(seed)[0] for seed in SEEDS])
    out["scale"] = {
        "arm_level_spread_sd": float(fit["mean_pred_woba"].std(ddof=1)),
        "arm_level_range": float(fit["mean_pred_woba"].max() - fit["mean_pred_woba"].min()),
        "seed_level_spread_sd": float(seed_means.std(ddof=1)),
        "seed_level_range": float(seed_means.max() - seed_means.min()),
        "seed_means": [float(v) for v in seed_means],
        "arm_over_seed_sd_ratio": float(fit["mean_pred_woba"].std(ddof=1) / seed_means.std(ddof=1)),
    }
    # the ranking-skill spread, for the same reason: if the arms are near-identical rankers, an
    # RMSE column that separates them cleanly is separating them on something other than ranking
    out["rank_spread"] = {"rank_decisive_min": float(fit["rank_decisive"].min()),
                          "rank_decisive_max": float(fit["rank_decisive"].max()),
                          "rank_decisive_range": float(fit["rank_decisive"].max()
                                                       - fit["rank_decisive"].min())}

    table.to_csv(OUT_DIR / "e12_level_confound.csv", index=False)
    (OUT_DIR / "e12_level_confound.json").write_text(json.dumps(out, indent=2))
    print(table.to_string(index=False, float_format="%.5f"))
    print()
    print(json.dumps(out, indent=2))
    print(f"\nwrote {OUT_DIR / 'e12_level_confound.json'}")


if __name__ == "__main__":
    main()
