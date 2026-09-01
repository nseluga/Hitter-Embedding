"""
Step 5 of the D.5 remediation: claim-1 for every ablation arm (D5-R16).

Frozen rule #2 says architecture decisions are made on claim-1. Zero of the eight `splithead` arms had
a claim-1 number; every verdict came from held-out per-pitch log loss, which `model_v1_ablation_report.py`'s own
docstring says "says which model predicts PITCHES better and nothing at all about whether it
projects HITTERS better than empirical Bayes". This module closes that.

Three things it deliberately does NOT do.

It does not compare `reference` columns. That column is log loss over the quality bins, and the
D5-R17 rebuild moved the bins, so a `splithead` and a `rebuild` reference are in different units -- and
`nospray` factorizes the output space differently again, so its 0.814994 is incomparable to
baseline's by construction rather than by accident. claim-1 has no such problem: it is
PA-weighted RMSE on wOBA, the same quantity whatever the model factorizes into.

It does not treat the seed spread as a test. The retired "x noise floor" was max-min over five
seeds, which GROWS with seed count and is not a standard error; redone as a paired bootstrap,
`block` is t ~ 9.2 where it was reported marginal and `bilinear` is t ~ 1.6 turning on one
outlier seed where it was reported a clean null. The spread is reported as context, next to the
interval, never instead of it.

It does not run five seeds per arm. That is forty extra composition runs at ~2.3 h each. Only
baseline gets a per-seed set, so every other arm's interval rests on the ASSUMPTION that its
seed noise resembles baseline's -- stated here rather than hidden, because it is the one
assumption in the step that could be wrong.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import claim1_eval as ce

# Pre-registered before any arm is scored: the AGGREGATE stratum decides arm selection. One
# number is needed to pick an architecture and the aggregate is the least noisy of the four. An
# arm that wins `low` while losing `all` is recorded as an explicit finding, not quietly
# promoted -- the low stratum is what the THESIS is judged on, which is a different question
# from which architecture to carry forward.
DECISIVE_STRATUM = "all"
BASELINE = "baseline"


def load_arm_frames(out_dir, stage, arms, pa_df, eval_season):
    """
    One eval frame per arm, from that arm's own predictions CSV.

    Arms with no CSV are skipped rather than defaulted, and the caller reports them: an arm
    silently absent from a ranking table reads as an arm that lost.
    """
    frames, missing = {}, []
    for arm in arms:
        path = Path(out_dir) / f"model_v1_predictions_{stage}_{arm}.csv"
        if not path.exists():
            missing.append(arm)
            continue
        frame, _ = ce.build_eval_frame(pa_df, pd.read_csv(path), eval_season)
        frames[arm] = frame
    return frames, missing


def seed_spread(out_dir, stage, arm, pa_df, eval_season, seeds=(0, 1, 2, 3, 4)):
    """
    Per-seed claim-1 for one arm, reported as CONTEXT beside the paired intervals.

    Two numbers, and the pair is the point: `sd` is what a standard error would be built from,
    `range` is the retired statistic. Printing both makes it visible that the retired one is
    roughly twice the other on five seeds and would keep growing on ten.
    """
    rows = []
    for seed in seeds:
        path = Path(out_dir) / f"model_v1_predictions_{stage}_{arm}_s{seed}.csv"
        if not path.exists():
            continue
        frame, _ = ce.build_eval_frame(pa_df, pd.read_csv(path), eval_season)
        scores = ce.score(frame).set_index("stratum")
        rows.append({"seed": seed,
                     **{name: scores.loc[name, "pa_weighted_rmse"]
                        for name in list(ce.STRATUM_NAMES) + ["all"]}})
    if not rows:
        return None, None
    per_seed = pd.DataFrame(rows)
    summary = pd.DataFrame([
        {"stratum": name, "n_seeds": len(per_seed),
         "mean": per_seed[name].mean(), "sd": per_seed[name].std(ddof=1),
         "range": per_seed[name].max() - per_seed[name].min()}
        for name in list(ce.STRATUM_NAMES) + ["all"]])
    return per_seed, summary


def compare_arms(frames, baseline=BASELINE, n_boot=2000, seed=0):
    """
    Every arm against baseline, paired and clustered on batter, per stratum.

    Sign conventions are inherited from `claim1_eval` and are opposite to each other on purpose,
    so they are restated here rather than left to the reader: NEGATIVE `rmse_difference` favours
    the ARM (lower error), POSITIVE `rank_difference` favours the ARM (better ordering). The arm
    is always side A.

    The interval is the verdict. `beats_baseline` requires it to exclude zero in the decisive
    stratum, which is what "pre-registered" means here -- an arm cannot qualify by winning
    whichever stratum happens to resolve.
    """
    assert baseline in frames, f"no {baseline} frame, so there is nothing to compare against"
    rows = []
    for arm, frame in frames.items():
        if arm == baseline:
            continue
        rmse = ce.paired_rmse_difference(frame, frames[baseline], n_boot=n_boot, seed=seed)
        rank = ce.paired_rank_difference(frame, frames[baseline], n_boot=n_boot, seed=seed)
        merged = rmse.merge(rank, on=["stratum", "n_hitters", "n_batters"],
                            suffixes=("_rmse", "_rank"))
        rows.append(merged.assign(arm=arm))
    table = pd.concat(rows, ignore_index=True)
    return table[["arm"] + [column for column in table.columns if column != "arm"]]


def rank_arms(frames, comparisons, stratum=DECISIVE_STRATUM):
    """
    The one table an architecture decision can be read off, with the split verdicts flagged.

    `wins_low_loses_decisive` is the finding the plan asks to be recorded rather than discarded:
    an arm better for low-exposure hitters and worse in aggregate is a real result about who the
    architecture helps, and it would vanish if the table only carried the decisive row.
    """
    rows = []
    for arm, frame in frames.items():
        scores = ce.score(frame).set_index("stratum")
        record = {"arm": arm,
                  "rmse_decisive": scores.loc[stratum, "pa_weighted_rmse"],
                  "rmse_low": scores.loc["low", "pa_weighted_rmse"],
                  "rank_decisive": scores.loc[stratum, "rank_corr_weighted"],
                  "rank_low": scores.loc["low", "rank_corr_weighted"]}
        part = comparisons[(comparisons["arm"] == arm) & (comparisons["stratum"] == stratum)]
        low = comparisons[(comparisons["arm"] == arm) & (comparisons["stratum"] == "low")]
        if len(part):
            row = part.iloc[0]
            resolved = row["ci_high_rmse"] < 0 or row["ci_low_rmse"] > 0
            record.update({
                "rmse_difference": row["rmse_difference"],
                "ci_low": row["ci_low_rmse"], "ci_high": row["ci_high_rmse"],
                "resolved": bool(resolved),
                "beats_baseline": bool(resolved and row["rmse_difference"] < 0),
                "wins_low_loses_decisive": bool(len(low)
                                                and low.iloc[0]["rmse_difference"] < 0
                                                and row["rmse_difference"] > 0)})
        rows.append(record)
    table = pd.DataFrame(rows).sort_values("rmse_decisive").reset_index(drop=True)
    return table


def main():
    parser = argparse.ArgumentParser(description="D.5 step 5: claim-1 for every ablation arm.")
    parser.add_argument("--stage", default="rebuild")
    parser.add_argument("--arms", nargs="+",
                        default=["baseline", "dim16", "dim64", "bilinear", "meanweight",
                                 "invfreq", "nospray", "block"])
    parser.add_argument("--eval-targets", default="data/processed/eval_targets_pa.parquet")
    parser.add_argument("--eval-season", type=int, default=2024)
    parser.add_argument("--final-run", action="store_true")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--out-dir", default="results/model_v1")
    args = parser.parse_args()

    ce.assert_not_test_season(args.eval_season, final_run=args.final_run)
    pa_df = pd.read_parquet(args.eval_targets)
    out_dir = Path(args.out_dir)

    frames, missing = load_arm_frames(out_dir, args.stage, args.arms, pa_df, args.eval_season)
    if missing:
        print(f"NOT SCORED (no predictions CSV): {', '.join(missing)}")
    assert BASELINE in frames, "baseline has no predictions, so nothing can be compared"

    comparisons = compare_arms(frames, n_boot=args.n_boot)
    comparisons.to_csv(out_dir / f"model_v1_arms_paired_{args.stage}.csv", index=False)

    table = rank_arms(frames, comparisons)
    table.to_csv(out_dir / f"model_v1_arms_{args.stage}.csv", index=False)
    print(f"\nclaim-1 by arm, decisive stratum '{DECISIVE_STRATUM}' (pre-registered). "
          f"NEGATIVE rmse_difference favours the arm.")
    print(table.to_string(index=False, float_format="%.5f"))

    per_seed, spread = seed_spread(out_dir, args.stage, BASELINE, pa_df, args.eval_season)
    if spread is not None:
        per_seed.to_csv(out_dir / f"model_v1_arms_baseline_per_seed_{args.stage}.csv", index=False)
        print(f"\nbaseline seed spread -- CONTEXT, never the test. Every other arm's interval "
              f"assumes its seed noise resembles this.")
        print(spread.to_string(index=False, float_format="%.5f"))

    verdict = {
        "stage": args.stage,
        "decisive_stratum": DECISIVE_STRATUM,
        "not_scored": missing,
        "arms": table.set_index("arm").to_dict("index"),
        "baseline_seed_spread": (spread.set_index("stratum").to_dict("index")
                                 if spread is not None else None),
        # the two facts that keep this table from being over-read, carried in the machine-readable
        # file rather than only in the printout
        "seed_noise_assumption": "only baseline has per-seed runs; other arms' intervals assume "
                                 "their seed noise resembles baseline's",
        "reference_column_incomparable": "sweep_log.csv `reference` is log loss over the quality "
                                         "bins, which the D5-R17 rebuild moved and which nospray "
                                         "factorizes differently; it is not comparable across "
                                         "arms or against splithead",
        "any_arm_beats_baseline": bool(table.get("beats_baseline", pd.Series(dtype=bool)).any()),
    }
    (out_dir / f"model_v1_arms_verdict_{args.stage}.json").write_text(json.dumps(verdict, indent=2))
    print(f"\nwrote {out_dir / f'model_v1_arms_verdict_{args.stage}.json'}")
    return 0


def _self_check():
    """
    The two things that would silently produce a wrong architecture verdict.

    A flipped sign would promote the worst arm, and a `beats_baseline` that reads an unresolved
    interval would promote noise. Both are checked against a planted arm that is genuinely
    better and one that is better but not resolvably so.
    """
    rng = np.random.default_rng(0)
    n = 240
    base = pd.DataFrame({
        "batter": np.arange(n), "season": 2024, "p_throws": ["L", "R"] * (n // 2),
        "woba": rng.normal(0.32, 0.06, n), "denominator": rng.uniform(50, 500, n),
        "noise_var": 0.0004,
        "stratum": pd.Categorical(["low", "high"] * (n // 2), categories=list(ce.STRATUM_NAMES))})
    base["pa"] = base["denominator"]
    baseline = base.assign(pred_woba=base["woba"] + rng.normal(0, 0.05, n))
    better = base.assign(pred_woba=base["woba"] + rng.normal(0, 0.005, n))
    tied = baseline.assign(pred_woba=baseline["pred_woba"] + rng.normal(0, 1e-6, n))

    comparisons = compare_arms({"baseline": baseline, "better": better, "tied": tied}, n_boot=200)
    table = rank_arms({"baseline": baseline, "better": better, "tied": tied},
                      comparisons).set_index("arm")
    assert table.loc["better", "rmse_difference"] < 0, "sign flipped: the better arm must be negative"
    assert table.loc["better", "beats_baseline"], table.loc["better"]
    assert not table.loc["tied", "beats_baseline"], \
        "an unresolved interval was read as a win"
    assert table.index[0] == "better", "the ranking is not sorted by the decisive stratum"
    print("self-check passed")


if __name__ == "__main__":
    raise SystemExit(main())
