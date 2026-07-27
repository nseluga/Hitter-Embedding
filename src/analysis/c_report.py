"""
Phase C report — every Phase C number in one committed, seeded command.

Follows the b1_report.py pattern and exists for the same reason: until now Phase C
figures lived in transcripts, which is not a reproducible record (§6 — numbers are
seeded, config-driven, with the config committed). Writes results/phase_c/.

What it produces:
  1. the fitted C.2 prior per batter type, with a bootstrap CI on rho
  2. the implied split-regression constant against The Book's 2200 / 1000
  3. claim-1 scores for every Phase C baseline on the same eval frame
  4. the paired C.2-vs-C.1 bootstrap, which is the actual head-to-head claim
  5. two honesty diagnostics: exposure-talent correlation, and a decode-one-hitter
     table for eyeballing against the source
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import c1_trailing as c1
from src.analysis import c2_bivariate_eb as c2
from src.analysis import claim1_eval as evaluation

DEFAULT_EVAL_SEASON = 2024
OUT_DIR = Path("results/phase_c")

# The Book p. 157 (via Tango, insidethebook.com 2009), converted to the rho it is
# equivalent to via n*_d = n*_side / (2(1 - rho)) with n*_side = 226 (B.1, hitters
# only). Approximate — assumes tau_L = tau_R. A REFERENCE ROW, never fitted.
BOOK_IMPLIED_RHO = {"R": 0.949, "L": 0.887, "S": 0.949}

# no-information reference: predict the side-specific league average for everyone.
# Its deconvolved error is the full between-hitter talent spread, so skill scores
# against it read as the share of talent variance a model captures.
NO_INFO_BUCKETS = ((0, 0.0),)


def load_targets(path):
    return pd.read_parquet(path)


def fit_and_describe(pa_df, eval_season, seed):
    """Fit the C.2 prior and return (params, parameter table, rho CIs)."""
    params = c2.fit(pa_df, eval_season)
    rho_ci = c2.bootstrap_rho(pa_df, eval_season, seed=seed)

    rows = []
    for batter_type, record in params.items():
        n_star_split, tau2_split = c2.implied_split_constant(record)
        point, low, high, at_bound = rho_ci.get(batter_type, (record["rho"], np.nan, np.nan, np.nan))
        for side_index, hand in enumerate(c2.SIDES):
            tau2 = record["tau2"][side_index]
            rows.append({
                "batter_type": batter_type,
                "vs_hand": hand,
                "n_hitters": record["n_hitters"],
                "mu": record["mu"][side_index],
                "tau": np.sqrt(tau2),
                "tau2": tau2,
                "sigma2_within_pa": record["sigma2"][side_index],
                "n_star_rate": record["sigma2"][side_index] / tau2 if tau2 > 0 else np.inf,
                "rho": point,
                "rho_ci_low": low,
                "rho_ci_high": high,
                "rho_at_bound": record["rho_at_bound"],
                "rho_bootstrap_share_at_bound": at_bound,
                "n_rho_hitters": record["n_rho_hitters"],
                "rho_unclipped": record["rho_unclipped"],
                "tau2_split_derived": tau2_split,
                "n_star_split_implied": n_star_split,
                "book_split_constant": c2.BOOK_SPLIT_CONSTANT.get(batter_type, np.nan),
                "book_implied_rho": BOOK_IMPLIED_RHO.get(batter_type, np.nan),
            })
    return params, pd.DataFrame(rows), rho_ci


def build_predictions(pa_df, eval_season, params):
    """Every Phase C baseline, plus the two reference rows, on one eval frame."""
    return {
        "c2_bivariate": c2.predict(pa_df, eval_season, params=params),
        "c2_book_rho_reference": c2.predict(pa_df, eval_season, params=params,
                                            rho_override=BOOK_IMPLIED_RHO),
        "c1_raw": c1.predict(pa_df, eval_season, variant="raw"),
        "c1_bucketed": c1.predict(pa_df, eval_season, variant="bucketed"),
        "no_info_league_average": c1.predict(pa_df, eval_season, variant="bucketed",
                                             buckets=NO_INFO_BUCKETS),
    }


def score_all(pa_df, predictions, eval_season):
    """Claim-1 metrics for every model, plus skill scores against the no-info reference."""
    frames, scores = {}, []
    for name, prediction in predictions.items():
        frame, coverage = evaluation.build_eval_frame(pa_df, prediction, eval_season)
        frames[name] = frame
        table = evaluation.score(frame)
        table.insert(0, "model", name)
        scores.append(table)
    scored = pd.concat(scores, ignore_index=True)

    reference = scored[scored["model"] == "no_info_league_average"].set_index("stratum")["model_rmse"]
    scored["skill_vs_no_info"] = [
        evaluation.skill_score(row["model_rmse"], reference[row["stratum"]])
        if reference[row["stratum"]] > 0 else np.nan
        for _, row in scored.iterrows()
    ]
    return frames, scored, coverage


def exposure_talent_correlation(frame):
    """
    Brown (2008) remark 3b: empirical Bayes assumes exchangeability, but better hitters
    get more PA, and in the platoon case managers ASSIGN weak-side exposure (§5.5
    deployment bias). Correlation between prior exposure and observed talent quantifies
    how far the assumption is from holding. Reported, not corrected for.
    """
    rows = []
    for hand in c2.SIDES:
        part = frame[frame["p_throws"] == hand]
        rows.append({
            "vs_hand": hand,
            "n_groups": len(part),
            "corr_prior_pa_vs_observed_woba": float(part["prior_pa"].corr(part["woba"], method="spearman")),
        })
    return pd.DataFrame(rows)


def decode_sample(frames, pa_df, eval_season, n_each=2, seed=0):
    """
    Decode-one-batch analog: real hitters at each exposure level with their inputs and
    every model's prediction side by side, for eyeballing against the source. Catches
    join and alignment bugs where every column is individually well formed but
    describes the wrong hitter.
    """
    base = frames["c2_bivariate"][evaluation.KEY + ["prior_pa", "stratum", "pa", "woba"]].copy()
    for name, frame in frames.items():
        base = base.merge(frame[evaluation.KEY + ["pred_woba"]].rename(columns={"pred_woba": name}),
                          on=evaluation.KEY, how="left")

    window = c2._fitting_window(pa_df, eval_season, c2.TRAILING_SEASONS)
    pairs = c2.to_pairs(c2.side_observations(window))
    base = base.merge(pairs, on="batter", how="left")

    rng = np.random.default_rng(seed)
    picks = [base[base["prior_pa"] == 0]]
    for stratum in evaluation.STRATUM_NAMES:
        part = base[base["stratum"] == stratum]
        if len(part):
            picks.append(part.iloc[rng.choice(len(part), min(n_each, len(part)), replace=False)])
    return pd.concat(picks).head(8)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run the Phase C baseline report.")
    parser.add_argument("--eval-targets", default="data/processed/eval_targets_pa.parquet")
    parser.add_argument("--eval-season", type=int, default=DEFAULT_EVAL_SEASON)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # never score against the frozen test season outside a final run
    evaluation.assert_not_test_season(args.eval_season)
    pa_df = load_targets(args.eval_targets)

    params, parameter_table, rho_ci = fit_and_describe(pa_df, args.eval_season, args.seed)
    print("fitted C.2 prior")
    print(parameter_table.drop(columns=["tau2", "rho_unclipped"]).to_string(index=False, float_format="%.5f"))

    print("\nrho vs The Book's implied value")
    for batter_type, (point, low, high, at_bound) in rho_ci.items():
        book = BOOK_IMPLIED_RHO.get(batter_type, np.nan)
        verdict = "NOT IDENTIFIED (at bound)" if params[batter_type]["rho_at_bound"] else (
            "consistent with The Book" if low <= book <= high else "DIFFERS from The Book")
        print(f"  type {batter_type}: rho {point:.3f} 95% CI [{low:.3f}, {high:.3f}] "
              f"vs Book-implied {book:.3f} -> {verdict}  [{at_bound:.0%} of resamples at bound]")

    predictions = build_predictions(pa_df, args.eval_season, params)
    frames, scored, coverage = score_all(pa_df, predictions, args.eval_season)
    print("\nclaim-1 scores")
    print(scored.to_string(index=False, float_format="%.4f"))

    print(f"\ncoverage: {json.dumps(coverage)}")

    paired = evaluation.paired_rmse_difference(frames["c2_bivariate"], frames["c1_bucketed"],
                                               seed=args.seed)
    print("\npaired bootstrap: C.2 bivariate minus C.1 bucketed (negative favours C.2)")
    print(paired.to_string(index=False, float_format="%.5f"))

    paired_book = evaluation.paired_rmse_difference(frames["c2_bivariate"],
                                                    frames["c2_book_rho_reference"], seed=args.seed)
    print("\npaired bootstrap: estimated rho minus The Book's implied rho")
    print(paired_book.to_string(index=False, float_format="%.5f"))

    exposure = exposure_talent_correlation(frames["c2_bivariate"])
    print("\nexposure-talent correlation (EB exchangeability check, reported not corrected)")
    print(exposure.to_string(index=False, float_format="%.4f"))

    decoded = decode_sample(frames, pa_df, args.eval_season, seed=args.seed)
    print("\ndecode check: real hitters, inputs and every model's prediction")
    print(decoded.to_string(index=False, float_format="%.4f"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parameter_table.to_csv(out_dir / "c2_prior_parameters.csv", index=False)
    scored.to_csv(out_dir / "c_claim1_scores.csv", index=False)
    paired.to_csv(out_dir / "c2_vs_c1_paired.csv", index=False)
    paired_book.to_csv(out_dir / "c2_vs_book_rho_paired.csv", index=False)
    exposure.to_csv(out_dir / "c2_exposure_talent_correlation.csv", index=False)
    decoded.to_csv(out_dir / "c_decode_sample.csv", index=False)
    (out_dir / "c_coverage.json").write_text(json.dumps(coverage, indent=2))
    print(f"\nwrote 6 files to {out_dir}")


if __name__ == "__main__":
    main()
