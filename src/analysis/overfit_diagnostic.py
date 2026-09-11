"""
Diagnostic checks for whether the level-query in-sample-vs-held-out gap
(docs/decision-log.md, 2026-09-08 "level-query correlation is inflated in-sample")
reflects overfitting -- the embedding table memorising sampling noise in
training-period process rates -- rather than 2024 simply being a noisier
target. Read-only over existing results, checkpoints, and
data/processed/{eval_targets_pa,pitch_events_labeled}.parquet. No retraining,
no code path shared with training, no claim-1 gate, no promotion decision.

Corrected framing (2026-09-10): outcome luck (where a ball lands) is gone by
training on process rates. What remains, and what a 32-free-number row with a
nonlinear trunk CAN memorise, is sampling noise in those process rates
themselves -- a hitter's observed swing rate over 2,000 pitches carries ~1pp
sampling error; a two-strike contact rate over 150 pitches carries several
points; a 300-ball-in-play EV90 estimate moves ~1mph season to season with no
change in the hitter. The thinner the slice, the more of what gets fit is
noise, not signal.

Test 1 (process-rate leak): for each head with both a model marginal
(results/model_visualization/head_marginals.csv) and an observed training-
period rate (swing <-> swing_rate, contact <-> contact_rate), partial-
correlate the model's marginal against the process-rate residual (observed
rate minus a shrunk rate, i.e. the part attributable to sampling noise),
controlling for the shrunk rate itself. Near zero means the model tracks the
shrunk (denoised) rate; positive means it is carrying sampling noise as if it
were signal. This replaces the original wOBA-residual-only version of test 1.

Test 2 (shrinkage slope): OLS slope of the model's pooled level query on EB's
pooled shrunk wOBA estimate, per stratum. Slope 1 means the same amount of
shrinkage as EB; above 1 means the model spreads hitters wider than the
evidence supports.

Test 3 (ceiling-normalised gap): divide the in-sample and held-out r_partial
for the level query by their own reliability ceiling (sqrt(n / (n + N_STAR)),
Spearman/Pearson attenuation). If the normalised gap closes, the raw gap was
target noise; if it persists, it is overfitting.

Test 4 (contrast probes, held-out labels): axis_screen's contrast candidates
(results/embedding_structure/axis_screen.py) are scored on season<=2024
labels, which mixes 2015-2023 training pitches into the metric the frozen
embedding is graded against -- contrasts are exactly the thin, few-hundred-
pitch slices most exposed to process-rate noise. Rerun build_contrasts with
season==2024 only (held out, same embedding, no retrain) and compare each
contrast's lda_cv_mean to its in-sample value in the committed axis_screen.json.
This is the level-query correction applied one tier down, onto the contrast
tier.

Run: .venv/bin/python -m src.analysis.overfit_diagnostic --out-dir results/overfit_diagnostic
"""

import argparse
import contextlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import baseline_ladder_bivariate_eb as eb
from src.analysis import claim1_eval
from src.analysis.level_query_heldout import (
    EVAL_TARGETS_PARQUET,
    HELDOUT_SEASON,
    bootstrap_ci_partial,
    paired_pearson,
)

HITTER_STATS_CSV = "results/model_visualization/hitter_stats.csv"
HEAD_MARGINALS_CSV = "results/model_visualization/head_marginals.csv"
HELDOUT_JSON = "results/paper_figures/level_query_heldout.json"
HELDOUT_CSV = "results/paper_figures/level_query_heldout.csv"
PITCH_EVENTS_PARQUET = "data/processed/pitch_events_labeled.parquet"
AXIS_SCREEN_JSON = "results/embedding_structure/axis_screen.json"
# both-sides stabilization point, decision-log 2026-09-08 ("low < 113 PA = n*/2,
# n*=226"); used here only to normalise a ceiling, not as a claim-1 constant.
N_STAR = 226
# head marginal <-> observed process-rate column, and which pitch subset the
# rate's denominator is drawn from ("all" pitches or "swings" only).
PROCESS_RATE_HEADS = {
    "swing": {"rate_col": "swing_rate", "numerator": "swing", "denominator": "all"},
    "contact": {"rate_col": "contact_rate", "numerator": "contact", "denominator": "swings"},
}


def _cell(a, b, confound):
    r_raw, _ = paired_pearson(a, b)
    r_partial, lo, hi, n = bootstrap_ci_partial(a, b, confound)
    return {"n": n, "r_raw": r_raw, "r_partial": r_partial, "ci_low": lo, "ci_high": hi}


# --------------------------------------------------------------------- Test 1

def eb_shrink_binomial(x, n, min_k=5.0):
    """
    Method-of-moments empirical-Bayes shrinkage for a per-hitter binomial rate.
    x, n: per-hitter successes and attempts. Shrinks each hitter's rate toward
    the attempt-weighted population mean by (n / (n + k)), where k is picked so
    the population variance of shrunk rates matches the between-hitter
    variance implied after removing expected sampling variance.

    ponytail: method-of-moments, not a fitted beta-binomial MLE -- k is a
    single pooled pseudo-count, not per-stratum. Upgrade path: fit
    scipy.stats.betabinom per stratum if this pooled k proves too coarse.
    """
    x = np.asarray(x, dtype="float64")
    n = np.asarray(n, dtype="float64")
    mask = n > 0
    x, n = x[mask], n[mask]
    p = x / n
    p_bar = x.sum() / n.sum()
    n_bar = n.mean()
    sample_var = np.average((p - p_bar) ** 2, weights=n)
    expected_sampling_var = np.average(p * (1 - p) / n, weights=n)
    between_var = max(sample_var - expected_sampling_var, 1e-9)
    k = max(p_bar * (1 - p_bar) / between_var - p_bar * (1 - p_bar) / n_bar, min_k)
    shrunk = (n * p + k * p_bar) / (n + k)
    return p, shrunk, k


def observed_process_rates(pitch_events_path, eval_season, n_seasons=eb.TRAILING_SEASONS):
    """
    Per-batter observed swing_rate (swings / all pitches) and contact_rate
    (contact / swings), over the same trailing training window
    baseline_ladder_bivariate_eb fits on (season < eval_season). Returns one
    row per batter with columns batter, swing_x, swing_n, contact_x, contact_n.
    """
    cols = ["batter", "season", "swing", "contact"]
    df = pd.read_parquet(pitch_events_path, columns=cols)
    df = eb.trailing_window(df, eval_season, n_seasons)

    swing_agg = df.groupby("batter")["swing"].agg(swing_x="sum", swing_n="count")
    swings = df[df["swing"] == 1]
    contact_agg = swings.groupby("batter")["contact"].agg(contact_x="sum", contact_n="count")
    return swing_agg.join(contact_agg, how="outer").reset_index()


def test1_process_rate_leak(head_marginals_csv=HEAD_MARGINALS_CSV,
                             hitters_csv=HITTER_STATS_CSV,
                             pitch_events_path=PITCH_EVENTS_PARQUET,
                             eval_season=HELDOUT_SEASON):
    marginals = pd.read_csv(head_marginals_csv)[["batter"] + list(PROCESS_RATE_HEADS)]
    hitters = pd.read_csv(hitters_csv)[["batter", "stratum", "log_prior_pa"]]
    rates = observed_process_rates(pitch_events_path, eval_season)

    merged = marginals.merge(hitters, on="batter", how="inner").merge(rates, on="batter", how="inner")

    out = {"n_hitters": len(merged)}
    for head, spec in PROCESS_RATE_HEADS.items():
        x_col, n_col = f"{head}_x", f"{head}_n"
        valid = merged[n_col].notna() & (merged[n_col] > 0)
        sub = merged[valid]
        p_obs, p_shrunk, k = eb_shrink_binomial(sub[x_col].values, sub[n_col].values)
        residual = p_obs - p_shrunk

        model_marginal = sub[head].values.astype("float64")
        cell_pooled = _cell(model_marginal, residual, p_shrunk)
        cell_pooled["k_pseudo_count"] = float(k)
        per_stratum = {}
        for stratum, idx in sub.groupby("stratum").groups.items():
            pos = sub.index.get_indexer(idx)
            per_stratum[str(stratum)] = _cell(model_marginal[pos], residual[pos], p_shrunk[pos])

        out[head] = {
            "rate_col": spec["rate_col"],
            "denominator": spec["denominator"],
            "n": int(valid.sum()),
            "pooled": cell_pooled,
            "by_stratum": per_stratum,
        }
    return out


# --------------------------------------------------------------------- Test 2

def eb_pooled_estimate(eval_targets_path=EVAL_TARGETS_PARQUET, eval_season=HELDOUT_SEASON):
    """
    Per-batter EB shrunk wOBA estimate pooled across both pitcher hands,
    weighted by each side's training PA. Returns batter/eb_pooled/train_pa.
    """
    claim1_eval.assert_not_test_season(eval_season, final_run=False)
    pa_df = pd.read_parquet(eval_targets_path)
    params = eb.fit(pa_df, eval_season)
    shrunk_long = eb.predict(pa_df, eval_season, params=params)  # batter/p_throws/pred_woba
    shrunk = shrunk_long.pivot(index="batter", columns="p_throws", values="pred_woba").reset_index()
    shrunk = shrunk.rename(columns={"L": "shrunk_L", "R": "shrunk_R"})

    window = eb._fitting_window(pa_df, eval_season, eb.TRAILING_SEASONS)
    pairs = eb.to_pairs(eb.side_observations(window))[["batter", "n_L", "n_R"]]

    merged = shrunk.merge(pairs, on="batter", how="left").fillna({"n_L": 0.0, "n_R": 0.0})
    total = merged["n_L"] + merged["n_R"]
    with np.errstate(invalid="ignore"):
        merged["eb_pooled"] = np.where(
            total > 0,
            (merged["n_L"] * merged["shrunk_L"] + merged["n_R"] * merged["shrunk_R"]) / total,
            np.nan,
        )
    merged["train_pa"] = total
    return merged[["batter", "eb_pooled", "train_pa"]]


def test2_shrinkage_slope(hitters, eb_frame):
    """
    OLS slope of level_query on eb_pooled, per stratum + pooled. Slope 1 means
    the model shrinks by the same amount as EB; above 1 means under-shrinkage.
    """
    merged = hitters.merge(eb_frame, on="batter", how="inner")

    def _slope(group):
        x = group["eb_pooled"].values.astype("float64")
        y = group["level_query"].values.astype("float64")
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        design = np.column_stack([np.ones_like(x), x])
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        return {
            "n": int(mask.sum()),
            "intercept": float(coefficients[0]),
            "slope": float(coefficients[1]),
            "mean_gap_vs_eb": float(np.mean(y - x)),
        }

    out = {"pooled": _slope(merged)}
    for stratum, group in merged.groupby("stratum"):
        out[str(stratum)] = _slope(group)
    return out


# --------------------------------------------------------------------- Test 3

def test3_ceiling_normalized_gap(heldout_stats, hitters_csv=HITTER_STATS_CSV,
                                  heldout_csv=HELDOUT_CSV, n_star=N_STAR):
    """
    Normalise the in-sample (r vs woba_level) and held-out (r vs woba_2024)
    r_partial by their own reliability ceiling sqrt(n / (n + n_star)), using
    median training PA for the in-sample ceiling and median pa_2024 for the
    held-out one, per stratum. If the normalised gap persists where the raw
    gap did, the drop is not explained by target noise alone.
    """
    hitters = pd.read_csv(hitters_csv)
    hitters["train_pa"] = hitters["prior_pa_L"] + hitters["prior_pa_R"]
    heldout = pd.read_csv(heldout_csv)

    out = {}
    for stratum in ("pooled", "low", "medium", "high"):
        merged = heldout if stratum == "pooled" else heldout[heldout["stratum"] == stratum]
        train_pa = (hitters["train_pa"] if stratum == "pooled"
                    else hitters.loc[hitters["stratum"] == stratum, "train_pa"]).median()
        held_pa = merged["pa_2024"].median()

        held_r_partial = heldout_stats.get(stratum, {}).get("r_partial")
        in_sample_cell = _cell(
            merged["level_query"].values.astype("float64"),
            merged["woba_level"].values.astype("float64"),
            merged["log_prior_pa"].values.astype("float64"),
        )
        in_sample_r_partial = in_sample_cell["r_partial"]

        train_ceiling = float(np.sqrt(train_pa / (train_pa + n_star)))
        held_ceiling = float(np.sqrt(held_pa / (held_pa + n_star)))
        cell = {
            "train_pa_median": float(train_pa),
            "held_pa_median": float(held_pa),
            "train_ceiling": train_ceiling,
            "held_ceiling": held_ceiling,
            "in_sample_r_partial": in_sample_r_partial,
            "held_r_partial": held_r_partial,
            "in_sample_normalized": in_sample_r_partial / train_ceiling if train_ceiling else None,
            "held_normalized": held_r_partial / held_ceiling
                                if held_r_partial is not None and held_ceiling else None,
        }
        if held_r_partial is not None:
            cell["raw_gap"] = in_sample_r_partial - held_r_partial
        if cell["in_sample_normalized"] is not None and cell["held_normalized"] is not None:
            cell["normalized_gap"] = cell["in_sample_normalized"] - cell["held_normalized"]
        out[stratum] = cell
    return out


# --------------------------------------------------------------------- Test 4

@contextlib.contextmanager
def _season_eq_2024_pitch_events(axis_screen):
    """
    Monkeypatch axis_screen.load_pitch_events to filter season==2024 instead
    of its module default season<=2024, so build_contrasts recomputes each
    contrast VALUE from held-out pitches only. The frozen embedding is not
    retrained or reloaded -- only the label side changes, same correction the
    level query got one tier up.
    """
    original = axis_screen.load_pitch_events

    def held_out_loader(batters):
        cols = ["batter", "pitch_type", "swing", "contact", "ev", "la", "balls",
                "strikes", "stand", "p_throws", "plate_z", "zone", "release_speed",
                "season"]
        df = pd.read_parquet(axis_screen.PITCH_EVENTS_PATH, columns=cols)
        df = df[df["season"] == HELDOUT_SEASON]
        df = df[df["batter"].isin(batters)]
        group = pd.Series(np.nan, index=df.index, dtype=object)
        for name, codes in axis_screen.PITCH_GROUPS.items():
            group[df["pitch_type"].isin(codes)] = name
        return df.assign(pitch_group=group)

    axis_screen.load_pitch_events = held_out_loader
    try:
        yield
    finally:
        axis_screen.load_pitch_events = original


def test4_contrast_probes_heldout(axis_screen_json=AXIS_SCREEN_JSON):
    """
    Rerun axis_screen's contrast candidates (only, not the marginals -- those
    already have a head-marginal-vs-hitter_stats comparison and aren't the
    thin-slice concern) against 2024-only pitch labels, same frozen embedding.
    Compares each contrast's held-out lda_cv_mean to the committed in-sample
    value.
    """
    from results.embedding_structure import axis_screen as axscreen

    in_sample = json.loads(Path(axis_screen_json).read_text())["candidates"]

    emb, hitters, _ = axscreen.es.load_hitters(
        axscreen.es.DEFAULT_CHECKPOINT_DIR, axscreen.es.DEFAULT_ARM,
        axscreen.es.HITTER_STATS_PATH, axscreen.es.NAMES_PATH)
    normalized, _ = axscreen.es.unit_normalize(emb)

    with _season_eq_2024_pitch_events(axscreen):
        contrasts = axscreen.build_contrasts(hitters)

    out = {}
    for name, series in contrasts.items():
        if name not in in_sample or in_sample[name]["kind"] != "contrast":
            continue
        definition, notes = axscreen.DEFINITIONS[name]
        result = axscreen.score_candidate(name, "contrast", series, hitters,
                                           normalized, hitters, definition, notes)
        entry = result[0]
        if entry is None:
            out[name] = {"skipped": True, "n": result[1], "min_n": axscreen.MIN_N}
            continue
        out[name] = {
            "n": entry["n"],
            "held_out_lda_cv_mean": entry["lda_cv_mean"],
            "in_sample_lda_cv_mean": in_sample[name]["lda_cv_mean"],
            "gap": in_sample[name]["lda_cv_mean"] - entry["lda_cv_mean"],
            "in_sample_n": in_sample[name]["n"],
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="results/overfit_diagnostic")
    parser.add_argument("--eval-targets", default=EVAL_TARGETS_PARQUET)
    parser.add_argument("--pitch-events", default=PITCH_EVENTS_PARQUET)
    parser.add_argument("--eval-season", type=int, default=HELDOUT_SEASON)
    parser.add_argument("--skip-test4", action="store_true",
                         help="test 4 loads a model checkpoint via embedding_structure.load_hitters; "
                              "skip it if only the EB-based tests (1-3) are needed.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hitters = pd.read_csv(HELDOUT_CSV)  # has level_query, woba_level, stratum, batter
    eb_frame = eb_pooled_estimate(args.eval_targets, args.eval_season)

    results = {
        "test1_process_rate_leak": test1_process_rate_leak(
            pitch_events_path=args.pitch_events, eval_season=args.eval_season),
        "test2_shrinkage_slope": test2_shrinkage_slope(hitters, eb_frame),
    }

    heldout_stats = json.loads(Path(HELDOUT_JSON).read_text())
    results["test3_ceiling_normalized_gap"] = test3_ceiling_normalized_gap(heldout_stats)

    if not args.skip_test4:
        results["test4_contrast_probes_heldout"] = test4_contrast_probes_heldout()

    (out_dir / "overfit_diagnostic.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {out_dir}/overfit_diagnostic.json")


if __name__ == "__main__":
    main()
