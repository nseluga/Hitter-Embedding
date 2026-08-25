"""
Step 3 of the D.5 remediation: attribute the +0.01771 level bias, and test its GRADIENT.

The bias is not flat. It rises monotonically with exposure -- +0.01179 low, +0.01785 mid,
+0.01941 high, a 61-65% rise that Phase C's comparator does not show. So there is nothing to
subtract: a single level correction cannot remove a gradient, and subtracting one is forbidden
anyway (decision log 2026-08-08, circular -- the correction would be validated on the metric the
bias is measured with). This module ATTRIBUTES.

Three of the four level contributors are already measurable by running `src.model.query` twice
with a flag flipped, so they need no code here:

    V's fitting subset   --no-unmeasured-split   vs default
    take surface         --take-count-offsets    vs default
    cell size M          --n-pitches 6 / 12 / 24

What needs code is the HBP diagnosis, which has to DISCRIMINATE between two candidate causes,
and the two gradient tests.

Test (b) IS here now, as `gradient_b`, but it reads the post-rebuild checkpoints -- the D5-R17
retrain replaces every embedding row, so a pre-rebuild reading would describe weights that no
longer exist. It runs per seed and never averages embeddings across seeds: two seeds' embedding
spaces are related by an arbitrary rotation, so a coordinate-wise average of them is not an
embedding of anything. Only the scalar summaries are pooled.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import c2_bivariate_eb, claim1_eval
from src.model import query_tables as qt

# Buckets for the HBP diagnosis, in grid cells from the nearest boundary. 0 is the outer ring,
# where `_grid_index` deposits every clipped coordinate.
EDGE_BUCKETS = (0, 1, 3, 6, 12)


def _edge_distance(x_index, z_index, n_x, n_z):
    """Cells from the nearest grid boundary. 0 means the outer ring."""
    return np.minimum(np.minimum(x_index, n_x - 1 - x_index),
                      np.minimum(z_index, n_z - 1 - z_index))


def hbp_diagnosis(frame, train_mask):
    """
    Which mechanism inflates modelled HBP -- coordinate CLIPPING or the SHRINKAGE target?

    Modelled HBP runs 0.00317 against 0.00221 observed per pitch at 0-0, 43% high on the
    highest-weighted non-hit terminal state. Two candidates, and the plan pre-registers a
    different fix for each, so the diagnosis has to separate them rather than confirm one:

    * CLIPPING -- `_grid_index` folds every out-of-range plate coordinate into the outer ring.
      Hit batsmen are far outside the zone, so they concentrate there. Signature: the observed
      HBP rate in the outer ring is far above the interior AND a large share of outer-ring rows
      were clipped rather than genuinely near the boundary. Fix: extend the grid, or drop
      out-of-range takes.

    * SHRINKAGE -- `shrink` pulls every cell toward the POOLED distribution over all cells,
      whose HBP share is the global HBP-among-takes rate. An interior cell with a true HBP
      probability of essentially zero is lifted toward that pool. Signature: `shrunk` exceeds
      `observed` in the interior buckets, where cell counts are lowest relative to alpha.
      Fix: the neighbour smoothing becomes active work rather than latent.

    The two are not exclusive. Returns one record per bucket plus the population totals.
    """
    klass = frame["description"].map(qt.TAKE_DESCRIPTION_TO_CLASS)
    usable = np.asarray(train_mask) & klass.notna().to_numpy()

    raw_x = frame["plate_x"].to_numpy(dtype="float64")
    raw_z = frame["plate_z"].to_numpy(dtype="float64")
    x_index, n_x = qt._grid_index(raw_x, qt.TAKE_GRID_X)
    z_index, n_z = qt._grid_index(raw_z, qt.TAKE_GRID_Z)
    # a coordinate is clipped when it fell OUTSIDE the grid, not merely in the outer ring
    clipped = ((raw_x < qt.TAKE_GRID_X[0]) | (raw_x >= qt.TAKE_GRID_X[1])
               | (raw_z < qt.TAKE_GRID_Z[0]) | (raw_z >= qt.TAKE_GRID_Z[1]))
    distance = _edge_distance(x_index, z_index, n_x, n_z)

    surfaces, alpha = qt.fit_take_surfaces(frame, usable)

    records, totals = [], {"observed": 0.0, "shrunk": 0.0, "n": 0.0}
    for low, high in zip(EDGE_BUCKETS, EDGE_BUCKETS[1:] + (max(n_x, n_z),)):
        band = usable & (distance >= low) & (distance < high)
        if not band.any():
            continue
        n = int(band.sum())
        observed = float((klass.to_numpy()[band] == 2).sum()) / n

        # the shrunk surface's HBP probability, weighted by the SAME rows -- so the gap
        # against `observed` is shrinkage's contribution and nothing else
        shrunk = 0.0
        for (stand, throws), surface in surfaces.items():
            cell = band & (frame["stand"] == stand).to_numpy() \
                & (frame["p_throws"] == throws).to_numpy()
            if cell.any():
                shrunk += float(surface[x_index[cell], z_index[cell], 2].sum())
        shrunk /= n

        records.append({
            "edge_distance_cells": f"{low}-{high - 1}",
            "n_takes": n,
            "clipped_share": float(clipped[band].sum()) / n,
            "observed_hbp": observed,
            "shrunk_hbp": shrunk,
            "shrinkage_inflation": shrunk - observed,
        })
        totals["observed"] += observed * n
        totals["shrunk"] += shrunk * n
        totals["n"] += n

    population = {"n_takes": int(totals["n"]),
                  "observed_hbp": totals["observed"] / totals["n"],
                  "shrunk_hbp": totals["shrunk"] / totals["n"],
                  "shrinkage_inflation": (totals["shrunk"] - totals["observed"]) / totals["n"],
                  "alpha": alpha,
                  "clipped_takes": int(clipped[usable].sum()),
                  "clipped_hbp": float((klass.to_numpy()[usable & clipped] == 2).mean())
                  if (usable & clipped).any() else float("nan"),
                  "unclipped_hbp": float((klass.to_numpy()[usable & ~clipped] == 2).mean())}
    return {"by_edge_distance": records, "population": population}


def _ols(design, y, weights):
    """Weighted least squares coefficients. `design` already carries its intercept column."""
    root = np.sqrt(weights)[:, None]
    solution, *_ = np.linalg.lstsq(design * root, y * np.sqrt(weights), rcond=None)
    return solution


def gradient_a(eval_frame, pa_df, eval_season, n_boot=2000, seed=0):
    """
    Does the per-hitter excess depend on EXPOSURE once talent is controlled for?

    The four level contributors explain a level. None explains why the bias grows with
    exposure. This regresses the excess jointly on prior plate appearances and a talent
    proxy, so an exposure coefficient survives only if it is not talent wearing exposure's
    clothes -- high-exposure hitters are better hitters, and a level error that scales with
    talent would show up as an exposure effect in a univariate fit.

    THE PROXY MUST BE C.2's PRIOR-SEASONS POSTERIOR, never observed eval-season wOBA.
    Regressing a residual on its own target induces a mechanical negative correlation through
    the shared `woba` term and would manufacture a talent effect out of arithmetic.

    Intervals are bootstrap, clustered on batter: a hitter contributes up to two rows (vs LHP
    and vs RHP) driven by the same talent, health and home park, so resampling rows would
    count them as independent and understate every interval.
    """
    proxy = c2_bivariate_eb.predict(pa_df, eval_season)[claim1_eval.KEY + ["pred_woba"]]
    frame = eval_frame.merge(proxy.rename(columns={"pred_woba": "c2_prior"}),
                             on=claim1_eval.KEY, how="inner")
    assert len(frame) > 0, "no rows survived the join to the C.2 posterior"

    frame = frame.assign(excess=frame["pred_woba"] - frame["woba"])
    # prior_pa in hundreds, so the coefficient reads as wOBA per 100 prior PA rather than as
    # a number with four leading zeros
    exposure = frame["prior_pa"].to_numpy(dtype="float64") / 100.0
    talent = frame["c2_prior"].to_numpy(dtype="float64")
    y = frame["excess"].to_numpy(dtype="float64")
    weights = frame["denominator"].to_numpy(dtype="float64")

    names = ("intercept", "prior_pa_per_100", "c2_prior")
    design = np.column_stack([np.ones(len(frame)), exposure, talent])
    point = _ols(design, y, weights)

    # the univariate fit is reported alongside, because the WHOLE question is whether the
    # exposure coefficient survives the control
    univariate = _ols(np.column_stack([np.ones(len(frame)), exposure]), y, weights)

    clusters = claim1_eval.batter_clusters(frame)
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, design.shape[1]))
    for draw in range(n_boot):
        rows = claim1_eval._resample(clusters, rng)
        draws[draw] = _ols(design[rows], y[rows], weights[rows])

    coefficients = {}
    for index, name in enumerate(names):
        low, high = np.percentile(draws[:, index], (2.5, 97.5))
        coefficients[name] = {"estimate": float(point[index]),
                              "ci_low": float(low), "ci_high": float(high),
                              "excludes_zero": bool(low > 0 or high < 0)}
    return {"n_rows": int(len(frame)), "n_batters": len(clusters),
            "coefficients": coefficients,
            "univariate_prior_pa_per_100": float(univariate[1]),
            "survives_talent_control":
                coefficients["prior_pa_per_100"]["excludes_zero"]}


def gradient_c(baseline, perturbed, eval_frame, n_boot=2000, seed=0):
    """
    Does a hitter-INDEPENDENT knob produce a hitter-DEPENDENT change?

    If it does, the level error interacts multiplicatively with the hitter rather than adding
    to him, and no additive correction can be right at both ends of the exposure range. The
    test is a regression of the per-hitter wOBA DELTA on exposure.

    The knob is `--no-unmeasured-split`: a fixed league measured share and a fixed league value
    for the unmeasured remainder, identical for every hitter by construction. So any exposure
    dependence in the delta is interaction, not the knob.

    Reusing the two runs step 3 already needs for the unmeasured fix's two-sided check, so this
    costs no additional composition.
    """
    merged = (baseline[claim1_eval.KEY + ["pred_woba"]]
              .merge(perturbed[claim1_eval.KEY + ["pred_woba"]], on=claim1_eval.KEY,
                     suffixes=("_base", "_perturbed")))
    assert len(merged) == len(baseline), \
        "the perturbed run must cover exactly the same hitters as the baseline"
    frame = merged.merge(eval_frame[claim1_eval.KEY + ["prior_pa", "denominator", "stratum"]],
                         on=claim1_eval.KEY, how="inner")

    delta = (frame["pred_woba_perturbed"] - frame["pred_woba_base"]).to_numpy(dtype="float64")
    exposure = frame["prior_pa"].to_numpy(dtype="float64") / 100.0
    weights = frame["denominator"].to_numpy(dtype="float64")
    design = np.column_stack([np.ones(len(frame)), exposure])

    point = _ols(design, delta, weights)
    clusters = claim1_eval.batter_clusters(frame)
    rng = np.random.default_rng(seed)
    draws = np.array([_ols(design[rows := claim1_eval._resample(clusters, rng)],
                           delta[rows], weights[rows])[1]
                      for _ in range(n_boot)])
    low, high = np.percentile(draws, (2.5, 97.5))

    by_stratum = {name: {"n": int(part.shape[0]),
                         "mean_delta": float(delta[part.index.to_numpy()].mean())}
                  for name, part in frame.reset_index(drop=True).groupby("stratum",
                                                                        observed=True)}
    return {"n_rows": int(len(frame)),
            "mean_delta": float(np.average(delta, weights=weights)),
            "slope_per_100_pa": {"estimate": float(point[1]), "ci_low": float(low),
                                 "ci_high": float(high),
                                 "excludes_zero": bool(low > 0 or high < 0)},
            "by_stratum": by_stratum,
            "interacts_multiplicatively": bool(low > 0 or high < 0)}


def hitter_exposure(data_dir, manifest):
    """
    Train pitches per embedding row -- the `n_h` test (b) regresses against.

    Prior PLATE APPEARANCES is the exposure variable everywhere else in this module, because
    that is what the evaluation frame strata are cut on. Here it has to be train pitches: the
    §5 decay-ratio argument is about how many batches carry a gradient for a row against how
    many steps decay it, and that ratio is set by pitches in the train seasons, not by PA in
    any season.

    Read memmapped straight off the two columns it needs rather than through `load_tensors`,
    which would pull the 1.08 GB context block into RAM beside a training run.
    """
    directory = Path(data_dir)
    seasons = np.load(directory / "season.npy", mmap_mode="r")
    hitters = np.load(directory / "hitter.npy", mmap_mode="r")
    train = np.isin(seasons, manifest["train_seasons"])
    counts = np.bincount(np.asarray(hitters)[train], minlength=manifest["n_hitters"] + 1)
    return counts.astype("float64")


def woba_direction(embedding, pred_woba, weights):
    """
    The direction in embedding space along which the model's own wOBA prediction rises.

    Estimated numerically from the model rather than assumed: a PA-weighted least-squares fit of
    per-hitter predicted wOBA on the 32 embedding coordinates, whose coefficient vector points
    the way predicted wOBA increases. Returned as a unit vector plus the fit's R^2, because the
    map from embedding to wOBA is a trunk and five heads and a Markov solve, none of which is
    linear -- a low R^2 means the projection below is measuring a weak shadow of the real
    direction and should be read as such.

    The alternative was a finite difference through the chain at one hitter. Rejected: it is a
    LOCAL gradient at whichever hitter is chosen, and the question is about a population.
    """
    design = np.column_stack([np.ones(len(embedding)), embedding])
    solution = _ols(design, pred_woba, weights)
    direction = solution[1:]
    fitted = design @ solution
    mean = np.average(pred_woba, weights=weights)
    residual = np.average((pred_woba - fitted) ** 2, weights=weights)
    total = np.average((pred_woba - mean) ** 2, weights=weights)
    return direction / np.linalg.norm(direction), float(1 - residual / total)


def gradient_b(embedding, exposure, pred_by_row, pa_by_row, n_boot=2000, seed=0,
               reserved=0, quantiles=5):
    """
    Is the embedding shrunk toward zero for low-exposure hitters, and shrunk where it counts?

    This is the D.7 diagnostic, run early because `phase-d-spec.md` §5 argues the AdamW
    decay-to-gradient ratio reaches 24:1 at the 10th percentile of exposure, which predicts
    that low-`n_h` rows should sit closer to the origin. Two readings, because they can
    disagree: the NORM says how far from the origin a row sits, and the PROJECTION onto the
    wOBA-raising direction says how much of that distance is in the subspace that moves the
    prediction. A row can be large and orthogonal, which is capacity spent on nothing.

    ROW 0 IS EXCLUDED, per D5-R18(1). Every hitter outside the training vocabulary routes to
    it, so it is not a hitter's embedding and its exposure is not a hitter's exposure; leaving
    it in adds one zero-norm point at zero exposure, which is a free confirming observation for
    a shrinkage story it cannot speak to.

    Bootstrap resamples HITTERS, not (hitter, hand) rows: there is one embedding row per
    hitter, so the clustering the rest of this module needs does not apply here.
    """
    rows = np.arange(len(embedding))
    trained = (rows != reserved) & (exposure > 0)
    embedding, exposure = embedding[trained], exposure[trained]
    norm = np.linalg.norm(embedding, axis=1)

    # scored rows are a SUBSET of trained rows -- a hitter needs eval-season PA to have a
    # predicted wOBA at all -- so the direction is fitted on the smaller set and then applied
    # to every trained row
    scored = np.isfinite(pred_by_row[trained]) & (pa_by_row[trained] > 0)
    direction, r_squared = woba_direction(embedding[scored], pred_by_row[trained][scored],
                                          pa_by_row[trained][scored])
    projection = embedding @ direction

    # per 1000 train pitches, so the coefficient is readable rather than four leading zeros
    design = np.column_stack([np.ones(len(exposure)), exposure / 1000.0])
    ones = np.ones(len(exposure))
    rng = np.random.default_rng(seed)
    # the one confound a norm reading has: a row that never moved keeps its initialization norm,
    # which would look like a large embedding without any training behind it. Reported beside the
    # means so "these rows are just at init" can be ruled out by the reader rather than asserted
    # by the caller
    from src.model.v1 import EMBEDDING_INIT_STD
    out = {"n_trained_rows": int(trained.sum()), "n_scored_rows": int(scored.sum()),
           "direction_r_squared": r_squared,
           "median_train_pitches": float(np.median(exposure)),
           "init_norm_reference": float(EMBEDDING_INIT_STD * np.sqrt(embedding.shape[1]))}
    for name, y in (("norm", norm), ("projection", projection),
                    ("abs_projection", np.abs(projection))):
        point = _ols(design, y, ones)
        draws = np.array([_ols(design[pick], y[pick], ones)[1] for pick in
                          (rng.integers(0, len(y), len(y)) for _ in range(n_boot))])
        low, high = np.percentile(draws, (2.5, 97.5))
        out[name] = {"mean": float(y.mean()),
                     "slope_per_1000_pitches": float(point[1]),
                     "ci_low": float(low), "ci_high": float(high),
                     "excludes_zero": bool(low > 0 or high < 0),
                     "spearman": float(pd.Series(y).corr(pd.Series(exposure), method="spearman"))}
    edges = np.quantile(exposure, np.linspace(0, 1, quantiles + 1))
    bucket = np.clip(np.digitize(exposure, edges[1:-1]), 0, quantiles - 1)
    out["by_exposure_quintile"] = [
        {"quintile": int(index + 1), "n": int((bucket == index).sum()),
         "median_train_pitches": float(np.median(exposure[bucket == index])),
         "mean_norm": float(norm[bucket == index].mean()),
         "mean_projection": float(projection[bucket == index].mean())}
        for index in range(quantiles)]
    # the sign convention worth stating once: POSITIVE means low-exposure rows sit closer to the
    # origin, which is the shrinkage §5 predicts. Negative is anti-shrinkage.
    out["shrinkage_in_norm"] = bool(out["norm"]["excludes_zero"]
                                    and out["norm"]["slope_per_1000_pitches"] > 0)
    # the wOBA flag reads |projection|, NOT the signed projection. The signed slope is positive
    # whenever rare rows sit on the negative side of the axis, which is the anti-shrinkage case
    # -- so the earlier signed test reported shrinkage exactly when there was none. Distance
    # from zero is the quantity "closer to the origin" was ever about.
    out["shrinkage_in_woba_direction"] = bool(out["abs_projection"]["excludes_zero"]
                                              and out["abs_projection"]["slope_per_1000_pitches"] > 0)
    return out


def per_hitter_predictions(predictions, vocabulary, n_rows):
    """
    Collapse a predictions CSV to one PA-weighted wOBA per embedding row.

    A hitter appears once per opposing hand, and the two rows are the same embedding seen
    through different contexts, so they average rather than count twice. Rows with no
    prediction stay NaN and are dropped downstream rather than filled.
    """
    frame = predictions.copy()
    frame["row"] = frame["batter"].astype(str).map(
        {key: int(value) for key, value in vocabulary.items()})
    frame = frame[frame["row"].notna()]
    weight = frame["pa"] if "pa" in frame else pd.Series(1.0, index=frame.index)
    frame = frame.assign(weight=weight, weighted=frame["pred_woba"] * weight)
    grouped = frame.groupby(frame["row"].astype(int))[["weight", "weighted"]].sum()

    pred = np.full(n_rows, np.nan)
    pa = np.zeros(n_rows)
    pred[grouped.index] = grouped["weighted"] / grouped["weight"]
    pa[grouped.index] = grouped["weight"]
    return pred, pa


def _self_check():
    """Smallest checks that fail if the two regressions stop measuring what they claim to."""
    rng = np.random.default_rng(0)
    n = 400
    exposure = rng.uniform(0, 600, n)
    y = 0.01 + 0.002 * (exposure / 100.0) + rng.normal(0, 1e-4, n)
    fit = _ols(np.column_stack([np.ones(n), exposure / 100.0]), y, np.ones(n))
    assert abs(fit[0] - 0.01) < 1e-3 and abs(fit[1] - 0.002) < 1e-3, fit

    # weights must actually bind: one row with enormous weight drags the intercept to it
    y2 = np.concatenate([np.zeros(n), [1.0]])
    design = np.ones((n + 1, 1))
    weights = np.concatenate([np.ones(n), [1e6]])
    assert _ols(design, y2, weights)[0] > 0.99, "weights are being ignored"

    n_x, n_z = 50, 50
    corner = _edge_distance(np.array([0]), np.array([0]), n_x, n_z)
    middle = _edge_distance(np.array([25]), np.array([25]), n_x, n_z)
    assert corner[0] == 0 and middle[0] == 24, (corner, middle)

    # test (b) on a planted shrinkage: rows are scaled by their own exposure, so the norm slope
    # must be positive, and the direction must come back as the one wOBA was built along
    n_rows, dim = 300, 8
    exposure = np.concatenate([[0.0], rng.uniform(200, 20000, n_rows - 1)])
    truth = np.zeros(dim)
    truth[2] = 1.0
    raw = rng.normal(0, 1, (n_rows, dim))
    embedding = raw * (exposure / exposure.max())[:, None]
    embedding[0] = 0.0
    pred = 0.30 + 0.05 * (embedding @ truth)
    pred[0] = np.nan  # row 0 has no hitter, so it has no prediction
    result = gradient_b(embedding, exposure, pred, np.where(np.isfinite(pred), 400.0, 0.0),
                        n_boot=200)
    assert result["n_trained_rows"] == n_rows - 1, "row 0 or a zero-exposure row got through"
    assert result["shrinkage_in_norm"], result["norm"]
    assert result["shrinkage_in_woba_direction"], result["abs_projection"]
    assert result["direction_r_squared"] > 0.99, result["direction_r_squared"]
    recovered, _ = woba_direction(embedding[1:], pred[1:], np.full(n_rows - 1, 400.0))
    assert abs(recovered @ truth) > 0.99, recovered

    # anti-shrinkage must read as anti-shrinkage: invert the scaling and the flag has to clear
    flipped = raw * (exposure.max() / np.maximum(exposure, 1.0))[:, None]
    flipped[0] = 0.0
    anti = gradient_b(flipped, exposure, pred, np.where(np.isfinite(pred), 400.0, 0.0),
                      n_boot=200)
    assert not anti["shrinkage_in_norm"], anti["norm"]
    assert not anti["shrinkage_in_woba_direction"], anti["abs_projection"]
    print("self-check passed")


def main():
    parser = argparse.ArgumentParser(description="D.5 level-bias attribution and gradient tests.")
    parser.add_argument("--predictions", default=None,
                        help="baseline d5_predictions_*.csv; gradient (a) needs it")
    parser.add_argument("--perturbed", default=None,
                        help="a second d5_predictions_*.csv from the same arm with one knob "
                             "flipped; enables gradient test (c)")
    parser.add_argument("--eval-targets", default="data/processed/eval_targets_pa.parquet")
    parser.add_argument("--pitch-events", default="data/processed/pitch_events_labeled.parquet")
    parser.add_argument("--data-dir", default="data/processed/phase_d")
    parser.add_argument("--eval-season", type=int, default=2024)
    parser.add_argument("--final-run", action="store_true")
    parser.add_argument("--gradient-b-arm", default=None,
                        help="checkpoint stem for test (b), e.g. d10_baseline. Reads "
                             "<arm>_s<seed>.pt beside <out-dir>/d5_predictions_<arm>_s<seed>.csv, "
                             "which is what the step 5 driver writes")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--checkpoint-dir", default="results/checkpoints")
    parser.add_argument("--skip-hbp", action="store_true",
                        help="the HBP diagnosis reads the aligned pitch frame, which is the "
                             "expensive part of this module")
    parser.add_argument("--out-dir", default="results/phase_d")
    parser.add_argument("--out-name", default="d5_level_attribution.json")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    if args.self_check:
        _self_check()
        return 0

    claim1_eval.assert_not_test_season(args.eval_season, final_run=args.final_run)
    pa_df = pd.read_parquet(args.eval_targets)
    report = {"eval_season": args.eval_season}

    if not args.skip_hbp:
        from src.model import loader
        tensors, manifest = loader.load_tensors(args.data_dir)
        frame = qt.align_pitch_frame(args.pitch_events, args.eval_targets, tensors["season"])
        # the same mask `build_tables` fits every auxiliary table under, so the surface this
        # diagnoses is the surface the composition actually used
        train_mask = np.isin(frame["season"].to_numpy(), manifest["train_seasons"])
        report["hbp"] = hbp_diagnosis(frame, train_mask)
        print("-- HBP: clipping or shrinkage? --")
        for record in report["hbp"]["by_edge_distance"]:
            print(f"  edge {record['edge_distance_cells']:>7} cells  n={record['n_takes']:>9}  "
                  f"clipped {record['clipped_share']:6.2%}  "
                  f"observed {record['observed_hbp']:.5f}  shrunk {record['shrunk_hbp']:.5f}  "
                  f"inflation {record['shrinkage_inflation']:+.5f}")
        population = report["hbp"]["population"]
        print(f"  population: observed {population['observed_hbp']:.5f}  "
              f"shrunk {population['shrunk_hbp']:.5f}  "
              f"inflation {population['shrinkage_inflation']:+.5f}  alpha {population['alpha']:.4f}")
        print(f"  clipped takes {population['clipped_takes']} at HBP "
              f"{population['clipped_hbp']:.5f} against {population['unclipped_hbp']:.5f} "
              f"unclipped")

    if args.predictions:
        predictions = pd.read_csv(args.predictions)
        eval_frame, coverage = claim1_eval.build_eval_frame(pa_df, predictions,
                                                            args.eval_season)
        report["coverage"] = coverage
        report["gradient_a"] = gradient_a(eval_frame, pa_df, args.eval_season)
        print("\n-- gradient test (a): excess ~ prior PA + C.2 posterior --")
        for name, value in report["gradient_a"]["coefficients"].items():
            print(f"  {name:>18}: {value['estimate']:+.6f} "
                  f"[{value['ci_low']:+.6f}, {value['ci_high']:+.6f}] "
                  f"{'EXCLUDES ZERO' if value['excludes_zero'] else 'contains zero'}")
        print(f"  univariate exposure coefficient: "
              f"{report['gradient_a']['univariate_prior_pa_per_100']:+.6f}")
        print(f"  exposure survives the talent control: "
              f"{report['gradient_a']['survives_talent_control']}")

        if args.perturbed:
            report["gradient_c"] = gradient_c(predictions, pd.read_csv(args.perturbed),
                                              eval_frame)
            slope = report["gradient_c"]["slope_per_100_pa"]
            print("\n-- gradient test (c): perturb and re-solve --")
            print(f"  mean delta {report['gradient_c']['mean_delta']:+.6f}")
            print(f"  slope per 100 prior PA: {slope['estimate']:+.8f} "
                  f"[{slope['ci_low']:+.8f}, {slope['ci_high']:+.8f}]")
            print(f"  interacts multiplicatively: "
                  f"{report['gradient_c']['interacts_multiplicatively']}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.gradient_b_arm:
        import torch

        manifest = json.loads((Path(args.data_dir) / "manifest.json").read_text())
        exposure = hitter_exposure(args.data_dir, manifest)
        report["gradient_b"] = {}
        print("\n-- gradient test (b): embedding norm and wOBA-projection against train "
              "exposure, row 0 excluded --")
        for seed in args.seeds:
            checkpoint = Path(args.checkpoint_dir) / f"{args.gradient_b_arm}_s{seed}.pt"
            csv = out_dir / f"d5_predictions_{args.gradient_b_arm}_s{seed}.csv"
            if not (checkpoint.exists() and csv.exists()):
                print(f"  seed {seed}: skipped (missing "
                      f"{checkpoint if not checkpoint.exists() else csv})")
                continue
            weight = torch.load(checkpoint, map_location="cpu",
                                weights_only=False)["model"]["embedding.weight"].numpy()
            assert not weight[manifest["reserved_hitter_index"]].any(), \
                "the reserved row is not zero, so it was trained and the exclusion is not enough"
            pred, pa = per_hitter_predictions(pd.read_csv(csv), manifest["vocabulary"],
                                             len(weight))
            result = gradient_b(weight.astype("float64"), exposure, pred, pa,
                                reserved=manifest["reserved_hitter_index"])
            report["gradient_b"][f"seed_{seed}"] = result
            for name in ("norm", "projection"):
                part = result[name]
                print(f"  seed {seed} {name:>10}: mean {part['mean']:.4f}  slope "
                      f"{part['slope_per_1000_pitches']:+.6f} per 1000 pitches "
                      f"[{part['ci_low']:+.6f}, {part['ci_high']:+.6f}] "
                      f"rho {part['spearman']:+.3f} "
                      f"{'EXCLUDES ZERO' if part['excludes_zero'] else 'contains zero'}")
            print(f"  seed {seed} direction R^2 {result['direction_r_squared']:.4f}  "
                  f"shrinkage in norm {result['shrinkage_in_norm']}  "
                  f"in wOBA direction {result['shrinkage_in_woba_direction']}")
    # named rather than fixed, so a test-(b) run cannot land on top of step 3's attribution file.
    # Merged rather than replaced on top of that: the three gradient tests run on different
    # populations -- (a) on all scored hitter-seasons, (c) on ~50 spanning the exposure range, (b)
    # per seed -- so they are separate invocations by construction, and a plain write means the
    # last one silently deletes the others' results. Already happened once: the full (a) run
    # removed (c) from this file. A test whose result vanishes because a sibling test ran later is
    # a worse failure than a stale key, so new keys win and absent ones are left alone.
    out_path = out_dir / args.out_name
    if out_path.exists():
        report = {**json.loads(out_path.read_text()), **report}
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
