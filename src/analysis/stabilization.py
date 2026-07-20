"""
Split-half reliability / stabilization analysis (Phase B step 1).

Measures how much of a metric's between-hitter variance is signal vs. noise at a
given sample size, i.e. how many observations until the metric "stabilizes". This
ranks candidate metrics by signal-per-PA, the currency of the small-sample thesis:
the headline question is whether process metrics (whiff rate, exit velocity)
stabilize faster than the outcome we project (side-specific wOBA).

Method (classical test theory). For sample size n, each qualifying hitter's n
observations are split into two random halves of n/2. The Pearson correlation r
of the half-metrics across hitters is the reliability of an n/2-sized measurement;
Spearman-Brown projects it up to n:  reliability(n) = 2r / (1 + r). Algebraically
this equals signal / (signal + noise/n), the reliability of an n-observation
estimate. The stabilization point is the n where reliability crosses 0.5 (half the
variance is signal) -- the widely-used baseball convention (Carleton / Baseball
Prospectus reliability work; unverified, to source before it is cited in the paper).

The metric is supplied as a function of a slice of observations, so it covers both
simple means (exit velocity) and ratios (wOBA = sum(points)/sum(denominator)). The
caller pre-filters to the metric's natural observation unit (swings for whiff, PAs
vs. a given pitcher hand for side-specific wOBA) and to train seasons.
"""

import numpy as np
import pandas as pd


def spearman_brown(r):
    """Project a split-half correlation to full-length reliability: 2r / (1 + r)."""
    return 2.0 * r / (1.0 + r)


def mean_metric(column):
    """Metric fn for a simple average, e.g. exit velocity or a 0/1 rate."""
    return lambda observations: observations[column].mean()


def ratio_metric(numerator, denominator):
    """Metric fn for a ratio like wOBA = sum(points) / sum(denominator); nan if denom is 0."""
    def compute(observations):
        denom = observations[denominator].sum()
        return observations[numerator].sum() / denom if denom > 0 else np.nan
    return compute


def _pearson(a, b):
    """Pearson correlation over finite pairs; nan if under 2 usable pairs or no spread."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2:
        return np.nan
    a, b = a[ok], b[ok]
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def reliability_at(groups, metric_fn, n, rng, n_resamples=1):
    """
    Spearman-Brown corrected split-half reliability of an n-observation estimate.

    groups: dict of hitter_id -> that hitter's observations (index reset).
    Splits n sampled rows into halves of n//2, computes the metric on each half,
    and correlates the halves across hitters. Averages over n_resamples random
    splits to damp Monte Carlo noise. Returns (reliability, n_qualifying_hitters).
    """
    half = n // 2
    qualifying = [g for g in groups.values() if len(g) >= n]
    if half < 1 or len(qualifying) < 2:
        return np.nan, len(qualifying)

    reliabilities = []
    for _ in range(n_resamples):
        half_a, half_b = [], []
        for group in qualifying:
            order = rng.permutation(len(group))[:n]
            half_a.append(metric_fn(group.iloc[order[:half]]))
            half_b.append(metric_fn(group.iloc[order[half:2 * half]]))
        r = _pearson(half_a, half_b)
        if np.isfinite(r):
            reliabilities.append(spearman_brown(r))

    reliability = float(np.mean(reliabilities)) if reliabilities else np.nan
    return reliability, len(qualifying)


def stabilization_curve(df, group_col, metric_fn, n_grid, seed=0, n_resamples=1):
    """
    Reliability vs. sample size for one metric. Returns a DataFrame with columns
    n, reliability, n_groups. n_groups is the qualifying-hitter count at each n and
    falls as n grows (survivorship) -- report it wherever the curve is reported.
    """
    rng = np.random.default_rng(seed)
    groups = {gid: sub.reset_index(drop=True) for gid, sub in df.groupby(group_col)}
    rows = []
    for n in n_grid:
        reliability, n_groups = reliability_at(groups, metric_fn, n, rng, n_resamples)
        rows.append({"n": n, "reliability": reliability, "n_groups": n_groups})
    return pd.DataFrame(rows)


def stabilization_point(curve, threshold=0.5):
    """
    Sample size where reliability first crosses `threshold`, linearly interpolated
    between the bracketing grid points. nan if the curve never reaches threshold.
    """
    usable = curve.dropna(subset=["reliability"]).sort_values("n").reset_index(drop=True)
    previous = None
    for _, row in usable.iterrows():
        if row["reliability"] >= threshold:
            if previous is None:
                return float(row["n"])
            span_r = row["reliability"] - previous["reliability"]
            if span_r <= 0:
                return float(row["n"])
            frac = (threshold - previous["reliability"]) / span_r
            return float(previous["n"] + frac * (row["n"] - previous["n"]))
        previous = row
    return np.nan


def _print_curve(title, curve, unit):
    print(f"\n{title}")
    print(curve.to_string(index=False))
    point = stabilization_point(curve)
    point_str = "not reached" if np.isnan(point) else f"{point:.1f} {unit}"
    print(f"  stabilization point (r=0.5): {point_str}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run the Phase B stabilization panel: process vs. outcome metrics on train seasons.")
    parser.add_argument("--labeled", default="data/processed/pitch_events_labeled.parquet")
    parser.add_argument("--eval-targets", default="data/processed/eval_targets_pa.parquet")
    parser.add_argument("--max-train-season", type=int, default=2023)
    parser.add_argument("--resamples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    grid = [10, 25, 50, 100, 200, 400, 800]
    train = lambda d: d[d["season"] <= args.max_train_season]

    # process metrics, one per model head, each in its natural observation unit
    labeled = pd.read_parquet(args.labeled, columns=["batter", "season", "swing", "contact", "ev", "la", "spray"])
    swings = train(labeled[labeled["swing"] == 1]).copy()
    swings["whiff"] = (swings["contact"] == 0).astype(float)
    in_play = train(labeled[labeled["ev"].notna()])
    sprayed = train(labeled[labeled["spray"].notna()])

    process_panel = [
        ("swing rate (per pitch) -> swing head", train(labeled), mean_metric("swing"), "pitches"),
        ("whiff rate (per swing) -> contact head", swings, mean_metric("whiff"), "swings"),
        ("exit velocity (per ball-in-play) -> quality head", in_play, mean_metric("ev"), "BBIP"),
        ("launch angle (per ball-in-play) -> quality head", in_play, mean_metric("la"), "BBIP"),
        ("spray / hit direction (per ball-in-play) -> quality head", sprayed, mean_metric("spray"), "BBIP"),
    ]
    for title, frame, metric, unit in process_panel:
        curve = stabilization_curve(frame, "batter", metric, grid, seed=args.seed, n_resamples=args.resamples)
        _print_curve(title, curve, unit)

    # outcome metric: side-specific wOBA, each pitcher hand
    pa = train(pd.read_parquet(args.eval_targets, columns=["batter", "season", "p_throws", "woba_points", "in_denominator"]))
    for hand, label in [("L", "vs LHP"), ("R", "vs RHP")]:
        side = pa[pa["p_throws"] == hand]
        curve = stabilization_curve(side, "batter", ratio_metric("woba_points", "in_denominator"),
                                    grid, seed=args.seed, n_resamples=args.resamples)
        _print_curve(f"wOBA {label} (per PA) -> outcome", curve, "PA")


if __name__ == "__main__":
    main()
