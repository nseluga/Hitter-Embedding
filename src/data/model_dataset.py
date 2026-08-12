"""
Phase D.1 — the pitch-level training tensors for the conditional-query model.

One row per PITCH (§2.2), not per hitter-season: the hitter enters as an integer
index into the embedding table and the model's job is p(process | hitter, context).
The (batter, season, hand) row unit belongs to `claim1_eval`, which SCORES the
model, and to the Phase C baselines, which train at that unit. Nothing here.

WHO IS IN THE TABLE. Pitchers taking their own at-bats are dropped (2026-08-01),
using the same `primarily_pitchers` definition every Phase C hitter-talent quantity
uses. `clean.py` filters position players on the PITCHING side only, so before this
they were 35.7% of the embedding table's rows while holding 1.60% of its pitches —
parameters spent on a population `claim1_eval` drops before scoring, and therefore
one the query machinery can never ask about.

THE THREE FROZEN QUANTITIES, all fit on TRAIN SEASONS ONLY. Each is a leakage
boundary enforced by assertion rather than convention, because all three are the
kind that fail silently:

  1. Hitter vocabulary. A batter seen in a train season gets an index; everyone
     else routes to RESERVED_HITTER_INDEX, whose embedding row is zero-initialized
     and never trained (2026-07-30). Building the vocabulary from all seasons would
     hand an unseen hitter a row that receives no gradient yet is not zero — a
     random personality, failing silently in exactly the population claim 1 is about.
  2. Contact-quality bin edges. Quantile edges on train values, so each bin carries
     equal train mass. Equal-WIDTH bins would put nearly every batted ball in a
     handful of EV bins and waste the head's capacity on empty tails.
  3. Context standardization. Already train-only by construction; reused from the
     Phase B vectorizer rather than refit, so Phase D conditions on exactly the
     vector B.2 screened.

THE SELECTION FRAME (2026-08-01). Train is 2015-2023 for the selection runs, with
2024 held out for early stopping and every §4 ablation; the winning configuration
refits on 2015-2024 and reports on 2025. `build` takes its train seasons as an
argument for that reason — the refit is the same call with one more season, not a
second code path.

FEATURE SELECTION LIVES HERE, NOT IN THE VECTORIZER. `context_features` stays the
frozen Phase B artifact; Phase D chooses which of its columns to use. Two reasons:
D.8 has to toggle the flagged block in and out anyway, so column selection must
exist regardless; and refitting the vectorizer would move the standardization
constants, making a feature ablation also a rescaling.

LABEL MASKING follows the §1.2 nesting exactly. `contact` is defined only on
swings and `ev`/`la`/`spray` only on balls in play, so each is emitted with an
explicit mask rather than a fill value. A masked label is stored as MASKED (-1),
never 0, because 0 is a legitimate class index for every one of these heads.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.splits import load_splits
from src.data.eval_targets import primarily_pitchers
from src.features import context_features

# Row 0 of the embedding table: zero-initialized, never trained, the destination for
# any batter absent from the train vocabulary (2026-07-30). Real hitters start at 1.
RESERVED_HITTER_INDEX = 0

# the committed Phase B fit, reused rather than refit — see FEATURE SELECTION above
VECTORIZER_PARAMS_PATH = "src/features/context_vectorizer_params.json"

# §1.5 puts each discretized quality head at roughly 20-40 bins. Quantile edges, so
# the count is a capacity choice rather than a claim about the distribution's shape.
DEFAULT_QUALITY_BINS = 24
QUALITY_DIMENSIONS = ("ev", "la", "spray")

# sentinel for "this label does not exist for this pitch" — see LABEL MASKING above
MASKED = -1

# dropped on measured redundancy: R^2 = 0.993 on the kept features once
# release_extension is among them (2026-08-01)
DROPPED_FEATURES = ("effective_speed",)

# B.2's remaining flagged five, entering D.8 as one pre-registered block ablation
ABLATION_BLOCK = ("release_extension", "release_pos_y", "release_pos_z",
                  "release_spin_rate", "spin_axis")

# the three-class contact split (2026-08-08) is read off `description`, which the
# context deliberately excludes but the label needs
SPLIT_DESCRIPTION_TO_CLASS = {"foul": 0, "foul_tip": 1, "hit_into_play": 2}

SOURCE_COLUMNS = (["batter", "season", "swing", "contact", "description"]
                  + list(QUALITY_DIMENSIONS)
                  + context_features.STANDARDIZED
                  + context_features.CIRCULAR
                  + context_features.CATEGORICAL)


def vectorizer_columns(field, names):
    """
    The context-matrix column names produced by one raw field.
    field: a raw Statcast column; names: the vectorizer's ordered feature names.
    A circular field contributes sin/cos, an optional field also a missingness flag.
    """
    produced = [field, f"{field}_sin", f"{field}_cos", f"{field}_missing"]
    return [name for name in names if name in produced]


def select_columns(names, drop=DROPPED_FEATURES, include_block=True):
    """
    Indices of the context columns Phase D uses, given the vectorizer's full ordering.
    drop: raw fields removed outright; include_block: whether B.2's flagged five are in.
    Returns (indices, kept_names) so the choice is recorded alongside the tensor.
    """
    excluded = set()
    for field in drop:
        excluded.update(vectorizer_columns(field, names))
    if not include_block:
        for field in ABLATION_BLOCK:
            excluded.update(vectorizer_columns(field, names))
    indices = [i for i, name in enumerate(names) if name not in excluded]
    assert indices, "feature selection removed every context column"
    return np.array(indices), [names[i] for i in indices]


def drop_pitcher_at_bats(pitch_df, excluded):
    """
    Remove pitches thrown to a pitcher taking his own turn at bat.
    excluded: set of (season, batter), from `eval_targets.primarily_pitchers` on the
    COMPLETE PA source — one definition of "pitcher-batter", shared with the Phase C
    quantities, so the two cannot drift apart.

    Applied to every season, not just train. These batters are dropped by
    `claim1_eval` before scoring, so a row trained for them is a row that can never
    be queried, and `clean.py` filters position players only on the PITCHING side.
    """
    if not excluded:
        return pitch_df
    keys = pd.MultiIndex.from_arrays([pitch_df["season"], pitch_df["batter"]])
    return pitch_df[~keys.isin(excluded)]


def build_vocabulary(pitch_df, train_seasons):
    """
    Map each batter appearing in train_seasons to an embedding index starting at 1.
    Index 0 is reserved and unassigned; sorted batter ids make the mapping stable
    across runs, which the reproducibility gate depends on.
    """
    train = pitch_df[pitch_df["season"].isin(train_seasons)]
    assert len(train) > 0, f"no pitches in train seasons {train_seasons}"
    batters = np.sort(train["batter"].unique())
    return {int(batter): index for index, batter in enumerate(batters, start=1)}


def hitter_indices(pitch_df, vocabulary):
    """Embedding index per pitch; batters outside the vocabulary get the reserved row."""
    return (pitch_df["batter"].map(vocabulary)
            .fillna(RESERVED_HITTER_INDEX).to_numpy(dtype="int64"))


# --- D5-R17: Statcast launch-angle placeholders ---------------------------------------
#
# Two (launch_angle, launch_speed) pairs are Statcast PLACEHOLDERS rather than measurements,
# verified 2026-08-11: at launch_angle -21.0 the paired launch_speed is exactly 82.9,
# hit_distance_sc is null for 77.5% of those rows against a <=5.5% baseline, and 33,993 of
# 34,001 are `ground_ball`; at 69.0 the speed is exactly 80.0, distance is null for 90.7%, and
# 22,354 of 22,372 are `popup`.
#
# They are a different measurement PROCESS, not missing data, so the Phase B imputation rule
# does not apply and they are DROPPED (settled Q2=C) -- from the bin-edge computation and from
# head_la's targets both. Masking alone would leave them in the edge computation, which is
# where they did the damage: the -21 spike is 59.9% of LA bin 2 and the 69 spike 44.0% of LA
# bin 23, producing a 3-degree-wide bin against 5-11 degree neighbours.
#
# Matched on the PAIR, not on launch_angle alone: about a fifth of rows at -21 degrees carry a
# real exit velocity and are genuine batted balls.
PLACEHOLDER_PAIRS = ((-21.0, 82.9), (69.0, 80.0))
# A re-pull that changes Statcast's placeholder values must fail loud rather than silently
# dropping nothing. Verified share on snapshot_2026-07-14 is ~5.5% of balls in play.
PLACEHOLDER_MIN_SHARE = 0.02

# How a dimension's interior edges are chosen. Compared on the JOINT within-cell wOBA variance
# by `joint_cell_variance`, never on the marginal: a 100 mph ball at 5 degrees and at 25 degrees
# are different outcomes, so choosing EV edges against wOBA averaged over launch angle optimizes
# the wrong quantity. The 1-D criterion below is only how a CANDIDATE is generated.
BIN_SCHEMES = ("equal_mass", "top_decile_split", "variance_min")
# `top_decile_split` puts this share of the mass in the tail and this many bins inside it
TAIL_SHARE = 0.10
TAIL_BINS = 6
# Pre-launch blocker threshold. `query_tables.fit_outcome_table` backs a joint (ev, la, spray)
# cell off to its (ev, la) marginal, and that backoff was fitted at an average joint cell count
# of 73.9. Unequal bins concentrate mass, so if a candidate pushes the SMALLEST occupied cell
# below this the backoff has to be re-fit BEFORE the overnight rather than discovered during it.
BACKOFF_MIN_CELL = 30


def placeholder_mask(pitch_df):
    """Rows carrying a Statcast (launch_angle, launch_speed) placeholder pair (D5-R17)."""
    la = pitch_df["la"].to_numpy(dtype="float64")
    ev = pitch_df["ev"].to_numpy(dtype="float64")
    mask = np.zeros(len(pitch_df), dtype=bool)
    for angle, speed in PLACEHOLDER_PAIRS:
        mask |= (la == angle) & (ev == speed)
    return mask


def assert_placeholders_present(pitch_df):
    """
    Fail loud if PLACEHOLDER_PAIRS no longer matches the snapshot (D5-R17).

    Called from `build` rather than from `fit_bin_edges`, because it is a statement about the
    REAL table: a hand-built frame legitimately contains no placeholders, while a real snapshot
    whose placeholder values Statcast has changed would silently drop nothing and quietly
    restore the defect. Returns the matched share.
    """
    placeholders = placeholder_mask(pitch_df)
    in_play = int(pitch_df["ev"].notna().sum())
    assert in_play > 0, "no balls in play, so the placeholder guard cannot be evaluated"
    share = float(placeholders.sum()) / in_play
    assert share >= PLACEHOLDER_MIN_SHARE, (
        f"placeholder pairs matched only {share:.4%} of balls in play, below the verified "
        f"~5.5% -- Statcast's placeholder values have changed and PLACEHOLDER_PAIRS is stale")
    return share


def _equal_mass_edges(values, n_bins):
    """Interior quantile edges: every bin carries the same share of the mass."""
    return np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])


def _top_decile_edges(values, n_bins, tail_share=TAIL_SHARE, tail_bins=TAIL_BINS):
    """
    Equal mass over the body, then `tail_bins` equal-mass bins inside the top `tail_share`.
    Equal mass alone cannot narrow the tail -- 24 equal-mass bins give a 15.1 mph top EV bin
    and 32 still give 14.3 at 2.4x the chain enumeration -- so only unequal mass can.
    """
    body_bins = n_bins - tail_bins
    assert body_bins >= 2, f"{n_bins} bins with {tail_bins} in the tail leaves too few"
    body = np.linspace(0.0, 1.0 - tail_share, body_bins + 1)[1:]
    tail = np.linspace(1.0 - tail_share, 1.0, tail_bins + 1)[1:-1]
    return np.quantile(values, np.concatenate([body, tail]))


def _variance_min_edges(values, target, n_bins):
    """
    Edges minimizing within-bin variance of `target`, by exact 1-D dynamic program.

    This is the Fisher-Jenks / `Ckmeans.1d.dp` recurrence (Wang & Song 2011, R Journal 3(2)
    29-33; Fisher 1958, JASA 53(284) 789-798) with one substitution: the cost of an interval is
    the sum of squared deviations of the TARGET inside it, not of the binned variable. That makes
    it a supervised partition -- it separates wOBA outcomes -- rather than a description of the
    feature's own shape.

    Exact, not greedy: the DP runs over DISTINCT feature values with the target's count, sum and
    sum-of-squares accumulated per value, which is what keeps an O(K * m^2) recurrence affordable
    on a million rows. Statcast reports exit velocity to 0.1 mph, so m is in the low thousands.
    """
    order = np.argsort(values, kind="stable")
    unique, start = np.unique(values[order], return_index=True)
    m = len(unique)
    assert m > n_bins, f"only {m} distinct values for {n_bins} bins"

    sorted_target = target[order]
    # prefix sums over distinct-value groups: [0, g1, g1+g2, ...]
    bounds = np.append(start, len(order))
    count = np.diff(bounds).astype("float64")
    total = np.add.reduceat(sorted_target, start)
    total_sq = np.add.reduceat(sorted_target ** 2, start)
    c = np.concatenate([[0.0], np.cumsum(count)])
    s = np.concatenate([[0.0], np.cumsum(total)])
    q = np.concatenate([[0.0], np.cumsum(total_sq)])

    def cost(i, j):
        """Sum of squared deviations of the target over distinct-value groups [i, j)."""
        n = c[j] - c[i]
        run = s[j] - s[i]
        with np.errstate(invalid="ignore", divide="ignore"):
            out = (q[j] - q[i]) - np.where(n > 0, run ** 2 / np.where(n > 0, n, 1.0), 0.0)
        return np.where(n > 0, out, np.inf)

    # dp[j] = best cost partitioning the first j groups into the bins placed so far
    dp = cost(0, np.arange(m + 1))
    dp[0] = 0.0
    back = np.zeros((n_bins, m + 1), dtype="int64")
    for k in range(1, n_bins):
        previous, dp = dp, np.full(m + 1, np.inf)
        for j in range(k + 1, m + 1):
            candidate = previous[k:j] + cost(np.arange(k, j), j)
            best = int(np.argmin(candidate))
            dp[j] = candidate[best]
            back[k, j] = k + best

    # walk the cut points back out; an interior edge sits at the first value of each bin
    cuts, j = [], m
    for k in range(n_bins - 1, 0, -1):
        j = int(back[k, j])
        cuts.append(j)
    return unique[sorted(cuts)]


def fit_bin_edges(pitch_df, train_seasons, n_bins=DEFAULT_QUALITY_BINS,
                  scheme="equal_mass", target=None, drop_placeholders=True):
    """
    Interior edges per contact-quality dimension, from train seasons only.
    Returns {dimension: list of n_bins-1 edges}. Null labels are excluded, so the edges
    describe balls in play rather than the mostly-null pitch population.

    scheme: one of BIN_SCHEMES. "variance_min" requires `target` -- a per-row realized wOBA
    array aligned to pitch_df, on TRAIN seasons only. Statcast's xwOBA would be circular here,
    since `V` is what estimates it.
    drop_placeholders: excludes D5-R17's Statcast placeholder pairs from the edge computation.
    """
    assert scheme in BIN_SCHEMES, f"unknown bin scheme {scheme!r}"
    # .to_numpy() can hand back a read-only view, and this mask is narrowed in place below
    keep = np.array(pitch_df["season"].isin(train_seasons).to_numpy(), dtype=bool)
    if drop_placeholders:
        keep &= ~placeholder_mask(pitch_df)

    edges = {}
    for dimension in QUALITY_DIMENSIONS:
        values = pitch_df[dimension].to_numpy(dtype="float64")
        usable = keep & np.isfinite(values)
        assert usable.sum() > n_bins, f"too few train values to bin {dimension}"
        if scheme == "equal_mass":
            cut = _equal_mass_edges(values[usable], n_bins)
        elif scheme == "top_decile_split":
            cut = _top_decile_edges(values[usable], n_bins)
        else:
            assert target is not None, "the variance_min scheme needs a realized-wOBA target"
            scored = usable & np.isfinite(np.asarray(target, dtype="float64"))
            cut = _variance_min_edges(values[scored],
                                      np.asarray(target, dtype="float64")[scored], n_bins)
        assert np.all(np.diff(cut) > 0), \
            f"{dimension} edges are not strictly increasing at {n_bins} bins ({scheme})"
        edges[dimension] = np.asarray(cut, dtype="float64").tolist()
    return edges


def joint_cell_variance(pitch_df, train_seasons, edges, target, n_bins=DEFAULT_QUALITY_BINS):
    """
    Within-cell wOBA variance over the joint (ev, la, spray) grid, plus the diagnostics that
    decide whether a candidate binning is affordable and safe.

    This is the objective the three candidate schemes are COMPARED on (D5-R7 + D5-R17). The
    joint grid is the right unit because the model predicts into a cell of it, not into three
    marginals: a 100 mph ball at 5 degrees and at 25 degrees are different outcomes.

    `min_cell_count` is the pre-launch blocker -- `query_tables.fit_outcome_table` backs a joint
    cell off to its (ev, la) marginal, and the backoff has to be re-fit if the new edges push
    top cells below the regime it was fitted in.
    """
    values = {d: pitch_df[d].to_numpy(dtype="float64") for d in QUALITY_DIMENSIONS}
    target = np.asarray(target, dtype="float64")
    usable = np.array(pitch_df["season"].isin(train_seasons).to_numpy(), dtype=bool)
    usable &= ~placeholder_mask(pitch_df) & np.isfinite(target)
    for value in values.values():
        usable &= np.isfinite(value)

    bins = {d: assign_bins(values[d][usable], edges[d]) for d in QUALITY_DIMENSIONS}
    flat = np.ravel_multi_index((bins["ev"], bins["la"], bins["spray"]),
                                (n_bins, n_bins, n_bins))
    scored = target[usable]
    size = n_bins ** 3
    count = np.bincount(flat, minlength=size).astype("float64")
    total = np.bincount(flat, weights=scored, minlength=size)
    total_sq = np.bincount(flat, weights=scored ** 2, minlength=size)

    occupied = count > 0
    within = (total_sq[occupied] - total[occupied] ** 2 / count[occupied]).sum()
    grand = total_sq.sum() - total.sum() ** 2 / count.sum()
    return {
        "scheme_rows": int(usable.sum()),
        # the headline: mean squared deviation of realized wOBA from its own cell's mean
        "within_cell_variance": float(within / count.sum()),
        # share of total wOBA variance the grid explains, so schemes are comparable on one scale
        "variance_explained": float(1.0 - within / grand),
        "occupied_cells": int(occupied.sum()),
        "min_cell_count": int(count[occupied].min()),
        "median_cell_count": float(np.median(count[occupied])),
        "top_ev_bin_span": float(values["ev"][usable].max() - edges["ev"][-1]),
        "ev_bin_spans": [round(b - a, 3) for a, b in zip(edges["ev"][:-1], edges["ev"][1:])],
    }


def assign_bins(values, edges):
    """Bin index per value against interior edges; non-finite values return MASKED."""
    values = np.asarray(values, dtype="float64")
    assigned = np.searchsorted(np.asarray(edges, dtype="float64"), values, side="right")
    return np.where(np.isfinite(values), assigned, MASKED).astype("int64")


def encode_labels(pitch_df, edges):
    """
    The three §1.2 factor labels with their nesting masks.
    Returns a dict of int64 arrays: swing (0/1), contact (0/1 or MASKED off swings),
    and one bin array per quality dimension (MASKED off balls in play).
    """
    swing = pitch_df["swing"].to_numpy(dtype="int64")
    contact = pitch_df["contact"]
    labels = {
        "swing": swing,
        "contact": np.where(contact.isna(), MASKED,
                            contact.fillna(0).to_numpy(dtype="int64")).astype("int64"),
    }
    for dimension in QUALITY_DIMENSIONS:
        labels[dimension] = assign_bins(pitch_df[dimension], edges[dimension])

    # the fourth factor: which of the three things a bat-on-ball event actually was.
    # Defined on every contact event, which is strictly more rows than the quality heads
    # get -- fouls and foul tips are contact and carry no batted-ball measurement.
    if "description" in pitch_df:
        mapped = pitch_df["description"].map(SPLIT_DESCRIPTION_TO_CLASS)
        labels["split"] = np.where(mapped.isna(), MASKED,
                                   mapped.fillna(0).to_numpy()).astype("int64")
        assert not (labels["split"][labels["contact"] != 1] != MASKED).any(), \
            "the contact split is defined off a contact event"
        assert not (labels["split"][labels["contact"] == 1] == MASKED).any(), \
            "a contact event carries no split label"

    assert set(np.unique(swing)) <= {0, 1}, "swing is not binary"
    assert not (labels["contact"][swing == 0] != MASKED).any(), \
        "contact is defined on a non-swing, violating the §1.2 nesting"
    for dimension in QUALITY_DIMENSIONS:
        assert not (labels[dimension][labels["contact"] != 1] != MASKED).any(), \
            f"{dimension} is defined off a batted ball, violating the §1.2 nesting"
    return labels


def build(pitch_df, train_seasons=None, n_bins=DEFAULT_QUALITY_BINS,
          include_block=True, params=None, config=None, excluded_batters=None):
    """
    Assemble the Phase D tensors from the labeled pitch table.
    pitch_df: labeled pitches; train_seasons: seasons the frozen quantities are fit
    on, defaulting to the frozen train split; excluded_batters: (season, batter)
    pairs to drop, normally pitchers taking their own at-bats. Returns
    (arrays, manifest), where arrays holds context/hitter/season plus the labels,
    and manifest records every frozen quantity so a rebuild is checkable.
    """
    config = config or load_splits()
    n_source = len(pitch_df)
    pitch_df = drop_pitcher_at_bats(pitch_df, excluded_batters or set())
    train_seasons = sorted(train_seasons or config["split"]["train"])
    assert set(train_seasons) <= set(config["seasons"]), "train seasons outside the split config"
    assert not set(train_seasons) & set(config["split"]["test"]), \
        "the frozen test season cannot fit the vocabulary, bins, or standardization"

    params = params or context_features.load_params(VECTORIZER_PARAMS_PATH)
    assert set(params.fit_seasons) <= set(config["split"]["train"]), \
        "the context vectorizer was fit outside the frozen train seasons"
    matrix, names = context_features.transform(pitch_df, params)
    indices, kept = select_columns(names, include_block=include_block)

    vocabulary = build_vocabulary(pitch_df, train_seasons)
    edges = fit_bin_edges(pitch_df, train_seasons, n_bins)

    # ponytail: dense float32 context, ~29 of 44 columns one-hot. Roughly 1.3 GB over
    # the full table, which fits; switch to categorical index columns embedded in the
    # model if memory ever binds.
    arrays = {
        "context": matrix[:, indices],
        "hitter": hitter_indices(pitch_df, vocabulary),
        "season": pitch_df["season"].to_numpy(dtype="int64"),
        **encode_labels(pitch_df, edges),
    }
    manifest = {
        "train_seasons": train_seasons,
        "n_pitches": int(len(pitch_df)),
        "n_pitcher_at_bat_pitches_dropped": int(n_source - len(pitch_df)),
        "n_hitters": len(vocabulary),
        # the batter -> embedding row map, kept rather than discarded: without it a
        # trained embedding table cannot be joined back to hitters, which the §5.1
        # probe, the query machinery, and the ‖e_h‖-vs-n_h diagnostic all require.
        # JSON keys are strings; callers cast back to int.
        "vocabulary": {str(batter): int(row) for batter, row in vocabulary.items()},
        "reserved_hitter_index": RESERVED_HITTER_INDEX,
        "n_quality_bins": n_bins,
        "quality_bin_edges": edges,
        "context_columns": kept,
        "dropped_features": list(DROPPED_FEATURES),
        "ablation_block_included": bool(include_block),
        "vectorizer_fit_seasons": list(params.fit_seasons),
    }
    return arrays, manifest


def save(arrays, manifest, out_dir):
    """Write each array as .npy beside a manifest.json, for memmapped reloading."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        # row-major, always. `context` arrives F-contiguous because it is built by
        # column selection, and np.save preserves order, which puts one pitch's 46
        # floats ~29 MB apart: a gather of 8,192 training rows measured 8.74 s
        # memmapped against 0.033 s row-major in RAM. The refit rebuilds through
        # this same function, so the fix belongs here rather than in a loader.
        np.save(out_dir / f"{name}.npy", np.ascontiguousarray(array))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out_dir


def load(out_dir, mmap_mode="r"):
    """Reload a built dataset; arrays are memmapped so a training run need not hold it all."""
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    arrays = {path.stem: np.load(path, mmap_mode=mmap_mode)
              for path in sorted(out_dir.glob("*.npy"))}
    assert len(arrays["hitter"]) == manifest["n_pitches"], "manifest disagrees with the arrays"
    return arrays, manifest


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build the Phase D pitch tensors.")
    parser.add_argument("--pitch-events", default="data/processed/pitch_events_labeled.parquet")
    parser.add_argument("--eval-targets", default="data/processed/eval_targets_pa.parquet",
                        help="complete PA source, used only to identify pitcher-batters")
    parser.add_argument("--out-dir", default="data/processed/phase_d")
    parser.add_argument("--n-bins", type=int, default=DEFAULT_QUALITY_BINS)
    parser.add_argument("--train-seasons", type=int, nargs="*", default=None,
                        help="defaults to the frozen train split; pass 2015..2024 for the refit")
    parser.add_argument("--no-ablation-block", action="store_true",
                        help="drop B.2's flagged five for the D.8 block ablation")
    args = parser.parse_args()

    pitch_df = pd.read_parquet(args.pitch_events, columns=sorted(set(SOURCE_COLUMNS)))
    excluded = primarily_pitchers(pd.read_parquet(args.eval_targets))
    arrays, manifest = build(pitch_df, train_seasons=args.train_seasons, n_bins=args.n_bins,
                             include_block=not args.no_ablation_block,
                             excluded_batters=excluded)
    save(arrays, manifest, args.out_dir)

    print(f"pitcher at-bat pitches dropped: {manifest['n_pitcher_at_bat_pitches_dropped']}")
    print(f"pitches: {manifest['n_pitches']}")
    print(f"hitters in vocabulary: {manifest['n_hitters']}")
    print(f"context columns: {len(manifest['context_columns'])}")
    reserved = int((arrays["hitter"] == RESERVED_HITTER_INDEX).sum())
    print(f"pitches routed to the reserved row: {reserved} ({reserved / len(arrays['hitter']):.4%})")
    for dimension in QUALITY_DIMENSIONS:
        labelled = int((arrays[dimension] != MASKED).sum())
        print(f"{dimension} labelled pitches: {labelled}")
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
