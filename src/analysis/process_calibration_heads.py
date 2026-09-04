"""
Phase F.3 -- every factor head scored against real held-out pitches.

The model is a process model: six heads over p(swing), p(contact | swing), the three-class
contact split, and the autoregressive quality chain p(ev), p(la | ev), p(spray | ev, la).
wOBA is not an output of any head. It is composed three steps downstream by
`query.expected_woba`, which solves the count chain and applies a frozen value table.

Until now exactly one head had ever been scored against real pitches (E.6, the swing head).
Every other head was visible only through the composed absorbing rates, where two errors of
opposite sign cancel and neither shows. Two residuals are open and unowned on exactly that
surface: the population-matched strikeout rate fails at -4.78%, and ~22% of the walk gap has
no owner after E.1-E.10. A contact or split head biased in two-strike counts produces that
signature precisely, and no artifact in the repo could currently see it.

This scores each head on the rows the training objective scores it on -- `v1.factor_masks`
is imported rather than reimplemented, so the nesting cannot drift: `la` is scored only
where `ev` is observed, `spray` only where `la` is, and `split` on every contact event,
which is strictly more rows than `ev` because fouls carry no batted-ball measurement. The
conditioned heads are fed OBSERVED bins, which is how training conditions them.

Ensemble convention follows E.6 and `query.expected_woba` and must not drift: PROBABILITIES
are averaged across seeds, never logits.

Diagnostic, not a gate. Nothing here re-opens the two pre-registered claim-1 gates, and
nothing here trains.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.analysis import claim1_eval, provenance
from src.data.model_dataset import MASKED
from src.model import query, query_tables as qt
from src.model import v1

DEFAULT_OUT_DIR = "results/process_calibration"
DEFAULT_BATCH = 32768
# heads scored here; the swing head keeps its own artifact from E.6 and is re-scored
# alongside the others so one table carries all six
BINARY_HEADS = ("swing", "contact")
CATEGORICAL_HEADS = ("split", "ev", "la", "spray")
ARRAY_NAMES = ("context", "hitter", "swing", "contact", "split", "ev", "la", "spray")


def load_season_arrays(data_dir, eval_season, names=ARRAY_NAMES):
    """
    Memmapped per-season subset of the built tensors, plus the manifest.
    Returns (row indices into the full build, {name: array}, manifest).
    Memmapped for the reason E.6 gives: the build is ~1.7GB and one season is wanted.
    """
    directory = Path(data_dir)
    season = np.load(directory / "season.npy", mmap_mode="r")
    rows = np.flatnonzero(np.asarray(season) == eval_season)
    assert len(rows), f"the built dataset carries no {eval_season} pitches"
    arrays = {}
    for name in names:
        path = directory / f"{name}.npy"
        assert path.exists(), f"{data_dir} has no {name}.npy -- wrong tensor build?"
        arrays[name] = np.ascontiguousarray(np.load(path, mmap_mode="r")[rows])
    manifest = json.loads((directory / "manifest.json").read_text())
    return rows, arrays, manifest


def ensemble_head_probabilities(models, arrays, n_bins, batch=DEFAULT_BATCH):
    """
    Mean predicted probability per head, per pitch row, averaged over the ensemble.
    Binary heads return (N,); categorical heads return (N, n_classes).
    Conditioned heads see the OBSERVED ev/la bins, matching how training conditions them.
    """
    n_rows = len(arrays["hitter"])
    out = {name: np.zeros(n_rows, dtype="float64") for name in BINARY_HEADS}
    widths = {"split": len(v1.SPLIT_CLASSES), "ev": n_bins, "la": n_bins, "spray": n_bins}
    for name, width in widths.items():
        out[name] = np.zeros((n_rows, width), dtype="float64")

    with torch.no_grad():
        for start in range(0, n_rows, batch):
            stop = min(start + batch, n_rows)
            size = stop - start
            hitter = torch.from_numpy(arrays["hitter"][start:stop].astype("int64"))
            context = torch.from_numpy(arrays["context"][start:stop])
            ev_bin = torch.from_numpy(arrays["ev"][start:stop].astype("int64"))
            la_bin = torch.from_numpy(arrays["la"][start:stop].astype("int64"))

            totals = {}
            for model in models:
                heads = model(hitter, context, ev_bin, la_bin)
                for name in BINARY_HEADS:
                    probability = torch.sigmoid(heads[name])
                    assert probability.shape == (size,), \
                        f"{name} head returned {tuple(probability.shape)}, expected {(size,)}"
                    totals[name] = probability + totals.get(name, 0.0)
                for name, width in widths.items():
                    if name not in heads:
                        continue
                    probability = torch.softmax(heads[name], dim=1)
                    assert probability.shape == (size, width), \
                        f"{name} head returned {tuple(probability.shape)}, expected {(size, width)}"
                    totals[name] = probability + totals.get(name, 0.0)

            for name, total in totals.items():
                out[name][start:stop] = (total / len(models)).double().numpy()
    return out


def head_masks(arrays):
    """Rows each head is scored on, from `v1.factor_masks` so the nesting cannot drift."""
    labels = {name: torch.from_numpy(arrays[name].astype("int64"))
              for name in ("swing", "contact", "split", "ev", "la", "spray")}
    return {name: mask.numpy() for name, mask in v1.factor_masks(labels).items()}


def binary_table(frame, observed_column, predicted_column, by, label, head):
    """Observed rate against mean predicted rate for a binary head, grouped by `by`."""
    grouped = frame.groupby(list(by), observed=True)
    out = grouped.agg(n_rows=(observed_column, "size"),
                      observed=(observed_column, "mean"),
                      predicted=(predicted_column, "mean")).reset_index()
    out["gap"] = out["predicted"] - out["observed"]
    out["relative_gap"] = out["gap"] / out["observed"].replace(0.0, np.nan)
    out.insert(0, "grouping", label)
    out.insert(0, "head", head)
    return out


def categorical_table(labels, probabilities, class_names, by_values, by_name, head):
    """
    Observed class frequency against mean predicted probability, one row per (cell, class).
    labels: observed class index per row; probabilities: (n_rows, n_classes).
    """
    rows = []
    frame = pd.DataFrame({"cell": by_values, "label": labels})
    for cell, part in frame.groupby("cell", observed=True):
        index = part.index.to_numpy()
        mean_predicted = probabilities[index].mean(axis=0)
        counts = np.bincount(part["label"].to_numpy(), minlength=len(class_names))
        observed = counts / counts.sum()
        for position, name in enumerate(class_names):
            gap = float(mean_predicted[position] - observed[position])
            rows.append({
                "head": head, "grouping": by_name, "cell": str(cell),
                "class": name, "n_rows": int(counts.sum()),
                "observed": float(observed[position]),
                "predicted": float(mean_predicted[position]),
                "gap": gap,
                "relative_gap": gap / observed[position] if observed[position] else np.nan,
            })
    return pd.DataFrame(rows)


def expected_index_table(labels, probabilities, by_values, by_name, head):
    """
    Mean observed bin index against mean predicted bin index, one row per cell.
    A distribution can match on average while every class is wrong, so this is reported
    beside the per-class table, never instead of it.
    """
    positions = np.arange(probabilities.shape[1])
    predicted = probabilities @ positions
    frame = pd.DataFrame({"cell": by_values, "observed": labels, "predicted": predicted})
    out = frame.groupby("cell", observed=True).agg(
        n_rows=("observed", "size"), observed=("observed", "mean"),
        predicted=("predicted", "mean")).reset_index()
    out["gap"] = out["predicted"] - out["observed"]
    out["relative_gap"] = out["gap"] / out["observed"].replace(0.0, np.nan)
    out.insert(0, "grouping", by_name)
    out.insert(0, "head", head)
    out["class"] = "expected_bin_index"
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Phase F.3 -- per-head calibration on real held-out pitches.")
    parser.add_argument("--arm", default="embedding_sgd_sgd_lr1")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--eval-season", type=int, default=2024)
    parser.add_argument("--final-run", action="store_true")
    # phase_d5 is the build D.10 trained on; the quality heads read ev/la/spray, which is
    # the 2026-08-19 Revisit-if condition, so the edges are asserted rather than trusted
    parser.add_argument("--data-dir", default=provenance.CANONICAL_DATA_DIR)
    parser.add_argument("--checkpoint-dir", default="results/checkpoints")
    parser.add_argument("--pitch-events", default="data/processed/pitch_events_labeled.parquet")
    parser.add_argument("--eval-targets", default="data/processed/eval_targets_pa.parquet")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    args = parser.parse_args()

    claim1_eval.assert_not_test_season(args.eval_season, final_run=args.final_run)
    provenance.assert_quality_bins(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, arrays, manifest = load_season_arrays(args.data_dir, args.eval_season)
    n_bins = len(manifest["quality_bin_edges"]["ev"]) + 1
    print(f"pitches scored: {len(rows)}")
    print(f"quality bins: {n_bins}")

    season = np.load(Path(args.data_dir) / "season.npy", mmap_mode="r")
    pitch_frame = qt.align_pitch_frame(args.pitch_events, args.eval_targets,
                                       np.asarray(season))
    frame = pitch_frame.iloc[rows].reset_index(drop=True)
    assert (frame["season"].to_numpy() == args.eval_season).all(), \
        "the pitch frame and the tensor season mask disagree"

    paths = [Path(args.checkpoint_dir) / f"{args.arm}_s{seed}.pt" for seed in args.seeds]
    models = query.load_ensemble(paths, manifest, arrays["context"].shape[1])

    probabilities = ensemble_head_probabilities(models, arrays, n_bins, batch=args.batch)

    # eval-mode / no-grad hygiene: the same rows scored twice must be bit-identical
    head = {name: arrays[name][:4096] for name in ARRAY_NAMES}
    repeat = ensemble_head_probabilities(models, head, n_bins, batch=args.batch)
    for name, values in repeat.items():
        assert np.array_equal(values, probabilities[name][:4096]), \
            f"scoring {name} twice changed the answer -- a model is not in eval mode"
    print("eval-mode hygiene: repeat scoring is bit-identical")

    masks = head_masks(arrays)
    for name, mask in masks.items():
        print(f"rows scored, {name}: {int(mask.sum())}")

    frame["two_strike"] = (frame["strikes"].to_numpy() == 2).astype("int64")
    frame["count"] = (frame["balls"].astype(str) + "-" + frame["strikes"].astype(str))
    frame["handedness"] = frame["stand"].astype(str) + "v" + frame["p_throws"].astype(str)
    frame["all"] = "all"

    # decode-one-batch: readable rows checked against the source parquet's own description
    sample_rows = np.flatnonzero(masks["split"])[:2000]
    sample = frame.iloc[sample_rows].sample(8, random_state=0)[
        ["batter", "balls", "strikes", "stand", "p_throws", "description"]].copy()
    sample["split_label"] = [v1.SPLIT_CLASSES[arrays["split"][i]] for i in sample.index]
    sample["p_in_play"] = probabilities["split"][sample.index,
                                                 v1.SPLIT_CLASSES.index("in_play")]
    print("\n-- decoded sample, split head --")
    print(sample.to_string(index=False))

    groupings = (("all", "overall"), ("count", "count"), ("handedness", "handedness"),
                 ("two_strike", "two_strike"))
    tables = []

    for head_name in BINARY_HEADS:
        mask = masks[head_name]
        part = frame.loc[mask].copy()
        part["observed"] = arrays[head_name][mask]
        part["predicted"] = probabilities[head_name][mask]
        for column, label in groupings:
            table = binary_table(part, "observed", "predicted", (column,), label, head_name)
            table = table.rename(columns={column: "cell"})
            table["cell"] = table["cell"].astype(str)
            table["class"] = "positive"
            tables.append(table)

    class_names = {"split": list(v1.SPLIT_CLASSES),
                   "ev": [f"bin_{i}" for i in range(n_bins)],
                   "la": [f"bin_{i}" for i in range(n_bins)],
                   "spray": [f"bin_{i}" for i in range(n_bins)]}
    for head_name in CATEGORICAL_HEADS:
        if head_name not in masks:
            continue
        mask = masks[head_name]
        labels = arrays[head_name][mask]
        assert (labels != MASKED).all(), f"{head_name} mask admitted a masked label"
        head_probabilities = probabilities[head_name][mask]
        part = frame.loc[mask].reset_index(drop=True)
        for column, label in groupings:
            tables.append(categorical_table(labels, head_probabilities,
                                            class_names[head_name],
                                            part[column].to_numpy(), label, head_name))
            if head_name != "split":
                tables.append(expected_index_table(labels, head_probabilities,
                                                  part[column].to_numpy(), label, head_name))

    table = pd.concat(tables, ignore_index=True)
    table = table[["head", "grouping", "cell", "class", "n_rows", "observed", "predicted",
                   "gap", "relative_gap"]]
    table.to_csv(out_dir / "heads_head_calibration.csv", index=False)

    overall = table[(table["grouping"] == "overall")]
    two_strike = table[(table["grouping"] == "two_strike")]
    summary = {
        "provenance": provenance.stamp(args.data_dir, arm=args.arm, seeds=args.seeds,
                                       eval_season=args.eval_season),
        "n_pitches": int(len(frame)),
        "n_bins": int(n_bins),
        "rows_scored": {name: int(mask.sum()) for name, mask in masks.items()},
        "overall_relative_gap": {
            name: float(overall[(overall["head"] == name)
                                & (overall["class"] == "positive")]["relative_gap"].iloc[0])
            for name in BINARY_HEADS},
        "split_overall": {
            row["class"]: {"observed": row["observed"], "predicted": row["predicted"],
                           "relative_gap": row["relative_gap"]}
            for _, row in overall[overall["head"] == "split"].iterrows()},
        "split_two_strike": {
            f"{row['cell']}_{row['class']}": row["relative_gap"]
            for _, row in two_strike[two_strike["head"] == "split"].iterrows()},
        "max_abs_relative_gap_by_head": {
            name: float(table[(table["head"] == name)
                              & (table["n_rows"] >= 1000)]["relative_gap"].abs().max())
            for name in table["head"].unique()},
    }
    (out_dir / "heads_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n-- overall, every head --")
    print(overall.to_string(index=False))
    print(f"\nwrote {out_dir / 'heads_head_calibration.csv'}")


if __name__ == "__main__":
    main()
