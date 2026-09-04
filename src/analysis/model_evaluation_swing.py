"""
Phase E.6 — swing-head calibration on real pitches (docs/phase-e-spec.md §8).

Two suspects own the composition's walk excess and D.5 ruled out neither:

  1. the SWING HEAD predicts too few swings, so too much mass reaches ball four;
  2. the PITCH RESAMPLER (`query._sample_grid`) draws a more out-of-zone pitch mix than
     real pitching, so the head is right and the pitches it is asked about are wrong.

They separate by removing the resampler from the measurement entirely: score REAL held-out
pitch rows and compare the ensemble's mean p(swing) against the observed swing rate on the
same rows. No repertoire draw, no (pitcher, count) cell, no backoff.

The ensemble convention is the one `query.expected_woba` uses and must not drift: swing
PROBABILITIES are averaged across seeds, never logits. Runs under `model.eval()` (set by
`query.load_ensemble`) and `torch.no_grad()`.

Read-only with respect to the model. Nothing trains.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.analysis import claim1_eval
from src.model import query, query_tables as qt
from src.model.v1 import HitterEmbeddingV1

DEFAULT_OUT_DIR = "results/model_evaluation"
DEFAULT_BATCH = 65536


def load_season_rows(data_dir, eval_season):
    """
    Context, hitter and observed-swing arrays for one season, memmapped and subset.

    Memmapped deliberately: the built table is ~1.7 GB and only one season of it is wanted,
    so a resident load would hold nine seasons of context to read one. The gather is a
    single fancy-index into the memmap, which is the access pattern memmapping is good at,
    unlike the shuffled per-batch gather `loader` warns about.
    """
    directory = Path(data_dir)
    season = np.load(directory / "season.npy", mmap_mode="r")
    rows = np.flatnonzero(np.asarray(season) == eval_season)
    assert len(rows), f"the built dataset carries no {eval_season} pitches"
    out = {}
    for name in ("context", "hitter", "swing"):
        array = np.load(directory / f"{name}.npy", mmap_mode="r")
        out[name] = np.ascontiguousarray(array[rows])
    manifest = json.loads((directory / "manifest.json").read_text())
    return rows, out, manifest


def ensemble_swing(models, hitter, context, batch=DEFAULT_BATCH):
    """
    Mean p(swing) over the ensemble, per pitch row. Probabilities averaged, not logits --
    the deep-ensemble mixture for a Bernoulli, and the same convention as `expected_woba`.
    """
    out = np.zeros(len(hitter), dtype="float64")
    with torch.no_grad():
        for start in range(0, len(hitter), batch):
            stop = min(start + batch, len(hitter))
            rows_h = torch.from_numpy(hitter[start:stop].astype("int64"))
            rows_c = torch.from_numpy(context[start:stop])
            total = None
            for model in models:
                p = torch.sigmoid(model.head_swing(query._trunk(model, rows_h, rows_c))
                                  .squeeze(-1))
                assert p.shape == (stop - start,), \
                    f"swing head returned {tuple(p.shape)}, expected {(stop - start,)}"
                total = p if total is None else total + p
            out[start:stop] = (total / len(models)).double().numpy()
    return out


def calibration_table(frame, by, label):
    """Observed swing rate against mean predicted, grouped by `by`. One row per cell."""
    grouped = frame.groupby(list(by), observed=True)
    out = grouped.agg(n_pitches=("swing", "size"), observed=("swing", "mean"),
                      predicted=("p_swing", "mean")).reset_index()
    out["gap"] = out["predicted"] - out["observed"]
    out["relative_gap"] = out["gap"] / out["observed"]
    out.insert(0, "grouping", label)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Phase E.6 — swing-head calibration on real held-out pitches.")
    parser.add_argument("--arm", default="embedding_sgd_sgd_lr1")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--eval-season", type=int, default=2024)
    parser.add_argument("--final-run", action="store_true")
    parser.add_argument("--data-dir", default="data/processed/phase_d5")
    parser.add_argument("--checkpoint-dir", default="results/checkpoints")
    parser.add_argument("--pitch-events", default="data/processed/pitch_events_labeled.parquet")
    parser.add_argument("--eval-targets", default="data/processed/eval_targets_pa.parquet")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    args = parser.parse_args()

    claim1_eval.assert_not_test_season(args.eval_season, final_run=args.final_run)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, arrays, manifest = load_season_rows(args.data_dir, args.eval_season)
    print(f"{len(rows)} pitches in {args.eval_season}")

    season = np.load(Path(args.data_dir) / "season.npy", mmap_mode="r")
    pitch_frame = qt.align_pitch_frame(args.pitch_events, args.eval_targets,
                                       np.asarray(season))
    frame = pitch_frame.iloc[rows].reset_index(drop=True)
    assert (frame["season"].to_numpy() == args.eval_season).all(), \
        "the pitch frame and the tensor season mask disagree"

    paths = [Path(args.checkpoint_dir) / f"{args.arm}_s{seed}.pt" for seed in args.seeds]
    models = query.load_ensemble(paths, manifest, arrays["context"].shape[1])

    p_swing = ensemble_swing(models, arrays["hitter"], arrays["context"], batch=args.batch)
    # eval-mode / no-grad hygiene: the same rows scored twice must be bit-identical
    check = ensemble_swing(models, arrays["hitter"][:4096], arrays["context"][:4096],
                           batch=args.batch)
    assert np.array_equal(check, p_swing[:4096]), \
        "scoring the same rows twice changed the answer -- a model is not in eval mode"

    frame["p_swing"] = p_swing
    frame["swing"] = arrays["swing"]
    frame["in_zone"] = (frame["zone"].to_numpy() <= 9).astype("int64")
    frame["trained"] = frame["batter"].isin(
        {int(b) for b, row in manifest["vocabulary"].items() if int(row) != 0})
    # the 12 non-terminal counts; a 3-2 foul stays at 3-2, so nothing else can appear
    frame = frame[frame["balls"].between(0, 3) & frame["strikes"].between(0, 2)]

    # decode-one-batch: a handful of scored rows in readable form, checked against the
    # source parquet's own description string rather than against an internal label
    sample = frame.sample(8, random_state=0)[
        ["batter", "balls", "strikes", "stand", "p_throws", "zone", "plate_x", "plate_z",
         "description", "swing", "p_swing"]]
    print("\n-- decoded sample --")
    print(sample.to_string(index=False))

    tables = [
        calibration_table(frame, ("balls", "strikes"), "count"),
        calibration_table(frame, ("stand", "p_throws"), "handedness"),
        calibration_table(frame, ("in_zone",), "zone"),
        calibration_table(frame, ("balls", "strikes", "in_zone"), "count_x_zone"),
        calibration_table(frame.assign(all="all"), ("all",), "overall"),
        calibration_table(frame[frame["trained"]].assign(all="trained"), ("all",),
                          "trained_hitters"),
    ]
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(out_dir / "swing_calibration_swing_calibration.csv", index=False)

    overall = table[table["grouping"] == "overall"].iloc[0]
    three_ball = frame[frame["balls"] == 3]
    other = frame[frame["balls"] < 3]
    summary = {
        "arm": args.arm, "seeds": args.seeds, "eval_season": args.eval_season,
        "n_pitches": int(len(frame)),
        "overall_observed": float(overall["observed"]),
        "overall_predicted": float(overall["predicted"]),
        "overall_relative_gap": float(overall["relative_gap"]),
        "three_ball_observed": float(three_ball["swing"].mean()),
        "three_ball_predicted": float(three_ball["p_swing"].mean()),
        "three_ball_relative_gap": float(three_ball["p_swing"].mean()
                                         / three_ball["swing"].mean() - 1.0),
        "under_three_ball_relative_gap": float(other["p_swing"].mean()
                                               / other["swing"].mean() - 1.0),
        "max_abs_count_relative_gap": float(
            table[table["grouping"] == "count"]["relative_gap"].abs().max()),
    }
    (out_dir / "swing_calibration_swing_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n-- swing calibration by count (predicted minus observed) --")
    print(table[table["grouping"] == "count"].to_string(index=False))
    print("\n-- by zone and handedness --")
    print(table[table["grouping"].isin(("zone", "handedness", "overall",
                                        "trained_hitters"))].to_string(index=False))
    print(f"\nwrote {out_dir / 'swing_calibration_swing_calibration.csv'}")


if __name__ == "__main__":
    main()
