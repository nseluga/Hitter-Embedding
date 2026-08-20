"""
Phase F.4 -- the process scored at the pitch level, against references that carry no hitter.

Every gate in this project collapses the model to one scalar per (batter, hand): a composed
wOBA. Per-seed validation loss exists but only ever compared seeds and arms to each other,
so nothing in the repo answered "is this a good model of a pitch", and nothing isolated what
the hitter embedding buys at the level the model actually predicts.

Two references, both carrying no hitter identity, because they answer different questions:

  COLD START -- the same trained network with the hitter index forced to the reserved
  zero row. The architecture, the context tower and every head are held fixed and only
  identity is removed, so the gap is exactly what knowing WHICH hitter is worth. This is
  the honest measure of the embedding's contribution.

  FREQUENCY TABLE -- observed rates by (balls, strikes, stand, p_throws), fit on the
  TRAIN seasons named in the build manifest and applied to the eval season. No network at
  all. This is the "is the model any good" reference. Fit on train only: fitting it on the
  eval season would let the reference see the answer key and beat the model for the wrong
  reason.

Scoring is mean negative log-likelihood per row, on the rows the training objective scores
each head on (`v1.factor_masks`, imported, not reimplemented). Lower is better. The ensemble
convention follows E.6: PROBABILITIES averaged across seeds, never logits, which is the
mixture a deep ensemble actually defines.

This is a diagnostic and NOT a claim-1 baseline. The frequency table is not a ladder rung,
does not enter the C ladder, and neither reference re-opens the two pre-registered gates.
Nothing here trains.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.analysis import claim1_eval, provenance
from src.analysis.f3_heads import (BINARY_HEADS, CATEGORICAL_HEADS, ARRAY_NAMES,
                                   head_masks, load_season_arrays)
from src.data.model_dataset import MASKED, RESERVED_HITTER_INDEX
from src.model import query, query_tables as qt
from src.model import v1

DEFAULT_OUT_DIR = "results/phase_f"
DEFAULT_BATCH = 32768
# the reference's conditioning set: everything a forecaster knows except who is batting
REFERENCE_KEYS = ("balls", "strikes", "stand", "p_throws")
# keeps a zero-count cell from returning an infinite loss; 1e-9 would dominate the mean
LAPLACE_ALPHA = 1.0


def ensemble_log_likelihood(models, arrays, n_bins, masks, cold_start=False,
                            batch=DEFAULT_BATCH):
    """
    Per-row log-likelihood of the OBSERVED outcome under the ensemble, per head.
    cold_start: force every hitter index to the reserved zero row, removing identity.
    Returns {head: array of log p(observed), NaN off that head's mask}.
    """
    n_rows = len(arrays["hitter"])
    out = {name: np.full(n_rows, np.nan, dtype="float64")
           for name in BINARY_HEADS + CATEGORICAL_HEADS}
    with torch.no_grad():
        for start in range(0, n_rows, batch):
            stop = min(start + batch, n_rows)
            size = stop - start
            index = arrays["hitter"][start:stop].astype("int64")
            if cold_start:
                index = np.full_like(index, RESERVED_HITTER_INDEX)
            hitter = torch.from_numpy(index)
            context = torch.from_numpy(arrays["context"][start:stop])
            ev_bin = torch.from_numpy(arrays["ev"][start:stop].astype("int64"))
            la_bin = torch.from_numpy(arrays["la"][start:stop].astype("int64"))

            totals = {}
            for model in models:
                heads = model(hitter, context, ev_bin, la_bin)
                for name in BINARY_HEADS:
                    probability = torch.sigmoid(heads[name])
                    assert probability.shape == (size,), \
                        f"{name} head returned {tuple(probability.shape)}"
                    totals[name] = probability + totals.get(name, 0.0)
                for name in CATEGORICAL_HEADS:
                    if name not in heads:
                        continue
                    probability = torch.softmax(heads[name], dim=1)
                    totals[name] = probability + totals.get(name, 0.0)

            for name, total in totals.items():
                mean = (total / len(models)).double().numpy()
                observed = arrays[name][start:stop].astype("int64")
                mask = masks[name][start:stop]
                if name in BINARY_HEADS:
                    probability = np.where(observed == 1, mean, 1.0 - mean)
                else:
                    safe = np.clip(observed, 0, mean.shape[1] - 1)
                    probability = mean[np.arange(size), safe]
                row = np.where(mask, np.log(np.clip(probability, 1e-12, 1.0)), np.nan)
                out[name][start:stop] = row
    return out


def frequency_reference(frame, arrays, masks, n_bins, train_rows, eval_rows):
    """
    Log-likelihood under observed rates by count and handedness, fit on TRAIN rows only.
    Unseen cells fall back to the head's train-season marginal, which is the same estimator
    with an empty conditioning set rather than a different model.
    """
    widths = {"split": len(v1.SPLIT_CLASSES), "ev": n_bins, "la": n_bins, "spray": n_bins}
    cells = frame[list(REFERENCE_KEYS)].astype(str).agg("|".join, axis=1).to_numpy()
    out = {}
    for name in BINARY_HEADS + CATEGORICAL_HEADS:
        mask = masks[name]
        width = widths.get(name, 2)
        fit = train_rows & mask
        assert fit.sum(), f"no train rows to fit the {name} reference"
        labels = arrays[name].astype("int64")
        table = {}
        marginal = (np.bincount(labels[fit], minlength=width) + LAPLACE_ALPHA)
        marginal = marginal / marginal.sum()
        for cell in np.unique(cells[fit]):
            rows = fit & (cells == cell)
            counts = np.bincount(labels[rows], minlength=width) + LAPLACE_ALPHA
            table[cell] = counts / counts.sum()
        scored = eval_rows & mask
        probability = np.array([table.get(cells[i], marginal)[labels[i]]
                                for i in np.flatnonzero(scored)])
        row = np.full(len(labels), np.nan)
        row[np.flatnonzero(scored)] = np.log(np.clip(probability, 1e-12, 1.0))
        out[name] = row
    return out


def summarize(name, model_ll, cold_ll, reference_ll, mask):
    """One row: mean NLL per scored pitch for the model and both references."""
    selected = mask & ~np.isnan(model_ll) & ~np.isnan(reference_ll)
    model_nll = float(-np.nanmean(model_ll[selected]))
    cold_nll = float(-np.nanmean(cold_ll[selected]))
    reference_nll = float(-np.nanmean(reference_ll[selected]))
    return {
        "head": name,
        "n_rows": int(selected.sum()),
        "model_nll": model_nll,
        "cold_start_nll": cold_nll,
        "frequency_reference_nll": reference_nll,
        "identity_gain": cold_nll - model_nll,
        "gain_vs_reference": reference_nll - model_nll,
        "identity_share_of_gain": ((cold_nll - model_nll) / (reference_nll - model_nll)
                                   if reference_nll > model_nll else np.nan),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Phase F.4 -- pitch-level process scoring against no-identity references.")
    parser.add_argument("--arm", default="d10_baseline")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--eval-season", type=int, default=2024)
    parser.add_argument("--final-run", action="store_true")
    parser.add_argument("--data-dir", default=provenance.CANONICAL_DATA_DIR)
    parser.add_argument("--checkpoint-dir", default="results/checkpoints")
    parser.add_argument("--pitch-events", default="data/processed/pitch_events_labeled.parquet")
    parser.add_argument("--eval-targets", default="data/processed/eval_targets_pa.parquet")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--reference-seasons", type=int, nargs="*", default=None)
    args = parser.parse_args()

    claim1_eval.assert_not_test_season(args.eval_season, final_run=args.final_run)
    provenance.assert_quality_bins(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    directory = Path(args.data_dir)
    manifest = json.loads((directory / "manifest.json").read_text())
    train_seasons = args.reference_seasons or manifest["train_seasons"]
    assert args.eval_season not in train_seasons, \
        "the reference would be fit on the season it is scored on"
    print(f"reference fit on train seasons: {train_seasons}")

    # both the eval season and the reference seasons are needed, so the whole build is
    # subset once on the union rather than twice
    season_all = np.asarray(np.load(directory / "season.npy", mmap_mode="r"))
    wanted = np.isin(season_all, list(train_seasons) + [args.eval_season])
    keep = np.flatnonzero(wanted)
    arrays = {}
    for name in ARRAY_NAMES:
        arrays[name] = np.ascontiguousarray(
            np.load(directory / f"{name}.npy", mmap_mode="r")[keep])
    season = season_all[keep]
    train_rows = np.isin(season, train_seasons)
    eval_rows = season == args.eval_season
    # split-boundary assertion: no row is in both, and the reference never sees eval
    assert not (train_rows & eval_rows).any(), "a row is in both the reference fit and the eval"
    print(f"reference fit rows: {int(train_rows.sum())}, eval rows: {int(eval_rows.sum())}")

    n_bins = len(manifest["quality_bin_edges"]["ev"]) + 1
    pitch_frame = qt.align_pitch_frame(args.pitch_events, args.eval_targets, season_all)
    frame = pitch_frame.iloc[keep].reset_index(drop=True)

    paths = [Path(args.checkpoint_dir) / f"{args.arm}_s{seed}.pt" for seed in args.seeds]
    models = query.load_ensemble(paths, manifest, arrays["context"].shape[1])
    masks = head_masks(arrays)

    scored = {name: mask & eval_rows for name, mask in masks.items()}
    model_ll = ensemble_log_likelihood(models, arrays, n_bins, scored, cold_start=False,
                                       batch=args.batch)
    cold_ll = ensemble_log_likelihood(models, arrays, n_bins, scored, cold_start=True,
                                      batch=args.batch)
    reference_ll = frequency_reference(frame, arrays, masks, n_bins, train_rows, eval_rows)

    rows = [summarize(name, model_ll[name], cold_ll[name], reference_ll[name], scored[name])
            for name in BINARY_HEADS + CATEGORICAL_HEADS if name in scored]
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "f4_process_nll.csv", index=False)

    cold_share = float(np.mean(arrays["hitter"][eval_rows] == RESERVED_HITTER_INDEX))
    summary = {
        "provenance": provenance.stamp(args.data_dir, arm=args.arm, seeds=args.seeds,
                                       eval_season=args.eval_season),
        "reference_seasons": list(train_seasons),
        "reference_conditioning": list(REFERENCE_KEYS),
        "laplace_alpha": LAPLACE_ALPHA,
        "cold_start_share_of_eval_pitches": cold_share,
        "by_head": {row["head"]: row for row in rows},
        "note": ("diagnostic only -- the frequency reference is not a claim-1 ladder rung "
                 "and neither reference re-opens the pre-registered gates"),
    }
    (out_dir / "f4_process_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    print("\n-- mean negative log-likelihood per scored pitch, lower is better --")
    print(table.to_string(index=False))
    print(f"\nwrote {out_dir / 'f4_process_nll.csv'}")


if __name__ == "__main__":
    main()
