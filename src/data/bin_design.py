"""
Step 4 of the D.5 remediation: measure three candidate binnings and pick on the measurement.

The three schemes are `model_dataset.BIN_SCHEMES`. The objective is within-cell realized-wOBA
variance across the JOINT 24^3 grid, not any marginal: the model predicts into a cell of that
grid, so choosing exit-velocity edges against wOBA averaged over launch angle optimizes a
quantity the model never uses (D5-R7).

The target is REALIZED wOBA on train seasons only. Statcast's `estimated_woba_using_speedangle`
would be circular, because the outcome table `V` is what estimates it.

Two things this reports besides the objective:

* `top_ev_bin_span` -- the defect that motivated the step. The shipped equal-mass top bin runs
  15.2 mph against neighbours at 1.8 and 2.3, holds 4.79% of balls in play and carries 11.64% of
  in-play wOBA points, so every ball in it is valued at a mean the 107-110 mph crowd sets.
* `min_cell_count` -- the PRE-LAUNCH BLOCKER. `query_tables.fit_outcome_table` backs a joint cell
  off to its (ev, la) marginal. If the winning edges push top joint cells below the regime that
  backoff was fitted in, it must be re-fit before the overnight rather than during it.

    python -m src.data.bin_design

Writes results/phase_d/d5_bin_design.json. Chooses nothing on its own -- the winner is recorded
in the decision log by hand, because freezing the edges is a decision and not an argmax.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import splits
from src.data import model_dataset as md
from src.model import query_tables as qt

TARGET_COLUMNS = qt.JOIN_KEYS + ["woba_points", "in_denominator"]


def per_row_target(pitch_df, pa_df):
    """
    Realized wOBA points per pitch row, joined from the plate appearance the pitch belongs to.

    From the EVAL-TARGET table rather than the modeling table, per the two-table principle
    (decision log 2026-07-15): the modeling table is deliberately filtered, and every evaluation
    quantity is built from the complete outcome record. NaN where the pitch's plate appearance is
    outside the wOBA denominator, which `fit_bin_edges` and `joint_cell_variance` both drop.
    """
    outcome = (pa_df[TARGET_COLUMNS].drop_duplicates(qt.JOIN_KEYS)
               .set_index(qt.JOIN_KEYS))
    assert outcome.index.is_unique, "a plate appearance key appears twice in the PA table"
    points = outcome["woba_points"].where(outcome["in_denominator"])
    keys = pd.MultiIndex.from_frame(pitch_df[qt.JOIN_KEYS])
    return points.reindex(keys).to_numpy(dtype="float64")


def measure(pitch_df, train_seasons, target, n_bins):
    """One record per scheme in BIN_SCHEMES: its edges plus the joint-grid diagnostics."""
    results = {}
    for scheme in md.BIN_SCHEMES:
        edges = md.fit_bin_edges(pitch_df, train_seasons, n_bins=n_bins, scheme=scheme,
                                 target=target, drop_placeholders=True)
        stats = md.joint_cell_variance(pitch_df, train_seasons, edges, target, n_bins=n_bins)
        results[scheme] = {"edges": edges, **stats}
        print(f"{scheme:18s} within-cell var {stats['within_cell_variance']:.6f}  "
              f"explained {stats['variance_explained']:.4f}  "
              f"top EV span {stats['top_ev_bin_span']:6.2f} mph  "
              f"min cell {stats['min_cell_count']:>5}  "
              f"occupied {stats['occupied_cells']:>6}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Measure three candidate quality binnings.")
    parser.add_argument("--pitch-events", default="data/processed/pitch_events_labeled.parquet")
    parser.add_argument("--eval-targets", default="data/processed/eval_targets_pa.parquet")
    parser.add_argument("--out-dir", default="results/phase_d")
    parser.add_argument("--n-bins", type=int, default=md.DEFAULT_QUALITY_BINS)
    parser.add_argument("--train-seasons", type=int, nargs="+", default=None,
                        help="defaults to the frozen split's train seasons")
    args = parser.parse_args()

    columns = (list(md.QUALITY_DIMENSIONS) + ["season", "launch_angle", "launch_speed"]
               + qt.JOIN_KEYS)
    pitch_df = pd.read_parquet(args.pitch_events, columns=columns)
    pa_df = pd.read_parquet(args.eval_targets, columns=TARGET_COLUMNS)

    train_seasons = args.train_seasons or list(splits.load_splits()["split"]["train"])
    print(f"train seasons: {train_seasons}")

    # the guard belongs here rather than inside fit_bin_edges: it is a statement about the REAL
    # table, and the unit tests build synthetic frames that legitimately carry no placeholders
    share = md.assert_placeholders_present(pitch_df)
    placeholders = int(md.placeholder_mask(pitch_df).sum())
    print(f"placeholder rows dropped from the edge fit: {placeholders} "
          f"({share:.4%} of balls in play)")

    target = per_row_target(pitch_df, pa_df)
    print(f"rows with a realized-wOBA target: {int(np.isfinite(target).sum())} "
          f"of {len(pitch_df)}")

    results = measure(pitch_df, train_seasons, target, args.n_bins)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "d5_bin_design.json"
    out_path.write_text(json.dumps({"n_bins": args.n_bins,
                                    "train_seasons": train_seasons,
                                    "placeholder_rows": placeholders,
                                    "placeholder_share_of_in_play": share,
                                    "schemes": results}, indent=2))
    print(f"\nwrote {out_path}")

    # `min_cell_count` cannot carry the blocker: it is 1 under the SHIPPED equal-mass edges too,
    # so a test against an absolute floor fires on the status quo and discriminates nothing. What
    # the backoff actually cares about is how much in-play MASS sits in cells too thin to fit, so
    # the blocker is that share measured RELATIVE to the shipped binning.
    reference = results["equal_mass"]["mass_share_below_backoff"]
    print(f"\nbackoff exposure (share of in-play mass in cells below {md.BACKOFF_MIN_CELL})")
    for name, record in results.items():
        delta = record["mass_share_below_backoff"] - reference
        flag = "  <- BLOCKER if this wins" if delta > 0 else ""
        print(f"  {name:18s} {record['mass_share_below_backoff']:.4%}  "
              f"cells {record['cells_below_backoff']:>6}  "
              f"vs shipped equal-mass {delta:+.4%}{flag}")


if __name__ == "__main__":
    main()
