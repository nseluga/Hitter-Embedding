"""
Blocking gate for the D.5 cleaning pass: widening `clean.RETAIN_COLUMNS` must be ADDITIVE.

`load_snapshot` reads `wanted & available`, so adding a column name should only append a
column. Should is not is. If the widening changed a filter's behaviour -- a new column with
nulls tripping `validate_core_context`, a dtype change altering `deduplicate`'s notion of a
duplicate, a sort key resolving ties differently -- then every downstream artefact built on
the old table is silently invalid, and the bins, the tensors and the outcome table all sit
downstream.

The assertion is bit-identity on the columns that already existed, not row counts. Two tables
can carry the same number of rows in a different order and produce different quantile edges.

This lives in the repository rather than in a scratch directory because the plan calls it
blocking, and a gate that a `/tmp` wipe can delete is not a gate. It is a script rather than a
pytest because it needs the 320 MB real snapshot; the unit tests cover the pipeline's logic.

    python -m src.data.verify_clean_additivity

Exit 0 additive, 1 not additive, 2 could not run the comparison.
"""

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd

from src.data import clean


def compare(baseline, rebuilt):
    """
    Returns (ok, findings, added). `baseline` is the table on disk, `rebuilt` the fresh pass.
    Findings are strings; an empty list is the pass condition.
    """
    findings = []
    added = [c for c in rebuilt.columns if c not in baseline.columns]

    missing = [c for c in baseline.columns if c not in rebuilt.columns]
    if missing:
        findings.append(f"columns present before and gone now: {missing}")

    if len(baseline) != len(rebuilt):
        findings.append(f"row count moved: {len(baseline)} -> {len(rebuilt)}")
        # every per-column comparison below is meaningless once the lengths differ
        return False, findings, added

    shared = [c for c in baseline.columns if c in rebuilt.columns]
    for column in shared:
        left, right = baseline[column], rebuilt[column]
        if left.dtype != right.dtype:
            findings.append(f"{column}: dtype {left.dtype} -> {right.dtype}")
            continue
        # `equals` treats NaN as equal to NaN, which is what bit-identity means for a column
        # that legitimately carries nulls. `==` would report every null row as a difference.
        if not left.reset_index(drop=True).equals(right.reset_index(drop=True)):
            differing = int((left.to_numpy() != right.to_numpy()).sum())
            findings.append(f"{column}: {differing} of {len(left)} values differ")

    return not findings, findings, added


def _self_check():
    """The smallest thing that fails if `compare` stops detecting a disturbance."""
    base = pd.DataFrame({"a": [1, 2, 3], "b": [0.5, float("nan"), 1.5]})
    assert compare(base, base.assign(c=[7, 8, 9]))[0], "an appended column is additive"
    assert compare(base, base.copy())[0], "NaN must compare equal to NaN"
    assert not compare(base, base.iloc[::-1])[0], "a reordering is not additive"
    assert not compare(base, base.assign(a=[1, 2, 4]))[0], "a changed value is not additive"
    assert not compare(base, base.drop(columns=["b"]))[0], "a dropped column is not additive"
    assert not compare(base, base.iloc[:2])[0], "a row-count change is not additive"
    print("self-check passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--snapshot-dir", default="data/raw/statcast/snapshot_2026-07-14")
    parser.add_argument("--baseline", default="data/processed/pitch_events.parquet",
                        help="the table the widening must not have disturbed")
    parser.add_argument("--seasons", type=int, nargs="+", default=None,
                        help="restrict both sides; omit for the full table")
    parser.add_argument("--self-check", action="store_true",
                        help="exercise `compare` on synthetic frames and exit")
    args = parser.parse_args()

    if args.self_check:
        _self_check()
        return 0

    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        print(f"no baseline at {baseline_path} -- nothing to compare against", file=sys.stderr)
        return 2

    baseline = pd.read_parquet(baseline_path)
    print(f"baseline: {len(baseline)} rows, {len(baseline.columns)} columns "
          f"from {baseline_path}")

    rebuilt, report = clean.clean(args.snapshot_dir, args.seasons)
    print(f"rebuilt:  {len(rebuilt)} rows, {len(rebuilt.columns)} columns")

    # written to a temp path deliberately. Overwriting the baseline is how a gate that is
    # supposed to protect the downstream artefacts destroys the thing it was comparing to.
    with tempfile.TemporaryDirectory() as scratch:
        out = Path(scratch) / "pitch_events.parquet"
        rebuilt.to_parquet(out, index=False)
        print(f"round-tripped through {out} ({out.stat().st_size / 1e6:.1f} MB)")
        rebuilt = pd.read_parquet(out)

    ok, findings, added = compare(baseline, rebuilt)

    print(f"\ncolumns added by the widening: {added or 'none'}")
    for stage, rows in report:
        print(f"  {stage:26s} {rows:>9}")

    if ok:
        print("\nADDITIVE: every pre-existing column is bit-identical and row order is unchanged")
        return 0
    print("\nNOT ADDITIVE -- the widening changed the table:", file=sys.stderr)
    for finding in findings:
        print(f"  {finding}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
