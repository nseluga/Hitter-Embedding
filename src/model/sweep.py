"""
Overnight driver for the Phase D runs: a queue, a ledger, and a deadline.

Designed around one constraint -- the machine is needed in the morning. So the
deadline does not kill anything. Before each run it asks whether the NEXT run fits
in the time left, using the median of the runs already observed, and exits cleanly
if it does not. You wake up to a finished run, never a half-written one.

Resume across nights is the ledger, not a checkpoint. `sweep_log.csv` records only
COMPLETED runs, so relaunching skips what finished and redoes anything interrupted.
Nothing partial ever enters the results, and no optimizer or RNG state is restored --
a partially restored resume produces a plausible number for a run that is no longer
the run its seed names, which is the failure this design refuses to risk.

Each run is a separate `python -m src.model.train` process. Memory is released
between runs, a crash in one arm cannot take the sweep down, and no state leaks
from one architecture into the next.

macOS sleeps the machine out from under this. Launch it under caffeinate, plugged
in, lid open:

    caffeinate -i -s .venv/bin/python -m src.model.sweep --stage early --hours 9
"""

import argparse
import csv
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from src.analysis import provenance
from src.model.train import LEARNING_RATE

OUT_DIR = Path("results/model_v1")
LEDGER = OUT_DIR / "sweep_log.csv"
LOG_DIR = OUT_DIR / "logs"
STOP_FLAG = OUT_DIR / "STOP"
# best_val_loss is the arm's OWN objective and is not comparable across arms -- rps,
# meanweight, and invfreq each train against something other than the likelihood.
# reference is the same quantity for every arm (unweighted log loss per scored row)
# and is the only column that may be read down.
# lr and warmup_steps were added 2026-08-20 for Phase O. Every pre-O row is backfilled
# with the constants those runs actually used (1e-3, no warmup) rather than left blank:
# blank would read as "unknown", and it is not unknown -- it is the single value the
# knob was pinned at for all 119 of them, which is itself the Phase O finding.
# data_dir was added at the same time and for the same reason `reference` carries the
# warning above: log loss over quality bins is only comparable within one tensor build, and
# until now the build a run used was recoverable only from shell history. The splithead/rebuild rows are
# backfilled to `phase_d5`, established 2026-08-20 by re-running rebuild baseline seed 0 on each
# candidate build: phase_d5 reproduces its epoch-0 train/val (1.05990/1.04681) exactly, and
# the older model_v1 build raises KeyError('split') because it predates the contact split, so
# no --split run could ever have used it.
LEDGER_FIELDS = ("stage", "config", "seed", "status", "seconds", "best_val_loss",
                 "reference", "best_epoch", "lr", "warmup_steps", "data_dir",
                 "finished_at")

# `splithead` is the retrain carrying the three-class contact split (2026-08-08). Every arm moves
# to it together, on purpose: the split head changes the loss, so a presplit `reference` and a splithead
# `reference` are in different units and must never be read down the same column. It is
# also the first stage where §7's last two items exist in code -- B.2's flagged five (a
# dataset rebuilt without the block) and spray as a quality dimension (a head removed) --
# which is why running them before this retrain would have produced an ablation table for
# an architecture that is not the one shipping.
NOBLOCK_DATA_DIR = "data/processed/phase_d_split_noblock"

# `rebuild` is `splithead`'s eight arms on the D5-R17 rebuild: the Statcast placeholder rows are gone from
# the quantile-edge fit AND from all three quality targets, so the 24 bins per dimension are
# different bins. That makes it a new stage rather than a relaunch of `splithead`, for two reasons that
# both bite. The ledger keys on (stage, config, seed) and every `splithead` row is `ok`, so a relaunch
# would silently skip all forty runs. And `reference` is log loss over the quality bins, so a splithead
# and a rebuild `reference` are in different units for exactly the reason the presplit -> splithead note gives.
# `block` gets its own rebuilt no-block dataset: leaving it on the splithead one would train seven arms
# on the new edges and one on the old, turning the block contrast into block + edges.
D10_NOBLOCK_DATA_DIR = "data/processed/phase_d5_noblock"
STAGES = {
    "screen": [("log", []), ("rps", ["--loss-rule", "rps"])],
    "early": [("baseline", [])],
    "presplit": [("baseline", []),
           ("dim16", ["--embedding-dim", "16"]),
           ("dim64", ["--embedding-dim", "64"]),
           ("bilinear", ["--bilinear"]),
           ("meanweight", ["--loss-weighting", "mean"]),
           ("invfreq", ["--contact-inverse-frequency"])],
    "splithead": [("baseline", ["--split"]),
           ("dim16", ["--split", "--embedding-dim", "16"]),
           ("dim64", ["--split", "--embedding-dim", "64"]),
           ("bilinear", ["--split", "--bilinear"]),
           ("meanweight", ["--split", "--loss-weighting", "mean"]),
           ("invfreq", ["--split", "--contact-inverse-frequency"]),
           ("nospray", ["--split", "--no-spray"]),
           ("block", ["--split", "--data-dir", NOBLOCK_DATA_DIR])],
    "rebuild": [("baseline", ["--split"]),
            ("dim16", ["--split", "--embedding-dim", "16"]),
            ("dim64", ["--split", "--embedding-dim", "64"]),
            ("bilinear", ["--split", "--bilinear"]),
            ("meanweight", ["--split", "--loss-weighting", "mean"]),
            ("invfreq", ["--split", "--contact-inverse-frequency"]),
            ("nospray", ["--split", "--no-spray"]),
            ("block", ["--split", "--data-dir", D10_NOBLOCK_DATA_DIR])],
}
# Phase O. A 3x2 factorial on the two knobs that were never varied, on the rebuild baseline
# architecture with everything else frozen. `lr1e3` is the incumbent control and must stay
# in the grid: without it the tuned build has nothing to be tuned RELATIVE TO, and a
# ledger `reference` from a different stage is not comparable (see the presplit->splithead note above).
# Warmup is one epoch of optimizer steps: ceil(5.88M train rows / 8,192) = 719, which is the
# step count every rebuild log reports. Not 718 -- an off-by-one here would be invisible.
O1_WARMUP_STEPS = "719"
# Pinned, not inherited. This module's --data-dir DEFAULT is `model_v1`, but rebuild trained on
# `phase_d5` (different quality-bin edges, different manifest sha), and `reference` is log
# loss over those bins. Inheriting the default would put selection and its own rebuild incumbent in
# different units while every column still lined up, which is the failure mode the
# presplit->splithead and splithead->rebuild notes above exist to prevent. Pin it to the build rebuild shipped on.
O1_DATA_DIR = provenance.CANONICAL_DATA_DIR
O1_BASE = ["--split", "--data-dir", O1_DATA_DIR]
STAGES["selection"] = [
    ("lr3e4", [*O1_BASE, "--lr", "3e-4"]),
    ("lr1e3", [*O1_BASE, "--lr", "1e-3"]),
    ("lr3e3", [*O1_BASE, "--lr", "3e-3"]),
    ("lr3e4_warm", [*O1_BASE, "--lr", "3e-4", "--warmup-steps", O1_WARMUP_STEPS]),
    ("lr1e3_warm", [*O1_BASE, "--lr", "1e-3", "--warmup-steps", O1_WARMUP_STEPS]),
    ("lr3e3_warm", [*O1_BASE, "--lr", "3e-3", "--warmup-steps", O1_WARMUP_STEPS]),
]

DEFAULT_SEEDS = {"screen": 2, "early": 5, "presplit": 5, "splithead": 5, "rebuild": 5, "selection": 2}


# The ledger is keyed and read as text, so `1e-3` and `0.001` are two different values in a
# column that has to be groupable. Everything written to `lr` goes through here, and the 119
# pre-O rows were re-backfilled to match.
def canonical_lr(value):
    return f"{float(value):g}"


QUARANTINED_FLAGS = ("--batch-size", "--weight-decay")


def knobs(extra, default_data_dir):
    """The Phase O knobs this config actually ran at, for the ledger.

    argparse takes the last occurrence, and `launch` puts `extra` after the sweep-level
    --data-dir, so a stage tuple's own --data-dir wins. Resolve it the same way here or the
    ledger records a build the run did not use."""
    lr, warmup, data_dir = canonical_lr(LEARNING_RATE), "0", default_data_dir
    for flag, value in zip(extra, extra[1:]):
        if flag == "--lr":
            lr = canonical_lr(value)
        elif flag == "--warmup-steps":
            warmup = value
        elif flag == "--data-dir":
            data_dir = value
    return lr, warmup, data_dir
FIRST_RUN_ESTIMATE_SECONDS = 45 * 60  # only used before any run has been timed


def read_ledger():
    """Completed (stage, config, seed) triples. Anything else is unfinished work."""
    if not LEDGER.exists():
        return set(), []
    with LEDGER.open() as handle:
        rows = list(csv.DictReader(handle))
    done = {(r["stage"], r["config"], int(r["seed"])) for r in rows if r["status"] == "ok"}
    seconds = [float(r["seconds"]) for r in rows if r["status"] == "ok"]
    return done, seconds


def append_ledger(row):
    """Appending a dict to a CSV trusts the header on disk to match LEDGER_FIELDS. When a
    column is added, an older ledger silently takes the new row's values in the new order
    under the old names -- every column after the insertion point shifts by one and nothing
    raises. Check before writing, not after."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    new = not LEDGER.exists()
    if not new:
        with LEDGER.open() as handle:
            header = tuple(next(csv.reader(handle), ()))
        if header != LEDGER_FIELDS:
            raise ValueError(
                f"{LEDGER} header does not match LEDGER_FIELDS. Appending would shift "
                f"columns silently.\n  on disk: {header}\n  expected: {LEDGER_FIELDS}")
    with LEDGER.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def queue(stage, seeds):
    return [(name, extra, seed) for name, extra in STAGES[stage] for seed in range(seeds)]


def launch(stage, name, extra, seed, args):
    """Run one config as its own process; returns (status, seconds, best loss, epoch)."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{stage}_{name}_s{seed}.log"
    command = [sys.executable, "-m", "src.model.train", "--seed", str(seed),
               "--device", args.device, "--run-name", f"{stage}_{name}",
               "--data-dir", args.data_dir, *args.train_args, *extra]

    started = time.time()
    with log_path.open("w") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                   env={**os.environ, "PYTHONPATH": "."})
    seconds = time.time() - started

    best_loss, best_epoch, reference = "", "", ""
    for line in log_path.read_text().splitlines():
        if line.startswith("best val loss"):
            fields = line.split()
            best_loss, best_epoch, reference = (fields[3], fields[6].rstrip(";"),
                                               fields[8].rstrip(";"))
    status = "ok" if completed.returncode == 0 else f"exit{completed.returncode}"
    return status, seconds, best_loss, best_epoch, reference


def main_argv(argv=None):
    """Argument parsing and the quarantine check, split out so both are testable without
    launching a training run."""
    parser = argparse.ArgumentParser(description="Run a Phase D stage overnight.")
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--hours", type=float, default=9.0,
                        help="stop starting runs once the next one would not fit")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--data-dir", default="data/processed/phase_d")
    parser.add_argument("--train-args", nargs=argparse.REMAINDER, default=[],
                        help="everything after this is passed straight to train.py")
    parser.add_argument("--dry-run", action="store_true", help="print the queue and exit")
    args = parser.parse_args(argv)

    # Phase O quarantines batch size and weight decay: they are one setting (the
    # decay-to-gradient ratio), the ratio is 23.9:1 at the 10th exposure percentile, and
    # moving either mid-phase would change what every earlier selection arm's `reference` means.
    # train.py still exposes --batch-size, and --train-args is forwarded verbatim, so the
    # quarantine is only real if it is enforced here. Batch size also silently rescales
    # `warmup_for`'s per-epoch cap, so a change would move a knob Phase O IS tuning.
    smuggled = [f for f in QUARANTINED_FLAGS if f in args.train_args]
    if smuggled and args.stage == "selection":
        parser.error(f"{', '.join(smuggled)} is quarantined for Phase O and is not recorded "
                     f"in the ledger. Unpin it in train.py and open a new stage instead.")
    return args


def main(argv=None):
    args = main_argv(argv)
    seeds = args.seeds or DEFAULT_SEEDS[args.stage]
    done, history = read_ledger()
    pending = [item for item in queue(args.stage, seeds)
               if (args.stage, item[0], item[2]) not in done]

    print(f"stage {args.stage}: {len(pending)} runs pending, {len(done)} already done")
    if args.dry_run or not pending:
        for name, _, seed in pending:
            print(f"  {name} seed {seed}")
        return

    STOP_FLAG.unlink(missing_ok=True)
    deadline = time.time() + args.hours * 3600
    print(f"deadline {datetime.fromtimestamp(deadline):%H:%M}; "
          f"touch {STOP_FLAG} to stop after the current run")

    for name, extra, seed in pending:
        if STOP_FLAG.exists():
            print("stop flag set; exiting")
            break
        # the next run has to FIT, not merely start: a run begun at hour 8:55 would
        # still be going at breakfast, which is the thing this exists to prevent
        expected = statistics.median(history) if history else FIRST_RUN_ESTIMATE_SECONDS
        if time.time() + expected > deadline:
            print(f"{(deadline - time.time()) / 60:.0f} min left, next run needs "
                  f"~{expected / 60:.0f}; stopping here")
            break

        print(f"[{datetime.now():%H:%M}] {name} seed {seed} ...", flush=True)
        status, seconds, best_loss, best_epoch, reference = launch(
            args.stage, name, extra, seed, args)
        lr, warmup, data_dir = knobs([*args.train_args, *extra], args.data_dir)
        append_ledger({"stage": args.stage, "config": name, "seed": seed, "status": status,
                       "seconds": round(seconds, 1), "best_val_loss": best_loss,
                       "reference": reference, "best_epoch": best_epoch,
                       "lr": lr, "warmup_steps": warmup, "data_dir": data_dir,
                       "finished_at": f"{datetime.now():%Y-%m-%d %H:%M}"})
        print(f"    {status} in {seconds / 60:.1f} min, ref {reference or '-'}")
        if status == "ok":
            history.append(seconds)
        else:
            print(f"    see {LOG_DIR / f'{args.stage}_{name}_s{seed}.log'}")

    remaining = len([i for i in pending if (args.stage, i[0], i[2]) not in read_ledger()[0]])
    print(f"done for now; {remaining} runs still pending — rerun the same command")


if __name__ == "__main__":
    main()
