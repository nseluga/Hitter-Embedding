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

    caffeinate -i -s .venv/bin/python -m src.model.sweep --stage d6 --hours 9
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

OUT_DIR = Path("results/phase_d")
LEDGER = OUT_DIR / "sweep_log.csv"
LOG_DIR = OUT_DIR / "logs"
STOP_FLAG = OUT_DIR / "STOP"
# best_val_loss is the arm's OWN objective and is not comparable across arms -- rps,
# meanweight, and invfreq each train against something other than the likelihood.
# reference is the same quantity for every arm (unweighted log loss per scored row)
# and is the only column that may be read down.
LEDGER_FIELDS = ("stage", "config", "seed", "status", "seconds", "best_val_loss",
                 "reference", "best_epoch", "finished_at")

# Arms that exist in code today. The two remaining §7 items -- B.2's flagged five and
# spray as a quality dimension -- need a dataset rebuild and a head change that are not
# written yet, so they are absent here rather than silently skipped.
STAGES = {
    "screen": [("log", []), ("rps", ["--loss-rule", "rps"])],
    "d6": [("baseline", [])],
    "d8": [("baseline", []),
           ("dim16", ["--embedding-dim", "16"]),
           ("dim64", ["--embedding-dim", "64"]),
           ("bilinear", ["--bilinear"]),
           ("meanweight", ["--loss-weighting", "mean"]),
           ("invfreq", ["--contact-inverse-frequency"])],
}
DEFAULT_SEEDS = {"screen": 2, "d6": 5, "d8": 5}
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    new = not LEDGER.exists()
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


def main():
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
    args = parser.parse_args()

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
        append_ledger({"stage": args.stage, "config": name, "seed": seed, "status": status,
                       "seconds": round(seconds, 1), "best_val_loss": best_loss,
                       "reference": reference, "best_epoch": best_epoch,
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
