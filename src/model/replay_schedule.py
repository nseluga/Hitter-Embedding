"""
Recover a finished run's stopping point and lr schedule, in optimizer steps.

The final refit has no validation season, so it cannot rediscover where the frozen-split
runs stopped or when ReduceLROnPlateau cut the rate -- it has to reproduce them. Both are
already on record: `results/model_v1/logs/<arm>_s<seed>.log` prints the lr every epoch and
the step count of every epoch, so a cut is visible as an epoch where the printed rate
changed. Nothing new has to be logged and no ledger column has to grow (2026-09-04
decision log's `unverified` item, resolved here).

TWO THINGS ARE EASY TO GET WRONG AND BOTH SHIFT THE BUDGET BY A WHOLE EPOCH.

1. THE OFF-BY-ONE. `best val loss ... at epoch 23` is 0-indexed and the checkpoint was
   written at the END of that epoch, so reproducing it takes 24 epochs of steps, not 23.
   The decision log's "best epoch x steps per epoch" understates the budget by one epoch.

2. WHICH CUTS COUNT. The printed rate on epoch e is the rate AFTER that epoch's scheduler
   step, so a change between epoch e-1 and e means the cut fired at the end of e-1 -- after
   epoch e-1's steps. A cut at the end of the best epoch itself lands exactly on the budget
   boundary and never affects the checkpoint, so only cuts strictly before it are replayed.

Seeds are combined by median, not by mean: the budget is a step count and the cut list a
count of cuts, and a mean of five stopping points is not a stopping point any run had.
"""

import argparse
import json
import re
import statistics
from pathlib import Path

DEFAULT_LOG_DIR = Path("results/model_v1/logs")
DEFAULT_ARM = "embedding_sgd_sgd_lr1"
DEFAULT_SEEDS = list(range(5))

EPOCH_LINE = re.compile(
    r"^epoch\s+(?P<epoch>\d+)\s+.*?\slr\s+(?P<lr>\S+)\s+.*?\((?P<steps>\d+) steps\)")
BEST_LINE = re.compile(r"^best val loss \S+ at epoch (?P<epoch>\d+);")


def parse_log(path):
    """{'best_epoch', 'steps_per_epoch', 'cut_epochs'} for one run. `cut_epochs` are the
    epochs at whose END the lr was cut, restricted to cuts that reached the checkpoint."""
    best_epoch, lrs, steps = None, {}, set()
    for line in Path(path).read_text().splitlines():
        epoch_match = EPOCH_LINE.match(line)
        if epoch_match:
            lrs[int(epoch_match["epoch"])] = float(epoch_match["lr"])
            steps.add(int(epoch_match["steps"]))
            continue
        best_match = BEST_LINE.match(line)
        if best_match:
            best_epoch = int(best_match["epoch"])
    assert best_epoch is not None, f"{path} has no 'best val loss' line; the run did not finish"
    assert lrs, f"{path} has no epoch lines"
    assert len(steps) == 1, (
        f"{path} has epochs of differing step counts {sorted(steps)}; a step budget derived "
        f"from a ragged epoch is not a step budget")
    cut_epochs = [epoch - 1 for epoch in sorted(lrs)[1:]
                  if lrs[epoch] != lrs[epoch - 1] and epoch - 1 < best_epoch]
    return {"best_epoch": best_epoch, "steps_per_epoch": steps.pop(),
            "cut_epochs": cut_epochs}


def recover(paths):
    """
    The median budget and cut points over several seeds' logs, as absolute step counts.

    Cut points are medianed as FRACTIONS of each run's own budget rather than as raw step
    counts: the runs stopped at different epochs, so the same cut sits at a different
    absolute step in each, and the fraction is what transfers to a budget none of them had.
    """
    runs = []
    for path in paths:
        run = parse_log(path)
        run["total_steps"] = (run["best_epoch"] + 1) * run["steps_per_epoch"]
        run["cut_fractions"] = [(epoch + 1) * run["steps_per_epoch"] / run["total_steps"]
                                for epoch in run["cut_epochs"]]
        runs.append(run)

    budget = int(statistics.median(run["total_steps"] for run in runs))
    n_cuts = int(statistics.median(len(run["cut_fractions"]) for run in runs))
    fractions = [statistics.median([run["cut_fractions"][i] for run in runs
                                    if len(run["cut_fractions"]) > i])
                 for i in range(n_cuts)]
    cut_steps = sorted({int(round(fraction * budget)) for fraction in fractions})
    assert len(cut_steps) == n_cuts, "two median cut fractions collapsed onto the same step"
    assert all(0 < step < budget for step in cut_steps), "a cut landed outside the budget"
    return {"step_budget": budget, "lr_cut_steps": cut_steps,
            "n_cuts_per_seed": [len(run["cut_fractions"]) for run in runs],
            "total_steps_per_seed": [run["total_steps"] for run in runs],
            "best_epoch_per_seed": [run["best_epoch"] for run in runs],
            "steps_per_epoch": runs[0]["steps_per_epoch"]}


def main():
    parser = argparse.ArgumentParser(description="Recover the refit's step budget and lr cuts.")
    parser.add_argument("--arm", default=DEFAULT_ARM)
    parser.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--out", default="results/model_v1/replay_schedule.json")
    args = parser.parse_args()

    paths = [Path(args.log_dir) / f"{args.arm}_s{seed}.log" for seed in args.seeds]
    schedule = {**recover(paths), "arm": args.arm, "seeds": args.seeds}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(schedule, indent=2) + "\n")
    print(json.dumps(schedule, indent=2))
    print(f"\n--step-budget {schedule['step_budget']} "
          f"--lr-cut-steps {' '.join(str(s) for s in schedule['lr_cut_steps'])}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
