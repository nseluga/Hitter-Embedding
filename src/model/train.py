"""
The Phase D training loop, transcribing docs/phase-d-spec.md §5.

Four failures here are silent, and every one of them produces a trained model.

1. THE FROZEN TEST SEASON. `assert_not_test_season` runs immediately after
   `parse_args`, mirroring `baseline_ladder_report.py`. Validation defaults to 2024; 2025 is
   reachable only through `--final-run`, which is the deliberate act of spending
   the season, not a convenience. Nothing downstream re-checks this.

2. WEIGHT DECAY GROUPS. AdamW applied to one flat parameter list decays the biases
   too, which is not the shrinkage §5 describes and cannot be seen in the loss
   curve. `weight_decay_groups` is the only correct way to build the optimizer here.

3. EVAL MODE. Dropout left live during validation adds noise to the number early
   stopping and the plateau schedule both read, so a run stops on a coin flip
   rather than on a plateau. `evaluate` sets `eval()` and restores nothing else.

4. SEEDING. `torch.manual_seed` covers initialisation; the batch order needs its
   own `Generator`, seeded separately and threaded through `loader.batches`. Miss
   the second and two runs at the same seed diverge without any error.

`--benchmark` runs one epoch, times it, and exits before validation. That number is
the deliverable D.6's compute decision rests on, so it is written to
`results/model_v1/benchmark.csv` as well as to MLflow, which is gitignored.
"""

import csv
import math
import os
import resource
import subprocess
import time
from pathlib import Path

# MLflow 3.x defaults to a sqlite store and puts the file store behind a flag, and
# mlflow-skinny ships no sqlalchemy, so the default raises. Skinny is what keeps the
# full package from downgrading this project's pandas, so the file store is the
# store -- pinned here, absolute, rather than left to the caller's environment.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import torch

from src.analysis import claim1_eval as evaluation
from src.config.splits import load_splits, season_split_map
from src.model import loader
from src.model.v1 import (DEFAULT_EMBEDDING_DIM, HitterEmbeddingV1, LOSS_RULES,
                          WEIGHTINGS, factor_masks, factorized_loss, weight_decay_groups)

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKING_URI = (REPO_ROOT / "mlruns").as_uri()
DATA_DIR = Path("data/processed/phase_d")
OUT_DIR = Path("results/model_v1")
CHECKPOINT_DIR = Path("results/checkpoints")
EXPERIMENT = "model_v1"

LEARNING_RATE = 1e-3
# Phase O tunes lr and warmup and nothing else. WEIGHT_DECAY and BATCH_SIZE are
# deliberately NOT exposed as arguments: they are one setting (they set the
# decay-to-gradient ratio an embedding row sees), that ratio is the shrinkage the
# low-exposure claim is about, and tuning it would reimplement C.2 inside the network
# and then score the result against C.2. Architecture plan section 3, Phase O.
WEIGHT_DECAY = 1e-2
BATCH_SIZE = 8192
# Linear warmup over optimizer steps, off by default so every pre-Phase-O run is
# reproduced exactly. Warmup must finish before ReduceLROnPlateau can first fire
# (patience 1 => epoch 2 at the earliest), or the two would fight over param_group lr;
# WARMUP_MAX_EPOCHS enforces that.
WARMUP_STEPS = 0
WARMUP_MAX_EPOCHS = 2
PLATEAU_FACTOR = 0.3
PLATEAU_PATIENCE = 1
EARLY_STOPPING_PATIENCE = 3
MAX_EPOCHS = 50


def run_name(args):
    """
    Identifies the arm, not just the seed. Two D.8 arms at the same seed and embedding
    dimension would otherwise write the same checkpoint file and the second would
    silently overwrite the first, leaving one arm scored against the other's weights.
    """
    if args.run_name:
        return f"{args.run_name}_s{args.seed}"
    parts = [f"v1_d{args.embedding_dim}", args.loss_rule]
    if args.bilinear:
        parts.append("bilinear")
    if args.loss_weighting != "sum":
        parts.append(args.loss_weighting)
    if args.contact_inverse_frequency:
        parts.append("invfreq")
    return "_".join(parts) + f"_s{args.seed}"


def git_commit():
    """Short commit of the working tree, so a logged run can be traced to code."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def peak_rss_gb():
    """Peak resident set. macOS reports bytes here, Linux kilobytes."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if peak > 1e7 else peak / 1e6


def scored_rows(labels):
    """Total rows entering the loss across all five factors -- the loss's denominator."""
    return int(sum(int(mask.sum()) for mask in factor_masks(labels).values()))


CANONICAL = {"rule": "log", "weighting": "sum", "contact_pos_weight": None}


class LinearWarmup:
    """
    Scales lr from base/steps up to base over the first `steps` optimizer steps, then
    stands down permanently and lets ReduceLROnPlateau own the learning rate.

    Why warmup is Phase O's second knob and not decoration: D.5 gradient (b) measured
    every embedding row starting at norm 0.057 and the RAREST rows finishing furthest
    from the origin, pointing the wrong way along the wOBA axis. AdamW normalises each
    coordinate by its own second moment, so a row seen in 30 batches takes a near-full
    step every one of those 30 and lands on a large-norm random walk. Capping early step
    size is the direct lever on that, and it is a lever weight decay cannot be, since at
    the 10th percentile of exposure decay already loses to the gradient 23.9:1.
    """

    def __init__(self, optimizer, steps, base_lr):
        self.optimizer, self.steps, self.base_lr = optimizer, int(steps), base_lr
        self.step_count = 0

    @property
    def done(self):
        return self.step_count >= self.steps

    def step(self):
        """Call BEFORE optimizer.step(). No-op once warmup is over."""
        if self.done:
            return
        self.step_count += 1
        scale = self.step_count / self.steps
        for group in self.optimizer.param_groups:
            group["lr"] = self.base_lr * scale


def warmup_for(optimizer, args, n_train_rows):
    """
    The warmup schedule for this run, or None. Refuses a warmup long enough to still be
    running when ReduceLROnPlateau can first cut the lr (patience 1, so epoch 2): past
    that point warmup would raise the rate the schedule just lowered and the run would be
    on neither schedule while both logged as if it were.
    """
    if args.warmup_steps <= 0:
        return None
    steps_per_epoch = math.ceil(n_train_rows / args.batch_size)
    limit = WARMUP_MAX_EPOCHS * steps_per_epoch
    if args.warmup_steps > limit:
        raise ValueError(
            f"--warmup-steps {args.warmup_steps} exceeds {WARMUP_MAX_EPOCHS} epochs "
            f"({limit} steps at batch {args.batch_size}). Past that point "
            f"ReduceLROnPlateau can already have cut the lr and warmup would raise it "
            f"back, so the run would be on neither schedule.")
    return LinearWarmup(optimizer, args.warmup_steps, args.lr)


def run_epoch(model, tensors, indices, optimizer, generator, args, on_step=None,
              objective=None, warmup=None):
    """
    One pass over `indices`. Returns (loss per scored row, steps, seconds).
    optimizer=None evaluates instead of training: no grad, no dropout, fixed order.
    objective overrides the arm's own loss settings, which is how the canonical
    yardstick is measured on an arm that trains against something else.
    """
    objective = objective or {"rule": args.loss_rule, "weighting": args.loss_weighting,
                              "contact_pos_weight": args.contact_pos_weight}
    training = optimizer is not None
    model.train(training)
    total, rows, steps = 0.0, 0, 0
    started = time.time()

    with torch.set_grad_enabled(training):
        for index in loader.batches(indices, args.batch_size, generator=generator,
                                    shuffle=training):
            hitter, context, labels = loader.gather(tensors, index, device=args.device)
            loss, _ = factorized_loss(model(hitter, context, labels["ev"], labels["la"]),
                                      labels, **objective)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if warmup is not None:
                    warmup.step()
                optimizer.step()
            batch_rows = scored_rows(labels)
            total, rows, steps = total + loss.item(), rows + batch_rows, steps + 1
            if on_step is not None:
                on_step(steps, loss.item() / max(batch_rows, 1))

    denominator = steps if objective["weighting"] == "mean" else rows
    return total / max(denominator, 1), steps, time.time() - started


def benchmark(model, tensors, indices, optimizer, generator, args):
    """One timed training epoch, no validation. Writes the CSV D.6's decision rests on."""
    per_row, steps, seconds = run_epoch(model, tensors, indices, optimizer, generator, args)
    row = {"device": args.device, "secs_per_epoch": round(seconds, 1),
           "secs_per_100_steps": round(100 * seconds / steps, 2),
           "peak_rss_gb": round(peak_rss_gb(), 2), "steps": steps,
           "batch_size": args.batch_size, "embedding_dim": args.embedding_dim,
           "bilinear": args.bilinear, "loss_per_row": round(per_row, 5)}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "benchmark.csv"
    path.touch()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)

    mlflow.log_metrics({k: float(v) for k, v in row.items() if k != "device"})
    return row, path


def fit(model, tensors, indices, optimizer, generator, args):
    """
    Train to the plateau schedule and early stopping of §5, both on validation loss.
    Returns (best validation loss per row, best epoch, checkpoint path).
    """
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=PLATEAU_FACTOR, patience=PLATEAU_PATIENCE)
    warmup = warmup_for(optimizer, args, len(indices["train"]))
    # Warmup finishes inside epoch 0 and the epoch-end line prints the rate AFTER it has
    # returned to base, so a warmed run and a cold one produce byte-identical logs. Print
    # the ramp itself, or the ledger records a knob with no evidence it ever moved.
    if warmup is not None:
        print(f"warmup: linear over {warmup.steps} steps to lr {args.lr:.2e} "
              f"(first step at lr {args.lr / warmup.steps:.2e})", flush=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = CHECKPOINT_DIR / f"{run_name(args)}.pt"
    best, best_epoch, since_best, best_reference = float("inf"), -1, 0, float("nan")

    for epoch in range(args.max_epochs):
        def log_step(step, loss_per_row, epoch=epoch):
            mlflow.log_metric("train_loss_per_row", loss_per_row,
                              step=epoch * 10_000 + step)
            if warmup is not None and not warmup.done:
                mlflow.log_metric("warmup_lr", optimizer.param_groups[0]["lr"],
                                  step=epoch * 10_000 + step)

        train_loss, steps, seconds = run_epoch(model, tensors, indices["train"], optimizer,
                                               generator, args, on_step=log_step,
                                               warmup=warmup)
        val_loss, _, _ = run_epoch(model, tensors, indices[args.eval_split], None,
                                   None, args)
        # the arm's own objective drives early stopping, but three of the six D.8 arms
        # train against something other than the likelihood, so their val_loss is in
        # different units and cannot be compared across arms. `reference` is the same
        # quantity for every arm -- unweighted log loss per scored row -- and is what
        # makes the ledger readable. It costs one extra forward pass over 705k rows.
        reference = val_loss if args.canonical else run_epoch(
            model, tensors, indices[args.eval_split], None, None, args,
            objective=CANONICAL)[0]
        schedule.step(val_loss)
        mlflow.log_metrics({"train_loss_epoch": train_loss, "val_loss_epoch": val_loss,
                            "val_reference_epoch": reference,
                            "lr": optimizer.param_groups[0]["lr"],
                            "epoch_seconds": seconds}, step=epoch)
        if warmup is not None and warmup.done and epoch == 0:
            print(f"warmup: complete after {warmup.step_count} steps", flush=True)
        print(f"epoch {epoch:>3d}  train {train_loss:.5f}  val {val_loss:.5f}  "
              f"ref {reference:.5f}  lr {optimizer.param_groups[0]['lr']:.2e}  "
              f"{seconds:.0f}s ({steps} steps)")

        if val_loss < best:
            best, best_epoch, since_best, best_reference = val_loss, epoch, 0, reference
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss,
                        "args": vars(args)}, checkpoint)
        else:
            since_best += 1
            if since_best >= EARLY_STOPPING_PATIENCE:
                print(f"early stop: {since_best} epochs without improvement")
                break

    mlflow.log_metrics({"best_val_loss": best, "best_epoch": best_epoch,
                        "best_val_reference": best_reference})
    return best, best_epoch, best_reference, checkpoint


def contact_pos_weight(tensors, train_index):
    """
    Inverse class frequency for the contact head, from the TRAIN split only (§2.2's
    D.8 arm). Counting on the full table would read the validation and test seasons
    to set a training constant, which is leakage no loss curve would reveal.
    """
    contact = tensors["contact"][train_index]
    observed = contact[contact != -1]
    positive = int((observed == 1).sum())
    negative = int(len(observed) - positive)
    return torch.tensor(negative / max(positive, 1), dtype=torch.float32)


def run(args):
    tensors, manifest = loader.load_tensors(args.data_dir)
    indices = loader.split_indices(tensors["season"])
    args.contact_pos_weight = (contact_pos_weight(tensors, indices["train"]).to(args.device)
                               if args.contact_inverse_frequency else None)
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    model = HitterEmbeddingV1(manifest["n_hitters"], tensors["context"].shape[1],
                              embedding_dim=args.embedding_dim,
                              n_bins=manifest["n_quality_bins"],
                              bilinear=args.bilinear,
                              split=getattr(args, "split", False),
                              spray=not getattr(args, "no_spray", False)).to(args.device)
    optimizer = torch.optim.AdamW(weight_decay_groups(model, WEIGHT_DECAY), lr=args.lr)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"{'bench' if args.benchmark else 'fit'}-"
                                   f"{args.device}-{run_name(args)}"):
        mlflow.log_params({
            "seed": args.seed, "device": args.device, "embedding_dim": args.embedding_dim,
            "batch_size": args.batch_size, "lr": args.lr,
            "warmup_steps": args.warmup_steps,
            "weight_decay": WEIGHT_DECAY, "loss_rule": args.loss_rule,
            "bilinear": args.bilinear, "loss_weighting": args.loss_weighting,
            "contact_inverse_frequency": args.contact_inverse_frequency,
            "contact_pos_weight": (float(args.contact_pos_weight)
                                   if args.contact_pos_weight is not None else None),
            "eval_split": args.eval_split,
            "eval_season": args.eval_season, "final_run": args.final_run,
            "train_seasons": manifest["train_seasons"], "n_hitters": manifest["n_hitters"],
            "n_context": tensors["context"].shape[1],
            "parameters": sum(p.numel() for p in model.parameters()),
            "git_commit": git_commit(),
        })
        mlflow.log_artifact(str(Path(args.data_dir) / "manifest.json"))

        if args.benchmark:
            row, path = benchmark(model, tensors, indices["train"], optimizer, generator, args)
            print(f"benchmark: {row}")
            print(f"wrote {path}")
            return row

        best, best_epoch, reference, checkpoint = fit(model, tensors, indices, optimizer,
                                                      generator, args)
        print(f"best val loss {best:.5f} at epoch {best_epoch}; "
              f"reference {reference:.5f}; checkpoint {checkpoint}")
        return {"best_val_loss": best, "best_epoch": best_epoch, "reference": reference}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train the Phase D v1 model.")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help="Phase O knob. Held at 1e-3 for all 119 pre-O ledger runs.")
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS,
                        help="Phase O knob. Linear lr warmup over this many optimizer "
                             "steps; 0 reproduces every pre-O run exactly.")
    parser.add_argument("--loss-rule", choices=list(LOSS_RULES), default="log",
                        help="'rps' is the promote-only screen arm, never the default")
    parser.add_argument("--loss-weighting", choices=list(WEIGHTINGS), default="sum",
                        help="'mean' averages each factor over its own rows; a D.8 arm")
    parser.add_argument("--contact-inverse-frequency", action="store_true",
                        help="inverse class frequency inside the contact head; a D.8 arm")
    parser.add_argument("--split", action="store_true",
                        help="train the three-class contact split head (2026-08-08)")
    parser.add_argument("--no-spray", action="store_true",
                        help="drop spray as a quality dimension — the §7 D.8 arm")
    parser.add_argument("--bilinear", action="store_true",
                        help="low-rank interaction term; a D.8 arm, off by default")
    parser.add_argument("--run-name", default=None,
                        help="names the arm; checkpoints and MLflow runs use it")
    parser.add_argument("--eval-season", type=int, default=None,
                        help="defaults to the frozen validation season")
    parser.add_argument("--benchmark", action="store_true",
                        help="time one training epoch and exit before validation")
    # the frozen test season is scored ONCE, for the final reported result;
    # this flag is the deliberate act of doing so, never a convenience
    parser.add_argument("--final-run", action="store_true",
                        help="permit scoring the frozen TEST season")
    args = parser.parse_args()

    # never score against the frozen test season outside a final run. the season is
    # an explicit argument and the split is derived FROM it, not from --final-run:
    # deriving it the other way would make this guard a tautology it can never fail.
    config = load_splits()
    args.eval_season = args.eval_season or config["split"]["val"][0]
    evaluation.assert_not_test_season(args.eval_season, final_run=args.final_run)
    args.eval_split = season_split_map(config)[args.eval_season]
    args.canonical = (args.loss_rule == "log" and args.loss_weighting == "sum"
                      and not args.contact_inverse_frequency)
    assert args.eval_split != "train", \
        f"season {args.eval_season} is a training season; validation would be in-sample"

    run(args)


if __name__ == "__main__":
    main()
