"""Loss curves straight from mlruns/ (no mlflow UI needed; full mlflow would downgrade pandas).
Usage: .venv/bin/python -m src.analysis.loss_curves [run-name pattern ...]
  Always writes results/loss_curves/all_runs.csv (per-epoch metrics, every run, every experiment).
  Each pattern -> results/loss_curves/<pattern>.png, train (dashed) vs val (solid), one colour per run.
  Patterns match the run name exactly after stripping "fit-cpu-" and a trailing "_s<seed>"."""
import csv, pathlib, re, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "results/loss_curves"
# per-step train_loss_per_row (~2.7M lines) deliberately left out
EPOCH_METRICS = ["train_loss_epoch", "val_loss_epoch", "val_reference_epoch", "lr", "embedding_lr", "epoch_seconds"]

def read(f):  # mlflow file store: "<timestamp> <value> <step>" per line
    return {int(s): float(v) for _, v, s in (l.split() for l in f.read_text().splitlines())} if f.exists() else {}

def runs():
    for exp in sorted((ROOT / "mlruns").iterdir()):
        if not (exp / "meta.yaml").exists():
            continue
        exp_name = re.search(r"^name: (.+)$", (exp / "meta.yaml").read_text(), re.M).group(1)
        for r in sorted(exp.iterdir()):
            tag = r / "tags/mlflow.runName"
            if tag.exists():
                name = tag.read_text().strip()
                yield exp_name, r.name, name, re.sub(r"_s\d+$", "", name.removeprefix("fit-cpu-")), r

def main(patterns):
    OUT.mkdir(parents=True, exist_ok=True)
    all_runs = list(runs())
    with open(OUT / "all_runs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["experiment", "run_id", "run_name", "group", "epoch", *EPOCH_METRICS])
        for exp, rid, name, group, r in all_runs:
            m = {k: read(r / "metrics" / k) for k in EPOCH_METRICS}
            for ep in sorted(set().union(*m.values())):
                w.writerow([exp, rid, name, group, ep, *(m[k].get(ep, "") for k in EPOCH_METRICS)])
    print(OUT / "all_runs.csv", len(all_runs), "runs")
    for pat in patterns:
        sel = sorted((x for x in all_runs if x[3] == pat), key=lambda x: (x[2], x[1]))
        if not sel:
            sys.exit(f"no runs in group {pat!r}")
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (_, rid, name, _, r) in enumerate(sel):
            label = f"{name.rsplit('_', 1)[-1]} {rid[:4]}"  # seed + run id: groups can hold reruns of one seed
            for metric, ls, kind in (("train_loss_epoch", "--", "train"), ("val_loss_epoch", "-", "val")):
                s = read(r / "metrics" / metric)
                if s:
                    ax.plot(*zip(*sorted(s.items())), ls, color=f"C{i % 10}", label=f"{label} {kind}")
        # final refit: one 2025 test loss per seed (src/analysis/final_loss_report.py), drawn as a marker
        # at the last epoch. A single end point, not a curve: the refit has no validation, test is never scored mid-training.
        final = OUT / "final_test_loss.csv"
        tested = {r["run_name"]: r for r in csv.DictReader(open(final))} if final.exists() else {}
        for t in tested.values():  # one per seed; the group can hold reruns of a seed
            if t["run_name"].removesuffix(t["run_name"].rsplit("_", 1)[-1]).rstrip("_") == pat:
                ax.plot(int(t["epoch"]), float(t["test"]), "k*", ms=10,
                        label="test 2025 (final, each seed)" if t["run_name"].endswith("_s0") else None)
        has_val = any((r / "metrics/val_loss_epoch").exists() for *_, r in sel)
        ax.set(xlabel="epoch", ylabel="loss per row (sum of 5 factor NLLs)",
               title=f"{pat}: train (dashed){' vs val (solid)' if has_val else ', no val logged'}"
                     f"{', test 2025 (star)' if any(n.removeprefix('fit-cpu-') in tested for _, _, n, _, _ in sel) else ''}, {len(sel)} runs")
        ax.legend(fontsize=6, ncol=2); fig.tight_layout()
        fig.savefig(OUT / f"{pat}.png", dpi=130); plt.close(fig)
        print(OUT / f"{pat}.png", len(sel), "runs")

if __name__ == "__main__":
    main(sys.argv[1:])
