"""
Phase O.1 -- pick a learning rate and warmup from the ledger, and nothing else.

PRE-REGISTERED BEFORE ANY selection RUN EXISTS. The rule below is fixed in code so that it
cannot be adjusted after seeing which arm it promotes.

WHY THIS MODULE MAY SELECT ON LIKELIHOOD WHEN PHASE E SAID IT COULD NOT. Across the seven
well-trained rebuild arms, held-out log likelihood and the claim-1 rank statistic correlate at
0.000. Seven points put a 95% interval of roughly +/-0.75 on that estimate, so it is NOT
evidence that the two are unrelated -- it is only evidence that likelihood cannot be shown
to RANK converged models. Phase O does not ask it to. It asks whether a run reached its own
training objective, which is the one thing a training loss is a direct measurement of. That
distinction, not the 0.000, is the licence.

THE SELECTION FRAME IS 2024, WHICH IS ALSO CLAIM 1's FRAME. `reference` is loss on the 2024
validation split, and every claim-1 number in Phases E and F is scored on 2024 as well. This
is the project's frozen walk-forward protocol -- 2024 is the SELECTION season and 2025 is the
untouched test season -- but it has a consequence Phase M must carry: any claim-1 number
computed on 2024 from a build tuned here is post-selection and must be labelled descriptive.
The confirmatory number is the 2025 run, which stays sealed until after Phase M.

CLAIM 1 IS NEVER READ HERE. Not as a tiebreak, not as a sanity check. This module does not
import `claim1_eval` and `test_selection_select.py` asserts that it does not, because a tuned
build chosen with one eye on claim 1 makes claim 1 a selected number and the project's
headline is a claim-1 number.

THE RULE.
1. Score each selection config by the MEAN of its `reference` column across seeds -- unweighted
   held-out 2024 log loss per scored row, the one ledger column comparable across configs.
2. The noise floor is the across-seed standard deviation of `reference` for the rebuild
   `baseline` arm (five seeds, same architecture, same build): 1.04e-4.
3. `lr1e3` (lr 1e-3, no warmup) is the incumbent -- the setting all 119 pre-O runs used.
   An arm is PROMOTED only if it beats the incumbent's mean by more than `MARGIN_SES`
   standard errors of the difference in means (the noise floor supplies the per-run sd). Otherwise the incumbent stands and Phase O reports
   that the untuned setting was already the best of the seven.
4. Lower `reference` is better (it is a loss).
5. Six challengers are tested against one incumbent at the same threshold, so a promoted
   arm carries `requires_confirmation` until it has CONFIRMATION_SEEDS seeds. A two-seed
   screen selects; it never concludes. The sixth, `lr1e3_warm2k`, was pre-registered on
   2026-08-31 after the first five had run; MARGIN_SES is unchanged, so it is one more
   chance to clear the same bar.

GRID COMPLETENESS. `sweep.py` queues config-major, so a night that runs short drops whole
arms off the END of the grid rather than thinning every arm evenly -- and a mean over the
three arms that happened to finish looks exactly like a mean over six. The expected arm set
is pre-registered as EXPECTED_ARMS and any absence refuses the selection. Same for depth: an arm
with fewer than MIN_SEEDS_PER_ARM seeds has no measurable spread and is not a mean.

GUARD. The incumbent arm re-runs the rebuild baseline recipe at a different stage, so its mean
`reference` must land within `INCUMBENT_TOLERANCE_SDS` of the rebuild baseline's. If it does
not, something moved underneath the comparison -- almost certainly the tensor build -- and
the selection is refused rather than reported.

Read-only. Nothing trains here.
"""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

LEDGER = Path("results/model_v1/sweep_log.csv")
OUT_DIR = Path("results/hyperparameter_tuning")
STAGE = "selection"
INCUMBENT = "lr1e3"
NOISE_FLOOR_STAGE, NOISE_FLOOR_CONFIG = "rebuild", "baseline"
MARGIN_SES = 2.0
INCUMBENT_TOLERANCE_SDS = 4.0
CONFIRMATION_SEEDS = 5
MIN_SEEDS_PER_ARM = 2
# The pre-registered grid, declared HERE rather than imported from `sweep`. Importing sweep
# transitively puts claim1_eval in sys.modules, and this module's whole licence is that claim
# 1 cannot reach it -- a firewall that holds only until someone adds a convenient import is
# not a firewall. `test_selection_select.py` asserts this list still matches sweep.STAGES["selection"]; a
# test may import both, because a test is not the thing being firewalled.
EXPECTED_ARMS = ("lr1e3", "lr1e3_warm", "lr1e3_warm2k", "lr3e3", "lr3e3_warm",
                 "lr3e4", "lr3e4_warm")

# Per-stage version of the four constants above: `selection` and `clean` score against
# different incumbents and noise floors, so the module-level names can no longer be the
# only source of truth. They stay bound to `selection`'s values because tests import them
# directly; `build_check`/`noise_floor`/`select` look the requested stage up here instead.
# `clean` arms are declared here, not imported from sweep -- same firewall as EXPECTED_ARMS above.
STAGE_RULES = {
    "selection": dict(incumbent=INCUMBENT, noise_floor=(NOISE_FLOOR_STAGE, NOISE_FLOOR_CONFIG),
                      expected_arms=EXPECTED_ARMS),
    # the noise floor IS the incumbent's own arm here, so the guard's drift is an identity
    # (build_check reports this via independent_of_noise_floor, not silently)
    # screen_seed: an arm left at ONE seed whose `reference` did not beat the incumbent's
    # seed-0 run was screened out per the pre-registration, not under-run; it is reported and
    # never promotable. An arm at one seed that DID beat it should have been deepened -> underpowered.
    "clean": dict(incumbent="clean_base", noise_floor=("clean", "clean_base"), screen_seed=0,
                 expected_arms=("clean_base", "clean_dim64", "clean_dropout",
                                "clean_wd3e-2", "clean_wd3e-3", "clean_wd_exposure")),
}


def ledger_rows(path=LEDGER):
    with Path(path).open() as handle:
        return [r for r in csv.DictReader(handle) if r["status"] == "ok"]


def references(rows, stage, config):
    """`reference` values for one (stage, config), ascending by seed."""
    matched = [r for r in rows if r["stage"] == stage and r["config"] == config]
    return [float(r["reference"]) for r in sorted(matched, key=lambda r: int(r["seed"]))
            if r["reference"] not in ("", None)]


def references_by_seed(rows, stage, config):
    """`reference` keyed by seed -- the drift guard needs seed identity, not just values."""
    return {int(r["seed"]): float(r["reference"]) for r in rows
            if r["stage"] == stage and r["config"] == config
            and r["reference"] not in ("", None)}


def build_check(rows, stage=STAGE):
    """
    Is the stage's incumbent scored in the same UNITS as the noise-floor arm?

    `reference` is log loss over quality bins, and `model_v1` and `phase_d5` carry different
    bin edges, so a run on the wrong build lands on a different scale while every ledger
    column still lines up. That is the one failure this check exists for.

    Two regimes, because the incumbent re-runs the noise-floor recipe and may therefore
    REPRODUCE it rather than merely resemble it:

    - shared seeds -> require EXACT equality on them. Two runs on different builds cannot
      agree to the ledger's precision, so this is a decisive units test.
    - no shared seeds -> fall back to the drift tolerance, which is all that is available.

    The distinction is reported, not hidden: a drift computed against a floor sample that
    CONTAINS the incumbent's own runs is an algebraic identity, and reporting it as a passed
    check overstates what was verified -- `clean`'s floor IS the incumbent arm, so every one
    of its seeds is shared and `independent_of_noise_floor` is always False there.
    """
    rule = STAGE_RULES[stage]
    floor_stage, floor_config = rule["noise_floor"]
    incumbent = references_by_seed(rows, stage, rule["incumbent"])
    floor = references_by_seed(rows, floor_stage, floor_config)
    shared = sorted(set(incumbent) & set(floor))
    mismatched = [seed for seed in shared if incumbent[seed] != floor[seed]]
    return {"shared_seeds": shared,
            "independent_of_noise_floor": not shared,
            "reproduces_shared_seeds": bool(shared) and not mismatched,
            "mismatched_seeds": mismatched}


def noise_floor(rows, stage=STAGE):
    """Across-seed sd of `reference` for the stage's noise-floor arm -- the smallest
    difference that is not seed noise. Computed from the ledger, never hardcoded, so a
    re-run moves it."""
    floor_stage, floor_config = STAGE_RULES[stage]["noise_floor"]
    values = references(rows, floor_stage, floor_config)
    if len(values) < 2:
        raise ValueError(
            f"need >=2 seeds of {floor_stage}/{floor_config} to set the noise "
            f"floor; found {len(values)}. Without it every margin is unfalsifiable.")
    return statistics.stdev(values), statistics.mean(values), len(values)


def select(rows, stage=STAGE):
    """The pre-registered rule. Returns the full record, verdict included."""
    if stage not in STAGE_RULES:
        raise ValueError(f"unknown stage {stage!r}; known stages are {sorted(STAGE_RULES)}")
    rule = STAGE_RULES[stage]
    incumbent_name = rule["incumbent"]
    expected = sorted(rule["expected_arms"])
    configs = sorted({r["config"] for r in rows if r["stage"] == stage})
    if not configs:
        raise ValueError(f"no completed {stage} runs in the ledger -- run the sweep first")
    missing_arms = [name for name in expected if name not in configs]
    # Symmetric on purpose. Checking only for MISSING arms stops a night that ran short;
    # it does nothing about an arm added to the ledger after the table was seen, which is
    # the failure that actually breaks a pre-registration.
    unexpected_arms = [name for name in configs if name not in expected]
    floor_sd, floor_mean, floor_n = noise_floor(rows, stage)

    arms = {}
    for name in configs:
        values = references(rows, stage, name)
        arms[name] = {
            "n_seeds": len(values),
            "reference_mean": statistics.mean(values),
            "reference_sd": statistics.stdev(values) if len(values) > 1 else None,
            "reference_values": values,
        }

    screen_seed = rule.get("screen_seed")
    screen_ref = (references_by_seed(rows, stage, incumbent_name).get(screen_seed)
                  if screen_seed is not None else None)
    for name, arm in arms.items():
        arm["screened_out"] = bool(
            screen_ref is not None and name != incumbent_name and arm["n_seeds"] == 1
            and arm["reference_values"][0] >= screen_ref)
    underpowered = sorted(k for k, v in arms.items()
                          if v["n_seeds"] < MIN_SEEDS_PER_ARM and not v["screened_out"])

    if incumbent_name not in arms:
        raise ValueError(
            f"the incumbent arm `{incumbent_name}` has no completed runs. It is the control: "
            f"without it nothing in this stage has anything to be tuned relative to.")

    if arms[incumbent_name]["n_seeds"] < MIN_SEEDS_PER_ARM:
        raise ValueError(
            f"the incumbent `{incumbent_name}` has {arms[incumbent_name]['n_seeds']} seed(s); "
            f"every margin in this stage is measured against it, so it needs at least "
            f"{MIN_SEEDS_PER_ARM}.")

    incumbent_mean = arms[incumbent_name]["reference_mean"]
    drift = abs(incumbent_mean - floor_mean)
    build = build_check(rows, stage)
    # Exact reproduction on a shared seed is strictly stronger than the drift tolerance, so it
    # supersedes it where available. Where it is not available the tolerance is all there is.
    guard_ok = (build["reproduces_shared_seeds"] if build["shared_seeds"]
                else drift <= INCUMBENT_TOLERANCE_SDS * floor_sd)

    # The threshold is a difference of two means, so it is scaled by the standard error of
    # THAT difference, not by a single run's sd. With n seeds per arm and the noise-floor arm's
    # sd standing in for the per-run sd, SE = sd * sqrt(1/n_a + 1/n_incumbent). At two seeds
    # each this happens to equal `floor_sd`, which is exactly why the earlier version looked
    # correct; it stops being correct the moment the seed counts differ.
    n_inc = arms[incumbent_name]["n_seeds"]
    for name, arm in arms.items():
        se = floor_sd * math.sqrt(1 / arm["n_seeds"] + 1 / n_inc)
        arm["margin_vs_incumbent"] = incumbent_mean - arm["reference_mean"]  # >0 is better
        arm["standard_error"] = se
        arm["margin_in_ses"] = arm["margin_vs_incumbent"] / se if se else 0.0
        arm["promotable"] = bool(name != incumbent_name and not arm["screened_out"]
                                 and arm["margin_in_ses"] > MARGIN_SES)

    challengers = {k: v for k, v in arms.items() if k != incumbent_name and v["promotable"]}
    winner = (max(challengers, key=lambda k: challengers[k]["margin_in_ses"])
              if challengers else incumbent_name)

    # Five challengers are each tested against one incumbent at the same threshold, so the
    # family-wise false-promotion rate is roughly five times the per-comparison one, and a
    # two-seed mean is a thin estimate to spend that on. The screen therefore SELECTS and
    # never CONCLUDES: a promoted arm is re-run at CONFIRMATION_SEEDS before anything
    # downstream trains on it. This is a field in the artifact rather than a line in the
    # spec so that a reader of the JSON cannot miss it.
    confirmed = arms[winner]["n_seeds"] >= CONFIRMATION_SEEDS
    requires_confirmation = winner != incumbent_name and not confirmed

    return {
        "stage": stage,
        "rule": {
            "metric": "reference (held-out unweighted log loss per scored row)",
            "incumbent": incumbent_name,
            "margin_ses": MARGIN_SES,
            "noise_floor_sd": floor_sd,
            "noise_floor_source": "/".join(rule["noise_floor"]),
            "noise_floor_n_seeds": floor_n,
            "claim1_read": False,
        },
        "guard": {
            "incumbent_mean": incumbent_mean,
            "rebuild_baseline_mean": floor_mean,
            "drift": drift,
            "tolerance": INCUMBENT_TOLERANCE_SDS * floor_sd,
            "drift_is_informative": build["independent_of_noise_floor"],
            "basis": ("exact reproduction on shared seeds"
                      if build["shared_seeds"] else "drift within tolerance"),
            **build,
            "passed": guard_ok,
        },
        "arms": arms,
        "grid": {
            "expected_arms": expected,
            "missing_arms": missing_arms,
            "underpowered_arms": underpowered,
            "screened_out_arms": sorted(k for k, v in arms.items() if v["screened_out"]),
            "unexpected_arms": unexpected_arms,
            "min_seeds_per_arm": MIN_SEEDS_PER_ARM,
            "complete": not missing_arms and not underpowered and not unexpected_arms,
        },
        "winner": winner,
        "n_challengers_tested": len(arms) - 1,
        "requires_confirmation": requires_confirmation,
        "confirmation_seeds": CONFIRMATION_SEEDS,
        "verdict": ("incomplete_grid" if missing_arms or underpowered or unexpected_arms else
                    "guard_failed" if not guard_ok else
                    "incumbent_stands" if winner == incumbent_name else
                    "tuned_pending_confirmation" if requires_confirmation else "tuned"),
    }


def report(result, out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "selection.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")

    csv_path = out_dir / "selection.csv"
    fields = ("config", "n_seeds", "reference_mean", "reference_sd",
              "margin_vs_incumbent", "standard_error", "margin_in_ses", "promotable")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, arm in sorted(result["arms"].items(),
                                key=lambda kv: kv[1]["reference_mean"]):
            writer.writerow({"config": name, **{f: arm[f] for f in fields[1:]}})
    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--ledger", default=str(LEDGER))
    parser.add_argument("--stage", default=STAGE)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    result = select(ledger_rows(args.ledger), stage=args.stage)
    json_path, csv_path = report(result, args.out_dir)

    floor = result["rule"]["noise_floor_sd"]
    print(f"noise floor (seed sd of {result['rule']['noise_floor_source']}): {floor:.2e}")
    grid = result["grid"]
    if not grid["complete"]:
        print(f"INCOMPLETE GRID: missing {grid['missing_arms'] or 'none'}, "
              f"under {grid['min_seeds_per_arm']} seeds "
              f"{grid['underpowered_arms'] or 'none'}. Selection refused.")
    if not result["guard"]["passed"]:
        print(f"GUARD FAILED: incumbent mean {result['guard']['incumbent_mean']:.5f} vs "
              f"noise-floor arm ({result['rule']['noise_floor_source']}) mean "
              f"{result['guard']['rebuild_baseline_mean']:.5f} "
              f"(drift {result['guard']['drift']:.2e} > "
              f"{result['guard']['tolerance']:.2e}). Selection refused.")
    elif not result["guard"]["drift_is_informative"]:
        print(f"guard passed on {result['guard']['basis']} -- the noise floor "
              f"({result['rule']['noise_floor_source']}) is the incumbent's own arm here, "
              f"so this is not independent evidence, only a units check.")
    for name, arm in sorted(result["arms"].items(), key=lambda kv: kv[1]["reference_mean"]):
        mark = "*" if name == result["winner"] else " "
        print(f" {mark} {name:<12} n={arm['n_seeds']}  ref {arm['reference_mean']:.5f}  "
              f"margin {arm['margin_in_ses']:+.1f} se")
    print(f"verdict: {result['verdict']}  winner: {result['winner']}")
    if result["requires_confirmation"]:
        print(f"a {result['arms'][result['winner']]['n_seeds']}-seed screen selects, it "
              f"does not conclude. Confirm at {CONFIRMATION_SEEDS} seeds before anything "
              f"downstream trains on this arm:\n"
              f"  PYTHONPATH=. .venv/bin/python -m src.model.sweep --stage {args.stage} "
              f"--seeds {CONFIRMATION_SEEDS} --hours 9")
    print(f"wrote {json_path} and {csv_path}")
    return result


if __name__ == "__main__":
    main()
