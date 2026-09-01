"""
Gates for Phase O.1 -- the tuning stage and its selection rule.

Three things can go wrong here and each of them is silent.

The first is that Phase O reads claim 1. The selection rule is legitimate only because it
never touches the metric the paper's headline is; if it does, the headline becomes a
selected number and nothing downstream can undo that. `test_selector_cannot_see_claim1`
imports the module in a clean interpreter and asserts nothing claim-1-shaped came
with it -- a source grep is not enough, and the first version of that test passed with the
import present.

The second is that selection trains on a different tensor build than the rebuild incumbent it is
compared against. `reference` is log loss over the quality bins, and `model_v1` and
`phase_d5` have different bin edges, so the two would be in different units while every
column still lined up. The stage pins its build; the guard catches it at report time.

The third is that warmup and ReduceLROnPlateau fight over `param_group["lr"]`. Warmup
raises the rate, the plateau schedule lowers it; if warmup is still running when the
schedule first fires, the run is on neither. `fit` refuses that configuration.
"""

import csv
import os
import subprocess
import sys
import inspect
from pathlib import Path

import pytest
import torch

from src.analysis import hyperparameter_tuning_select
from src.model import sweep, train


# --------------------------------------------------------------------------- warmup

def optimizer_at(lr):
    return torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))], lr=lr)


def test_warmup_ramps_from_one_step_to_base_and_stops():
    optimizer = optimizer_at(1e-3)
    warmup = train.LinearWarmup(optimizer, steps=4, base_lr=1e-3)
    seen = []
    for _ in range(6):
        warmup.step()
        seen.append(optimizer.param_groups[0]["lr"])
    assert seen[:4] == pytest.approx([2.5e-4, 5e-4, 7.5e-4, 1e-3])
    # once done it must stand down permanently, or it would overwrite every plateau cut
    assert warmup.done
    assert seen[4] == seen[5] == pytest.approx(1e-3)


def test_warmup_leaves_plateau_cuts_alone_once_done():
    optimizer = optimizer_at(1e-3)
    warmup = train.LinearWarmup(optimizer, steps=2, base_lr=1e-3)
    warmup.step()
    warmup.step()
    optimizer.param_groups[0]["lr"] = 3e-4     # a ReduceLROnPlateau cut
    warmup.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)


class Args:
    def __init__(self, warmup_steps=0, batch_size=8192, lr=1e-3):
        self.warmup_steps, self.batch_size, self.lr = warmup_steps, batch_size, lr


def test_zero_warmup_builds_no_schedule_so_pre_o_runs_are_bit_identical():
    assert train.WARMUP_STEPS == 0
    assert train.warmup_for(optimizer_at(1e-3), Args(warmup_steps=0), 5_880_000) is None


def test_one_epoch_of_warmup_is_allowed():
    warmup = train.warmup_for(optimizer_at(1e-3), Args(warmup_steps=718), 5_880_000)
    assert isinstance(warmup, train.LinearWarmup)
    assert warmup.steps == 718 and warmup.base_lr == 1e-3


def test_warmup_longer_than_the_plateau_can_wait_is_refused():
    """It could still be raising the lr after ReduceLROnPlateau first cuts it."""
    rows = 5_880_000
    steps_per_epoch = -(-rows // 8192)
    limit = train.WARMUP_MAX_EPOCHS * steps_per_epoch
    train.warmup_for(optimizer_at(1e-3), Args(warmup_steps=limit), rows)   # boundary is ok
    with pytest.raises(ValueError, match="exceeds"):
        train.warmup_for(optimizer_at(1e-3), Args(warmup_steps=limit + 1), rows)


def test_warmup_limit_scales_with_batch_size():
    """Fewer, larger batches means fewer steps per epoch and a tighter cap."""
    big = train.warmup_for(optimizer_at(1e-3), Args(warmup_steps=718, batch_size=8192),
                           5_880_000)
    assert big is not None
    with pytest.raises(ValueError):
        train.warmup_for(optimizer_at(1e-3), Args(warmup_steps=718, batch_size=8_000_000),
                         5_880_000)


# ------------------------------------------------------------------- frozen knobs

def test_batch_size_and_weight_decay_are_not_tunable():
    """Quarantined by the architecture plan: they set the decay-to-gradient ratio, which
    IS the shrinkage the low-exposure claim measures. An argument here would let a Phase O
    run reimplement C.2 inside the network and then be scored against C.2."""
    parser_source = inspect.getsource(train.main)
    assert "--weight-decay" not in parser_source
    for _, extra in sweep.STAGES["selection"]:
        assert "--batch-size" not in extra
        assert "--weight-decay" not in extra


def test_selection_pins_the_build_rebuild_trained_on():
    from src.analysis import provenance
    for _, extra in sweep.STAGES["selection"]:
        assert "--data-dir" in extra
        assert extra[extra.index("--data-dir") + 1] == provenance.CANONICAL_DATA_DIR


def test_selection_grid_contains_the_untuned_incumbent():
    names = [name for name, _ in sweep.STAGES["selection"]]
    assert hyperparameter_tuning_select.INCUMBENT in names
    incumbent = dict(sweep.STAGES["selection"])[hyperparameter_tuning_select.INCUMBENT]
    assert incumbent[incumbent.index("--lr") + 1] == "1e-3"
    assert "--warmup-steps" not in incumbent


def test_knobs_reads_what_the_config_actually_ran_at():
    default = sweep.canonical_lr(train.LEARNING_RATE)
    assert sweep.knobs(["--split"], "DEFAULT") == (default, "0", "DEFAULT")
    assert sweep.knobs(["--lr", "3e-3", "--warmup-steps", "719"], "DEFAULT") \
        == ("0.003", "719", "DEFAULT")
    # last flag wins, matching argparse, so a --train-args override is recorded truthfully
    assert sweep.knobs(["--lr", "1e-3", "--lr", "3e-4"], "D") == ("0.0003", "0", "D")
    # a stage tuple's --data-dir wins over the sweep-level default, as `launch` orders them
    assert sweep.knobs(["--data-dir", "pinned"], "DEFAULT")[2] == "pinned"


def test_one_spelling_of_a_learning_rate():
    """`1e-3` and `0.001` are the same number and two different strings, and the ledger is
    read as text. Everything that writes the column goes through canonical_lr."""
    assert sweep.canonical_lr("1e-3") == sweep.canonical_lr(0.001) == "0.001"
    assert sweep.canonical_lr("3e-4") == "0.0003"


def test_quarantined_knobs_cannot_be_smuggled_through_train_args():
    for flag in sweep.QUARANTINED_FLAGS:
        with pytest.raises(SystemExit):
            sweep.main_argv(["--stage", "selection", "--train-args", flag, "2048"])


def test_ledger_carries_the_knobs():
    assert "lr" in sweep.LEDGER_FIELDS and "warmup_steps" in sweep.LEDGER_FIELDS
    with Path("results/model_v1/sweep_log.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows, "ledger is empty"
    assert set(rows[0]) == set(sweep.LEDGER_FIELDS)
    pre_o = [r for r in rows if r["stage"] != "selection"]
    assert pre_o and all(r["lr"] == "0.001" and r["warmup_steps"] == "0" for r in pre_o), \
        "the pre-O history is the finding: one lr, no warmup, 119 runs"
    # every rebuild row names the build the 2026-08-20 probe established it trained on
    rebuild = [r for r in pre_o if r["stage"] == "rebuild"]
    assert rebuild and all(r["data_dir"].startswith("data/processed/phase_d5") for r in rebuild)


# ------------------------------------------------------------------- selection rule

def ledger(rows):
    """(stage, config, seed, reference) tuples -> ledger dicts."""
    return [{"stage": s, "config": c, "seed": str(i), "status": "ok", "reference": str(v)}
            for s, c, i, v in rows]


D10 = [("rebuild", "baseline", i, v) for i, v in
       enumerate([1.02584, 1.02585, 1.02578, 1.02563, 1.02590])]
NEUTRAL = 1.02580


def full_grid(*rows, seeds=2):
    """The selection rows under test, padded out to the whole pre-registered grid.

    `select` refuses a partial grid on purpose -- `sweep.queue` is config-major, so a night
    that runs short drops whole arms and a mean over the arms that finished is
    indistinguishable from a mean over all seven. Every fixture therefore has to supply
    seven arms; the padding ones sit at the incumbent's value so only the arm under test moves.

    Fixtures put the incumbent on seeds 7-8, off the noise floor's 0-4. Sharing a seed with
    the floor puts the guard in its exact-reproduction regime, which is a separate question
    from the margin these fixtures are about -- the shared-seed tests below cover it."""
    named = {r[1] for r in rows}
    padding = [("selection", arm, i, NEUTRAL)
               for arm in hyperparameter_tuning_select.EXPECTED_ARMS if arm not in named
               for i in range(seeds)]
    return list(rows) + padding


def test_the_pre_registered_grid_matches_the_stage_it_scores():
    """selection_select declares EXPECTED_ARMS itself instead of importing sweep.STAGES, because
    importing sweep drags claim1_eval into sys.modules. The duplication is deliberate and
    this is the test that keeps the two copies honest."""
    assert sorted(hyperparameter_tuning_select.EXPECTED_ARMS) == sorted(n for n, _ in sweep.STAGES["selection"])


def test_a_short_night_is_refused_rather_than_averaged():
    rows = ledger(D10 + [("selection", "lr1e3", 7, NEUTRAL), ("selection", "lr1e3", 8, NEUTRAL),
                         ("selection", "lr3e3", 0, 1.02500), ("selection", "lr3e3", 1, 1.02500)])
    result = hyperparameter_tuning_select.select(rows)
    assert result["verdict"] == "incomplete_grid"
    assert set(result["grid"]["missing_arms"]) == {
        "lr3e4", "lr3e4_warm", "lr1e3_warm", "lr1e3_warm2k", "lr3e3_warm"}


def test_a_one_seed_arm_is_refused_rather_than_meaned():
    rows = ledger(D10 + full_grid(("selection", "lr3e3", 0, 1.02500), seeds=2))
    result = hyperparameter_tuning_select.select(rows)
    assert result["verdict"] == "incomplete_grid"
    assert result["grid"]["underpowered_arms"] == ["lr3e3"]


def test_selector_cannot_see_claim1():
    """Behaviour, not text. The previous version of this test grepped the source and still
    passed with `from src.analysis.claim1_eval import evaluate` inserted at the top -- the
    docstring's own mention of claim1_eval satisfied one clause and the other only matched a
    spelling nobody would write. Import the module for real, in a clean interpreter, and ask
    what came with it: an import anywhere in the transitive graph shows up in sys.modules."""
    probe = (
        "import sys; import src.analysis.hyperparameter_tuning_select as m;"
        "leaked = [n for n in sys.modules if 'claim1' in n];"
        "print(leaked)"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                         cwd=Path(__file__).resolve().parents[1],
                         env={**os.environ, "PYTHONPATH": "."})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", f"claim 1 reached the selector: {out.stdout}"


def test_incumbent_stands_when_nothing_clears_the_margin():
    rows = ledger(D10 + full_grid(("selection", "lr1e3", 7, 1.02580), ("selection", "lr1e3", 8, 1.02580),
                         ("selection", "lr3e3", 0, 1.02575), ("selection", "lr3e3", 1, 1.02575)))
    result = hyperparameter_tuning_select.select(rows)
    assert result["guard"]["passed"]
    # 5e-5 better is half a seed sd -- real-looking, and noise
    assert result["arms"]["lr3e3"]["margin_in_ses"] < hyperparameter_tuning_select.MARGIN_SES
    assert result["winner"] == "lr1e3"
    assert result["verdict"] == "incumbent_stands"


def test_a_clear_winner_is_promoted():
    rows = ledger(D10 + full_grid(("selection", "lr1e3", 7, 1.02580), ("selection", "lr1e3", 8, 1.02580),
                         ("selection", "lr3e4_warm", 0, 1.02000), ("selection", "lr3e4_warm", 1, 1.02010)))
    result = hyperparameter_tuning_select.select(rows)
    assert result["winner"] == "lr3e4_warm"
    assert result["verdict"] == "tuned_pending_confirmation"
    assert result["arms"]["lr3e4_warm"]["margin_in_ses"] > hyperparameter_tuning_select.MARGIN_SES


def test_a_worse_arm_is_never_promoted():
    rows = ledger(D10 + full_grid(("selection", "lr1e3", 7, 1.02580), ("selection", "lr1e3", 8, 1.02580),
                         ("selection", "lr3e3", 0, 1.19000), ("selection", "lr3e3", 1, 1.19000)))
    result = hyperparameter_tuning_select.select(rows)
    assert result["winner"] == "lr1e3"
    assert result["arms"]["lr3e3"]["margin_in_ses"] < 0


def test_guard_fires_when_the_incumbent_drifts_off_its_own_rebuild_run():
    """A moved tensor build shows up here and nowhere else: every column still lines up."""
    rows = ledger(D10 + full_grid(("selection", "lr1e3", 7, 1.04000), ("selection", "lr1e3", 8, 1.04000)))
    result = hyperparameter_tuning_select.select(rows)
    assert not result["guard"]["passed"]
    assert result["verdict"] == "guard_failed"


def test_missing_incumbent_is_refused_not_defaulted():
    rows = ledger(D10 + [("selection", arm, i, 1.02000) for arm in hyperparameter_tuning_select.EXPECTED_ARMS
                         if arm != hyperparameter_tuning_select.INCUMBENT for i in range(2)])
    with pytest.raises(ValueError, match="incumbent"):
        hyperparameter_tuning_select.select(rows)


def test_noise_floor_needs_more_than_one_seed():
    rows = ledger([("rebuild", "baseline", 0, 1.02584),
                   ("selection", "lr1e3", 7, 1.02580), ("selection", "lr1e3", 8, 1.02580)])
    with pytest.raises(ValueError, match="noise floor"):
        hyperparameter_tuning_select.select(rows)


def test_noise_floor_is_read_from_the_ledger_not_hardcoded():
    tight = ledger(D10 + full_grid(("selection", "lr1e3", 7, 1.02580), ("selection", "lr1e3", 8, 1.02580),
                          ("selection", "lr3e3", 0, 1.02540), ("selection", "lr3e3", 1, 1.02540)))
    wide = ledger([("rebuild", "baseline", i, v) for i, v in
                   enumerate([1.020, 1.026, 1.031, 1.024, 1.029])] +
                  full_grid(("selection", "lr1e3", 7, 1.02580), ("selection", "lr1e3", 8, 1.02580),
                            ("selection", "lr3e3", 0, 1.02540), ("selection", "lr3e3", 1, 1.02540)))
    assert hyperparameter_tuning_select.select(tight)["winner"] == "lr3e3"
    assert hyperparameter_tuning_select.select(wide)["verdict"] == "incumbent_stands"


def test_report_round_trips(tmp_path):
    rows = ledger(D10 + full_grid(("selection", "lr1e3", 7, 1.02580), ("selection", "lr1e3", 8, 1.02580),
                         ("selection", "lr3e3", 0, 1.02000), ("selection", "lr3e3", 1, 1.02000)))
    result = hyperparameter_tuning_select.select(rows)
    json_path, csv_path = hyperparameter_tuning_select.report(result, tmp_path)
    import json
    assert json.loads(json_path.read_text())["winner"] == "lr3e3"
    written = list(csv.DictReader(csv_path.open()))
    assert [r["config"] for r in written][0] == "lr3e3"   # best first
    assert len(written) == len(hyperparameter_tuning_select.EXPECTED_ARMS)
    assert written[0]["promotable"] == "True"


def test_empty_stage_is_refused():
    with pytest.raises(ValueError, match="no completed"):
        hyperparameter_tuning_select.select(ledger(D10))


def test_a_two_seed_promotion_is_flagged_for_confirmation():
    """Five challengers, one incumbent, one threshold: the family-wise false-promotion
    rate is several times the per-comparison one, and a two-seed mean is thin. The
    artifact must say so rather than leaving it to the spec."""
    rows = ledger(D10 + full_grid(("selection", "lr1e3", 7, 1.02580), ("selection", "lr1e3", 8, 1.02580),
                         ("selection", "lr3e3", 0, 1.02000), ("selection", "lr3e3", 1, 1.02000)))
    result = hyperparameter_tuning_select.select(rows)
    assert result["winner"] == "lr3e3"
    assert result["requires_confirmation"] is True
    assert result["verdict"] == "tuned_pending_confirmation"


def test_a_five_seed_promotion_needs_no_confirmation():
    rows = ledger(D10 + full_grid(*[("selection", "lr1e3", i, 1.02580) for i in (7, 8)],
                                  *[("selection", "lr3e3", i, 1.02000) for i in range(5)]))
    result = hyperparameter_tuning_select.select(rows)
    assert result["requires_confirmation"] is False
    assert result["verdict"] == "tuned"


def test_the_incumbent_standing_never_asks_for_confirmation():
    rows = ledger(D10 + full_grid(("selection", "lr1e3", 7, 1.02580), ("selection", "lr1e3", 8, 1.02580),
                         ("selection", "lr3e3", 0, 1.02579), ("selection", "lr3e3", 1, 1.02579)))
    result = hyperparameter_tuning_select.select(rows)
    assert result["verdict"] == "incumbent_stands"
    assert result["requires_confirmation"] is False


# ------------------------------------------------- the guard, and what it can actually see

def test_an_undeclared_arm_is_refused_rather_than_ranked():
    """The grid check is symmetric. Missing arms catch a night that ran short; unexpected
    arms catch an arm appended after the table was read, which is the failure that actually
    voids a pre-registration. Without this the injected arm wins and `complete` reads true."""
    rows = ledger(D10 + full_grid(("selection", "lr9e9_undeclared", 0, 1.02000),
                                  ("selection", "lr9e9_undeclared", 1, 1.02000)))
    result = hyperparameter_tuning_select.select(rows)
    assert result["grid"]["unexpected_arms"] == ["lr9e9_undeclared"]
    assert result["grid"]["complete"] is False
    assert result["verdict"] == "incomplete_grid"


def test_drift_is_flagged_uninformative_when_the_incumbent_reruns_floor_seeds():
    """The real selection case. `selection/lr1e3` seeds 0-1 reproduce `rebuild/baseline` seeds 0-1 exactly, so
    the drift is an algebraic function of the floor sample and says nothing about the build.
    The guard must pass on the REPRODUCTION and say the drift is not evidence."""
    rows = ledger(D10 + full_grid(("selection", "lr1e3", 0, 1.02584), ("selection", "lr1e3", 1, 1.02585)))
    guard = hyperparameter_tuning_select.select(rows)["guard"]
    assert guard["shared_seeds"] == [0, 1]
    assert guard["drift_is_informative"] is False
    assert guard["basis"] == "exact reproduction on shared seeds"
    assert guard["reproduces_shared_seeds"] is True
    assert guard["passed"] is True


def test_a_shared_seed_that_does_not_reproduce_fails_the_guard():
    """The units failure this exists for: the same recipe and seed on a build with different
    quality-bin edges cannot land on the floor's value. A drift small enough to sit inside
    the 4 sd tolerance must still fail, which the old tolerance-only guard let through."""
    rows = ledger(D10 + full_grid(("selection", "lr1e3", 0, 1.02584 + 1e-5),
                                  ("selection", "lr1e3", 1, 1.02585)))
    result = hyperparameter_tuning_select.select(rows)
    assert result["guard"]["mismatched_seeds"] == [0]
    assert result["guard"]["passed"] is False
    assert result["verdict"] == "guard_failed"
    # and the tolerance alone would have waved it through
    assert result["guard"]["drift"] < result["guard"]["tolerance"]


def test_without_shared_seeds_the_guard_falls_back_to_the_tolerance():
    """An incumbent trained on fresh seeds has nothing to reproduce, so drift is all there
    is -- and it is then genuinely informative."""
    rows = ledger(D10 + full_grid(("selection", "lr1e3", 7, 1.02584), ("selection", "lr1e3", 8, 1.02585)))
    guard = hyperparameter_tuning_select.select(rows)["guard"]
    assert guard["shared_seeds"] == []
    assert guard["drift_is_informative"] is True
    assert guard["basis"] == "drift within tolerance"
    assert guard["passed"] is True


# ------------------------------------------- the ml-engineer gates, on the warm path

def _warm_run(steps=400, warmup_steps=50, seed=3):
    """One overfit-a-single-batch run with the warmup schedule actually in the loop.

    A scheduler cannot change tensor shapes, the loss at init, the split boundary, or the
    decode -- those gates are properties of the model and the build, and D.4 discharged
    them. What a scheduler CAN break is the two gates below: a wrong scale can stop the
    run learning at all, and a scheduler carrying state can make two same-seed runs
    diverge. So those two are re-run here and the other five are not."""
    from src.model.v1 import HitterEmbeddingV1, factorized_loss
    from tests.test_model_v1 import make_batch, N_CONTEXT, N_HITTERS

    torch.manual_seed(seed)
    model = HitterEmbeddingV1(N_HITTERS, N_CONTEXT, dropout=0.0)
    hitter, context, labels = make_batch(64, seed=seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    warmup = train.LinearWarmup(optimizer, steps=warmup_steps, base_lr=1e-2)

    losses = []
    for _ in range(steps):
        outputs = model(hitter, context, labels["ev"], labels["la"])
        loss, _ = factorized_loss(outputs, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        warmup.step()
        optimizer.step()
        losses.append(loss.item())
    return losses


def test_the_warm_path_still_overfits_one_batch():
    losses = _warm_run()
    assert losses[-1] < losses[0] * 0.05, \
        f"loss only fell from {losses[0]:.1f} to {losses[-1]:.1f} with warmup in the loop"


def test_two_warm_runs_at_the_same_seed_are_bit_identical():
    assert _warm_run() == _warm_run()
