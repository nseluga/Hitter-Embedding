# Pass A1 — C.3-full refit + review fixes

Prompt for a fresh opus high-effort window. First action: invoke `/research-partner hitter-embedding` and confirm the standup block. This file is the pass scope; the decision log remains the authority.

## Scope — exactly these tasks, nothing else

1. **C.3-full refit.** Run the supported path at `c_report.py:364`. Fold C.3-full into both existing tables (`m1_differential_scores.csv`, `m2_stratum_ceiling.csv`) with every opponent plus intervals. CPU, minutes–1 h — run it in this pass, do not defer.
2. **`min_eval_pa` sensitivity** at 5/10/25, reported as a sensitivity; the population is unchanged (2026-08-31 decision).
3. **Rename** `ceiling_range` → `ceiling_range_finite_only` everywhere it appears, tests included.
4. **Three write-up sentences** in the appropriate docs: M.3's route was not pre-registered; the 32 dropped low-wOBA groups inflate Route A's τ²; Route C's retirement is a deviation from the pre-registered branch.
5. **Prepare tonight's warmup launch**: one command for the o-ledger training path with lr 1e-3, warmup 2,000 steps, seeds 0–4. Config-only, so the ML verification gate is exempt. Print the command and expected wall-clock (~1.5 h GPU); do not launch it.

## Forbidden

No model, loss, or pipeline code changes. No new analysis beyond the sensitivity. The exhibit pipeline is Pass A2 — do not start it.

## Done when

- Both tables carry C.3-full with intervals; sensitivity artifact in `results/`.
- All tests pass, exit 0 (441 at last count).
- Decision-log entries (five-field format) for anything settled.
- Warmup launch command printed for Nate.
