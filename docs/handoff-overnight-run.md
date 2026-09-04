# Handoff — hitter-embedding: oversee the overnight run (monitor, don't build)
Repo `~/hitter-embedding`. Precondition: build handoff (`docs/handoff-build-abstract-pipeline.md`) done, committed,
tests green, `scripts/overnight.sh` reviewed. Read `docs/decision-log.md` last three entries and
`docs/run-notes-eval-chain-pass.md` (last night's traps: serial jobs only, orphaned trainer children,
`--data-dir data/processed/phase_d5` everywhere, pgrep self-match).

## Just done
- Code for prior-on chain, no-decay ablation, replay check, refit, 2025 chain built and gated.
- Nate's calls: step 4 (refit, unseals 2025) fires on the replay gate alone — unless he says otherwise at launch.

## What the script does that the design didn't say
- Stage 4 builds a NEW tensor dir `data/processed/phase_d5_final` (train 2015-2024) and a new
  `results/model_visualization_final/hitter_stats.csv`; the refit cannot run on `phase_d5` (2024 debutants
  would route to the padding row). `reference` is therefore in different units after stage 4 — the gate is
  before it, on `phase_d5`, deliberately.
- Stage 6 writes to `results/model_evaluation_final/` and `results/process_calibration_final/`. These modules
  write arm-less filenames, so the default dirs would overwrite the committed 2024 exhibit in place.
- `measurement_ceiling_report`, `model_evaluation_platoon_ceiling`, `model_visualization_heads` and
  `_disagreement` are NOT in stage 6; they stay 2024 artifacts (the first depends on the second, which cannot
  be pointed at 2025).
- Stage 3's gate writes `/tmp/hitter-overnight/gate_passed`; stage 4 refuses to start without it. `FROM=4`
  alone will not get past that — `GATE_OVERRIDE=1` is the deliberate way, and only after a gate that passed.

## Run
- Launch `scripts/overnight.sh` with nohup; ~13 h serial. Monitor with the Monitor tool on pid + log tail, not
  polling loops. Check swap (`vm_stat`); if two heavy procs appear, kill the newer, resume from the stage script.
- Stage ETAs: chain ~5 h, nodecay ~30 min, replay ~30 min, refit 5×~25 min, 2025 queries+chain ~5 h.
- If the replay GATE fails: stop. Do not run the refit. Log the miss (reference value, spread) and stop for Nate.
- After each stage: write the numbers to `docs/run-notes-overnight.md` (untracked ok) — nothing lives only in chat.
- After all stages: fresh Sonnet verifier: stale arm names, pins reproduce, 2025 read only by refit/query stages,
  row counts. Then full pytest. Then commit results + one pins entry + lab-notebook entry (Did/Why/Found/Learned/Next).
- Do NOT: change model/loss/pipeline code, rerun 2025 a second time, decide anything Tier 3.

## Report to Nate (morning)
nodecay slope vs +0.438 (mechanism: step size or decay); replay gate value; refit 2025 claim-1 by stratum vs
`eb_bivariate`/`gbm_full`, prior-on 2024 chain deltas, fidelity bb/k; three Tier 2 leftovers if any.
## Next
`/research-review` on the refit, then abstract draft. Update `~/personal-os/projects/hitter-embedding/README.md`.
