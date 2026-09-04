# Handoff — hitter-embedding: build the abstract pipeline (Opus, code only, no long runs)
Repo `~/hitter-embedding`, main 634d9a8+. Read first: `docs/research-manifest.md`, `docs/decision-log.md` last two
entries (2026-09-04 prior adopted; refit design), `~/.claude/skills/ml-engineer/SKILL.md` gates. Rules: no model/loss change; frozen split untouched; 2025 sealed; smoke runs to /tmp only; one heavy job at a time (8 GB).

## Just done
- Chain + pins on canonical `embedding_sgd_sgd_lr1_s0..s4`; review 7/10. Nate: prior adopted, refit design ok, grill waived (logged).

## Build (four items, in this order)
1. Prior default on: `src/model/query.py` `cold_start_prior` default = `ensemble_cold_start_prior` (see
   `src/analysis/cold_start_prior_eval.py:55`); every chain script/query CLI passes it; `eb_bivariate` gets
   `eb_debut_mu` where compared to the model. Add `--no-cold-start-prior` opt-out. Update tests.
2. Refit code per the log entry: `split_config` schema accepts `final_run: true` with `val: []` (validator in
   `src/config/splits.py:23`); `loader.split_indices` optional split arg; `train.py fit()` fixed `--step-budget`
   with early stop + `ReduceLROnPlateau` off and lr cuts replayed at given step fractions (`--lr-cut-steps`).
   Budget = median best step over `sgd_lr1` s0–s4 (`results/model_v1/sweep_log.csv` has best_epoch, NOT cut
   epochs — recover cut epochs from run logs/checkpoints or add a ledger column; resolve the log's `unverified`).
3. No-decay ablation config: `sweep.py` stage `embedding_sgd_nodecay`, `weight_decay=0` on the table group only.
4. `scripts/overnight.sh`: serial: chain (prior on) → nodecay s0 + norm check → replay check on frozen split
   (1 seed, val off) → GATE `reference` within 1.02386 ± 0.00018 (2 SD) else exit 1 → refit 5 seeds 2015–2024
   (`final_run` config in a NEW file) → 2025 ensemble + per-seed queries → 2025 chain. Chain order: copy
   `/tmp/hitter-eval-chain/run_queries.sh`, `run_chain.sh`, `run_post.sh`, `run_post2.sh`. Log to /tmp/hitter-overnight/.

## Verification (blocking before handing to the run)
- ML gates (pipeline changed): shape asserts, one-batch overfit, loss-scale, determinism 2 seeds, split-boundary
  assert (no 2025 row in any train/val tensor under both configs), eval-mode hygiene. Keep as pytest.
- `fit()` with budget = a run's actual best step and replayed cuts, on the frozen split, 200 steps smoke: loss
  trajectory bit-identical to the same steps of the normal path (proves the replay is a no-op).
- Full `pytest` green (was 529). Fresh Sonnet subagent adversarially reviews `overnight.sh` + configs (path
  mismatches, wrong data-dir `phase_d5`, gate logic, 2025 leakage). Dry-run the script with `DRY=1` echo mode.
- Commit code (no heavy launches). Next: hand `docs/handoff-overnight-run.md` to the run session.
