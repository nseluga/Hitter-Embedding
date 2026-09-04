# Kickoff spec + running notes — eval-chain pass on embedding_sgd_sgd_lr1 (2026-09-03)

## Goal / pass condition
Every Phase E/M/V artifact in `results/` reads the canonical build `embedding_sgd_sgd_lr1`:
no script default, notebook, or JSON provenance names `rebuild_baseline` as the model arm
(grep-checkable). Claim-1 old vs new is in the lab notebook. lr 3.0/10.0 screened at one seed
and logged. Ruf 573131 absent from every anchor-naming artifact. personal-os committed, README
next_step moved, stale worktrees pruned. /research-review verdict on Phase V + post-V in the notebook.

## Decisions taken at kickoff (Nate, 2026-09-03)
- Cold-start prior OFF in the chain; substitution eval reruns on the new frame, reported alongside.
- Script defaults flip to `embedding_sgd_sgd_lr1` in one commit (config-only, ML gate exempt).
- Re-execute notebooks 06 and 08. Not 07 (its inputs do not change).

## Guardrails
- No model, loss, or pipeline code change. No frozen-split edit. No cold-start adoption decision.
- `results/` and `data/processed/` written only by the named regeneration steps; smoke runs go to /tmp.
- Canonical build does not move on a one-seed lr screen. Log entry BEFORE launching the screen.
- Decision-log entries: five fields, STE. Lab-notebook entry at the end. No session narration in either.

## Autonomy
Proceed without asking. Interrupt only for: a gate/assert failure I cannot explain, a frozen-rule
conflict, or a Tier 3 call not already taken above.

## Verification
Fresh-context verifier after step 1 (grep for stale arm names + spot-check three regenerated
numbers against the scripts that wrote them) and before declaring done (research-review is the
final verifier). Workers: none needed — the chain is serial on the predictions file.

## Progress grounding
Every "done" below cites a file path or command output from this session.

## Running notes (corrections + confirmed approaches only)
- `query.py` DEFAULT_DATA_DIR is `phase_d`; the canonical arm trained on `phase_d5`. The two manifests differ only in `quality_bin_edges`, so the wrong dir decodes contact bins silently. Every query and every script that reads the manifest for bin edges must get `--data-dir data/processed/phase_d5`.
- main called `eb_bivariate_eb.*` in `model_evaluation_probe_coverage.py` and `model_v1_level.py` with only `baseline_ladder_bivariate_eb` imported (NameError at run time). The fix sat uncommitted in the `clever-hertz` worktree; applied to main with its regression test.
- Ensemble query launched 17:5x, log `/tmp/hitter-eval-chain/query_ensemble.log`.
- Defaults flipped and NameError fix committed as fc451e6. Worktrees pruned (all four: three squash-merged per `git cherry`, clever-hertz's only diff applied to main).
- lr screen (sgd_lr3, sgd_lr10, seed 0) launched concurrently with the query; log `/tmp/hitter-eval-chain/lr_screen.log`. `STAGES["embedding_sgd"]` carries the two configs uncommitted; revert after the screen.
- Ruf 573131 appears only as an ordinary row in all-hitter tables and as base64 noise in notebook 09 PNGs. No anchor artifact names him; step 4 closes with no rerun.
- `~/os/knowledge/library/INDEX.md` has uncommitted edits from another session (tango, klaassen, stollenwerk lines) that diverge from `~/personal-os`'s copy; not touched here — flag to Nate.
- Per-seed 2024 queries (`--seeds N --label embedding_sgd_sgd_lr1_sN`) are needed by probe_coverage and measurement_ceiling_report; run after the ensemble query.
- 18:03 lr screen + query together thrashed 8 GB (10 GB swap, both procs state U). Killed lr screen at sgd_lr3 s0 partial; relaunch after per-seed queries.
- 18:08 ensemble query never recovered after the thrash (state U, no log progress 12 min); killed and relaunched ensemble + per-seed queries serially via /tmp/hitter-eval-chain/run_queries.sh
- orphaned trainer child (pid 9704, src.model.train from the sweep) survived pkill of the sweep parent and kept eating memory; killed
- 18:10 queries relaunched serial: ensemble (~1.3 h) then 5 per-seed (~30 min each, needed by probe_coverage + measurement_ceiling_report; the ensemble run does not emit per-seed CSVs). ETA all queries ~22:00, then chain. lr screen relaunches after that.
- 22:44 all six queries landed. Fidelity gate (query.py) FAIL on the new build: ensemble bb 0.08620 vs 0.08042 obs (+7.2%), k 0.22852 vs 0.22338 (+2.3%); old build was bb +6.2%, k -1.7%. Per-seed bb 7.0-7.5%. Carry to notebook/log; chain launched 22:45.
- 22:5x chain died at E.13 bip_value: pre-existing argparse mismatch (--e-report vs args.model_evaluation_report, from the presentability pass) + pinned old-build E3 modelled-bip constant; fixed both, resumed from bip_value
- 23:1x measurement_ceiling_report COMMITTED coverage pin 0.855527 -> 0.869452 (only model-dependent pin; routes A/B/B' tau2 unchanged, data-derived). Rerun M report + nb08 after run_post (run_post2.sh). V.4 level_query r_partial pooled 0.56->0.37, low 0.57->0.27 on the new build; process_calibration identity share of gain up on every head (ev 0.35->0.50); fidelity k now passes vs matched/eval, fails vs train_pooled (+2.3%); bb worse (+7.2%).
- 23:2x decision-log chain-rerun entry written. run_post2 auto-chains after run_post (pid-wait watcher). Exhibit reproduction-gate expected values in tests/test_measurement_ceiling_report.py still old-build; update from regenerated differential_exhibit.csv after post2.
- 03:2x post2 done: coverage reproduces_committed True; exhibit reproduction-gate expected values moved to new build (pooled model 0.16263, low 0.17922, medium 0.14392, high 0.12574); 54 M tests pass. Cold-start eval on new frame: model prior rmse 0.06322->0.06245 (diff -0.00077, CI [-0.00119,-0.00020]), rank unchanged; eb debut_mu 0.0921->0.0655. Old frame was 0.064->0.062.
- 03:5x claim-1 old -> new (min_pa sweep arm, PA-weighted RMSE / rank corr weighted): low 0.05923/0.259 -> 0.05831/0.259; medium 0.04921/0.339 -> 0.04712/0.351; high 0.04357/0.489 -> 0.04268/0.481; all 0.04793/0.459 -> 0.04679/0.447. Paired vs eb_bivariate: low RMSE -0.00253 [-0.00484,-0.00025] -> -0.00345 [-0.00741,+0.00057] (interval now spans zero, wider); medium -0.00141 [-0.00424,+0.00194] -> -0.00350 [-0.00591,-0.00088] and rank +0.158 [+0.012,+0.294] (both gates True, only cell); high still loses on RMSE (+0.00163 [+0.00001,+0.00326]). Vs gbm_full: medium -0.00237 [-0.00445,-0.00035] (was -0.00027 n.s.); high +0.00187 (loses). debiased_mean_excess_removed 0.0143 -> 0.0070. Overall claim-1 gates False both builds; decisive stratum low.
- 03:43 lr screen launched alone (sgd_lr3 s0, sgd_lr10 s0; ~39 min each from the sgd_lr1 ledger time). Log /tmp/hitter-eval-chain/lr_screen2.log. Verifier subagent running concurrently (read-only).
- Old-arm files model_v1_*_min_pa_sweep_rebuild_baseline.* still sit in results/model_evaluation next to the new ones (historical; not deleted — Nate's call).
- 04:0x verifier: pins/arm/row-counts PASS; only stale hits were the 8 superseded model_v1_*_min_pa_sweep_rebuild_baseline.* files -> git rm (history keeps them; reversible).
- 04:05 sgd_lr3 s0 done: reference 1.02409 vs sgd_lr1 s0 1.02376 (worse by 0.00033), best epoch 23, 1355 s. Swap 6.5/7.2 GB during training -> norm checks deferred until sgd_lr10 finishes.
- 04:4x sgd_lr10 s0: ref 1.02470, epoch 31. Norm checks: lr3 slope +0.505, lr10 +0.572 (lr1 +0.438). Rates above 1.0 lose reference monotonically; canonical stays; sweep.py reverted; finding logged.
- 05:0x review 7/10, no Tier 1; notebook entry written; screen entry SE arithmetic corrected; cold-start new-frame finding logged.
- 05:1x W5 closed: lr1e-1/1e-2 norm checks regenerated from checkpoints (slopes +0.325/+0.206 reproduce the pick entry), committed. personal-os README f5a1655.
- 2026-09-04 morning: Nate adopted the cold-start prior (Tier 3); refit design entry written (steps, replay, val-off check). Next: flip default + chain rerun.
