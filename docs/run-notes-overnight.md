# Overnight run notes — 2026-09-04

Launched 20:00. Watchdog session. Stage ETAs from handoff: chain ~5 h, nodecay ~30 m,
replay ~30 m, refit 5x~25 m, 2025 queries+chain ~5 h.

## Stage 1 — 2024 chain, prior on (in progress)

Memory pressure observed during the ensemble query (`src.model.query --seeds 0 1 2 3 4`):

| wall | query RSS | swap used / total | note |
|------|-----------|-------------------|------|
| 12 m | 6.5 GB    | -                 | LHB vs LHP block, ~122 s per 64 pitchers |
| 42 m | 12.6 GB   | 10.6 / 12.3 GB    | RHB vs LHP block, ~590 s per 64 pitchers |

RSS grows monotonically across platoon blocks; the 4.8x slowdown from the LHB block to the
RHB block is larger than the 2.0x hitter-count ratio (210 -> 423), so the excess is swap.
Single heavy process throughout — the handoff's "two heavy procs, kill the newer" rule did
not fire. No action taken: memory pressure is not a script/path/flag/env fault.

Block pacing (ensemble query, seeds 0-4):

| block | grid | wall |
|-------|------|------|
| LHB vs LHP | 210 x 540 | 1159 s |
| RHB vs LHP | 423 x 544 | 2536 s |
| LHB vs RHP | 287 x 1475 | started 3715 s |

Time scales with the hitter x pitcher grid, so the two RHP blocks project to ~78 min and
~115 min: ensemble query ~4.2 h against a whole-stage ETA of ~5 h, before the five per-seed
queries and the 14-module chain. Schedule risk flagged, not acted on.

## 2026-09-05 — completion (Claude, interactive; watchdog died 21:32 on session limit)

Stages 1–5 finished unattended by 06:44. Stage 6 crashed on its 3rd module; four resumes (`FROM=6`) needed:
1. `take_mass` reads `draw_price` summary, which `price_draw` writes later; `price_draw` reads the resampler audit no stage-6 module wrote. Fixed order: eval, swing, resample, price_draw, take_mass. (2024 dir had both files from earlier runs, so stage 1 never hit it.)
2. `probe_coverage` (E.14) asserts against `--final-run` by design → dropped from stage 6.
3. `bip_value` (E.13) hard-pins the 2024 E.3 constants (`E3_VALUE_*_BIP`) and asserts the report matches → cannot run on 2025 without a code change. Dropped; **Tier 2 leftover**.
4. `process_calibration_heads/process` assert eval-build bin edges == `phase_d5`; refit arm trained on `phase_d5_final` whose quantile edges shifted (ev 0.1, la 1.0, spray 0.18). Guard now references the build recorded in the arm's checkpoint `args.data_dir` (`provenance.trained_data_dir`); test added. Analysis-guard change only.
Stage 6 done 12:32. No 2025 stage was re-run except the deterministic stage-6 analyses over the same predictions.

### Numbers
- Replay gate: reference 1.02393 vs 1.02386 ± 0.00018 → PASSED (miss 0.00007). Refit ran on the gate alone.
- No-decay arm (seed 0): slope **+2.08** [2.00, 2.16] vs +0.438 with decay; reference 1.02618 (−3.3 SE vs incumbent). Removing decay steepens the slope → slope comes from step size; decay was damping it. Norm-check FAIL is the reference bar, advisory.
- 2024 chain, prior on (canonical arm): low RMSE 0.05860 (rank 0.247), medium 0.04712 (rank 0.351), high 0.04268. vs eb_bivariate low −0.0032 [−0.0073, +0.0010]; medium both gates pass vs eb_bivariate. Fidelity bb +7.2 % (matched 0.0831 vs modelled 0.0862), k −0.9 %.
- **2025 refit (sealed, one pass):** claim-1 gates False (low RMSE vs gbm_full +0.0014 [−0.0014, +0.0042]; low rank Δ +0.026 [−0.089, +0.141]). Medium passes both gates vs both baselines (RMSE −0.0022 [−0.0045, −0.0001] vs gbm_full; −0.0036 vs eb_bivariate; rank +0.141 [0.014, 0.264] vs gbm_full). All-strata rank Δ +0.048 [0.008, 0.090] vs gbm_full, +0.066 [0.012, 0.124] vs eb_bivariate (RMSE n.s.). High: n.s. both ways.
- 2025 fidelity: bb modelled 0.0851 vs train-matched 0.0812 (+4.8 %, fail at 2 %), k 0.2250 vs eval-matched 0.2219 (+1.4 %, pass).
- 2025 draw price: channel 1 −0.00038, channel 2 +0.0028, both +0.0024 of residual 0.0039 (62 %).
- Outputs: `results/model_evaluation_final/` (27 files), `results/process_calibration_final/` (6 files), refit checkpoints `embedding_sgd_sgd_lr1_final_s0..4.pt`, `data/processed/phase_d5_final/`.
