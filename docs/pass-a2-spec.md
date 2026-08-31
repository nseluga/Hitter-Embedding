# Pass A2 — the main exhibit pipeline

Prompt for a fresh opus high-effort window. First action: invoke `/research-partner hitter-embedding` and confirm the standup block. Requires Pass A1 merged (C.3-full artifacts exist). Does NOT wait on the warmup arm.

## Scope

1. **Promote the scratch computations into `src/analysis/`.** `boot_m1.py` (paired hitter-level bootstrap on M.1) and `m2_c2.py` (M.2 stratum rescoring with C.2) are currently scratch-only — their numbers do not exist until they are committed, tested, and rerunnable. Reference values to reproduce: model 0.14626 [0.0504, 0.2488], C.2 0.14583 [0.0515, 0.2371], paired diff +0.0012 [−0.079, +0.087], P(model > C.2) = 0.513.
2. **Build the one main table.** Rows: model / C.2 / C.3-full. Columns: pooled + three exposure strata. Each cell carries two fractions — RMSE vs the noise-floor bound (`noise_floor`/`deconvolve` already compute it) and rank vs the Monte Carlo Spearman ceiling (0.30931, not the analytic Pearson 0.29645) — both with paired bootstrap intervals. This replaces `m1_differential_scores.csv` and `m2_stratum_ceiling.csv`.
3. **Stamp the output pre-tuning.** Filename or column marks it as the pre-07 build. One command must regenerate it after any model change.

## Forbidden

No model, loss, or pipeline code changes. No new metrics beyond the two fractions defined by manifest rule 2. No stratum redefinition.

## Done when

- Committed `src/analysis/` scripts reproduce the scratch numbers above; scratch files deleted.
- The table generates from one command; old CSVs replaced, downstream readers updated.
- All tests pass, exit 0. Decision-log entries in five-field format.
