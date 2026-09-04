# Handoff — evaluation-chain pass on the canonical SGD build

Written 2026-09-03 at the close of the post-V pass. Input for `/fable-orchestrate`.

## State
- Canonical build: `embedding_sgd_sgd_lr1_s0..s4` (decision log 2026-09-03). Gate passed
  on all three conditions; `reference` 1.02386 vs incumbent 1.02580.
- Phase V regenerated on it. V.4 (`level_query.json`) and V.10 (`disagreement_*`) still read
  `results/model_v1/model_v1_predictions_rebuild_baseline.csv`, the OLD arm's predictions.
- Main exhibit, notebook 07, `results/model_evaluation/` all point at the old build
  (post-v-spec §0.7 defers them to this pass).
- Cold-start prior: diagnostic separates; substitution eval done; prior is opt-in.
  Adoption is Nate's Tier 3 call — do NOT decide it.
- Tree clean at HEAD. 528 tests pass.

## Scope of this pass, in order
1. Rerun the evaluation chain on the canonical build: predictions file → main exhibit →
   notebook 07 → `results/model_evaluation/`. Then re-run V.4 and V.10 so every Phase V
   artifact reads the same build. One decision-log entry for the chain rerun.
2. Report the claim-1 numbers old vs new in the lab notebook. If the cold-start substitution
   eval needs the new claim-1 frame, rerun it; leave adoption to Nate.
3. Screen `--embedding-lr` above 1.0 (3.0 and 10.0, one seed each, stage `embedding_sgd`).
   Config-only, no ML gate. Needs a decision-log entry BEFORE launch stating the question
   (is 1.0 a peak or a floor) and that the canonical build does not move on a one-seed result.
4. Ruf-era anchor reruns per the 2026-09-02 anchor entry, if any V.5 artifact still names Ruf.
5. Run `/research-review` on Phase V + post-V (spec §4 requires the verdict before the
   2025 refit). Record the verdict in the lab notebook.

## Out of scope
Items 9–12 (`docs/post-v.md`): 9–10 wait on the 2025 refit, 11 gets its own grill,
12 is Nate's. No model, loss, or pipeline change. No decision on the cold-start default.

## Rules that bind
- `docs/research-manifest.md` frozen rules; `CLAUDE.md` blocking-question list.
- Smoke runs to the scratchpad only; `results/` and `data/processed/` written only by the
  named regeneration steps above.
- Decision-log entries: five fields, STE, no narration. Lab-notebook entry at the end.
- `~/personal-os/projects/hitter-embedding/README.md` per `~/os/clients/_TEMPLATE.md`
  when the next step moves. That repo has uncommitted edits from this pass; commit them.
- Four stale worktrees under `.claude/worktrees` (two locked). Unlock and prune only after
  confirming each has no uncommitted work.

## Known traps
- `embedding_norm_check.py --seeds` takes seed ints, not a count.
- Its norm-sd proxy prints FAIL on the passing build by construction; not a gate input.
- Full pytest takes ~15 min alongside an eval; run it in the background.
- A full 2024 ensemble query pass is ~1.3 h; the chain has at least one.
