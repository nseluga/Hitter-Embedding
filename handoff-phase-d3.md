# Handoff — hitter-embedding: build Phase D.3 onward

Repo `/Users/nateseluga/hitter-embedding`, branch `main`. Run as `PYTHONPATH=. .venv/bin/python`. 206 tests pass.
Read first: `docs/phase-d-spec.md` (APPROVED — build to it), `docs/decision-log.md` (last 7 entries, all 2026-08-01),
`docs/lab-notebook.md` (2026-08-01), `~/os/knowledge/library/baseball-research/Layer1_Architecture_Plan_v2.md` §1.3, §1.5, §5.
Invoke `/research-partner hitter-embedding` + `/ml-engineer`. D.0–D.2 are closed and signed off.

## What was just done
- **D.0** decisions settled; `src/analysis/d0_checks.py` + `results/phase_d/` (redundancy R², frame structure).
- **Weighted rank correlation** added to `claim1_eval` (wCorr construction); Phase C regenerated on **2024**.
- **D.1** `src/data/model_dataset.py` — built: 7,293,321 pitches, 1,762-hitter vocabulary, 46 context columns,
  24 quantile bins/dimension, 1.6 GB memmapped at `data/processed/phase_d` (gitignored).
- **D.2** `docs/phase-d-spec.md` approved: equations, shapes, loss, optimizer, ~207k params at `d=32`.

## Key decisions and why (non-obvious only)
- **Select on 2024, refit through 2024, report on 2025 — once.** §2.2 already allocated the seasons this way; Phase D is the
  first phase needing a selection budget. The refit is not optional: without it 75% of the 2025 low stratum lands on the
  untrained embedding row vs 42.7% with it. `c_report --final-run` is the deliberate gate; **do not run it early.**
- **Loss is raw sums, not per-head means** (Nate's call, and correct). It is the plain likelihood, and D.5's Markov
  composition must reproduce league run scoring (§5.4) — any re-weighting decalibrates the conditionals. Re-weighting
  variants are D.8 arms, expected null.
- **Batch size and weight decay are ONE setting.** AdamW decays every step; an embedding row updates only when that hitter
  is in the batch. Ratio ≈1:1 at the median, 23.9:1 at the 10th percentile — that ratio *is* v1's shrinkage. Neither is
  swept: tuning either is tuning shrinkage, which 2026-07-30 refused on attribution grounds.
- **Plateau, not cosine** — cosine needs a total step count that early stopping makes unknowable.
- Only `effective_speed` was redundant; the three release features carry the most independent information of B.2's six.

## Next step
1. **D.3** — model to spec §3, then **D.4** verification gates *before* any real run (CLAUDE.md's blocking ML gate applies).
2. **Benchmark one epoch first**, before committing to 5 seeds × N configs. Wall-clock is the only unmeasured quantity.
3. Then **D.5** query machinery (unresolved: sample the quality chain rather than enumerate 24³ — decide and log),
   **D.6** first run + §5.1 probe, **D.7** ensemble, **D.8** ablations (dim sweep, bilinear, B.2 block of five, spray,
   two loss arms) — all scored on the **2024** frame.
4. Still owed: a decision-log entry for the bat-tracking exclusion in Nate's own words.
