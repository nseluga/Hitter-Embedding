# Research Manifest — Hitter Embedding

Config for `/research-partner` and `/research-review` (see `~/os/skills/`).
Format governed by `~/os/knowledge/frameworks/research-standards.md`.

## Architecture file
- `~/os/knowledge/library/baseball-research/Layer1_Architecture_Plan_v2.md` — canonical Layer 1 spec; frozen decision log (§5.13), non-negotiable rules (§2)

## Authority documents
- `~/os/knowledge/library/baseball-research/` (Project Handoff v2) — framing, Layer 2 scope, cross-layer open items
- `docs/` (this repo) — Claude Code project context

## Frozen rules
1. **Baseline-first gate:** no deep-learning claim without beating bucketed averages, XGBoost, and empirical-Bayes platoon regression out-of-sample, especially in low-exposure strata
2. **Ablation-decided:** every unclear choice settled by ablation on the claim-1 metric; negative results kept and reported
3. **Frozen walk-forward splits:** season splits frozen in config before any model comparison; flag leakage immediately
4. **No luck/price/salary leakage into Layer 1:** never train on outcome luck or market prices

## Decision log
- Path: `docs/decision-log.md`

## Lab notebook
- Path: `docs/lab-notebook.md`

## Library area
- Path: `~/os/knowledge/library/baseball-research/`

## Domain sanity questions
- What is the baseball mechanism? Would a scout, coach, or front-office analyst believe this explanation — and if not, why not?
- Does the effect survive the way the game is actually managed (platooning, deployment/selection bias, pinch-hitting)?
- Is the market claim about measurement-and-identification, or does it quietly assume team ignorance?

## Review bar
- The processed papers in `~/os/knowledge/library/baseball-research/` — rigorous justification for modeling choices, explicit confound handling, clear evaluation design, honest limitations
- Powers & Yurko (measurement confound), Krautmann (MRP-estimate critique) for the economics layer
