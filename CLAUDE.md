# Hitter Embedding — CLAUDE.md

## Key Documents
- **Research manifest:** [`docs/research-manifest.md`](docs/research-manifest.md) — frozen rules, architecture decisions, working style
- **Architecture spec:** `~/os/knowledge/library/baseball-research/Layer1_Architecture_Plan_v2.md`
- **Project index:** `~/os/projects/hitter-embedding/README.md`

## Project Structure
```
hitter-embedding/
├── src/
│   ├── data/          # data pull, cleaning, labeling, eval targets
│   ├── features/      # context feature engineering
│   ├── config/        # frozen split config, woba weights
│   └── analysis/      # stabilization and other analysis utilities
├── tests/             # pytest unit tests for all src modules
├── notebooks/         # exploratory notebooks (numbered, named)
├── data/
│   ├── raw/statcast/  # raw Statcast pulls (gitignored)
│   └── processed/     # cleaned parquets and reports
├── results/           # ablation outputs and evaluation results
└── docs/              # decision log, lab notebook, research manifest
```

## Stack & Compute
- **Stack:** PyTorch, W&B or MLflow, seeded config-driven runs, pytest
- **Scale:** <1M parameters, 7–8M pitches max, no distributed training
- **Compute budget:** <$200 total (Colab Pro ~$10–12/mo or A10/T4 ~$0.30–0.80/hr)

## ML Verification Gate
Any session (interactive, dt-*, or autonomous) that **changes model architecture, loss code, or the data pipeline** must pass the blocking verification gates in `~/.claude/skills/ml-engineer/SKILL.md` before launching a real training run or recording an ablation result. Config-only changes (learning rate, batch size) are exempt.

## Blocking Questions
If you encounter a decision not explicitly covered in the architecture plan or decision log, **stop and ask** rather than proceeding. Common candidates: feature engineering not in Phase B, loss function changes, scope expansion, new uncertainty quantification methods, interpretability additions.

## Session Startup
1. Check [`docs/research-manifest.md`](docs/research-manifest.md) for frozen rules and current architecture
2. Check `~/os/projects/hitter-embedding/README.md` for current phase status
3. For research/build sessions, invoke `/research-partner hitter-embedding`
