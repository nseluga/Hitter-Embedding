# Hitter Embedding Project

Deep-learning hitter representation model trained on pitch-by-pitch process signals (Layer 1) + platoon-skill and market-value query framework (Layer 2).

**Research question:** Identify and value MLB hitters whose market price is suppressed by deployable hitting weaknesses (specifically platoon splits), using a process-signal embedding model to outperform incumbent methods while recovering real market mispricing.

**Target venue:** MIT Sloan Sports Analytics Conference (SSAC27)  
**Abstract due:** October 1, 2026  
**Paper due:** December 4, 2026 (if selected)

## Project Structure

- `docs/` — Governing documents, architecture plan, research notes
- `src/` — Model code, training pipeline, evaluation
- `data/` — Data processing, feature engineering (Statcast)
- `results/` — Model outputs, checkpoints, evaluation artifacts
- `notebooks/` — Exploratory analysis, visualization

## Documentation

All project documentation lives in `docs/`:

- **`docs/README.md`** — Project overview, references to canonical architecture and research documents, working document templates
- **`docs/decision-log.md`** — Append-only record of all modeling and design decisions with rationale
- **`docs/lab-notebook.md`** — Per-session working notes: what was built, why, what was learned, and next steps

Working documents (feature analysis, baseline results, training logs, evaluation checkpoints) are created in `docs/` as needed during development.
