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

See `~/os/knowledge/library/baseball-research/Layer1_Architecture_Plan_v2.md` for the canonical architecture specification and build order. Session-facing documents live in `docs/`.

For full project context (literature references, handoff document, feature details), see `~/os/knowledge/library/baseball-research/`.

## Key Resources

- **Architecture:** `~/os/knowledge/library/baseball-research/Layer1_Architecture_Plan_v2.md` (frozen decision log)
- **Research manifest:** `docs/research-manifest.md` (config for `/research-partner` and `/research-review`)
- **Decision log:** `docs/decision-log.md`
- **Lab notebook:** `docs/lab-notebook.md`
- **Literature:** `~/os/knowledge/library/baseball-research/` (PDFs + summaries)
