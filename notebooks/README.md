# Notebooks

Exploratory analysis, visualization, and prototyping (not production code).

## Usage

Keep notebooks for:
- EDA on Statcast data (pitch distributions, hitter/pitcher splits, count-state patterns)
- Feature stabilization analysis (split-half reliability plots)
- Baseline model visualization and error analysis
- Result interpretation and paper figure generation

Move to source code once:
- The approach is finalized (via ablation)
- The code needs to run in production (training pipeline, evaluation)
- The results need to be reproducible from config (not hand-tuned in a notebook)

## Naming Convention

- `phase_X_*.ipynb` — exploratory work for that phase
- `exploratory_*.ipynb` — ad-hoc investigation (label by topic)
- `paper_figure_*.ipynb` — analysis for paper figures/tables (archive once finalized)

## Important Notes

- Notebooks are exploratory and disposable; treat config-driven source code as the authority
- Do not hardcode paths, model seeds, or decision thresholds in notebooks — reference `src/config/` instead
- Comment data availability constraints and versioning (Statcast snapshot date, etc.)
