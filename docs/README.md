# Project Documentation

This folder contains references and links to the canonical project documents.

## Canonical Locations

- **Layer 1 Architecture Plan (v2):** `~/os/knowledge/library/baseball-research/Layer1_Architecture_Plan_v2.md`
  - Full architecture specification, build order (Phases A–G), decision log (§5.13), evaluation protocol
  - Authority document for all modeling decisions

- **Project Handoff (v2):** `~/os/knowledge/library/baseball-research/`
  - Cross-layer context, Layer 2 scope, open items, literature gaps
  - Framing and project-level decisions

- **Project Index:** `~/os/projects/hitter-embedding/README.md`
  - Phase status, non-negotiables, build order summary
  - Cross-layer open items and standing risks

## Reference Research

Every external source this project cites is shelved under `~/os/knowledge/library/`,
one file per document, on one of two shelves:

- `baseball-research/` — domain findings, measurements, and conventions. Player
  embeddings (Alcorn 2018, Heaton 2022/2023), pitch sequencing and fielding
  (Melville 2023/2024), the bat-tracking measurement confound (Powers & Yurko
  2025), strike-zone provenance, foul-ball and count-state work, and the
  sabermetric reliability posts behind the stabilization numbers.
- `ml-research/` — techniques, estimators, and statistical methods. Optimizer
  and warmup precedent (Loshchilov 2019, Goyal 2017, Liu 2020, Ma & Yarats 2021,
  Duchi 2011, Kunstner 2024, Qiu 2025), scoring rules and ensembles (Gneiting &
  Raftery 2007, Lakshminarayanan 2017), and the reliability/attenuation lineage
  (Spearman 1904, Franks 2016, Brown 2008, Revelle ch. 7).

Each entry states what it establishes, what it does not, the verbatim passage
this project leans on, and which decision-log entries cite it. The shelving rule
is `~/os/knowledge/frameworks/library-policy.md`; `ls` the two folders for the
current list rather than trusting this summary.

Cited but unreachable, so deliberately unshelved: Efron & Morris (1972),
*Biometrika* 59(2); Lord & Novick (1968); *The Book* p.157. Each is flagged in
the decision log at the entry that cites it.

## Working Documents (Session-Specific)

Add session-specific notes here as they emerge (exploratory results, feature engineering notes, etc.):

- `feature_analysis.md` — Phase B feature-value results
- `baseline_results.md` — Phase C incumbent model performance
- `training_log.md` — v1 model training runs and ablations
- `evaluation_checkpoints.md` — Probe, claim-1, and composition validation results

(These are templates — create as needed during build.)
