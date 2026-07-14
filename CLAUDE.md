# Hitter Embedding — Claude Code Instructions

This file embeds the **non-negotiable rules** and **working style** for all Claude Code sessions on this project.

For full context, architecture specification, decision log, and build order, see:
- **Layer 1 Architecture Plan (v2):** `~/os/knowledge/library/baseball-research/Layer1_Architecture_Plan_v2.md`
- **Project Handoff (v2):** `~/os/knowledge/library/baseball-research/` (full document)
- **Project Index:** `~/os/projects/hitter-embedding/README.md`

---

## Non-Negotiable Rules

### Baseline-First Gate
No deep-learning claim without beating these incumbents **out-of-sample**, especially in **low-exposure strata**:
- Bucketed side-specific trailing averages
- XGBoost with context-interaction features
- Empirical-Bayes platoon regression (The Book)

Beating baselines only on high-exposure hitters (2,000+ PA) is a **null result** for this thesis.

### Ablation-Decided
- Every unclear architectural choice, feature, or outcome dimension is settled by **ablation on the claim-1 metric** (PA-weighted RMSE + rank correlation, stratified by prior exposure)
- Negative results are **kept and reported**, never discarded
- If a finding's ablation status is unclear, flag it before it enters the model

### Frozen Walk-Forward Splits
- Season splits are **frozen in config** before any model comparison
- Never revisit, never use random splits
- Flag any leakage risk immediately (encoder inputs, feature engineering, future-dated information)

### No Luck/Price/Salary Leakage into Layer 1
- Never train on outcome luck (batted-ball quality beyond EV/LA, fielder positioning luck)
- Never train on market prices or salary data
- Team-controlled players are surplus-value **outputs**, never validation targets (avoids monopsony confound and service-time contamination)

### ML Verification Gates
Any session (interactive, dt-*, or autonomous) that changes model architecture, loss code, or the data pipeline must pass the blocking verification gates in `~/.claude/skills/ml-engineer/SKILL.md` before launching a real training run or recording an ablation result. Config-only changes (learning rate, batch size) are exempt.

---

## Working Style (Apply in Every Session)

1. **State open problems plainly.** Never imply something unresolved is solved.
2. **Report search/source provenance.** Cite what was searched and where information came from.
3. **Push back on weak reasoning.** Nate expects genuine pushback and will do the same — default to agreement is not helpful.
4. **Calibrate explanations to ML fundamentals.** Nate knows ML principles but not necessarily exact technique implementations. Explain specific techniques (e.g., autoregressive factorization, deep ensembles) when they first appear.
5. **Precision in terminology matters.** For paper writing: "supervised representation learning," not "unsupervised" (model trains on cross-entropy/regression against labels; embedding structure is emergent, not the training objective).
6. **Record decisions in Nate's own words** where possible, for handoff-document durability.
7. **Honest reporting of non-results is non-negotiable**, on par with baseline-first discipline.

---

## Key Architecture Decisions (Frozen, per §5.13)

These are settled and documented in the decision log; do not re-litigate them:

1. **Conditional-query reframe** (p(process | hitter, pitch context)) adopted as the organizing principle — supersedes "archetype emergence"
2. **Count state only** (no baserunners/score/inning) — count is within-PA process
3. **Characteristics-first pitcher representation** with ID-residual gated on a specific diagnostic
4. **Distributional contact-quality head** (autoregressive factorization), surviving dimensions decided by ablation
5. **Deep ensembles (5 seeds)** for v1 uncertainty; calibration risk logged with named fallback
6. **Phase B process** is the bat-tracking placement decision — feature value vs. history depth is empirical
7. **Contrastive learning formally deferred**; masked-event pretraining also deferred to future work

---

## Quick-Reference Checklist

Before writing or modifying modeling code, confirm:

- [ ] Does this touch the frozen walk-forward split config? Stop and flag rather than modify.
- [ ] Does this feature/architecture choice already have an ablation result, or does one need to be run?
- [ ] Does any input trace back to outcome luck, market price, or salary data? If yes, it does not belong in Layer 1.
- [ ] Is a claim about model performance being made without an out-of-sample comparison against baselines?
- [ ] Are low-exposure strata reported separately, not just aggregate metrics?
- [ ] Any negative/null result — is it being kept and reported, not discarded?

---

## Tools & Compute

- **Stack:** PyTorch, W&B or MLflow, seeded config-driven runs, unit-tested data pipeline
- **Scale:** <1M parameters, 7–8M pitches max, no distributed training
- **Compute budget:** <$200 total (Colab Pro ~$10–12/mo or rented A10/T4 ~$0.30–0.80/hr)
- **Workflow:** Claude Code sessions with this file freezing non-negotiables so agent sessions cannot drift from them

---

## Blocking Questions

If you encounter a decision not explicitly covered in the architecture plan or decision log, **ask Nate rather than proceeding**. Common candidates:
- Feature engineering choices not in Phase B
- Changes to the loss function or training procedure
- Scope expansion (e.g., pitcher-side representation, batter team affiliation)
- Uncertainty quantification methods beyond deep ensembles
- Interpretability/explainability techniques to add

---

## Session Startup

1. Read `~/os/knowledge/library/baseball-research/Layer1_Architecture_Plan_v2.md` in full (or at least §5 — architecture spec and build order)
2. Check `~/os/projects/hitter-embedding/README.md` for the latest Phase status
3. Reference this file for non-negotiable rules before taking any modeling action
