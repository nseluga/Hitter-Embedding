# Decision Log — Hitter Embedding

Append-only. Format fixed by `~/os/knowledge/frameworks/research-standards.md`
§4 — `/research-partner` and `/research-review` both parse against it.

Seeded from Architecture Plan §5.13 (approved by Nate, July 2026).

---

## 2026-07-12 — Conditional-query reframe as organizing principle

- **Decision:** p(process outcome | hitter, pitch context) adopted as the core framing; supersedes "archetype emergence" as the mechanism description. Archetypes remain a testable secondary hypothesis with a concrete fallback (platoon-skill direction in embedding space).
- **Alternatives:** Global embedding trained on a generic objective hoping platoon-relevant structure emerges as a by-product.
- **Rationale:** Global objectives impose no pressure for conditional structure to be separable. Explicit conditioning makes platoon skill a directly computable query, not a hoped-for emergent property. The embedding's role narrows to what it can be relied on: parameter sharing across hitters.
- **Reference:** Architecture Plan §5.1, §5.3; Appendix ("Why conditional-query, not global embedding?")
- **Tier:** 1 (frozen)
- **Revisit if:** A global objective is found that explicitly forces context-conditional structure to be separable at the granularity the platoon query requires.

---

## 2026-07-12 — Count state included; baserunners/score/inning excluded

- **Decision:** Count (0-0 through 3-2) is the within-PA context. Baserunners, score, and inning are excluded.
- **Alternatives:** Full game-state conditioning; no within-PA context at all; pitch-sequence conditioning.
- **Rationale:** Count drives swing intent and is within the hitter's process (Powers & Yurko). Game state is hitter-uncontrollable and introduces managerial selection confounds (pinch-hitting, matchup deployment) that distort the process estimate.
- **Reference:** Architecture Plan §5.1, §5.3; Powers & Yurko (arXiv:2507.01238)
- **Tier:** 1 (frozen)
- **Revisit if:** Evidence emerges that game-state conditioning improves process separation without introducing selection confounds.

---

## 2026-07-12 — Characteristics-first pitcher representation; ID-residual gated

- **Decision:** Pitcher represented by characteristics vector (pitch type, velocity, spin, movement, plate location, release point/extension, handedness). A per-pitcher ID residual embedding is added only if a specific diagnostic fires: the characteristics-only model systematically under-predicts observed platoon splits of high-exposure hitters.
- **Alternatives:** Primary pitcher-ID embedding (small learned vector); characteristics only with no residual path.
- **Rationale:** Finer-platoon queries require characteristics conditioning (otherwise query space collapses to just handedness). Primary ID embedding reintroduces pitcher-side small-sample sparsity and breaks generalization to unseen pitchers. Residual covers deception/sequencing effects if the diagnostic fires.
- **Reference:** Architecture Plan §5.6, §5.8
- **Tier:** 1 (frozen; residual path is Tier 2 gated on the named diagnostic)
- **Revisit if:** Characteristics-only model passes the diagnostic — in that case, the residual gate never fires and this stands as-is.

---

## 2026-07-12 — Distributional contact-quality head via autoregressive factorization

- **Decision:** Contact quality predicted as a joint distribution over {EV, LA, launch direction, batted-ball spin angle, batted-ball spin rate} using autoregressive factorization (~20–40 bins per factor). Point regression rejected. Surviving outcome dimensions decided by ablation on the claim-1 metric. Spin fields contingent on public Statcast availability; excluded if unavailable (Sloan open-source requirement).
- **Alternatives:** Point regression (documented failure in closest precedent); monolithic joint softmax (20^5 = 3.2M classes, untrainable); mixture density network (fallback if autoregressive underperforms).
- **Rationale:** Point regression documented failure in the (batter|pitcher)2vec precedent. Full joint grid is combinatorially untrainable. Autoregressive factorization applies the same chain-rule decomposition as language models: each factor head is tractable (~20–40 classes), the full joint is recoverable, and no outcome luck enters.
- **Reference:** Architecture Plan §5.5; Alcorn 2018 (batter2vec, point regression failure); Architecture Plan Appendix
- **Tier:** 1 (head design frozen; outcome dimensions Tier 2 / ablation-decided)
- **Revisit if:** Ablation shows mixture density network outperforms autoregressive factorization on claim-1 metric (this is the named fallback).

---

## 2026-07-12 — Deep ensembles (5 seeds) for v1 uncertainty

- **Decision:** 5-seed deep ensemble. Prediction = ensemble mean; uncertainty = spread across seeds. Naturally widens for low-exposure hitters.
- **Alternatives:** Hierarchical-Bayes hybrid (named fallback if calibration fails); MC dropout; conformal prediction.
- **Rationale:** Simplest reliable method for calibrated per-hitter error bars. Layer 2's error-bar gating on candidacy decisions requires them. Hierarchical-Bayes hybrid is a significant rebuild — kept as fallback pending calibration check (§5.11.3).
- **Reference:** Architecture Plan §5.7; Lakshminarayanan et al. 2017 — unverified, parametric
- **Tier:** 2 (empirically resolvable; calibration check in §5.11.3 is the gate)
- **Revisit if:** Calibration reliability diagrams on very-low-exposure hitters show systematic miscalibration — triggers the hierarchical-Bayes fallback.

---

## 2026-07-12 — Phase B process decides bat-tracking placement

- **Decision:** Whether bat-tracking features (bat speed, attack angle) enter v1 directly vs. requiring the v2 history encoder is decided empirically by Phase B ablations, not assumed.
- **Alternatives:** Include bat-tracking from the start; exclude it entirely; decide by analogy to prior literature.
- **Rationale:** Feature value vs. history-depth requirement is an empirical question. Ablations in Phase B (common-window comparison, window trade test) are the cheapest way to answer it without committing the history-encoder build prematurely.
- **Reference:** Architecture Plan §5.10 Phase B; §5.8 Upgrade 1
- **Tier:** 2 (ablation-decided; Phase B ablations are the decision mechanism)
- **Revisit if:** Never — the ablation is the decision.

---

## 2026-07-12 — Contrastive learning and masked-event pretraining deferred

- **Decision:** Contrastive learning formally deferred in all roles (primary objective and auxiliary). Masked-event pretraining (Heaton & Mitra MGM-style) and within-PA sequence encoders deferred to future work. v2 scope = history encoder + pitcher-ID residual only.
- **Alternatives:** Contrastive as primary objective; contrastive as auxiliary geometry-shaping loss; MGM-style masked pretraining as Phase D/E addition.
- **Rationale:** As primary objective, contrastive trades calibration for similarity geometry — incompatible with the conditional-query framing. As auxiliary, it risks manufacturing the archetype-cluster structure the project intends to test, tainting the evaluation. Masked pretraining is viable but adds scope risk without a clear gate; deferred alongside sequence encoders, their natural container.
- **Reference:** Architecture Plan §5.8, §5.9; Heaton & Mitra 2023 (MGM)
- **Tier:** 1 (deferred, not permanently rejected)
- **Revisit if:** Claim-1 test and archetype hypothesis test are both complete; contrastive or masked pretraining could enter post-hoc without tainting the primary evaluation.
