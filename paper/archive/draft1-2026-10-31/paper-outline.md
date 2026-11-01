# SSAC27 full paper — proposed outline (v3: standard structure, Alcorn + Melville moves)

Title: Conditional Queries Against a Learned Hitter Process Model
Numbers: clean build only (`main.tex` comments are the licensed source). `outline.md` in
this folder is the *abstract* outline with stale old-build numbers; do not copy from it.

## Structure

Introduction · Related Work · Methods · Results · Discussion · Conclusion.

How the model papers compare:

| Paper | Sections |
|---|---|
| Alcorn 2018 | Introduction (related work one paragraph inside) · Methods (Data, Model) · Results · Future directions |
| Melville 2023 | Introduction · Related Work · Overview with worked example · components · three game variants · Conclusion |
| Melville 2024 | Introduction (related work two sentences) · method sections with results inline · extension · Conclusion |

Related Work is separate in 4 of the 7 library papers; kept separate here because this
paper cites more than Alcorn did. No library paper has a section called Discussion: its job
is done by "Future directions" (Alcorn) or the Conclusion (Melville). Here Discussion holds
practitioner use + limits + future directions, and the Conclusion stays one paragraph.

## Moves borrowed inside that structure

| Move | Source | Where |
|---|---|---|
| Critique of current practice → scout example → this paper | Alcorn §1 | §1 |
| First intro paragraph states claim *and* finding; objective as an equation | Melville 2024 §1 | §1 |
| Each prior paper positioned in one line: "none of these…" | Melville 2023 §2 | §2 |
| Methods split into Data and Model, numbered equations | Alcorn §2 | §3 |
| One subsection per distribution factor; section ends on one named hitter's fitted distributions | Melville 2024 §2 (Nathaniel Lowe) | §3.3, §3.6 |
| Pitch-outcome net with a batter embedding, checked by calibration plots | Melville 2023 §4.1 | §3.6 |
| Baseline as an equation; held-out test in one table | Alcorn §3.4, Table 2 | §4.1 |
| Embedding read by stat-coloured maps → neighbours → handedness | Alcorn §3.1–3.3 | §4.4 |
| Named-player application; wOBA → runs per team | Melville 2024 §3.5 | §5.1 |
| Future directions in Alcorn's shape | Alcorn §4 | §5.3 |
| One-paragraph conclusion ending on a pointed implication | Melville 2023/2024 | §6 |

Framing hook: Alcorn's Future directions proposes "modifying (batter|pitcher)2vec to predict
PITCHf/x measurements as opposed to discrete outcomes". This paper is that step.

**Deliberate deviations:**
1. **Within Results, the held-out test comes before the embedding** (Alcorn puts it last).
   His test was a win; ours is the claim and it failed. Leading with embedding pictures
   would read as cushioning.
2. **No t-SNE neighbourhood claims, no clusters** (Alcorn makes both). Decision log rules
   them out: silhouette 0.064; pooled cosine fails because norm tracks exposure (r = 0.93).

## Shape
- Body **~6,500 words**, single column, numbered sections.
- **8 figures, 2 tables, ~11 numbered equations.**
- **Name the model** (both do: "(batter|pitcher)2vec", "OptimusPitch"). Nate's pick.
- Check the SSAC27 call for page cap and template.

## Abstract (~180 words, Alcorn's 5-sentence shape)
(1) What the model is and what it learns from. (2) How a split is queried. (3) The
pre-registered sealed-2025 test did not pass: RMSE tied, rank beat both baselines.
(4) 60% / 64% of the measurable ceiling on 2024; one exploratory band beats both.
(5) The embedding separates power, contact, discipline without being told them.

## 1. Introduction (~800)
1. **Claim and finding up front** (Melville 2024). Teams decide platoons before the split
   is measurable. We learn each hitter's pitch-level process and query it for the split.
   A pre-registered test on sealed 2025 did not pass, and we measure how much of a platoon
   split one season can reveal at all.
2. **How it is done today**: platoon the opposite hand, regress the split toward league
   average; stabilization needs hundreds of PA.
3. **Scout example** (Alcorn): a scout does not read 60 PA vs LHP; he sees the hitter
   cannot lay off the slider away from a left-hander. That signal exists on every pitch.
4. **This paper**: Alcorn's proposed next step, taken. The query as an equation
   (Melville 2024 eqs 1–2): score_S(h) = Σ_j w_j^S · V(h, j). A new split is a new w.
5. **Roadmap.**

## 2. Related Work (~500)
Melville 2023 style: each paper, what it did, one line on how this differs.
- **Player embeddings.** Alcorn 2018 (at-bat result softmax: carries outcome luck);
  Heaton 2021/22 (form); Heaton 2023 (game prediction); 2025 transformer (pitch outcome;
  point regression on EV/LA failed); Melville 2023 (batter embedding in a pitch-outcome
  RNN). None composes the embedding into a season split.
- **Batted-ball distributions and PA composition.** Melville 2024 (batter-specific
  p(spray)·p(LA|spray)·p(EV|LA)); Powers & Yurko (PA as a count Markov chain; intent
  confound).
- **Platoon shrinkage and stabilization.** Tango (The Book), Klaassen 2010, Carleton
  2017, Pemstein 2016. The baseline to beat.

## 3. Methods (~2,000)
3.1 **Data.** Statcast 2015–2024, 7,245,923 pitches, 1,672 hitters, career
    pitcher-batters excluded. Frozen walk-forward: train 2015–23, select 2024, refit
    2015–24, sealed 2025 scored once. Strata: low <113, medium 113–452, high ≥452 prior
    side-specific PA; 42.7% of low is cold start. **Table 1.**
3.2 **Factorizing a pitch.** p(swing) · p(contact | swing) · p(foul, tip, in play |
    contact) · p(quality | in play) (eq). Takes from an empirical location surface, not
    the network (Statcast's zone is fit on umpire calls). Three contact classes because a
    two-strike tip ends the PA (68,244 events).
3.3 **Batted-ball quality: p(EV) · p(LA | EV) · p(spray | EV, LA).** 24 quantile bins
    each, full chain rule. Contrast Melville 2024, which assumes conditional
    independence; we assume none.
3.4 **Architecture and training** (Alcorn §2.2 style): embedding lookup, context tower,
    concatenation trunk (no bilinear term; `bilinear False`), masked multi-task log-likelihood (eqs).
    64-dim, <1M params, 5 seeds averaged before composition. Cuts (history encoder,
    pitcher-ID residual) one sentence each. **Figure 1:** architecture schematic.
3.5 **Composing a platoon split.** All 2,027 training-season pitchers, batters-faced
    weighted, one pitcher held for a whole PA (eq); why not pooled pitches (Jensen).
    12-state count chain by backward induction (eq); W(b,2) = A / (1 − E[P_foul]) (eq);
    wOBA = W(0,0) (eq). Exact, not Monte Carlo (seed noise floor 0.00096).
3.6 **Model checks.** Head calibration (Melville 2023 §4.1); composed values reproduce
    league-level outcomes on held-out data. **Figure 2:** head calibration. End on Alec
    Bohm's fitted EV/LA/spray distributions vs LHP and RHP (Melville 2024's Lowe figure;
    Bohm is the pre-registered righty-platoon anchor). **Figure 3.**
3.7 **Baselines and evaluation design.** EB bivariate platoon regression as an equation
    (Alcorn eq 8–9 style), XGBoost with context interactions, trailing averages. The
    pre-registered test and both criteria, fixed before the 2025 read. Measurement
    ceiling (route B′) in baseball terms. Label each later number confirmatory (2025),
    post-selection (2024) or exploratory.

## 4. Results (~2,000)
4.1 **The pre-registered test did not pass.** Pooled RMSE 0.0452 vs 0.0453 GBM / 0.0460
    EB; pooled rank 0.47 vs 0.42 / 0.40. **Table 2:** stratum × model × {RMSE, rank} with
    paired CIs.
4.2 **Where it wins and loses.** Medium band beats both on both criteria (one exploratory
    cell of 16). Low: too spread, not misordered (calibration slope 0.449); cold start;
    5,158 batters needed for 80% power. **Figure 4:** 2025 calibration by stratum.
4.3 **How much of a split is measurable.** 60% of the platoon rank ceiling, 64% of level,
    platoon error at the noise floor; 2024 post-selection. **Figure 5 (hero):** ceiling
    vs achieved for every method.
4.4 **What the embedding learned** (Alcorn §3.1–3.3, with held-out tests):
    - *Attribute maps.* PCA coloured by power, contact, chase, zone-swing. Held-out
      accuracy 0.73–0.79 vs 0.33 chance; wOBA rises toward high power and contact.
      **Figure 6.**
    - *Nearest neighbours.* High-exposure only; 10-NN power-tercile purity 0.591 vs null
      0.400, p = 0.001 (`neighbour_purity.csv`, stratum high). Why pooled cosine fails;
      neighbourhoods, not clusters. **Figure 7.**
    - *Platoon direction.* Within-stand sign agreement 0.569 vs permutation null 0.498,
      p = 0.001; never raw sign agreement (handedness rule alone gets 0.64). High-stratum
      projection reported on both builds.

## 5. Discussion (~900)
5.1 **Using the model.** Ranking platoon options in the medium band, where model and EB
    order differ most; named hitters chosen by that stated rule. **Figure 8.** Scale in
    runs per team via the wOBA scale (Melville 2024), for the size of the platoon spread
    only; no "runs saved" claim, since the test failed.
5.2 **Other queries, same interface.** Pitcher type, velocity, location, pitch mix;
    public explorer. Exposed, not validated (16 of 18 contrast CIs cover 0).
5.3 **Limitations and future directions** (Alcorn §4 shape). History encoder cut (the
    largest concession for low-PA hitters); cold-start prior; sequencing beyond count
    (Melville 2023's RNN does this); pitcher-ID residual; ceiling on a held-out season;
    selection bias in who faces LHP.

## 6. Conclusion (~200)
One paragraph. Built, reached, did not reach. End on the practitioner implication: with
one season, a platoon split is mostly noise for every method, so the useful question is
ordering hitters within a split, not sizing it.

## Exhibits
| # | Exhibit | Source / status |
|---|---|---|
| Fig 1 | architecture schematic | **new** (draw) |
| Fig 2 | head calibration | `results/v_chain_clean/paper_figures/fig_reliability_curve.png` (confirm content) |
| Fig 3 | Bohm batted-ball distributions by pitcher hand | **new render**, clean build |
| Fig 4 | 2025 calibration by stratum | `results/v_chain_clean/paper_figures/fig_calibration_2025.png` |
| Fig 5 | ceiling vs achieved (hero) | **new render** from `results/measurement_ceiling_clean/` |
| Fig 6 | attribute PCA maps | `results/v_chain_clean/embedding_structure/fig_attribute_maps.png` / `paper/figs/fig1_power_contact.png` |
| Fig 7 | high-exposure neighbours | `fig_neighbours_high.png` is old-build; **regenerate clean** |
| Fig 8 | platoon-ranking case study | **new**; `disagreement_top.csv` is old-build only |
| Tab 1 | data and strata counts | `data/processed/phase_d5_clean_final/manifest.json` |
| Tab 2 | sealed 2025 test | `results/model_evaluation_final_clean/` claim-1 strata |

## Decisions open before drafting
1. SSAC27 page cap/template.
2. Model name.
3. Conclusion's closing implication leans on post-V item 12 ("platoon split barely
   measurable at current exposure"), a Tier-3 headline not yet promoted. Approve or soften.
4. Phase M on sealed 2025: turns §4.3 from post-selection descriptive into held-out.
   Irreversible; Nate's call.
5. `refs.bib`: Heaton "2022" may be the 2021 SSAC paper. Add Melville 2023 and 2024.
