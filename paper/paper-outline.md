# Paper outline v4 (from the abstract)

Sources: Nate's Overleaf abstract (`main.tex`), this repo, Alcorn 2018 only.
v3 outline, draft and flow reviews are archived in `archive/draft1-2026-10-31/` and not used here.

Rule: every section exists to deliver sentences of the abstract. Nothing new
gets claimed in the body that the abstract does not set up, except the
supporting detail a reader needs to trust it.

Skeleton follows Alcorn: Abstract, 1 Introduction (motivation and related
work together, no separate related-work section), 2 Methods (Data, Model),
3 Results, then Discussion and Conclusion. Additions forced by the abstract
or by Nate: 2.3 Querying (Alcorn has no equivalent), a separate Discussion
built on player case studies, and a Conclusion that recaps briefly and then
gives next steps.

Decisions (Nate, 2026-10-31): results in abstract order; platoon direction
included; Discussion and Conclusion separate, Discussion carries case studies
of specific players, Conclusion carries next steps; abstract wording fixes
applied.

Abstract sentences are numbered S1 to S12 below.

| # | Abstract sentence (short) | Section |
|---|---|---|
| S1 | Great hitters are easy, play them | 1 |
| S2 | Edge lives in remaining hitters, thin data, slow stabilization | 1 |
| S3 | Neural net inspired by (batter\|pitcher)2vec, hitter-controlled pitch outcomes, less luck | 1, 2.2 |
| S4 | 5 heads, each conditioned on pitcher and game state | 2.2 |
| S5 | 6.5M pitches 2015–2024, 1,672 hitters, 64-dim | 2.1, 2.2 |
| S6 | Queried by reweighting pitchers (hand, velo, mix, specific pitcher) | 2.3 |
| S7 | Sealed 2025 handedness: RMSE tie, rank 0.47 vs 0.42/0.40 | 3.1 |
| S8 | Ceiling on 2024: 60% rank, 64% level, best at 113–451 PA | 3.2 |
| S9 | Embedding separates power, contact, discipline untold | 3.3, 3.4 |
| S10 | Better at ordering than sizing, helps most at moderate track record | 4 (case studies) |
| S11 | Velo, location, pitch-mix queries not validated | 5 |
| S12 | Code and data link | end of 5 / footnote |

---

## Abstract

Nate's text with the two number fixes (6.5M, 113 to 451) and three wording
fixes: "player embeddings", "separates", and "64% of the level signal
pooled over both hands".

## 1. Introduction

Alcorn parallel: opens on baseball's stat culture, argues aggregate stats
miss context, imports the word2vec idea, then a paragraph on prior work.
We keep that arc but open on the decision instead of on stats.

1. **The decision (S1, S2).** Stars play every day. The edge is in the rest
   of the roster, and especially platoon calls, where a team needs a hitter's
   split against one kind of pitcher. Those hitters have the fewest PA, and
   split stats take hundreds of PA to stabilize. One concrete number for
   "hundreds" if we have a sourced one (The Book's platoon regression);
   otherwise keep it qualitative.
2. **Why outcomes are slow (S3, first half).** wOBA mixes what the hitter
   controls with fielders, park, luck on balls in play. The hitter controls
   whether he swings, whether he makes contact, and how hard and at what
   angle and direction he hits it. Those signals stabilize faster.
3. **The (batter|pitcher)2vec idea (S3).** Alcorn learned player embeddings
   by predicting at-bat outcomes from batter and pitcher identity, and showed
   they carry baseball meaning and predict unseen matchups. His Future
   directions suggested predicting PITCHf/x measurements instead of discrete
   outcomes. That is roughly our step: predict per-pitch Statcast process,
   condition on the pitch's context, and treat the embedding as something to
   query.
4. **What this paper does (contribution list).** One short paragraph or
   three bullets mirroring S6, S7–S8, S9. State the honest result up front:
   ties on error, wins on ordering.
5. **Related work, kept inside the intro as Alcorn does.** Alcorn 2018;
   The Book's empirical-Bayes platoon regression (it is our baseline, so it
   gets named here). Keep it to what the paper uses.

Length target: about 1 page.

## 2. Methods

### 2.1 Data (S5)

Alcorn parallel: source, unit of observation, filtering, final counts.

- Statcast pitch-level, 2015–2025: 7,245,923 pitches.
- Train 2015–2024: 6,540,616 pitches, 1,672 hitters, 2,182 pitchers.
- 2024 is the selection season. 2025 is sealed and read once for the final test.
- Walk-forward, season-frozen splits, no random splits.
- Pitcher-batters removed (the clean build).
- Prior-exposure strata, named once here and reused: low, medium (113 to 451
  prior PA against that hand), high.
- Leakage controls in one or two sentences (prior-season-only features).

Table: data counts (could merge into Table 1 or a small Table A).

### 2.2 Model (S3, S4, S5)

Alcorn parallel: a figure of the architecture, then equations, then
training details in one paragraph.

- **Figure: architecture.** Hitter ID → 64-dim embedding; pitch context
  (46 dims: pitcher characteristics, pitch, count) → shared trunk → 5 heads.
- **The 5 heads, in PA order.** Swing decision, contact given swing, then
  contact quality as exit velocity, launch angle, spray direction (chained so
  each conditions on the ones before). One sentence on why these are
  "hitter-controlled".
- **Conditioning (S4).** Every head sees the pitcher (characteristics-first)
  and game state, which here is count only. One line on why only count.
- **Loss.** Sum of per-head likelihoods. One equation in Alcorn's style.
- **Training.** 5-seed deep ensemble averaged before composition, tuned on
  2024, refit on 2015–2024. Under 1M parameters. Config-driven, seeded.

### 2.3 Querying the model (S6)

New relative to Alcorn, and the heart of the abstract.

- From pitch process to wOBA. Heads compose into a PA outcome through the
  12-state count chain (backward induction from each count), then to wOBA.
- **A query is a reweighting of pitchers.** Pick the pitchers the projection
  is drawn over, weight by batters faced, run the chain. Handedness = all
  same-hand or opposite-hand pitchers. Velocity, pitch mix or one named
  pitcher are the same operation with a different set.
- Small figure or equation showing the reweighting.
- One sentence: only handedness is evaluated in this paper (sets up Section 5).

Length target for Section 2: about 2 pages.

## 3. Results

Abstract order: prediction, ceiling, then embeddings. (Alcorn puts
embeddings first; Nate chose the abstract's order.)

### 3.1 Handedness splits on sealed 2025 (S7)

Alcorn parallel: 3.4, held-out matchups against a naive baseline and a
simpler model, one table.

- Target: each hitter's 2025 wOBA against each hand.
- Baselines: gradient-boosted model with context interactions (GBM), and
  empirical-Bayes platoon regression (EB). One line each on why they are the
  right bar.
- Metrics: PA-weighted RMSE and rank correlation, pooled and by stratum.
- **Table 1** (already in Nate's abstract doc): RMSE 0.0452 / 0.0453 / 0.0460,
  rank 0.470 / 0.421 / 0.402.
- **Figure:** `results/paper_figures/fig_claim1_2025_strata.png` (by stratum).
- Honest framing required by the project: the pre-registered test (beat both
  baselines on both metrics) did not pass. Error is a tie. Ordering wins.
  State this plainly in this section, not only in the discussion.
- Medium stratum beats both baselines. It is one exploratory cell among
  sixteen, so say "exploratory". Low stratum is too dispersed (43% cold start; that figure is from the Phase C notes, so recheck it on the clean build before quoting it).

### 3.2 How much of the signal is there to get (S8)

No Alcorn parallel. Needed because S8 is a headline number.

- **Telepathic model / noise ceiling.** Plain-language definition: a model
  that knew each hitter's true split would still miss by the sampling noise of
  one season. That bounds what any method can reach.
- 2024, the selection season. Say so, and that this makes it descriptive.
- Model recovers 60% of the rank signal and 64% of the level signal (level
  pooled over both hands).
- By stratum: most reliable in the medium band (113 to 451 prior PA).
- Platoon error sits at the noise floor, report it beside the rank fraction.
- **Figure:** `results/paper_figures/fig_level_query_strata.png` or a
  ceiling-fraction bar chart.

### 3.3 What the embedding learned (S9)

Alcorn parallel: 3.1 visual inspection and 3.2 nearest neighbours.

- **Figure 1** (in Nate's abstract doc): `fig1_power_contact.png`.
- Attribute maps for power, contact, plate discipline. The model was never
  given these definitions. Attribute accuracy 73–79% against 33% chance.
- Nearest neighbours as Alcorn did, named hitters, high-exposure only (where
  the embedding is stable). Two or three pairs a reader will recognize.
- Keep it qualitative, as the abstract says.

### 3.4 A platoon direction in the embedding

Alcorn parallel: 3.3 opposite-handed doppelgängers. Not promised by the
abstract; included by Nate's choice as support for S9 and the platoon story.

- Hitters cluster by batting side: neighbour purity 0.645 against a
  permutation null of 0.506.
- A direction in the 64 dims lines up with a hitter's platoon split:
  0.569 against a null of 0.498. Projection onto it correlates with the
  observed split in every stratum (partial r 0.07 low, 0.14 medium, 0.13 high).
- **Figure:** `results/v_chain_clean/platoon/fig_platoon_direction.png`.
- Keep the claim small: the structure is there, it is weak, and it is
  descriptive.

Length target for Section 3: about 3 pages.

## 4. Discussion (S10)

Alcorn's paper has no discussion; his player names live in Results. Here the
Results sections stay aggregate and the players come in Discussion, so each
case study answers "what does S10 look like for a real roster decision".

**Opening.** Ordering beats sizing. A team picking between two bench bats
needs the order more than the exact points of wOBA. The biggest help is at a
moderate track record, which is where a platoon call is actually unsure.

**Case studies: the five Phase V anchor hitters.** Phase V (notebook 09,
`docs/phase-v-spec.md` V.8 to V.10) already studied five named hitters, each
with an expectation written down before the result was read:
- **Joc Pederson, Kyle Schwarber:** classic platoon lefties. Expected worse
  against LHP than RHP, and neighbours that share their side and platoon sign.
- **Alec Bohm:** expected the reverse split.
- **Mike Trout, Juan Soto:** stars, expected near-flat splits and high-wOBA
  neighbours.

For each anchor the Discussion uses three existing Phase V pieces:
1. **Conditional surfaces (V.9).** Expected wOBA against LHP and against RHP
   through the query, with the heads shown separately (swing, contact,
   in-play quality), so the reader sees which part of his process the split
   comes from.
2. **Nearest neighbours (V.8).** Neighbours in the embedding beside
   neighbours by plain stats, as Alcorn did with Trout and Goldschmidt.
3. **Model vs EB disagreement (V.10).** The hitters the model and EB rank
   most differently on platoon split. The case studies come from here and
   from the 113–451 PA band, which ties them to S10.

Graded against 2025 in a separate module (`src/analysis/case_studies_2025.py`,
`results/case_studies_2025/`), so V.10 stays "on whom", as registered.

**What the clean build shows (re-run 2026-10-31, `results/v_chain_clean/`).**
- **Surfaces (V.9), model L-R gap with the league offset removed:** Pederson
  -0.004 and Bohm +0.009 match their expectations. Schwarber is -0.0002, so
  flat, not lower vs LHP. Trout -0.013 and Soto +0.008 are near-flat.
  Every gap is far smaller than the hitter's observed history (0.03 to 0.08).
  This is the magnitude compression S10 talks about, shown in five players.
  Sign agreement with observed history is 4 of 5 (Soto is the miss, and
  Schwarber's agreement is a tie).
- **Neighbours (V.8):** embedding and stat neighbour lists share 0 of 8
  names for four anchors and 1 of 8 for Schwarber. Pederson's are the best
  story: Tellez, Schwarber, Sheets, Gallo, mostly left-handed power bats.
  The pooled lists pull in low-exposure hitters (Trout's top three are Matt
  Hague, Wilyer Abreu, Iván Herrera), so use the high-stratum lists. Trout's
  high-stratum embedding list is Acuña, Alonso, Belt, Suárez, Austin
  Jackson. Stat neighbours are Álvarez, Goldschmidt, Soto, Betts. The
  embedding groups by process, not by output level. Say that, do not hide
  Austin Jackson.
- **Disagreement (V.10), 545 hitters on 2024:** model and EB rank platoon
  split with Spearman 0.77 pooled (0.80 low, 0.79 medium, 0.69 high). They
  disagree most among established hitters. Biggest gaps: Robbie Grossman
  (EB has him first, model 324th of R hitters), Marcell Ozuna (the reverse),
  Luis Robert, Kyle Farmer, James McCann, Aaron Judge, José Altuve.
- **Graded on 2025:** across 539 hitters the model is closer on 296, EB on
  243, PA-weighted error 0.0448 vs 0.0462. EB is closer on all five
  anchors, by 0.002 to 0.013. Of the top-20 V.10 hitters, 16 played enough
  in 2025: model closer on 9, EB on 7. For every case-study hitter the 2025
  split sits within 1.96 SE of the midpoint between the two predictions
  (519 of 539 hitters overall), so noise alone could flip the grade and no
  one case study is evidence. Grossman, J. D. Davis, Nido and Sanó (top-20
  gaps) have no 2025 split. Use them as illustrations and say so.
- **Candidate cases (picked by rule, not by look):** Pederson (expectation
  met in surfaces and neighbours, but 19 PA vs LHP in 2025, SE 0.12).
  Altuve and Judge (two of the three largest 2024 gaps where the model was
  closer in 2025, McCann the third on 136 PA).
  Ozuna (large gap, EB closer). Trout (neighbours that show process over
  level).

**Limits, stated plainly.** Failed pre-registered gate, ceiling measured on
the selection season, medium band exploratory, count-only game state, one
season of 2025 noise per hitter (so a single case study can be luck; say so).

Length target: about 1.5 pages.

## 5. Conclusion (S11, S12)

Technical-writing note: a conclusion that is only next steps reads as
unfinished. Standard form is a short recap, then next steps. Alcorn ends on
"Future directions" with no recap, so a brief recap is the only addition.

- **Recap, 2–3 sentences.** Process-based embedding, queried by reweighting
  pitchers. Ties on handedness error, orders better, most help at a moderate
  track record.
- **Next steps.**
  - Validate the other queries (velocity, location, pitch mix, one named
    pitcher). Each needs a held-out target and a matching baseline.
  - The low band: what a cold-start hitter needs (a prior, or minor league
    data mapped in, as Alcorn suggests).
  - Game state beyond the count.
  - Using the query in a real decision tool (lineup or matchup card).
- Close on S12: code and processed data at github.com/nseluga/hitter-embedding.

Length target: about half a page.

---

## Figures and tables

| Item | Source | Status |
|---|---|---|
| Table 1, 2025 handedness results | Nate's `main.tex` | exists, numbers checked |
| Fig 1, power vs contact map | `fig1_power_contact.png` | exists in Overleaf |
| Architecture diagram | to draw | new |
| Query/reweighting diagram | to draw, could merge with architecture | new |
| 2025 results by stratum | `results/paper_figures/fig_claim1_2025_strata.png` | exists |
| Ceiling fractions by stratum | `results/paper_figures/fig_level_query_strata.png` or new | check fit |
| Platoon direction | `results/v_chain_clean/platoon/fig_platoon_direction.png` | exists |
| Anchor surfaces, neighbours, disagreement | `results/v_chain_clean/fig_surfaces_*.png`, `fig_neighbours_high.png`, `fig_disagreement.png` | exists, clean arm |
| Case studies graded on 2025 | `results/case_studies_2025/case_studies_2025.csv` | exists, table not figure |
