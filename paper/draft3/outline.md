# Draft 3 outline (v5, Alcorn shape)

Basis: Nate's 2026-11-01 notes on draft 2. Model: Alcorn 2018 (batter|pitcher)2vec.
Voice: scientific, neutral, no first person in the body, passive where natural. Describe the
finished model, how to use it, why it matters. Design choices as "X and Y were compared because A;
Y was chosen because B." No process narrative. Target body ~10 pages with figures (draft 2: 16).

## Abstract
`paper/abstract.tex` verbatim, same format (Introduction / Methods / Results / Conclusion run-in
headings, Table 1 strata table, Figure 1 power/contact, code link). Only change: 5 heads -> 6.
Body cites Table 1 and Figure 1 from here; no duplicates.

## 1 Introduction (~1 page, 4 paragraphs)
- Lineup decisions with an edge concern part-time hitters; side-specific wOBA needs ~226 PA vs LHP
  to be half signal.
- Pitch-level outcomes the hitter controls (swing, contact, EV/LA/spray) carry less outcome luck.
- Related work folded in: Alcorn 2018 embeddings + PITCHf/x suggestion; platoon regression
  (Tango 2007), empirical Bayes.
- What the paper provides: the model, a query that turns it into a split for any pitcher set
  without retraining, a test on 2025 handedness splits, a look inside the embedding.

## 2 Methods
### 2.1 Data (1 para) Statcast 2015-2025, cleaning, season splits (select on 2024, refit, score 2025).
### 2.2 Model (2-3 para + Fig arch(a) + loss eq)
- Embedding + 46-dim pitch context, six heads in PA order, take table.
- Design choices: embedding sizes 16/32/64 compared by held-out 2024 log loss (five seeds); 64
  chosen (lowest loss). Embedding table trained with plain SGD separate from the rest because it
  gave lower held-out loss and embeddings that grow with exposure. Five-seed ensemble.
- Code link.
### 2.3 Querying the model (3 steps + chain eq + query eq + Fig arch(b))
- How a user builds a split: choose S. Handedness tested; level, location, velocity are the same
  code.
### 2.4 Evaluation (2 para)
- Units, strata. Built mainly to help lower-exposure hitters, but meant to be accurate for all.
  Cuts at half and twice n* = 226: a simple convention, not tuned. Floor 10 PA.
- Baselines (EB, GBM), RMSE + weighted rank, batter bootstrap. One-sentence registration fact
  (low-stratum gate specified before 2025 model trained; 2025 scored twice, second pass reported).

## 3 Results
### 3.1 How much signal one season holds  [Fig reliability, Fig ceiling]
- Reliability vs PA, n* = 226. Level ceiling 0.742 (model 0.475, 64%); split ceiling 0.296 main
  (model 0.177, 60%; EB 49%, GBM 48%); alt 0.167. RMSE floor 0.0639 vs model 0.0633.
### 3.2 Handedness splits on 2025  [Table 1, Fig strata, Fig calibration]
- Pooled: RMSE tie, rank 0.47 vs 0.42/0.40 (CIs). Medium: beats both on both. Low: no difference
  (cold-start 43%, power 5,200 vs 246). High: even. Calibration slopes. Split grade 539: 296/243.
### 3.3 What the embedding learned  [Fig 1, Fig head partial, Fig platoon direction]
- Power/contact probes 0.79/0.75. Head-statistic partials. Seed stability. Neighbourhoods, no
  types. Weak platoon direction.

## 4 Discussion
### 4.1 Ordering more than sizing (interpretation, why medium band)
### 4.2 Case studies  [Table cases, Fig cases, Fig disagreement]
- Five anchors chosen in advance, neighbours (Pederson, Trout), disagreements (Grossman, Ozuna,
  Judge, Altuve, Robert), medium-band six.
### 4.3 What the model sees in a hitter  [Fig surfaces, Fig miss/damage]
- Location surfaces for anchors; miss vs damage by pitch family.
### 4.4 Other queries  [Fig velocity if clean rerun works]
- Level query held-out r 0.378; velocity/pitcher type (descriptive, reliability ~0).
### 4.5 Limits (1 short para)

## 5 Future directions (1-2 para, Alcorn's heading)
- Validate other queries, medium band test with fixed rule, cold start, game state, calibrated
  intervals, matchup card. Code link.
