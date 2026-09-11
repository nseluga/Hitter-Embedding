# SSAC27 abstract outline

Title: Conditional Queries Against a Learned Hitter Process Model

Format: one paragraph, no headers, under 350 words, ends on the repo link. No dashes.
Numbers only where marked.

1. **Problem.** Most of what a team wants about a hitter is a split (vs lefties, vs high
   velocity, vs pitchers who live up in the zone). Each needs hundreds of PA to stabilize;
   the decision comes before then.
2. **Current answer and its cost.** Shrinkage toward a league or handedness average throws
   away the pitch-level record.
3. **Prior work, one sentence, no names.** Player embeddings exist but are read out for form
   or game predictions, not composed into a season split.
4. **What we do.** Train a hitter model on pitches, query it for any split. Handedness is
   the first validated query.
5. **Model.** Under 1M params; 32-dim hitter embedding plus pitch context; predicts swing,
   then contact, then EV, LA, spray as bins. *(Numbers: 1M, 32.)*
6. **Data.** 7.3M Statcast pitches, 2015 to 2024, 1,870 hitters. *(Numbers: all three.)*
7. **Query.** A weight vector over pitchers (BF-weighted), composed through the count-state
   Markov chain exactly to wOBA. A different split is a different weight vector, no retraining.
8. **Gate result, stated first.** Sealed 2025, pre-registered test vs gradient-boosted and
   empirical Bayes baselines did not pass: pooled error tied, pooled rank beat both.
9. **Ceiling result, the headline.** Model recovers 55% of platoon signal and 63% of level
   signal one season can measure. *(Numbers: 55, 63.)*
10. **Where it wins and loses.** (a) Beats both baselines only at 113 to 452 prior PA;
    (b) below that 43% are unseen and predictions are too spread out, not misordered.
    *(Numbers: 113, 452, 43.)*
11. **Embedding finding.** Trained only on pitch outcomes, separates hitters on power,
    contact, discipline at 0.75 to 0.80 held-out. *(Numbers: 0.75, 0.80.)*
12. **Exposed queries + close.** One clause: velocity, fastball location, and pitch-mix
    queries use the same interface; one season is too noisy to score them per hitter, so
    they are exposed, not validated. Then the repo link.

Cut order if over length: 3, then 11, then 10b. Never cut 8 or 9.

Left out on purpose: five-seed ensemble, walk-forward detail, sign test, calibration slope,
any pitcher-type result number (2024 exploration, descriptive, no sealed target), any named
prior paper, anchor or similarity claims, raw sign agreement, in-sample level query.

Source for sentence 12: `results/pitcher_type_query/ceiling.csv` (2024 split-half
reliability of top-vs-bottom tercile contrasts, 16 of 18 CIs cover 0).
