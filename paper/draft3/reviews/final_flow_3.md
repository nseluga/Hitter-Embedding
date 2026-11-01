# Final review, draft 3: length, proportion, claim calibration

Body (excluding the abstract) runs about 5,200 words. Methods takes about 1,600 of them, Results about 1,100 and Discussion about 1,600. Most of the excess is repetition and process residue, not unearned sections. Cutting it brings the body to about 4,300 words without losing a result.

## 1. Top changes (ranked)

**1. Methods 2.2, the optimizer paragraph ("The network has 266,317 parameters...", about 330 words).** This is the longest paragraph in the paper, and it is where the most process history is left. "as the architecture plan called for" refers to an internal document. The seed-stability result (cosine 0.83 against 0.12) is a finding, not a method. The code URL appears here, in the abstract and again at the end of the paper. Move the cosine sentence to the start of 3.3, where seed alignment is already described, and drop the URL from this paragraph. Replacement (about 220 words):

> The network has 266,317 parameters. Weights outside the embedding table are trained with AdamW \citep{loshchilov2019} at a learning rate of $10^{-3}$ and batches of 8,192 pitches. The embedding table is trained with plain stochastic gradient descent at a learning rate of 1.0, because an adaptive optimizer gives a rarely seen hitter steps as large as a regular's. Plain gradient descent moves each embedding in proportion to the hitter's data, so embedding length grows with exposure, and on an earlier build of the data it gave lower held-out loss. Embedding sizes of 32 and 64 were compared, and 64 was chosen because it gave the lower 2024 log loss averaged over five seeds (1.02327 against 1.02378, a gap about four times the seed-to-seed standard deviation). Weight decay of $10^{-2}$ on all weights scored better than stronger and weaker settings in a single-seed screen, and dropout of 0.1 in the trunk was fixed in advance. The final model was trained for 18,538 steps, the median early-stopping point of the five selection runs. Five copies are trained from different seeds as a deep ensemble \citep{lakshminarayanan2017}, and each head's predicted distribution is averaged across the five before any query. The model was implemented in PyTorch and trained on a CPU in about 25 minutes per seed.

If this is not done, the paragraph keeps reading like a lab notebook, which is the thing the author asked to remove.

**2. Discussion 4.1, second paragraph ("The model was built to be accurate for every hitter...").** This paragraph repeats three things from earlier in the paper. The intent sentence comes from 2.4, nearly word for word but in reverse order. "The strata cuts are a convention" also comes from 2.4, and the 43% unseen-hitter fact from 3.2. Its mechanism sentence is also stated as fact ("this middle range is where they add the most") when it was never tested. What the paragraph leaves out is the most useful interpretation of the low stratum: the model ranks those hitters well, but their projections are overspread (slope 0.59). Replacement (about 150 words, down from about 200):

> The model's advantage over the baselines varies with exposure, and because the strata cuts are a convention, the pattern is better read as a curve than as three groups. With heavy exposure, several seasons of outcomes already measure a hitter well, which leaves little for any method to add. With moderate exposure, a hitter has seen enough pitches from a hand for the embedding to describe how the hitter handles them, while the outcomes of those plate appearances are still too few to stabilize. Pitch-level outcomes accumulate several times faster than plate appearance results, which would explain why this range shows the clearest gain. With little exposure, the model's point estimates still rank hitters ahead of both baselines, but its projections are spread more widely than the outcomes support (calibration slope 0.59), and the stratum holds too few batters to confirm the margin.

**3. Future directions has one oversell and one undersell.**
- The oversell: "a matchup card that ranks a team's options against a given pitcher". A single-pitcher query is the one this same section later says is untested. Replace with: "a matchup card that ranks a team's options against a pitcher hand, the query tested here, with a band around each projection."
- The undersell: "The low-exposure margin, positive but far from significant, would need many seasons to confirm." "Far from significant" reads as a negative result, but the author's framing is "positive but underpowered". Replace with: "The low-exposure margin is positive but underpowered, and confirming it at its observed size would take far more batters than one season holds (Section~\ref{sec:test})."
- Also drop "with no retraining for a new split" from the first sentence. That point is already made in the abstract, the introduction, the Figure 1 caption and Section 2.3.

**4. Discussion 4.3, the surfaces paragraph (about 175 words).** The paragraph reads the figure aloud in seven numbers (170, 80, 200 to 220, 30 to 40, 70, 44, 48). Only the Pederson pair supports an argument. It also stops one step short of its payoff, which is that Pederson's platoon gap comes from swing decisions, not contact. Replacement (about 110 words):

> Because the network scores single pitches, it can be read one pitch location at a time. Figure~\ref{fig:surfaces} shows, for each case-study hitter, the expected wOBA on a ball in play by location, relative to the average of 100 hitters on the same pitches. Schwarber and Soto do their damage on pitches up in the zone, and Soto falls below average on low pitches. Trout's damage sits low, peaking at about 200 points above average on low pitches over the middle. Bohm's surface is nearly flat. Each surface changes little between pitcher hands. Pederson's average edge on contact is 44 points against left-handed pitchers and 48 against right-handed pitchers, so the projected platoon gap comes mainly from the swing decisions described in Section~\ref{sec:anchors}, not from the quality of contact.

**5. Results 3.1, the ceiling comparison.** The sentence "60% of that ceiling, compared with 49% for empirical Bayes and 48% for the boosted baseline" invites a ranking of the three methods that the interval (26% to 91%) does not support. Replace "The share is imprecise (interval 26\% to 91\%) and depends on..." with "The share is imprecise (interval 26\% to 91\%, wide enough to hold both baselines' shares) and depends on...". That costs seven words and closes the one place where Results quietly oversells.

**6. A repetition sweep, worth about 120 words.**
- The 43% of low-stratum units held by unseen hitters appears in 3.2, 4.1 and Section 5. Keep it in 3.2 and 5.
- Embedding length tracking exposure (0.93) appears in 3.3 and again in the neighbors paragraph. Cut "The neighbor search also uses direction only, ignoring the vector length, which tracks playing time (correlation 0.93 with log prior PA)." Section 3.3 already says the vectors are scaled to unit length.
- "New split, new weights, no retraining" appears five times. Keep it in the abstract, the introduction and the Figure 1 caption. Cut "Any other pitcher set uses the same calculation with no retraining." from 2.3.
- The code URL appears three times. Keep it in the abstract and on the final line.

**7. The strata framing is consistent except for two spots.** Section 2.4 states the intent exactly right: built mainly for low-exposure hitters, meant to be accurate for all, cuts are a convention. The abstract, Section 3.2 and Section 5 agree with it.
- The introduction never states the intent, so a reader first meets it in the evaluation section. Append to the last sentence of the introduction: "...against empirical Bayes and gradient-boosted baselines, for all hitters and by how much history each hitter has against a hand, and the embedding is examined..."
- In Section 3.2, "The clearest signal is in the medium stratum" is followed by "all four intervals exclude zero" with no reminder that this was not the registered test. Add one word: "The clearest signal, in exploratory comparisons, is in the medium stratum...". That makes the claim strong and correctly labeled, not hedged.

## 2. What should not change

- Section 2.4's statement of intent and its "a simple convention rather than a tuned choice". This is the framing the author wanted, stated once and cleanly.
- The footnote on the earlier 32-dimension build. It shows the direction of the low-stratum error estimate is fragile, and it is the paper's most honest calibration point.
- The 3.2 power calculation (5,200 against 246 batters) and the floor-sensitivity sentence. They turn "not passed" into "underpowered" with evidence.
- "The evidence on accuracy is the 539-hitter comparison", which keeps the case studies illustrative.
- The structure of Results (ceiling, test, what it learned) and the figure load in Discussion. Both match the author's brief.
- The 4.1 heading and its claim, "ordering more than sizing". It is calibrated to the pooled tie on error and the pooled win on ranking.
- Section 4.4 is the right length.

## 3. Word-count targets (body text, excluding tables and captions)

| Section | Now (approx.) | Target |
|---|---|---|
| 1 Introduction | 500 | 450 |
| 2.1 Data | 160 | 150 |
| 2.2 Model | 670 | 520 |
| 2.3 Querying | 400 | 360 |
| 2.4 Evaluation | 380 | 360 |
| 3.1 Ceiling | 240 | 240 |
| 3.2 Handedness 2025 | 460 | 430 |
| 3.3 Embedding | 420 | 420 (gains the seed sentence) |
| 4.1 Ordering | 340 | 280 |
| 4.2 Case studies | 570 (incl. table) | 500 |
| 4.3 What the model sees | 540 | 420 |
| 4.4 Other queries | 190 | 190 |
| 5 Future directions | 310 | 270 |
| **Body total** | **~5,200** | **~4,600** |

The total gets to about 4,300 if the eight rows of Table 2 are folded into Figure 10 (the figure already shows each observed split, its error bar and both projections). That cut is optional and lower priority.

## 4. Verdict

This draft is close. No section is structurally too long, and the strata story is now told the way the author wants it in 2.4, 3.2 and Section 5. What remains is about 600 to 900 words of excess, nearly all of it repetition (the intent sentence, the 43%, the 0.93, no-retraining, the URL), process residue in the optimizer paragraph, and a surfaces paragraph that recites the figure. Calibration is good overall, and the paper errs toward honesty. It has three real misses. The matchup-card sentence promises an untested query. "Far from significant" undersells the low stratum against the author's own "positive but underpowered". The 60% against 49% ceiling comparison implies a ranking its interval cannot support. Fix those three, apply the cuts in items 1, 2, 4 and 6, and the paper reads as a finished result rather than a record of the process.
