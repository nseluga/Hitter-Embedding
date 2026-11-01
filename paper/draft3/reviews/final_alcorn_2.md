# Final review, lens: Alcorn comparability, tone and voice

Baseline register (from the Alcorn PDF itself). Methods are passive past tense stating what was done ("Play-by-play data ... were obtained", "The model was trained for 100 epochs"). Hyperparameters are listed with no history. Where a choice gets a reason, it is a purpose clause ("To ensure an adequate amount of data was available ..., only the most frequent batters ... were included"). Results are stated confidently. The draft already matches this in most places ("This paper introduces ...", "It orders hitters better than both", "The model's prediction intervals are too narrow"). What follows is what is left over.

## 1. Top changes (ranked)

**1. Methods 2.2, training paragraph ("Adaptive and plain stochastic gradient descent were both tested ...").** This is the last large block of process narration in the paper. "on an earlier build of the data", "as the architecture plan called for", "was kept after stronger and weaker settings each scored worse in a single-seed screen", "was fixed in advance" and "replayed" all describe the history of the work, not the model. "As the architecture plan called for" is also not a reason, and the author explicitly asked for "x was tested because y". If it stays, the paragraph keeps reading like a lab notebook beside Alcorn's three-sentence equivalent. Replacement, from "Weights outside" up to "Five copies":

> Weights outside the embedding table are trained with AdamW \citep{loshchilov2019} at a learning rate of $10^{-3}$ in batches of 8,192 pitches. The embedding table is trained with plain stochastic gradient descent at a learning rate of 1.0. An adaptive optimizer would give a rarely seen hitter steps as large as a regular's, while plain gradient descent moves each embedding in proportion to the hitter's data, so embedding length grows with exposure. Embedding sizes of 32 and 64 were compared to [AUTHOR: state the reason, e.g. test whether a larger space holds more hitter detail without overfitting rare hitters], and 64 gave the lower 2024 log loss averaged over five seeds (1.02327 against 1.02378, a gap about four times the seed-to-seed standard deviation). Weight decay of $10^{-2}$ applies to all weights, including the embedding table, and scored better on 2024 than stronger or weaker settings in single-seed runs. Dropout of 0.1 in the trunk was not tuned. The final model was trained for 18,538 steps, the median early-stopping point on 2024, with the same learning-rate reductions.

What gets lost: the claim that SGD "gave lower held-out loss" on an earlier build. That evidence came from a different dataset and is weak, and the mechanism argument covers the choice.

**2. Methods 2.4, pre-registration sentences and the footnote.** "The model had to beat the boosted baseline" is informal, and the footnote opens with history ("2025 was also scored with an earlier build"). Keep the content. It is the paper's most important integrity disclosure. Replacement:

> The low stratum was fixed as the primary test before the 2025 model was trained. Passing required a lower RMSE than the boosted baseline and a higher rank correlation than both baselines. All other comparisons are exploratory.\footnote{An earlier build, with a 32-dimension embedding and pitchers batting retained, was also scored on 2025. No interval changed whether it excluded zero, although the low-stratum error differences changed from nominally favoring the baselines to nominally favoring the model.}

**3. Discussion 4.2, opening ("Four had an expected platoon direction fixed before any projection was examined, and Alec Bohm replaced a pre-registered right-handed hitter ...").** This is process narration in the second sentence of a section meant to be interpretive. Keep the disclosure but move it to a footnote:

> Five well-known hitters serve as case studies, each with an expected platoon direction set before projections were examined.\footnote{Alec Bohm replaced the originally chosen right-handed hitter, whose training data showed no platoon split, before any projection was examined.}

[AUTHOR: confirm that Bohm's substitution happened before projections were seen. If it did not, the current wording has to stay.]

**4. Discussion 4.1, "The model was built to be accurate for every hitter, with particular attention to hitters who have little history against a hand, and its advantage over the baselines varies with exposure."** This repeats Methods 2.4 almost word for word. Alcorn never restates a design goal. Replace with:

> The model's advantage over the baselines varies with exposure.

**5. Hedge stacks (two places).**
- Results 3.3: "is only weakly aligned with a direction fit separately ... (cosine 0.14, interval 0.08 to 0.20), positive but far from parallel." This uses three qualifiers for one number. Replace with: "aligns weakly with a direction fit separately to decode the empirical Bayes split from the embedding (cosine 0.14, interval 0.08 to 0.20)."
- Discussion 4.3: "One possible reason is that exit velocity is observed only on balls in play, ..., but this was not tested." Replace with: "Exit velocity is observed only on balls in play, which leaves fewer labels per pitch family than swings and contact have and may limit how finely power separates by pitch type."

**6. "selection model" / "selection runs" / "The chosen configuration was then retrained" (2.1, 2.2, 3.3).** This is workflow vocabulary that a reader has to decode. Define it once, in Data: "Hyperparameters were selected by training on 2015 through 2023 and scoring on 2024, and the final model was trained on 2015 through 2024 ...". Then in 3.3 write "The analyses below use the model trained through 2023." In 2.2 write "the five 2024 runs" instead of "the five selection runs".

**7. Informal phrasing.** Quote, then replacement.
- "was built mainly to help with hitters" (2.4) becomes "targets hitters".
- "six real pitches that pitcher threw" (2.3) becomes "six pitches that pitcher threw".
- "Schwarber and Soto do their damage on pitches in the upper middle" / "Trout's damage sits low" (4.3) becomes "Schwarber and Soto gain most on pitches in the upper middle" / "Trout's advantage is lowest in the zone".
- "mostly left- and right-handed power bats" (4.3) becomes "mostly power hitters from both sides".
- "These neighborhoods shade into one another." (4.3) becomes "The neighborhoods overlap."
- "Checking it is harder than checking the projection." (4.4) becomes "Its outcomes are harder to check."
- In the Figure 6 caption, "Each head aligns with the statistic it should" becomes "Each head aligns with its corresponding statistic".

**8. Defensive framing and repetition.**
- 2.4, "a simple convention rather than a tuned choice" becomes "The cuts, half and twice the 226 PA ..., were set by convention and not tuned." This states it plainly instead of defending it.
- Delete the final line of Section 5 ("The code is available at ..."). The URL already appears in the abstract and in 2.2, and Alcorn gives it once.

## 2. Things that should NOT change

These are the places where the draft is more careful than Alcorn. Keep them.
- **Intervals and power.** Alcorn reports "significantly (p < 0.001)" and "These results prove ...". The draft gives difference intervals, the 5,200-batter power figure, and "so the test was not passed". This is better science and costs no confidence in tone.
- **Measurement ceiling (3.1) and the caveat that 2024 is the selection season.** Alcorn has nothing comparable. It tells readers how much signal exists before they judge the model.
- **Noise caveats on single hitters.** These include the z column in Table 3, "the cases show how the two methods differ rather than which is right", and Pederson's 19 PA. Alcorn's case studies rely on reputation ("an unsurprising result considering ..."). The draft's version is stronger.
- **Null baselines.** These are the seed-stability null (0.83 against 0.12), shuffled-label neighbors, and the control for log prior PA on vector length.
- **Things that already match Alcorn.** The neighbor comparison against stat-based neighbors (4.3) mirrors Alcorn's Goldschmidt point and should stay. So should the "This paper introduces ..." sentence and the counterfactual-roster tie-back to Alcorn in Section 5.
- Do not add Alcorn's evaluative adjectives ("intriguing", "fascinating"). The draft's flatter voice fits the author's "unbiased" direction better.

## 3. Verdict

The voice is now mostly impersonal, plain and confident, and closer to Alcorn than any earlier draft. Results sentences in particular state findings without throat-clearing. Process narration is left in only a few pockets. The worst is the training paragraph in 2.2, which still tells the story of how hyperparameters were reached instead of stating them with a one-clause reason, as the author asked. Next are the pre-registration wording, the Bohm substitution and the "selection model" jargon. Two hedge stacks and a handful of informal phrases are left. Fixing items 1 to 4 makes the paper read as a final account rather than a project log. The careful parts (intervals, ceilings, noise caveats) are where the draft beats Alcorn and should be defended against any pressure to sound bolder.
