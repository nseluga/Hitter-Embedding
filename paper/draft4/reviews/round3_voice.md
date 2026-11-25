# Round 3 review: sentence-level voice (final round)

Lens: rhythm, paragraph openers and closers, repeated constructions, AI tells. Checked against round1_voice and round2_voice (and the round 1/2 content and structure notes) so nothing below re-proposes an applied or deliberately skipped fix. Two items fix regressions introduced while earlier fixes were applied (item 2's "empirical Bayes's" and item 6's stacked appositive).

The draft is now close to both targets. The one pattern still loud enough to matter is ", so" as the default connective. It appears 32 times in the body, three of them in one six-sentence paragraph. Alcorn varies the same move ("which is suggestive of", "thus appears", "which is to be expected"), and Nate uses "suggesting", "indicating", "As a result", and "Overall,". Items 3 to 5 cut nine of the 32.

Every replacement keeps all numbers and hedges, adds no facts, and uses no first person, gendered pronouns, dashes, colons, or semicolons. The abstract, the tab:test caption, and the fig:powercontact caption are untouched. Line numbers are from the current .tex files.

Ranked by impact.

---

### 1. part3.tex:89 (Discussion, last two sentences). The body's final sentence repeats the abstract word for word, and "that" points at the wrong thing
**Current (a).** Pitch-level outcomes accumulate several times faster than plate appearance results (Section~\ref{sec:intro}), which would explain why this range shows the clearest gain, although that gain was found in exploratory analysis.

**Replace with.** Pitch-level outcomes accumulate several times faster than plate appearance results (Section~\ref{sec:intro}). This would explain why the clearest gain appears in this range, although that gain was found in exploratory analysis.

**Current (b).** For a team, that makes the model most useful for ranking platoon options among hitters with a moderate track record, where the decision is actually in question, not as a replacement for a projection system on raw accuracy.

**Replace with.** For a team, the model is most useful for ranking platoon options among hitters with a moderate track record, where the decision is still in question. It is not a replacement for a projection system on raw accuracy.

**Why.** (b) follows the low-stratum sentence, so "that" reads as if the poor low-stratum calibration is what makes the model useful. "actually" is filler, and "not as a replacement" is a negative tack-on at the end of a 38-word sentence. Nate closes on a short plain claim ("Overall, these results paint a clear picture..."). (a) is a which-then-although chain.

### 2. part2.tex:48 (3.2, verdict). Dangling "Taken together", and a possessive that round 2 removed elsewhere
**Current.** Its RMSE is about 2 points of wOBA lower than the boosted baseline's and 4 lower than empirical Bayes's, against a typical error near 45 points, so for these hitters the model's advantage lies mostly in ordering them. Among high-exposure hitters the three methods are even. Taken together, the one pre-registered test was not passed. In exploratory comparisons the model ties both baselines on error and orders hitters better than both, most clearly for hitters with a moderate track record.

**Replace with.** Its RMSE is about 2 points of wOBA below the boosted baseline and 4 below empirical Bayes, against a typical error near 45 points. For these hitters the model's advantage lies mostly in ordering them. Among high-exposure hitters the three methods are even. Overall, the model did not pass the one pre-registered test. In exploratory comparisons it ties both baselines on error and orders hitters better than both, most clearly for hitters with a moderate track record.

**Why.** Nothing in "the one pre-registered test was not passed" is taken together, so the opener dangles on the paper's most important sentence. "Overall," is Nate's own summary move. "empirical Bayes's" came back when round 2 item 7 was applied.

### 3. ", so" as the default connective (32 uses). Four spots where it stacks
**(a) part2.tex:13. Current.** With that variance taken from 2024 alone, the ceiling falls to 0.167, below the model's 0.177, so the share is a rough guide rather than a measurement.
**Replace with.** With that variance taken from 2024 alone, the ceiling falls to 0.167, below the model's 0.177. The share is a rough guide, not a measurement.

**(b) part3.tex:30. Current.** Empirical Bayes was closer for all five in 2025, by 0.002 to 0.014, but every $z$ in Table~\ref{tab:cases} is below 1.96, so noise alone could reverse which projection is closer.
**Replace with.** Empirical Bayes was closer for all five in 2025, by 0.002 to 0.014. Every $z$ in Table~\ref{tab:cases} is below 1.96, however, which means noise alone could reverse which projection is closer.

**(c) part2.tex:71. Current.** A platoon decision depends on the gap between a hitter's two sides, so the model's projected split and the empirical Bayes split were also graded against each hitter's observed 2025 split.
**Replace with.** Because a platoon decision depends on the gap between a hitter's two sides, the model's projected split and the empirical Bayes split were also graded against each hitter's observed 2025 split.

**(d) part3.tex:68. Current.** Across the five case-study hitters the two kinds of list share 9 of 40 names, so the embedding's notion of similarity differs from one built on season statistics.
**Replace with.** Across the five case-study hitters the two kinds of list share 9 of 40 names, which suggests that the embedding's notion of similarity differs from one built on season statistics.

**Why.** Each of these paragraphs already has another ", so" (part2:13 has two, part3:30 two, part2:71 two). (d) borrows Alcorn's "which is suggestive of" move for the same kind of neighbor comparison.

### 4. part1.tex:62 (2.2, first paragraph). Three "Each" openers and three ", so" tails in six sentences
**Current.** Each hitter has one embedding for all training seasons, so a projection describes the hitter across those seasons together. Each pitch is described by a 46-dimension context vector built from its physical measurements, location, type, the handedness of both players, and the count. The pitcher enters only through the pitches thrown, not through an identifier, so the model can describe any pitcher whose pitches it has seen. The count is the only game state included, so predictions reflect the hitter and the pitch rather than the game situation.

**Replace with.** The same embedding covers all training seasons, and a projection describes the hitter across those seasons together. Each pitch is described by a 46-dimension context vector built from its physical measurements, location, type, the handedness of both players, and the count. The pitcher enters only through the pitches thrown, not through an identifier, which lets the model describe any pitcher whose pitches it has seen. Because the count is the only game state included, predictions reflect the hitter and the pitch rather than the game situation.

**Why.** The paragraph reads as a template (Each X, so Y). Alcorn's model section varies its openers ("The model takes...", "Here,...", "The batter and pitcher embeddings are then..."). This is the reader's first look at the model.

### 5. part2.tex:91 (3.3, pitch families). Each point is made twice, once mid-paragraph and once in the closer
**Current.** Their observed whiff rates on the two families correlate at 0.53, so the embedding keeps a distinction that is present in the data. For power, measured by 90th-percentile exit velocity, the two directions correlate at 0.94 and the observed values at 0.83, so power is close to a single direction in the embedding.

**Replace with.** Their observed whiff rates on the two families correlate at 0.53. For power, measured by 90th-percentile exit velocity, the two directions correlate at 0.94 and the observed values at 0.83.

**Why.** The closer that round 2 content added ("The embedding follows the data on both skills. Whiff rate differs by pitch family... power is close to a single skill in both.") already states both conclusions. Keeping it and cutting the two ", so" tails lets the numbers run, then gives the interpretation once, which is how Nate's results paragraphs work.

### 6. part3.tex:78 (3.7, level query). Two appositives between subject and verb
**Current.** Its correlation with 2024 wOBA, the selection season, after removing log prior PA, is 0.38 across 521 hitters (interval 0.30 to 0.45).

**Replace with.** After removing log prior PA, its correlation with wOBA in the 2024 selection season is 0.38 across 521 hitters (interval 0.30 to 0.45).

**Why.** "the selection season" (round 2 content) was inserted as a second interrupter, and the verb now arrives after 17 words.

### 7. part3.tex:70 (3.6, neighbors closer). Mid-sentence appositive, and "better read as X than as Y" twice in the paper
**Current.** The silhouette score, near zero when groups overlap, is at most 0.054 for any number of clusters from two to ten, so the space is better read as neighborhoods of similar hitters than as a set of hitter types.

**Replace with.** The silhouette score is near zero when groups overlap, and here it is at most 0.054 for any number of clusters from two to ten. The space holds neighborhoods of similar hitters, not a set of hitter types.

**Why.** part3.tex:89 uses the same "better read as a curve than as three groups" frame. A repeated hedge construction reads as generated. The short closer matches Alcorn's "At the least, ... appears to be capable of...".

### 8. part5.tex:11 (5). Stacked appositives and an interrupted "either ... or"
**Current.** Hitters absent from training, who now receive one of four generic projections, need a better starting point, either a prior built from what is known about them or, as \citet{alcorn2018} suggested, minor league data mapped into the same space.

**Replace with.** Hitters absent from training now receive one of four generic projections and need a better starting point. One option is a prior built from what is known about them. Another is minor league data mapped into the same space, as \citet{alcorn2018} suggested.

**Why.** Three comma-bound interrupters in one sentence. The split mirrors how Alcorn lists his own future directions ("For example, ... Likewise, ...").

### 9. part2.tex:58 (3.2, calibration). "-ing" tack-on
**Current.** A PA-weighted regression of each unit's observed 2025 wOBA on its projection gives a slope near one in the medium (0.98) and high (0.93) strata, meaning the projections there are spread about as widely as the outcomes they predict.

**Replace with.** A PA-weighted regression of each unit's observed 2025 wOBA on its projection gives a slope near one in the medium (0.98) and high (0.93) strata. There the projections are spread about as widely as the outcomes they predict.

**Why.** ", meaning ..." is the participial gloss that the humanizer rules flag. The next sentence ("In the low stratum the slope is 0.59, so...") then reads as a direct contrast.

### 10. part1.tex:91 (2.3, step one). "exit velocity, launch angle, and spray bins" three times in three sentences
**Current.** Each ball in play is thus valued by the league's average result for its exit velocity, launch angle, and spray bins, not by its actual result, which sets aside the fielder and park effects described in Section~\ref{sec:intro}.

**Replace with.** Each ball in play is thus valued by the league's average result for its bins, not by its actual result. This sets aside the fielder and park effects described in Section~\ref{sec:intro}.

**Why.** The two sentences before it already name the bins twice. The split also removes a trailing "which" clause. The link back to the intro that round 1 content asked for stays.

---

Not proposed, and left as is: the 2.4 opener, the "Three steps" list in Section 5, the 3.3 head-output closer (it interprets and does not only restate), and the remaining ", so" sentences. Each of those is a real cause-and-effect, and changing more would trade one template for another.
