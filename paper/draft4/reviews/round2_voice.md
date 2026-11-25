# Round 2 review: sentence-level voice

Lens: make each sentence read like Nate's NBA shot-value writeup and Alcorn (2018), and remove AI-writing tells. Round 1 fixes are not re-proposed. Several items below are sentences round 1 wrote or reshaped, which picked up tells of their own.

Overall the draft reads close to the target. The remaining problems fall into four groups. Paragraphs close on "X therefore Y" five times, which becomes a tic by Section 3.3. A few long sentences still carry an appositive between subject and verb or a stacked comma chain ("most clearly, and only in exploratory analysis, for"). Two Methods sentences say something slightly different from the math. "Rather than" appears eight times, mostly as a negative-parallel tack-on.

Every replacement keeps all numbers, adds no facts, uses no first person or gendered pronouns, and has no dashes, colons or semicolons in prose. The abstract, the Table 1 caption and the Figure 1 caption are untouched. Line numbers are from the .tex files.

---

### 1. part2.tex:78, part2.tex:100, part3.tex:65, part3.tex:89. "X therefore Y" as the paragraph closer, four times
**Problem.** Four results paragraphs in a row end on "The embedding therefore ...", "The platoon split is therefore ...", "The embedding therefore ...", "The level query therefore ...". Two of them only restate the paragraph's first sentence. Nate's own closers are "This indicates that ...", "suggesting ...", "As a result, ...", or simply ending on the last result. Keep the two mid-paragraph uses (part1.tex:66, part2.tex:13).

**(a) part2.tex:78. Current.** The embedding therefore organizes hitters by power and contact without being given either statistic.
**Replace with.** This indicates that the embedding organizes hitters by power and contact, although the model was never given either statistic.

**(b) part2.tex:100. Current (opener).** The embedding holds the platoon split more weakly.
**Current (closer).** The platoon split is therefore present in the embedding, but more weakly than power or contact.
**Replace with.** Opener becomes "The embedding also holds the platoon split, but more weakly than power or contact." Delete the closer. The paragraph then ends on the stratum correlations.

**(c) part3.tex:65. Current.** The embedding therefore keeps a pitch-family distinction for whiffs that it largely does not keep for power.
**Replace with.** Delete. The opener ("The embedding separates some skills by pitch type and not others.") already makes the point, and "that it largely does not keep" is awkward. See item 13 for the sentence that becomes the new last one.

**(d) part3.tex:89. Current.** The level query therefore carries information about a hitter's 2024 wOBA beyond how much the hitter had played.
**Replace with.** This suggests that the level query carries information about a hitter's 2024 wOBA beyond how much the hitter had played.

### 2. part2.tex:48 and part5.tex:7. "most clearly, and only in exploratory analysis, for"
**Current (a), part2.tex:48.** Taken together, the model ties both baselines on error and orders hitters better than both, most clearly, and only in exploratory analysis, for hitters with a moderate track record.

**Current (b), part5.tex:7.** On 2025 handedness splits it matched gradient boosting and empirical Bayes on error and ordered hitters better than both, most clearly, in an exploratory comparison, for hitters with a moderate track record against a hand.

**Problem.** Two commas of hedging wedged between "most clearly" and its "for" phrase. A reader loses the thread and has to reread to find what "most clearly" attaches to. (b) also says "gradient boosting" where the body says "the boosted baseline" throughout.

**Replace (a) with.** Taken together, the model ties both baselines on error and orders hitters better than both. The ordering gain is clearest for hitters with a moderate track record, although that comparison is exploratory.

**Replace (b) with.** On 2025 handedness splits it matched the boosted and empirical Bayes baselines on error and ordered hitters better than both. The gain was clearest for hitters with a moderate track record against a hand, although that comparison was exploratory.

### 3. part1.tex:31 (Intro, Tango). Four clauses in one sentence, "split" three times
**Current.** In \citet[p.~157]{tango2007}, a right-handed hitter needs about 2,200 PA against left-handed pitchers before the observed split receives half the weight, and a left-handed hitter about 1,000, so for most hitters the estimate stays close to the league-average split, and two part-time hitters from the same side receive nearly the same projected split.

**Problem.** 58 words, an elided verb ("a left-handed hitter about 1,000"), then "so ... and ..." chained on. The consequence deserves its own sentence, and "As a result" is Nate's connector.

**Replace with.** In \citet[p.~157]{tango2007}, a right-handed hitter needs about 2,200 PA against left-handed pitchers before the observed split receives half the weight, and a left-handed hitter needs about 1,000. As a result, the estimate for most hitters stays close to the league-average split, and two part-time hitters from the same side receive nearly the same projection.

### 4. part1.tex:25 (Intro, paragraph 1). Stacked appositives and a broken subject
**Current.** The decisions that leave room for an edge involve the rest of the roster, the bench and platoon hitters, and they usually depend on a split, meaning how a hitter performs against one kind of pitcher. Choosing which of two bench hitters starts against a left-handed pitcher, or which pinch hitter to send up against the reliever warming up, are decisions of this kind.

**Problem.** The first sentence carries two appositives ("the rest of the roster, the bench and platoon hitters" and "meaning how ..."). In the second, "or which pinch hitter" has no verb of its own, and the gerund subject takes a plural "are". A careful reader stops on it in the first paragraph of the paper.

**Replace with.** The decisions that leave room for an edge involve the bench and platoon hitters, and they usually depend on a split, which is how a hitter performs against one kind of pitcher. Examples include choosing which of two bench hitters starts against a left-handed pitcher and which pinch hitter to send up against the reliever warming up.

### 5. part1.tex:93 (2.3, step two). The recursion is described without its sum
**Current.** Each pitch moves the count to a new state or ends the plate appearance, so the expected wOBA from any count is the chance of each pitch outcome times the expected wOBA of where it leads.

**Problem.** As written it is one product, not a sum over outcomes, and "where it leads" is casual. An expert reading it beside Equation 2 will notice the mismatch.

**Replace with.** Each pitch moves the count to a new state or ends the plate appearance, so the expected wOBA from any count is a sum over the possible pitch outcomes. Each term is the probability of that outcome times the expected wOBA of the count or result it leads to.

### 6. part1.tex:91 (2.3, step one). 25-word subject before the verb
**Current.** For pitcher $j$ and each count, six real pitches that pitcher threw in that count to hitters who bat from the same side as $h$ during the training seasons are run through the network for hitter $h$ and averaged.

**Problem.** The reader holds "six real pitches that ... during the training seasons" before reaching "are run". "during the training seasons" also reads as if it modifies "bat".

**Replace with.** For pitcher $j$ and each count, the network is run for hitter $h$ on six real pitches that pitcher $j$ threw in that count during the training seasons to hitters who bat from the same side as $h$, and the six outputs are averaged.

### 7. part2.tex:46 (3.2, medium stratum). Compressed numbers and "gain" twice
**Current.** In wOBA terms the error gain is small, about 2 and 4 points against a typical error near 45, so the gain for these hitters lies mostly in ordering them.

**Problem.** "about 2 and 4 points against a typical error near 45" does not say which baseline each number belongs to or that 45 is in points. "gain ... gain" in one sentence.

**Replace with.** In wOBA terms the error gain is small, about 2 points over the boosted baseline and 4 over empirical Bayes against a typical error near 45 points. For these hitters the model's advantage lies mostly in ordering them.

(Checked against Table 1. 0.0479 minus 0.0455 is 2.4 points, 0.0492 minus 0.0455 is 3.7.)

### 8. part2.tex:44 (3.2, pooled result). Three numbers in an appositive
**Current.** Across all 1,157 units the model ties both baselines on error, 0.0452 against 0.0453 for the boosted baseline and 0.0460 for empirical Bayes, and both difference intervals include zero.

**Problem.** The appositive carries the three RMSE values between the claim and its interval check. Round 1 split the matching rank-correlation sentence right after it, so the two now read unevenly.

**Replace with.** Across all 1,157 units the model ties both baselines on error. Its RMSE is 0.0452 against 0.0453 for the boosted baseline and 0.0460 for empirical Bayes, and both difference intervals include zero.

### 9. part2.tex:58 (3.2, calibration). Definition wedged between subject and verb
**Current.** On 2024, its nominal 95\% intervals, built from the spread of the five seeds and each unit's sampling noise, covered the observed value for 87\% of 1,149 units.

**Problem.** Same pattern round 1 fixed in the head-output sentence. Define, then report.

**Replace with.** The intervals are built from the spread across the five seeds and each unit's sampling noise. On 2024, the nominal 95\% intervals covered the observed value for 87\% of 1,149 units.

### 10. part2.tex:11 (3.1). Tacked-on appositive, "spread" three times
**Current.** Comparing the spread of hitters' observed values with the spread that sampling noise alone would produce gives the share of the spread that is real, the same quantity as the reliability in Section~\ref{sec:eval}. That share sets the highest rank correlation any method could reach against those outcomes.

**Problem.** The appositive is tacked on after an already long sentence. Moving it into the next sentence gives a plain copula and keeps both facts.

**Replace with.** Comparing the spread of hitters' observed values with the spread that sampling noise alone would produce gives the share of the spread that is real. This share is the same quantity as the reliability in Section~\ref{sec:eval}, and it sets the highest rank correlation any method could reach against those outcomes.

### 11. part2.tex:13 (3.1). "sits at" and a "with" tack-on
**Current.** The model's RMSE for the split, 0.0633, sits at the 0.0639 floor that sampling noise sets, with empirical Bayes about as close.

**Problem.** "sits at" avoids "is" (humanizer 8), and 0.0633 is slightly below the floor, so "at" is loose. "with empirical Bayes about as close" is a tack-on with no verb.

**Replace with.** The model's RMSE for the split is 0.0633, close to the 0.0639 floor that sampling noise sets, and empirical Bayes is about as close.

### 12. part2.tex:69 (3.2, split grading). "empirical Bayes's"
**Current.** Across the 539 hitters with at least 10 PA against each hand in 2025, the model's projection was closer for 296 (55\%, interval 51\% to 59\%) and empirical Bayes's for 243.

**Problem.** The possessive "empirical Bayes's" plus an elided noun is hard to read aloud. Round 1 removed the same possessive from the pooled-result sentence.

**Replace with.** Across the 539 hitters with at least 10 PA against each hand in 2025, the model's projection was closer for 296 (55\%, interval 51\% to 59\%) and the empirical Bayes projection for 243.

### 13. part3.tex:65 (4.4, pitch families). "stand much worse" and a which-and chain
**Current (a).** Relative to other hitters, they stand much worse against one pitch family than the other.

**Replace with.** Relative to other hitters, they rank much worse against one pitch family than against the other.

**Current (b).** Exit velocity is observed only on balls in play, which leaves fewer labels per pitch family than swings and contact have and may account for the difference between the two skills.

**Replace with.** Exit velocity is observed only on balls in play, so it has fewer labels per pitch family than swings and contact do. This may account for the difference between the two skills.

**Problem.** (a) "stand ... worse" is not idiomatic. (b) "which leaves ... and may account" hangs two predicates on one "which". With item 1(c) the paragraph ends on (b), a result followed by a modest interpretation, which is Nate's pattern.

### 14. part3.tex:79 (4.5, neighbors). Dangling opener plus a stacked "-ing" phrase
**Current.** Ranked by ten observed statistics instead, covering plate discipline, contact quality, and wOBA, Trout's eight nearest hitters include Yordan Álvarez, Paul Goldschmidt, Juan Soto, and Mookie Betts, and only Ronald Acuña appears on both lists.

**Problem.** "Ranked by ... instead, covering ..." stacks two modifiers before the subject. "instead" in the middle of the opener is easy to miss.

**Replace with.** When hitters are instead ranked by ten observed statistics covering plate discipline, contact quality, and wOBA, Trout's eight nearest include Yordan Álvarez, Paul Goldschmidt, Juan Soto, and Mookie Betts, and only Ronald Acuña appears on both lists.

### 15. part3.tex:81 (4.5). "Neighborhoods are similar in skill"
**Current.** Neighborhoods are similar in skill but do not form separate groups.

**Problem.** A neighborhood is not similar in skill. The hitters inside it are. An expert reader trips on the subject.

**Replace with.** Hitters within a neighborhood are similar in skill, but the neighborhoods do not form separate groups.

### 16. part1.tex:114 and part3.tex:98. "rather than" as a tack-on
**Problem.** "rather than" appears eight times. Most are fine, but two are tacked-on negations that add no information.

**Current (a), part1.tex:114.** The cuts are half and twice $n^*$, a simple convention rather than a tuned choice.
**Replace with.** The cuts were set at half and twice $n^*$ by convention and were not tuned.

**Current (b), part3.tex:98.** Part of this is a property of the data rather than the model.
**Replace with.** Part of this is a property of the data.
(The next sentence already says "for any method", so the contrast is carried there.)

### 17. part1.tex:46 (2.1, data). "-ing" tack-on and a tense switch
**Current.** Pitchouts, automatic balls and strikes, bunts, pitches thrown by position players, pitches to pitchers batting, and pitches missing a core measurement were removed, leaving 7,245,923 pitches. No minimum exposure was imposed, so every hitter in the training seasons receives an embedding, and hitters absent from them share one reserved embedding.

**Problem.** "leaving" is a tack-on. Alcorn states the count as its own sentence ("The final data set consisted of 461,231 at-bats ..."). The next sentence switches from "was imposed" to "receives", and "absent from them" can point at the hitters or the seasons.

**Replace with.** Pitchouts, automatic balls and strikes, bunts, pitches thrown by position players, pitches to pitchers batting, and pitches missing a core measurement were removed. The final data set consisted of 7,245,923 pitches. No minimum exposure was imposed, so every hitter in the training seasons has an embedding. Hitters who do not appear in the training seasons share one reserved embedding.

### 18. part1.tex:66 (2.2, taken pitches). Center-embedded phrase
**Current.** Their outcomes therefore come from a table, fit on the training seasons, of the probability of a ball, a called strike, or a hit batter by plate location and by the pairing of batter and pitcher hand.

**Problem.** "fit on the training seasons" splits "a table" from "of the probability".

**Replace with.** Their outcomes therefore come from a table fit on the training seasons, which gives the probability of a ball, a called strike, or a hit batter for each plate location and pairing of batter and pitcher hand.

### 19. part1.tex:107 (2.3). "every such hitter of a batting side"
**Current.** A hitter absent from the training seasons has only the shared reserved embedding, so every such hitter of a batting side receives the same projection against a pitcher hand.

**Problem.** "every such hitter of a batting side" is hard to parse.

**Replace with.** A hitter absent from the training seasons has only the shared reserved embedding, so all such hitters who bat from the same side receive the same projection against a given pitcher hand.

### 20. part1.tex:127, footnote (round 1 wording). "where" as a contrast, and a tense switch
**Current.** No comparison changed whether its interval excluded zero, although its low-stratum error differences nominally favored the baselines, where the final model's nominally favor the model.

**Problem.** "where" used as "whereas", an elided noun ("the final model's"), and favored/favor in one clause pair.

**Replace with.** No comparison changed whether its interval excluded zero. Its low-stratum error differences nominally favored the baselines, while those of the final model nominally favor the model.

---

## Minor, optional

- **part3.tex:52 and :54. "edge".** "an average edge of about 30 to 40 points" and "Pederson's average edge on contact is 44 points" use "edge" as a term the figure never defines, and the intro already uses "edge" for a team's advantage. Replace with "an average of about 30 to 40 points above the average hitter" and "On contact, Pederson averages 44 points above the average hitter against left-handed pitchers and 48 against right-handed pitchers."
- **part5.tex:11. "Three extensions" right after "Three steps".** Two numbered openers back to back read as a template (humanizer 10). Replace "Three extensions would widen what the model can answer." with "Other extensions would widen what the model can answer."
- **part1.tex:27. "move a wOBA outcome".** Replace "all move a wOBA outcome without reflecting anything the hitter did" with "all change a plate appearance's wOBA value without reflecting anything the hitter did."
- **Rhythm.** Paragraph lengths and sentence lengths are in Nate's range. The splits above (items 3, 5, 8, 9, 13, 17) add a few short sentences where the draft runs long, so no further short sentences need to be manufactured.
