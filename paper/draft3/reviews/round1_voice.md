# Round 1 review: sentence-level voice

Lens: make each sentence read like Nate's NBA shot-value writeup and Alcorn (2018), and remove AI-writing tells. Structure and numbers are not reviewed here except where a sentence's wording misstates them.

Overall the draft is already close to the target register. It uses "can be seen in Figure", plain passive methods sentences, and result-then-interpretation ("so...", "which would explain..."), all of which match both references. The problems are mostly local. There are gendered pronouns in 4.2 and 4.4, a few sentences where the wording says something different from what is meant (an expert will trip on these), sentences carrying six to eight numbers, and long comma chains that end in a tacked-on appositive or "-ing" phrase.

All replacements below keep every number unless noted, add no facts, use no first person, no gendered pronouns, and no dashes, colons or semicolons in prose. The abstract and the Table 1 caption are untouched.

Line numbers are from the .tex files.

---

## Rule violations

### 1. part3.tex:72 (4.4, Pederson head outputs). Gendered pronouns
**Current.** His head outputs show where his projected platoon gap comes from instead. He swings less than an average hitter against both hands, and that shortfall is larger against left-handed pitchers, so the gap lies more in his swing decisions than in the quality of his contact.

**Problem.** "his/He" five times, which breaks the no-gendered-pronoun rule. "show ... instead" also has nothing clear to be instead of.

**Replace with.** The head outputs place the projected platoon gap elsewhere. Pederson swings less than an average hitter against both hands, and the shortfall is larger against left-handed pitchers, so the gap lies more in swing decisions than in the quality of contact.

### 2. part3.tex:41 (4.2, case-study results). Gendered pronouns
**Current (a).** Soto, expected to be flat, projects 0.041 worse against left-handed pitchers, and his observed 2025 split, $-0.074$, runs the same way.

**Replace with.** Soto, expected to be flat, projects 0.041 worse against left-handed pitchers, and Soto's observed 2025 split of $-0.074$ runs the same way.

**Current (b).** With 19 PA against left-handed pitchers in 2025, his observed split of $+0.172$ has a standard error of 0.121, wide enough to hold either projection and the opposite sign.

**Replace with.** With 19 PA against left-handed pitchers in 2025, Pederson's observed split of $+0.172$ has a standard error of 0.121, wide enough to hold either projection and the opposite sign.

**Problem.** "his" in both. Repeating the name is how Nate handles this in his essays (blunt repetition over pronouns).

---

## Sentences an expert reader will misread

### 3. part1.tex:74 (2.2, loss definition). "split" means two things
**Current.** where for pitch $i$, $a_i$ is the swing decision, $c_i$ the contact outcome, $t_i$ the foul, foul tip, or ball in play split, and $e_i$, $l_i$, and $s_i$ the exit velocity, launch angle, and spray bins.

and

Each mask $m_i$ is one when the outcome exists and was measured, so contact is scored only on swings, the split only on contact, and the three quality terms only on measured balls in play.

**Problem.** Everywhere else in the paper "split" is the platoon split. Here it names the foul / foul tip / ball-in-play head. A reader hits "the split only on contact" and has to stop.

**Replace with.** where for pitch $i$, $a_i$ is the swing decision, $c_i$ the contact outcome, $t_i$ the contact type (foul, foul tip, or ball in play), and $e_i$, $l_i$, and $s_i$ the exit velocity, launch angle, and spray bins.

and

Each mask $m_i$ is one when the outcome exists and was measured, so contact is scored only on swings, contact type only on contact, and the three quality terms only on measured balls in play.

### 4. part2.tex:78 (3.2, split grading). Grading reads backwards, and the errors are unlabeled
**Current (a).** A platoon decision depends on the gap between a hitter's two sides, so each hitter's 2025 split was also graded against the model's projected split and the empirical Bayes split.

**Problem.** As written, the observed split is the thing being graded and the projections are the key. It should be the other way round.

**Replace with.** A platoon decision depends on the gap between a hitter's two sides, so the model's projected split and the empirical Bayes split were also graded against each hitter's observed 2025 split.

**Current (b).** Weighted by the harmonic mean of each hitter's PA against the two hands, the mean absolute errors are 0.0448 and 0.0462, a difference of 0.0014 (interval 0.0002 to 0.0026).

**Problem.** The sentence never says which error belongs to which method. The reader has to infer the order from the previous sentence.

**Replace with.** Weighted by the harmonic mean of each hitter's PA against the two hands, the model's mean absolute error is 0.0448 against 0.0462 for empirical Bayes, a difference of 0.0014 (interval 0.0002 to 0.0026).

(This takes 0.0448 as the model's error, following the order in the source comment, `wMAE 0.0448/0.0462`. Confirm before applying.)

### 5. part3.tex:11 (4.1, exposure curve). "rank hitters ahead of both baselines"
**Current.** With little exposure, the model's point estimates still rank hitters ahead of both baselines, but its projections are spread more widely than the outcomes support (calibration slope 0.59), and the stratum holds too few batters to confirm the margin.

**Problem.** This reads as if hitters were ranked above the baselines. The intended meaning is that the model orders hitters better than the baselines do.

**Replace with.** With little exposure, the model's point estimates still order hitters better than both baselines do, but its projections are spread more widely than the outcomes support (calibration slope 0.59), and the stratum holds too few batters to confirm the margin.

### 6. part1.tex:109 (2.4, evaluation). Dangling "called a unit"
**Current.** The handedness query is scored on a hitter's wOBA against one pitcher hand in 2025, called a unit.

**Problem.** It is not clear what "called a unit" attaches to (the hand, the season, or the hitter-hand pair). This term is used for the rest of the paper, so it should be defined cleanly.

**Replace with.** The handedness query is scored on units, where a unit is one hitter's wOBA against one pitcher hand in 2025.

### 7. part1.tex:115, footnote (2.4). Direction of the shift is unclear
**Current.** 2025 was also scored by a version of the model with a 32-dimension embedding that kept pitches to pitchers batting. No comparison changed whether its interval excluded zero, although the low-stratum error differences moved from nominally favoring the baselines to nominally favoring the model.

**Problem.** "moved from X to Y" does not say which version is X. The sentence also opens on a numeral.

**Replace with.** A version of the model with a 32-dimension embedding, trained with pitches to pitchers batting left in, was also scored on 2025. No comparison changed whether its interval excluded zero, although the low-stratum error differences nominally favored the baselines under that version and nominally favor the model under the final one.

### 8. part1.tex:115 (2.4). Tense in the pass rule
**Current.** Before the 2025 model was trained, the low stratum was specified as the primary test, which the model passed only if it beat the boosted baseline on RMSE and both baselines on rank correlation.

**Problem.** "which the model passed only if" mixes a past result with a conditional rule and reads as if it passed. Two sentences are cleaner.

**Replace with.** Before the 2025 model was trained, the low stratum was specified as the primary test. To pass it, the model had to beat the boosted baseline on RMSE and both baselines on rank correlation.

---

## Sentences carrying too many numbers

### 9. part2.tex:55 (3.2, pooled result). Seven numbers in one sentence
**Current.** It orders hitters better than both, with a weighted rank correlation of 0.47 against the boosted baseline's 0.42 (difference interval 0.011 to 0.089) and empirical Bayes's 0.40 (0.015 to 0.125).

**Problem.** Seven numbers and an awkward "empirical Bayes's".

**Replace with.** It orders hitters better than both, with a weighted rank correlation of 0.47 against 0.42 for the boosted baseline and 0.40 for empirical Bayes. The intervals on those two differences, 0.011 to 0.089 and 0.015 to 0.125, both exclude zero.

### 10. part2.tex:57 (3.2, medium stratum). Eight numbers and stacked appositives
**Current.** The clearest signal, an exploratory result, is in the medium stratum, hitters with 113 to 451 prior PA against a hand. There the model beats both baselines on both measures, with rank correlation 0.42 against 0.28 and 0.22 and RMSE 0.0455 against 0.0479 and 0.0492 (5\% and 7\% lower), and all four intervals exclude zero.

**Problem.** Two appositives in a row in the first sentence. The second carries eight numbers, and "0.28 and 0.22" never says which baseline is which. The raw RMSE values are already in Table 1, so cutting them here is honest.

**Replace with.** The clearest signal is in the medium stratum, hitters with 113 to 451 prior PA against a hand, although this comparison is exploratory. There the model beats both baselines on both measures, and all four intervals exclude zero. Its rank correlation is 0.42 against 0.28 for the boosted baseline and 0.22 for empirical Bayes, and its RMSE is 5\% and 7\% lower than theirs (Table~\ref{tab:test}).

---

## Awkward constructions and long chains

### 11. part1.tex:24 (1, intro paragraph 1). Unclear "These", and "wOBA takes PA against them"
**Current.** These are also the hitters with the least data. In the training seasons, a hitter's wOBA against left-handed pitchers takes about 226 plate appearances (PA) against them before half of its variation across hitters reflects skill rather than noise (Section~\ref{sec:ceiling}), and a part-time hitter can take several seasons to get there.

**Problem.** "These" follows a sentence that ends on "one kind of pitcher", so it could point to splits or to pitchers. "wOBA ... takes ... PA against them" gives the PA to the statistic, says "against" twice, and has "its" referring back to wOBA.

**Replace with.** Bench and platoon hitters are also the ones with the least data. In the training seasons, it takes about 226 plate appearances (PA) against left-handed pitchers before half of the variation in hitters' wOBA against them reflects skill rather than noise (Section~\ref{sec:ceiling}), and a part-time hitter can take several seasons to reach that many.

### 12. part1.tex:34 (1, last paragraph). "so" claims the wrong cause
**Current.** The model learns from every pitch a hitter has seen and shares structure across hitters through the embedding, so a new split requires a new set of pitcher weights rather than a retrained model.

**Problem.** Sharing structure is not why a new split needs only new weights. The reason is that the query is a weighted average over pitchers (eq. 3). An expert will notice the non sequitur.

**Replace with.** The model learns from every pitch a hitter has seen and shares structure across hitters through the embedding. A projection is a weighted average over pitchers, so a new split requires a new set of pitcher weights rather than a retrained model.

### 13. part3.tex:9 (4.1). Three clauses with three subjects
**Current.** Empirical Bayes sits just as close to the error floor for the split (Section~\ref{sec:ceiling}), so one season of outcomes cannot show much skill at sizing a split for any method, and the model's intervals are too narrow to size one reliably.

**Problem.** The paragraph says this is "a property of the data". The last clause is a property of the model, but it is chained on as if it followed from the data point.

**Replace with.** Empirical Bayes sits just as close to the error floor for the split (Section~\ref{sec:ceiling}), so one season of outcomes cannot show much skill at sizing a split for any method. The model's intervals are also too narrow to size one reliably.

### 14. part2.tex:22 (3.1). Vague "idea" and an elliptical second clause
**Current.** The same idea gives a ceiling for each target. Comparing the spread of hitters' observed values with the spread that sampling noise alone would produce gives the share of the spread that is real, and from that share the highest rank correlation any method could reach against those outcomes.

**Problem.** "idea" is an abstract noun standing in for "reliability". The second sentence drops its verb after "and from that share", so the reader has to supply "gives" a second time.

**Replace with.** Reliability also gives a ceiling for each target. Comparing the spread of hitters' observed values with the spread that sampling noise alone would produce gives the share of the spread that is real. That share sets the highest rank correlation any method could reach against those outcomes.

### 15. part2.tex:24 (3.1). "one season of them", plus a tacked-on appositive
**Current (a).** True splits vary little across hitters compared with the noise in one season of them, so even a perfect projection would reach a rank correlation of only about 0.30 with the observed split.

**Replace with.** True splits vary little across hitters compared with the noise in one season's observed split, so even a perfect projection would reach a rank correlation of only about 0.30 with it.

**Current (b).** The 2025 test in the next section scores each side's wOBA separately, a target that falls between these two, with more real variation across hitters than the split and fewer PA behind each value than the level.

**Replace with.** The 2025 test in the next section scores each side's wOBA separately. This target falls between the other two, with more real variation across hitters than the split and fewer PA behind each value than the level.

**Problem.** (a) "one season of them" is clumsy. (b) Two appositives chained onto the main clause make a 40-word sentence that loses its subject.

### 16. part1.tex:63 (2.2). Six heads in one nested list
**Current.** They give the probability of a swing, the probability of contact given a swing, whether contact is a foul, a foul tip, or a ball in play (a two-strike foul tip is a strikeout and a foul is not), and the exit velocity, launch angle, and spray direction of a ball in play, each in 24 bins.

**Problem.** There are lists inside a list plus a parenthetical, and it is unclear what "each in 24 bins" covers.

**Replace with.** They give the probability of a swing, the probability of contact given a swing, and whether contact is a foul, a foul tip, or a ball in play (a two-strike foul tip is a strikeout and a foul is not). The other three give the exit velocity, launch angle, and spray direction of a ball in play, each in 24 bins.

### 17. part1.tex:86 (2.3, query step one). Hard appositive and a split verb
**Current.** This gives, for each count, the probability that the pitch ends as a ball, a strike, a foul, a hit batter, or a ball in play, and the expected wOBA from a ball in play, its probability times its average value. The average value sums, over every combination of exit velocity, launch angle, and spray bins, the hitter's predicted probability of that combination times the league's average wOBA for it.

**Problem.** "its probability times its average value" is an appositive with two "its". "sums, over ..., the hitter's" separates the verb from its object by 12 words.

**Replace with.** This gives, for each count, the probability that the pitch ends as a ball, a strike, a foul, a hit batter, or a ball in play. The expected wOBA from a ball in play is that probability times the ball in play's average value. The average value adds up the league's average wOBA for every combination of exit velocity, launch angle, and spray bins, each weighted by the hitter's predicted probability of that combination.

(Also in this paragraph, "to hitters of the same side" could be "to hitters who bat from the same side as $h$". It is the same fact, and the comparison is now explicit.)

### 18. part2.tex:99 (3.3, head outputs). Definition wedged between verb and object
**Current.** Figure~\ref{fig:headpartial} correlates each head's hitter-specific output, its average prediction for the hitter over a fixed set of 2,000 training-season pitches, with the hitter's observed training-season statistics, after removing log prior PA.

**Replace with.** A head's hitter-specific output is its average prediction for the hitter over a fixed set of 2,000 training-season pitches. Figure~\ref{fig:headpartial} correlates this output with the hitter's observed training-season statistics after removing log prior PA.

**Problem.** The reader carries "correlates ..." across a 17-word appositive before reaching "with". Defining the output first matches the Alcorn habit of defining a quantity, then using it.

### 19. part2.tex:111 (3.3, platoon direction). 26-word subject, plus synonym cycling
**Current.** The direction in the embedding along which the projected left-minus-right gap grows, found by perturbing each embedding dimension, has a cosine of 0.14 (interval 0.08 to 0.20) with a direction fit separately to decode the empirical Bayes split, so the two are related but far from the same. A hitter's position along the model's direction correlates with the observed split in the training seasons, after controlling for log prior PA, at 0.07 in the low stratum, 0.14 in the medium, and 0.13 in the high, each with an interval above zero.

**Problem.** The main verb arrives after 26 words. "after controlling for log prior PA" is the only place that does not say "after removing log prior PA" (3.3 head outputs, 4.6). That is synonym cycling, and a reader may wonder whether it is a different adjustment.

**Replace with.** Perturbing each embedding dimension gives the direction along which the projected left-minus-right gap grows. This direction has a cosine of 0.14 (interval 0.08 to 0.20) with a separate direction fit to decode the empirical Bayes split, so the two are related but far from the same. A hitter's position along the model's direction correlates with the observed split in the training seasons, after removing log prior PA, at 0.07 in the low stratum, 0.14 in the medium, and 0.13 in the high, each with an interval above zero.

---

## AI tells (-ing tack-ons, copula avoidance, dangling openers)

### 20. part3.tex:48 and :50 (4.3, disagreements). -ing tack-on, dangling opener, elliptical chain
**Current (a).** Across 545 hitters in 2024, the two methods' projected splits have a Spearman correlation of 0.77, falling from 0.80 and 0.79 in the low and medium strata to 0.69 in the high stratum (Figure~\ref{fig:disagreement}).

**Replace with.** Across 545 hitters in 2024, the two methods' projected splits have a Spearman correlation of 0.77. The correlation is 0.80 and 0.79 in the low and medium strata and 0.69 in the high stratum (Figure~\ref{fig:disagreement}).

**Current (b).** Ranking within batting side, the largest disagreement is Robbie Grossman, first among right-handed hitters by empirical Bayes and 324th by the model.

**Replace with.** With hitters ranked within batting side, the largest disagreement is Robbie Grossman, first among right-handed hitters by empirical Bayes and 324th by the model.

**Current (c).** Most of these observed splits have a standard error between 0.04 and 0.06, and a few near 0.1, at least the size of the gaps being graded, so the cases show how the two methods differ rather than which is right.

**Replace with.** Most of these observed splits have a standard error between 0.04 and 0.06, and a few have one near 0.1. These errors are at least the size of the gaps being graded, so the cases show how the two methods differ rather than which is right.

**Problem.** (a) "falling from ..." is the -ing tack-on pattern and carries four numbers. (b) The disagreement is not "ranking", so the opener dangles. (c) The verb is dropped after "and a few", then an appositive is chained on.

### 21. part3.tex:83 (4.4, pitch families). -ing tack-on, dangling appositive, abstract noun
**Current (a).** For whiff rate the two directions correlate at 0.64 across hitters, and many hitters sit well off the diagonal, standing much worse against one pitch family than the other relative to other hitters.

**Replace with.** For whiff rate the two directions correlate at 0.64 across hitters, and many hitters sit well off the diagonal. Relative to other hitters, they stand much worse against one pitch family than the other.

**Current (b).** For power, measured by 90th-percentile exit velocity, the two directions correlate at 0.94 and the observed values at 0.83, close to a single direction.

**Replace with.** For power, measured by 90th-percentile exit velocity, the two directions correlate at 0.94 and the observed values at 0.83, so power is close to a single direction in the embedding.

**Current (c).** Both correlations in the embedding run higher than the observed ones, partly because the two directions share the space's dominant variation.

**Replace with.** Both correlations in the embedding run higher than the observed ones, partly because the two directions share the embedding's main direction of variation.

**Problem.** (a) "standing ..." tacked on. (b) "close to a single direction" has no subject. (c) "the space's dominant variation" is a new name for what 3.3 calls "the main direction of variation".

### 22. part3.tex:18 (4.2). Copula avoidance
**Current.** Five well-known hitters serve as case studies, each with an expected platoon direction fixed before any projection was examined.

**Problem.** "serve as" is humanizer pattern 8.

**Replace with.** The case studies are five well-known hitters, each with an expected platoon direction fixed before any projection was examined.

### 23. part3.tex:105 and :107 (4.6, other queries). Abstract phrasing, number pile-up, 25-word subject
**Current (a).** Two others show what the approach supports and what limits checking it.

**Replace with.** Two others show what the approach supports and how far one season of outcomes can check it.

**Current (b).** Its correlation with held-out 2024 wOBA, after removing log prior PA, is 0.38 across 521 hitters (interval 0.30 to 0.45), and it rises with exposure, from 0.17 in the low stratum to 0.38 in the medium and 0.52 in the high (Figure~\ref{fig:levelquery}).

**Replace with.** Its correlation with held-out 2024 wOBA, after removing log prior PA, is 0.38 across 521 hitters (interval 0.30 to 0.45). The correlation rises with exposure, from 0.17 in the low stratum to 0.38 in the medium and 0.52 in the high (Figure~\ref{fig:levelquery}).

**Current (c).** A hitter's observed wOBA against the top third of right-handed pitchers by fastball velocity, minus that against the bottom third, has a split-half reliability of 0.08 across 301 hitters in 2024 (interval $-0.15$ to 0.26), and the same right-handed contrasts for fastball height and breaking-ball share are no higher.

**Replace with.** Checking it needs an observed contrast, such as a hitter's wOBA against the top third of right-handed pitchers by fastball velocity minus the wOBA against the bottom third. Across 301 hitters in 2024 this contrast has a split-half reliability of 0.08 (interval $-0.15$ to 0.26), and the same right-handed contrasts for fastball height and breaking-ball share are no higher.

**Problem.** (a) "what limits checking it" is abstract and hard to parse. (b) Seven numbers in one sentence. (c) The subject runs 25 words before "has".

### 24. part5.tex:7 and :9 (5). Misplaced appositive, tack-on, and "Two limits" followed by three
**Current (a).** The most direct application is a matchup card that ranks a team's options against a pitcher hand, the query tested here, with a band around each projection.

**Replace with.** The most direct application is a matchup card that ranks a team's options against a pitcher hand, the query tested here, and shows a band around each projection.

**Current (b).** ...can be set beside those of the hitters already on the roster, a single-split version of the counterfactual roster comparison that \citet{alcorn2018} proposed for player embeddings.

**Replace with.** ...can be set beside those of the hitters already on the roster. This is a single-split version of the counterfactual roster comparison that \citet{alcorn2018} proposed for player embeddings.

**Current (c).** The prediction intervals are too narrow and need recalibration before a card can show a band, and the moderate-exposure advantage was found in exploratory analysis and needs a confirmatory test, which may need to pool 2026 with later seasons to reach 80\% power. The low-exposure margin is positive but underpowered, and confirming it at its observed size would take far more batters than one season holds.

**Replace with.** The prediction intervals are too narrow and need recalibration before a card can show a band. The moderate-exposure advantage was found in exploratory analysis and needs a confirmatory test, which may have to pool 2026 with later seasons to reach 80\% power. The low-exposure margin is a separate question. It is positive but underpowered, and confirming it at its observed size would take far more batters than one season holds.

**Problem.** (a) "with a band" reads as attached to "the query tested here". (b) Tacked-on appositive. (c) Two limits are joined by "and" into one 45-word sentence with "need ... needs ... need". A third item then follows "Two limits" without being marked as separate.

### 25. part2.tex:89 (3.3, classifiers). Number pile-up, a casual tack-on, and "emerge"
**Current (a).** Across 1,493 hitters it recovers power (90th-percentile exit velocity) for 79\% and contact rate on swings for 75\%, against 33\% by chance, with in-zone swing rate (75\%) and chase rate (73\%) close behind.

**Replace with.** Across 1,493 hitters it recovers the power third (90th-percentile exit velocity) for 79\% of hitters and the contact-rate third (contact on swings) for 75\%, against 33\% by chance. In-zone swing rate (75\%) and chase rate (73\%) are recovered nearly as well.

**Current (b).** Hitter-level skills therefore emerge from pitch-level training without being defined for the model.

**Replace with.** The model therefore learns hitter-level skills from pitch-level training without being given their definitions.

**Problem.** (a) Six numbers, and "close behind" is a sportswriting tack-on that does not fit a paper. (b) "emerge" is a mild AI-vocabulary tell, and abstract-noun subject. The replacement echoes the author's own abstract ("Without being given definitions of power or contact").

---

## Minor, optional

- **part3.tex:72.** "Trout's best results come low, peaking at about 200 to 220 points above average on low pitches over the middle." The -ing tack-on means "low" is said twice. Replace with "Trout's best results come on low pitches over the middle, at about 200 to 220 points above average."
- **part3.tex:98.** "The neighborhoods also overlap." There is nothing before it for "also" to add to. Replace with "The neighborhoods overlap."
- **part1.tex:47.** "Feature scaling was fit on 2015 through 2023 only, and the baselines see only seasons before the one being scored." The tense switches mid-sentence. Replace with "...and the baselines saw only seasons before the one being scored."
- **part2.tex:94, Figure (power/contact) caption.** "Left: observed power tercile. Right: mean observed wOBA per cell." This is the only prose with colons outside the Table 1 caption. It is the author's own caption from abstract.tex, so it is left alone here. If the no-colon rule is meant to cover it, "Left, observed power tercile. Right, mean observed wOBA per cell." would match the other captions.
- **Rhythm, whole draft.** Most paragraphs run five to seven sentences of 25 to 40 words, which is Nate's normal range. The splits in items 9, 10, 14, 15, 17 and 23 also add the occasional short sentence the NBA writeup has ("The decision tree provides a simple nonlinear model."). No further short sentences need to be manufactured.
