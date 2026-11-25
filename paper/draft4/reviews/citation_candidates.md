# Citation candidates for draft4/main.tex

Checked: 2026-11-17. Scope: prior work the paper relies on but does not cite. The 14 existing entries in refs.bib were not re-checked or touched. No edits were made to main.tex or refs.bib.

Rule applied: a candidate is ADD only if I read the source's text and it supports the specific sentence. Sources: the `~/os/knowledge/library/baseball-research/` folder first. After that, a small number of web reads: the arXiv PDFs of Powers & Yurko and Deshpande & Wyner, three Sloan PDFs and pages, and the MLB xwOBA glossary.

## Summary

| # | Source | Attaches to | Verdict |
|---|--------|-------------|---------|
| 1 | Melville et al. 2023 (SSAC) | L79 | ADD |
| 2 | Powers & Yurko 2025 (arXiv) | L139 | ADD |
| 3 | Deshpande & Wyner 2017 (JQAS) | L112 | ADD |
| 4 | MLB.com xwOBA glossary | L137 (also L74) | ADD |
| 5 | Yonushonis 2011 (BRJ) | L144 | OPTIONAL |
| 6 | Heaton & Mitra 2022 / 2023 (SSAC) | new clause at L79 | OPTIONAL |
| 7 | Kneita 2025 (SSAC; was `anon-2025-transformer-pitch-outcome`) | new clause at L79 or L108 | OPTIONAL |
| 8 | Tenneal 2015 (FanGraphs Community) | L139 | REJECT |
| 9 | Melville, Mott, Grimsman & Archibald 2025 (SSAC; was `anon-2025-ai-manager-strategies`) | none | REJECT |
| 10 | Dolinar & Pemstein 2015; Pemstein 2016 | L158 / L74 | REJECT |
| 11 | Klaassen 2010, Carleton 2017, Melville 2024, Clemens x2, Baumann 2024, BP 2018, Freiman 2018, Weise 2017 | none | REJECT |

---

## ADD

### 1. Melville et al. 2023, "A Game Theoretical Approach to Optimal Pitch Sequencing"

**Tex, L79 (last sentence):** "Each hitter keeps a learned embedding, but the network predicts the hitter's response to each pitch from the pitch's measured characteristics, rather than the at-bat result from the pitcher's identity."

**Source, §4, p. 11 (Sloan PDF):** "x_t is a vector containing the relevant information about pitch t, specifically the release speed, spin rate, movement, plate location, the ball and strike count, and the batter and pitcher handedness. ... B is an index representing the batter". Also: "we first pass the batter index through an embedding layer as in [6]" ([6] is Alcorn). **p. 3:** "A key element of our pitch outcome model is the batter embedding. ... the idea for this came from Alcorn's work on the (batter|pitcher)2vec model."

**Why:** this is the closest prior model. Like this paper, it keeps a learned batter embedding (built on Alcorn), takes the measured pitch features, and predicts the pitch outcome. Leaving it out makes L79 read as if this paper is the first to follow Alcorn's suggestion at the pitch level. Novelty at L81 still holds, because Melville et al. use the model for pitch sequencing and have no matchup or handedness projection query. Suggested wording: "...as did \citet{melville2023} for pitch sequencing."

```bibtex
@inproceedings{melville2023,
  author    = {Melville, William and Melville, Jesse and Dawson, Theodore and Nieves-Rivera, Delma and Archibald, Christopher and Grimsman, David},
  title     = {A Game Theoretical Approach to Optimal Pitch Sequencing},
  booktitle = {MIT Sloan Sports Analytics Conference},
  year      = {2023},
  url       = {https://www.sloansportsconference.com/research-papers/a-game-theoretical-approach-to-optimal-pitch-sequencing}
}
```
**Assumption:** the year 2023 comes from the PDF's creation date (13 Feb 2023) and the matching 2023 BYU thesis. The Sloan page itself does not print a year; its site navigation only shows SSAC26. The authors and title match the PDF title page.

**Verdict: ADD.** It is directly comparable prior work, and the paper's own framing at L79 leans on the gap it fills.

### 2. Powers & Yurko 2025, "Swinging, Fast and Slow"

**Tex, L139:** "The count moves as a Markov chain over 12 states from 0-0 to 3-2, and the expected wOBA from count $(b,s)$ satisfies"

**Source, §3.3, p. 10:** "We model the progression of a plate appearance as a Markov reward process whose nonterminal states are all possible ball-strike counts S_N ≡ {0,1,2,3} × {0,1,2} and whose terminal states are S_T ≡ {Strikeout, Walk, Hit by Pitch, Fair Ball}." Transition probabilities come from a pitch outcome model, and the "value of the initial state s = (0, 0) gives us the estimated run value."

**Why:** this is the same 12-state count chain, driven by per-pitch outcome probabilities and read off at 0-0. It is a peer-reviewable preprint by established authors. They solve it iteratively (Bellman) and score in linear-weight runs, whereas this paper solves it directly and scores in wOBA. Those are differences in detail; the construction is the same.

```bibtex
@misc{powers2025,
  author        = {Powers, Scott and Yurko, Ronald},
  title         = {Swinging, Fast and Slow: Interpreting Variation in Baseball Swing Tracking Metrics},
  year          = {2025},
  eprint        = {2507.01238},
  archivePrefix = {arXiv},
  primaryClass  = {stat.AP},
  url           = {https://arxiv.org/abs/2507.01238}
}
```
**Verdict: ADD.** The paper presents the count chain without precedent, and this is a citable, recent and exact precedent.

### 3. Deshpande & Wyner 2017, "A Hierarchical Bayesian Model of Pitch Framing"

**Tex, L112 (second sentence):** "Their outcomes therefore come from a table fit on the training seasons, which gives the probability of a ball, a called strike, or a hit batter for each plate location and pairing of batter and pitcher hand."

**Source, §2.2 "Adjusting for Pitch Location", p. 8 (arXiv v2):** "The differences in spatial distribution of pitches seen in Figure 3 motivate us to use a separate smoothed estimate of the historical log-odds of a called strike for each combination of batter and pitcher handedness."

**Why:** this is the precedent for modeling called-strike probability by location, separately for each handedness pairing. It supports the design choice in the sentence. It does **not** support the first sentence of L112 ("depend mostly on the pitcher and the umpire"), so attach it to the second sentence only.

```bibtex
@article{deshpande2017,
  author  = {Deshpande, Sameer K. and Wyner, Abraham J.},
  title   = {A Hierarchical {B}ayesian Model of Pitch Framing},
  journal = {Journal of Quantitative Analysis in Sports},
  volume  = {13},
  number  = {3},
  pages   = {95--112},
  year    = {2017},
  doi     = {10.1515/jqas-2017-0027},
  eprint  = {1704.00823},
  archivePrefix = {arXiv}
}
```
**Assumption:** I confirmed the title, authors and arXiv id from arXiv v2. The volume, number, pages and DOI come from the library note and were not re-fetched from the publisher. Check the DOI resolves before submitting. The arXiv v1 title was "SAFE2: A Hierarchical Model of Pitch Framing", so cite the journal title.

**Verdict: ADD.** The handedness-by-location table is an established modeling choice, and this is the standard reference for it.

### 4. MLB.com glossary, "Expected Weighted On-base Average (xwOBA)"

**Tex, L137:** "A ball in play is valued at the league's average wOBA for its exit velocity, launch angle, and spray bins, weighted by the hitter's predicted probability of each combination, so its value sets aside the fielder and park effects described in Section~\ref{sec:intro}."

**Source (glossary, "Definition" and "Why it's useful"):** "every batted ball is given a single, double, triple and home run probability based on the results of comparable batted balls since Statcast was implemented ... For the majority of batted balls, this is achieved using only exit velocity and launch angle." And: "xwOBA removes defense from the equation. Hitters ... are able to influence exit velocity and launch angle but have no control over what happens to a batted ball once it is put into play."

**Why:** L137 is an xwOBA-style valuation, and the paper never names it. Readers will recognize it as xwOBA, so crediting it also heads off a reviewer's "this is just xwOBA" remark. Differences to note: this paper adds spray, uses bins, and does not use sprint speed. The second quote also gives L74 ("Fielders, parks ... without reflecting anything the hitter did") a source, though that one is a league glossary rather than research.

```bibtex
@misc{mlbxwoba,
  author       = {{MLB Advanced Media}},
  title        = {Expected Weighted On-base Average (xwOBA)},
  howpublished = {MLB.com Glossary},
  url          = {https://www.mlb.com/glossary/statcast/expected-woba},
  note         = {Accessed 17 November 2026}
}
```
**Verdict: ADD** at L137. Using it at L74 as well is optional.

---

## OPTIONAL

### 5. Yonushonis 2011, "The Infinitely Long MLB Plate Appearance"

**Tex, L144:** "A two-strike foul leaves the count unchanged, and that loop is solved directly."

**Source (sabr.org, section "Infinite Geometric Series"):** "An infinite geometric series is an infinite series whose successive terms have a common ratio. Such a series converges if (and only if) the absolute value of the common ratio is less than one (|r| < 1)." The article groups two-strike foul sequences into geometric families and sums them in closed form.

**Why optional:** this is the standard closed form for a self-loop. Most readers will not need a citation for it, and Powers & Yurko already covers the chain. It is useful only if a reviewer asks why there is no truncation.

```bibtex
@article{yonushonis2011,
  author  = {Yonushonis, Brian},
  title   = {The Infinitely Long {MLB} Plate Appearance},
  journal = {Baseball Research Journal},
  year    = {2011},
  note    = {Spring 2011},
  publisher = {Society for American Baseball Research},
  url     = {https://sabr.org/journal/article/the-infinitely-long-mlb-plate-appearance/}
}
```
The volume number was not verified, so it is left out.

**Verdict: OPTIONAL.** It supports the sentence, but the fact is elementary.

### 6. Heaton & Mitra 2022 and 2023 (SSAC)

**Tex:** no current sentence. It would need a new clause in the L79 paragraph, for example "Later work learned context-dependent player representations from event sequences \citep{heaton2022,heaton2023}."

**Source:** 2022, §2.1.1, p. 3, on (batter|pitcher)2vec: "While useful, we question the assumption that the same player can or should be described the same way in all situations." The 2023 paper extends this to contextual event embeddings pre-trained on MLB event sequences.

**Why optional:** this is the main line of player-embedding work after Alcorn, and a Sloan audience will know it. But the paper makes no claim it supports. Adding it means adding a sentence, which is the author's call.

```bibtex
@inproceedings{heaton2022,
  author    = {Heaton, Connor and Mitra, Prasenjit},
  title     = {Using Machine Learning to Describe How Players Impact the Game in the {MLB}},
  booktitle = {MIT Sloan Sports Analytics Conference},
  year      = {2022}
}
@inproceedings{heaton2023,
  author    = {Heaton, Connor and Mitra, Prasenjit},
  title     = {Learning Contextual Event Embeddings to Predict Player Performance in the {MLB}},
  booktitle = {MIT Sloan Sports Analytics Conference},
  year      = {2023}
}
```
**Verdict: OPTIONAL.** It is related work, not support for an existing claim.

### 7. Kneita 2025 (published version of `anon-2025-transformer-pitch-outcome`)

**Published version:** yes. The Sloan page titled "Transformer-Based Baseball Modeling for Pitch Outcome Prediction and Strategy Optimization" lists **Declan Kneita** as the only author. The PDF linked from that page is the same anonymized file as the library copy (header "Baseball 20251439", created Jan 2025). Cite the Sloan page, not the anonymous copy.

**Tex:** the closest is L108, "The pitcher enters only through the pitches thrown, not through an identifier, which lets the model describe any pitcher whose pitches it has seen." It would fit better as a related-work clause at L79.

**Source (§3 data, p. ~5):** "Meta-information such as player names and game identifiers were used solely for constructing sequences and were not included in the model's input features." Results section: the model "adapts to individual player tendencies without explicit player identifiers in the input data." Introduction: static player embeddings "may not generalize well to unseen players."

**Why optional:** it is a pitch-level outcome model that drops player identifiers entirely. That is a useful contrast, since this paper keeps the hitter embedding but drops the pitcher's identity. However, L108 states a design choice, not a borrowed claim.

```bibtex
@inproceedings{kneita2025,
  author    = {Kneita, Declan},
  title     = {Transformer-Based Baseball Modeling for Pitch Outcome Prediction and Strategy Optimization},
  booktitle = {MIT Sloan Sports Analytics Conference},
  year      = {2025},
  url       = {https://www.sloansportsconference.com/research-papers/transformer-based-baseball-modeling-for-pitch-outcome-prediction-and-strategy-optimization}
}
```
**Assumption:** the author name is taken only from the Sloan page, because the PDF is anonymized and one web search found nothing else. The year 2025 is inferred from the submission id (2025xxxx) and the Jan 2025 PDF date; the page does not print a year.

**Verdict: OPTIONAL.** It is a relevant contrast but does not support an existing sentence, and the author attribution comes from a single source.

---

## REJECT

- **8. Tenneal 2015**, FanGraphs Community, "Using Markov Chains to Predict K% and BB%". It uses the same 12 count states with absorbing K/BB/BIP. However, it is pseudonymous ("Captain Tenneal") and unreviewed, and Powers & Yurko covers L139 better.
- **9. Melville, Mott, Grimsman & Archibald**, "An Extensive Investigation of Strategies in Baseball". This is the published version of `anon-2025-ai-manager-strategies`; the full paper is posted on the Sloan page, and the submission id 20251404 implies SSAC 2025. The earlier note left out Mott. It is rejected because no sentence relies on it: its handedness projections use Marcel, and the paper does not discuss manager decision AI.
- **10. Dolinar & Pemstein 2015; Pemstein 2016.** They define reliability as true-talent variance over observed variance, which is the same idea as L158. franks2016 is already cited there, so a second source adds nothing. I did not confirm that Pemstein 2016 shows exit velocity is more reliable than wOBA, so it cannot be used for L74.
- **11. The rest:**
  - **Klaassen 2010:** corroborates the 2,200/1,000 constants, but tango2007 is already cited.
  - **Carleton 2017:** compares split methods, but no sentence matches.
  - **Melville 2024:** factorizes spray, then launch, then speed, which is a different order from L110 and does not support it.
  - **Clemens x2, Baumann 2024, BP 2018, Freiman 2018, Weise 2017:** no matching sentence.

---

## Uncited claims with no verified source

These sentences lean on outside work. I could not find and read a source that supports them, so none is proposed.

| Line | Sentence (short) | What it leans on | Suggested action |
|------|------------------|------------------|------------------|
| L74 | "Fielders, parks, and where a ball in play happens to land all change a plate appearance's wOBA value without reflecting anything the hitter did." | DIPS/BABIP lineage (McCracken 2001), park factors | Partly covered by the xwOBA glossary (#4). Otherwise soften or cite DIPS after reading it. |
| L74 | "...they carry less of the luck that follows contact." | That exit velocity and launch angle stabilize faster than wOBA | No source read. Either cite a stabilization study after reading it, or point to the paper's own results. |
| L112 | "Taken pitches depend mostly on the pitcher and the umpire." | Framing and umpire literature | Deshpande & Wyner model umpire, catcher and pitcher effects, but do not say "mostly". Reword as a modeling assumption or drop "mostly". |
| L160 | "2,000 bootstrap draws that resample batters." | Cluster or block bootstrap | Standard method, optional. No source read. |
| L245 fn | "after rotating each onto the first" | Orthogonal Procrustes (Schönemann 1966) | Not read. Name the method, and cite it only after reading. |
| L285 | "the silhouette score is at most 0.054" | Rousseeuw 1987 | Not read. A reviewer may expect it, so read it before adding. |
| L292 | "Pederson and Schwarber, left-handed hitters with reputations as platoon bats" | Reputational claim | No source. Either point to their career split usage (the paper's own data) or reword as "who are often platooned". |

**Note on an existing entry, without editing it:** tango2007 is cited at L77 for p. 157 (2,200 / 1,000). The library note shows these constants were taken from Tango's insidethebook.com comment quoting the book, not from the book page itself. Confirm p. 157 in a physical or ebook copy before submission.

---

## Recommendation

- **Add #1 to #4.** Each one fixes a gap a reviewer is likely to name:
  - #1: "Melville et al. already did a pitch-level batter embedding."
  - #2: "The count chain is not new."
  - #3: "Where does the handedness-by-location take table come from?"
  - #4: "This is xwOBA."
- All four were read at the quoted locations. Together they add four bib entries and change no claims.
- **#1 matters most**, because it is the only one that touches the paper's novelty framing at L79.
- **Of the OPTIONAL items,** add Heaton & Mitra only if the author wants a one-sentence related-work clause. Leave Yonushonis and Kneita out unless a reviewer asks.
- **Before submission,** fix the L112 "mostly" and L292 "reputations" sentences, since they make claims nothing in the bib supports.
