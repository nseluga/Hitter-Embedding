# Draft 2 flow reviews (2026-10-31)

Three independent opus reviewers read `draft2/main.tex`, `paper-outline.md`, and Alcorn 2018. They covered flow and structure only, since facts were already checked section by section. Nothing below is applied yet.

- **FO** = front-office R&D analyst
- **ML** = ML / evaluation-methodology researcher
- **ST** = skeptical sabermetric statistician

Priority codes: M = must, S = should, C = could, blank = not raised.

## Where reviewers agree (2+ of 3)

| # | Change | FO | ML | ST | Argument | My take |
|---|---|---|---|---|---|---|
| 1 | Move the 539-hitter 2025 split grade (296 vs 243, MAE 0.0448 vs 0.0462) from 4.3 into Results, labeled exploratory | S | S | M | It is the only out-of-sample result on the split itself, and it first appears in a section framed as anecdote | **Yes.** ST also wants a batter-clustered bootstrap CI. I can compute it (small). If the CI spans zero, say so |
| 2 | Aim the case studies at the medium band. Say up front that all 5 anchors are high-exposure, where no method adds anything. Add medium-band cases by a stated rule | M | S | S | 4.1 says the edge is in 113–451 PA, yet every named case is outside it | **Yes.** Rule: lead 4.3 with the six medium-band top-20 disagreement hitters (already graded, 3–3). This adds no new pull and leaves no room to cherry-pick |
| 3 | Add a 2.4 Evaluation subsection: targets, strata as the operational meaning of "less data", baselines, metrics, registered gate, bootstrap, the two-pass 2025 story told once | S | S | | The protocol is scattered across 4 places. The two-pass story is told 4 times (l.102, 176, 201, 316), and EB is described twice | **Yes.** This departs from Alcorn (baselines in his Results), but he had one test and we have a gate, strata and two passes |
| 4 | State the payoff/thesis in the Intro contributions, flagged as interpretation, after the failure sentence | M | | S | The reader first learns why to care around l.272, about 4,500 words in. The roadmap sells the five hitters instead | **Yes.** Use ST's wording ("a reading built mostly on descriptive results") so it doesn't soften the failure |
| 5 | Fold 3.4 (platoon direction) into one paragraph at the end of 3.3. Gradient method and Fig 4 to an appendix, or keep the fig if pages allow | C | C | C | The abstract doesn't promise it, the effect is small, and it's method detail sitting in Results | **Yes**, keep the figure for now. All three agree but rate it low priority |
| 6 | Move the 87%-coverage sentence out of 3.2 (ceiling) | C | S | C | Calibration is not a ceiling result, and it interrupts 3.2's argument | **Yes**, to 4.1 as sizing evidence (ST's placement), labeled 2024 |
| 7 | Replace Table 2's "Closer" column with (obs − midpoint)/SE, or drop it | C | | S | It invites a 0-for-5 / 2-to-6 scorecard that the caption then disowns | **Yes**, use the z column |
| 8 | Two-panel Figure 1: network + query pipeline (pitches → per-count probs → count chain → W_hj → BF-weighted mean) | S | S | | The query is the abstract's distinctive idea and the hardest prose to follow without a picture | **Yes.** It's still a TODO. I can draft a TikZ version, or you draw it |
| 9 | Move the neighbour lists (Pederson, Trout, 9/40 overlap) from 4.2 into 3.3, as Alcorn does | S | (C) | S | Neighbours describe the embedding, and the forward pointer at l.241 shows they're out of place | **Lean no.** You asked for players to be examined in Discussion. Compromise: put the 9/40 aggregate in 3.3 and keep the named hitters in 4.2. **Your call** |

## Raised by one reviewer, but strong

| # | Change | Who | Argument | My take |
|---|---|---|---|---|
| 10 | Label each claim, not each section. Only the low-stratum gate was registered (verified: decision log 2026-07-30 says "in the claimed stratum"). The pooled rank win is not confirmatory. Mark Table 1 rows Low = registered and Medium = exploratory, and caveat the Fig 2 caption | ST (M) | l.163 calls all of 3.1 "confirmatory", so the pooled win reads as confirmed | **Yes, must.** The l.163 wording is actually inaccurate, not just a framing issue |
| 11 | Rebuild 4.1. Rank the three legs by evidence (2025 pooled rank strongest, 2024 ceiling descriptive on the selection season, direction weak). Note the sizing tie is a noise floor every method hits. Put "exploratory" before "where that call is open" | ST (M), ML (C: drop the direction leg) | It reads as three independent confirmations, and two of them share the selection season | **Yes.** Swap the direction leg for the 2025 split grade (#1), per ML |
| 12 | Make the abstract's "Figure~1" true by moving the power/contact figure into the Intro as a teaser | ML (M) | Verified: the abstract cites Figure 1 = power/contact, but the first float is the architecture TODO | **Yes.** Alcorn also has an intro figure. The alternative is renumbering the abstract, which you said not to touch |
| 13 | Reconcile the abstract's "5 heads" with the body's six | ML (M) | A reviewer sees this on page 2 | **Your call.** (a) change the abstract to 6 (a number fix like the others), or (b) the body says "five parts of a plate appearance, with contact read by two heads". I'd pick (a) |
| 14 | Low-stratum paragraph: open with "not distinguishable from either baseline", not the nominal wins | ST (S) | It matches the intro's "beat neither". The nominal wins first reads as spin | **Yes** to reordering. See disagreement B on the power number |
| 15 | Fix 3.2's opener ("0.47 sounds low until…") | ML (S) | 0.47 is 2025 side wOBA and the ceiling is 2024 pooled level 0.475, so the near-twins invite conflation | **Yes.** Use a bridge instead: "the decision turns on the gap, so how measurable is it?" |
| 16 | Tie Limits to next steps, and add interval calibration as a next step | FO (S) | The matchup card promises "an honest band" but coverage is 87% at a nominal 95%, and this isn't in the next steps | **Yes**, add calibration to §5. Keep 4.4 |
| 17 | Retitle 3.1 / Table 1 away from "sealed" | ST (C) | l.102 admits 2025 isn't as sealed as one pass would make it | **Yes**, the heading becomes "Handedness splits on 2025". The abstract stays as is |
| 18 | Expand the Pederson per-head breakdown into a short paragraph with the V.9 head figure | FO (S, part of 5) | The swing-vs-damage story is what coaches act on | **Could.** It costs a figure. Only if pages allow |
| 19 | Move the Tango/EB intro paragraph up to follow paragraph 1 | ML (C) | It puts 226 PA and 2,200 PA together, and lets the Alcorn paragraph hand straight to the contributions | **Neutral.** Fine either way |

## Where reviewers disagree

| | Issue | Positions | My take |
|---|---|---|---|
| A | Where the neighbour lists go | FO + ST: into 3.3. ML: only a bridge at the top of 3.3. Your outline: players in Discussion | Split it: the aggregate in 3.3, the named hitters in 4.2 (#9) |
| B | The 5,200-batter power number | ST: post-hoc power just restates the p-value and reads as an excuse, so move it to Limits or recast it as a 2026 design number. FO + ML: keep it next to the failure (both list it under "keep") | **Recast in place** as the 2026 design number ("a test on 2026 would need…"). That keeps the context and drops the excuse reading |
| C | The 539-hitter grade | ST: move it with a CI, or cut it. FO + ML: move it, labeled exploratory | Move it **with** the CI. It's cheap and it closes ST's objection |
| D | Coverage destination | FO + ML: end of 3.1. ST: 4.1 as sizing evidence | 4.1. It's a 2024 number, so it doesn't belong in the 2025 test section |
| E | How far to follow Alcorn | FO: baselines in Methods departs from Alcorn. ML: the protocol has outgrown his layout. ST: the draft is already better ordered than Alcorn | Depart where the paper has more protocol than he did (#3). Otherwise keep his arc |

## Kept by all three

- The Results preamble labeling confirmatory vs descriptive. #10 makes it accurate.
- The failed test stated everywhere the reader looks: abstract, l.84, l.197, Limits, Conclusion.
- Anchor expectations written before results, with the Duvall→Bohm swap and the 1.96-SE caveat disclosed.

## Verdicts

- **FO:** The skeleton is sound and faithful to Alcorn. The payoff comes too late, the case studies sit in the stratum with no edge, and repeated caveats crowd out the decision story.
- **ML:** The order and honesty are right. Fix the abstract–body mismatches, consolidate evaluation into Methods, and re-aim the case studies at the medium band.
- **ST:** Honest, and better ordered than Alcorn. The Discussion still lets descriptive and post-hoc results stand as a confirmed thesis, so #10, #11 and #1 are needed before submission.
