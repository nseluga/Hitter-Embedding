# Decision Log — Hitter Embedding

Append-only. Format fixed by `~/os/knowledge/frameworks/research-standards.md`
§4 — `/research-partner` and `/research-review` both parse against it.

---

## <YYYY-MM-DD> — <decision title>
- **Decision:** what was chosen
- **Alternatives:** what was considered and rejected
- **Rationale:** why, in terms a skeptical reviewer would accept
- **Revisit if:** the condition under which this should be reopened

---

## 2026-07-14 — Statcast raw-snapshot storage design
- **Decision:** Raw Statcast frozen to `data/raw/statcast/snapshot_<date>/season=<YYYY>.parquet` — per-season, all columns, immutable; committed `manifest.json` holds pull metadata. Data gitignored; script + manifest committed.
- **Alternatives:** Monolithic parquet (forces full-memory loads); column pruning at pull time (unrecoverable without re-pull).
- **Rationale:** Date-stamped snapshots handle Statcast's retroactive revisions; immutable raw lets the pitch-event table re-derive without re-fetching.
- **Revisit if:** snapshot outgrows local disk.

---

## 2026-07-15 — Statcast cleaning spec for the modeling pitch table
- **Decision:** Modeling table is regular season only. It removes position-player pitches, pitchouts, automatic balls and strikes, and bunts, and keeps intentional balls. It drops rows missing or physically impossible on core context of type, velocity, location, and movement, and keeps optional spin context with missingness indicators. It drops the 8 deprecated columns, dedupes on pitch key, and sorts by game_date then pitch key. Filters apply only to the modeling table, never to evaluation targets.
- **Alternatives:** A minimum-PA hitter floor was rejected because it deletes the low-exposure population the thesis targets. A velocity-based position-player rule was rejected because it misclassifies hard-armed position players. Dropping spin columns was deferred to a Phase B ablation.
- **Rationale:** Every filter is backed by the profiling notebook, and the modeling-versus-target split keeps sharpening filters from biasing ground truth.
- **Revisit if:** Phase B feature screening changes the retained context set, or target construction needs a filtered field.

---

## 2026-07-17 — Contact-quality label domain
- **Decision:** The contact-quality head (EV, LA, spray) is labeled on balls in play only (`hit_into_play`); `ev`/`la`/`spray` are null on all other pitches.
- **Alternatives:** Labeling all contact-with-EV was rejected because fouls carry EV/LA ~70% of the time but have no spray and no batted-ball outcome, giving ragged masking and no run-value mapping.
- **Rationale:** Fouls are contact for the whiff head but are non-terminal count transitions in the Markov composition; only in-play balls have the run-value-bearing outcome the quality head feeds.
- **Revisit if:** the outcome space or run-value mapping (§1.5) changes to need foul-ball measurements.

---

## 2026-07-17 — Spray angle derivation
- **Decision:** Spray = `atan((hc_x − 125.42)/(198.27 − hc_y))` in degrees, then mirrored so positive = pull for both hands (flip sign for RHB).
- **Alternatives:** Empirically calibrating the home-plate origin was dropped once the constants were sourced; raw field-side angle (no mirroring) was rejected as not batter-intrinsic.
- **Rationale:** Formula and constants corroborated by three sources (abdwr3e App. C, BGSU, Weise); MLB does not publish the origin, so a real-data regression-guard (field mean ≈ 0, pull-mean > 0) confirms our export matches the scale.
- **Revisit if:** the near-plate artifact (`|spray| > 90°`, ~1% of in-play) needs clipping, decided in Phase B.

---

## 2026-07-17 — Walk-forward split frozen
- **Decision:** Contiguous walk-forward, train 2015-2023 / val 2024 / test 2025, single fold, frozen in `src/config/split_config.json` and validated on load.
- **Alternatives:** Random k-fold rejected (leaks a hitter's future PAs into his own ID embedding, memorizing the outcome we claim to project and manufacturing a positive result). Rolling multi-fold deferred (multiplies compute against the <$200 budget with no gain on the exposure axis we grade on). A val/test gap season rejected (wastes data, breaks Phase B's same-season window trade).
- **Rationale:** Projection is a forecasting task, so eval must mirror deployment: train on the past, test on a strictly-later season. Freezing before any model comparison keeps the held-out season from becoming a shopped hyperparameter (pre-registration). Contiguity minimizes distribution shift so the metric measures projection skill, not regime drift. Robustness comes from exposure stratification + dual-sampler, not temporal folds.
- **Revisit if:** never for this project (frozen rule); a new fold requires a new entry naming this one.
