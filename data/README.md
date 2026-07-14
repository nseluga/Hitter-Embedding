# Data

Raw and processed data, Statcast snapshots, feature tables, and splits.

## Structure (TBD)

- `raw/` — Statcast snapshots (versioned by pull date), raw pitch-event exports
- `processed/` — Feature-engineered pitch tables, labels, walk-forward splits
- `external/` — Supplementary data (player names/IDs, pitcher rosters, run-value mappings)

## Important Notes

### Statcast Snapshots
- Statcast revises retroactively; snapshots must be frozen by pull date + script
- Store both snapshot parquet and the pull script that generated it
- Document: Python version, pybaseball version, date pulled, seasons included

### Frozen Walk-Forward Splits
- Season splits are **frozen in config** before any model comparison (see `src/config/`)
- Once frozen, never revisit or use random splits
- Splits are: train ≤ t, validate t+1, test t+2 (strict walk-forward)
- Commit split config to repo; treat it as immutable

### Leakage Safeguards
- No future-dated features (all inputs strictly precede predicted pitch)
- No outcome-luck data (batted-ball quality beyond EV/LA/direction/spin)
- No market prices, salary data, or team-controlled player outcome labels in Layer 1
- All features unit-tested for leakage before use

### Data Availability
- EV, LA: public, 2015+
- Launch direction: derivable from hit coordinates
- Batted-ball spin: verify availability before design freeze (historically not public)
- Bat-tracking (swing length, speed, attack angle): 2023+ only
