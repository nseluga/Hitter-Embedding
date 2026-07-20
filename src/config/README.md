# Configuration

Frozen decisions and immutable settings for reproducible, non-drifting model runs.

## Frozen Items (Never Change Without Flagging)

These live in YAML/JSON config files and must be committed to the repo:

### Walk-Forward Season Splits
```
train_seasons: [2015, ..., t]
validation_season: t+1
test_season: t+2
```
Contiguous walk-forward per architecture doc §2.2: train on seasons <= t,
validate on t+1, test on t+2. Frozen instance: train 2015-2023, val 2024,
test 2025 (see split_config.json).

**Rule:** Frozen once defined. Random splits are forbidden. If you need to change this, stop and ask Nate.

### Outcome Space (Phase B Decision)
Candidate dimensions for contact-quality head:
- Exit velocity (EV)
- Launch angle (LA)
- Hit launch direction (spray angle)
- Batted-ball spin angle
- Batted-ball spin rate

**Status:** TBD after Phase B ablations. Once decided, locked in.

### Pitcher Query Configuration
- **Hierarchical (primary):** Sample pitchers by batters-faced weighting; simulate PAs; average run value
- **Pooled (comparison):** Pool all pitches; sample i.i.d.; report rank stability as robustness check
- Both run in parallel; divergence reported

### Leakage Guardrails (Tested, Not Configured)

All data pipeline unit tests verify:
- No features with future-dated information
- No outcome-luck labels (only process outcomes)
- No market prices or salary data
- No team-controlled player labels in Layer 1
- Encoder inputs strictly precede the predicted pitch (if applicable)

## Configuration Files (TBD)

Create as build progresses:

- `split_config.yaml` — Walk-forward season definitions (immutable)
- `feature_config.yaml` — Phase B decision on outcome dimensions
- `model_config.yaml` — Architecture spec (hitter embedding dim, context MLP size, ensemble size)
- `training_config.yaml` — Loss function, optimizer, batch size, learning rate schedule
- `query_config.yaml` — Monte Carlo sampler config (pitcher population, num sims, seed)

## Reproducibility

Every run is seeded and reproducible from:
1. Data snapshot (Statcast pull date + script)
2. Exact config file
3. Random seed
4. Code commit hash

Store this metadata with every result.
