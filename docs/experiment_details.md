# Experimental Design

## Research question

Does a game-theoretic inference rule add coordination benefits beyond a learned joint-compatibility baseline when the predictive model, state representation, candidate set, test pieces, and continuation length are held fixed?

## Data representation

The project uses the voice-separated 16th-note JSB Chorales representation and keeps only soprano and bass. Consecutive repeated notes are converted to a `HOLD=-2` token while rests remain `REST=-1`.

The final model uses a 16-step context window plus six duration-aware features computed only from that visible context:

1. upper current hold length
2. lower current hold length
3. upper steps since last onset
4. lower steps since last onset
5. whether the previous upper token is HOLD
6. whether the previous lower token is HOLD

## Shared utility model

A single multi-head MLP predicts:

- upper/soprano next-token logits
- lower/bass next-token logits
- joint pair-compatibility logits

All final inference methods use the same trained duration-aware checkpoint. The main controlled comparison changes only the inference-time rule used to select a pair from the same candidate grid.

## Inference rules

- **No-game joint:** selects from learned pair compatibility without an explicit strategic solver.
- **QRE:** iterative soft-response game with a compatibility filter.
- **Stackelberg:** leader/follower inference with either the lower or upper voice as leader.
- **Pure Nash:** equilibrium-based strategic ablation with deterministic tie-breaking/fallback.

Validation-selected settings are stored in `configs/final_inference.yaml`.

## Final objective evaluation

The authoritative objective evaluation uses all 77 held-out test chorales. Each method generates one matched 128-step continuation per test chorale; the 16-step real seed prefix is excluded from evaluation.

Metrics include melodic-interval, harmonic interval-class, pitch-class, voice-distance, duration/hold, entropy, and local 3-gram diversity statistics. For the corpus-similarity distances and gaps reported in the main comparison, lower values indicate closer agreement with the real test corpus.

## Uncertainty

Cross-chorale uncertainty is estimated using 10,000 paired percentile-bootstrap resamples. The same resampled chorale indices are used for every method within a bootstrap replicate. Reported deltas use:

```text
method - no_game_joint
```

Because the main distance/gap metrics are lower-is-better, a negative delta favors the compared method. A confidence interval that crosses zero is treated as insufficient evidence of a directional difference; it is not interpreted as proof of equivalence.

See `results/paired_bootstrap_vs_no_game.csv` for the full table.
