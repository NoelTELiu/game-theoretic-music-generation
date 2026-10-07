# Experimental Design

## Research question

Does a game-theoretic decision rule improve two-voice generation beyond a no-game learned joint-compatibility baseline under the same model, state representation, candidate set, seeds, and continuation length?

## Data representation

The project uses the voice-separated 16th-note JSB Chorales representation and keeps soprano and bass. Consecutive repeated non-rest pitches are converted to `HOLD=-2`; rests remain `REST=-1`.

The final duration-aware state contains a 16-step two-voice context, bar position, and six explicit duration features:

1. upper current hold length
2. lower current hold length
3. upper steps since last onset
4. lower steps since last onset
5. whether the previous upper token is HOLD
6. whether the previous lower token is HOLD

The first four duration features are normalized by 16.

## Shared utility model

A single multi-head MLP predicts:

- an upper/soprano private score,
- a lower/bass private score, and
- a shared upper-lower pair-compatibility score.

All final methods use the same trained duration-aware checkpoint. At each timestep, the upper and lower private heads define top-12 candidate sets, producing a shared 12 x 12 candidate grid for No-game joint, QRE-style, Stackelberg, and Nash inference.

## Inference rules

- **No-game joint:** samples from learned pair compatibility without constructing player-specific utilities or solving a game.
- **Compatibility-filtered QRE-style:** uses iterative logit soft responses over player-specific utility matrices and renormalizes the joint policy over the highest-compatibility pairs. It is an iterative soft-response approximation rather than an exact analytical QRE solver.
- **Stackelberg:** models asymmetric leader-follower interaction. The reported main configuration is lower-leading soft Stackelberg; an upper-leading variant is retained as a leader ablation.
- **Pure Nash:** selects a locally stable pure equilibrium when one exists, with welfare tie-breaking and a welfare-maximizing fallback. It is treated as a strict-equilibrium ablation rather than a competitive generator.

Validation-selected settings are stored in `configs/final_inference.yaml`.

## Validation selection caveat

The strategic methods were selected from different validation search spaces:

- QRE-style: 9 settings
- Stackelberg: 6 settings
- Nash: 3 settings
- No-game joint: unswept

All selected settings are frozen before the final held-out test. However, this tuning asymmetry limits attribution of every performance difference solely to the decision-rule family.

## Final objective evaluation

The authoritative objective evaluation uses all **77 held-out test chorales**. Each method receives the same 16-step real prefix and generates one matched 128-step continuation per test chorale. The seed prefix is removed before scoring.

Metrics include melodic-interval, harmonic interval-class, pitch-class, voice-distance, duration/hold, entropy, and local 3-gram diversity statistics. For the corpus-similarity distances and scalar gaps reported in the main comparison, lower values indicate closer agreement with the real test corpus.

## Paired uncertainty analysis

Cross-chorale uncertainty is estimated using **10,000 paired percentile-bootstrap resamples**. The same resampled chorale indices are applied to every method within a bootstrap replicate, while the real test corpus remains fixed.

Reported deltas use:

```text
method - no_game_joint
```

Because the main distance/gap metrics are lower-is-better, a negative delta favors the compared method. A 95% confidence interval crossing zero is treated as an uncertain directional difference and is not interpreted as proof of equivalence.

The most robust positive strategic effect is the reduction in unique 3-gram ratio gap for both Stackelberg leader variants. Pure Nash instead shows a persistent collapse toward excessive sustain and very low local diversity.

See `results/paired_bootstrap_vs_no_game.csv` for the compact paired-bootstrap table and `docs/papers/extended_technical_report.pdf` for the full analysis.
