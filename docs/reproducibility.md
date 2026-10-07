# Reproducibility Notes

## What is included

- preprocessing code
- training code
- all game-theoretic inference implementations
- generation and evaluation scripts
- paired-bootstrap code
- listening-stimulus construction scripts
- validation-selected final inference settings
- compact final objective, bootstrap, and listening-study result tables

## What is intentionally not included

- raw JSB Chorales files
- processed dataset artifacts
- model checkpoints
- thousands of intermediate/timestamped experiment outputs
- rendered listening-study media
- intermediate development notes

These exclusions keep the public repository small and avoid redistributing third-party data or machine-specific experiment artifacts.

## Randomness

Training defaults to seed 42 in `configs/train_ctx16_duration_aware.yaml`. Generation scripts expose explicit RNG seeds, and the final paired-bootstrap analysis used 10,000 replicates with seed `20260816`.

## Checkpoints

The final study used a duration-aware context-16 checkpoint. Checkpoints are ignored by Git and should be stored locally under `checkpoints/` or attached separately as a GitHub Release if public inference without retraining is desired.
