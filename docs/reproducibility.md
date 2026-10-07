# Reproducibility Notes

## What is included

- preprocessing code
- training code
- QRE-style, Stackelberg, and Nash inference implementations
- generation and evaluation scripts
- paired-bootstrap code
- listening-stimulus construction scripts
- validation-selected final inference settings
- compact final objective, bootstrap, and listening-study result tables
- accepted ISMIR 2026 LBD camera-ready paper and extended technical report

## What is intentionally not included

- raw JSB Chorales files
- processed dataset artifacts
- model checkpoints
- thousands of intermediate/timestamped experiment outputs
- rendered listening-study media
- intermediate development notes

These exclusions keep the public repository small and avoid redistributing third-party data or machine-specific experiment artifacts.

## Randomness

Training defaults to seed 42 in `configs/train_ctx16_duration_aware.yaml`. Generation scripts expose explicit RNG seeds. The final paired-bootstrap analysis used **10,000 resamples** with seed `20260816`.

## Final-test protocol

Solver hyperparameters are selected on validation data and frozen before final testing. The final objective study evaluates all 77 held-out test chorales with the same real 16-step seed for every method and a matched 128-step generated continuation. The seed prefix is excluded from scoring.

The bootstrap quantifies cross-chorale uncertainty, not model-training uncertainty or repeated stochastic generations from multiple seeds within a chorale.

## Checkpoints

The final study used a duration-aware context-16 checkpoint. Checkpoints are ignored by Git and should be stored locally under `checkpoints/` or attached separately as a GitHub Release if public inference without retraining is desired.

## Papers

- `docs/papers/ismir_lbd_2026_submission.pdf`: compact 3-page camera-ready paper accepted at ISMIR 2026 LBD. The PDF is preserved exactly as submitted to the conference.
- `docs/papers/extended_technical_report.pdf`: 10-page extended report with the complete experimental narrative.
