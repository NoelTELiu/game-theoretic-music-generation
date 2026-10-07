# Data

The raw dataset is intentionally **not redistributed** in this repository.

This project expects the 16th-note, voice-separated JSB Chorales JSON file:

`data/raw/Jsb16thSeparated.json`

A public copy is available from:

- https://github.com/czhuang/JSB-Chorales-dataset

That repository provides the standard `train`, `valid`, and `test` splits used in prior JSB Chorales work. The preprocessing code in this repository extracts soprano and bass, represents silence as `REST=-1`, and converts consecutive repeated non-rest pitches into `HOLD=-2` tokens.

After placing the JSON file in `data/raw/`, run:

```bash
python scripts/preprocess_data.py
```

This creates:

```text
data/processed/jsb_soprano_bass_16th.json
data/processed/jsb_two_voice_tokens.npz
```

Expected split sizes are 229 train, 76 validation, and 77 test chorales.
