#!/usr/bin/env python3
"""
Preprocess JSB Chorales SATB data into a two-voice Soprano+Bass dataset.

Input:
    data/raw/Jsb16thSeparated.json

Output:
    data/processed/jsb_soprano_bass_16th.json
    data/processed/jsb_two_voice_tokens.npz

Assumptions:
    - Each time step has 4 voice-separated pitches: [Soprano, Alto, Tenor, Bass].
    - REST / silence is represented by -1 in JSON.
    - Optional HOLD token is created by replacing consecutive repeated pitches
      with HOLD = -2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


REST = -1
HOLD = -2

SPLITS = ("train", "valid", "test")


def load_jsb_json(path: Path) -> Dict[str, Any]:
    """Load Jsb16thSeparated.json."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    for split in SPLITS:
        if split not in data:
            raise ValueError(f"Missing split '{split}' in {path}")

    return data


def extract_soprano_bass(piece: List[List[int]]) -> List[List[int]]:
    """
    Extract Soprano and Bass from one SATB piece.

    Expected input time step:
        [soprano, alto, tenor, bass]

    Output time step:
        [soprano, bass]
    """
    sb_piece: List[List[int]] = []

    for t, row in enumerate(piece):
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError(f"Invalid SATB row at t={t}: {row}")

        soprano = int(row[0])
        bass = int(row[3])
        sb_piece.append([soprano, bass])

    return sb_piece


def add_hold_tokens(sb_piece: List[List[int]]) -> List[List[int]]:
    """
    Convert consecutive repeated pitches to HOLD tokens.

    Example:
        [[74, 58], [74, 58], [75, 55]]
    becomes:
        [[74, 58], [HOLD, HOLD], [75, 55]]

    REST is kept as REST. Repeated REST remains REST, not HOLD.
    """
    if not sb_piece:
        return []

    output: List[List[int]] = []
    prev = [None, None]

    for row in sb_piece:
        new_row: List[int] = []

        for voice_idx in range(2):
            current = row[voice_idx]
            previous = prev[voice_idx]

            if current == REST:
                token = REST
            elif previous is not None and current == previous and previous != REST:
                token = HOLD
            else:
                token = current

            new_row.append(int(token))

        output.append(new_row)
        prev = row

    return output


def detokenize_hold(tokens: List[List[int]]) -> List[List[int]]:
    """
    Convert HOLD tokens back to currently sounding pitches.

    This is used for sanity checking:
        original MIDI sequence should equal detokenized HOLD-token sequence.
    """
    restored: List[List[int]] = []
    prev = [REST, REST]

    for t, row in enumerate(tokens):
        restored_row: List[int] = []

        for voice_idx in range(2):
            token = row[voice_idx]

            if token == HOLD:
                restored_pitch = prev[voice_idx]
                if restored_pitch == REST:
                    raise ValueError(
                        f"HOLD appears after REST at t={t}, voice={voice_idx}"
                    )
            else:
                restored_pitch = token

            restored_row.append(int(restored_pitch))

        restored.append(restored_row)
        prev = restored_row

    return restored


def process_dataset(
    data: Dict[str, Any],
    use_hold: bool = True,
) -> Dict[str, Any]:
    """Extract Soprano+Bass for all splits."""
    processed: Dict[str, Any] = {
        "metadata": {
            "source": "JSB Chorales Jsb16thSeparated.json",
            "representation": "two_voice_soprano_bass",
            "time_resolution": "1/16 note",
            "voices": ["soprano", "bass"],
            "voice_indices_from_satb": [0, 3],
            "rest_token": REST,
            "hold_token": HOLD if use_hold else None,
            "uses_hold_tokens": use_hold,
            "note": (
                "original_midi stores currently sounding pitches; "
                "tokens may replace repeated pitches with HOLD."
            ),
        }
    }

    for split in SPLITS:
        processed[split] = []

        for i, piece in enumerate(data[split]):
            original_sb = extract_soprano_bass(piece)
            tokens = add_hold_tokens(original_sb) if use_hold else original_sb

            item = {
                "piece_id": f"{split}_{i:03d}",
                "tokens": tokens,
                "original_midi": original_sb,
                "length": len(original_sb),
            }
            processed[split].append(item)

    return processed


def save_json(processed: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2)


def save_npz(processed: Dict[str, Any], output_path: Path) -> None:
    """
    Save variable-length sequences as object arrays.

    Loading later:
        data = np.load(path, allow_pickle=True)
        train_tokens = data["train_tokens"]
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    npz_data = {}

    for split in SPLITS:
        tokens = [
            np.array(item["tokens"], dtype=np.int16)
            for item in processed[split]
        ]
        original = [
            np.array(item["original_midi"], dtype=np.int16)
            for item in processed[split]
        ]

        npz_data[f"{split}_tokens"] = np.array(tokens, dtype=object)
        npz_data[f"{split}_original_midi"] = np.array(original, dtype=object)
        npz_data[f"{split}_piece_ids"] = np.array(
            [item["piece_id"] for item in processed[split]],
            dtype=object,
        )

    np.savez_compressed(output_path, **npz_data)


def collect_all_rows(processed: Dict[str, Any], split: str, field: str) -> np.ndarray:
    arrays = [np.array(item[field], dtype=np.int16) for item in processed[split]]
    if not arrays:
        return np.empty((0, 2), dtype=np.int16)
    return np.vstack(arrays)


def run_sanity_checks(processed: Dict[str, Any], use_hold: bool = True) -> None:
    """
    Basic tests:
    1. Splits exist.
    2. Every row has exactly two voices.
    3. Detokenizing HOLD restores original_midi.
    4. Soprano average pitch is higher than bass average pitch.
    5. Token vocabulary only contains REST, HOLD, or MIDI pitch range.
    """
    print("\n=== Sanity checks ===")

    for split in SPLITS:
        pieces = processed[split]
        print(f"{split}: {len(pieces)} pieces")

        if len(pieces) == 0:
            raise AssertionError(f"{split} split is empty")

        for item in pieces:
            tokens = item["tokens"]
            original = item["original_midi"]

            if len(tokens) != len(original):
                raise AssertionError(f"Length mismatch in {item['piece_id']}")

            for row in tokens:
                if len(row) != 2:
                    raise AssertionError(
                        f"Expected 2 voices in {item['piece_id']}, got row={row}"
                    )

            if use_hold:
                restored = detokenize_hold(tokens)
                if restored != original:
                    raise AssertionError(
                        f"HOLD detokenization mismatch in {item['piece_id']}"
                    )

        all_original = collect_all_rows(processed, split, "original_midi")
        soprano = all_original[:, 0]
        bass = all_original[:, 1]

        soprano_nonrest = soprano[soprano != REST]
        bass_nonrest = bass[bass != REST]

        print(
            f"  soprano range: {soprano_nonrest.min()}–{soprano_nonrest.max()}, "
            f"mean={soprano_nonrest.mean():.2f}"
        )
        print(
            f"  bass range:    {bass_nonrest.min()}–{bass_nonrest.max()}, "
            f"mean={bass_nonrest.mean():.2f}"
        )

        if soprano_nonrest.mean() <= bass_nonrest.mean():
            raise AssertionError(
                f"Unexpected voice statistics in {split}: "
                "soprano mean is not higher than bass mean"
            )

        all_tokens = collect_all_rows(processed, split, "tokens").flatten()
        valid_mask = (
            (all_tokens == REST)
            | (all_tokens == HOLD)
            | ((all_tokens >= 0) & (all_tokens <= 127))
        )

        if not np.all(valid_mask):
            bad = all_tokens[~valid_mask][:20]
            raise AssertionError(f"Invalid tokens found in {split}: {bad}")

        rest_rate = float(np.mean(all_tokens == REST))
        hold_rate = float(np.mean(all_tokens == HOLD)) if use_hold else 0.0
        print(f"  REST rate: {rest_rate:.4f}")
        print(f"  HOLD rate: {hold_rate:.4f}")

    print("All sanity checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/Jsb16thSeparated.json"),
        help="Path to Jsb16thSeparated.json",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/processed/jsb_soprano_bass_16th.json"),
        help="Output processed JSON path",
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=Path("data/processed/jsb_two_voice_tokens.npz"),
        help="Output processed NPZ path",
    )
    parser.add_argument(
        "--no-hold",
        action="store_true",
        help="Disable HOLD-token conversion",
    )

    args = parser.parse_args()
    use_hold = not args.no_hold

    raw = load_jsb_json(args.input)
    processed = process_dataset(raw, use_hold=use_hold)

    run_sanity_checks(processed, use_hold=use_hold)

    save_json(processed, args.output_json)
    save_npz(processed, args.output_npz)

    print("\nSaved:")
    print(f"  JSON: {args.output_json}")
    print(f"  NPZ:  {args.output_npz}")


if __name__ == "__main__":
    main()