from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


REST = -1
HOLD = -2


DEFAULT_RUN_DIRS = {
    "no_game_joint": "outputs/generated/20260531_135919__no_game_joint__dur_test_no_game_joint",
    "qre_b0p5_cf20": "outputs/generated/20260531_135928__qre_game__dur_test_qre_b0p5_cf20",
    "stk_lower_soft_b0p75": "outputs/generated/20260531_135939__stackelberg_game__dur_test_stk_lower_soft_b0p75",
}


# Reuse the same run_index / piece_idx structure as the original AB design.
# Original:
#   no_game_joint vs stackelberg_upper_soft_b10 -> use for no_game_joint vs stk_lower_soft_b0p75
#   no_game_joint vs qre_cf30                  -> use for no_game_joint vs qre_b0p5_cf20
#   qre_cf30 vs stackelberg_upper_soft_b10     -> use for qre_b0p5_cf20 vs stk_lower_soft_b0p75
DEFAULT_AB_ITEMS = {
    ("no_game_joint", "stk_lower_soft_b0p75"): [
        (27, 65),
        (12, 61),
        (26, 57),
        (20, 64),
    ],
    ("no_game_joint", "qre_b0p5_cf20"): [
        (29, 51),
        (4, 46),
        (16, 59),
        (17, 25),
    ],
    ("qre_b0p5_cf20", "stk_lower_soft_b0p75"): [
        (10, 29),
        (23, 55),
        (2, 8),
        (15, 7),
    ],
}


def load_tokens(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path, allow_pickle=True)
    elif path.suffix.lower() == ".txt":
        arr = np.loadtxt(path, dtype=np.int64)
    else:
        raise ValueError(f"Unsupported token sample file: {path}")

    arr = np.asarray(arr, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected token array shape [T, 2], got {arr.shape} from {path}")
    return arr


def find_sample(run_dir: Path, run_index: int, piece_idx: int) -> Path:
    samples_dir = run_dir / "samples"
    search_dir = samples_dir if samples_dir.exists() else run_dir

    patterns = [
        f"*seed{run_index:03d}_piece{piece_idx:03d}.npy",
        f"*seed{run_index:03d}_piece{piece_idx:03d}.txt",
        f"*seed{run_index:03d}*piece{piece_idx:03d}*.npy",
        f"*seed{run_index:03d}*piece{piece_idx:03d}*.txt",
    ]

    for pattern in patterns:
        matches = sorted(
            p for p in search_dir.glob(pattern)
            if not p.name.endswith(".metrics.txt")
        )
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"Could not find sample seed{run_index:03d}_piece{piece_idx:03d} in {search_dir}"
    )


def tokens_to_musicxml(tokens: np.ndarray, out_path: Path, tempo_bpm: int = 90) -> None:
    """
    Export two-voice raw token sequence to MusicXML with music21.

    Token convention:
      REST = -1
      HOLD = -2
      MIDI pitch = 0..127

    Each timestep is a 16th note, so quarterLength = 0.25.
    HOLD extends the previous sounding note/rest in that voice.
    """
    try:
        from music21 import converter, duration, instrument, metadata, note, stream, tempo
    except Exception as exc:
        raise RuntimeError(
            "music21 is required to export MusicXML. Install it with: pip install music21"
        ) from exc

    tokens = np.asarray(tokens, dtype=np.int64)
    score = stream.Score()
    score.metadata = metadata.Metadata()
    score.metadata.title = out_path.stem
    score.append(tempo.MetronomeMark(number=int(tempo_bpm)))

    part_names = ["Upper", "Lower"]
    instruments = [instrument.Soprano(), instrument.Bass()]

    for voice_idx, part_name in enumerate(part_names):
        part = stream.Part()
        part.id = part_name
        part.partName = part_name
        part.append(instruments[voice_idx])

        events: List[Tuple[int, float]] = []
        current_token = None
        current_len = 0

        for raw in tokens[:, voice_idx]:
            raw = int(raw)

            if raw == HOLD:
                if current_token is None:
                    current_token = REST
                    current_len = 1
                else:
                    current_len += 1
                continue

            if current_token is not None:
                events.append((current_token, current_len * 0.25))

            current_token = raw
            current_len = 1

        if current_token is not None:
            events.append((current_token, current_len * 0.25))

        for raw, ql in events:
            if raw == REST:
                obj = note.Rest()
            else:
                obj = note.Note(int(raw))
            obj.duration = duration.Duration(float(ql))
            part.append(obj)

        score.append(part)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    score.write("musicxml", fp=str(out_path))


def write_txt(path: Path, tokens: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, tokens, fmt="%d")


def make_output_name(comparison: Tuple[str, str], item_idx: int, side: str, ext: str) -> str:
    return f"item{item_idx:03d}_{side}.{ext}"


def build_ab(
    run_dirs: Dict[str, Path],
    output_dir: Path,
    rng_seed: int,
    drop_prefix: int,
    tempo_bpm: int,
) -> None:
    rng = random.Random(int(rng_seed))

    output_dir.mkdir(parents=True, exist_ok=True)

    hidden_rows: List[Dict[str, object]] = []
    answer_rows: List[Dict[str, object]] = []
    selected_rows: List[Dict[str, object]] = []

    for comparison_idx, (method_a, method_b) in enumerate(DEFAULT_AB_ITEMS.keys(), start=1):
        items = DEFAULT_AB_ITEMS[(method_a, method_b)]
        comp_dir_name = f"{method_a}__vs__{method_b}"
        comp_dir = output_dir / comp_dir_name
        comp_dir.mkdir(parents=True, exist_ok=True)

        for item_idx, (run_index, piece_idx) in enumerate(items):
            methods = [method_a, method_b]
            sides = ["A", "B"]
            rng.shuffle(sides)
            method_to_side = {methods[0]: sides[0], methods[1]: sides[1]}

            for method in methods:
                side = method_to_side[method]
                sample_path = find_sample(run_dirs[method], run_index, piece_idx)
                tokens = load_tokens(sample_path)

                if drop_prefix > 0:
                    tokens = tokens[int(drop_prefix):]

                txt_path = comp_dir / make_output_name((method_a, method_b), item_idx, side, "txt")
                xml_path = comp_dir / make_output_name((method_a, method_b), item_idx, side, "musicxml")

                write_txt(txt_path, tokens)
                tokens_to_musicxml(tokens, xml_path, tempo_bpm=tempo_bpm)

                hidden_rows.append(
                    {
                        "comparison": comp_dir_name,
                        "comparison_idx": comparison_idx,
                        "item_idx": item_idx,
                        "side": side,
                        "method": method,
                        "run_index": run_index,
                        "piece_idx": piece_idx,
                        "source_sample": str(sample_path),
                        "txt_path": str(txt_path),
                        "musicxml_path": str(xml_path),
                    }
                )

            selected_rows.append(
                {
                    "comparison": comp_dir_name,
                    "comparison_idx": comparison_idx,
                    "item_idx": item_idx,
                    "run_index": run_index,
                    "piece_idx": piece_idx,
                }
            )

            answer_rows.append(
                {
                    "trial_id": f"{comparison_idx:02d}_{item_idx:03d}",
                    "comparison": comp_dir_name,
                    "item_idx": item_idx,
                    "coordination_preference": "",
                    "coherence_preference": "",
                    "overall_preference": "",
                    "notes": "",
                }
            )

    hidden_manifest = output_dir / "manifest_key_hidden.csv"
    participant_sheet = output_dir / "participant_answer_sheet.csv"
    selected_piece_list = output_dir / "selected_piece_list_for_researchers.csv"
    config_path = output_dir / "ab_config.json"
    readme_path = output_dir / "README.md"

    with hidden_manifest.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "comparison",
            "comparison_idx",
            "item_idx",
            "side",
            "method",
            "run_index",
            "piece_idx",
            "source_sample",
            "txt_path",
            "musicxml_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(hidden_rows)

    with participant_sheet.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "trial_id",
            "comparison",
            "item_idx",
            "coordination_preference",
            "coherence_preference",
            "overall_preference",
            "notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(answer_rows)

    with selected_piece_list.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["comparison", "comparison_idx", "item_idx", "run_index", "piece_idx"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)

    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Duration-aware A/B listening stimuli using the same item structure as the original AB test.",
        "run_dirs": {k: str(v) for k, v in run_dirs.items()},
        "pairs": [list(k) for k in DEFAULT_AB_ITEMS.keys()],
        "num_items_per_pair": 4,
        "total_trials": len(answer_rows),
        "rng_seed": int(rng_seed),
        "drop_prefix": int(drop_prefix),
        "tempo_bpm": int(tempo_bpm),
        "output_dir": str(output_dir),
        "hidden_manifest": str(hidden_manifest),
        "participant_answer_sheet": str(participant_sheet),
        "selected_piece_list_for_researchers": str(selected_piece_list),
    }
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    readme_lines = [
        "# Duration-Aware A/B Listening Stimuli",
        "",
        "This folder contains anonymized A/B listening stimuli for the duration-aware model.",
        "",
        "## Comparisons",
        "",
        "- no_game_joint vs qre_b0p5_cf20",
        "- no_game_joint vs stk_lower_soft_b0p75",
        "- qre_b0p5_cf20 vs stk_lower_soft_b0p75",
        "",
        "## Files",
        "",
        "- `participant_answer_sheet.csv`: anonymized answer sheet for responses.",
        "- `manifest_key_hidden.csv`: hidden key mapping A/B labels to methods. Do not show this to participants.",
        "- `selected_piece_list_for_researchers.csv`: fixed run_index / piece_idx list.",
        "- comparison folders contain `itemXXX_A.musicxml`, `itemXXX_B.musicxml`, and matching `.txt` token files.",
        "",
        "## Rendering",
        "",
        "Render MusicXML to MP3/WAV using MuseScore, then create black-screen MP4 files with ffmpeg.",
    ]
    readme_path.write_text("\n".join(readme_lines), encoding="utf-8")

    print(f"Done. Wrote duration-aware AB stimuli to: {output_dir}")
    print(f"Hidden manifest: {hidden_manifest}")
    print(f"Participant answer sheet: {participant_sheet}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build duration-aware A/B listening stimuli from final test run folders."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/listening_ab/20260531_duration_aware_final_ab_12",
    )
    parser.add_argument("--rng_seed", type=int, default=20260524)
    parser.add_argument("--drop_prefix", type=int, default=16)
    parser.add_argument("--tempo_bpm", type=int, default=90)

    parser.add_argument("--no_game_joint_dir", type=str, default=DEFAULT_RUN_DIRS["no_game_joint"])
    parser.add_argument("--qre_dir", type=str, default=DEFAULT_RUN_DIRS["qre_b0p5_cf20"])
    parser.add_argument("--stackelberg_dir", type=str, default=DEFAULT_RUN_DIRS["stk_lower_soft_b0p75"])

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dirs = {
        "no_game_joint": Path(args.no_game_joint_dir),
        "qre_b0p5_cf20": Path(args.qre_dir),
        "stk_lower_soft_b0p75": Path(args.stackelberg_dir),
    }

    for method, run_dir in run_dirs.items():
        if not run_dir.exists():
            raise FileNotFoundError(f"Run dir for {method} does not exist: {run_dir}")

    build_ab(
        run_dirs=run_dirs,
        output_dir=Path(args.output_dir),
        rng_seed=args.rng_seed,
        drop_prefix=args.drop_prefix,
        tempo_bpm=args.tempo_bpm,
    )


if __name__ == "__main__":
    main()
