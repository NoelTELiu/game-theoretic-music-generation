from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from two_voice_game.generation.musicxml_export import load_raw_tokens, write_musicxml  # noqa: E402


SAMPLE_RE = re.compile(r"seed(?P<run_index>\d+)_piece(?P<piece_idx>\d+)")
DEFAULT_PAIR_PRIORITY = [
    ("no_game_joint", "stackelberg_upper_soft_b10"),
    ("no_game_joint", "qre_cf30"),
    ("qre_cf30", "stackelberg_upper_soft_b10"),
    ("no_game_joint", "nash_game"),
]


def parse_run_dir_spec(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(
            "Run dirs must be written as method=path, e.g. "
            "no_game_joint=outputs/generated/..."
        )
    method, path_text = spec.split("=", 1)
    method = method.strip()
    if not method:
        raise ValueError(f"Missing method name in run dir spec: {spec!r}")
    return method, Path(path_text.strip())


def parse_pair_spec(spec: str) -> Tuple[str, str]:
    if ":" not in spec:
        raise ValueError(
            "Pair specs must be written as method_a:method_b, e.g. "
            "no_game_joint:stackelberg_upper_soft_b10"
        )
    left, right = spec.split(":", 1)
    return left.strip(), right.strip()


def safe_name(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[^A-Za-z0-9_.=-]+", "-", text)
    return text.strip("-") or "item"


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [jsonable(x) for x in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    return value


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(jsonable(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def find_sample_files(run_dir: Path) -> List[Path]:
    samples_dir = run_dir / "samples"
    search_dir = samples_dir if samples_dir.exists() else run_dir

    npy_files = sorted(search_dir.glob("*.npy"))
    if npy_files:
        return npy_files

    return sorted(
        p
        for p in search_dir.glob("*.txt")
        if not p.name.endswith(".metrics.txt")
    )


def index_samples(run_dir: Path) -> Dict[Tuple[int, int], Path]:
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    indexed: Dict[Tuple[int, int], Path] = {}
    for path in find_sample_files(run_dir):
        match = SAMPLE_RE.search(path.stem)
        if not match:
            continue
        key = (int(match.group("run_index")), int(match.group("piece_idx")))
        indexed[key] = path

    if not indexed:
        raise ValueError(f"No seedXXX_pieceYYY token samples found in {run_dir}")

    return indexed


def load_clip_tokens(path: Path, drop_prefix: int) -> np.ndarray:
    tokens = load_raw_tokens(path)
    if tokens.ndim != 2 or tokens.shape[1] != 2:
        raise ValueError(f"Expected token sample shape [T, 2], got {tokens.shape} in {path}")
    if drop_prefix > 0:
        tokens = tokens[int(drop_prefix) :]
    if len(tokens) == 0:
        raise ValueError(f"Sample became empty after drop_prefix={drop_prefix}: {path}")
    return tokens


def choose_default_pairs(methods: set[str]) -> List[Tuple[str, str]]:
    pairs = [pair for pair in DEFAULT_PAIR_PRIORITY if pair[0] in methods and pair[1] in methods]
    if pairs:
        return pairs

    sorted_methods = sorted(methods)
    return [
        (sorted_methods[i], sorted_methods[j])
        for i in range(len(sorted_methods))
        for j in range(i + 1, len(sorted_methods))
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build anonymized pairwise A/B MusicXML stimuli from matched generated "
            "two-voice samples."
        )
    )
    parser.add_argument(
        "--run_dirs",
        nargs="+",
        required=True,
        help=(
            "Run directories as method=path. Example: "
            "no_game_joint=outputs/generated/... stackelberg_upper_soft_b10=outputs/generated/..."
        ),
    )
    parser.add_argument(
        "--pairs",
        nargs="*",
        default=None,
        help=(
            "Method pairs as method_a:method_b. If omitted, recommended pairs are "
            "chosen from available methods."
        ),
    )
    parser.add_argument("--num_items", type=int, default=8, help="Number of matched seeds per method pair.")
    parser.add_argument("--rng_seed", type=int, default=20260524)
    parser.add_argument(
        "--drop_prefix",
        type=int,
        default=16,
        help=(
            "Initial timesteps to remove before writing listening clips. "
            "Use 16 to evaluate only generated continuation, or 0 to include the real seed prefix."
        ),
    )
    parser.add_argument("--tempo_bpm", type=int, default=90)
    parser.add_argument("--output_root", type=str, default="outputs/listening_ab")
    parser.add_argument("--output_name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.rng_seed)

    method_to_run_dir = dict(parse_run_dir_spec(spec) for spec in args.run_dirs)
    method_to_samples = {
        method: index_samples(run_dir)
        for method, run_dir in method_to_run_dir.items()
    }

    if args.pairs:
        pairs = [parse_pair_spec(spec) for spec in args.pairs]
    else:
        pairs = choose_default_pairs(set(method_to_run_dir.keys()))

    if not pairs:
        raise ValueError("No method pairs available. Provide at least two --run_dirs or explicit --pairs.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = safe_name(args.output_name) if args.output_name else "ab_listening"
    output_dir = Path(args.output_root) / f"{timestamp}__{output_name}"
    output_dir.mkdir(parents=True, exist_ok=False)

    hidden_rows: List[Dict[str, Any]] = []
    participant_rows: List[Dict[str, Any]] = []

    for method_a, method_b in pairs:
        if method_a not in method_to_samples or method_b not in method_to_samples:
            raise KeyError(f"Unknown pair methods: {method_a}, {method_b}")

        common_keys = sorted(set(method_to_samples[method_a]) & set(method_to_samples[method_b]))
        if not common_keys:
            raise ValueError(f"No matched seed/piece samples for pair {method_a}:{method_b}")

        rng.shuffle(common_keys)
        selected_keys = common_keys[: int(args.num_items)]
        pair_name = f"{safe_name(method_a)}__vs__{safe_name(method_b)}"
        pair_dir = output_dir / pair_name
        pair_dir.mkdir(parents=True, exist_ok=True)

        for item_index, key in enumerate(selected_keys):
            run_index, piece_idx = key
            a_first = rng.random() < 0.5
            label_to_method = {
                "A": method_a if a_first else method_b,
                "B": method_b if a_first else method_a,
            }

            for label, method in label_to_method.items():
                source_path = method_to_samples[method][key]
                tokens = load_clip_tokens(source_path, drop_prefix=args.drop_prefix)
                stem = f"item{item_index:03d}_{label}"

                np.savetxt(pair_dir / f"{stem}.txt", tokens, fmt="%d")
                write_musicxml(
                    raw_tokens=tokens,
                    output_path=pair_dir / f"{stem}.musicxml",
                    tempo_bpm=args.tempo_bpm,
                    title=f"{pair_name} {stem}",
                )

                hidden_rows.append(
                    {
                        "pair_name": pair_name,
                        "item_index": item_index,
                        "run_index": run_index,
                        "piece_idx": piece_idx,
                        "label": label,
                        "method": method,
                        "source_path": source_path,
                        "stimulus_txt": pair_dir / f"{stem}.txt",
                        "stimulus_musicxml": pair_dir / f"{stem}.musicxml",
                    }
                )

            participant_rows.append(
                {
                    "pair_name": pair_name,
                    "item_index": item_index,
                    "A_musicxml": pair_dir / f"item{item_index:03d}_A.musicxml",
                    "B_musicxml": pair_dir / f"item{item_index:03d}_B.musicxml",
                    "better_coordination": "",
                    "more_coherent": "",
                    "overall_preference": "",
                    "notes": "",
                }
            )

    hidden_manifest = output_dir / "manifest_key_hidden.csv"
    participant_sheet = output_dir / "participant_answer_sheet.csv"

    with hidden_manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(hidden_rows[0].keys()))
        writer.writeheader()
        for row in hidden_rows:
            writer.writerow({k: jsonable(v) for k, v in row.items()})

    with participant_sheet.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(participant_rows[0].keys()))
        writer.writeheader()
        for row in participant_rows:
            writer.writerow({k: jsonable(v) for k, v in row.items()})

    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dirs": method_to_run_dir,
        "pairs": pairs,
        "num_items": args.num_items,
        "rng_seed": args.rng_seed,
        "drop_prefix": args.drop_prefix,
        "tempo_bpm": args.tempo_bpm,
        "output_dir": output_dir,
        "hidden_manifest": hidden_manifest,
        "participant_answer_sheet": participant_sheet,
    }
    write_json(output_dir / "ab_config.json", config)

    readme_lines = [
        "# A/B Listening Stimuli",
        "",
        "Files in each pair folder are anonymized as `A` and `B`.",
        "Use `participant_answer_sheet.csv` for data collection.",
        "Keep `manifest_key_hidden.csv` hidden until after responses are collected.",
        "",
        f"- Drop prefix: `{args.drop_prefix}` timesteps",
        f"- Tempo: `{args.tempo_bpm}` BPM",
        f"- Items per pair: `{args.num_items}`",
        "",
        "Suggested questions:",
        "",
        "- Which clip has better coordination between the two voices?",
        "- Which clip sounds more musically coherent?",
        "- Which clip do you prefer overall?",
    ]
    (output_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")

    print(f"Saved A/B stimuli: {output_dir}")
    print(f"Hidden key: {hidden_manifest}")
    print(f"Participant sheet: {participant_sheet}")


if __name__ == "__main__":
    main()
