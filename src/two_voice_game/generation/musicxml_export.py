from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

HOLD_TOKEN = -2
REST_TOKEN = -1


def load_raw_tokens(path: str | Path) -> np.ndarray:
    """Load two-voice raw tokens from either .npy or human-readable .txt."""
    path = Path(path)
    if path.suffix == ".npy":
        raw_tokens = np.load(path, allow_pickle=True)
    elif path.suffix == ".txt" and not path.name.endswith(".metrics.txt"):
        raw_tokens = np.loadtxt(path, dtype=np.int64)
    else:
        raise ValueError(f"Unsupported token sample file: {path}")

    raw_tokens = np.asarray(raw_tokens, dtype=np.int64)
    if raw_tokens.ndim == 1:
        raw_tokens = raw_tokens.reshape(1, -1)
    return raw_tokens


def token_sequence_to_segments(seq: np.ndarray) -> List[Tuple[Optional[int], int]]:
    """
    Convert timestep tokens into duration segments.

    Returns:
        [(midi_pitch_or_None_for_rest, duration_steps), ...]

    Rules:
        pitch >= 0 starts a new note
        REST_TOKEN starts a rest
        HOLD_TOKEN extends the previous note/rest
        HOLD at the beginning is treated as a rest
    """
    seq = np.asarray(seq, dtype=np.int64)
    segments: List[Tuple[Optional[int], int]] = []

    current_pitch: Optional[int] = None
    current_duration = 0
    has_current = False

    def flush():
        nonlocal current_pitch, current_duration, has_current
        if has_current and current_duration > 0:
            segments.append((current_pitch, current_duration))
        current_pitch = None
        current_duration = 0
        has_current = False

    for tok in seq:
        tok = int(tok)
        if tok == HOLD_TOKEN:
            if not has_current:
                current_pitch = None
                current_duration = 1
                has_current = True
            else:
                current_duration += 1
            continue

        flush()
        if tok == REST_TOKEN:
            current_pitch = None
        elif tok >= 0:
            current_pitch = tok
        else:
            current_pitch = None
        current_duration = 1
        has_current = True

    flush()
    return segments


def write_musicxml(
    raw_tokens: np.ndarray,
    output_path: str | Path,
    tempo_bpm: int = 90,
    time_signature: str = "4/4",
    quarter_length_per_step: float = 0.25,
    title: str = "Generated Two-Voice Piece",
) -> None:
    """
    Export generated two-voice timestep tokens to MusicXML using music21.

    The dataset uses a sixteenth-note grid, so each timestep defaults to
    quarterLength = 0.25.
    """
    try:
        from music21 import stream, note, instrument, meter, tempo, metadata
    except ImportError as exc:
        raise ImportError("music21 is required for MusicXML export. Install it with: pip install music21") from exc

    raw_tokens = np.asarray(raw_tokens, dtype=np.int64)
    if raw_tokens.ndim != 2 or raw_tokens.shape[1] != 2:
        raise ValueError(f"Expected raw_tokens shape [T, 2], got {raw_tokens.shape}")

    score = stream.Score(id="two_voice_score")
    score.insert(0, metadata.Metadata())
    score.metadata.title = title
    score.metadata.composer = "Two-Voice Game Generator"
    score.insert(0, tempo.MetronomeMark(number=tempo_bpm))
    score.insert(0, meter.TimeSignature(time_signature))

    voice_specs = [
        ("Soprano", instrument.Piano(), raw_tokens[:, 0]),
        ("Bass", instrument.Piano(), raw_tokens[:, 1]),
    ]

    for part_name, inst, seq in voice_specs:
        part = stream.Part(id=part_name)
        part.partName = part_name
        part.insert(0, inst)
        for pitch, duration_steps in token_sequence_to_segments(seq):
            ql = max(float(duration_steps) * quarter_length_per_step, quarter_length_per_step)
            n = note.Rest(quarterLength=ql) if pitch is None else note.Note(int(pitch), quarterLength=ql)
            part.append(n)
        score.insert(0, part)

    measured = score.makeMeasures(inPlace=False)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    measured.write("musicxml", fp=str(output_path))
    
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert existing generated .npy or .txt token samples to MusicXML."
    )
    parser.add_argument(
        "--run_dir",
        required=True,
        help="Generated run folder, e.g. outputs/generated/20260516_120003__qre_game__q1",
    )
    parser.add_argument(
        "--samples_subdir",
        default="samples",
        help="Subfolder containing .npy or .txt token samples. Default: samples",
    )
    parser.add_argument(
        "--output_subdir",
        default="musicxml",
        help="Subfolder to write MusicXML files. Default: musicxml",
    )
    parser.add_argument("--tempo_bpm", type=int, default=90)
    parser.add_argument("--time_signature", default="4/4")
    parser.add_argument("--quarter_length_per_step", type=float, default=0.25)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of files to convert.",
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    samples_dir = run_dir / args.samples_subdir
    output_dir = run_dir / args.output_subdir

    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

    if not samples_dir.exists():
        raise FileNotFoundError(f"samples_dir does not exist: {samples_dir}")

    sample_files = sorted(samples_dir.glob("*.npy"))
    if not sample_files:
        sample_files = sorted(
            p
            for p in samples_dir.glob("*.txt")
            if not p.name.endswith(".metrics.txt")
        )

    if args.limit is not None:
        sample_files = sample_files[: args.limit]

    if not sample_files:
        raise FileNotFoundError(f"No .npy or .txt token files found in: {samples_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(sample_files)} token files")
    print(f"Input : {samples_dir}")
    print(f"Output: {output_dir}")

    converted = 0
    skipped = 0

    for sample_path in sample_files:
        raw_tokens = load_raw_tokens(sample_path)

        if raw_tokens.ndim != 2 or raw_tokens.shape[1] != 2:
            print(f"[SKIP] {sample_path.name}: expected shape [T, 2], got {raw_tokens.shape}")
            skipped += 1
            continue

        output_path = output_dir / f"{sample_path.stem}.musicxml"

        write_musicxml(
            raw_tokens=raw_tokens,
            output_path=output_path,
            tempo_bpm=args.tempo_bpm,
            time_signature=args.time_signature,
            quarter_length_per_step=args.quarter_length_per_step,
            title=sample_path.stem,
        )

        print(f"[OK] {sample_path.name} -> {output_path.name}")
        converted += 1

    print(f"Done. Converted: {converted}, skipped: {skipped}")


if __name__ == "__main__":
    main()
