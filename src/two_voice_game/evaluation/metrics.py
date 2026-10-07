from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import math

import numpy as np


REST = -1
HOLD = -2

# Bins used for corpus-similarity histograms.
MELODIC_MIN = -24
MELODIC_MAX = 24
HARMONIC_N_CLASSES = 12
VOICE_DISTANCE_MIN = -24
VOICE_DISTANCE_MAX = 48

# A simple consonance set for two-voice diagnostic analysis.
# Perfect unison/octave, minor/major third, perfect fifth, minor/major sixth.
CONSONANT_INTERVAL_CLASSES = {0, 3, 4, 7, 8, 9}
PERFECT_INTERVAL_CLASSES = {0, 7}


def ensure_two_voice_array(x: np.ndarray | Sequence[Sequence[int]]) -> np.ndarray:
    arr = np.asarray(x, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected a two-voice array with shape [T, 2], got {arr.shape}")
    return arr[:, :2]


def detokenize_hold(tokens: np.ndarray | Sequence[Sequence[int]], rest: int = REST, hold: int = HOLD) -> np.ndarray:
    """
    Convert HOLD tokens back to currently sounding pitches.

    If a generated sequence starts with an invalid HOLD before any pitch has
    sounded, the HOLD is treated as REST instead of crashing. This makes the
    evaluator robust to imperfect generated samples.
    """
    arr = ensure_two_voice_array(tokens)
    restored = np.zeros_like(arr, dtype=np.int64)
    prev = [rest, rest]

    for t in range(arr.shape[0]):
        for voice in range(2):
            tok = int(arr[t, voice])
            if tok == hold:
                restored[t, voice] = prev[voice] if prev[voice] != rest else rest
            else:
                restored[t, voice] = tok
            prev[voice] = int(restored[t, voice])

    return restored


def _safe_slice(arr: np.ndarray, drop_prefix: int = 0, max_steps: int | None = None) -> np.ndarray:
    arr = ensure_two_voice_array(arr)
    start = max(int(drop_prefix), 0)
    out = arr[start:]
    if max_steps is not None and max_steps > 0:
        out = out[: int(max_steps)]
    return out


def _hist(values: Iterable[int], min_value: int, max_value: int) -> np.ndarray:
    size = max_value - min_value + 1
    counts = np.zeros(size, dtype=np.float64)
    for v in values:
        iv = int(v)
        if iv < min_value:
            iv = min_value
        elif iv > max_value:
            iv = max_value
        counts[iv - min_value] += 1.0
    return counts


def _entropy_from_counts(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total <= 0:
        return 0.0
    p = counts.astype(np.float64) / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def js_divergence(p_counts: np.ndarray, q_counts: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence between two count vectors, in natural-log units."""
    p = np.asarray(p_counts, dtype=np.float64) + eps
    q = np.asarray(q_counts, dtype=np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


@dataclass
class CorpusStats:
    name: str
    n_pieces: int = 0
    n_steps: int = 0

    soprano_interval_counts: np.ndarray = field(default_factory=lambda: np.zeros(MELODIC_MAX - MELODIC_MIN + 1, dtype=np.float64))
    bass_interval_counts: np.ndarray = field(default_factory=lambda: np.zeros(MELODIC_MAX - MELODIC_MIN + 1, dtype=np.float64))
    harmonic_interval_counts: np.ndarray = field(default_factory=lambda: np.zeros(HARMONIC_N_CLASSES, dtype=np.float64))
    voice_distance_counts: np.ndarray = field(default_factory=lambda: np.zeros(VOICE_DISTANCE_MAX - VOICE_DISTANCE_MIN + 1, dtype=np.float64))
    pitch_class_counts: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.float64))
    pitch_counts: np.ndarray = field(default_factory=lambda: np.zeros(128, dtype=np.float64))

    hold_count: int = 0
    token_count: int = 0

    crossing_count: int = 0
    vertical_count: int = 0
    dissonance_count: int = 0

    parallel_perfect_count: int = 0
    parallel_transition_count: int = 0

    excessive_leap_count: int = 0
    melodic_transition_count: int = 0

    unique_3gram_count: int = 0
    total_3gram_count: int = 0

    voice_distance_sum: float = 0.0
    voice_distance_n: int = 0

    def scalar_metrics(self) -> Dict[str, float]:
        return {
            "n_pieces": float(self.n_pieces),
            "n_steps": float(self.n_steps),
            "hold_ratio": _ratio(self.hold_count, self.token_count),
            "voice_crossing_rate": _ratio(self.crossing_count, self.vertical_count),
            "dissonance_rate": _ratio(self.dissonance_count, self.vertical_count),
            "parallel_perfect_rate": _ratio(self.parallel_perfect_count, self.parallel_transition_count),
            "excessive_leap_rate": _ratio(self.excessive_leap_count, self.melodic_transition_count),
            "pitch_entropy": _entropy_from_counts(self.pitch_counts),
            "pitch_class_entropy": _entropy_from_counts(self.pitch_class_counts),
            "soprano_interval_entropy": _entropy_from_counts(self.soprano_interval_counts),
            "bass_interval_entropy": _entropy_from_counts(self.bass_interval_counts),
            "harmonic_interval_entropy": _entropy_from_counts(self.harmonic_interval_counts),
            "unique_3gram_ratio": _ratio(self.unique_3gram_count, self.total_3gram_count),
            "mean_voice_distance": _ratio(self.voice_distance_sum, self.voice_distance_n),
        }


def _collect_melodic_intervals(detok: np.ndarray, voice: int) -> List[int]:
    values: List[int] = []
    for t in range(1, len(detok)):
        prev_pitch = int(detok[t - 1, voice])
        cur_pitch = int(detok[t, voice])
        if prev_pitch == REST or cur_pitch == REST:
            continue
        values.append(cur_pitch - prev_pitch)
    return values


def _update_piece_stats(stats: CorpusStats, raw_tokens: np.ndarray) -> None:
    raw = ensure_two_voice_array(raw_tokens)
    if raw.size == 0:
        return

    detok = detokenize_hold(raw)
    stats.n_pieces += 1
    stats.n_steps += int(raw.shape[0])
    stats.hold_count += int(np.sum(raw == HOLD))
    stats.token_count += int(raw.size)

    # Melodic interval distributions and excessive leaps.
    for voice, attr in [(0, "soprano_interval_counts"), (1, "bass_interval_counts")]:
        intervals = _collect_melodic_intervals(detok, voice)
        setattr(stats, attr, getattr(stats, attr) + _hist(intervals, MELODIC_MIN, MELODIC_MAX))
        stats.excessive_leap_count += sum(1 for x in intervals if abs(int(x)) > 12)
        stats.melodic_transition_count += len(intervals)

    # Vertical distributions and pitch distributions.
    for t in range(len(detok)):
        s = int(detok[t, 0])
        b = int(detok[t, 1])

        for p in (s, b):
            if p != REST and 0 <= p <= 127:
                stats.pitch_counts[p] += 1.0
                stats.pitch_class_counts[p % 12] += 1.0

        if s == REST or b == REST:
            continue

        distance = s - b
        abs_interval = abs(distance)
        interval_class = abs_interval % HARMONIC_N_CLASSES

        stats.vertical_count += 1
        stats.voice_distance_sum += float(distance)
        stats.voice_distance_n += 1
        stats.harmonic_interval_counts[interval_class] += 1.0
        stats.voice_distance_counts += _hist([distance], VOICE_DISTANCE_MIN, VOICE_DISTANCE_MAX)

        if s < b:
            stats.crossing_count += 1
        if interval_class not in CONSONANT_INTERVAL_CLASSES:
            stats.dissonance_count += 1

    # Parallel perfect fifth/octave diagnostic.
    for t in range(1, len(detok)):
        s0, b0 = int(detok[t - 1, 0]), int(detok[t - 1, 1])
        s1, b1 = int(detok[t, 0]), int(detok[t, 1])
        if REST in (s0, b0, s1, b1):
            continue
        stats.parallel_transition_count += 1
        prev_ic = abs(s0 - b0) % 12
        cur_ic = abs(s1 - b1) % 12
        ds = s1 - s0
        db = b1 - b0
        same_direction = (ds > 0 and db > 0) or (ds < 0 and db < 0)
        both_move = ds != 0 and db != 0
        if prev_ic in PERFECT_INTERVAL_CLASSES and cur_ic in PERFECT_INTERVAL_CLASSES and same_direction and both_move:
            stats.parallel_perfect_count += 1

    # Diversity: unique 3-gram ratio over detokenized pitch pairs.
    if len(detok) >= 3:
        grams = []
        for t in range(len(detok) - 2):
            gram = tuple(map(tuple, detok[t : t + 3].tolist()))
            grams.append(gram)
        stats.unique_3gram_count += len(set(grams))
        stats.total_3gram_count += len(grams)


def collect_corpus_stats(
    name: str,
    sequences: Sequence[np.ndarray],
    drop_prefix: int = 0,
    max_steps: int | None = None,
) -> CorpusStats:
    stats = CorpusStats(name=name)
    for seq in sequences:
        arr = _safe_slice(np.asarray(seq, dtype=np.int64), drop_prefix=drop_prefix, max_steps=max_steps)
        if len(arr) == 0:
            continue
        _update_piece_stats(stats, arr)
    return stats


def load_npz_split(data_path: str | Path, split: str, token_field: str = "tokens") -> List[np.ndarray]:
    data = np.load(Path(data_path), allow_pickle=True)
    key = f"{split}_{token_field}"
    if key not in data:
        raise KeyError(f"Missing key '{key}' in {data_path}. Available keys: {list(data.keys())}")
    return [ensure_two_voice_array(np.asarray(x, dtype=np.int64)) for x in data[key]]


def load_generated_npy(method_dir: str | Path) -> List[np.ndarray]:
    method_dir = Path(method_dir)
    if not method_dir.exists():
        return []
    files = sorted(method_dir.glob("*.npy"))
    return [ensure_two_voice_array(np.load(path, allow_pickle=True)) for path in files]


def comparison_to_real(real: CorpusStats, method: CorpusStats) -> Dict[str, float]:
    """Corpus-similarity distances and scalar gaps from a method to real music."""
    real_scalars = real.scalar_metrics()
    method_scalars = method.scalar_metrics()

    return {
        "soprano_melodic_interval_js": js_divergence(real.soprano_interval_counts, method.soprano_interval_counts),
        "bass_melodic_interval_js": js_divergence(real.bass_interval_counts, method.bass_interval_counts),
        "harmonic_interval_js": js_divergence(real.harmonic_interval_counts, method.harmonic_interval_counts),
        "voice_distance_js": js_divergence(real.voice_distance_counts, method.voice_distance_counts),
        "pitch_class_js": js_divergence(real.pitch_class_counts, method.pitch_class_counts),
        "hold_ratio_gap": abs(method_scalars["hold_ratio"] - real_scalars["hold_ratio"]),
        "pitch_entropy_gap": abs(method_scalars["pitch_entropy"] - real_scalars["pitch_entropy"]),
        "unique_3gram_ratio_gap": abs(method_scalars["unique_3gram_ratio"] - real_scalars["unique_3gram_ratio"]),
        "mean_voice_distance_gap": abs(method_scalars["mean_voice_distance"] - real_scalars["mean_voice_distance"]),
    }


def write_wide_csv(path: str | Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_float(x: object) -> object:
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return ""
        return f"{x:.6f}"
    return x
