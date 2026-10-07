from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


COUNT_ATTRS = [
    "soprano_interval_counts",
    "bass_interval_counts",
    "harmonic_interval_counts",
    "voice_distance_counts",
    "pitch_class_counts",
    "pitch_counts",
]
SCALAR_ATTRS = [
    "hold_count",
    "token_count",
    "unique_3gram_count",
    "total_3gram_count",
    "voice_distance_sum",
    "voice_distance_n",
]
METRIC_KEYS = [
    "bass_melodic_interval_js",
    "harmonic_interval_js",
    "hold_ratio_gap",
    "mean_voice_distance_gap",
    "pitch_class_js",
    "pitch_entropy_gap",
    "soprano_melodic_interval_js",
    "unique_3gram_ratio_gap",
    "voice_distance_js",
]
PAIR_COLUMNS = ["run_index", "piece_idx", "sample_rng_seed"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Paired percentile bootstrap for matched final-test continuations. "
            "The real test corpus is kept fixed; the matched generated continuation indices "
            "are resampled with replacement and shared across all methods."
        )
    )
    p.add_argument("--data", default="data/processed/jsb_two_voice_tokens.npz")
    p.add_argument("--real_split", default="test", choices=["train", "valid", "test"])
    p.add_argument(
        "--run_specs",
        nargs="+",
        required=True,
        help=(
            "label=run_dir entries. Example: "
            "no_game=outputs/generated/... qre=outputs/generated/..."
        ),
    )
    p.add_argument("--baseline", default="no_game")
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260816)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument(
        "--metrics_py",
        default="src/two_voice_game/evaluation/metrics.py",
        help="Exact evaluator metrics.py used for the paper.",
    )
    p.add_argument("--output_dir", default="outputs/bootstrap/final_test_paired")
    return p.parse_args()


def load_metrics_module(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"metrics.py not found: {path}")
    spec = importlib.util.spec_from_file_location("bootstrap_paper_metrics", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load evaluator module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_run_specs(specs: List[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected label=run_dir, got: {spec}")
        label, path = spec.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Empty label in: {spec}")
        if label in out:
            raise ValueError(f"Duplicate label: {label}")
        out[label] = Path(path.strip())
    return out


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def pairing_signature(rows: List[Dict[str, str]]) -> List[Tuple[int, int, int]]:
    rows = sorted(rows, key=lambda r: int(r["run_index"]))
    return [
        (int(r["run_index"]), int(r["piece_idx"]), int(r["sample_rng_seed"]))
        for r in rows
    ]


def infer_drop_prefix(run_dir: Path) -> int:
    cfg_path = run_dir / "run_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing run_config.json in {run_dir}")
    cfg = read_json(cfg_path)
    if "context_len" not in cfg:
        raise KeyError(f"context_len missing from {cfg_path}")
    return int(cfg["context_len"])


def find_sample_file(run_dir: Path, run_index: int, piece_idx: int) -> Path:
    samples_dir = run_dir / "samples"
    pattern = f"*seed{run_index:03d}_piece{piece_idx:03d}.npy"
    matches = sorted(samples_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one sample for {pattern} in {samples_dir}, got {len(matches)}"
        )
    return matches[0]


def load_real_pieces(data_path: Path, split: str) -> List[np.ndarray]:
    data = np.load(data_path, allow_pickle=True)
    key = f"{split}_tokens"
    if key not in data:
        raise KeyError(f"Missing {key} in {data_path}. Available: {list(data.keys())}")
    return [np.asarray(x, dtype=np.int64)[:, :2] for x in data[key]]


def precompute_piece_features(metrics_mod, run_dir: Path, manifest_rows: List[Dict[str, str]]):
    rows = sorted(manifest_rows, key=lambda r: int(r["run_index"]))
    drop_prefix = infer_drop_prefix(run_dir)
    piece_stats = []
    shapes = []

    for row in rows:
        run_index = int(row["run_index"])
        piece_idx = int(row["piece_idx"])
        path = find_sample_file(run_dir, run_index, piece_idx)
        arr = np.asarray(np.load(path, allow_pickle=True), dtype=np.int64)
        shapes.append(tuple(arr.shape))
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"Expected [T,2], got {arr.shape}: {path}")
        arr = arr[drop_prefix:]
        if len(arr) == 0:
            raise ValueError(f"Sample empty after drop_prefix={drop_prefix}: {path}")
        piece_stats.append(metrics_mod.collect_corpus_stats("piece", [arr]))

    features: Dict[str, np.ndarray] = {}
    for attr in COUNT_ATTRS:
        features[attr] = np.stack([getattr(s, attr) for s in piece_stats], axis=0)
    for attr in SCALAR_ATTRS:
        features[attr] = np.asarray([getattr(s, attr) for s in piece_stats], dtype=np.float64)

    return features, drop_prefix, sorted(set(shapes))


def batch_js(real_counts: np.ndarray, generated_counts: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Vectorized copy of the paper evaluator's natural-log JS divergence."""
    p = np.asarray(real_counts, dtype=np.float64) + eps
    p = p / p.sum()
    q = np.asarray(generated_counts, dtype=np.float64) + eps
    q = q / q.sum(axis=1, keepdims=True)
    m = 0.5 * (q + p[None, :])
    return 0.5 * np.sum(p[None, :] * np.log(p[None, :] / m), axis=1) + 0.5 * np.sum(
        q * np.log(q / m), axis=1
    )


def aggregate_metrics(weights: np.ndarray, features: Dict[str, np.ndarray], real_stats) -> Dict[str, np.ndarray]:
    counts = {attr: weights @ features[attr] for attr in COUNT_ATTRS}
    scalars = {attr: weights @ features[attr] for attr in SCALAR_ATTRS}

    hold_ratio = scalars["hold_count"] / scalars["token_count"]
    unique_3gram_ratio = scalars["unique_3gram_count"] / scalars["total_3gram_count"]
    mean_voice_distance = scalars["voice_distance_sum"] / scalars["voice_distance_n"]

    pitch_counts = counts["pitch_counts"]
    pitch_totals = pitch_counts.sum(axis=1, keepdims=True)
    pitch_p = np.divide(
        pitch_counts,
        pitch_totals,
        out=np.zeros_like(pitch_counts, dtype=np.float64),
        where=pitch_totals > 0,
    )
    log_pitch_p = np.zeros_like(pitch_p)
    np.log(pitch_p, out=log_pitch_p, where=pitch_p > 0)
    pitch_entropy = -np.sum(np.where(pitch_p > 0, pitch_p * log_pitch_p, 0.0), axis=1)

    real_scalars = real_stats.scalar_metrics()
    return {
        "soprano_melodic_interval_js": batch_js(real_stats.soprano_interval_counts, counts["soprano_interval_counts"]),
        "bass_melodic_interval_js": batch_js(real_stats.bass_interval_counts, counts["bass_interval_counts"]),
        "harmonic_interval_js": batch_js(real_stats.harmonic_interval_counts, counts["harmonic_interval_counts"]),
        "voice_distance_js": batch_js(real_stats.voice_distance_counts, counts["voice_distance_counts"]),
        "pitch_class_js": batch_js(real_stats.pitch_class_counts, counts["pitch_class_counts"]),
        "hold_ratio_gap": np.abs(hold_ratio - real_scalars["hold_ratio"]),
        "pitch_entropy_gap": np.abs(pitch_entropy - real_scalars["pitch_entropy"]),
        "unique_3gram_ratio_gap": np.abs(unique_3gram_ratio - real_scalars["unique_3gram_ratio"]),
        "mean_voice_distance_gap": np.abs(mean_voice_distance - real_scalars["mean_voice_distance"]),
    }


def quantile_ci(values: np.ndarray, ci: float) -> Tuple[float, float]:
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(values, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not (0.0 < args.ci < 1.0):
        raise ValueError("--ci must be between 0 and 1")
    if args.n_boot <= 0:
        raise ValueError("--n_boot must be positive")

    run_dirs = parse_run_specs(args.run_specs)
    if args.baseline not in run_dirs:
        raise ValueError(f"Baseline '{args.baseline}' not found in run_specs: {list(run_dirs)}")

    metrics_mod = load_metrics_module(Path(args.metrics_py))
    real_pieces = load_real_pieces(Path(args.data), args.real_split)
    real_stats = metrics_mod.collect_corpus_stats(f"real_{args.real_split}", real_pieces)

    manifests: Dict[str, List[Dict[str, str]]] = {}
    signatures: Dict[str, List[Tuple[int, int, int]]] = {}
    features: Dict[str, Dict[str, np.ndarray]] = {}
    run_meta = []

    for label, run_dir in run_dirs.items():
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        manifest_path = run_dir / "sample_manifest.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing sample_manifest.csv: {manifest_path}")
        manifests[label] = read_manifest(manifest_path)
        signatures[label] = pairing_signature(manifests[label])
        features[label], drop_prefix, shapes = precompute_piece_features(metrics_mod, run_dir, manifests[label])
        run_meta.append(
            {
                "label": label,
                "run_dir": str(run_dir),
                "n_samples": len(signatures[label]),
                "drop_prefix": drop_prefix,
                "original_shapes": ";".join(map(str, shapes)),
            }
        )

    baseline_sig = signatures[args.baseline]
    n = len(baseline_sig)
    if n < 2:
        raise ValueError("Need at least two matched continuations")
    for label, sig in signatures.items():
        if sig != baseline_sig:
            raise ValueError(
                f"Pairing mismatch for {label}. run_index/piece_idx/sample_rng_seed must exactly match baseline."
            )

    # Observed corpus metrics using all matched continuations exactly once.
    observed_weights = np.ones((1, n), dtype=np.int16)
    observed: Dict[str, Dict[str, float]] = {}
    for label in run_dirs:
        vals = aggregate_metrics(observed_weights, features[label], real_stats)
        observed[label] = {k: float(vals[k][0]) for k in METRIC_KEYS}

    # Paired bootstrap: one set of resampled continuation indices shared across all methods.
    rng = np.random.default_rng(args.seed)
    sampled_indices = rng.integers(0, n, size=(args.n_boot, n))
    weights = np.zeros((args.n_boot, n), dtype=np.int16)
    for b in range(args.n_boot):
        weights[b] = np.bincount(sampled_indices[b], minlength=n)

    bootstrap: Dict[str, Dict[str, np.ndarray]] = {}
    for label in run_dirs:
        bootstrap[label] = aggregate_metrics(weights, features[label], real_stats)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairing_rows = [
        {"run_index": a, "piece_idx": b, "sample_rng_seed": c}
        for a, b, c in baseline_sig
    ]
    write_csv(out_dir / "paired_manifest.csv", pairing_rows)
    write_csv(out_dir / "run_metadata.csv", run_meta)

    observed_rows = []
    ci_rows = []
    for label in run_dirs:
        for metric in METRIC_KEYS:
            observed_rows.append(
                {"method": label, "metric": metric, "observed": f"{observed[label][metric]:.9f}"}
            )
            lo, hi = quantile_ci(bootstrap[label][metric], args.ci)
            ci_rows.append(
                {
                    "method": label,
                    "metric": metric,
                    "observed": f"{observed[label][metric]:.9f}",
                    "bootstrap_mean": f"{float(np.mean(bootstrap[label][metric])):.9f}",
                    "ci_low": f"{lo:.9f}",
                    "ci_high": f"{hi:.9f}",
                }
            )
    write_csv(out_dir / "observed_metrics.csv", observed_rows)
    write_csv(out_dir / "bootstrap_metric_ci.csv", ci_rows)

    delta_rows = []
    base = args.baseline
    for label in run_dirs:
        if label == base:
            continue
        for metric in METRIC_KEYS:
            # All reported metrics are lower-is-better.
            # delta > 0 => method metric is higher => baseline is better.
            delta = bootstrap[label][metric] - bootstrap[base][metric]
            observed_delta = observed[label][metric] - observed[base][metric]
            lo, hi = quantile_ci(delta, args.ci)
            delta_rows.append(
                {
                    "baseline": base,
                    "method": label,
                    "metric": metric,
                    "observed_delta_method_minus_baseline": f"{observed_delta:.9f}",
                    "bootstrap_mean_delta": f"{float(np.mean(delta)):.9f}",
                    "ci_low": f"{lo:.9f}",
                    "ci_high": f"{hi:.9f}",
                    "ci_excludes_zero": str(bool(lo > 0.0 or hi < 0.0)),
                    "direction_if_ci_excludes_zero": (
                        "baseline_better" if lo > 0.0 else ("method_better" if hi < 0.0 else "uncertain")
                    ),
                    "bootstrap_fraction_baseline_better": f"{float(np.mean(delta > 0.0)):.6f}",
                }
            )
    write_csv(out_dir / "paired_delta_vs_baseline.csv", delta_rows)

    summary = [
        "# Paired bootstrap: final objective test",
        "",
        f"- Matched continuations: {n}",
        f"- Bootstrap replicates: {args.n_boot}",
        f"- RNG seed: {args.seed}",
        f"- Percentile CI: {args.ci * 100:.1f}%",
        f"- Baseline: `{args.baseline}`",
        f"- Real reference: full `{args.real_split}` split, kept fixed",
        "- Resampling unit: matched generated continuation index",
        "- The same resampled indices are used for every method in each bootstrap replicate.",
        "- Delta convention: `method - baseline`; because all Table 1 metrics are lower-is-better, positive delta favors the baseline.",
        "- `bootstrap_fraction_baseline_better` is descriptive, not a p-value.",
        "",
        "## Pairwise differences vs baseline",
        "",
        "| Method | Metric | Observed delta | CI low | CI high | Interpretation |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in delta_rows:
        if row["ci_excludes_zero"] == "True":
            interp = row["direction_if_ci_excludes_zero"]
        else:
            interp = "CI crosses 0"
        summary.append(
            f"| {row['method']} | {row['metric']} | "
            f"{float(row['observed_delta_method_minus_baseline']):+.6f} | "
            f"{float(row['ci_low']):+.6f} | {float(row['ci_high']):+.6f} | {interp} |"
        )
    (out_dir / "bootstrap_summary.md").write_text("\n".join(summary), encoding="utf-8")

    config = {
        "data": args.data,
        "real_split": args.real_split,
        "run_specs": {k: str(v) for k, v in run_dirs.items()},
        "baseline": args.baseline,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "ci": args.ci,
        "metrics_py": args.metrics_py,
        "output_dir": str(out_dir),
        "n_matched_continuations": n,
        "real_reference_fixed": True,
    }
    (out_dir / "bootstrap_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Pairing check: PASS ({n} matched continuations)")
    print(f"Saved: {out_dir / 'observed_metrics.csv'}")
    print(f"Saved: {out_dir / 'bootstrap_metric_ci.csv'}")
    print(f"Saved: {out_dir / 'paired_delta_vs_baseline.csv'}")
    print(f"Saved: {out_dir / 'bootstrap_summary.md'}")


if __name__ == "__main__":
    main()
