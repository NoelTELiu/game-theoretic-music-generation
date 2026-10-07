from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from two_voice_game.evaluation.metrics import (  # noqa: E402
    collect_corpus_stats,
    comparison_to_real,
)


METHODS_DEFAULT = ["independent", "no_game_joint", "qre_game"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated two-voice music against a held-out real split."
    )

    parser.add_argument("--data", type=str, default="data/processed/jsb_two_voice_tokens.npz")
    parser.add_argument("--real_split", type=str, default="test", choices=["train", "valid", "test"])

    # New recommended interface.
    parser.add_argument(
        "--run_dirs",
        nargs="*",
        default=None,
        help=(
            "Generation run directories to evaluate. Use either plain paths or method=path, e.g. "
            "independent=outputs/generated/runs/<run1> no_game_joint=outputs/generated/runs/<run2>."
        ),
    )

    # Legacy interface retained for older outputs: generated_root/method/*.npy.
    parser.add_argument(
        "--generated_root",
        type=str,
        default=None,
        help="Legacy generated root containing method subfolders. Prefer --run_dirs for new run-managed outputs.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=METHODS_DEFAULT,
        help="Methods to evaluate when using --generated_root. Also used to validate method names.",
    )

    parser.add_argument(
        "--drop_prefix",
        type=int,
        default=None,
        help=(
            "Number of initial seed steps to drop from generated samples before evaluation. "
            "If omitted for run-managed folders, it is read from run_config.json context_len."
        ),
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/evaluation",
        help="Root directory where timestamped evaluation folders are created.",
    )
    parser.add_argument("--eval_name", type=str, default=None, help="Optional human-readable suffix for eval folder.")
    parser.add_argument("--overwrite", action="store_true")

    # Backward compatibility.
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Deprecated alias for a fixed output directory. Prefer --output_root.",
    )

    return parser.parse_args()


def safe_name(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[^A-Za-z0-9_.=-]+", "-", text)
    return text.strip("-") or "eval"


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


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_real_pieces(data_path: Path, split: str) -> List[np.ndarray]:
    data = np.load(data_path, allow_pickle=True)
    key = f"{split}_tokens"
    if key not in data:
        raise KeyError(f"Could not find {key} in {data_path}. Available keys: {list(data.keys())}")
    return [np.asarray(piece, dtype=np.int64)[:, :2] for piece in data[key]]


def infer_method_from_run_dir(run_dir: Path) -> str:
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        cfg = load_json(config_path)
        if "method" in cfg:
            return str(cfg["method"])
    # Fallback: try to infer from folder name.
    name = run_dir.name
    for method in METHODS_DEFAULT:
        if method in name:
            return method
    return run_dir.name


def infer_drop_prefix_from_run_dir(run_dir: Path, user_value: Optional[int]) -> Tuple[int, str]:
    """
    Return the number of generated-sample prefix steps to drop and where the value came from.

    For run-managed generation folders, generated .npy samples contain:
        [real seed prefix of length context_len] + [generated continuation]

    Therefore, when --drop_prefix is omitted, we read context_len from run_config.json
    and drop exactly that many initial seed steps.
    """
    if user_value is not None:
        return int(user_value), "argument"

    config_path = run_dir / "run_config.json"
    if config_path.exists():
        cfg = load_json(config_path)
        if "context_len" in cfg:
            return int(cfg["context_len"]), "run_config.context_len"

    return 0, "default_0"


def parse_run_dir_spec(spec: str) -> Tuple[Optional[str], Path]:
    if "=" in spec:
        method, path_text = spec.split("=", 1)
        return method.strip(), Path(path_text.strip())
    return None, Path(spec)


def find_sample_files_in_run(run_dir: Path) -> List[Path]:
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


def load_token_array(path: Path) -> np.ndarray:
    """Load a generated token sample from either .npy or human-readable .txt."""
    if path.suffix == ".npy":
        arr = np.load(path, allow_pickle=True)
    elif path.suffix == ".txt" and not path.name.endswith(".metrics.txt"):
        arr = np.loadtxt(path, dtype=np.int64)
    else:
        raise ValueError(f"Unsupported generated sample file: {path}")

    arr = np.asarray(arr, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def load_generated_samples(files: Iterable[Path], drop_prefix: int) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    samples: List[np.ndarray] = []
    file_records: List[Dict[str, Any]] = []

    for path in files:
        arr = load_token_array(path)

        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"Expected generated sample shape [T, 2], got {arr.shape} in {path}")

        original_len = int(len(arr))
        if drop_prefix > 0:
            arr = arr[drop_prefix:]

        kept_len = int(len(arr))
        loaded = kept_len > 0
        if loaded:
            samples.append(arr)

        file_records.append(
            {
                "path": str(path),
                "original_len": original_len,
                "drop_prefix": int(drop_prefix),
                "kept_len": kept_len,
                "loaded": loaded,
            }
        )

    return samples, file_records


def collect_run_managed_outputs(args: argparse.Namespace) -> Tuple[Dict[str, List[np.ndarray]], List[Dict[str, Any]]]:
    if not args.run_dirs:
        return {}, []

    method_to_samples: Dict[str, List[np.ndarray]] = {}
    run_records: List[Dict[str, Any]] = []

    for spec in args.run_dirs:
        explicit_method, run_dir = parse_run_dir_spec(spec)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        method = explicit_method or infer_method_from_run_dir(run_dir)
        run_config_path = run_dir / "run_config.json"
        run_cfg = load_json(run_config_path) if run_config_path.exists() else {}
        drop_prefix, drop_prefix_source = infer_drop_prefix_from_run_dir(run_dir, args.drop_prefix)
        files = find_sample_files_in_run(run_dir)
        samples, file_records = load_generated_samples(files, drop_prefix=drop_prefix)

        if not samples:
            raise ValueError(
                f"No usable .npy or .txt samples found for method {method} in {run_dir}. "
                f"Found {len(files)} files, drop_prefix={drop_prefix}. "
                "Check that generated samples contain more timesteps than the seed prefix."
            )

        method_to_samples.setdefault(method, []).extend(samples)

        min_original_len = min((r["original_len"] for r in file_records), default=0)
        min_kept_len = min((r["kept_len"] for r in file_records), default=0)

        run_records.append(
            {
                "method": method,
                "run_dir": str(run_dir),
                "num_files": len(files),
                "num_samples_loaded": len(samples),
                "drop_prefix": drop_prefix,
                "drop_prefix_source": drop_prefix_source,
                "min_original_len": min_original_len,
                "min_kept_len": min_kept_len,
                "run_config": str(run_config_path) if run_config_path.exists() else "",
                "duration_aware_state": run_cfg.get("duration_aware_state", ""),
                "duration_feature_dim": run_cfg.get("duration_feature_dim", ""),
                "duration_feature_names": ",".join(run_cfg.get("duration_feature_names", []) or []),
            }
        )

    return method_to_samples, run_records


def collect_legacy_outputs(args: argparse.Namespace) -> Tuple[Dict[str, List[np.ndarray]], List[Dict[str, Any]]]:
    if args.generated_root is None:
        return {}, []

    root = Path(args.generated_root)
    if not root.exists():
        raise FileNotFoundError(f"generated_root not found: {root}")

    method_to_samples: Dict[str, List[np.ndarray]] = {}
    run_records: List[Dict[str, Any]] = []
    drop_prefix = int(args.drop_prefix or 0)

    for method in args.methods:
        method_dir = root / method
        if not method_dir.exists():
            print(f"[Warning] Missing method directory: {method_dir}")
            continue
        files = sorted(method_dir.glob("*.npy"))
        if not files:
            files = sorted(
                p
                for p in method_dir.glob("*.txt")
                if not p.name.endswith(".metrics.txt")
            )
        samples, file_records = load_generated_samples(files, drop_prefix=drop_prefix)
        if samples:
            method_to_samples[method] = samples

        min_original_len = min((r["original_len"] for r in file_records), default=0)
        min_kept_len = min((r["kept_len"] for r in file_records), default=0)

        run_records.append(
            {
                "method": method,
                "run_dir": str(method_dir),
                "num_files": len(files),
                "num_samples_loaded": len(samples),
                "drop_prefix": drop_prefix,
                "drop_prefix_source": "argument" if args.drop_prefix is not None else "default_0_legacy",
                "min_original_len": min_original_len,
                "min_kept_len": min_kept_len,
                "run_config": "",
            }
        )

    return method_to_samples, run_records


def make_eval_id(args: argparse.Namespace) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [timestamp, "eval", f"real-{args.real_split}"]
    if args.eval_name:
        parts.append(safe_name(args.eval_name))
    return "__".join(parts)


def write_scalar_metrics_csv(path: Path, rows: Dict[str, Dict[str, float]]) -> None:
    if not rows:
        return
    all_keys = sorted({k for metrics in rows.values() for k in metrics.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", *all_keys])
        for method, metrics in rows.items():
            writer.writerow([method, *[f"{float(metrics.get(k, 0.0)):.6f}" for k in all_keys]])


def write_comparison_csv(path: Path, rows: Dict[str, Dict[str, float]]) -> None:
    if not rows:
        return
    all_keys = sorted({k for metrics in rows.values() for k in metrics.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", *all_keys])
        for method, metrics in rows.items():
            writer.writerow([method, *[f"{float(metrics.get(k, 0.0)):.6f}" for k in all_keys]])


def write_run_records_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = [
        "method",
        "run_dir",
        "num_files",
        "num_samples_loaded",
        "drop_prefix",
        "drop_prefix_source",
        "min_original_len",
        "min_kept_len",
        "run_config",
        "duration_aware_state",
        "duration_feature_dim",
        "duration_feature_names",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def format_metric_table(methods: List[str], comparison: Dict[str, Dict[str, float]]) -> List[str]:
    metric_keys = sorted({k for m in comparison.values() for k in m.keys()})
    lines = ["| Metric | " + " | ".join(methods) + " | Best |", "|---|" + "---:|" * len(methods) + "---|"]
    for key in metric_keys:
        values = {method: float(comparison.get(method, {}).get(key, 0.0)) for method in methods}
        best_method = min(values, key=values.get) if values else ""
        row = [key] + [f"{values[method]:.6f}" for method in methods] + [best_method]
        lines.append("| " + " | ".join(row) + " |")
    return lines


def format_scalar_table(methods_with_real: List[str], scalars: Dict[str, Dict[str, float]]) -> List[str]:
    important_keys = [
        "hold_ratio",
        "voice_crossing_rate",
        "dissonance_rate",
        "parallel_perfect_rate",
        "excessive_leap_rate",
        "pitch_entropy",
        "unique_3gram_ratio",
        "mean_voice_distance",
    ]
    lines = ["| Metric | " + " | ".join(methods_with_real) + " |", "|---|" + "---:|" * len(methods_with_real)]
    for key in important_keys:
        row = [key]
        for method in methods_with_real:
            value = scalars.get(method, {}).get(key, 0.0)
            row.append(f"{float(value):.6f}")
        lines.append("| " + " | ".join(row) + " |")
    return lines


def write_summary_md(
    path: Path,
    eval_config: Dict[str, Any],
    scalar_metrics: Dict[str, Dict[str, float]],
    comparison: Dict[str, Dict[str, float]],
) -> None:
    real_name = eval_config["real_name"]
    methods = [m for m in scalar_metrics.keys() if m != real_name]
    methods_with_real = [real_name, *methods]

    lines = [
        "# Two-Voice Generation Evaluation Summary",
        "",
        "This report compares generated outputs against the held-out real corpus using external metrics.",
        "Lower corpus-similarity distances/gaps indicate that the generated distribution is closer to real music.",
        "",
        "## Evaluation run",
        "",
        f"- Eval ID: `{eval_config['eval_id']}`",
        f"- Real split: `{eval_config['real_split']}`",
        f"- Real reference name: `{real_name}`",
        f"- Data: `{eval_config['data']}`",
        f"- Output directory: `{eval_config['eval_dir']}`",
        "",
        "## Methods / runs",
        "",
    ]

    for record in eval_config["run_records"]:
        duration_note = ""
        if record.get("duration_aware_state", "") != "":
            duration_note = (
                f", duration_aware_state={record.get('duration_aware_state')}, "
                f"duration_feature_dim={record.get('duration_feature_dim')}"
            )
        lines.append(
            f"- `{record['method']}`: `{record['run_dir']}` "
            f"({record['num_samples_loaded']} samples, "
            f"drop_prefix={record['drop_prefix']} from {record['drop_prefix_source']}, "
            f"min kept length={record['min_kept_len']}"
            f"{duration_note})"
        )

    lines.extend([
        "",
        "## Corpus-similarity metrics",
        "",
        *format_metric_table(methods, comparison),
        "",
        "## Diagnostic and diversity metrics",
        "",
        *format_scalar_table(methods_with_real, scalar_metrics),
        "",
        "## Suggested interpretation",
        "",
        "- If `no_game_joint` improves over `independent`, then modeling upper-lower interaction helps.",
        "- If `qre_game` improves over `no_game_joint`, then the game solver adds value beyond ordinary joint prediction.",
        "- Internal game metrics such as regret and payoff should be reported separately from these external music-quality metrics.",
        "- Generated continuation metrics should exclude the real seed prefix. Check `evaluated_runs.csv` for the applied `drop_prefix`.",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    real_pieces = load_real_pieces(Path(args.data), args.real_split)
    real_name = f"real_{args.real_split}"
    real_stats = collect_corpus_stats(real_name, real_pieces)
    real_metrics = real_stats.scalar_metrics()

    run_managed_samples, run_records = collect_run_managed_outputs(args)
    legacy_samples, legacy_records = collect_legacy_outputs(args)

    method_to_samples = {**legacy_samples}
    for method, samples in run_managed_samples.items():
        method_to_samples.setdefault(method, []).extend(samples)

    all_records = legacy_records + run_records

    if not method_to_samples:
        raise ValueError("No generated samples found. Provide --run_dirs or --generated_root.")

    if args.output_dir is not None:
        eval_dir = Path(args.output_dir)
        eval_id = eval_dir.name
    else:
        eval_id = make_eval_id(args)
        eval_dir = Path(args.output_root) / eval_id

    if eval_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Evaluation directory already exists: {eval_dir}\n"
            "Use --eval_name to create a different eval or --overwrite if this is intentional."
        )
    eval_dir.mkdir(parents=True, exist_ok=True)

    scalar_metrics: Dict[str, Dict[str, float]] = {real_name: real_metrics}
    comparison: Dict[str, Dict[str, float]] = {}

    for method, samples in sorted(method_to_samples.items()):
        method_stats = collect_corpus_stats(method, samples)
        scalar_metrics[method] = method_stats.scalar_metrics()
        comparison[method] = comparison_to_real(real_stats, method_stats)

    eval_config = {
        "eval_id": eval_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data": args.data,
        "real_split": args.real_split,
        "real_name": real_name,
        "drop_prefix_argument": args.drop_prefix,
        "generated_root": args.generated_root,
        "run_dirs_argument": args.run_dirs,
        "methods_argument": args.methods,
        "eval_dir": str(eval_dir),
        "run_records": all_records,
    }

    write_json(eval_dir / "eval_config.json", eval_config)
    write_run_records_csv(eval_dir / "evaluated_runs.csv", all_records)
    write_scalar_metrics_csv(eval_dir / "per_method_scalar_metrics.csv", scalar_metrics)
    write_comparison_csv(eval_dir / "comparison_to_real.csv", comparison)
    write_summary_md(eval_dir / "evaluation_summary.md", eval_config, scalar_metrics, comparison)

    print(f"Saved eval config: {eval_dir / 'eval_config.json'}")
    print(f"Saved evaluated runs: {eval_dir / 'evaluated_runs.csv'}")
    print(f"Saved scalar metrics: {eval_dir / 'per_method_scalar_metrics.csv'}")
    print(f"Saved comparison: {eval_dir / 'comparison_to_real.csv'}")
    print(f"Saved summary: {eval_dir / 'evaluation_summary.md'}")
    print("Done.")


if __name__ == "__main__":
    main()
