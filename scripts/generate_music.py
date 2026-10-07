from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from two_voice_game.generation import (  # noqa: E402
    DEVICE,
    generate_independent,
    generate_nash_game,
    generate_no_game_joint,
    generate_no_game_joint_global,
    generate_qre_game,
    generate_stackelberg_game,
    load_generator,
    save_generation_outputs,
    select_heldout_seeds,
)


def parse_seed_indices(text: str | None) -> Optional[List[int]]:
    if text is None or text.strip() == "":
        return None
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def safe_float_tag(value: float) -> str:
    """Convert a float into a filename-safe short tag, e.g. 1.25 -> 1p25."""
    text = f"{value:g}"
    text = text.replace("-", "m").replace(".", "p")
    return text


def safe_name(text: str) -> str:
    """Make a compact filesystem-safe name component."""
    text = text.strip()
    text = re.sub(r"[^A-Za-z0-9_.=-]+", "-", text)
    return text.strip("-") or "run"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate two-voice symbolic music from a trained utility model."
    )

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, default="data/processed/jsb_two_voice_tokens.npz")
    parser.add_argument("--split", type=str, default="valid", choices=["train", "valid", "test"])

    parser.add_argument(
        "--method",
        type=str,
        default="qre_game",
        choices=[
            "independent",
            "no_game_joint",
            "no_game_joint_global",
            "qre_game",
            "nash_game",
            "stackelberg_game",
        ],
        help=(
            "Generation decision mode: independent=no interaction; "
            "no_game_joint=fair grid joint baseline over A_U x A_L; "
            "no_game_joint_global=original global pair-head top-k baseline; "
            "qre_game=QRE-style game solver over the same A_U x A_L grid; "
            "nash_game=pure Nash equilibrium solver over the same A_U x A_L grid; "
            "stackelberg_game=leader/follower game solver over the same A_U x A_L grid."
        ),
    )

    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument(
        "--seed_indices",
        type=str,
        default=None,
        help="Comma-separated piece indices, e.g. '0,3,9'. If omitted, seeds are sampled from the split.",
    )
    parser.add_argument("--rng_seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=128, help="Continuation length in sixteenth-note steps.")

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=12)

    parser.add_argument("--alpha", type=float, default=1.0, help="Upper/soprano private utility weight.")
    parser.add_argument("--beta_private", type=float, default=1.0, help="Lower/bass private utility weight.")
    parser.add_argument("--qre_beta", type=float, default=1.0, help="QRE rationality parameter.")
    parser.add_argument("--qre_iterations", type=int, default=20)
    parser.add_argument(
        "--compat_top_pairs",
        type=int,
        default=None,
        help=(
            "Optional QRE compatibility filter. For qre_game, keep only the top-N "
            "candidate pairs by shared compatibility C before sampling from the QRE "
            "joint policy. Example: --compat_top_pairs 40 when top_k=12 keeps the "
            "best 40 pairs out of 144."
        ),
    )
    parser.add_argument(
        "--nash_tie_break",
        type=str,
        default="welfare",
        choices=["welfare", "compatibility", "upper", "lower"],
        help=(
            "For nash_game: how to choose when multiple pure Nash equilibria exist. "
            "welfare maximizes payoff_upper + payoff_lower."
        ),
    )
    parser.add_argument(
        "--nash_fallback",
        type=str,
        default="welfare",
        choices=["welfare", "compatibility", "upper", "lower"],
        help=(
            "For nash_game: deterministic fallback when no pure Nash equilibrium exists."
        ),
    )
    parser.add_argument(
        "--nash_atol",
        type=float,
        default=1e-9,
        help="For nash_game: tolerance used when detecting best responses.",
    )
    parser.add_argument(
        "--stackelberg_leader",
        type=str,
        default="lower",
        choices=["lower", "upper"],
        help="For stackelberg_game: which voice commits first. Use 'lower' for bass-leading Stackelberg.",
    )
    parser.add_argument(
        "--stackelberg_mode",
        type=str,
        default="deterministic",
        choices=["deterministic", "soft"],
        help="For stackelberg_game: deterministic argmax leader/follower or soft Stackelberg sampling.",
    )
    parser.add_argument(
        "--stackelberg_beta",
        type=float,
        default=1.0,
        help="For stackelberg_game with --stackelberg_mode soft: rationality parameter for leader/follower softmax policies.",
    )

    parser.add_argument("--tempo_bpm", type=int, default=90)
    parser.add_argument("--no_musicxml", action="store_true")

    # New run-management arguments.
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/generated",
        help="Root directory where timestamped generation run folders are created.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional human-readable suffix for the run folder.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing run folder with the same name. Normally not needed.",
    )

    # Backward compatibility: accept old --output_dir, but use it as output_root.
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Deprecated alias for --output_root. Prefer --output_root.",
    )

    return parser.parse_args()


def make_run_id(args: argparse.Namespace, context_len: int) -> str:
    """Create a short run folder name.

    Keep the folder name short to avoid Windows path-length problems.
    Detailed parameters such as context length, split, seed, steps, top-k,
    alpha, beta_private, and qre_beta are stored in run_config.json instead
    of being duplicated in the folder name.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parts = [timestamp, args.method]

    if args.run_name:
        parts.append(safe_name(args.run_name))

    return "__".join(parts)


def sample_file_tag(method: str) -> str:
    """Return a readable but compact sample filename prefix."""
    aliases = {
        "qre_game": "qre",
        "nash_game": "nash",
        "stackelberg_game": "stackelberg",
        "no_game_joint": "no_game_joint",
        "no_game_joint_global": "no_game_joint_global",
        "independent": "independent",
    }
    return aliases.get(method, safe_name(method))


def jsonable(value: Any) -> Any:
    """Convert common non-JSON objects into JSON-friendly values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [jsonable(x) for x in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    return value


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(jsonable(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def build_run_config(
    args: argparse.Namespace,
    context_len: int,
    run_id: str,
    run_dir: Path,
    seed_info: List[Tuple[int, Any]],
    generator_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    selected_piece_indices = [int(piece_idx) for piece_idx, _ in seed_info]
    generator_metadata = generator_metadata or {}

    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": DEVICE,
        "checkpoint": args.checkpoint,
        "data": args.data,
        "split": args.split,
        "method": args.method,
        "context_len": context_len,
        "duration_aware_state": bool(generator_metadata.get("duration_aware_state", False)),
        "duration_feature_dim": int(generator_metadata.get("duration_feature_dim", 0)),
        "duration_feature_names": list(generator_metadata.get("duration_feature_names", []) or []),
        "duration_feature_normalizer": generator_metadata.get("duration_feature_normalizer", None),
        "num_seeds_requested": args.num_seeds,
        "num_seeds_used": len(seed_info),
        "seed_indices_argument": args.seed_indices,
        "selected_piece_indices": selected_piece_indices,
        "rng_seed": args.rng_seed,
        "steps": args.steps,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "alpha": args.alpha,
        "beta_private": args.beta_private,
        "qre_beta": args.qre_beta,
        "qre_iterations": args.qre_iterations,
        "compat_top_pairs": args.compat_top_pairs,
        "nash_tie_break": args.nash_tie_break,
        "nash_fallback": args.nash_fallback,
        "nash_atol": args.nash_atol,
        "stackelberg_leader": args.stackelberg_leader,
        "stackelberg_mode": args.stackelberg_mode,
        "stackelberg_beta": args.stackelberg_beta,
        "tempo_bpm": args.tempo_bpm,
        "write_musicxml": not args.no_musicxml,
        "output_root": args.output_root if args.output_dir is None else args.output_dir,
        "run_dir": str(run_dir),
        "samples_dir": str(run_dir / "samples"),
        "notes": {
            "independent": "No-interaction baseline: soprano and bass are sampled independently from private heads.",
            "no_game_joint": "Main fair baseline: builds the same A_U x A_L candidate grid as QRE, then samples using only the pair compatibility head.",
            "no_game_joint_global": "Strong supplementary baseline: directly samples from the global pair-head top-k without using the QRE candidate grid.",
            "qre_game": "Interaction with game solver: uses the same A_U x A_L candidate grid, solves a QRE-style iterative logit-response game, and can optionally apply a compatibility filter with --compat_top_pairs.",
            "nash_game": "Pure-equilibrium ablation: uses the same A_U x A_L candidate grid, chooses a pure Nash equilibrium when available, and reports fallback/regret when not.",
            "stackelberg_game": "Role-structured game solver: one voice leads and the other best-responds or soft-responds over the same A_U x A_L candidate grid.",
        },
    }


def write_aggregate_metrics(path: Path, all_metrics: List[Dict[str, float]]) -> None:
    if not all_metrics:
        return

    metric_keys = sorted(k for k in all_metrics[0].keys() if k != "piece_idx")

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean"])
        for key in metric_keys:
            values = [float(m[key]) for m in all_metrics if key in m]
            mean_value = sum(values) / len(values) if values else 0.0
            writer.writerow([key, f"{mean_value:.6f}"])


def write_sample_manifest(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames = [
        "run_id",
        "method",
        "run_index",
        "piece_idx",
        "sample_rng_seed",
        "npy_path",
        "txt_path",
        "metrics_path",
        "musicxml_path",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_generation_summary(path: Path, run_config: Dict[str, Any], all_metrics: List[Dict[str, float]]) -> None:
    lines = [
        "# Generation Run Summary",
        "",
        f"- Run ID: `{run_config['run_id']}`",
        f"- Method: `{run_config['method']}`",
        f"- Split: `{run_config['split']}`",
        f"- Context length: `{run_config['context_len']}`",
        f"- Duration-aware state: `{run_config.get('duration_aware_state', False)}`",
        f"- Generated continuation steps: `{run_config['steps']}`",
        f"- Number of seeds used: `{run_config['num_seeds_used']}`",
        f"- RNG seed: `{run_config['rng_seed']}`",
        f"- Temperature: `{run_config['temperature']}`",
        f"- Top-k: `{run_config['top_k']}`",
    ]

    if run_config.get("duration_aware_state", False):
        lines.extend(
            [
                f"- Duration feature dim: `{run_config.get('duration_feature_dim', 0)}`",
                f"- Duration feature normalizer: `{run_config.get('duration_feature_normalizer')}`",
                f"- Duration features: `{', '.join(run_config.get('duration_feature_names', []))}`",
            ]
        )

    if run_config["method"] == "qre_game":
        lines.extend(
            [
                f"- QRE beta: `{run_config['qre_beta']}`",
                f"- Alpha: `{run_config['alpha']}`",
                f"- Beta private: `{run_config['beta_private']}`",
                f"- QRE iterations: `{run_config['qre_iterations']}`",
                f"- Compatibility top pairs: `{run_config['compat_top_pairs']}`",
            ]
        )

    if run_config["method"] == "nash_game":
        lines.extend(
            [
                f"- Nash tie-break: `{run_config['nash_tie_break']}`",
                f"- Nash fallback: `{run_config['nash_fallback']}`",
                f"- Nash tolerance: `{run_config['nash_atol']}`",
                f"- Alpha: `{run_config['alpha']}`",
                f"- Beta private: `{run_config['beta_private']}`",
            ]
        )

    if run_config["method"] == "stackelberg_game":
        lines.extend(
            [
                f"- Stackelberg leader: `{run_config['stackelberg_leader']}`",
                f"- Stackelberg mode: `{run_config['stackelberg_mode']}`",
                f"- Stackelberg beta: `{run_config['stackelberg_beta']}`",
                f"- Alpha: `{run_config['alpha']}`",
                f"- Beta private: `{run_config['beta_private']}`",
            ]
        )

    lines.extend([
        "",
        "## Files",
        "",
        "- `run_config.json`: full generation configuration and selected seeds.",
        "- `sample_manifest.csv`: generated sample file paths.",
        "- `aggregate_metrics.csv`: mean internal metrics across samples.",
        "- `samples/`: generated `.npy`, `.txt`, `.metrics.txt`, and optional `.musicxml` files.",
    ])

    if all_metrics:
        metric_keys = sorted(k for k in all_metrics[0].keys() if k != "piece_idx")
        lines.extend(["", "## Aggregate internal metrics", "", "| Metric | Mean |", "|---|---:|"])
        for key in metric_keys:
            values = [float(m[key]) for m in all_metrics if key in m]
            mean_value = sum(values) / len(values) if values else 0.0
            lines.append(f"| {key} | {mean_value:.6f} |")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    if args.output_dir is not None:
        args.output_root = args.output_dir
        print("[Warning] --output_dir is deprecated. It is treated as --output_root.")

    gen = load_generator(args.checkpoint, device=DEVICE)

    seeds = select_heldout_seeds(
        data_path=args.data,
        split=args.split,
        context_len=gen.context_len,
        num_seeds=args.num_seeds,
        seed_indices=parse_seed_indices(args.seed_indices),
        rng_seed=args.rng_seed,
    )

    output_root = Path(args.output_root)
    run_id = make_run_id(args, gen.context_len)
    run_dir = output_root / run_id
    samples_dir = run_dir / "samples"

    if run_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Run directory already exists: {run_dir}\n"
            "Use --run_name to create a different run or --overwrite if you intentionally want to reuse it."
        )

    samples_dir.mkdir(parents=True, exist_ok=True)

    run_config = build_run_config(
        args=args,
        context_len=gen.context_len,
        run_id=run_id,
        run_dir=run_dir,
        seed_info=seeds,
        generator_metadata={
            "duration_aware_state": gen.duration_aware_state,
            "duration_feature_dim": gen.duration_feature_dim,
            "duration_feature_names": gen.duration_feature_names or [],
            "duration_feature_normalizer": gen.duration_feature_normalizer,
        },
    )
    write_json(run_dir / "run_config.json", run_config)

    print(f"Using device: {DEVICE}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Context length: {gen.context_len}")
    print(f"Duration-aware state: {gen.duration_aware_state}")
    if gen.duration_aware_state:
        print(f"Duration feature dim: {gen.duration_feature_dim}")
        print(f"Duration features: {', '.join(gen.duration_feature_names or [])}")
    print(f"Method: {args.method}")
    print(f"Seeds: {len(seeds)}")
    print(f"Continuation steps: {args.steps}")
    print(f"Run ID: {run_id}")
    print(f"Run dir: {run_dir}")

    all_metrics: List[Dict[str, float]] = []
    manifest_rows: List[Dict[str, Any]] = []

    for run_index, (piece_idx, seed_raw) in enumerate(seeds):
        sample_rng_seed = args.rng_seed + run_index

        if args.method == "independent":
            raw_tokens, metrics = generate_independent(
                gen=gen,
                seed_raw=seed_raw,
                num_steps=args.steps,
                temperature=args.temperature,
                top_k=args.top_k,
                rng_seed=sample_rng_seed,
                device=DEVICE,
            )
        elif args.method == "no_game_joint":
            raw_tokens, metrics = generate_no_game_joint(
                gen=gen,
                seed_raw=seed_raw,
                num_steps=args.steps,
                temperature=args.temperature,
                top_k=args.top_k,
                rng_seed=sample_rng_seed,
                device=DEVICE,
            )
        elif args.method == "no_game_joint_global":
            raw_tokens, metrics = generate_no_game_joint_global(
                gen=gen,
                seed_raw=seed_raw,
                num_steps=args.steps,
                temperature=args.temperature,
                top_k=args.top_k,
                rng_seed=sample_rng_seed,
                device=DEVICE,
            )
        elif args.method == "qre_game":
            raw_tokens, metrics = generate_qre_game(
                gen=gen,
                seed_raw=seed_raw,
                num_steps=args.steps,
                alpha=args.alpha,
                beta_private=args.beta_private,
                qre_beta=args.qre_beta,
                iterations=args.qre_iterations,
                temperature=args.temperature,
                top_k=args.top_k,
                compat_top_pairs=args.compat_top_pairs,
                rng_seed=sample_rng_seed,
                device=DEVICE,
            )
        elif args.method == "nash_game":
            raw_tokens, metrics = generate_nash_game(
                gen=gen,
                seed_raw=seed_raw,
                num_steps=args.steps,
                alpha=args.alpha,
                beta_private=args.beta_private,
                top_k=args.top_k,
                tie_break=args.nash_tie_break,
                fallback=args.nash_fallback,
                atol=args.nash_atol,
                rng_seed=sample_rng_seed,
                device=DEVICE,
            )
        elif args.method == "stackelberg_game":
            raw_tokens, metrics = generate_stackelberg_game(
                gen=gen,
                seed_raw=seed_raw,
                num_steps=args.steps,
                alpha=args.alpha,
                beta_private=args.beta_private,
                leader=args.stackelberg_leader,
                mode=args.stackelberg_mode,
                stackelberg_beta=args.stackelberg_beta,
                temperature=args.temperature,
                top_k=args.top_k,
                rng_seed=sample_rng_seed,
                device=DEVICE,
            )
        else:
            raise ValueError(f"Unsupported method: {args.method}")

        metrics = {"piece_idx": float(piece_idx), **metrics}
        all_metrics.append(metrics)

        file_tag = sample_file_tag(args.method)
        prefix = samples_dir / f"{file_tag}_seed{run_index:03d}_piece{piece_idx:03d}"
        save_generation_outputs(
            raw_tokens=raw_tokens,
            output_prefix=prefix,
            metrics=metrics,
            write_musicxml_file=not args.no_musicxml,
            tempo_bpm=args.tempo_bpm,
        )

        manifest_rows.append(
            {
                "run_id": run_id,
                "method": args.method,
                "run_index": run_index,
                "piece_idx": int(piece_idx),
                "sample_rng_seed": sample_rng_seed,
                "npy_path": str(prefix.with_suffix(".npy")),
                "txt_path": str(prefix.with_suffix(".txt")),
                "metrics_path": str(prefix.with_suffix(".metrics.txt")),
                "musicxml_path": str(prefix.with_suffix(".musicxml")) if not args.no_musicxml else "",
            }
        )

        print(f"Saved sample {run_index + 1:03d}/{len(seeds):03d}: {prefix}")

    write_aggregate_metrics(run_dir / "aggregate_metrics.csv", all_metrics)
    write_sample_manifest(run_dir / "sample_manifest.csv", manifest_rows)
    write_generation_summary(run_dir / "generation_summary.md", run_config, all_metrics)

    print(f"Saved run config: {run_dir / 'run_config.json'}")
    print(f"Saved aggregate metrics: {run_dir / 'aggregate_metrics.csv'}")
    print(f"Saved sample manifest: {run_dir / 'sample_manifest.csv'}")
    print(f"Saved generation summary: {run_dir / 'generation_summary.md'}")
    print("Done.")


if __name__ == "__main__":
    main()
