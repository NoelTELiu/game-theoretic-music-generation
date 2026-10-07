from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def safe_float_tag(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def timestamp_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass
class SweepJob:
    label: str
    group: str
    method: str
    params: Dict[str, object]


def build_jobs(include_independent: bool = False) -> List[SweepJob]:
    """
    Duration-aware validation sweep.

    Baseline:
      - no_game_joint

    QRE:
      - qre_beta in {0.25, 0.5, 1.0}
      - compat_top_pairs in {20, 30, 40}
      - alpha = beta_private = 0.5

    Stackelberg:
      - leader in {upper, lower}
      - stackelberg_beta in {0.5, 0.75, 1.0}
      - mode = soft
      - alpha = beta_private = 0.5

    Nash:
      - alpha = beta_private in {0.0, 0.5, 1.0}
      - tie_break = fallback = welfare
    """
    jobs: List[SweepJob] = []

    if include_independent:
        jobs.append(
            SweepJob(
                label="independent",
                group="baseline",
                method="independent",
                params={},
            )
        )

    jobs.append(
        SweepJob(
            label="no_game_joint",
            group="baseline",
            method="no_game_joint",
            params={},
        )
    )

    for qre_beta in [0.25, 0.5, 1.0]:
        for compat_top_pairs in [20, 30, 40]:
            jobs.append(
                SweepJob(
                    label=f"qre_b{safe_float_tag(qre_beta)}_cf{compat_top_pairs}",
                    group="qre",
                    method="qre_game",
                    params={
                        "alpha": 0.5,
                        "beta_private": 0.5,
                        "qre_beta": qre_beta,
                        "qre_iterations": 20,
                        "compat_top_pairs": compat_top_pairs,
                    },
                )
            )

    for leader in ["upper", "lower"]:
        for beta in [0.5, 0.75, 1.0]:
            jobs.append(
                SweepJob(
                    label=f"stk_{leader}_soft_b{safe_float_tag(beta)}",
                    group="stackelberg",
                    method="stackelberg_game",
                    params={
                        "alpha": 0.5,
                        "beta_private": 0.5,
                        "stackelberg_leader": leader,
                        "stackelberg_mode": "soft",
                        "stackelberg_beta": beta,
                    },
                )
            )

    for alpha in [0.0, 0.5, 1.0]:
        jobs.append(
            SweepJob(
                label=f"nash_a{safe_float_tag(alpha)}",
                group="nash",
                method="nash_game",
                params={
                    "alpha": alpha,
                    "beta_private": alpha,
                    "nash_tie_break": "welfare",
                    "nash_fallback": "welfare",
                    "nash_atol": 1e-9,
                },
            )
        )

    return jobs


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(x) for x in value]
    return value


def write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def write_manifest(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "status",
        "label",
        "group",
        "method",
        "run_name",
        "run_dir",
        "log_path",
        "return_code",
        "params_json",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def run_and_tee(cmd: List[str], log_path: Path, cwd: Path) -> int:
    """
    Run a subprocess while streaming output to both terminal and a log file.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8", newline="") as log_f:
        log_f.write("$ " + " ".join(cmd) + "\n\n")
        log_f.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_f.write(line)
            log_f.flush()

        proc.wait()
        log_f.write(f"\n[return_code] {proc.returncode}\n")
        return int(proc.returncode)


def find_generated_run_dir(output_root: Path, method: str, run_name: str) -> Optional[Path]:
    """
    generate_music.py creates:
        <timestamp>__<method>__<run_name>

    This function finds the newest matching directory.
    """
    if not output_root.exists():
        return None

    candidates = [
        p
        for p in output_root.iterdir()
        if p.is_dir()
        and p.name.endswith(f"__{method}__{run_name}")
    ]

    if not candidates:
        # Fallback: tolerate small filename differences.
        candidates = [
            p
            for p in output_root.iterdir()
            if p.is_dir() and method in p.name and run_name in p.name
        ]

    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)


def find_latest_eval_dir(eval_output_root: Path, eval_name: str) -> Optional[Path]:
    if not eval_output_root.exists():
        return None

    candidates = [
        p
        for p in eval_output_root.iterdir()
        if p.is_dir() and eval_name in p.name
    ]

    if not candidates:
        candidates = [p for p in eval_output_root.iterdir() if p.is_dir()]

    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)


def build_generate_command(args: argparse.Namespace, job: SweepJob, run_name: str) -> List[str]:
    cmd = [
        sys.executable,
        "scripts/generate_music.py",
        "--checkpoint",
        args.checkpoint,
        "--data",
        args.data,
        "--split",
        args.split,
        "--method",
        job.method,
        "--num_seeds",
        str(args.num_seeds),
        "--steps",
        str(args.steps),
        "--top_k",
        str(args.top_k),
        "--temperature",
        str(args.temperature),
        "--rng_seed",
        str(args.rng_seed),
        "--run_name",
        run_name,
        "--output_root",
        str(args.generated_root),
    ]

    if args.no_musicxml:
        cmd.append("--no_musicxml")

    for key, value in job.params.items():
        cmd.extend([f"--{key}", str(value)])

    return cmd


def write_sweep_summary(
    path: Path,
    args: argparse.Namespace,
    sweep_id: str,
    manifest_rows: List[Dict[str, object]],
    eval_dir: Optional[Path],
) -> None:
    completed = [r for r in manifest_rows if r.get("status") == "completed"]
    failed = [r for r in manifest_rows if r.get("status") == "failed"]

    lines = [
        "# Duration-Aware Validation Sweep",
        "",
        f"- Sweep ID: `{sweep_id}`",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Data: `{args.data}`",
        f"- Split: `{args.split}`",
        f"- Num seeds: `{args.num_seeds}`",
        f"- Steps: `{args.steps}`",
        f"- Top-k: `{args.top_k}`",
        f"- Temperature: `{args.temperature}`",
        f"- RNG seed: `{args.rng_seed}`",
        f"- Generated root: `{args.generated_root}`",
        f"- Evaluation root: `{args.eval_root}`",
        f"- Completed jobs: `{len(completed)}`",
        f"- Failed jobs: `{len(failed)}`",
        "",
        "## Evaluation",
        "",
    ]

    if eval_dir is not None:
        lines.extend(
            [
                f"- Eval dir: `{eval_dir}`",
                f"- Main summary: `{eval_dir / 'summary.md'}`",
                f"- Corpus comparison CSV: `{eval_dir / 'comparison_to_real.csv'}`",
                f"- Diagnostic metrics CSV: `{eval_dir / 'scalar_metrics.csv'}`",
                "",
            ]
        )
    else:
        lines.append("- Evaluation did not run or no evaluation directory was detected.")
        lines.append("")

    lines.extend(
        [
            "## Runs",
            "",
            "| Status | Label | Group | Method | Run dir | Log |",
            "|---|---|---|---|---|---|",
        ]
    )

    for row in manifest_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("status", "")),
                    f"`{row.get('label', '')}`",
                    str(row.get("group", "")),
                    str(row.get("method", "")),
                    f"`{row.get('run_dir', '')}`",
                    f"`{row.get('log_path', '')}`",
                ]
            )
            + " |"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full duration-aware validation parameter sweep for "
            "no_game_joint, QRE, Stackelberg, and Nash, then evaluate all runs."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/utility_ctx16_duration_aware/best.pt",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/jsb_two_voice_tokens.npz",
    )
    parser.add_argument("--split", type=str, default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--num_seeds", type=int, default=30)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--top_k", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--rng_seed", type=int, default=42)

    parser.add_argument(
        "--sweep_name",
        type=str,
        default="dur_valid_param_sweep",
        help="Human-readable sweep name used in output folders.",
    )
    parser.add_argument(
        "--generated_root",
        type=str,
        default=None,
        help=(
            "Where generation run folders are placed. "
            "Default: outputs/generated/<sweep_id>"
        ),
    )
    parser.add_argument(
        "--eval_root",
        type=str,
        default=None,
        help=(
            "Where evaluation folder is placed. "
            "Default: outputs/evaluation/<sweep_id>"
        ),
    )
    parser.add_argument(
        "--logs_root",
        type=str,
        default=None,
        help=(
            "Where sweep logs/manifests are placed. "
            "Default: outputs/sweeps/<sweep_id>"
        ),
    )

    parser.add_argument(
        "--include_independent",
        action="store_true",
        help="Also run independent baseline. Default is off because it is not a game solver.",
    )
    parser.add_argument(
        "--no_musicxml",
        action="store_true",
        default=True,
        help="Disable MusicXML export. Default: enabled as a flag here, so MusicXML is not generated.",
    )
    parser.add_argument(
        "--write_musicxml",
        action="store_true",
        help="Override default and write MusicXML files.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue remaining jobs if one generation command fails.",
    )
    parser.add_argument(
        "--skip_evaluation",
        action="store_true",
        help="Only generate samples; do not run evaluate_generation.py.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.write_musicxml:
        args.no_musicxml = False

    project_root = Path.cwd()

    sweep_id = f"{timestamp_now()}__{args.sweep_name}"

    if args.generated_root is None:
        args.generated_root = str(Path("outputs/generated") / sweep_id)
    if args.eval_root is None:
        args.eval_root = str(Path("outputs/evaluation") / sweep_id)
    if args.logs_root is None:
        args.logs_root = str(Path("outputs/sweeps") / sweep_id)

    args.generated_root = Path(args.generated_root)
    args.eval_root = Path(args.eval_root)
    args.logs_root = Path(args.logs_root)

    logs_dir = args.logs_root / "logs"
    manifest_path = args.logs_root / "sweep_manifest.csv"
    config_path = args.logs_root / "sweep_config.json"
    summary_path = args.logs_root / "sweep_summary.md"

    args.generated_root.mkdir(parents=True, exist_ok=True)
    args.eval_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(include_independent=args.include_independent)

    write_json(
        config_path,
        {
            "sweep_id": sweep_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "checkpoint": args.checkpoint,
            "data": args.data,
            "split": args.split,
            "num_seeds": args.num_seeds,
            "steps": args.steps,
            "top_k": args.top_k,
            "temperature": args.temperature,
            "rng_seed": args.rng_seed,
            "generated_root": args.generated_root,
            "eval_root": args.eval_root,
            "logs_root": args.logs_root,
            "include_independent": args.include_independent,
            "no_musicxml": args.no_musicxml,
            "jobs": [asdict(job) for job in jobs],
        },
    )

    print("=" * 80)
    print("Duration-aware validation sweep")
    print("=" * 80)
    print(f"Sweep ID:       {sweep_id}")
    print(f"Checkpoint:     {args.checkpoint}")
    print(f"Generated root: {args.generated_root}")
    print(f"Eval root:      {args.eval_root}")
    print(f"Logs root:      {args.logs_root}")
    print(f"Jobs:           {len(jobs)}")
    print("=" * 80)

    manifest_rows: List[Dict[str, object]] = []

    for idx, job in enumerate(jobs, start=1):
        run_name = f"{args.sweep_name}__{job.label}"
        log_path = logs_dir / f"{idx:02d}__{job.label}.log"

        print()
        print("=" * 80)
        print(f"[{idx:02d}/{len(jobs):02d}] {job.label}")
        print("=" * 80)

        cmd = build_generate_command(args, job, run_name=run_name)
        return_code = run_and_tee(cmd=cmd, log_path=log_path, cwd=project_root)

        run_dir = find_generated_run_dir(args.generated_root, job.method, run_name)
        status = "completed" if return_code == 0 and run_dir is not None else "failed"

        row = {
            "status": status,
            "label": job.label,
            "group": job.group,
            "method": job.method,
            "run_name": run_name,
            "run_dir": str(run_dir) if run_dir is not None else "",
            "log_path": str(log_path),
            "return_code": return_code,
            "params_json": json.dumps(job.params, ensure_ascii=False),
        }
        manifest_rows.append(row)
        write_manifest(manifest_path, manifest_rows)

        if status != "completed":
            print(f"[FAILED] {job.label}. See log: {log_path}")
            if not args.continue_on_error:
                write_sweep_summary(summary_path, args, sweep_id, manifest_rows, eval_dir=None)
                raise SystemExit(return_code if return_code != 0 else 1)

    eval_dir: Optional[Path] = None

    if not args.skip_evaluation:
        completed_rows = [r for r in manifest_rows if r.get("status") == "completed"]
        if completed_rows:
            eval_name = args.sweep_name
            eval_log_path = logs_dir / "evaluation.log"

            eval_cmd = [
                sys.executable,
                "scripts/evaluate_generation.py",
                "--data",
                args.data,
                "--real_split",
                args.split,
                "--output_root",
                str(args.eval_root),
                "--eval_name",
                eval_name,
                "--run_dirs",
            ]

            for row in completed_rows:
                label = str(row["label"])
                run_dir = str(row["run_dir"])
                eval_cmd.append(f"{label}={run_dir}")

            print()
            print("=" * 80)
            print("Running evaluation for completed sweep jobs")
            print("=" * 80)

            eval_return_code = run_and_tee(cmd=eval_cmd, log_path=eval_log_path, cwd=project_root)

            if eval_return_code != 0:
                print(f"[FAILED] Evaluation failed. See log: {eval_log_path}")
                if not args.continue_on_error:
                    write_sweep_summary(summary_path, args, sweep_id, manifest_rows, eval_dir=None)
                    raise SystemExit(eval_return_code)

            eval_dir = find_latest_eval_dir(args.eval_root, eval_name)

    write_sweep_summary(summary_path, args, sweep_id, manifest_rows, eval_dir=eval_dir)

    print()
    print("=" * 80)
    print("Sweep complete")
    print("=" * 80)
    print(f"Sweep config:   {config_path}")
    print(f"Manifest:       {manifest_path}")
    print(f"Summary:        {summary_path}")
    if eval_dir is not None:
        print(f"Evaluation dir: {eval_dir}")
        print(f"Eval summary:   {eval_dir / 'summary.md'}")
        print(f"Comparison CSV: {eval_dir / 'comparison_to_real.csv'}")
        print(f"Scalars CSV:    {eval_dir / 'scalar_metrics.csv'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
