from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_PAIRED_METHODS = ["simswap", "inswap"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def script_path(root: Path, name: str) -> Path:
    return root / "prototype" / "scripts" / name


def runner_python(root: Path) -> Path:
    candidate = root / ".conda" / "passive-pilot" / "python.exe"
    return candidate if candidate.exists() else Path(sys.executable)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def find_arg(command: list[str], flag: str) -> str | None:
    for index, item in enumerate(command):
        if item == flag and index + 1 < len(command):
            return command[index + 1]
    return None


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return -1
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def run_command(command: list[str], cwd: Path, dry_run: bool) -> dict[str, Any]:
    record: dict[str, Any] = {
        "command": command,
        "started_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }
    if dry_run:
        record["returncode"] = None
        record["dry_run"] = True
        return record
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    record["finished_at"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    record["returncode"] = completed.returncode
    record["stdout_tail"] = completed.stdout[-4000:]
    record["stderr_tail"] = completed.stderr[-4000:]
    if completed.returncode != 0:
        raise RuntimeError(json.dumps(record, indent=2, ensure_ascii=False))
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay one df40aux8 rerun checkpoint through the canonical six-dataset bridge, "
            "the held-out FF-paired simswap/inswap bridge, and the fixed literal routing policy."
        )
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--pipeline-summary",
        default="prototype/reports/effort_sfrank_df40aux8_2026-04-19_pipeline_summary.json",
    )
    parser.add_argument(
        "--rules-yaml",
        default="prototype/reports/route_label_rules_sfrank_df40aux8_transfertree_literal_fpr08.yaml",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--paths", default="prototype/configs/paths.local.yaml")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=64)
    parser.add_argument("--paired-method", action="append", dest="paired_methods")
    parser.add_argument("--paired-profile-alias", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    reports = root / "prototype" / "reports"
    runner = runner_python(root)

    summary = load_json(resolve_path(root, args.pipeline_summary))
    suffix = summary["suffix"]
    checkpoint = root / "pilot" / "passive_baseline" / "runs" / args.run_name / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    rules_yaml = resolve_path(root, args.rules_yaml)
    if not rules_yaml.exists():
        raise FileNotFoundError(f"Missing rules yaml: {rules_yaml}")

    first_dataset = next(iter(summary["datasets"].values()))
    passive_template = first_dataset["passive"]["command"]
    config_path = find_arg(passive_template, "--config") or "prototype/configs/uadfv_smoke_v0_1.yaml"
    cache_dir = find_arg(passive_template, "--cache-dir") or "workspace/checkpoints/huggingface"

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "run_name": args.run_name,
        "tag": args.tag,
        "checkpoint": str(checkpoint),
        "threshold": args.threshold,
        "rules_yaml": str(rules_yaml),
        "canonical": [],
        "paired": [],
        "commands": [],
    }

    canonical_bridge_csvs: list[Path] = []
    for dataset_name, dataset_summary in summary["datasets"].items():
        manifest = Path(dataset_summary["manifest"])
        manifest_stem = manifest.stem
        provenance_csv = reports / f"{manifest_stem}_provenance_results_{suffix}.csv"
        passive_csv = reports / f"{manifest_stem}_passive_results_effort_{args.tag}.csv"
        merged_csv = reports / f"{manifest_stem}_merged_effort_{args.tag}.csv"
        if not provenance_csv.exists():
            raise FileNotFoundError(f"Missing provenance csv for {dataset_name}: {provenance_csv}")

        if args.force or count_csv_rows(passive_csv) <= 0:
            command = [
                str(runner),
                str(script_path(root, "execute_effort_manifest_probe.py")),
                "--config",
                config_path,
                "--manifest",
                str(manifest),
                "--output",
                str(passive_csv),
                "--checkpoint",
                str(checkpoint),
                "--cache-dir",
                cache_dir,
                "--device",
                args.device,
                "--batch-size",
                str(args.batch_size),
                "--threshold",
                str(args.threshold),
            ]
            report["commands"].append(run_command(command, root, args.dry_run))

        if args.force or count_csv_rows(merged_csv) <= 0:
            command = [
                sys.executable,
                str(script_path(root, "merge_smoke_outputs.py")),
                "--config",
                config_path,
                "--manifest",
                str(manifest),
                "--provenance-csv",
                str(provenance_csv),
                "--passive-csv",
                str(passive_csv),
                "--output",
                str(merged_csv),
            ]
            report["commands"].append(run_command(command, root, args.dry_run))

        canonical_bridge_csvs.append(merged_csv)
        report["canonical"].append(
            {
                "dataset": dataset_name,
                "manifest": str(manifest),
                "provenance_csv": str(provenance_csv),
                "passive_csv": str(passive_csv),
                "merged_csv": str(merged_csv),
            }
        )

    paired_methods = args.paired_methods or DEFAULT_PAIRED_METHODS
    paired_bridge_csvs: list[Path] = []
    profile_alias = args.paired_profile_alias or f"aux8_{args.tag}"
    for method in paired_methods:
        command = [
            sys.executable,
            str(script_path(root, "run_df40_ffpp_paired_probe.py")),
            "--method",
            method,
            "--profile",
            "aux8",
            "--profile-alias",
            profile_alias,
            "--checkpoint-override",
            str(checkpoint),
            "--rules-override",
            str(rules_yaml),
            "--ffpp-split",
            "test",
            "--max-pairs",
            str(args.max_pairs),
            "--device",
            args.device,
            "--paths",
            args.paths,
        ]
        if args.force:
            command.append("--force-config")
        experiment_name = f"df40_{method}_ffpp_test_paired_{profile_alias}_v0_1"
        merged_csv = reports / f"{experiment_name}_merged.csv"
        if args.force or count_csv_rows(merged_csv) <= 0:
            report["commands"].append(run_command(command, root, args.dry_run))
        paired_bridge_csvs.append(merged_csv)
        report["paired"].append({"method": method, "merged_csv": str(merged_csv)})

    output_csv = reports / f"single_passive_{args.tag}_lodo.csv"
    summary_json = reports / f"single_passive_{args.tag}_lodo_summary.json"
    eval_command = [
        sys.executable,
        str(script_path(root, "evaluate_transfer_rules_fixed_lodo.py")),
        "--rules-yaml",
        str(rules_yaml),
        "--output-csv",
        str(output_csv),
        "--summary-json",
        str(summary_json),
        "--tag",
        args.tag,
    ]
    for path in canonical_bridge_csvs:
        eval_command.extend(["--canonical-bridge-csv", str(path)])
    for path in paired_bridge_csvs:
        eval_command.extend(["--paired-bridge-csv", str(path)])
    report["commands"].append(run_command(eval_command, root, args.dry_run))

    report["output_csv"] = str(output_csv)
    report["summary_json"] = str(summary_json)
    report_path = reports / f"{args.tag}_replay_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "summary_json": str(summary_json), "status": "ok"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
