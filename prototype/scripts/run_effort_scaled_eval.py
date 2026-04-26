from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATASETS = [
    ("UADFV", "prototype/manifests/uadfv_metadata_train128_v0_1.csv"),
    ("CelebDF-v1", "prototype/manifests/celebdfv1_metadata_train128_v0_1.csv"),
    ("FF++", "prototype/manifests/ffpp_metadata_train512_v0_1.csv"),
    ("Celeb-DF-v2", "prototype/manifests/celebdfv2_metadata_eval512_v0_1.csv"),
    ("DFDCP", "prototype/manifests/dfdcp_metadata_eval512_v0_1.csv"),
    ("DF40-deepfacelab", "prototype/manifests/df40_deepfacelab_eval49_v0_1.csv"),
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return -1
    with path.open("r", newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def manifest_row_count(path: Path) -> int:
    return count_csv_rows(path)


def stem(path: Path) -> str:
    return path.stem


def script(root: Path, name: str) -> str:
    return str(root / "prototype" / "scripts" / name)


def rel(path: Path) -> str:
    return str(path).replace("\\", "/")


def run_command(command: list[str], cwd: Path, log_path: Path, dry_run: bool) -> dict[str, Any]:
    printable = " ".join(command)
    record: dict[str, Any] = {"command": command, "started_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")}
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n\n=== {record['started_at']} RUN {printable}\n")
        log.flush()
        if dry_run:
            record["returncode"] = None
            record["dry_run"] = True
            return record
        completed = subprocess.run(command, cwd=cwd, text=True, stdout=log, stderr=subprocess.STDOUT)
    record["finished_at"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    record["returncode"] = completed.returncode
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with {completed.returncode}: {printable}")
    return record


def output_paths(root: Path, manifest: Path, suffix: str) -> dict[str, Path]:
    name = stem(manifest)
    reports = root / "prototype" / "reports"
    return {
        "provenance": reports / f"{name}_provenance_results_{suffix}.csv",
        "passive": reports / f"{name}_passive_results_effort_{suffix}.csv",
        "merged": reports / f"{name}_merged_effort_{suffix}.csv",
        "routing": reports / f"{name}_routing_effort_{suffix}_balanced_security.csv",
        "routing_summary": reports / f"{name}_routing_effort_{suffix}_balanced_security_summary.json",
    }


def maybe_run_csv_step(
    expected_rows: int,
    output_csv: Path,
    command: list[str],
    cwd: Path,
    log_path: Path,
    dry_run: bool,
    force: bool,
) -> dict[str, Any]:
    existing_rows = count_csv_rows(output_csv)
    if not force and existing_rows == expected_rows:
        return {"status": "skipped_existing", "path": str(output_csv), "rows": existing_rows}
    result = run_command(command, cwd, log_path, dry_run)
    result["path"] = str(output_csv)
    result["rows"] = count_csv_rows(output_csv) if not dry_run else None
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scaled Effort-style hybrid evaluation over project manifests.")
    parser.add_argument("--suffix", default="scaled_2026-04-18")
    parser.add_argument("--checkpoint", default="pilot/passive_baseline/runs/effort_clipl_multisource_rank4_v1/best.pt")
    parser.add_argument("--cache-dir", default="workspace/checkpoints/huggingface")
    parser.add_argument("--config", default="prototype/configs/uadfv_smoke_v0_1.yaml")
    parser.add_argument("--paths", default="prototype/configs/paths.local.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.44)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    reports = root / "prototype" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    log_path = reports / f"effort_{args.suffix}_pipeline.log"
    summary_path = reports / f"effort_{args.suffix}_pipeline_summary.json"
    rules_path = reports / f"route_label_rules_effort_{args.suffix}_balanced_security.yaml"
    search_path = reports / f"routing_search_results_effort_{args.suffix}_balanced_security.tsv"
    lodo_csv = reports / f"single_effort_lodo_{args.suffix}_balanced_security.csv"
    lodo_summary = reports / f"single_effort_lodo_{args.suffix}_balanced_security_summary.json"

    summary: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "suffix": args.suffix,
        "checkpoint": args.checkpoint,
        "datasets": {},
        "commands": [],
    }

    try:
        bridge_args: list[str] = []
        lodo_args: list[str] = []
        for dataset_name, manifest_text in DATASETS:
            manifest = root / manifest_text
            expected_rows = manifest_row_count(manifest)
            if expected_rows <= 0:
                raise FileNotFoundError(f"Manifest has no rows or is missing: {manifest}")
            paths = output_paths(root, manifest, args.suffix)
            dataset_summary: dict[str, Any] = {"manifest": str(manifest), "expected_rows": expected_rows}

            provenance_command = [
                sys.executable,
                script(root, "execute_videoseal_probe.py"),
                "--config",
                args.config,
                "--manifest",
                rel(manifest),
                "--paths",
                args.paths,
                "--output",
                rel(paths["provenance"]),
                "--device",
                args.device,
            ]
            provenance_result = maybe_run_csv_step(
                expected_rows, paths["provenance"], provenance_command, root, log_path, args.dry_run, args.force
            )
            dataset_summary["provenance"] = provenance_result

            passive_command = [
                sys.executable,
                script(root, "execute_effort_manifest_probe.py"),
                "--config",
                args.config,
                "--manifest",
                rel(manifest),
                "--output",
                rel(paths["passive"]),
                "--checkpoint",
                args.checkpoint,
                "--cache-dir",
                args.cache_dir,
                "--device",
                args.device,
                "--batch-size",
                str(args.batch_size),
                "--threshold",
                str(args.threshold),
            ]
            passive_result = maybe_run_csv_step(
                expected_rows, paths["passive"], passive_command, root, log_path, args.dry_run, args.force
            )
            dataset_summary["passive"] = passive_result

            merge_command = [
                sys.executable,
                script(root, "merge_smoke_outputs.py"),
                "--config",
                args.config,
                "--manifest",
                rel(manifest),
                "--provenance-csv",
                rel(paths["provenance"]),
                "--passive-csv",
                rel(paths["passive"]),
                "--output",
                rel(paths["merged"]),
            ]
            merge_result = maybe_run_csv_step(
                expected_rows, paths["merged"], merge_command, root, log_path, args.dry_run, True
            )
            dataset_summary["merged"] = merge_result

            summary["datasets"][dataset_name] = dataset_summary
            bridge_args.extend(["--bridge-csv", rel(paths["merged"])])
            lodo_args.extend(["--bridge", f"{dataset_name}={rel(paths['merged'])}"])

        search_command = [
            sys.executable,
            script(root, "search_routing_thresholds_constrained.py"),
            *bridge_args,
            "--results-tsv",
            rel(search_path),
            "--best-rules-yaml",
            rel(rules_path),
            "--max-protected-real-fpr",
            "0.25",
            "--min-protected-fake-recall",
            "0.60",
            "--min-unprotected-fake-recall",
            "0.60",
        ]
        summary["commands"].append(run_command(search_command, root, log_path, args.dry_run))

        lodo_command = [
            sys.executable,
            script(root, "evaluate_single_passive_lodo.py"),
            *lodo_args,
            "--output-csv",
            rel(lodo_csv),
            "--summary-json",
            rel(lodo_summary),
            "--max-protected-real-fpr",
            "0.25",
            "--min-protected-fake-recall",
            "0.60",
            "--min-unprotected-fake-recall",
            "0.60",
        ]
        summary["commands"].append(run_command(lodo_command, root, log_path, args.dry_run))

        for dataset_name, manifest_text in DATASETS:
            manifest = root / manifest_text
            paths = output_paths(root, manifest, args.suffix)
            route_command = [
                sys.executable,
                script(root, "run_routing_sanity.py"),
                "--bridge-csv",
                rel(paths["merged"]),
                "--rules",
                rel(rules_path),
                "--output-csv",
                rel(paths["routing"]),
                "--summary-json",
                rel(paths["routing_summary"]),
                "--backbone-name",
                "VideoSeal",
            ]
            summary["commands"].append(run_command(route_command, root, log_path, args.dry_run))

        summary["status"] = "ok"
        summary["rules"] = str(rules_path)
        summary["search_results"] = str(search_path)
        summary["lodo_csv"] = str(lodo_csv)
        summary["lodo_summary"] = str(lodo_summary)
    except Exception as exc:
        summary["status"] = "failed"
        summary["error_type"] = type(exc).__name__
        summary["error"] = str(exc)
        raise
    finally:
        summary["finished_at"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"status": summary["status"], "summary": str(summary_path), "log": str(log_path)}, indent=2))


if __name__ == "__main__":
    main()
