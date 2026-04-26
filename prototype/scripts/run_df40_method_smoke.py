from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def method_slug(method: str) -> str:
    return method.strip().replace(" ", "_").replace("-", "_").lower()


def script_path(root: Path, name: str) -> Path:
    return root / "prototype" / "scripts" / name


def config_path(root: Path, method: str) -> Path:
    return root / "prototype" / "configs" / f"df40_{method_slug(method)}_smoke_v0_1.yaml"


def experiment_name(method: str) -> str:
    return f"df40_{method_slug(method)}_smoke_v0_1"


def default_runner_python(root: Path) -> Path:
    candidate = root / ".conda" / "passive-pilot" / "python.exe"
    return candidate if candidate.exists() else Path(sys.executable)


def count_dirs(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for child in path.iterdir() if child.is_dir())


def write_config(root: Path, method: str, max_real: int, max_fake: int, force: bool) -> Path:
    cfg_path = config_path(root, method)
    if cfg_path.exists() and not force:
        return cfg_path

    slug = method_slug(method)
    exp = experiment_name(method)
    data = {
        "experiment_name": exp,
        "paths": {
            "manifest_path": f"prototype/manifests/{exp}.csv",
            "provenance_jobs_path": f"prototype/reports/{exp}_provenance_jobs.csv",
            "passive_jobs_path": f"prototype/reports/{exp}_passive_jobs.csv",
            "merged_report_path": f"prototype/reports/{exp}_merged.csv",
            "route_rules_path": "prototype/configs/route_label_rules.yaml",
        },
        "dataset": {
            "name": f"DF40-{method}",
            "real_root": f"workspace/datasets/raw/df40/{method}/real/frames",
            "fake_root": f"workspace/datasets/raw/df40/{method}/fake/frames",
            "extensions": [".png"],
            "max_real_videos": max_real,
            "max_fake_videos": max_fake,
            "frame_selector": "middle",
            "split_ratios": {"train": 0.5, "val": 0.25, "test": 0.25},
        },
        "derivations": {
            "include_unprotected_clean": True,
            "protected_variants": [
                {"name": "clean", "degradation_type": "none", "degradation_level": "clean"},
                {"name": "jpeg_q75", "degradation_type": "jpeg", "degradation_level": "q75"},
                {"name": "resize_half", "degradation_type": "resize", "degradation_level": "0.5x"},
            ],
        },
        "outputs": {
            "derived_root": f"workspace/datasets/derived/df40_{slug}_protected/v0_1",
            "provenance_feature_root": f"workspace/features/provenance/{exp}",
            "passive_feature_root": f"workspace/features/passive/{exp}",
        },
    }
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return cfg_path


def run_command(command: list[str], cwd: Path, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {"command": command, "returncode": None, "dry_run": True}
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    result = {
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }
    if completed.returncode != 0:
        raise RuntimeError(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a full DF40 method smoke bridge from method frames to routing summary.")
    parser.add_argument("--method", required=True)
    parser.add_argument("--max-real-videos", type=int, default=16)
    parser.add_argument("--max-fake-videos", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--runner-python", default="")
    parser.add_argument("--rules", default="prototype/reports/route_label_rules_autosearch.yaml")
    parser.add_argument("--paths", default="prototype/configs/paths.local.yaml")
    parser.add_argument("--force-config", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Only write config, manifest, and job sheets.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = project_root()
    method = args.method
    exp = experiment_name(method)
    cfg = write_config(root, method, args.max_real_videos, args.max_fake_videos, args.force_config)

    real_root = root / "workspace" / "datasets" / "raw" / "df40" / method / "real" / "frames"
    fake_root = root / "workspace" / "datasets" / "raw" / "df40" / method / "fake" / "frames"
    if count_dirs(real_root) == 0 or count_dirs(fake_root) == 0:
        raise SystemExit(f"Missing DF40 method frames. real={real_root}, fake={fake_root}")

    runner = Path(args.runner_python) if args.runner_python else default_runner_python(root)
    if not runner.is_absolute():
        runner = root / runner

    commands: list[list[str]] = [
        [sys.executable, str(script_path(root, "build_video_dir_smoke_manifest.py")), "--config", str(cfg)],
        [sys.executable, str(script_path(root, "run_videoseal_probe.py")), "--config", str(cfg), "--paths", args.paths],
        [sys.executable, str(script_path(root, "run_passive_probe.py")), "--config", str(cfg), "--paths", args.paths],
    ]

    if not args.prepare_only:
        commands.extend(
            [
                [str(runner), str(script_path(root, "execute_videoseal_probe.py")), "--config", str(cfg), "--paths", args.paths, "--device", args.device],
                [str(runner), str(script_path(root, "execute_passive_pilot_probe.py")), "--config", str(cfg), "--device", args.device],
                [
                    sys.executable,
                    str(script_path(root, "merge_smoke_outputs.py")),
                    "--config",
                    str(cfg),
                    "--provenance-csv",
                    f"prototype/reports/{exp}_provenance_results.csv",
                    "--passive-csv",
                    f"prototype/reports/{exp}_passive_results.csv",
                    "--output",
                    f"prototype/reports/{exp}_merged.csv",
                ],
                [
                    sys.executable,
                    str(script_path(root, "run_routing_sanity.py")),
                    "--bridge-csv",
                    f"prototype/reports/{exp}_merged.csv",
                    "--rules",
                    args.rules,
                    "--output-csv",
                    f"prototype/reports/{exp}_routing_sanity.csv",
                    "--summary-json",
                    f"prototype/reports/{exp}_routing_sanity_summary.json",
                    "--backbone-name",
                    "VideoSeal",
                ],
            ]
        )

    results = []
    status = "ok"
    try:
        for command in commands:
            print("RUN", " ".join(command))
            results.append(run_command(command, root, args.dry_run))
    except Exception as exc:
        status = "failed"
        results.append({"error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        report = {
            "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "status": status,
            "method": method,
            "experiment_name": exp,
            "config": str(cfg),
            "prepare_only": args.prepare_only,
            "commands": results,
        }
        report_path = root / "prototype" / "reports" / f"{exp}_run_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"report": str(report_path), "status": status, "experiment_name": exp}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
