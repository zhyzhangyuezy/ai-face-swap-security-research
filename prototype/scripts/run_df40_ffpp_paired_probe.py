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


def script_path(root: Path, name: str) -> Path:
    return root / "prototype" / "scripts" / name


PROFILE_SETTINGS = {
    "sfrg05": {
        "checkpoint": "pilot/passive_baseline/runs/effort_clipl_rank4_patchattn_mean_sfrank_realguard_w10_m15_g05_e5_v1/best.pt",
        "rules": "prototype/reports/route_label_rules_sfrg05_fpr08.yaml",
        "label": "sfrg05",
    },
    "aux8": {
        "checkpoint": "pilot/passive_baseline/runs/effort_clipl_rank4_patchattn_mean_sfrank_df40aux8_w10_m15_e5_v1/best.pt",
        "rules": "prototype/reports/route_label_rules_sfrank_df40aux8_fpr08.yaml",
        "label": "sfrank_df40aux8",
    },
}


def runner_python(root: Path) -> Path:
    candidate = root / ".conda" / "passive-pilot" / "python.exe"
    return candidate if candidate.exists() else Path(sys.executable)


def config_path(root: Path, experiment_name: str) -> Path:
    return root / "prototype" / "configs" / f"{experiment_name}.yaml"


def write_config(
    root: Path,
    method: str,
    profile_label: str,
    ffpp_split: str,
    max_pairs: int,
    force: bool,
    fake_root_override: str,
    experiment_tag: str,
) -> tuple[str, Path]:
    split_suffix = ffpp_split if ffpp_split else "all"
    tag_suffix = f"_{experiment_tag}" if experiment_tag else ""
    experiment_name = f"df40_{method}{tag_suffix}_ffpp_{split_suffix}_paired_{profile_label}_v0_1"
    cfg_path = config_path(root, experiment_name)
    if cfg_path.exists() and not force:
        return experiment_name, cfg_path

    pair_split_json = (
        str((root / "workspace" / "datasets" / "raw" / "FaceForensics++" / f"{ffpp_split}.json").resolve())
        if ffpp_split
        else ""
    )
    fake_root = (
        Path(fake_root_override).resolve()
        if str(fake_root_override).strip()
        else (root / "workspace" / "datasets" / "raw" / "df40" / method / "frames").resolve()
    )
    data = {
        "experiment_name": experiment_name,
        "paths": {
            "manifest_path": f"prototype/manifests/{experiment_name}.csv",
            "provenance_jobs_path": f"prototype/reports/{experiment_name}_provenance_jobs.csv",
            "passive_jobs_path": f"prototype/reports/{experiment_name}_passive_jobs.csv",
            "merged_report_path": f"prototype/reports/{experiment_name}_merged.csv",
        },
        "dataset": {
            "name": f"DF40-{method}{tag_suffix}-FFPP-paired-{split_suffix}",
            "method": method,
            "ffpp_real_root": str(
                (root / "workspace" / "datasets" / "raw" / "FaceForensics++" / "original_sequences" / "youtube" / "c23" / "frames").resolve()
            ),
            "fake_root": str(fake_root),
            "ffpp_pair_split_name": split_suffix,
            "ffpp_pair_split_json": pair_split_json,
            "extensions": [".png", ".jpg", ".jpeg"],
            "max_pairs": max_pairs,
            "source_token_index": 0,
            "real_frame_selector": "middle",
            "fake_frame_selector": "middle",
            "split_ratios": {"train": 0.5, "val": 0.25, "test": 0.25},
        },
        "derivations": {
            "protected_variants": [
                {"name": "clean", "degradation_type": "none", "degradation_level": "clean"},
                {"name": "jpeg_q75", "degradation_type": "jpeg", "degradation_level": "q75"},
                {"name": "resize_half", "degradation_type": "resize", "degradation_level": "0.5x"},
            ],
        },
        "outputs": {
            "derived_root": f"workspace/datasets/derived/{experiment_name}",
            "provenance_feature_root": f"workspace/features/provenance/{experiment_name}",
            "passive_feature_root": f"workspace/features/passive/{experiment_name}",
        },
    }
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return experiment_name, cfg_path


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
    parser = argparse.ArgumentParser(
        description="Run a paired DF40-method FF++ bridge using a locked passive checkpoint and locked routing rules."
    )
    parser.add_argument("--method", required=True, choices=["faceswap", "simswap", "fsgan", "inswap"])
    parser.add_argument("--profile", choices=sorted(PROFILE_SETTINGS), default="sfrg05")
    parser.add_argument("--ffpp-split", choices=["", "train", "val", "test"], default="test")
    parser.add_argument("--max-pairs", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--paths", default="prototype/configs/paths.local.yaml")
    parser.add_argument("--force-config", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--checkpoint-override", default="")
    parser.add_argument("--rules-override", default="")
    parser.add_argument("--profile-alias", default="")
    parser.add_argument("--fake-root-override", default="")
    parser.add_argument("--experiment-tag", default="")
    args = parser.parse_args()

    root = project_root()
    profile_cfg = dict(PROFILE_SETTINGS[args.profile])
    if args.checkpoint_override:
        profile_cfg["checkpoint"] = args.checkpoint_override
    if args.rules_override:
        profile_cfg["rules"] = args.rules_override
    profile_label = args.profile_alias or profile_cfg["label"]
    experiment_name, cfg_path = write_config(
        root,
        args.method,
        profile_label,
        args.ffpp_split,
        args.max_pairs,
        args.force_config,
        args.fake_root_override,
        args.experiment_tag,
    )
    runner = runner_python(root)

    provenance_results = root / "prototype" / "reports" / f"{experiment_name}_provenance_results.csv"
    passive_results = root / "prototype" / "reports" / f"{experiment_name}_passive_results.csv"
    routing_csv = root / "prototype" / "reports" / f"{experiment_name}_routing.csv"
    routing_summary = root / "prototype" / "reports" / f"{experiment_name}_routing_summary.json"
    manifest_summary = root / "prototype" / "reports" / f"{experiment_name}_manifest_summary.json"

    commands: list[list[str]] = [
        [
            sys.executable,
            str(script_path(root, "build_df40_ffpp_paired_manifest.py")),
            "--config",
            str(cfg_path),
            "--summary-output",
            str(manifest_summary),
        ],
        [sys.executable, str(script_path(root, "run_videoseal_probe.py")), "--config", str(cfg_path), "--paths", args.paths],
        [sys.executable, str(script_path(root, "run_passive_probe.py")), "--config", str(cfg_path), "--paths", args.paths],
    ]

    if not args.prepare_only:
        commands.extend(
            [
                [
                    str(runner),
                    str(script_path(root, "execute_videoseal_probe.py")),
                    "--config",
                    str(cfg_path),
                    "--paths",
                    args.paths,
                    "--output",
                    str(provenance_results),
                    "--device",
                    args.device,
                ],
                [
                    str(runner),
                    str(script_path(root, "execute_effort_manifest_probe.py")),
                    "--config",
                    str(cfg_path),
                    "--output",
                    str(passive_results),
                    "--checkpoint",
                    str((root / profile_cfg["checkpoint"]).resolve()),
                    "--device",
                    args.device,
                ],
                [
                    sys.executable,
                    str(script_path(root, "merge_smoke_outputs.py")),
                    "--config",
                    str(cfg_path),
                    "--provenance-csv",
                    str(provenance_results),
                    "--passive-csv",
                    str(passive_results),
                    "--output",
                    str((root / "prototype" / "reports" / f"{experiment_name}_merged.csv").resolve()),
                ],
                [
                    sys.executable,
                    str(script_path(root, "run_routing_sanity.py")),
                    "--bridge-csv",
                    str((root / "prototype" / "reports" / f"{experiment_name}_merged.csv").resolve()),
                    "--rules",
                    str((root / profile_cfg["rules"]).resolve()),
                    "--output-csv",
                    str(routing_csv),
                    "--summary-json",
                    str(routing_summary),
                    "--backbone-name",
                    "VideoSeal",
                ],
            ]
        )

    report_commands = []
    status = "ok"
    try:
        for command in commands:
            print("RUN", " ".join(command))
            report_commands.append(run_command(command, root, args.dry_run))
    except Exception as exc:
        status = "failed"
        report_commands.append({"error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        report = {
            "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "status": status,
            "experiment_name": experiment_name,
            "method": args.method,
            "profile": args.profile,
            "profile_label": profile_label,
            "ffpp_split": args.ffpp_split or "all",
            "config": str(cfg_path),
            "fake_root": args.fake_root_override or str((root / "workspace" / "datasets" / "raw" / "df40" / args.method / "frames").resolve()),
            "checkpoint": str((root / profile_cfg["checkpoint"]).resolve()) if not Path(profile_cfg["checkpoint"]).is_absolute() else str(Path(profile_cfg["checkpoint"]).resolve()),
            "rules": str((root / profile_cfg["rules"]).resolve()) if not Path(profile_cfg["rules"]).is_absolute() else str(Path(profile_cfg["rules"]).resolve()),
            "commands": report_commands,
        }
        report_path = root / "prototype" / "reports" / f"{experiment_name}_run_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"report": str(report_path), "status": status, "experiment_name": experiment_name}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
