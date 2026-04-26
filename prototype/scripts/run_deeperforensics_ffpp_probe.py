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
    "aux8_literal_fixed": {
        "checkpoint": "pilot/passive_baseline/runs/effort_clipl_rank4_patchattn_mean_sfrank_df40aux8_w10_m15_e5_v1/best.pt",
        "rules": "prototype/reports/route_label_rules_sfrank_df40aux8_transfertree_literal_fpr08.yaml",
        "label": "sfrank_df40aux8_literalfixed",
    },
}


def runner_python(root: Path) -> Path:
    candidate = root / ".conda" / "passive-pilot" / "python.exe"
    return candidate if candidate.exists() else Path(sys.executable)


def config_path(root: Path, experiment_name: str) -> Path:
    return root / "prototype" / "configs" / f"{experiment_name}.yaml"


def write_config(
    root: Path,
    *,
    split_name: str,
    subset_names: list[str],
    max_fake_videos: int,
    max_fake_per_subset: int,
    config_tag: str,
    force: bool,
) -> tuple[str, Path]:
    tag_suffix = f"_{config_tag}" if config_tag else ""
    experiment_name = f"deeperforensics_ffpp_{split_name}{tag_suffix}_v0_1"
    cfg_path = config_path(root, experiment_name)
    if cfg_path.exists() and not force:
        return experiment_name, cfg_path

    deeper_root = root / "workspace" / "datasets" / "raw" / "DeeperForensics-1.0"
    data = {
        "experiment_name": experiment_name,
        "paths": {
            "manifest_path": f"prototype/manifests/{experiment_name}.csv",
            "provenance_jobs_path": f"prototype/reports/{experiment_name}_provenance_jobs.csv",
            "passive_jobs_path": f"prototype/reports/{experiment_name}_passive_jobs.csv",
            "merged_report_path": f"prototype/reports/{experiment_name}_merged.csv",
        },
        "dataset": {
            "name": f"DeeperForensics-FFPP-{split_name}",
            "deeperforensics_root": str(deeper_root.resolve()),
            "manipulated_root": str((deeper_root / "manipulated_videos").resolve()),
            "splits_root": str((deeper_root / "lists" / "splits").resolve()),
            "ffpp_real_root": str(
                (root / "workspace" / "datasets" / "raw" / "FaceForensics++" / "original_sequences" / "youtube" / "c23" / "frames").resolve()
            ),
            "fake_frame_root": str((root / "workspace" / "datasets" / "derived" / f"deeperforensics_ffpp_{split_name}_frames_v1").resolve()),
            "split_name": split_name,
            "subset_names": subset_names,
            "video_extensions": [".mp4"],
            "image_extensions": [".png", ".jpg", ".jpeg"],
            "real_frame_selector": "middle",
            "max_fake_videos": max_fake_videos,
            "max_fake_per_subset": max_fake_per_subset,
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
        description="Run a DeeperForensics deployment-stress bridge by reusing local FF++ real anchors."
    )
    parser.add_argument("--profile", choices=sorted(PROFILE_SETTINGS), default="sfrg05")
    parser.add_argument("--df-split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--subset", action="append", default=[], help="Optional manipulated subset name. Repeat to keep only selected subsets.")
    parser.add_argument("--max-fake-videos", type=int, default=0)
    parser.add_argument("--max-fake-per-subset", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--paths", default="prototype/configs/paths.local.yaml")
    parser.add_argument("--force-config", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--checkpoint-override", default="")
    parser.add_argument("--rules-override", default="")
    parser.add_argument("--profile-alias", default="")
    parser.add_argument("--config-tag", default="")
    args = parser.parse_args()

    root = project_root()
    profile_cfg = dict(PROFILE_SETTINGS[args.profile])
    if args.checkpoint_override:
        profile_cfg["checkpoint"] = args.checkpoint_override
    if args.rules_override:
        profile_cfg["rules"] = args.rules_override
    profile_label = args.profile_alias or profile_cfg["label"]

    dataset_name, cfg_path = write_config(
        root,
        split_name=args.df_split,
        subset_names=args.subset,
        max_fake_videos=args.max_fake_videos,
        max_fake_per_subset=args.max_fake_per_subset,
        config_tag=args.config_tag,
        force=args.force_config,
    )
    experiment_name = f"{dataset_name}_{profile_label}"
    runner = runner_python(root)

    provenance_results = root / "prototype" / "reports" / f"{experiment_name}_provenance_results.csv"
    passive_results = root / "prototype" / "reports" / f"{experiment_name}_passive_results.csv"
    merged_csv = root / "prototype" / "reports" / f"{experiment_name}_merged.csv"
    routing_csv = root / "prototype" / "reports" / f"{experiment_name}_routing.csv"
    routing_summary = root / "prototype" / "reports" / f"{experiment_name}_routing_summary.json"
    manifest_summary = root / "prototype" / "reports" / f"{dataset_name}_manifest_summary.json"

    commands: list[list[str]] = [
        [
            sys.executable,
            str(script_path(root, "build_deeperforensics_ffpp_manifest.py")),
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
                    str(merged_csv),
                ],
                [
                    sys.executable,
                    str(script_path(root, "run_routing_sanity.py")),
                    "--bridge-csv",
                    str(merged_csv),
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
            "dataset_name": dataset_name,
            "experiment_name": experiment_name,
            "profile": args.profile,
            "profile_label": profile_label,
            "config": str(cfg_path),
            "split": args.df_split,
            "subset_names": args.subset,
            "checkpoint": str((root / profile_cfg["checkpoint"]).resolve()),
            "rules": str((root / profile_cfg["rules"]).resolve()),
            "commands": report_commands,
        }
        report_path = root / "prototype" / "reports" / f"{experiment_name}_run_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"report": str(report_path), "status": status, "experiment_name": experiment_name}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
