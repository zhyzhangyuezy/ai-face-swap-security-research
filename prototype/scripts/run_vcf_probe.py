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
    domain_name: str,
    fake_methods: list[str],
    resolution_names: list[str],
    bucket_names: list[str],
    max_target_videos: int,
    max_fake_per_subset: int,
    config_tag: str,
    force: bool,
) -> tuple[str, Path]:
    methods_tag = "allmethods" if not fake_methods else str(len(fake_methods)) + "methods"
    tag_suffix = f"_{config_tag}" if config_tag else ""
    experiment_name = f"vcf_{domain_name}_{methods_tag}{tag_suffix}_v0_1"
    cfg_path = config_path(root, experiment_name)
    if cfg_path.exists() and not force:
        return experiment_name, cfg_path

    vcf_root = root / "workspace" / "datasets" / "raw" / "vcf"
    data = {
        "experiment_name": experiment_name,
        "paths": {
            "manifest_path": f"prototype/manifests/{experiment_name}.csv",
            "provenance_jobs_path": f"prototype/reports/{experiment_name}_provenance_jobs.csv",
            "passive_jobs_path": f"prototype/reports/{experiment_name}_passive_jobs.csv",
            "merged_report_path": f"prototype/reports/{experiment_name}_merged.csv",
        },
        "dataset": {
            "name": f"VCF-{domain_name}",
            "vcf_root": str(vcf_root.resolve()),
            "domain_root": str((vcf_root / domain_name).resolve()),
            "frame_root": str((root / "workspace" / "datasets" / "derived" / f"{experiment_name}_frames_v1").resolve()),
            "domain_name": domain_name,
            "split_name": "test",
            "fake_methods": fake_methods,
            "resolution_names": resolution_names,
            "bucket_names": bucket_names,
            "video_extensions": [".mp4"],
            "max_target_videos": max_target_videos,
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
    parser = argparse.ArgumentParser(description="Run a VCF deployment-stress bridge from internal target/fake video pairs.")
    parser.add_argument("--profile", choices=sorted(PROFILE_SETTINGS), default="sfrg05")
    parser.add_argument("--domain", choices=["raw", "c23", "c40"], default="c23")
    parser.add_argument("--method", action="append", default=[], help="Optional fake method name. Repeat to keep only selected methods.")
    parser.add_argument("--resolution", action="append", default=[], help="Optional resolution name. Repeat to filter.")
    parser.add_argument("--bucket", action="append", default=[], help="Optional bucket name. Repeat to filter.")
    parser.add_argument("--max-target-videos", type=int, default=0)
    parser.add_argument("--max-fake-per-subset", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--paths", default="prototype/configs/paths.local.yaml")
    parser.add_argument("--force-config", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--checkpoint-override", default="")
    parser.add_argument("--rules-override", default="")
    parser.add_argument("--profile-alias", default="")
    parser.add_argument("--config-tag", default="full")
    parser.add_argument("--reuse-provenance-csv", default="")
    parser.add_argument("--reuse-passive-csv", default="")
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
        domain_name=args.domain,
        fake_methods=args.method,
        resolution_names=args.resolution,
        bucket_names=args.bucket,
        max_target_videos=args.max_target_videos,
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

    reuse_provenance = Path(args.reuse_provenance_csv).resolve() if args.reuse_provenance_csv else None
    reuse_passive = Path(args.reuse_passive_csv).resolve() if args.reuse_passive_csv else None
    if reuse_provenance and not reuse_provenance.exists():
        raise FileNotFoundError(f"Provenance CSV to reuse does not exist: {reuse_provenance}")
    if reuse_passive and not reuse_passive.exists():
        raise FileNotFoundError(f"Passive CSV to reuse does not exist: {reuse_passive}")

    commands: list[list[str]] = [
        [
            sys.executable,
            str(script_path(root, "build_vcf_manifest.py")),
            "--config",
            str(cfg_path),
            "--summary-output",
            str(manifest_summary),
        ]
    ]

    actual_provenance_csv = reuse_provenance or provenance_results
    actual_passive_csv = reuse_passive or passive_results

    if not args.prepare_only and not reuse_provenance:
        commands.extend(
            [
                [sys.executable, str(script_path(root, "run_videoseal_probe.py")), "--config", str(cfg_path), "--paths", args.paths],
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
            ]
        )

    if not args.prepare_only and not reuse_passive:
        commands.extend(
            [
                [sys.executable, str(script_path(root, "run_passive_probe.py")), "--config", str(cfg_path), "--paths", args.paths],
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
            ]
        )

    if not args.prepare_only:
        commands.extend(
            [
                [
                    sys.executable,
                    str(script_path(root, "merge_smoke_outputs.py")),
                    "--config",
                    str(cfg_path),
                    "--provenance-csv",
                    str(actual_provenance_csv),
                    "--passive-csv",
                    str(actual_passive_csv),
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
            "domain_name": args.domain,
            "fake_methods": args.method,
            "resolutions": args.resolution,
            "buckets": args.bucket,
            "checkpoint": str((root / profile_cfg["checkpoint"]).resolve()),
            "rules": str((root / profile_cfg["rules"]).resolve()),
            "reused_provenance_csv": str(reuse_provenance) if reuse_provenance else "",
            "reused_passive_csv": str(reuse_passive) if reuse_passive else "",
            "commands": report_commands,
        }
        report_path = root / "prototype" / "reports" / f"{experiment_name}_run_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"report": str(report_path), "status": status, "experiment_name": experiment_name}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
