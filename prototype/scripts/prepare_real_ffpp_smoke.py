from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def bootstrap_src() -> None:
    script_dir = Path(__file__).resolve().parent
    src_root = script_dir.parent / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


bootstrap_src()

from hybrid_router.config_io import load_yaml_config


def run_step(args: list[str]) -> None:
    cmd = [sys.executable, *args]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the real FF++ smoke pipeline from one YAML config.")
    parser.add_argument("--config", required=True, help="Path to the real FF++ import pipeline YAML config.")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    source_cfg = config["source"]
    targets_cfg = config["targets"]
    pipeline_cfg = config["pipeline"]
    script_dir = Path(__file__).resolve().parent

    import_script = str((script_dir / "import_ffpp_subset.py").resolve())
    build_script = str((script_dir / "build_ffpp_smoke_manifest.py").resolve())
    prov_script = str((script_dir / "run_videoseal_probe.py").resolve())
    passive_script = str((script_dir / "run_passive_probe.py").resolve())
    merge_script = str((script_dir / "merge_smoke_outputs.py").resolve())

    import_args = [
        import_script,
        "--target-real-dir",
        targets_cfg["real_dir"],
        "--target-fake-dir",
        targets_cfg["fake_dir"],
        "--max-real",
        str(source_cfg.get("max_real", 32)),
        "--max-fake",
        str(source_cfg.get("max_fake", 32)),
        "--mode",
        source_cfg.get("mode", "copy"),
        "--report",
        targets_cfg["import_report"],
    ]

    source_root = source_cfg.get("source_root")
    if source_root:
        import_args.extend(["--source-root", source_root])

    for path in source_cfg.get("real_source", []):
        import_args.extend(["--real-source", path])
    for path in source_cfg.get("fake_source", []):
        import_args.extend(["--fake-source", path])

    if source_cfg.get("prefer_videos", False):
        import_args.append("--prefer-videos")

    run_step(import_args)
    run_step([build_script, "--config", pipeline_cfg["smoke_config"]])
    run_step([prov_script, "--config", pipeline_cfg["smoke_config"], "--paths", pipeline_cfg["paths_config"]])
    run_step([passive_script, "--config", pipeline_cfg["smoke_config"], "--paths", pipeline_cfg["paths_config"]])
    run_step([merge_script, "--config", pipeline_cfg["smoke_config"]])

    print("Real FF++ smoke pipeline preparation finished.")


if __name__ == "__main__":
    main()
