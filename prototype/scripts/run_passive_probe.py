from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List


def _bootstrap_src() -> None:
    script_dir = Path(__file__).resolve().parent
    src_root = script_dir.parent / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


_bootstrap_src()

from hybrid_router.config_io import load_optional_yaml_config, load_yaml_config
from hybrid_router.manifest_schema import load_manifest


JOB_FIELDS = [
    "sample_id",
    "split",
    "label",
    "protection",
    "input_path",
    "passive_key",
    "repo_root",
    "entrypoint",
    "detector_config",
    "env_hint",
    "command_hint",
    "command_template",
]


def default_paths_file(config_path: Path) -> Path:
    return config_path.parent / "paths.local.yaml"


def build_jobs(manifest_rows: List[Dict[str, str]], repo_root: str) -> List[Dict[str, str]]:
    entrypoint = str((Path(repo_root) / "training" / "test.py").resolve()) if repo_root else ""
    detector_config = str((Path(repo_root) / "training" / "config" / "detector" / "clip.yaml").resolve()) if repo_root else ""
    jobs: List[Dict[str, str]] = []
    for row in manifest_rows:
        input_path = row["planned_output_path"] if row["protection"] == "protected" else row["source_path"]
        jobs.append(
            {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "label": row["label"],
                "protection": row["protection"],
                "input_path": input_path,
                "passive_key": row["passive_key"],
                "repo_root": repo_root,
                "entrypoint": entrypoint,
                "detector_config": detector_config,
                "env_hint": "DeepfakeBench environment (README suggests Python 3.7.2)",
                "command_hint": "run DeepfakeBench test.py with CLIP detector config and export score plus penultimate feature",
                "command_template": "python training/test.py --detector_path training/config/detector/clip.yaml --test_dataset FaceForensics++ --weights_path <clip_checkpoint>",
            }
        )
    return jobs


def write_jobs(path: Path, jobs: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=JOB_FIELDS)
        writer.writeheader()
        writer.writerows(jobs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a passive-detector probe job sheet from the smoke-test manifest.")
    parser.add_argument("--config", required=True, help="Path to the smoke-test YAML config.")
    parser.add_argument("--manifest", help="Optional explicit manifest path.")
    parser.add_argument("--paths", help="Optional local paths YAML config.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_yaml_config(config_path)
    manifest_path = Path(args.manifest or config["paths"]["manifest_path"])
    jobs_path = Path(config["paths"]["passive_jobs_path"])
    paths_cfg = load_optional_yaml_config(args.paths or default_paths_file(config_path))
    repo_root = str(paths_cfg.get("repos", {}).get("deepfakebench", ""))

    manifest_rows = load_manifest(manifest_path)
    jobs = build_jobs(manifest_rows, repo_root)
    write_jobs(jobs_path, jobs)

    print(f"Wrote {len(jobs)} passive probe jobs to {jobs_path}")


if __name__ == "__main__":
    main()
