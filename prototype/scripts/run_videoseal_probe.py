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
    "source_path",
    "planned_output_path",
    "degradation_type",
    "degradation_level",
    "provenance_key",
    "repo_root",
    "entrypoint",
    "env_hint",
    "command_hint",
    "command_template",
]


def default_paths_file(config_path: Path) -> Path:
    return config_path.parent / "paths.local.yaml"


def build_jobs(manifest_rows: List[Dict[str, str]], repo_root: str) -> List[Dict[str, str]]:
    entrypoint = str((Path(repo_root) / "videoseal" / "__init__.py").resolve()) if repo_root else ""
    jobs: List[Dict[str, str]] = []
    for row in manifest_rows:
        if row["protection"] != "protected":
            continue
        jobs.append(
            {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "label": row["label"],
                "source_path": row["source_path"],
                "planned_output_path": row["planned_output_path"],
                "degradation_type": row["degradation_type"],
                "degradation_level": row["degradation_level"],
                "provenance_key": row["provenance_key"],
                "repo_root": repo_root,
                "entrypoint": entrypoint,
                "env_hint": "fssec-provenance (Python >= 3.9, torch >= 2.3.1)",
                "command_hint": "embed protected media, apply planned degradation, then export provenance statistics with videoseal.load('videoseal')",
                "command_template": "python -c \"import videoseal; print(videoseal.load('videoseal'))\"",
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
    parser = argparse.ArgumentParser(description="Create a VideoSeal probe job sheet from the smoke-test manifest.")
    parser.add_argument("--config", required=True, help="Path to the smoke-test YAML config.")
    parser.add_argument("--manifest", help="Optional explicit manifest path.")
    parser.add_argument("--paths", help="Optional local paths YAML config.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_yaml_config(config_path)
    manifest_path = Path(args.manifest or config["paths"]["manifest_path"])
    jobs_path = Path(config["paths"]["provenance_jobs_path"])
    paths_cfg = load_optional_yaml_config(args.paths or default_paths_file(config_path))
    repo_root = str(paths_cfg.get("repos", {}).get("videoseal", ""))

    manifest_rows = load_manifest(manifest_path)
    jobs = build_jobs(manifest_rows, repo_root)
    write_jobs(jobs_path, jobs)

    print(f"Wrote {len(jobs)} provenance probe jobs to {jobs_path}")


if __name__ == "__main__":
    main()
