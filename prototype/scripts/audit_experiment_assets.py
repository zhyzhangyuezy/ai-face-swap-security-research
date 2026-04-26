from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def count_media(root: Path, extensions: set[str]) -> tuple[int, int]:
    if not root.exists():
        return 0, 0
    videos = 0
    frames = 0
    for child in sorted(root.iterdir()):
        if child.is_dir():
            videos += 1
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            frames += 1
    return videos, frames


def maybe_gpu() -> dict:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception as exc:  # pragma: no cover
        return {"available": False, "error": str(exc), "gpus": []}

    gpus = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 5:
            gpus.append(
                {
                    "index": parts[0],
                    "name": parts[1],
                    "memory_total": parts[2],
                    "memory_used": parts[3],
                    "utilization": parts[4],
                }
            )
    return {"available": bool(gpus), "gpus": gpus}


def maybe_conda() -> dict:
    try:
        result = subprocess.run(
            ["conda", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        return {"available": True, "version": result.stdout.strip()}
    except Exception as exc:  # pragma: no cover
        return {"available": False, "error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit datasets, repos, checkpoints, and environment readiness.")
    parser.add_argument("--output", help="Optional JSON output path.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    workspace = project_root / "workspace"
    raw_root = workspace / "datasets" / "raw"
    checkpoints_root = workspace / "checkpoints"
    repos_root = workspace / "repos"
    downloads_root = workspace / "downloads" / "rgb_datasets"
    reports_root = project_root / "prototype" / "reports"
    manifests_root = project_root / "prototype" / "manifests"

    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    datasets = {
        "UADFV_real": count_media(raw_root / "UADFV" / "real" / "frames", image_exts),
        "UADFV_fake": count_media(raw_root / "UADFV" / "fake" / "frames", image_exts),
        "CelebDF_CelebReal": count_media(raw_root / "celebdf" / "Celeb-real" / "frames", image_exts),
        "CelebDF_YouTubeReal": count_media(raw_root / "celebdf" / "YouTube-real" / "frames", image_exts),
        "CelebDF_Fake": count_media(raw_root / "celebdf" / "Celeb-synthesis" / "frames", image_exts),
        "FFPP_real": count_media(raw_root / "faceforensicspp" / "real", image_exts),
        "FFPP_fake": count_media(raw_root / "faceforensicspp" / "fake", image_exts),
    }

    dataset_summary = {
        name: {"video_dirs": videos, "frames": frames}
        for name, (videos, frames) in datasets.items()
    }

    downloads = {
        path.name: round(path.stat().st_size / (1024**3), 2)
        for path in sorted(downloads_root.glob("*"))
        if path.is_file()
    }

    repos = {
        "DeepfakeBench": (repos_root / "DeepfakeBench").exists(),
        "VideoSeal": (repos_root / "videoseal").exists(),
    }

    checkpoints = {
        "deepfakebench_pretrained_zip": (checkpoints_root / "deepfakebench" / "pretrained.zip").exists(),
        "deepfakebench_xception": (checkpoints_root / "deepfakebench" / "weights" / "xception_best.pth").exists(),
        "deepfakebench_ucf": (checkpoints_root / "deepfakebench" / "weights" / "ucf_best.pth").exists(),
        "shape_predictor_81": (checkpoints_root / "deepfakebench" / "shape_predictor_81_face_landmarks.dat").exists(),
    }

    smoke_outputs = {
        "uadfv_manifest": (manifests_root / "uadfv_smoke_v0_1.csv").exists(),
        "uadfv_provenance_jobs": (reports_root / "uadfv_smoke_v0_1_provenance_jobs.csv").exists(),
        "uadfv_passive_jobs": (reports_root / "uadfv_smoke_v0_1_passive_jobs.csv").exists(),
        "uadfv_merged": (reports_root / "uadfv_smoke_v0_1_merged.csv").exists(),
    }

    uadfv_total_frames = dataset_summary["UADFV_real"]["frames"] + dataset_summary["UADFV_fake"]["frames"]
    celebdf_total_frames = (
        dataset_summary["CelebDF_CelebReal"]["frames"]
        + dataset_summary["CelebDF_YouTubeReal"]["frames"]
        + dataset_summary["CelebDF_Fake"]["frames"]
    )
    ffpp_total_frames = dataset_summary["FFPP_real"]["frames"] + dataset_summary["FFPP_fake"]["frames"]

    readiness = {
        "smoke_ready": repos["DeepfakeBench"] and repos["VideoSeal"] and uadfv_total_frames > 0,
        "passive_pilot_ready": uadfv_total_frames > 0 and celebdf_total_frames > 0,
        "hybrid_m1_ready": uadfv_total_frames > 0 and celebdf_total_frames > 0 and checkpoints["deepfakebench_pretrained_zip"],
        "topconf_ready": ffpp_total_frames > 0 and celebdf_total_frames > 0 and repos["DeepfakeBench"] and repos["VideoSeal"],
    }

    data = {
        "project_root": str(project_root),
        "python": sys.version,
        "conda": maybe_conda(),
        "gpu": maybe_gpu(),
        "datasets": dataset_summary,
        "downloads_gb": downloads,
        "repos": repos,
        "checkpoints": checkpoints,
        "smoke_outputs": smoke_outputs,
        "readiness": readiness,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
