from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass(frozen=True)
class DatasetTarget:
    key: str
    role: str
    local_roots: tuple[str, ...]
    metadata_file: str | None = None
    required_for_topconf: bool = False
    note: str = ""


TARGETS = [
    DatasetTarget(
        key="UADFV",
        role="current_anchor",
        local_roots=("UADFV",),
        metadata_file="UADFV.json",
        note="Current local pilot anchor.",
    ),
    DatasetTarget(
        key="Celeb-DF-v1",
        role="current_anchor",
        local_roots=("Celeb-DF-v1", "celebdf"),
        metadata_file="Celeb-DF-v1.json",
        note="Current local cross-domain pilot anchor.",
    ),
    DatasetTarget(
        key="FaceForensics++",
        role="ffpp_bootstrap",
        local_roots=("FaceForensics++", "faceforensicspp"),
        metadata_file="FaceForensics++.json",
        required_for_topconf=True,
        note="Primary c23 training/bootstrap dataset.",
    ),
    DatasetTarget(
        key="FF-DF",
        role="ffpp_manipulation",
        local_roots=("FaceForensics++", "faceforensicspp"),
        metadata_file="FF-DF.json",
        note="FF++ DeepFakes manipulation slice.",
    ),
    DatasetTarget(
        key="FF-F2F",
        role="ffpp_manipulation",
        local_roots=("FaceForensics++", "faceforensicspp"),
        metadata_file="FF-F2F.json",
        note="FF++ Face2Face manipulation slice.",
    ),
    DatasetTarget(
        key="FF-FS",
        role="ffpp_manipulation",
        local_roots=("FaceForensics++", "faceforensicspp"),
        metadata_file="FF-FS.json",
        note="FF++ FaceSwap manipulation slice.",
    ),
    DatasetTarget(
        key="FF-NT",
        role="ffpp_manipulation",
        local_roots=("FaceForensics++", "faceforensicspp"),
        metadata_file="FF-NT.json",
        note="FF++ NeuralTextures manipulation slice.",
    ),
    DatasetTarget(
        key="Celeb-DF-v2",
        role="cross_dataset",
        local_roots=("Celeb-DF-v2", "celebdfv2"),
        metadata_file="Celeb-DF-v2.json",
        required_for_topconf=True,
        note="Stronger Celeb-DF benchmark for cross-dataset reporting.",
    ),
    DatasetTarget(
        key="DeepFakeDetection",
        role="cross_dataset",
        local_roots=("FaceForensics++", "faceforensicspp", "DeepFakeDetection", "deepfakedetection"),
        metadata_file="DeepFakeDetection.json",
        note="Google/Jigsaw DFD slice in the DeepfakeBench layout.",
    ),
    DatasetTarget(
        key="DFDCP",
        role="cross_dataset",
        local_roots=("DFDCP", "dfdcp"),
        metadata_file="DFDCP.json",
        note="DFDC preview split for cross-dataset stress.",
    ),
    DatasetTarget(
        key="DFDC",
        role="cross_dataset",
        local_roots=("DFDC", "dfdc"),
        metadata_file="DFDC.json",
        note="Large DFDC benchmark; may be heavy and access constrained.",
    ),
    DatasetTarget(
        key="FaceShifter",
        role="cross_dataset",
        local_roots=("FaceForensics++", "faceforensicspp", "FaceShifter", "faceshifter"),
        metadata_file="FaceShifter.json",
        note="Additional FF++-aligned manipulation dataset.",
    ),
    DatasetTarget(
        key="DF40",
        role="unseen_generator",
        local_roots=("df40", "DF40"),
        required_for_topconf=True,
        note="Broad unseen-generator benchmark with 40 generation techniques.",
    ),
    DatasetTarget(
        key="DeepFaceGen",
        role="unseen_generator",
        local_roots=("deepfacegen", "DeepFaceGen"),
        note="Large universal face-forgery benchmark.",
    ),
    DatasetTarget(
        key="VCF",
        role="deployment_stress",
        local_roots=("vcf", "VCF"),
        required_for_topconf=True,
        note="Video-conference scenario stress benchmark.",
    ),
    DatasetTarget(
        key="DeeperForensics-1.0",
        role="deployment_stress",
        local_roots=("deeperforensics", "DeeperForensics-1.0", "DeeperForensics"),
        note="Large real-world perturbation benchmark.",
    ),
]


OFFICIAL_ACCESS = {
    "FaceForensics++": {
        "url": "https://kaldir.vc.in.tum.de/faceforensics_benchmark/documentation",
        "status": "tos_form_required",
        "project_path": "workspace/datasets/raw/FaceForensics++ or workspace/datasets/raw/faceforensicspp",
    },
    "DeepFakeDetection": {
        "url": "https://kaldir.vc.in.tum.de/faceforensics_benchmark/documentation",
        "status": "tos_form_required",
        "project_path": "workspace/datasets/raw/FaceForensics++",
    },
    "Celeb-DF-v2": {
        "url": "https://github.com/yuezunli/celeb-deepfakeforensics",
        "status": "form_required",
        "project_path": "workspace/datasets/raw/Celeb-DF-v2",
    },
    "DF40": {
        "url": "https://github.com/YZY-stack/DF40",
        "status": "public_large_download",
        "project_path": "workspace/datasets/raw/df40",
    },
    "DeepFaceGen": {
        "url": "https://github.com/HengruiLou/DeepFaceGen",
        "status": "openxlab_download",
        "project_path": "workspace/datasets/raw/deepfacegen",
    },
    "VCF": {
        "url": "https://github.com/mirmashel/vcf_dataset",
        "status": "form_required",
        "project_path": "workspace/datasets/raw/vcf",
    },
    "DeeperForensics-1.0": {
        "url": "https://github.com/EndlessSora/DeeperForensics-1.0",
        "status": "noncommercial_research_access",
        "project_path": "workspace/datasets/raw/deeperforensics",
    },
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def normalize_relative_path(value: str) -> Path:
    parts = [part for part in value.replace("\\", "/").split("/") if part and part != "."]
    if not parts:
        return Path()
    return Path(*parts)


def iter_video_entries(node: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], dict[str, Any]]]:
    if not isinstance(node, dict):
        return
    if "label" in node and "frames" in node and isinstance(node["frames"], list):
        yield path, node
        return
    for key, value in node.items():
        yield from iter_video_entries(value, path + (str(key),))


def summarize_metadata(metadata_path: Path) -> dict[str, Any]:
    if not metadata_path.exists():
        return {
            "exists": False,
            "path": str(metadata_path),
            "videos": 0,
            "frames": 0,
            "labels": {},
            "splits": {},
            "first_frames": [],
        }

    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    labels: dict[str, int] = {}
    splits: dict[str, int] = {}
    first_frames: list[str] = []
    videos = 0
    frames = 0

    for entry_path, entry in iter_video_entries(data):
        videos += 1
        label = str(entry.get("label", "unknown"))
        labels[label] = labels.get(label, 0) + 1
        split = next((part for part in entry_path if part in {"train", "test", "val", "valid", "validation"}), "unknown")
        splits[split] = splits.get(split, 0) + 1
        frame_list = entry.get("frames", [])
        frames += len(frame_list)
        if frame_list and len(first_frames) < 32:
            first_frames.append(str(frame_list[0]))

    return {
        "exists": True,
        "path": str(metadata_path),
        "videos": videos,
        "frames": frames,
        "labels": labels,
        "splits": splits,
        "first_frames": first_frames,
    }


def candidate_frame_paths(raw_root: Path, local_roots: Iterable[str], relative_value: str) -> list[Path]:
    rel_path = normalize_relative_path(relative_value)
    if not rel_path.parts:
        return []

    roots = [raw_root / local_root for local_root in local_roots]
    roots.append(raw_root)
    rel_tail = Path(*rel_path.parts[1:]) if len(rel_path.parts) > 1 else rel_path

    candidates: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        for suffix in (rel_path, rel_tail):
            candidate = root / suffix
            key = str(candidate.resolve()) if candidate.exists() else str(candidate)
            if key not in seen:
                seen.add(key)
                candidates.append(candidate)
    return candidates


def sample_metadata_presence(
    metadata_path: Path,
    raw_root: Path,
    local_roots: Iterable[str],
    sample_size: int,
) -> dict[str, Any]:
    if not metadata_path.exists():
        return {"checked": 0, "existing": 0, "examples_missing": [], "examples_existing": []}

    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    checked = 0
    existing = 0
    examples_missing: list[str] = []
    examples_existing: list[str] = []

    for _, entry in iter_video_entries(data):
        frames = entry.get("frames", [])
        if not frames:
            continue
        relative_value = str(frames[len(frames) // 2])
        candidates = candidate_frame_paths(raw_root, local_roots, relative_value)
        match = next((candidate for candidate in candidates if candidate.exists()), None)
        checked += 1
        if match is not None:
            existing += 1
            if len(examples_existing) < 5:
                examples_existing.append(str(match))
        elif len(examples_missing) < 5:
            examples_missing.append(relative_value)
        if checked >= sample_size:
            break

    return {
        "checked": checked,
        "existing": existing,
        "examples_missing": examples_missing,
        "examples_existing": examples_existing,
    }


def count_local_media(root: Path, max_media_files: int) -> dict[str, Any]:
    if not root.exists():
        return {
            "path": str(root),
            "exists": False,
            "media_files": 0,
            "image_files": 0,
            "video_files": 0,
            "capped": False,
        }

    image_files = 0
    video_files = 0
    capped = False
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            image_files += 1
        elif suffix in VIDEO_EXTENSIONS:
            video_files += 1
        if image_files + video_files >= max_media_files:
            capped = True
            break

    return {
        "path": str(root),
        "exists": True,
        "media_files": image_files + video_files,
        "image_files": image_files,
        "video_files": video_files,
        "capped": capped,
    }


def summarize_local_roots(raw_root: Path, local_roots: Iterable[str], max_media_files: int) -> dict[str, Any]:
    root_summaries = [count_local_media(raw_root / local_root, max_media_files) for local_root in local_roots]
    seen_existing: set[str] = set()
    unique_roots = []
    for summary in root_summaries:
        raw_path = Path(summary["path"])
        path_key = str(raw_path.resolve() if raw_path.exists() else raw_path).casefold()
        if path_key in seen_existing:
            continue
        seen_existing.add(path_key)
        unique_roots.append(summary)

    return {
        "roots": unique_roots,
        "media_files": sum(item["media_files"] for item in unique_roots if item["exists"]),
        "image_files": sum(item["image_files"] for item in unique_roots if item["exists"]),
        "video_files": sum(item["video_files"] for item in unique_roots if item["exists"]),
        "existing_roots": [item["path"] for item in unique_roots if item["exists"]],
        "capped": any(item["capped"] for item in unique_roots),
    }


def summarize_downloads(workspace: Path) -> dict[str, Any]:
    downloads_root = workspace / "downloads"
    rgb_root = downloads_root / "rgb_datasets"
    metadata_root = downloads_root / "dataset_json"

    rgb_files = []
    if rgb_root.exists():
        for path in sorted(rgb_root.iterdir()):
            if path.is_file():
                rgb_files.append(
                    {
                        "name": path.name,
                        "size_gb": round(path.stat().st_size / (1024**3), 4),
                        "is_partial": path.suffix.lower() in {".part", ".crdownload", ".tmp"} or ".part" in path.name,
                    }
                )

    metadata_files = []
    if metadata_root.exists():
        for path in sorted(metadata_root.glob("*.json")):
            metadata_files.append({"name": path.name, "size_mb": round(path.stat().st_size / (1024**2), 3)})

    return {
        "rgb_downloads_path": str(rgb_root),
        "rgb_files": rgb_files,
        "metadata_path": str(metadata_root),
        "metadata_files": metadata_files,
    }


def has_data(summary: dict[str, Any]) -> bool:
    local = summary["local"]
    sample = summary["metadata_sample_presence"]
    return local["media_files"] > 0 or sample["existing"] > 0


def compute_readiness(dataset_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    available = {key: has_data(value) for key, value in dataset_summaries.items()}
    anchor_ready = available.get("UADFV", False) and available.get("Celeb-DF-v1", False)
    ffpp_bootstrap_ready = available.get("FaceForensics++", False)
    cross_keys = ["Celeb-DF-v2", "DeepFakeDetection", "DFDCP", "DFDC", "FaceShifter", "DeeperForensics-1.0"]
    cross_available = [key for key in cross_keys if available.get(key, False)]
    unseen_keys = ["DF40", "DeepFaceGen"]
    unseen_available = [key for key in unseen_keys if available.get(key, False)]
    deployment_keys = ["VCF", "DeeperForensics-1.0"]
    deployment_available = [key for key in deployment_keys if available.get(key, False)]

    return {
        "current_anchor_ready": anchor_ready,
        "ffpp_bootstrap_ready": ffpp_bootstrap_ready,
        "cross_dataset_available": cross_available,
        "cross_dataset_count": len(cross_available),
        "unseen_generator_available": unseen_available,
        "deployment_stress_available": deployment_available,
        "m1_current_data_ready": anchor_ready,
        "topconf_minimum_ready": ffpp_bootstrap_ready and len(cross_available) >= 1,
        "topconf_main_ready": ffpp_bootstrap_ready
        and len(cross_available) >= 2
        and bool(unseen_available)
        and bool(deployment_available),
        "missing_for_topconf_main": [
            name
            for name, ok in {
                "FaceForensics++ c23 bootstrap": ffpp_bootstrap_ready,
                "at least two cross-dataset benchmarks": len(cross_available) >= 2,
                "one unseen-generator benchmark": bool(unseen_available),
                "one deployment-stress benchmark": bool(deployment_available),
            }.items()
            if not ok
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit top-conference data readiness inside the project workspace.")
    parser.add_argument("--output", default="", help="JSON report path. Defaults to prototype/reports/topconf_data_audit_latest.json.")
    parser.add_argument("--metadata-sample-size", type=int, default=2048)
    parser.add_argument("--max-media-files-per-root", type=int, default=1_000_000)
    args = parser.parse_args()

    root = project_root()
    workspace = root / "workspace"
    raw_root = workspace / "datasets" / "raw"
    metadata_root = workspace / "downloads" / "dataset_json"
    reports_root = root / "prototype" / "reports"
    output_path = Path(args.output) if args.output else reports_root / "topconf_data_audit_latest.json"

    dataset_summaries: dict[str, dict[str, Any]] = {}
    for target in TARGETS:
        metadata_path = metadata_root / target.metadata_file if target.metadata_file else Path()
        metadata_summary = summarize_metadata(metadata_path) if target.metadata_file else {
            "exists": False,
            "path": "",
            "videos": 0,
            "frames": 0,
            "labels": {},
            "splits": {},
            "first_frames": [],
        }
        local_summary = summarize_local_roots(raw_root, target.local_roots, args.max_media_files_per_root)
        sample_presence = (
            sample_metadata_presence(metadata_path, raw_root, target.local_roots, args.metadata_sample_size)
            if target.metadata_file
            else {"checked": 0, "existing": 0, "examples_missing": [], "examples_existing": []}
        )
        dataset_summaries[target.key] = {
            "role": target.role,
            "required_for_topconf": target.required_for_topconf,
            "note": target.note,
            "metadata": metadata_summary,
            "local": local_summary,
            "metadata_sample_presence": sample_presence,
            "official_access": OFFICIAL_ACCESS.get(target.key, {}),
        }

    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    report = {
        "generated_at": now,
        "project_root": str(root),
        "workspace": str(workspace),
        "raw_root": str(raw_root),
        "python": sys.version,
        "downloads": summarize_downloads(workspace),
        "datasets": dataset_summaries,
        "readiness": compute_readiness(dataset_summaries),
        "policy": {
            "all_downloads_and_data_must_stay_under_project_root": True,
            "download_root": str(workspace / "downloads"),
            "raw_data_root": str(raw_root),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    dated_path = output_path.with_name(f"topconf_data_audit_{datetime.now().strftime('%Y-%m-%d')}.json")
    if dated_path != output_path:
        dated_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({"output": str(output_path), "dated_output": str(dated_path), "readiness": report["readiness"]}, indent=2))


if __name__ == "__main__":
    main()
