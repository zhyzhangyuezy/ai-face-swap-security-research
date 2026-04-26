from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def count_media(root: Path) -> dict[str, Any]:
    if not root.exists():
        return {"exists": False, "image_files": 0, "video_files": 0, "video_dirs": 0}
    image_files = 0
    video_files = 0
    video_dirs = 0
    frames_root = root / "fake" / "frames"
    if frames_root.exists():
        video_dirs += sum(1 for path in frames_root.iterdir() if path.is_dir())
    frames_root = root / "real" / "frames"
    if frames_root.exists():
        video_dirs += sum(1 for path in frames_root.iterdir() if path.is_dir())
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            image_files += 1
        elif suffix in VIDEO_EXTENSIONS:
            video_files += 1
    return {"exists": True, "image_files": image_files, "video_files": video_files, "video_dirs": video_dirs}


def validate_inside(base: Path, candidate: Path) -> None:
    try:
        candidate.resolve().relative_to(base.resolve())
    except ValueError as exc:
        raise SystemExit(f"Unsafe archive path escapes target: {candidate}") from exc


def infer_method_name(archive: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    name = archive.name
    if name.lower().startswith("df40_"):
        name = name[5:]
    if name.lower().endswith(".zip"):
        name = name[:-4]
    return name


def extract_archive(archive: Path, raw_root: Path, method: str, force: bool) -> dict[str, Any]:
    if not archive.exists():
        raise FileNotFoundError(f"Archive not found: {archive}")
    if not zipfile.is_zipfile(archive):
        raise ValueError(f"Archive is not a valid zip file: {archive}")

    target_root = raw_root / "df40"
    method_root = target_root / method
    if method_root.exists() and any(method_root.iterdir()) and not force:
        return {
            "status": "skipped_existing",
            "archive": str(archive),
            "target": str(method_root),
            "media": count_media(method_root),
        }
    if force and method_root.exists():
        shutil.rmtree(method_root)

    target_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        members = zf.infolist()
        for member in members:
            validate_inside(target_root, target_root / member.filename)
        zf.extractall(target_root)

    extracted_root = method_root
    if not extracted_root.exists():
        candidates = [
            path
            for path in target_root.iterdir()
            if path.is_dir() and path.name.lower() == method.lower()
        ]
        if candidates:
            extracted_root = candidates[0]

    return {
        "status": "ok",
        "archive": str(archive),
        "target": str(extracted_root),
        "entries": len(members),
        "media": count_media(extracted_root),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely extract a DF40 method zip under workspace/datasets/raw/df40.")
    parser.add_argument("--archive", required=True)
    parser.add_argument("--method", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    root = project_root()
    archive = Path(args.archive)
    if not archive.is_absolute():
        archive = root / archive
    method = infer_method_name(archive, args.method or None)
    raw_root = root / "workspace" / "datasets" / "raw"

    result = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "method": method,
        **extract_archive(archive, raw_root, method, args.force),
    }

    report_path = Path(args.report) if args.report else root / "prototype" / "reports" / f"df40_{method}_extract_report.json"
    if not report_path.is_absolute():
        report_path = root / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"report": str(report_path), **result}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
