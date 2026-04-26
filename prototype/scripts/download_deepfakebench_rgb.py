from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gdown


DEEPFAKEBENCH_RGB_FOLDER_URL = "https://drive.google.com/drive/folders/1N4X3rvx9IhmkEZK-KIk4OxBrQb9BRUcs?usp=drive_link"


@dataclass(frozen=True)
class DatasetDownload:
    key: str
    file_name: str
    google_drive_id: str
    raw_target: str
    priority: int
    note: str


DATASETS = {
    "celeb-df-v1": DatasetDownload(
        key="Celeb-DF-v1",
        file_name="Celeb-DF-v1.zip",
        google_drive_id="1cacTBE9apKTZBhUS8cW0Cyzf-ObA4vVY",
        raw_target="workspace/datasets/raw/celebdf",
        priority=99,
        note="Already available locally; keep only as checksum/reference backup.",
    ),
    "celeb-df-v2": DatasetDownload(
        key="Celeb-DF-v2",
        file_name="Celeb-DF-v2.zip",
        google_drive_id="1oSihXtB0caSGAX0Tt3MxgFbsuY46ecml",
        raw_target="workspace/datasets/raw/Celeb-DF-v2",
        priority=2,
        note="Cross-dataset benchmark for main results.",
    ),
    "dfdc": DatasetDownload(
        key="DFDC",
        file_name="DFDC.zip",
        google_drive_id="1lPBtwJObgsQZVvDFV4wFC3UJL1vAwQkB",
        raw_target="workspace/datasets/raw/DFDC",
        priority=4,
        note="Large cross-dataset benchmark; DeepfakeBench release includes test data.",
    ),
    "dfdcp": DatasetDownload(
        key="DFDCP",
        file_name="DFDCP.zip",
        google_drive_id="1MShgs4zx9ScN9DX8wkBfid6modOwhSgx",
        raw_target="workspace/datasets/raw/DFDCP",
        priority=3,
        note="DFDC preview split; good short-term cross-dataset target.",
    ),
    "faceforensics++": DatasetDownload(
        key="FaceForensics++",
        file_name="FaceForensics++.zip",
        google_drive_id="1mZ9NNtgW_4oo9S996uQh9-SmRYaLxPnb",
        raw_target="workspace/datasets/raw/FaceForensics++",
        priority=1,
        note="Primary FF++ c23 bootstrap dataset.",
    ),
    "uadfv": DatasetDownload(
        key="UADFV",
        file_name="UADFV.zip",
        google_drive_id="1kEMijoDJKaIMKjJFK7YJVA71z3Femd0y",
        raw_target="workspace/datasets/raw/UADFV",
        priority=99,
        note="Already available locally; keep only as checksum/reference backup.",
    ),
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def normalize_key(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def downloads_root(root: Path) -> Path:
    return root / "workspace" / "downloads" / "rgb_datasets"


def reports_root(root: Path) -> Path:
    return root / "prototype" / "reports"


def archive_path(root: Path, item: DatasetDownload) -> Path:
    return downloads_root(root) / item.file_name


def raw_path(root: Path, item: DatasetDownload) -> Path:
    return root / item.raw_target


def media_count(path: Path, limit: int = 100_000) -> dict[str, Any]:
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    if not path.exists():
        return {"exists": False, "media_files": 0, "capped": False}
    count = 0
    capped = False
    for child in path.rglob("*"):
        if child.is_file() and child.suffix.lower() in image_exts | video_exts:
            count += 1
            if count >= limit:
                capped = True
                break
    return {"exists": True, "media_files": count, "capped": capped}


def list_official_folder(root: Path) -> list[dict[str, Any]]:
    scratch = downloads_root(root) / "_deepfakebench_rgb_index"
    files = gdown.download_folder(
        url=DEEPFAKEBENCH_RGB_FOLDER_URL,
        output=str(scratch),
        quiet=True,
        use_cookies=False,
        skip_download=True,
    )
    rows = []
    for file_info in files:
        rows.append(
            {
                "id": getattr(file_info, "id", ""),
                "name": Path(getattr(file_info, "path", "")).name,
                "suggested_local_path": str(getattr(file_info, "local_path", "")),
            }
        )
    return rows


def current_status(root: Path, official_files: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    official_by_name = {row["name"]: row for row in official_files or []}
    rows = []
    for item in sorted(DATASETS.values(), key=lambda value: value.priority):
        archive = archive_path(root, item)
        raw = raw_path(root, item)
        official = official_by_name.get(item.file_name, {})
        rows.append(
            {
                "key": item.key,
                "priority": item.priority,
                "archive_path": str(archive),
                "archive_exists": archive.exists(),
                "archive_size_gb": round(archive.stat().st_size / (1024**3), 4) if archive.exists() else 0,
                "partial_candidates": [str(path) for path in sorted(downloads_root(root).glob(f"{item.file_name}*")) if path.name != item.file_name],
                "raw_target": str(raw),
                "raw_status": media_count(raw),
                "google_drive_id": item.google_drive_id,
                "official_folder_id": official.get("id", ""),
                "note": item.note,
            }
        )
    return {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "project_root": str(root),
        "folder_url": DEEPFAKEBENCH_RGB_FOLDER_URL,
        "datasets": rows,
    }


def write_status(root: Path, data: dict[str, Any]) -> Path:
    output = reports_root(root) / "deepfakebench_rgb_download_status.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def append_attempt(root: Path, event: dict[str, Any]) -> Path:
    output = reports_root(root) / "deepfakebench_rgb_download_attempts.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    return output


def resolve_dataset(name: str) -> DatasetDownload:
    key = normalize_key(name)
    aliases = {
        "ffpp": "faceforensics++",
        "faceforensics": "faceforensics++",
        "faceforensicspp": "faceforensics++",
        "cdfv2": "celeb-df-v2",
        "celebdfv2": "celeb-df-v2",
        "cdf-v2": "celeb-df-v2",
        "dfdc-preview": "dfdcp",
    }
    key = aliases.get(key, key)
    if key not in DATASETS:
        allowed = ", ".join(item.key for item in sorted(DATASETS.values(), key=lambda value: value.priority))
        raise SystemExit(f"Unknown dataset '{name}'. Allowed: {allowed}")
    return DATASETS[key]


def download_dataset(root: Path, item: DatasetDownload, force: bool, quiet: bool) -> Path:
    target = archive_path(root, item)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0 and not force:
        print(f"Archive already exists, skipping: {target}")
        return target
    print(f"Downloading {item.key} to {target}")
    gdown.download(
        id=item.google_drive_id,
        output=str(target),
        quiet=quiet,
        use_cookies=False,
        resume=True,
    )
    return target


def extract_archive(root: Path, item: DatasetDownload, force: bool) -> Path:
    archive = archive_path(root, item)
    target = raw_path(root, item)
    if not archive.exists():
        raise SystemExit(f"Cannot extract missing archive: {archive}")
    if target.exists() and any(target.iterdir()) and not force:
        print(f"Raw target is non-empty, skipping extraction: {target}")
        return target
    target.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive} to {target}")
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise SystemExit(f"Unsafe archive member: {member.filename}")
        zf.extractall(target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="List/download DeepfakeBench RGB-format dataset archives into this project.")
    parser.add_argument("--list", action="store_true", help="List official Google Drive folder contents and local status.")
    parser.add_argument("--download", action="append", default=[], help="Dataset key to download. May be repeated.")
    parser.add_argument("--extract", action="append", default=[], help="Dataset key to extract after the archive exists. May be repeated.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    root = project_root()
    if not str(downloads_root(root)).startswith(str(root)):
        raise SystemExit("Refusing to write outside the project root.")

    official_files = list_official_folder(root) if args.list else []
    attempts = []
    for dataset_name in args.download:
        item = resolve_dataset(dataset_name)
        event = {
            "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "action": "download",
            "dataset": item.key,
            "archive_path": str(archive_path(root, item)),
            "status": "started",
        }
        try:
            downloaded = download_dataset(root, item, force=args.force, quiet=args.quiet)
            event.update(
                {
                    "status": "ok",
                    "archive_exists": downloaded.exists(),
                    "archive_size_gb": round(downloaded.stat().st_size / (1024**3), 4) if downloaded.exists() else 0,
                }
            )
        except Exception as exc:  # gdown exposes several network/Drive failure types.
            event.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
            print(f"Download failed for {item.key}: {type(exc).__name__}: {exc}", file=sys.stderr)
        append_attempt(root, event)
        attempts.append(event)
    for dataset_name in args.extract:
        extract_archive(root, resolve_dataset(dataset_name), force=args.force)

    status = current_status(root, official_files)
    status["attempts_this_run"] = attempts
    output = write_status(root, status)
    print(json.dumps({"status_path": str(output), "datasets": status["datasets"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
