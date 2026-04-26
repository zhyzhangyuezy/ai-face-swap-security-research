from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def discover_files(roots: Sequence[Path], exts: set[str]) -> List[Path]:
    files: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in exts:
                files.append(path)
    files.sort()
    return files


def discover_from_source_root(source_root: Path) -> Tuple[List[Path], List[Path], List[Path], List[Path]]:
    real_roots: List[Path] = []
    fake_roots: List[Path] = []

    direct_real = source_root / "original_sequences"
    direct_fake = source_root / "manipulated_sequences"
    if direct_real.exists():
        real_roots.append(direct_real)
    if direct_fake.exists():
        fake_roots.append(direct_fake)

    if not real_roots and (source_root / "real").exists():
        real_roots.append(source_root / "real")
    if not fake_roots and (source_root / "fake").exists():
        fake_roots.append(source_root / "fake")

    real_images = discover_files(real_roots, IMAGE_EXTS)
    fake_images = discover_files(fake_roots, IMAGE_EXTS)
    real_videos = discover_files(real_roots, VIDEO_EXTS)
    fake_videos = discover_files(fake_roots, VIDEO_EXTS)
    return real_images, fake_images, real_videos, fake_videos


def safe_name(src: Path) -> str:
    digest = hashlib.sha1(src.as_posix().encode("utf-8")).hexdigest()[:10]
    return f"{src.stem}_{digest}{src.suffix.lower()}"


def transfer_file(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            shutil.copy2(src, dst)
            return "copy_fallback"
    shutil.copy2(src, dst)
    return "copy"


def extract_first_frame(video_path: Path, dst_png: Path) -> str:
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise RuntimeError("imageio is required for video-frame extraction. Install imageio and imageio-ffmpeg.") from exc

    dst_png.parent.mkdir(parents=True, exist_ok=True)
    if dst_png.exists():
        return "exists"
    frame = iio.imread(video_path, index=0)
    iio.imwrite(dst_png, frame)
    return "extract_frame"


def import_subset(files: Sequence[Path], target_dir: Path, limit: int, mode: str, label: str) -> List[dict]:
    rows: List[dict] = []
    for src in files[:limit]:
        dst = target_dir / safe_name(src)
        action = transfer_file(src, dst, mode)
        rows.append(
            {
                "label": label,
                "source_path": str(src.resolve()),
                "target_path": str(dst.resolve()),
                "action": action,
            }
        )
    return rows


def import_videos_as_frames(files: Sequence[Path], target_dir: Path, limit: int, label: str) -> List[dict]:
    rows: List[dict] = []
    for src in files[:limit]:
        dst = target_dir / f"{safe_name(src)}.png"
        action = extract_first_frame(src, dst)
        rows.append(
            {
                "label": label,
                "source_path": str(src.resolve()),
                "target_path": str(dst.resolve()),
                "action": action,
            }
        )
    return rows


def write_report(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["label", "source_path", "target_path", "action"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a small FF++ subset into the project workspace.")
    parser.add_argument("--source-root", help="Canonical FF++ root containing original_sequences and manipulated_sequences, or a root with real/ and fake/.")
    parser.add_argument("--real-source", action="append", default=[], help="Explicit real-image source directory. Can be repeated.")
    parser.add_argument("--fake-source", action="append", default=[], help="Explicit fake-image source directory. Can be repeated.")
    parser.add_argument("--target-real-dir", required=True, help="Target directory for imported real images.")
    parser.add_argument("--target-fake-dir", required=True, help="Target directory for imported fake images.")
    parser.add_argument("--max-real", type=int, default=32, help="Maximum number of real images to import.")
    parser.add_argument("--max-fake", type=int, default=32, help="Maximum number of fake images to import.")
    parser.add_argument("--mode", choices=["copy", "hardlink"], default="copy", help="Transfer mode.")
    parser.add_argument("--prefer-videos", action="store_true", help="Prefer extracting one frame from each video even when images are available.")
    parser.add_argument("--report", required=True, help="CSV report path.")
    args = parser.parse_args()

    real_images: List[Path] = []
    fake_images: List[Path] = []
    real_videos: List[Path] = []
    fake_videos: List[Path] = []

    if args.source_root:
        src_root = Path(args.source_root)
        real_images, fake_images, real_videos, fake_videos = discover_from_source_root(src_root)

    if args.real_source:
        real_images = discover_files([Path(p) for p in args.real_source], IMAGE_EXTS)
        real_videos = discover_files([Path(p) for p in args.real_source], VIDEO_EXTS)
    if args.fake_source:
        fake_images = discover_files([Path(p) for p in args.fake_source], IMAGE_EXTS)
        fake_videos = discover_files([Path(p) for p in args.fake_source], VIDEO_EXTS)

    target_real = Path(args.target_real_dir)
    target_fake = Path(args.target_fake_dir)

    rows: List[dict] = []
    use_real_videos = args.prefer_videos or (not real_images and bool(real_videos))
    use_fake_videos = args.prefer_videos or (not fake_images and bool(fake_videos))

    if use_real_videos and real_videos:
        rows.extend(import_videos_as_frames(real_videos, target_real, args.max_real, "real"))
    else:
        rows.extend(import_subset(real_images, target_real, args.max_real, args.mode, "real"))

    if use_fake_videos and fake_videos:
        rows.extend(import_videos_as_frames(fake_videos, target_fake, args.max_fake, "fake"))
    else:
        rows.extend(import_subset(fake_images, target_fake, args.max_fake, args.mode, "fake"))

    write_report(Path(args.report), rows)

    print(f"Imported {sum(1 for r in rows if r['label'] == 'real')} real images and {sum(1 for r in rows if r['label'] == 'fake')} fake images.")
    print(f"Report written to {args.report}")


if __name__ == "__main__":
    main()
