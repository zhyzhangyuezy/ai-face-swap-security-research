from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


FIELDS = [
    "sample_id",
    "split",
    "label",
    "protection",
    "degradation_type",
    "degradation_level",
    "source_path",
    "planned_output_path",
    "identity_id",
    "provenance_key",
    "passive_key",
    "pair_group_id",
    "notes",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def discover_video_dirs(root: Path) -> list[Path]:
    dirs = [path for path in root.iterdir() if path.is_dir()]
    dirs.sort()
    return dirs


def discover_frames(video_dir: Path, extensions: set[str]) -> list[Path]:
    frames = [path for path in video_dir.iterdir() if path.is_file() and path.suffix.lower() in extensions]
    frames.sort()
    return frames


def select_frame(frames: list[Path], selector: str) -> Path:
    if selector == "first":
        return frames[0]
    if selector == "last":
        return frames[-1]
    if selector == "middle":
        return frames[len(frames) // 2]
    raise ValueError(f"Unsupported selector: {selector}")


def sample_id(method: str, frame: Path) -> str:
    digest = hashlib.sha1(f"df40:{method}:{frame.as_posix()}".encode("utf-8")).hexdigest()[:12]
    return f"df40_{method}_f_{digest}"


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fake-only DF40 manifest for method-level passive stress tests.")
    parser.add_argument("--method", action="append", required=True, help="DF40 method directory name. Repeatable.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-videos-per-method", type=int, default=512)
    parser.add_argument("--frame-selector", choices=["first", "middle", "last"], default="middle")
    parser.add_argument("--extensions", default=".png,.jpg,.jpeg")
    args = parser.parse_args()

    root = project_root()
    extensions = {item.strip().lower() for item in args.extensions.split(",") if item.strip()}
    rows: list[dict[str, str]] = []
    stats: dict[str, dict[str, int]] = {}
    for method in args.method:
        method_root = root / "workspace" / "datasets" / "raw" / "df40" / method / "frames"
        if not method_root.exists():
            raise FileNotFoundError(f"Missing DF40 fake frames root: {method_root}")
        selected = 0
        scanned = 0
        for video_dir in discover_video_dirs(method_root):
            scanned += 1
            frames = discover_frames(video_dir, extensions)
            if not frames:
                continue
            frame = select_frame(frames, args.frame_selector)
            sid = sample_id(method, frame)
            rows.append(
                {
                    "sample_id": f"{sid}_u_clean",
                    "split": "eval",
                    "label": "fake",
                    "protection": "unprotected",
                    "degradation_type": "none",
                    "degradation_level": "clean",
                    "source_path": str(frame.resolve()),
                    "planned_output_path": str(frame.resolve()),
                    "identity_id": f"{method}/{video_dir.name}",
                    "provenance_key": "",
                    "passive_key": str(
                        (
                            root
                            / "workspace"
                            / "features"
                            / "passive"
                            / "df40_fake_only"
                            / method
                            / f"{sid}.npz"
                        ).resolve()
                    ),
                    "pair_group_id": f"pair_{sid}",
                    "notes": f"df40 fake-only method stress; method={method}",
                }
            )
            selected += 1
            if selected >= args.max_videos_per_method:
                break
        stats[method] = {"scanned_dirs": scanned, "selected_rows": selected}

    write_csv(Path(args.output), rows)
    print({"output": args.output, "rows": len(rows), "stats": stats})


if __name__ == "__main__":
    main()
