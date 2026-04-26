from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List


def _bootstrap_src() -> None:
    script_dir = Path(__file__).resolve().parent
    src_root = script_dir.parent / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


_bootstrap_src()

from hybrid_router.manifest_schema import FIELDNAMES, load_manifest


def parse_notes(notes: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for part in notes.split(";"):
        token = part.strip()
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def metadata_frame_count(video_path: Path) -> int:
    import imageio.v3 as iio

    try:
        meta = iio.immeta(video_path)
    except Exception:
        return 0
    fps = float(meta.get("fps") or 0.0)
    duration = float(meta.get("duration") or 0.0)
    if fps > 0 and duration > 0:
        return max(1, int(round(fps * duration)))
    nframes = meta.get("nframes") or meta.get("n_frames")
    try:
        return max(1, int(nframes))
    except Exception:
        return 0


def sample_indices(video_path: Path, frame_count: int) -> List[int]:
    total = metadata_frame_count(video_path)
    if frame_count <= 1:
        return [0]
    if total <= 1:
        return [0] + [max(1, 30 * idx) for idx in range(1, frame_count)]
    fractions = [0.0] + [0.1 + 0.8 * (idx / (frame_count - 1)) for idx in range(1, frame_count)]
    indices = [min(total - 1, max(0, int(round((total - 1) * frac)))) for frac in fractions]
    deduped: List[int] = []
    for index in indices:
        if index not in deduped:
            deduped.append(index)
    next_index = 1
    while len(deduped) < frame_count:
        candidate = min(total - 1, next_index)
        if candidate not in deduped:
            deduped.append(candidate)
        next_index += 1
    return deduped[:frame_count]


def read_frame(video_path: Path, index: int):
    import imageio.v3 as iio

    try:
        return iio.imread(video_path, index=index)
    except Exception:
        if index == 0:
            raise
        return iio.imread(video_path, index=0)


def write_frame(video_path: Path, dst_path: Path, index: int) -> str:
    import imageio.v3 as iio

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        return "exists"
    frame = read_frame(video_path, index)
    iio.imwrite(dst_path, frame)
    return "extracted"


def protected_variant(row: Dict[str, str]) -> str:
    planned = Path(row["planned_output_path"])
    parts = planned.parts
    try:
        protected_idx = parts.index("protected")
        return parts[protected_idx + 1]
    except Exception:
        if row["degradation_type"] == "none":
            return "clean"
        if row["degradation_type"] == "jpeg":
            return f"jpeg_{row['degradation_level']}"
        if row["degradation_type"] == "resize":
            return "resize_half" if row["degradation_level"] == "0.5x" else f"resize_{row['degradation_level']}"
    raise ValueError(f"Cannot infer protected variant for row {row['sample_id']}")


def feature_path(root: Path, record_id: str, suffix: str) -> str:
    return str((root / f"{record_id}{suffix}").resolve())


def raw_video_path(raw_root: Path, notes: Dict[str, str]) -> Path:
    domain = notes.get("domain", "c23")
    method = notes.get("method") or "targets"
    resolution = notes["resolution"]
    bucket = notes["bucket"]
    stem = notes["stem"]
    return (raw_root / domain / method / resolution / "mp4" / bucket / f"{stem}.mp4").resolve()


def temporal_frame_path(frame_root: Path, notes: Dict[str, str], slot: int, index: int, short_names: bool) -> Path:
    domain = notes.get("domain", "c23")
    method = notes.get("method") or "targets"
    resolution = notes["resolution"]
    bucket = notes["bucket"]
    stem = notes["stem"]
    if short_names:
        name = f"tf_{temporal_hash(f'{domain}:{method}:{resolution}:{bucket}:{stem}:{slot}:{index}')}_s{slot:02d}.png"
    else:
        name = f"{stem}_tf{slot:02d}_i{index:04d}.png"
    return (frame_root / domain / method / resolution / bucket / name).resolve()


def temporal_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def expand_rows(
    rows: Iterable[Dict[str, str]],
    *,
    raw_root: Path,
    frame_root: Path,
    derived_root: Path,
    provenance_root: Path,
    passive_root: Path,
    frame_count: int,
    benchmark_name: str,
    short_frame_names: bool,
) -> tuple[List[Dict[str, str]], dict]:
    expanded: List[Dict[str, str]] = []
    frame_cache: Dict[tuple[str, int], Path] = {}
    index_cache: Dict[str, List[int]] = {}
    extracted = Counter()
    source_video_keys = set()
    failures: List[dict] = []

    for row in rows:
        notes = parse_notes(row.get("notes", ""))
        video_path = raw_video_path(raw_root, notes)
        if not video_path.exists():
            failures.append({"sample_id": row["sample_id"], "video_path": str(video_path), "error": "missing_raw_video"})
            continue
        video_key = str(video_path)
        source_video_keys.add(video_key)
        if video_key not in index_cache:
            index_cache[video_key] = sample_indices(video_path, frame_count)
        indices = index_cache[video_key]
        temporal_video_key = temporal_hash(
            "|".join(
                [
                    notes.get("domain", "c23"),
                    notes.get("method") or "targets",
                    notes.get("resolution", ""),
                    notes.get("bucket", ""),
                    notes.get("stem", ""),
                ]
            )
        )

        for slot, frame_index in enumerate(indices):
            new_row = {field: row.get(field, "") for field in FIELDNAMES}
            if slot == 0:
                frame_path = Path(row["source_path"]).resolve()
                record_id = row["sample_id"]
            else:
                cache_key = (video_key, slot)
                if cache_key not in frame_cache:
                    dst = temporal_frame_path(frame_root, notes, slot, frame_index, short_frame_names)
                    action = write_frame(video_path, dst, frame_index)
                    extracted[action] += 1
                    frame_cache[cache_key] = dst
                frame_path = frame_cache[cache_key]
                record_id = f"{row['sample_id']}_tf{slot:02d}"
                new_row["sample_id"] = record_id
                new_row["source_path"] = str(frame_path)
                new_row["passive_key"] = feature_path(passive_root, record_id, ".npz")
                if row["protection"] == "protected":
                    variant = protected_variant(row)
                    new_row["planned_output_path"] = str(
                        (
                            derived_root
                            / row["split"]
                            / row["label"]
                            / "protected"
                            / variant
                            / row["identity_id"]
                            / frame_path.name
                        ).resolve()
                    )
                    new_row["provenance_key"] = feature_path(provenance_root, record_id, ".json")
                else:
                    new_row["planned_output_path"] = str(frame_path)
                    new_row["provenance_key"] = ""

            extra_notes = (
                f"temporal_benchmark={benchmark_name}; temporal_sample_id={row['sample_id']}; "
                f"temporal_video_key={temporal_video_key}; temporal_frame_slot={slot}; "
                f"temporal_frame_index={frame_index}; temporal_frame_count={frame_count}; "
                f"raw_video_path={video_path}"
            )
            new_row["notes"] = f"{row.get('notes', '').rstrip()}; {extra_notes}" if row.get("notes") else extra_notes
            expanded.append(new_row)

    summary = {
        "benchmark_name": benchmark_name,
        "source_rows": len(list(rows)) if not isinstance(rows, list) else len(rows),
        "expanded_rows": len(expanded),
        "frame_count": frame_count,
        "unique_source_videos": len(source_video_keys),
        "extracted": dict(extracted),
        "failures": failures[:25],
        "failure_count": len(failures),
        "frame_root": str(frame_root.resolve()),
        "derived_root": str(derived_root.resolve()),
        "provenance_root": str(provenance_root.resolve()),
        "passive_root": str(passive_root.resolve()),
        "short_frame_names": short_frame_names,
    }
    return expanded, summary


def write_manifest(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand a first-frame VCF manifest into a multi-frame temporal manifest.")
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--raw-root", default="workspace/datasets/raw/vcf")
    parser.add_argument("--frame-root", default="workspace/datasets/derived/vcf_c23_temporal_f3_frames_v1")
    parser.add_argument("--derived-root", default="workspace/datasets/derived/vcf_c23_temporal_f3_v1")
    parser.add_argument("--provenance-root", default="workspace/features/provenance/vcf_c23_temporal_f3_v1")
    parser.add_argument("--passive-root", default="workspace/features/passive/vcf_c23_temporal_f3_v1")
    parser.add_argument("--frame-count", type=int, default=3)
    parser.add_argument("--benchmark-name", default="vcf_c23_temporal_f3_v1")
    parser.add_argument("--short-frame-names", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_manifest(Path(args.source_manifest))
    expanded, summary = expand_rows(
        rows,
        raw_root=Path(args.raw_root),
        frame_root=Path(args.frame_root),
        derived_root=Path(args.derived_root),
        provenance_root=Path(args.provenance_root),
        passive_root=Path(args.passive_root),
        frame_count=args.frame_count,
        benchmark_name=args.benchmark_name,
        short_frame_names=args.short_frame_names,
    )
    if summary["failure_count"]:
        raise RuntimeError(json.dumps(summary, indent=2, ensure_ascii=False))
    write_manifest(Path(args.output_manifest), expanded)
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"manifest": str(Path(args.output_manifest).resolve()), **summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
