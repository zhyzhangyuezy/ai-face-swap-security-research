from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
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

from hybrid_router.manifest_schema import FIELDNAMES


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
REAL_SLOTS_CACHE: Dict[str, List[tuple[int, Path, int]]] = {}
VIDEO_INDEX_CACHE: Dict[str, List[int]] = {}
VIDEO_FRAME_CACHE: Dict[tuple[str, str, int, int], Path] = {}


def parse_notes(notes: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for part in notes.split(";"):
        token = part.strip()
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_manifest(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def feature_path(root: Path, record_id: str, suffix: str) -> str:
    return str((root / f"{record_id}{suffix}").resolve())


def temporal_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def discover_frames(root: Path) -> List[Path]:
    files = [path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS]

    def key(path: Path) -> tuple[int, str]:
        try:
            return (int(path.stem), path.name)
        except ValueError:
            return (10**12, path.name)

    files.sort(key=key)
    return files


def frame_number(path: Path) -> int:
    digits = re.sub(r"\D", "", path.stem)
    return int(digits) if digits else 0


def chronological_real_slots(anchor_path: Path) -> List[tuple[int, Path, int]]:
    cache_key = str(anchor_path.resolve())
    if cache_key in REAL_SLOTS_CACHE:
        return REAL_SLOTS_CACHE[cache_key]
    frames = discover_frames(anchor_path.parent)
    if not frames:
        result = [(0, anchor_path, frame_number(anchor_path))]
        REAL_SLOTS_CACHE[cache_key] = result
        return result
    anchor = anchor_path.resolve()
    first = frames[0].resolve()
    last = frames[-1].resolve()
    if anchor not in [frame.resolve() for frame in frames]:
        anchor = frames[len(frames) // 2].resolve()
    selected = [first, anchor, last]
    deduped: List[Path] = []
    for path in selected:
        if path not in deduped:
            deduped.append(path)
    cursor = 0
    while len(deduped) < 3 and cursor < len(frames):
        candidate = frames[cursor].resolve()
        if candidate not in deduped:
            deduped.append(candidate)
        cursor += max(1, len(frames) // 3)
    while len(deduped) < 3:
        deduped.append(deduped[-1])
    result = [(slot, path, frame_number(path)) for slot, path in enumerate(deduped[:3])]
    REAL_SLOTS_CACHE[cache_key] = result
    return result


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


def sample_video_indices(video_path: Path, frame_count: int) -> List[int]:
    cache_key = f"{video_path.resolve()}::{frame_count}"
    if cache_key in VIDEO_INDEX_CACHE:
        return VIDEO_INDEX_CACHE[cache_key]
    total = metadata_frame_count(video_path)
    if total <= 1:
        result = list(range(frame_count))
        VIDEO_INDEX_CACHE[cache_key] = result
        return result
    fractions = [0.0, 0.5, 0.9]
    indices = [min(total - 1, max(0, int(round((total - 1) * frac)))) for frac in fractions[:frame_count]]
    deduped: List[int] = []
    for index in indices:
        if index not in deduped:
            deduped.append(index)
    cursor = 1
    while len(deduped) < frame_count:
        candidate = min(total - 1, cursor)
        if candidate not in deduped:
            deduped.append(candidate)
        cursor += 1
    result = deduped[:frame_count]
    VIDEO_INDEX_CACHE[cache_key] = result
    return result


def read_frame(video_path: Path, index: int):
    import imageio.v3 as iio

    try:
        return iio.imread(video_path, index=index)
    except Exception:
        if index == 0:
            raise
        return iio.imread(video_path, index=0)


def write_video_frame(video_path: Path, dst_path: Path, index: int) -> str:
    import imageio.v3 as iio

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        return "exists"
    iio.imwrite(dst_path, read_frame(video_path, index))
    return "extracted"


def variant_name(row: Dict[str, str]) -> str:
    if row["degradation_type"] == "none":
        return "clean"
    if row["degradation_type"] == "jpeg":
        return f"jpeg_{row['degradation_level']}"
    if row["degradation_type"] == "resize":
        return "resize_half" if row["degradation_level"] == "0.5x" else f"resize_{row['degradation_level']}"
    return f"{row['degradation_type']}_{row['degradation_level']}".replace("/", "_")


def fake_video_path(manipulated_root: Path, row: Dict[str, str], notes: Dict[str, str]) -> Path:
    if notes.get("video_path"):
        return Path(notes["video_path"])
    subset, stem = row["identity_id"].split("__", 1)
    return manipulated_root / subset / f"{stem}.mp4"


def fake_frame_path(frame_root: Path, video_path: Path, subset: str, slot: int, index: int) -> Path:
    name = f"df_{temporal_hash(f'{subset}:{video_path.stem}:{slot}:{index}')}_s{slot:02d}.png"
    return (frame_root / "fake" / subset / name).resolve()


def cached_fake_frame(frame_root: Path, video_path: Path, subset: str, slot: int, index: int, extracted: Counter) -> Path:
    cache_key = (str(frame_root.resolve()), str(video_path.resolve()), slot, index)
    if cache_key in VIDEO_FRAME_CACHE:
        return VIDEO_FRAME_CACHE[cache_key]
    frame_path = fake_frame_path(frame_root, video_path, subset, slot, index)
    action = write_video_frame(video_path, frame_path, index)
    extracted[action] += 1
    VIDEO_FRAME_CACHE[cache_key] = frame_path
    return frame_path


def protected_output_path(
    derived_root: Path,
    split: str,
    label: str,
    row: Dict[str, str],
    frame_path: Path,
) -> str:
    return str((derived_root / split / label / "protected" / variant_name(row) / row["identity_id"] / frame_path.name).resolve())


def temporal_video_key(row: Dict[str, str], notes: Dict[str, str], raw_path: str) -> str:
    key = "|".join([row["label"], row["identity_id"], notes.get("source_id", ""), raw_path])
    return temporal_hash(key)


def keep_row(row: Dict[str, str], variants: set[str]) -> bool:
    if not variants:
        return True
    return variant_name(row) in variants or row["protection"] == "unprotected" and "clean" in variants


def source_id_for_row(row: Dict[str, str]) -> str:
    notes = parse_notes(row.get("notes", ""))
    if notes.get("source_id"):
        return notes["source_id"]
    if row["label"] == "real":
        return row["identity_id"]
    return row["identity_id"].split("__", 1)[-1].split("_", 1)[0]


def split_source_ids(rows: List[Dict[str, str]], seed: int, calib_fraction: float) -> tuple[set[str], set[str]]:
    ids = sorted({source_id_for_row(row) for row in rows})
    rng = random.Random(seed)
    rng.shuffle(ids)
    cutoff = max(1, min(len(ids) - 1, int(round(len(ids) * calib_fraction))))
    return set(ids[:cutoff]), set(ids[cutoff:])


def expand_row(
    row: Dict[str, str],
    *,
    manipulated_root: Path,
    frame_root: Path,
    derived_root: Path,
    provenance_root: Path,
    passive_root: Path,
    benchmark_name: str,
) -> tuple[List[Dict[str, str]], Counter, List[dict]]:
    notes = parse_notes(row.get("notes", ""))
    source_id = source_id_for_row(row)
    extracted = Counter()
    failures: List[dict] = []

    slots: List[tuple[int, Path, int, str]]
    raw_path_text = ""
    if row["label"] == "real":
        raw_path_text = str(Path(row["source_path"]).parent.resolve())
        slots = [(slot, frame_path, frame_index, raw_path_text) for slot, frame_path, frame_index in chronological_real_slots(Path(row["source_path"]))]
    else:
        video_path = fake_video_path(manipulated_root, row, notes).resolve()
        if not video_path.exists():
            return [], extracted, [{"sample_id": row["sample_id"], "error": "missing_video", "video_path": str(video_path)}]
        subset = row["identity_id"].split("__", 1)[0]
        indices = sample_video_indices(video_path, 3)
        slots = []
        for slot, index in enumerate(indices):
            if slot == 0:
                frame_path = Path(row["source_path"]).resolve()
            else:
                frame_path = cached_fake_frame(frame_root, video_path, subset, slot, index, extracted)
            slots.append((slot, frame_path, index, str(video_path)))
        raw_path_text = str(video_path)

    output: List[Dict[str, str]] = []
    t_video_key = temporal_video_key(row, notes, raw_path_text)
    for slot, frame_path, frame_index, raw_ref in slots:
        frame_path = frame_path.resolve()
        if frame_path == Path(row["source_path"]).resolve():
            record_id = row["sample_id"]
            new_row = {field: row.get(field, "") for field in FIELDNAMES}
        else:
            record_id = f"{row['sample_id']}_tf{slot:02d}"
            new_row = {field: row.get(field, "") for field in FIELDNAMES}
            new_row["sample_id"] = record_id
            new_row["source_path"] = str(frame_path)
            new_row["passive_key"] = feature_path(passive_root, record_id, ".npz")
            if row["protection"] == "protected":
                new_row["planned_output_path"] = protected_output_path(derived_root, row["split"], row["label"], row, frame_path)
                new_row["provenance_key"] = feature_path(provenance_root, record_id, ".json")
            else:
                new_row["planned_output_path"] = str(frame_path)
                new_row["provenance_key"] = ""

        extra_notes = (
            f"temporal_benchmark={benchmark_name}; temporal_sample_id={row['sample_id']}; "
            f"temporal_video_key={t_video_key}; temporal_split_key={source_id}; "
            f"temporal_frame_slot={slot}; temporal_frame_index={frame_index}; temporal_frame_count=3; "
            f"raw_video_path={raw_ref}"
        )
        new_row["notes"] = f"{row.get('notes', '').rstrip()}; {extra_notes}" if row.get("notes") else extra_notes
        output.append(new_row)
    return output, extracted, failures


def build(args: argparse.Namespace) -> dict:
    source_rows = load_csv(Path(args.source_manifest))
    variants = {item.strip() for item in args.variants.split(",") if item.strip()}
    filtered = [row for row in source_rows if keep_row(row, variants)]
    calib_ids, heldout_ids = split_source_ids(filtered, args.seed, args.calib_fraction)

    calib_rows: List[Dict[str, str]] = []
    heldout_rows: List[Dict[str, str]] = []
    extracted_total = Counter()
    failures: List[dict] = []

    for row in filtered:
        expanded, extracted, row_failures = expand_row(
            row,
            manipulated_root=Path(args.manipulated_root),
            frame_root=Path(args.frame_root),
            derived_root=Path(args.derived_root),
            provenance_root=Path(args.provenance_root),
            passive_root=Path(args.passive_root),
            benchmark_name=args.benchmark_name,
        )
        extracted_total.update(extracted)
        failures.extend(row_failures)
        target = calib_rows if source_id_for_row(row) in calib_ids else heldout_rows
        target.extend(expanded)

    write_manifest(Path(args.calib_manifest), calib_rows)
    write_manifest(Path(args.heldout_manifest), heldout_rows)
    return {
        "benchmark_name": args.benchmark_name,
        "source_manifest": str(Path(args.source_manifest).resolve()),
        "source_rows": len(source_rows),
        "filtered_source_rows": len(filtered),
        "variants": sorted(variants) if variants else ["all"],
        "seed": args.seed,
        "calib_fraction": args.calib_fraction,
        "calib_source_ids": len(calib_ids),
        "heldout_source_ids": len(heldout_ids),
        "calib_rows": len(calib_rows),
        "heldout_rows": len(heldout_rows),
        "calib_groups": len({parse_notes(row.get("notes", "")).get("temporal_sample_id", row["sample_id"]) for row in calib_rows}),
        "heldout_groups": len({parse_notes(row.get("notes", "")).get("temporal_sample_id", row["sample_id"]) for row in heldout_rows}),
        "extracted": dict(extracted_total),
        "failure_count": len(failures),
        "failures": failures[:25],
        "calib_manifest": str(Path(args.calib_manifest).resolve()),
        "heldout_manifest": str(Path(args.heldout_manifest).resolve()),
        "frame_root": str(Path(args.frame_root).resolve()),
        "derived_root": str(Path(args.derived_root).resolve()),
        "provenance_root": str(Path(args.provenance_root).resolve()),
        "passive_root": str(Path(args.passive_root).resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a DeeperForensics/FF++ three-frame temporal manifest.")
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--calib-manifest", required=True)
    parser.add_argument("--heldout-manifest", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--manipulated-root", default="workspace/datasets/raw/DeeperForensics-1.0/manipulated_videos")
    parser.add_argument("--frame-root", default="workspace/datasets/derived/dtf3fs")
    parser.add_argument("--derived-root", default="workspace/datasets/derived/dtf3s")
    parser.add_argument("--provenance-root", default="workspace/features/provenance/dtf3s")
    parser.add_argument("--passive-root", default="workspace/features/passive/dtf3s")
    parser.add_argument("--benchmark-name", default="deeperforensics_ffpp_test_temporal_f3_v1")
    parser.add_argument("--variants", default="clean", help="Comma-separated protected variants to include, e.g. clean,jpeg_q75,resize_half. Empty means all.")
    parser.add_argument("--seed", type=int, default=20260422)
    parser.add_argument("--calib-fraction", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build(args)
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
