from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, List


def _bootstrap_src() -> None:
    script_dir = Path(__file__).resolve().parent
    src_root = script_dir.parent / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


_bootstrap_src()

from hybrid_router.manifest_schema import SampleRecord, write_manifest


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


def select_frame(frames: list[str], selector: str) -> str:
    if not frames:
        raise ValueError("Cannot select from an empty frame list.")
    if selector == "first":
        return frames[0]
    if selector == "last":
        return frames[-1]
    if selector == "middle":
        return frames[len(frames) // 2]
    raise ValueError(f"Unsupported frame selector: {selector}")


def assign_split(index: int, total: int, ratios: dict[str, float]) -> str:
    if total <= 0:
        return "train"
    train_cut = math.floor(total * float(ratios.get("train", 0.7)))
    val_cut = train_cut + math.floor(total * float(ratios.get("val", 0.15)))
    if index < train_cut:
        return "train"
    if index < val_cut:
        return "val"
    return "test"


def split_from_metadata_path(path_parts: Iterable[str], fallback_index: int, total: int, ratios: dict[str, float]) -> str:
    normalized = {"valid": "val", "validation": "val"}
    for part in path_parts:
        lowered = part.lower()
        if lowered in {"train", "test", "val", "valid", "validation"}:
            return normalized.get(lowered, lowered)
    return assign_split(fallback_index, total, ratios)


def infer_label(raw_label: str, real_labels: set[str], fake_labels: set[str]) -> str | None:
    if raw_label in real_labels:
        return "real"
    if raw_label in fake_labels:
        return "fake"

    lowered = raw_label.lower()
    if "real" in lowered or lowered in {"ff-real", "original", "target", "targets"}:
        return "real"

    fake_tokens = [
        "fake",
        "synthesis",
        "ff-df",
        "ff-f2f",
        "ff-fs",
        "ff-nt",
        "faceshifter",
        "face2face",
        "faceswap",
        "deepfakes",
        "neuraltextures",
        "dfd",
    ]
    if any(token in lowered for token in fake_tokens):
        return "fake"
    return None


def candidate_paths(dataset_roots: list[Path], relative_value: str) -> list[Path]:
    rel_path = normalize_relative_path(relative_value)
    if not rel_path.parts:
        return []
    rel_tail = Path(*rel_path.parts[1:]) if len(rel_path.parts) > 1 else rel_path

    candidates: list[Path] = []
    seen: set[str] = set()
    for root in dataset_roots:
        for suffix in (rel_path, rel_tail):
            candidate = root / suffix
            key = str(candidate)
            if key not in seen:
                seen.add(key)
                candidates.append(candidate)
    return candidates


def resolve_existing_frame(dataset_roots: list[Path], relative_value: str) -> Path | None:
    return next((path for path in candidate_paths(dataset_roots, relative_value) if path.exists()), None)


def sample_id_from_path(path: Path, label: str, dataset_name: str) -> str:
    digest = hashlib.sha1(f"{dataset_name}:{label}:{path.as_posix()}".encode("utf-8")).hexdigest()[:12]
    return f"{label[:1]}_{digest}"


def planned_output_path(derived_root: Path, split: str, label: str, protection: str, variant_name: str, source_path: Path) -> str:
    return str((derived_root / split / label / protection / variant_name / source_path.name).resolve())


def parse_label_set(values: list[str] | None) -> set[str]:
    if not values:
        return set()
    labels: set[str] = set()
    for value in values:
        labels.update(part.strip() for part in value.split(",") if part.strip())
    return labels


def build_records(
    metadata_json: Path,
    dataset_roots: list[Path],
    dataset_name: str,
    max_real_videos: int,
    max_fake_videos: int,
    frame_selector: str,
    split_ratios: dict[str, float],
    derived_root: Path,
    provenance_root: Path,
    passive_root: Path,
    real_labels: set[str],
    fake_labels: set[str],
) -> tuple[List[SampleRecord], dict[str, Any]]:
    data = json.loads(metadata_json.read_text(encoding="utf-8"))
    real_count = 0
    fake_count = 0
    skipped_missing = 0
    skipped_unknown_label = 0
    missing_examples: list[str] = []
    unknown_labels: dict[str, int] = {}
    selected_sources: list[tuple[tuple[str, ...], dict[str, Any], Path, str]] = []

    entries = list(iter_video_entries(data))
    for entry_path, entry in entries:
        raw_label = str(entry.get("label", "unknown"))
        label = infer_label(raw_label, real_labels, fake_labels)
        if label is None:
            skipped_unknown_label += 1
            unknown_labels[raw_label] = unknown_labels.get(raw_label, 0) + 1
            continue
        if label == "real" and real_count >= max_real_videos:
            continue
        if label == "fake" and fake_count >= max_fake_videos:
            continue

        frames = [str(frame) for frame in entry.get("frames", [])]
        if not frames:
            skipped_missing += 1
            continue
        selected_relative = select_frame(frames, frame_selector)
        source_path = resolve_existing_frame(dataset_roots, selected_relative)
        if source_path is None:
            skipped_missing += 1
            if len(missing_examples) < 10:
                missing_examples.append(selected_relative)
            continue

        selected_sources.append((entry_path, entry, source_path.resolve(), label))
        if label == "real":
            real_count += 1
        else:
            fake_count += 1
        if real_count >= max_real_videos and fake_count >= max_fake_videos:
            break

    records: List[SampleRecord] = []
    for index, (entry_path, entry, source_path, label) in enumerate(selected_sources):
        split = split_from_metadata_path(entry_path, index, len(selected_sources), split_ratios)
        base_sample_id = sample_id_from_path(source_path, label, dataset_name)
        pair_group_id = f"pair_{base_sample_id}"
        identity_id = source_path.parent.name

        records.append(
            SampleRecord(
                sample_id=f"{base_sample_id}_u_clean",
                split=split,
                label=label,
                protection="unprotected",
                degradation_type="none",
                degradation_level="clean",
                source_path=str(source_path),
                planned_output_path=str(source_path),
                identity_id=identity_id,
                provenance_key="",
                passive_key=f"{base_sample_id}_u_clean",
                pair_group_id=pair_group_id,
                notes=f"{dataset_name} metadata clean sample; source label={entry.get('label', '')}",
            )
        )

        for variant_name, degradation_type, degradation_level in [
            ("clean", "none", "clean"),
            ("jpeg_q75", "jpeg", "q75"),
            ("resize_half", "resize", "0.5x"),
        ]:
            record_id = f"{base_sample_id}_p_{variant_name}"
            records.append(
                SampleRecord(
                    sample_id=record_id,
                    split=split,
                    label=label,
                    protection="protected",
                    degradation_type=degradation_type,
                    degradation_level=degradation_level,
                    source_path=str(source_path),
                    planned_output_path=planned_output_path(derived_root, split, label, "protected", variant_name, source_path),
                    identity_id=identity_id,
                    provenance_key=str((provenance_root / f"{record_id}.json").resolve()),
                    passive_key=str((passive_root / f"{record_id}.npz").resolve()),
                    pair_group_id=pair_group_id,
                    notes=f"{dataset_name} metadata protected variant {variant_name}; source label={entry.get('label', '')}",
                )
            )

    summary = {
        "metadata_json": str(metadata_json),
        "dataset_roots": [str(root) for root in dataset_roots],
        "dataset_name": dataset_name,
        "metadata_videos": len(entries),
        "selected_videos": len(selected_sources),
        "selected_real_videos": real_count,
        "selected_fake_videos": fake_count,
        "manifest_rows": len(records),
        "skipped_missing": skipped_missing,
        "skipped_unknown_label": skipped_unknown_label,
        "missing_examples": missing_examples,
        "unknown_labels": unknown_labels,
    }
    return records, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a smoke manifest from a DeepfakeBench dataset_json file.")
    parser.add_argument("--metadata-json", required=True)
    parser.add_argument("--dataset-root", action="append", required=True, help="May be provided multiple times.")
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", default="")
    parser.add_argument("--max-real-videos", type=int, default=16)
    parser.add_argument("--max-fake-videos", type=int, default=16)
    parser.add_argument("--frame-selector", choices=["first", "middle", "last"], default="middle")
    parser.add_argument("--real-label", action="append", default=[])
    parser.add_argument("--fake-label", action="append", default=[])
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()

    metadata_json = Path(args.metadata_json)
    dataset_roots = [Path(value) for value in args.dataset_root]
    dataset_name = args.dataset_name or metadata_json.stem
    output_path = Path(args.output)
    summary_path = Path(args.summary_output) if args.summary_output else output_path.with_suffix(".summary.json")

    project = Path(__file__).resolve().parents[2]
    derived_root = project / "workspace" / "datasets" / "derived" / f"{output_path.stem}_protected"
    provenance_root = project / "workspace" / "features" / "provenance" / output_path.stem
    passive_root = project / "workspace" / "features" / "passive" / output_path.stem

    records, summary = build_records(
        metadata_json=metadata_json,
        dataset_roots=dataset_roots,
        dataset_name=dataset_name,
        max_real_videos=args.max_real_videos,
        max_fake_videos=args.max_fake_videos,
        frame_selector=args.frame_selector,
        split_ratios={"train": 0.7, "val": 0.15, "test": 0.15},
        derived_root=derived_root,
        provenance_root=provenance_root,
        passive_root=passive_root,
        real_labels=parse_label_set(args.real_label),
        fake_labels=parse_label_set(args.fake_label),
    )

    if not records and not args.allow_empty:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        raise SystemExit(
            "No manifest rows were built. Check dataset roots, labels, or use --allow-empty "
            f"after inspecting {summary_path}."
        )

    write_manifest(output_path, records)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"manifest": str(output_path), "summary": str(summary_path), **summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
