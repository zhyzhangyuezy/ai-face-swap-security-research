from __future__ import annotations

import argparse
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

from hybrid_router.config_io import load_yaml_config
from hybrid_router.manifest_schema import SampleRecord, write_manifest


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def discover_media_files(root: Path, extensions: Iterable[str]) -> List[Path]:
    normalized = {ext.lower() for ext in extensions}
    files = [path for path in root.iterdir() if path.is_file() and path.suffix.lower() in normalized]
    files.sort()
    return files


def discover_video_files(root: Path, extensions: Iterable[str]) -> List[Path]:
    normalized = {ext.lower() for ext in extensions}
    files = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in normalized]
    files.sort()
    return files


def select_frame(files: List[Path], selector: str) -> Path:
    if not files:
        raise ValueError("Cannot select a frame from an empty file list.")
    if selector == "first":
        return files[0]
    if selector == "last":
        return files[-1]
    if selector == "middle":
        return files[len(files) // 2]
    raise ValueError(f"Unsupported frame selector: {selector}")


def extract_first_frame(video_path: Path, dst_png: Path) -> str:
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise RuntimeError("imageio is required for DeeperForensics frame extraction.") from exc

    dst_png.parent.mkdir(parents=True, exist_ok=True)
    if dst_png.exists():
        return "exists"
    frame = iio.imread(video_path, index=0)
    iio.imwrite(dst_png, frame)
    return "extract_frame"


def stable_sample_id(namespace: str, value: str) -> str:
    digest = hashlib.sha1(f"{namespace}:{value}".encode("utf-8")).hexdigest()[:12]
    prefix = "r" if namespace.endswith(":real") else "f"
    return f"{prefix}_{digest}"


def protected_output_path(
    derived_root: Path,
    split: str,
    label: str,
    variant_name: str,
    identity_id: str,
    source_path: Path,
) -> str:
    return str((derived_root / split / label / "protected" / variant_name / identity_id / source_path.name).resolve())


def passive_feature_path(passive_root: Path, record_id: str) -> str:
    return str((passive_root / f"{record_id}.npz").resolve())


def provenance_feature_path(provenance_root: Path, record_id: str) -> str:
    return str((provenance_root / f"{record_id}.json").resolve())


def load_split_ids(splits_root: Path, split_name: str) -> Dict[str, str]:
    if split_name == "all":
        mapping: Dict[str, str] = {}
        for candidate in ["train", "val", "test"]:
            mapping.update(load_split_ids(splits_root, candidate))
        return mapping

    split_path = splits_root / f"{split_name}.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"DeeperForensics split file does not exist: {split_path}")

    mapping: Dict[str, str] = {}
    for raw_line in split_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        source_id = line.split("_", 1)[0]
        mapping[source_id] = split_name
    return mapping


def parse_source_id(name: str) -> str | None:
    parts = name.split("_", 1)
    if len(parts) != 2:
        return None
    source_id = parts[0]
    return source_id if source_id.isdigit() else None


def subset_name_from_video(manipulated_root: Path, video_path: Path) -> str:
    relative = video_path.relative_to(manipulated_root)
    if len(relative.parts) < 2:
        raise ValueError(f"Unexpected manipulated video path layout: {video_path}")
    return relative.parts[0]


def build_records(config: dict) -> tuple[List[SampleRecord], dict]:
    dataset_cfg = config["dataset"]
    outputs_cfg = config["outputs"]
    derivations_cfg = config["derivations"]

    deeper_root = Path(dataset_cfg["deeperforensics_root"])
    manipulated_root = Path(dataset_cfg.get("manipulated_root", deeper_root / "manipulated_videos"))
    splits_root = Path(dataset_cfg.get("splits_root", deeper_root / "lists" / "splits"))
    ffpp_real_root = Path(dataset_cfg["ffpp_real_root"])
    fake_frame_root = Path(dataset_cfg["fake_frame_root"])
    split_name = str(dataset_cfg.get("split_name", "test")).strip() or "test"
    subset_names = list(dataset_cfg.get("subset_names", []))
    video_extensions = dataset_cfg.get("video_extensions", sorted(VIDEO_EXTS))
    image_extensions = dataset_cfg.get("image_extensions", sorted(IMAGE_EXTS))
    max_fake_videos = int(dataset_cfg.get("max_fake_videos", 0))
    max_fake_per_subset = int(dataset_cfg.get("max_fake_per_subset", 0))
    real_selector = str(dataset_cfg.get("real_frame_selector", "middle"))

    derived_root = Path(outputs_cfg["derived_root"])
    provenance_root = Path(outputs_cfg["provenance_feature_root"])
    passive_root = Path(outputs_cfg["passive_feature_root"])

    if not deeper_root.exists():
        raise FileNotFoundError(f"DeeperForensics root does not exist: {deeper_root}")
    if not manipulated_root.exists():
        raise FileNotFoundError(f"DeeperForensics manipulated root does not exist: {manipulated_root}")
    if not ffpp_real_root.exists():
        raise FileNotFoundError(f"FF++ real root does not exist: {ffpp_real_root}")

    split_id_to_name = load_split_ids(splits_root, split_name)
    allowed_ids = set(split_id_to_name)
    real_frame_by_source: Dict[str, Path] = {}
    records: List[SampleRecord] = []

    extracted_new = 0
    extracted_existing = 0
    skipped_bad_name = 0
    skipped_outside_split = 0
    skipped_missing_real_dir = 0
    skipped_missing_real_frames = 0
    subset_counter: Counter[str] = Counter()
    pair_examples: List[str] = []

    source_ids_with_fake: set[str] = set()
    fake_entries: List[dict] = []

    video_files = discover_video_files(manipulated_root, video_extensions)
    for video_path in video_files:
        subset_name = subset_name_from_video(manipulated_root, video_path)
        if subset_names and subset_name not in subset_names:
            continue

        source_id = parse_source_id(video_path.stem)
        if source_id is None:
            skipped_bad_name += 1
            continue
        if source_id not in allowed_ids:
            skipped_outside_split += 1
            continue
        if max_fake_per_subset > 0 and subset_counter[subset_name] >= max_fake_per_subset:
            continue
        if max_fake_videos > 0 and len(fake_entries) >= max_fake_videos:
            break

        real_dir = ffpp_real_root / source_id
        if not real_dir.exists():
            skipped_missing_real_dir += 1
            continue
        real_frames = discover_media_files(real_dir, image_extensions)
        if not real_frames:
            skipped_missing_real_frames += 1
            continue
        if source_id not in real_frame_by_source:
            real_frame_by_source[source_id] = select_frame(real_frames, real_selector)

        extracted_path = (fake_frame_root / subset_name / f"{video_path.stem}.png").resolve()
        extraction_action = extract_first_frame(video_path, extracted_path)
        if extraction_action == "exists":
            extracted_existing += 1
        else:
            extracted_new += 1

        fake_entries.append(
            {
                "source_id": source_id,
                "split": split_id_to_name[source_id],
                "subset_name": subset_name,
                "video_path": video_path.resolve(),
                "frame_path": extracted_path,
                "video_stem": video_path.stem,
            }
        )
        subset_counter[subset_name] += 1
        source_ids_with_fake.add(source_id)
        if len(pair_examples) < 12:
            pair_examples.append(str(video_path.relative_to(manipulated_root)).replace("\\", "/"))

    protected_variants = derivations_cfg.get("protected_variants", [])
    ordered_source_ids = sorted(source_ids_with_fake)

    for source_id in ordered_source_ids:
        split = split_id_to_name[source_id]
        real_frame = real_frame_by_source[source_id]
        real_identity = source_id
        real_base = stable_sample_id(f"deeperforensics:{split}:real", f"{source_id}:{real_frame.as_posix()}")
        real_pair_group = f"pair_{real_base}"

        records.append(
            SampleRecord(
                sample_id=f"{real_base}_u_clean",
                split=split,
                label="real",
                protection="unprotected",
                degradation_type="none",
                degradation_level="clean",
                source_path=str(real_frame.resolve()),
                planned_output_path=str(real_frame.resolve()),
                identity_id=real_identity,
                provenance_key="",
                passive_key=passive_feature_path(passive_root, f"{real_base}_u_clean"),
                pair_group_id=real_pair_group,
                notes=f"DeeperForensics real anchor reused from local FF++; source_id={source_id}; split={split}",
            )
        )
        for variant in protected_variants:
            variant_name = variant["name"]
            record_id = f"{real_base}_p_{variant_name}"
            records.append(
                SampleRecord(
                    sample_id=record_id,
                    split=split,
                    label="real",
                    protection="protected",
                    degradation_type=variant["degradation_type"],
                    degradation_level=variant["degradation_level"],
                    source_path=str(real_frame.resolve()),
                    planned_output_path=protected_output_path(derived_root, split, "real", variant_name, real_identity, real_frame),
                    identity_id=real_identity,
                    provenance_key=provenance_feature_path(provenance_root, record_id),
                    passive_key=passive_feature_path(passive_root, record_id),
                    pair_group_id=real_pair_group,
                    notes=f"DeeperForensics protected real anchor; source_id={source_id}; split={split}",
                )
            )

    for fake_info in fake_entries:
        split = fake_info["split"]
        subset_name = fake_info["subset_name"]
        source_id = fake_info["source_id"]
        frame_path = Path(fake_info["frame_path"])
        video_path = Path(fake_info["video_path"])
        video_stem = str(fake_info["video_stem"])
        identity_id = f"{subset_name}__{video_stem}"
        fake_base = stable_sample_id(
            f"deeperforensics:{split}:fake",
            f"{subset_name}:{video_stem}:{video_path.as_posix()}",
        )
        fake_pair_group = f"pair_{fake_base}"

        records.append(
            SampleRecord(
                sample_id=f"{fake_base}_u_clean",
                split=split,
                label="fake",
                protection="unprotected",
                degradation_type="none",
                degradation_level="clean",
                source_path=str(frame_path.resolve()),
                planned_output_path=str(frame_path.resolve()),
                identity_id=identity_id,
                provenance_key="",
                passive_key=passive_feature_path(passive_root, f"{fake_base}_u_clean"),
                pair_group_id=fake_pair_group,
                notes=(
                    f"DeeperForensics fake frame extracted from manipulated video; source_id={source_id}; "
                    f"subset={subset_name}; split={split}; video_path={video_path}"
                ),
            )
        )
        for variant in protected_variants:
            variant_name = variant["name"]
            record_id = f"{fake_base}_p_{variant_name}"
            records.append(
                SampleRecord(
                    sample_id=record_id,
                    split=split,
                    label="fake",
                    protection="protected",
                    degradation_type=variant["degradation_type"],
                    degradation_level=variant["degradation_level"],
                    source_path=str(frame_path.resolve()),
                    planned_output_path=protected_output_path(
                        derived_root,
                        split,
                        "fake",
                        variant_name,
                        identity_id,
                        frame_path,
                    ),
                    identity_id=identity_id,
                    provenance_key=provenance_feature_path(provenance_root, record_id),
                    passive_key=passive_feature_path(passive_root, record_id),
                    pair_group_id=fake_pair_group,
                    notes=(
                        f"DeeperForensics protected fake frame; source_id={source_id}; "
                        f"subset={subset_name}; split={split}"
                    ),
                )
            )

    summary = {
        "split_name": split_name,
        "rows": len(records),
        "real_sources": len(ordered_source_ids),
        "fake_videos": len(fake_entries),
        "available_split_ids": len(allowed_ids),
        "subset_counts": dict(sorted(subset_counter.items())),
        "extracted_new": extracted_new,
        "extracted_existing": extracted_existing,
        "skipped_bad_name": skipped_bad_name,
        "skipped_outside_split": skipped_outside_split,
        "skipped_missing_real_dir": skipped_missing_real_dir,
        "skipped_missing_real_frames": skipped_missing_real_frames,
        "pair_examples": pair_examples,
        "fake_frame_root": str(fake_frame_root.resolve()),
        "ffpp_real_root": str(ffpp_real_root.resolve()),
        "manipulated_root": str(manipulated_root.resolve()),
    }
    return records, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a split-aware DeeperForensics deployment-stress manifest using local FF++ real anchors."
    )
    parser.add_argument("--config", required=True, help="Path to the DeeperForensics bridge YAML config.")
    parser.add_argument("--summary-output", default="", help="Optional JSON summary output path.")
    args = parser.parse_args()

    config = load_yaml_config(Path(args.config))
    manifest_path = Path(config["paths"]["manifest_path"])
    records, summary = build_records(config)
    write_manifest(manifest_path, records)

    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({"manifest_path": str(manifest_path.resolve()), **summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
