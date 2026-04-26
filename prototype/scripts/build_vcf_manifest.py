from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _bootstrap_src() -> None:
    script_dir = Path(__file__).resolve().parent
    src_root = script_dir.parent / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


_bootstrap_src()

from hybrid_router.config_io import load_yaml_config
from hybrid_router.manifest_schema import SampleRecord, write_manifest


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def discover_video_files(root: Path, extensions: Iterable[str]) -> List[Path]:
    normalized = {ext.lower() for ext in extensions}
    files = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in normalized]
    files.sort()
    return files


def extract_first_frame(video_path: Path, dst_png: Path) -> str:
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise RuntimeError("imageio is required for VCF frame extraction.") from exc

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


def stable_identity_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(f"{prefix}:{value}".encode("utf-8")).hexdigest()[:16]
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


def parse_video_layout(domain_root: Path, video_path: Path) -> Tuple[str, str, str, str, str]:
    relative = video_path.relative_to(domain_root)
    if len(relative.parts) < 5:
        raise ValueError(f"Unexpected VCF path layout: {video_path}")
    method, resolution, container, bucket, filename = relative.parts[:5]
    stem = Path(filename).stem
    return method, resolution, container, bucket, stem


def build_records(config: dict) -> tuple[List[SampleRecord], dict]:
    dataset_cfg = config["dataset"]
    outputs_cfg = config["outputs"]
    derivations_cfg = config["derivations"]

    vcf_root = Path(dataset_cfg["vcf_root"])
    domain_name = str(dataset_cfg.get("domain_name", "c23")).strip() or "c23"
    domain_root = Path(dataset_cfg.get("domain_root", vcf_root / domain_name))
    frame_root = Path(dataset_cfg["frame_root"])
    split_name = str(dataset_cfg.get("split_name", "test")).strip() or "test"
    fake_methods = list(dataset_cfg.get("fake_methods", []))
    resolution_names = list(dataset_cfg.get("resolution_names", []))
    bucket_names = list(dataset_cfg.get("bucket_names", []))
    video_extensions = dataset_cfg.get("video_extensions", sorted(VIDEO_EXTS))
    max_target_videos = int(dataset_cfg.get("max_target_videos", 0))
    max_fake_per_subset = int(dataset_cfg.get("max_fake_per_subset", 0))

    derived_root = Path(outputs_cfg["derived_root"])
    provenance_root = Path(outputs_cfg["provenance_feature_root"])
    passive_root = Path(outputs_cfg["passive_feature_root"])

    if not vcf_root.exists():
        raise FileNotFoundError(f"VCF root does not exist: {vcf_root}")
    if not domain_root.exists():
        raise FileNotFoundError(f"VCF domain root does not exist: {domain_root}")

    all_video_files = discover_video_files(domain_root, video_extensions)
    if not all_video_files:
        raise RuntimeError(f"No VCF videos found under: {domain_root}")

    discovered_methods = sorted({parse_video_layout(domain_root, path)[0] for path in all_video_files})
    if not fake_methods:
        fake_methods = [method for method in discovered_methods if method != "targets"]
    if "targets" not in discovered_methods:
        raise RuntimeError("VCF targets directory is missing; cannot build target/fake paired benchmark.")

    records: List[SampleRecord] = []
    extracted_new = 0
    extracted_existing = 0
    skipped_no_target = 0
    pair_examples: List[str] = []
    fake_subset_counter: Counter[str] = Counter()
    target_counter: Counter[str] = Counter()

    targets_by_key: Dict[Tuple[str, str, str], dict] = {}
    fake_entries: List[dict] = []

    for video_path in all_video_files:
        method, resolution, container, bucket, stem = parse_video_layout(domain_root, video_path)
        if container != "mp4":
            continue
        if resolution_names and resolution not in resolution_names:
            continue
        if bucket_names and bucket not in bucket_names:
            continue

        if method == "targets":
            key = (resolution, bucket, stem)
            targets_by_key[key] = {
                "resolution": resolution,
                "bucket": bucket,
                "stem": stem,
                "video_path": video_path.resolve(),
            }

    ordered_target_keys = sorted(targets_by_key)
    if max_target_videos > 0:
        ordered_target_keys = ordered_target_keys[:max_target_videos]
    allowed_target_keys = set(ordered_target_keys)

    for key in ordered_target_keys:
        target_info = targets_by_key[key]
        resolution = target_info["resolution"]
        bucket = target_info["bucket"]
        stem = target_info["stem"]
        target_path = Path(target_info["video_path"])
        extracted_path = (frame_root / domain_name / "targets" / resolution / bucket / f"{stem}.png").resolve()
        extraction_action = extract_first_frame(target_path, extracted_path)
        if extraction_action == "exists":
            extracted_existing += 1
        else:
            extracted_new += 1
        target_info["frame_path"] = extracted_path
        target_counter[f"{resolution}"] += 1

    for video_path in all_video_files:
        method, resolution, container, bucket, stem = parse_video_layout(domain_root, video_path)
        if container != "mp4" or method == "targets":
            continue
        if method not in fake_methods:
            continue
        if resolution_names and resolution not in resolution_names:
            continue
        if bucket_names and bucket not in bucket_names:
            continue

        key = (resolution, bucket, stem)
        if key not in allowed_target_keys:
            if key not in targets_by_key:
                skipped_no_target += 1
            continue

        subset_name = f"{method}__{resolution}"
        if max_fake_per_subset > 0 and fake_subset_counter[subset_name] >= max_fake_per_subset:
            continue

        extracted_path = (frame_root / domain_name / method / resolution / bucket / f"{stem}.png").resolve()
        extraction_action = extract_first_frame(video_path, extracted_path)
        if extraction_action == "exists":
            extracted_existing += 1
        else:
            extracted_new += 1

        fake_entries.append(
            {
                "method": method,
                "resolution": resolution,
                "bucket": bucket,
                "stem": stem,
                "subset_name": subset_name,
                "video_path": video_path.resolve(),
                "frame_path": extracted_path,
                "target": targets_by_key[key],
            }
        )
        fake_subset_counter[subset_name] += 1
        if len(pair_examples) < 12:
            pair_examples.append(str(video_path.relative_to(domain_root)).replace("\\", "/"))

    protected_variants = derivations_cfg.get("protected_variants", [])

    for key in ordered_target_keys:
        target_info = targets_by_key[key]
        resolution = target_info["resolution"]
        bucket = target_info["bucket"]
        stem = target_info["stem"]
        real_frame = Path(target_info["frame_path"])
        real_identity = stable_identity_id("vr", f"{domain_name}:{resolution}:{bucket}:{stem}")
        real_base = stable_sample_id("vcf:test:real", f"{domain_name}:{resolution}:{bucket}:{stem}:{real_frame.as_posix()}")
        real_pair_group = f"pair_{real_base}"

        records.append(
            SampleRecord(
                sample_id=f"{real_base}_u_clean",
                split=split_name,
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
                notes=(
                    f"VCF target real anchor; domain={domain_name}; resolution={resolution}; "
                    f"bucket={bucket}; stem={stem}"
                ),
            )
        )
        for variant in protected_variants:
            variant_name = variant["name"]
            record_id = f"{real_base}_p_{variant_name}"
            records.append(
                SampleRecord(
                    sample_id=record_id,
                    split=split_name,
                    label="real",
                    protection="protected",
                    degradation_type=variant["degradation_type"],
                    degradation_level=variant["degradation_level"],
                    source_path=str(real_frame.resolve()),
                    planned_output_path=protected_output_path(derived_root, split_name, "real", variant_name, real_identity, real_frame),
                    identity_id=real_identity,
                    provenance_key=provenance_feature_path(provenance_root, record_id),
                    passive_key=passive_feature_path(passive_root, record_id),
                    pair_group_id=real_pair_group,
                    notes=(
                        f"VCF protected real anchor; domain={domain_name}; resolution={resolution}; "
                        f"bucket={bucket}; stem={stem}"
                    ),
                )
            )

    for fake_info in fake_entries:
        method = fake_info["method"]
        resolution = fake_info["resolution"]
        bucket = fake_info["bucket"]
        stem = fake_info["stem"]
        subset_name = fake_info["subset_name"]
        frame_path = Path(fake_info["frame_path"])
        video_path = Path(fake_info["video_path"])
        identity_id = stable_identity_id("vf", f"{domain_name}:{method}:{resolution}:{bucket}:{stem}")
        fake_base = stable_sample_id(
            "vcf:test:fake",
            f"{domain_name}:{method}:{resolution}:{bucket}:{stem}:{video_path.as_posix()}",
        )
        fake_pair_group = f"pair_{fake_base}"

        records.append(
            SampleRecord(
                sample_id=f"{fake_base}_u_clean",
                split=split_name,
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
                    f"VCF fake frame extracted from video; domain={domain_name}; method={method}; "
                    f"resolution={resolution}; bucket={bucket}; stem={stem}; target_stem={stem}"
                ),
            )
        )
        for variant in protected_variants:
            variant_name = variant["name"]
            record_id = f"{fake_base}_p_{variant_name}"
            records.append(
                SampleRecord(
                    sample_id=record_id,
                    split=split_name,
                    label="fake",
                    protection="protected",
                    degradation_type=variant["degradation_type"],
                    degradation_level=variant["degradation_level"],
                    source_path=str(frame_path.resolve()),
                    planned_output_path=protected_output_path(
                        derived_root,
                        split_name,
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
                        f"VCF protected fake frame; domain={domain_name}; method={method}; "
                        f"resolution={resolution}; bucket={bucket}; stem={stem}"
                    ),
                )
            )

    summary = {
        "split_name": split_name,
        "domain_name": domain_name,
        "rows": len(records),
        "real_targets": len(ordered_target_keys),
        "fake_videos": len(fake_entries),
        "fake_methods": fake_methods,
        "resolutions": resolution_names or sorted({key[0] for key in ordered_target_keys}),
        "buckets": bucket_names or sorted({key[1] for key in ordered_target_keys}),
        "target_resolution_counts": dict(sorted(target_counter.items())),
        "subset_counts": dict(sorted(fake_subset_counter.items())),
        "extracted_new": extracted_new,
        "extracted_existing": extracted_existing,
        "skipped_no_target": skipped_no_target,
        "pair_examples": pair_examples,
        "frame_root": str(frame_root.resolve()),
        "domain_root": str(domain_root.resolve()),
    }
    return records, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a VCF deployment-stress manifest from internal target/fake video pairs.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--summary-output", default="", help="Optional path to write a JSON summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_yaml_config(config_path)

    records, summary = build_records(config)
    manifest_path = Path(config["paths"]["manifest_path"])
    write_manifest(manifest_path, records)

    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({"manifest_path": str(manifest_path.resolve()), **summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
