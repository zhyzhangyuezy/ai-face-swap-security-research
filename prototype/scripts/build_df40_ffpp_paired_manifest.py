from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def _bootstrap_src() -> None:
    script_dir = Path(__file__).resolve().parent
    src_root = script_dir.parent / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


_bootstrap_src()

from hybrid_router.config_io import load_yaml_config
from hybrid_router.manifest_schema import SampleRecord, write_manifest


def discover_video_dirs(root: Path) -> List[Path]:
    video_dirs = [path for path in root.iterdir() if path.is_dir()]
    video_dirs.sort()
    return video_dirs


def discover_media_files(root: Path, extensions: Iterable[str]) -> List[Path]:
    normalized = {ext.lower() for ext in extensions}
    files = [path for path in root.iterdir() if path.is_file() and path.suffix.lower() in normalized]
    files.sort()
    return files


def select_frame(files: Sequence[Path], selector: str) -> Path:
    if not files:
        raise ValueError("Cannot select a frame from an empty file list.")
    if selector == "first":
        return files[0]
    if selector == "last":
        return files[-1]
    if selector == "middle":
        return files[len(files) // 2]
    raise ValueError(f"Unsupported frame selector: {selector}")


def assign_split(index: int, total: int, ratios: dict) -> str:
    if total <= 0:
        return "train"
    train_cut = math.floor(total * float(ratios.get("train", 0.7)))
    val_cut = train_cut + math.floor(total * float(ratios.get("val", 0.15)))
    if index < train_cut:
        return "train"
    if index < val_cut:
        return "val"
    return "test"


def stable_sample_id(namespace: str, value: str) -> str:
    digest = hashlib.sha1(f"{namespace}:{value}".encode("utf-8")).hexdigest()[:12]
    prefix = "r" if namespace.endswith(":real") else "f"
    return f"{prefix}_{digest}"


def parse_pair_name(name: str) -> Tuple[str, str] | None:
    parts = name.split("_")
    if len(parts) != 2:
        return None
    if not all(part.isdigit() for part in parts):
        return None
    return parts[0], parts[1]


def normalized_pair_key(left: str, right: str) -> str:
    return "_".join(sorted([left, right]))


def load_allowed_pair_keys(path: Path) -> set[str]:
    pairs = json.loads(path.read_text(encoding="utf-8"))
    allowed: set[str] = set()
    for item in pairs:
        if not isinstance(item, list) or len(item) != 2:
            continue
        left, right = str(item[0]), str(item[1])
        allowed.add(normalized_pair_key(left, right))
    return allowed


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


def build_records(config: dict) -> tuple[List[SampleRecord], dict]:
    dataset_cfg = config["dataset"]
    outputs_cfg = config["outputs"]
    derivations_cfg = config["derivations"]

    method = str(dataset_cfg["method"])
    ffpp_real_root = Path(dataset_cfg["ffpp_real_root"])
    fake_root = Path(dataset_cfg["fake_root"])
    extensions = dataset_cfg.get("extensions", [".png", ".jpg", ".jpeg"])
    split_ratios = dataset_cfg.get("split_ratios", {"train": 0.7, "val": 0.15, "test": 0.15})
    fake_selector = dataset_cfg.get("fake_frame_selector", "middle")
    real_selector = dataset_cfg.get("real_frame_selector", "middle")
    max_pairs = int(dataset_cfg.get("max_pairs", 0))
    source_token_index = int(dataset_cfg.get("source_token_index", 0))
    pair_split_json_value = str(dataset_cfg.get("ffpp_pair_split_json", "")).strip()
    pair_split_name = str(dataset_cfg.get("ffpp_pair_split_name", "")).strip() or "all"

    derived_root = Path(outputs_cfg["derived_root"])
    provenance_root = Path(outputs_cfg["provenance_feature_root"])
    passive_root = Path(outputs_cfg["passive_feature_root"])

    if not ffpp_real_root.exists():
        raise FileNotFoundError(f"FF++ real root does not exist: {ffpp_real_root}")
    if not fake_root.exists():
        raise FileNotFoundError(f"DF40 fake root does not exist: {fake_root}")

    allowed_pair_keys: set[str] | None = None
    if pair_split_json_value:
        pair_split_json = Path(pair_split_json_value)
        if not pair_split_json.exists():
            raise FileNotFoundError(f"FF++ split JSON does not exist: {pair_split_json}")
        allowed_pair_keys = load_allowed_pair_keys(pair_split_json)

    source_records: Dict[str, dict] = {}
    fake_entries_by_source: Dict[str, List[dict]] = defaultdict(list)
    skipped_bad_name = 0
    skipped_split_filter = 0
    skipped_missing_real_dir = 0
    skipped_missing_real_frames = 0
    skipped_missing_fake_frames = 0
    kept_pairs = 0
    pair_examples: List[str] = []
    missing_real_examples: List[str] = []

    for fake_video_dir in discover_video_dirs(fake_root):
        parsed = parse_pair_name(fake_video_dir.name)
        if parsed is None:
            skipped_bad_name += 1
            continue
        left_id, right_id = parsed
        if source_token_index not in {0, 1}:
            raise ValueError("source_token_index must be 0 or 1.")
        source_id = [left_id, right_id][source_token_index]
        target_id = [right_id, left_id][source_token_index]

        if allowed_pair_keys is not None and normalized_pair_key(left_id, right_id) not in allowed_pair_keys:
            skipped_split_filter += 1
            continue

        fake_frames = discover_media_files(fake_video_dir, extensions)
        if not fake_frames:
            skipped_missing_fake_frames += 1
            continue

        real_video_dir = ffpp_real_root / source_id
        if not real_video_dir.exists():
            skipped_missing_real_dir += 1
            if len(missing_real_examples) < 10:
                missing_real_examples.append(fake_video_dir.name)
            continue
        real_frames = discover_media_files(real_video_dir, extensions)
        if not real_frames:
            skipped_missing_real_frames += 1
            if len(missing_real_examples) < 10:
                missing_real_examples.append(fake_video_dir.name)
            continue

        if source_id not in source_records:
            real_frame = select_frame(real_frames, real_selector)
            source_records[source_id] = {
                "source_id": source_id,
                "real_frame": real_frame,
            }

        fake_frame = select_frame(fake_frames, fake_selector)
        fake_entries_by_source[source_id].append(
            {
                "pair_name": fake_video_dir.name,
                "source_id": source_id,
                "target_id": target_id,
                "fake_frame": fake_frame,
            }
        )
        kept_pairs += 1
        if len(pair_examples) < 10:
            pair_examples.append(fake_video_dir.name)
        if max_pairs > 0 and kept_pairs >= max_pairs:
            break

    ordered_source_ids = sorted(source_records)
    split_by_source = {
        source_id: assign_split(index, len(ordered_source_ids), split_ratios)
        for index, source_id in enumerate(ordered_source_ids)
    }

    records: List[SampleRecord] = []
    real_source_count = 0
    fake_pair_count = 0
    for source_id in ordered_source_ids:
        source_info = source_records[source_id]
        split = split_by_source[source_id]
        real_frame = Path(source_info["real_frame"])
        real_identity = source_id
        real_base = stable_sample_id(f"df40ffpp:{method}:real", f"{source_id}:{real_frame.as_posix()}")
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
                notes=(
                    f"df40-ffpp paired real source; method={method}; ffpp_pair_split={pair_split_name}; "
                    "source_identity assumed from first token convention."
                ),
            )
        )
        for variant in derivations_cfg.get("protected_variants", []):
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
                    notes=(
                        f"df40-ffpp paired protected real; method={method}; source_id={source_id}; "
                        f"ffpp_pair_split={pair_split_name}"
                    ),
                )
            )
        real_source_count += 1

        for fake_info in fake_entries_by_source.get(source_id, []):
            pair_name = str(fake_info["pair_name"])
            fake_frame = Path(fake_info["fake_frame"])
            target_id = str(fake_info["target_id"])
            fake_base = stable_sample_id(f"df40ffpp:{method}:fake", f"{pair_name}:{fake_frame.as_posix()}")
            fake_pair_group = f"pair_{fake_base}"
            records.append(
                SampleRecord(
                    sample_id=f"{fake_base}_u_clean",
                    split=split,
                    label="fake",
                    protection="unprotected",
                    degradation_type="none",
                    degradation_level="clean",
                    source_path=str(fake_frame.resolve()),
                    planned_output_path=str(fake_frame.resolve()),
                    identity_id=pair_name,
                    provenance_key="",
                    passive_key=passive_feature_path(passive_root, f"{fake_base}_u_clean"),
                    pair_group_id=fake_pair_group,
                    notes=(
                        f"df40-ffpp paired fake; method={method}; source_id={source_id}; target_id={target_id}; "
                        f"ffpp_pair_split={pair_split_name}; source_identity assumed from first token."
                    ),
                )
            )
            for variant in derivations_cfg.get("protected_variants", []):
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
                        source_path=str(fake_frame.resolve()),
                        planned_output_path=protected_output_path(derived_root, split, "fake", variant_name, pair_name, fake_frame),
                        identity_id=pair_name,
                        provenance_key=provenance_feature_path(provenance_root, record_id),
                        passive_key=passive_feature_path(passive_root, record_id),
                        pair_group_id=fake_pair_group,
                        notes=(
                            f"df40-ffpp paired protected fake; method={method}; source_id={source_id}; "
                            f"target_id={target_id}; ffpp_pair_split={pair_split_name}"
                        ),
                    )
                )
            fake_pair_count += 1

    summary = {
        "method": method,
        "pair_split_name": pair_split_name,
        "ffpp_real_root": str(ffpp_real_root),
        "fake_root": str(fake_root),
        "rows": len(records),
        "real_sources": real_source_count,
        "fake_pairs": fake_pair_count,
        "kept_pairs": kept_pairs,
        "split_counts": {
            split: sum(1 for source_id in ordered_source_ids if split_by_source[source_id] == split)
            for split in sorted(set(split_by_source.values()))
        },
        "skipped_bad_name": skipped_bad_name,
        "skipped_split_filter": skipped_split_filter,
        "skipped_missing_real_dir": skipped_missing_real_dir,
        "skipped_missing_real_frames": skipped_missing_real_frames,
        "skipped_missing_fake_frames": skipped_missing_fake_frames,
        "pair_examples": pair_examples,
        "missing_real_examples": missing_real_examples,
    }
    return records, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a paired DF40-method manifest by attaching fake-only FF-domain methods to local FF++ real sources."
    )
    parser.add_argument("--config", required=True, help="Path to the paired-manifest YAML config.")
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

    print(json.dumps({"manifest_path": str(manifest_path), **summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
