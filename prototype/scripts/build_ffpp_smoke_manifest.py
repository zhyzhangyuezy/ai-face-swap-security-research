from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path
from typing import Iterable, List


def _bootstrap_src() -> None:
    script_dir = Path(__file__).resolve().parent
    src_root = script_dir.parent / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


_bootstrap_src()

from hybrid_router.config_io import load_yaml_config
from hybrid_router.manifest_schema import SampleRecord, write_manifest


def discover_media_files(root: Path, extensions: Iterable[str]) -> List[Path]:
    normalized = {ext.lower() for ext in extensions}
    files = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in normalized]
    files.sort()
    return files


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


def sample_id_from_path(path: Path, label: str) -> str:
    digest = hashlib.sha1(f"{label}:{path.as_posix()}".encode("utf-8")).hexdigest()[:12]
    return f"{label[:1]}_{digest}"


def planned_output_path(derived_root: Path, split: str, label: str, protection: str, variant_name: str, source_path: Path) -> str:
    filename = source_path.name
    return str((derived_root / split / label / protection / variant_name / filename).resolve())


def build_records(config: dict) -> List[SampleRecord]:
    dataset_cfg = config["dataset"]
    derivations_cfg = config["derivations"]
    outputs_cfg = config["outputs"]

    real_root = Path(dataset_cfg["real_root"])
    fake_root = Path(dataset_cfg["fake_root"])
    extensions = dataset_cfg.get("extensions", [".png", ".jpg", ".jpeg"])
    max_real = int(dataset_cfg.get("max_real", 64))
    max_fake = int(dataset_cfg.get("max_fake", 64))
    split_ratios = dataset_cfg.get("split_ratios", {"train": 0.7, "val": 0.15, "test": 0.15})
    derived_root = Path(outputs_cfg["derived_root"])
    provenance_root = Path(outputs_cfg["provenance_feature_root"])
    passive_root = Path(outputs_cfg["passive_feature_root"])

    if not real_root.exists():
        raise FileNotFoundError(f"Real root does not exist: {real_root}")
    if not fake_root.exists():
        raise FileNotFoundError(f"Fake root does not exist: {fake_root}")

    real_files = discover_media_files(real_root, extensions)[:max_real]
    fake_files = discover_media_files(fake_root, extensions)[:max_fake]

    source_rows = [(path, "real") for path in real_files] + [(path, "fake") for path in fake_files]
    records: List[SampleRecord] = []

    for index, (source_path, label) in enumerate(source_rows):
        split = assign_split(index, len(source_rows), split_ratios)
        base_sample_id = sample_id_from_path(source_path, label)
        pair_group_id = f"pair_{base_sample_id}"

        if derivations_cfg.get("include_unprotected_clean", True):
            records.append(
                SampleRecord(
                    sample_id=f"{base_sample_id}_u_clean",
                    split=split,
                    label=label,
                    protection="unprotected",
                    degradation_type="none",
                    degradation_level="clean",
                    source_path=str(source_path.resolve()),
                    planned_output_path=str(source_path.resolve()),
                    identity_id="unknown",
                    provenance_key="",
                    passive_key=f"{base_sample_id}_u_clean",
                    pair_group_id=pair_group_id,
                    notes="clean unprotected baseline",
                )
            )

        for variant in derivations_cfg.get("protected_variants", []):
            variant_name = variant["name"]
            degradation_type = variant["degradation_type"]
            degradation_level = variant["degradation_level"]
            record_id = f"{base_sample_id}_p_{variant_name}"
            output_path = planned_output_path(derived_root, split, label, "protected", variant_name, source_path)
            records.append(
                SampleRecord(
                    sample_id=record_id,
                    split=split,
                    label=label,
                    protection="protected",
                    degradation_type=degradation_type,
                    degradation_level=degradation_level,
                    source_path=str(source_path.resolve()),
                    planned_output_path=output_path,
                    identity_id="unknown",
                    provenance_key=str((provenance_root / f"{record_id}.json").resolve()),
                    passive_key=str((passive_root / f"{record_id}.npz").resolve()),
                    pair_group_id=pair_group_id,
                    notes=f"protected variant {variant_name}",
                )
            )

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the canonical FaceForensics++ smoke-test manifest.")
    parser.add_argument("--config", required=True, help="Path to the smoke-test YAML config.")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    manifest_path = Path(config["paths"]["manifest_path"])
    records = build_records(config)
    write_manifest(manifest_path, records)

    print(f"Wrote {len(records)} manifest rows to {manifest_path}")


if __name__ == "__main__":
    main()
