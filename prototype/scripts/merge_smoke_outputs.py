from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List


def _bootstrap_src() -> None:
    script_dir = Path(__file__).resolve().parent
    src_root = script_dir.parent / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


_bootstrap_src()

from hybrid_router.config_io import load_yaml_config
from hybrid_router.manifest_schema import load_manifest


def load_optional_csv(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = {}
        for row in reader:
            sample_id = row.get("sample_id")
            if sample_id:
                rows[sample_id] = row
        return rows


def merge_rows(
    manifest_rows: List[Dict[str, str]],
    provenance_rows: Dict[str, Dict[str, str]],
    passive_rows: Dict[str, Dict[str, str]],
) -> List[Dict[str, str]]:
    merged: List[Dict[str, str]] = []
    for row in manifest_rows:
        sample_id = row["sample_id"]
        out = dict(row)
        for key, value in provenance_rows.get(sample_id, {}).items():
            if key != "sample_id":
                out[f"prov_{key}"] = value
        for key, value in passive_rows.get(sample_id, {}).items():
            if key != "sample_id":
                out[f"passive_{key}"] = value
        merged.append(out)
    return merged


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge smoke-test manifest, provenance exports, and passive-detector exports.")
    parser.add_argument("--config", required=True, help="Path to the smoke-test YAML config.")
    parser.add_argument("--manifest", help="Optional explicit manifest path.")
    parser.add_argument("--provenance-csv", help="Optional provenance result CSV.")
    parser.add_argument("--passive-csv", help="Optional passive result CSV.")
    parser.add_argument("--output", help="Optional merged output CSV.")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    manifest_path = Path(args.manifest or config["paths"]["manifest_path"])
    provenance_csv = Path(args.provenance_csv or config["paths"]["provenance_jobs_path"])
    passive_csv = Path(args.passive_csv or config["paths"]["passive_jobs_path"])
    output_path = Path(args.output or config["paths"]["merged_report_path"])

    manifest_rows = load_manifest(manifest_path)
    provenance_rows = load_optional_csv(provenance_csv)
    passive_rows = load_optional_csv(passive_csv)
    merged = merge_rows(manifest_rows, provenance_rows, passive_rows)
    write_csv(output_path, merged)

    print(f"Wrote merged smoke-test table with {len(merged)} rows to {output_path}")


if __name__ == "__main__":
    main()
