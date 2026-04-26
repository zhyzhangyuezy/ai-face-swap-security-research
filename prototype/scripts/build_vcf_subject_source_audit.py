#!/usr/bin/env python
"""Build a subject/source-family leakage audit for the VCF temporal manifests."""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = PROJECT_ROOT / "prototype" / "manifests"
REPORT_DIR = PROJECT_ROOT / "prototype" / "reports"

CALIB_MANIFEST = MANIFEST_DIR / "vcf_c23_calib_temporal_f3_seed20260421_sw010_grouped.csv"
HELDOUT_MANIFEST = MANIFEST_DIR / "vcf_c23_heldout_temporal_f3_seed20260421_sw010_grouped.csv"

JSON_OUT = REPORT_DIR / "vcf_temporal_subject_source_audit_2026-04-25.json"
TSV_OUT = REPORT_DIR / "vcf_temporal_subject_source_audit_2026-04-25.tsv"


AUDIT_FIELDS = [
    (
        "identity_id",
        "Subject proxy stored in the VCF manifest.",
        "pass",
        "No overlap supports subject-proxy disjoint calibration and held-out rows.",
    ),
    (
        "pair_group_id",
        "Protected/unprotected pairing family used by the router.",
        "pass",
        "No overlap means paired source baselines are not shared across splits.",
    ),
    (
        "temporal_video_key",
        "Temporal group/video key parsed from manifest notes.",
        "pass",
        "No overlap means ordered temporal groups are split-disjoint.",
    ),
    (
        "temporal_sample_id",
        "Three-frame grouped sample id parsed from manifest notes.",
        "pass",
        "No overlap means group-level examples are not replayed across splits.",
    ),
    (
        "raw_video_path",
        "Full raw-video path parsed from manifest notes.",
        "pass",
        "No overlap means exact raw videos are not shared across splits.",
    ),
    (
        "source_path",
        "Frame/source image path stored in the manifest.",
        "pass",
        "No overlap means exact source frames are not reused across splits.",
    ),
    (
        "planned_output_path",
        "Protected or unprotected output path stored in the manifest.",
        "pass",
        "No overlap means generated media paths are split-disjoint.",
    ),
    (
        "passive_key",
        "Stored passive-feature path.",
        "pass",
        "No overlap means feature files are not shared across splits.",
    ),
    (
        "provenance_key",
        "Stored provenance-feature path for protected rows.",
        "pass",
        "No overlap means provenance features are not reused across splits.",
    ),
    (
        "target_stem",
        "Coarse VCF content stem, ignoring method/resolution/path.",
        "boundary",
        "Overlap is reported as a source-family boundary; the paper does not claim target-stem-disjoint generalization.",
    ),
]


def parse_notes(notes: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for chunk in notes.split(";"):
        chunk = chunk.strip()
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def value_for(row: dict[str, str], notes: dict[str, str], field: str) -> str:
    if field == "temporal_video_key":
        return notes.get("temporal_video_key", "")
    if field == "temporal_sample_id":
        return notes.get("temporal_sample_id", "")
    if field == "raw_video_path":
        return notes.get("raw_video_path", "")
    if field == "target_stem":
        return notes.get("target_stem") or notes.get("stem", "")
    return row.get(field, "")


def collect(path: Path) -> dict[str, object]:
    rows: list[dict[str, str]] = []
    values = {field: set() for field, *_ in AUDIT_FIELDS}
    label_counts: dict[str, int] = {}
    protection_counts: dict[str, int] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
            label_counts[row["label"]] = label_counts.get(row["label"], 0) + 1
            protection_counts[row["protection"]] = protection_counts.get(row["protection"], 0) + 1
            notes = parse_notes(row.get("notes", ""))
            for field, *_ in AUDIT_FIELDS:
                value = value_for(row, notes, field)
                if value:
                    values[field].add(value)
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "rows": len(rows),
        "label_counts": label_counts,
        "protection_counts": protection_counts,
        "values": values,
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    calib = collect(CALIB_MANIFEST)
    heldout = collect(HELDOUT_MANIFEST)

    records = []
    for field, description, expected_status, reading in AUDIT_FIELDS:
        calib_values = calib["values"][field]
        heldout_values = heldout["values"][field]
        overlap = sorted(calib_values & heldout_values)
        status = expected_status
        if expected_status == "pass" and overlap:
            status = "fail"
        records.append(
            {
                "field": field,
                "description": description,
                "calib_unique": len(calib_values),
                "heldout_unique": len(heldout_values),
                "overlap": len(overlap),
                "status": status,
                "reading": reading,
                "overlap_examples": overlap[:10],
            }
        )

    summary = {
        "calibration_manifest": calib["path"],
        "heldout_manifest": heldout["path"],
        "calibration_rows": calib["rows"],
        "heldout_rows": heldout["rows"],
        "calibration_label_counts": calib["label_counts"],
        "heldout_label_counts": heldout["label_counts"],
        "calibration_protection_counts": calib["protection_counts"],
        "heldout_protection_counts": heldout["protection_counts"],
        "records": records,
        "pass": all(r["status"] != "fail" for r in records),
        "boundary": "The strict audited keys are split-disjoint. Coarse target_stem overlap is retained as an explicit source-family boundary, not hidden.",
    }

    JSON_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with TSV_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "field",
                "description",
                "calib_unique",
                "heldout_unique",
                "overlap",
                "status",
                "reading",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for record in records:
            writer.writerow({k: record[k] for k in writer.fieldnames})

    print(f"Wrote {JSON_OUT.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {TSV_OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
