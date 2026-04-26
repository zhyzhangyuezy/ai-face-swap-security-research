from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_note_map(notes: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in notes.split(";"):
        item = item.strip()
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def target_key(row: dict[str, str]) -> str:
    note_map = parse_note_map(row.get("notes", ""))
    stem = note_map.get("stem") or row.get("identity_id") or row.get("pair_group_id") or row.get("sample_id", "")
    resolution = note_map.get("resolution", "")
    bucket = note_map.get("bucket", "")
    domain = note_map.get("domain", "")
    return "|".join([domain, resolution, bucket, stem])


def deterministic_score(value: str, seed: str) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16 - 1)


def target_source_group(value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"vcf-target-{digest}"


def keep_bucket(row: dict[str, str], buckets: set[str]) -> bool:
    if not buckets:
        return True
    return parse_note_map(row.get("notes", "")).get("bucket", "") in buckets


def write_manifest(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def row_counter(rows: list[dict[str, str]]) -> dict[str, object]:
    counts = Counter()
    bucket_counts = Counter()
    fake_degradation_counts = Counter()
    for row in rows:
        counts[f"{row.get('label', '')}/{row.get('protection', '')}"] += 1
        note_map = parse_note_map(row.get("notes", ""))
        bucket_counts[note_map.get("bucket", "")] += 1
        if row.get("label") == "fake":
            fake_degradation_counts[row.get("degradation_type", "")] += 1
    return {
        "rows": len(rows),
        "label_protection": dict(sorted(counts.items())),
        "buckets": dict(sorted(bucket_counts.items())),
        "fake_degradation_types": dict(sorted(fake_degradation_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build VCF target-real calibration and heldout manifests without target leakage. "
            "Calibration contains real rows only; heldout keeps all labels for disjoint target keys."
        )
    )
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output-calib-real", required=True)
    parser.add_argument("--output-calib-all", default="")
    parser.add_argument("--output-heldout-all", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--seed", default="20260421")
    parser.add_argument("--calib-fraction", type=float, default=0.5)
    parser.add_argument("--sample-weight", type=float, default=0.25)
    parser.add_argument("--fake-sample-weight", type=float, default=-1.0)
    parser.add_argument("--add-target-sourceaware-group", action="store_true")
    parser.add_argument("--bucket", action="append", default=[], help="Optional bucket filter for calibration and heldout.")
    parser.add_argument("--calib-protection", action="append", default=[], help="Optional real protection subset for calibration.")
    args = parser.parse_args()

    source_path = Path(args.source_manifest)
    with source_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        source_fieldnames = list(reader.fieldnames or [])
        source_rows = [row for row in reader if keep_bucket(row, set(args.bucket))]

    if not source_rows:
        raise ValueError(f"No rows remain after bucket filtering: {args.bucket}")

    rows_by_target: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        rows_by_target[target_key(row)].append(row)

    if not rows_by_target:
        raise ValueError(f"No target keys found in {source_path}")

    calib_fraction = min(0.95, max(0.05, args.calib_fraction))
    calib_keys: set[str] = set()
    heldout_keys: set[str] = set()
    for key in sorted(rows_by_target):
        if deterministic_score(key, args.seed) < calib_fraction:
            calib_keys.add(key)
        else:
            heldout_keys.add(key)

    if not calib_keys or not heldout_keys:
        ordered = sorted(rows_by_target, key=lambda key: deterministic_score(key, args.seed))
        split_at = max(1, min(len(ordered) - 1, round(len(ordered) * calib_fraction)))
        calib_keys = set(ordered[:split_at])
        heldout_keys = set(ordered[split_at:])

    protection_filter = set(args.calib_protection)
    calib_rows: list[dict[str, str]] = []
    calib_all_rows: list[dict[str, str]] = []
    heldout_rows: list[dict[str, str]] = []
    for key, key_rows in rows_by_target.items():
        if key in calib_keys:
            for row in key_rows:
                if args.output_calib_all:
                    updated_all = dict(row)
                    updated_all["split"] = "train"
                    if args.add_target_sourceaware_group:
                        updated_all["source_group_id"] = target_source_group(key)
                    label_weight = args.fake_sample_weight if row.get("label") == "fake" and args.fake_sample_weight >= 0 else args.sample_weight
                    updated_all["sample_weight"] = f"{max(0.0, label_weight):.6f}"
                    notes_all = updated_all.get("notes", "").strip()
                    extra_notes_all = [
                        "source_manifest=vcf_target_calib_all",
                        "target_split=calib",
                        f"calib_seed={args.seed}",
                        f"sample_weight={max(0.0, label_weight):.6f}",
                    ]
                    updated_all["notes"] = "; ".join([item for item in [notes_all, *extra_notes_all] if item])
                    calib_all_rows.append(updated_all)
                if row.get("label") != "real":
                    continue
                if protection_filter and row.get("protection", "") not in protection_filter:
                    continue
                updated = dict(row)
                updated["split"] = "train"
                if args.add_target_sourceaware_group:
                    updated["source_group_id"] = target_source_group(key)
                updated["sample_weight"] = f"{max(0.0, args.sample_weight):.6f}"
                notes = updated.get("notes", "").strip()
                extra_notes = [
                    "source_manifest=vcf_target_real_calib",
                    f"target_split=calib",
                    f"calib_seed={args.seed}",
                    f"sample_weight={max(0.0, args.sample_weight):.6f}",
                ]
                updated["notes"] = "; ".join([item for item in [notes, *extra_notes] if item])
                calib_rows.append(updated)
        else:
            for row in key_rows:
                updated = dict(row)
                updated["split"] = "test"
                if args.add_target_sourceaware_group:
                    updated["source_group_id"] = target_source_group(key)
                notes = updated.get("notes", "").strip()
                extra_notes = ["target_split=heldout", f"calib_seed={args.seed}"]
                updated["notes"] = "; ".join([item for item in [notes, *extra_notes] if item])
                heldout_rows.append(updated)

    fieldnames = list(source_fieldnames)
    if "sample_weight" not in fieldnames:
        fieldnames.append("sample_weight")
    if args.add_target_sourceaware_group and "source_group_id" not in fieldnames:
        fieldnames.append("source_group_id")
    write_manifest(Path(args.output_calib_real), calib_rows, fieldnames)
    if args.output_calib_all:
        write_manifest(Path(args.output_calib_all), calib_all_rows, fieldnames)
    write_manifest(Path(args.output_heldout_all), heldout_rows, fieldnames)

    summary = {
        "source_manifest": str(source_path),
        "seed": args.seed,
        "calib_fraction": calib_fraction,
        "sample_weight": max(0.0, args.sample_weight),
        "fake_sample_weight": args.fake_sample_weight if args.fake_sample_weight >= 0 else args.sample_weight,
        "add_target_sourceaware_group": args.add_target_sourceaware_group,
        "bucket_filter": args.bucket,
        "calib_protection_filter": args.calib_protection,
        "target_keys": {
            "total": len(rows_by_target),
            "calib": len(calib_keys),
            "heldout": len(heldout_keys),
        },
        "calib_real": row_counter(calib_rows),
        "calib_all": row_counter(calib_all_rows) if args.output_calib_all else None,
        "heldout_all": row_counter(heldout_rows),
        "outputs": {
            "calib_real": str(Path(args.output_calib_real)),
            "calib_all": str(Path(args.output_calib_all)) if args.output_calib_all else "",
            "heldout_all": str(Path(args.output_heldout_all)),
        },
    }
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
