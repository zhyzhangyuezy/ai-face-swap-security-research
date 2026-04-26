from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(value: str, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def source_dataset_name(manifest_path: Path) -> str:
    stem = manifest_path.stem.lower()
    if "ffpp" in stem or "faceforensics" in stem:
        return "ffpp"
    if "uadfv" in stem:
        return "uadfv"
    if "celebdfv1" in stem or "celebdf_v1" in stem or "celebdf" in stem:
        return "celebdfv1"
    return stem


def stable_digest(value: str, length: int = 10) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def source_group_id(row: dict[str, str], dataset: str) -> str:
    identity = row.get("identity_id", "") or Path(row.get("source_path", "")).parent.name
    label = row.get("label", "")
    if dataset == "uadfv":
        base = identity.removesuffix("_fake")
        return f"uadfv-source-{base}"
    if dataset == "ffpp":
        # FF++ manipulated ids are source_target pairs. Keep the manipulated video as
        # one group; real source videos remain source-separated.
        return f"ffpp-{label}-{identity}"
    if dataset.startswith("celebdf"):
        return f"{dataset}-{label}-{identity}"
    return f"{dataset}-{label}-{identity}"


def reliable_pair_source_group(row: dict[str, str], dataset: str) -> str:
    identity = row.get("identity_id", "") or Path(row.get("source_path", "")).parent.name
    if dataset == "uadfv":
        return f"uadfv-source-{identity.removesuffix('_fake')}"
    if dataset == "ffpp":
        source_identity = identity.split("_", 1)[0]
        if source_identity:
            return f"ffpp-source-{source_identity}"
    return ""


def label_signature(rows: list[dict[str, str]]) -> str:
    labels = sorted({row.get("label", "") for row in rows})
    if labels == ["fake", "real"]:
        return "mixed"
    if labels:
        return labels[0]
    return "unknown"


def choose_split(index: int, count: int, train_ratio: float, val_ratio: float) -> str:
    train_cut = round(count * train_ratio)
    val_cut = train_cut + round(count * val_ratio)
    if index < train_cut:
        return "train"
    if index < val_cut:
        return "val"
    return "test"


def assign_splits(
    groups: dict[str, list[dict[str, str]]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, str]:
    by_signature: dict[str, list[str]] = defaultdict(list)
    for group_id, rows in groups.items():
        by_signature[label_signature(rows)].append(group_id)

    assignments: dict[str, str] = {}
    for signature, group_ids in by_signature.items():
        rng = random.Random(seed + int(stable_digest(signature, 6), 16))
        group_ids = group_ids[:]
        rng.shuffle(group_ids)
        for index, group_id in enumerate(group_ids):
            assignments[group_id] = choose_split(index, len(group_ids), train_ratio, val_ratio)
    return assignments


def append_note(row: dict[str, str], text: str) -> None:
    previous = row.get("notes", "")
    row["notes"] = f"{previous}; {text}" if previous else text


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a source-video-aware curriculum manifest by explicitly assigning "
            "train/val/test splits at source group granularity."
        )
    )
    parser.add_argument("--manifest", action="append", required=True, help="Input manifest. Repeat for multiple sources.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows-per-label", type=int, default=0)
    parser.add_argument("--include-protected-real", action="store_true")
    args = parser.parse_args()

    root = project_root()
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    missing = 0
    skipped = 0
    for manifest_value in args.manifest:
        manifest_path = resolve_path(manifest_value, root)
        dataset = source_dataset_name(manifest_path)
        with manifest_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for name in reader.fieldnames or []:
                if name not in fieldnames:
                    fieldnames.append(name)
            for row in reader:
                protection = row.get("protection", "")
                label = row.get("label", "")
                if protection != "unprotected" and not (
                    args.include_protected_real and protection == "protected" and label == "real"
                ):
                    skipped += 1
                    continue
                if label not in {"real", "fake"}:
                    skipped += 1
                    continue
                path_value = row.get("planned_output_path", "") if protection == "protected" else row.get("source_path", "")
                source_path = resolve_path(path_value, root)
                if not source_path.exists():
                    missing += 1
                    continue
                row = dict(row)
                row["_source_dataset"] = dataset
                row["_source_manifest"] = manifest_path.name
                row["_source_group_id"] = source_group_id(row, dataset)
                row["_pair_source_group"] = reliable_pair_source_group(row, dataset)
                row["_split_group_id"] = row["_pair_source_group"] or row["_source_group_id"]
                rows.append(row)

    if args.max_rows_per_label > 0:
        limited_rows: list[dict[str, str]] = []
        rng = random.Random(args.seed)
        buckets: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            buckets[(row["_source_dataset"], row["label"])].append(row)
        for bucket_rows in buckets.values():
            bucket_rows = bucket_rows[:]
            rng.shuffle(bucket_rows)
            limited_rows.extend(bucket_rows[: args.max_rows_per_label])
        rows = limited_rows

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["_split_group_id"]].append(row)
    assignments = assign_splits(groups, args.train_ratio, args.val_ratio, args.seed)

    for row in rows:
        group_id = row["_source_group_id"]
        pair_group = row["_pair_source_group"]
        split_group = row["_split_group_id"]
        split = assignments[split_group]
        row["split"] = split
        row["source_group_id"] = group_id
        row["pair_source_group"] = pair_group
        append_note(
            row,
            (
                f"sourceaware_group={group_id}; pair_source_group={pair_group or 'none'}; "
                f"sourceaware_split_group={split_group}; sourceaware_split={split}; "
                f"source_manifest={row['_source_manifest']}"
            ),
        )
        for private_key in ["_source_dataset", "_source_manifest", "_source_group_id", "_pair_source_group", "_split_group_id"]:
            row.pop(private_key, None)

    if "split" not in fieldnames:
        fieldnames.insert(1, "split")
    for name in rows[0].keys() if rows else []:
        if name not in fieldnames:
            fieldnames.append(name)

    rows.sort(key=lambda row: (row.get("split", ""), row.get("label", ""), row.get("sample_id", "")))
    write_csv(Path(args.output), rows, fieldnames)

    split_counts = Counter(row.get("split", "") for row in rows)
    label_counts = Counter((row.get("split", ""), row.get("label", "")) for row in rows)
    pair_counts = Counter(row.get("pair_source_group", "") for row in rows if row.get("pair_source_group", ""))
    summary = {
        "output": args.output,
        "rows": len(rows),
        "groups": len(groups),
        "split_groups": len(groups),
        "pair_source_groups": len(pair_counts),
        "split_counts": dict(split_counts),
        "label_counts": {f"{split}:{label}": count for (split, label), count in label_counts.items()},
        "missing_rows": missing,
        "skipped_rows": skipped,
        "include_protected_real": args.include_protected_real,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
    }
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
