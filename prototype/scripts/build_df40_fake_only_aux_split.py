from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a method-balanced train/heldout split from a DF40 fake-only manifest."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--heldout-output", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--train-per-method", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260419)
    return parser.parse_args()


def method_name(row: dict[str, str]) -> str:
    sample_id = row.get("sample_id", "")
    parts = sample_id.split("_")
    if len(parts) >= 2 and parts[0] == "df40":
        return parts[1]
    notes = row.get("notes", "")
    for item in notes.split(";"):
        item = item.strip()
        if item.startswith("method="):
            return item.split("=", 1)[1].strip()
    raise ValueError(f"Cannot infer DF40 method for sample_id={sample_id!r}")


def video_key(row: dict[str, str]) -> str:
    identity = row.get("identity_id", "")
    if identity:
        return identity
    path_value = row.get("source_path") or row.get("planned_output_path") or ""
    if path_value:
        path = Path(path_value)
        return f"{method_name(row)}/{path.parent.name}"
    return row.get("sample_id", "")


def append_note(row: dict[str, str], note: str) -> dict[str, str]:
    out = dict(row)
    notes = out.get("notes", "")
    out["notes"] = f"{notes}; {note}" if notes else note
    return out


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    with input_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_method[method_name(row)].append(row)

    rng = random.Random(args.seed)
    train_rows: list[dict[str, str]] = []
    heldout_rows: list[dict[str, str]] = []
    summary = {
        "input": str(input_path),
        "seed": args.seed,
        "train_per_method": args.train_per_method,
        "methods": {},
    }

    for method in sorted(by_method):
        method_rows = sorted(by_method[method], key=lambda row: (video_key(row), row.get("sample_id", "")))
        keys = [video_key(row) for row in method_rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Expected one row per video for {method}; found duplicate video keys.")
        shuffled_keys = keys[:]
        rng.shuffle(shuffled_keys)
        selected = set(shuffled_keys[: args.train_per_method])
        method_train = []
        method_heldout = []
        for row in method_rows:
            row_split = "train" if video_key(row) in selected else "eval"
            out = append_note(row, f"df40_fake_aux_split={row_split}; seed={args.seed}")
            out["split"] = row_split
            if row_split == "train":
                method_train.append(out)
            else:
                method_heldout.append(out)
        train_rows.extend(method_train)
        heldout_rows.extend(method_heldout)
        summary["methods"][method] = {
            "total": len(method_rows),
            "train": len(method_train),
            "heldout": len(method_heldout),
        }

    train_rows.sort(key=lambda row: (method_name(row), video_key(row), row.get("sample_id", "")))
    heldout_rows.sort(key=lambda row: (method_name(row), video_key(row), row.get("sample_id", "")))
    write_rows(Path(args.train_output), fieldnames, train_rows)
    write_rows(Path(args.heldout_output), fieldnames, heldout_rows)
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
