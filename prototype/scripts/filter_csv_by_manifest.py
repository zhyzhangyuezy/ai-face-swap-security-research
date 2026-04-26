from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_sample_ids(manifest_path: Path) -> set[str]:
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {row.get("sample_id", "") for row in reader if row.get("sample_id", "")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter a CSV to sample_id values present in a manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    sample_ids = read_sample_ids(Path(args.manifest))
    input_path = Path(args.input)
    with input_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [row for row in reader if row.get("sample_id", "") in sample_ids]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"kept_rows={len(rows)} sample_ids={len(sample_ids)} output={output_path}")


if __name__ == "__main__":
    main()
