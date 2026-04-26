from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load_passive_probs(path: Path) -> dict[str, float]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        out = {}
        for row in reader:
            sample_id = row.get("sample_id", "")
            prob = row.get("prob_fake", "")
            if sample_id and prob:
                out[sample_id] = float(prob)
        return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_pair(value: str) -> tuple[Path, Path]:
    if "=" not in value:
        raise ValueError(f"Expected MANIFEST=PASSIVE_CSV, got {value}")
    manifest, passive = value.split("=", 1)
    return Path(manifest), Path(passive)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a source-only hard fake curriculum manifest from passive predictions.")
    parser.add_argument("--source", action="append", required=True, help="MANIFEST=PASSIVE_CSV. Repeat for source datasets.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()

    rows_out: list[dict] = []
    selected = 0
    scanned_fake = 0
    for source_value in args.source:
        manifest_path, passive_path = parse_pair(source_value)
        probs = load_passive_probs(passive_path)
        with manifest_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("label") != "fake" or row.get("protection") != "unprotected":
                    continue
                scanned_fake += 1
                prob = probs.get(row.get("sample_id", ""))
                if prob is None or prob >= args.threshold:
                    continue
                selected += 1
                for rep in range(args.repeat):
                    out = dict(row)
                    out["sample_id"] = f"{row['sample_id']}_hardfake_r{rep + 1}"
                    out["pair_group_id"] = f"{row.get('pair_group_id', row['sample_id'])}_hardfake_r{rep + 1}"
                    out["notes"] = f"{row.get('notes', '')}; hard_fake_source_prob={prob:.6f}; source_manifest={manifest_path.name}"
                    rows_out.append(out)

    write_csv(Path(args.output), rows_out)
    print(
        {
            "output": args.output,
            "rows": len(rows_out),
            "selected_hard_fake": selected,
            "scanned_fake": scanned_fake,
            "threshold": args.threshold,
            "repeat": args.repeat,
        }
    )


if __name__ == "__main__":
    main()
