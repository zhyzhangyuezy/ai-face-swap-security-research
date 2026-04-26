from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate a DF40 fake-only manifest with per-method sample weights."
    )
    parser.add_argument("--input", required=True, help="Input CSV manifest.")
    parser.add_argument("--output", required=True, help="Output CSV manifest with sample_weight column.")
    parser.add_argument("--summary-json", required=True, help="Output JSON summary.")
    parser.add_argument(
        "--method-weight",
        action="append",
        required=True,
        help="Method-specific weight, e.g. --method-weight simswap=1.25",
    )
    parser.add_argument(
        "--default-weight",
        type=float,
        default=1.0,
        help="Fallback weight for methods not explicitly listed.",
    )
    return parser.parse_args()


def infer_method(row: dict[str, str]) -> str:
    sample_id = row.get("sample_id", "")
    parts = sample_id.split("_")
    if len(parts) >= 2 and parts[0] == "df40":
        return parts[1]
    for note in row.get("notes", "").split(";"):
        note = note.strip()
        if note.startswith("method="):
            return note.split("=", 1)[1].strip()
    raise ValueError(f"Cannot infer method for row with sample_id={sample_id!r}")


def parse_method_weights(values: list[str]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected method=weight, got {value!r}")
        method, weight_text = value.split("=", 1)
        method = method.strip()
        if not method:
            raise ValueError(f"Empty method name in {value!r}")
        weight = float(weight_text)
        if weight < 0:
            raise ValueError(f"Weight must be non-negative, got {value!r}")
        weights[method] = weight
    return weights


def append_note(row: dict[str, str], note: str) -> dict[str, str]:
    out = dict(row)
    notes = out.get("notes", "")
    out["notes"] = f"{notes}; {note}" if notes else note
    return out


def main() -> None:
    args = parse_args()
    method_weights = parse_method_weights(args.method_weight)

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary_json)

    with input_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if "sample_weight" not in fieldnames:
        fieldnames.append("sample_weight")

    summary_counts: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "weight_sum": 0.0})
    output_rows: list[dict[str, str]] = []
    for row in rows:
        method = infer_method(row)
        weight = method_weights.get(method, args.default_weight)
        out = append_note(row, f"sample_weight={weight:.6f}; weight_source=methodaware")
        out["sample_weight"] = f"{weight:.6f}"
        output_rows.append(out)
        summary_counts[method]["count"] += 1
        summary_counts[method]["weight_sum"] += weight

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "default_weight": args.default_weight,
        "method_weights": method_weights,
        "methods": {
            method: {
                "count": int(stats["count"]),
                "mean_weight": stats["weight_sum"] / max(1, stats["count"]),
            }
            for method, stats in sorted(summary_counts.items())
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
