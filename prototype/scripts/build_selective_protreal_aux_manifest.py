from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def source_manifest(notes: str) -> str:
    match = re.search(r"source_manifest=([^;]+)", notes or "")
    return match.group(1) if match else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mine high-risk protected-real train rows from an existing manifest using "
            "passive probe probabilities, and emit a small aux-only stabilizer manifest."
        )
    )
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--probe-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--top-k", type=int, default=96)
    parser.add_argument("--min-prob-fake", type=float, default=0.0)
    parser.add_argument("--sample-weight", type=float, default=0.25)
    parser.add_argument("--note-token", default="selective_prot_real=1")
    parser.add_argument("--tag", default="selective_protreal_hardmine")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.input_manifest)
    probe_path = Path(args.probe_csv)
    output_path = Path(args.output)
    summary_path = Path(args.summary_output)

    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        manifest_rows = {row["sample_id"]: row for row in reader}
    if "sample_weight" not in fieldnames:
        fieldnames.append("sample_weight")
    if "source_group_id" not in fieldnames:
        fieldnames.append("source_group_id")
    if "pair_source_group" not in fieldnames:
        fieldnames.append("pair_source_group")

    candidates = []
    with probe_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for probe in csv.DictReader(handle):
            sample_id = probe.get("sample_id", "")
            row = manifest_rows.get(sample_id)
            if not row:
                continue
            if row.get("split", "").lower() != "train":
                continue
            if row.get("label") != "real" or row.get("protection") != "protected":
                continue
            try:
                prob_fake = float(probe.get("prob_fake", ""))
            except ValueError:
                continue
            if prob_fake < args.min_prob_fake:
                continue
            candidates.append((prob_fake, row, probe))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[: max(0, args.top_k)]
    output_rows = []
    for rank, (prob_fake, row, probe) in enumerate(selected, start=1):
        new_row = dict(row)
        new_row["sample_weight"] = f"{args.sample_weight:.6f}"
        notes = new_row.get("notes", "")
        extras = [
            args.note_token,
            f"{args.tag}=1",
            f"selective_rank={rank}",
            f"mined_prob_fake={prob_fake:.6f}",
        ]
        new_row["notes"] = f"{notes}; " + "; ".join(extras) if notes else "; ".join(extras)
        if not new_row.get("source_group_id"):
            new_row["source_group_id"] = new_row.get("pair_source_group") or new_row.get("identity_id", "")
        if not new_row.get("pair_source_group"):
            new_row["pair_source_group"] = new_row.get("source_group_id", "")
        output_rows.append(new_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    source_counts = Counter(source_manifest(row.get("notes", "")) for _, row, _ in selected)
    degradation_counts = Counter((row.get("degradation_type", ""), row.get("degradation_level", "")) for _, row, _ in selected)
    score_values = [prob for prob, _, _ in selected]
    summary = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "input_manifest": str(manifest_path),
        "probe_csv": str(probe_path),
        "output": str(output_path),
        "top_k": args.top_k,
        "min_prob_fake": args.min_prob_fake,
        "sample_weight": args.sample_weight,
        "note_token": args.note_token,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "selected_min_prob_fake": min(score_values) if score_values else None,
        "selected_max_prob_fake": max(score_values) if score_values else None,
        "source_manifest_counts": dict(source_counts),
        "degradation_counts": {f"{key[0]}:{key[1]}": value for key, value in degradation_counts.items()},
        "selected_examples": [
            {
                "sample_id": row.get("sample_id"),
                "identity_id": row.get("identity_id"),
                "degradation_type": row.get("degradation_type"),
                "degradation_level": row.get("degradation_level"),
                "prob_fake": prob,
                "source_manifest": source_manifest(row.get("notes", "")),
            }
            for prob, row, _ in selected[:20]
        ],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
