from __future__ import annotations

import argparse
import csv
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


SOURCE_RE = re.compile(r"(?:^|;\s*)source_id=([^;]+)")
METHOD_RE = re.compile(r"(?:^|;\s*)method=([^;]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a small train-only hard-generator paired auxiliary manifest from DF40 FF++ paired manifests. "
            "The output adds pair_source_group/source_group_id columns so source-family ranking can push "
            "simswap/inswap fakes away from their source-real family."
        )
    )
    parser.add_argument("--input-manifest", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--max-pairs-per-method", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260422)
    parser.add_argument("--sample-weight", type=float, default=0.25)
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_note(pattern: re.Pattern[str], notes: str) -> str:
    match = pattern.search(notes or "")
    return match.group(1).strip() if match else ""


def method_name(row: Dict[str, str], manifest_path: Path) -> str:
    parsed = parse_note(METHOD_RE, row.get("notes", ""))
    if parsed:
        return parsed
    for token in ["simswap", "inswap", "faceswap", "fsgan"]:
        if token in manifest_path.name:
            return token
    return "unknown"


def source_id(row: Dict[str, str]) -> str:
    parsed = parse_note(SOURCE_RE, row.get("notes", ""))
    if parsed:
        return parsed
    identity = row.get("identity_id", "")
    return identity.split("_", 1)[0]


def augment_row(row: Dict[str, str], method: str, source: str, sample_weight: float) -> Dict[str, str]:
    out = dict(row)
    group = f"hardgen-{method}-src-{source}"
    out["split"] = "train"
    out["source_group_id"] = group
    out["pair_source_group"] = group
    out["sample_weight"] = f"{sample_weight:.6f}"
    tag = "hardgen_target_fake=1" if out.get("label") == "fake" else "hardgen_source_real=1"
    notes = out.get("notes", "") or ""
    suffix = (
        f"hardgen_paired_aux=1; hardgen_method={method}; hardgen_source_id={source}; "
        f"sourceaware_group={group}; pair_source_group={group}; sample_weight={sample_weight:.6f}; {tag}"
    )
    out["notes"] = f"{notes}; {suffix}" if notes else suffix
    return out


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    selected_rows: List[Dict[str, str]] = []
    summary: Dict[str, dict] = {}

    for manifest_value in args.input_manifest:
        manifest_path = Path(manifest_value)
        rows = read_csv(manifest_path)
        if not rows:
            continue
        method = method_name(rows[0], manifest_path)
        rows_by_source: dict[str, list[Dict[str, str]]] = defaultdict(list)
        for row in rows:
            if row.get("split") != "train":
                continue
            rows_by_source[source_id(row)].append(row)

        pairable_sources = []
        for source, source_rows in rows_by_source.items():
            has_real = any(row.get("label") == "real" for row in source_rows)
            has_fake = any(row.get("label") == "fake" for row in source_rows)
            if has_real and has_fake:
                pairable_sources.append(source)
        pairable_sources = sorted(pairable_sources)
        rng.shuffle(pairable_sources)
        keep_sources = sorted(pairable_sources[: args.max_pairs_per_method])

        for source in keep_sources:
            source_rows = sorted(rows_by_source[source], key=lambda item: item.get("sample_id", ""))
            for row in source_rows:
                selected_rows.append(augment_row(row, method, source, args.sample_weight))

        summary[method] = {
            "input_manifest": str(manifest_path),
            "available_pairable_sources": len(pairable_sources),
            "selected_sources": keep_sources,
            "selected_source_count": len(keep_sources),
            "selected_rows": sum(1 for row in selected_rows if f"hardgen_method={method}" in row.get("notes", "")),
        }

    write_csv(Path(args.output), selected_rows)
    import json

    payload = {
        "output": args.output,
        "seed": args.seed,
        "max_pairs_per_method": args.max_pairs_per_method,
        "sample_weight": args.sample_weight,
        "total_rows": len(selected_rows),
        "methods": summary,
    }
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
