from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


STRATA = (
    ("real", "protected"),
    ("real", "unprotected"),
    ("fake", "protected"),
    ("fake", "unprotected"),
)


def load_csv(path: Path) -> List[Dict[str, str]]:
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


def parse_notes(notes: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for part in notes.split(";"):
        token = part.strip()
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def target_stem(row: Dict[str, str]) -> str:
    notes = parse_notes(row.get("notes", ""))
    stem = notes.get("stem")
    if stem:
        return stem
    source = Path(row.get("source_path", ""))
    return source.stem or row.get("pair_group_id") or row["sample_id"]


def temporal_group_id(row: Dict[str, str]) -> str:
    notes = parse_notes(row.get("notes", ""))
    return notes.get("temporal_sample_id") or row["sample_id"]


def group_manifest_rows(rows: Iterable[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[temporal_group_id(row)].append(row)
    return dict(grouped)


def group_stratum(rows: List[Dict[str, str]]) -> Tuple[str, str]:
    labels = {row["label"] for row in rows}
    protections = {row["protection"] for row in rows}
    if len(labels) != 1 or len(protections) != 1:
        raise ValueError(f"Temporal group mixes labels/protection: {rows[0].get('sample_id')}")
    return (next(iter(labels)), next(iter(protections)))


def stem_counters(groups_by_id: Dict[str, List[Dict[str, str]]]) -> Dict[str, Counter]:
    counters: Dict[str, Counter] = defaultdict(Counter)
    for rows in groups_by_id.values():
        stem = target_stem(rows[0])
        counters[stem][group_stratum(rows)] += 1
    return dict(counters)


def score_split(eval_counts: Counter, total_counts: Counter, eval_fraction: float) -> float:
    score = 0.0
    for stratum in STRATA:
        target = total_counts[stratum] * eval_fraction
        score += abs(eval_counts[stratum] - target)
    return score


def choose_eval_stems(stem_counts: Dict[str, Counter], eval_fraction: float, seed: int) -> set[str]:
    stems = list(stem_counts)
    rng = random.Random(seed)
    rng.shuffle(stems)
    stems.sort(key=lambda stem: (-sum(stem_counts[stem].values()), stem))

    total_counts = Counter()
    for counts in stem_counts.values():
        total_counts.update(counts)

    eval_counts: Counter = Counter()
    eval_stems: set[str] = set()
    for stem in stems:
        current_score = score_split(eval_counts, total_counts, eval_fraction)
        with_stem = eval_counts.copy()
        with_stem.update(stem_counts[stem])
        next_score = score_split(with_stem, total_counts, eval_fraction)
        current_groups = sum(eval_counts.values())
        target_groups = sum(total_counts.values()) * eval_fraction
        if next_score < current_score or current_groups < target_groups * 0.92:
            eval_stems.add(stem)
            eval_counts = with_stem
    return eval_stems


def counts_by_stratum(groups: Iterable[List[Dict[str, str]]]) -> Dict[str, int]:
    counts = Counter(group_stratum(rows) for rows in groups)
    return {f"{label}_{protection}": counts[(label, protection)] for label, protection in STRATA}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a VCF temporal split with zero target_stem overlap.")
    parser.add_argument("--calib-manifest", required=True)
    parser.add_argument("--heldout-manifest", required=True)
    parser.add_argument("--calib-passive", required=True)
    parser.add_argument("--heldout-passive", required=True)
    parser.add_argument("--out-calib-manifest", required=True)
    parser.add_argument("--out-heldout-manifest", required=True)
    parser.add_argument("--out-calib-passive", required=True)
    parser.add_argument("--out-heldout-passive", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--eval-fraction", type=float, default=0.47)
    parser.add_argument("--seed", type=int, default=20260425)
    args = parser.parse_args()

    manifest_rows = load_csv(Path(args.calib_manifest)) + load_csv(Path(args.heldout_manifest))
    passive_rows = load_csv(Path(args.calib_passive)) + load_csv(Path(args.heldout_passive))
    groups = group_manifest_rows(manifest_rows)
    stem_counts = stem_counters(groups)
    eval_stems = choose_eval_stems(stem_counts, args.eval_fraction, args.seed)

    calib_ids = set()
    eval_ids = set()
    for group_id, rows in groups.items():
        if target_stem(rows[0]) in eval_stems:
            eval_ids.add(group_id)
        else:
            calib_ids.add(group_id)

    calib_rows = [dict(row, split="train") for row in manifest_rows if temporal_group_id(row) in calib_ids]
    eval_rows = [dict(row, split="test") for row in manifest_rows if temporal_group_id(row) in eval_ids]
    calib_sample_ids = {row["sample_id"] for row in calib_rows}
    eval_sample_ids = {row["sample_id"] for row in eval_rows}
    calib_passive = [row for row in passive_rows if row.get("sample_id") in calib_sample_ids]
    eval_passive = [row for row in passive_rows if row.get("sample_id") in eval_sample_ids]

    calib_stems = {target_stem(row) for row in calib_rows}
    heldout_stems = {target_stem(row) for row in eval_rows}
    sample_overlap = calib_sample_ids & eval_sample_ids
    summary = {
        "seed": args.seed,
        "eval_fraction": args.eval_fraction,
        "manifest_rows": {"calib": len(calib_rows), "heldout": len(eval_rows)},
        "temporal_groups": {"calib": len(calib_ids), "heldout": len(eval_ids)},
        "target_stems": {
            "calib_unique": len(calib_stems),
            "heldout_unique": len(heldout_stems),
            "overlap": len(calib_stems & heldout_stems),
        },
        "sample_id_overlap": len(sample_overlap),
        "calib_group_strata": counts_by_stratum(groups[group_id] for group_id in calib_ids),
        "heldout_group_strata": counts_by_stratum(groups[group_id] for group_id in eval_ids),
        "passive_rows": {"calib": len(calib_passive), "heldout": len(eval_passive)},
    }

    write_csv(Path(args.out_calib_manifest), calib_rows)
    write_csv(Path(args.out_heldout_manifest), eval_rows)
    write_csv(Path(args.out_calib_passive), calib_passive)
    write_csv(Path(args.out_heldout_passive), eval_passive)
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
