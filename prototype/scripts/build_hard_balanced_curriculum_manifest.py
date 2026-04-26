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


def clone_row(row: dict, tag: str, rep: int, prob: float, manifest_path: Path) -> dict:
    out = dict(row)
    out["sample_id"] = f"{row['sample_id']}_{tag}_r{rep + 1}"
    out["pair_group_id"] = f"{row.get('pair_group_id', row['sample_id'])}_{tag}_r{rep + 1}"
    out["notes"] = f"{row.get('notes', '')}; {tag}_source_prob={prob:.6f}; source_manifest={manifest_path.name}"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a source-only hard-example curriculum manifest with hard fake recall "
            "and matched hard real replay."
        )
    )
    parser.add_argument("--source", action="append", required=True, help="MANIFEST=PASSIVE_CSV. Repeat for source datasets.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fake-threshold", type=float, default=0.6)
    parser.add_argument("--real-threshold", type=float, default=0.4)
    parser.add_argument("--fake-repeat", type=int, default=2)
    parser.add_argument("--real-repeat", type=int, default=2)
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Keep the same number of hard fake and hard real base videos, using the smaller selected pool.",
    )
    args = parser.parse_args()

    hard_fakes: list[tuple[float, dict, Path]] = []
    hard_reals: list[tuple[float, dict, Path]] = []
    scanned_fake = 0
    scanned_real = 0

    for source_value in args.source:
        manifest_path, passive_path = parse_pair(source_value)
        probs = load_passive_probs(passive_path)
        with manifest_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("protection") != "unprotected":
                    continue
                label = row.get("label")
                prob = probs.get(row.get("sample_id", ""))
                if prob is None:
                    continue
                if label == "fake":
                    scanned_fake += 1
                    if prob < args.fake_threshold:
                        hard_fakes.append((prob, row, manifest_path))
                elif label == "real":
                    scanned_real += 1
                    if prob > args.real_threshold:
                        hard_reals.append((prob, row, manifest_path))

    hard_fakes.sort(key=lambda item: item[0])
    hard_reals.sort(key=lambda item: item[0], reverse=True)
    selected_fake = len(hard_fakes)
    selected_real = len(hard_reals)
    if args.balance:
        keep = min(selected_fake, selected_real)
        hard_fakes = hard_fakes[:keep]
        hard_reals = hard_reals[:keep]

    rows_out: list[dict] = []
    for prob, row, manifest_path in hard_fakes:
        for rep in range(args.fake_repeat):
            rows_out.append(clone_row(row, "hardfake", rep, prob, manifest_path))
    for prob, row, manifest_path in hard_reals:
        for rep in range(args.real_repeat):
            rows_out.append(clone_row(row, "hardreal", rep, prob, manifest_path))

    write_csv(Path(args.output), rows_out)
    print(
        {
            "output": args.output,
            "rows": len(rows_out),
            "selected_hard_fake": len(hard_fakes),
            "selected_hard_real": len(hard_reals),
            "scanned_fake": scanned_fake,
            "scanned_real": scanned_real,
            "fake_threshold": args.fake_threshold,
            "real_threshold": args.real_threshold,
            "fake_repeat": args.fake_repeat,
            "real_repeat": args.real_repeat,
            "balance": args.balance,
            "raw_selected_hard_fake": selected_fake,
            "raw_selected_hard_real": selected_real,
        }
    )


if __name__ == "__main__":
    main()
