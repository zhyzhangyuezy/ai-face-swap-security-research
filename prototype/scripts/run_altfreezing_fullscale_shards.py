from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def count_rows(path: Path) -> int:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def complete_csv(path: Path, expected_rows: int) -> bool:
    if not path.exists():
        return False
    try:
        rows = read_csv(path)
    except Exception:
        return False
    return len(rows) == expected_rows


def shard_offsets(total: int, shard_size: int) -> Iterable[tuple[int, int]]:
    for offset in range(0, total, shard_size):
        yield offset, min(shard_size, total - offset)


def run_shard(args: argparse.Namespace, split: str, manifest: Path, output_dir: Path, offset: int, limit: int) -> Path:
    shard_csv = output_dir / f"{split}_offset{offset:06d}_limit{limit:04d}.csv"
    if complete_csv(shard_csv, limit):
        print(f"[skip] {split} offset={offset} limit={limit} -> {shard_csv}", flush=True)
        return shard_csv

    script = project_root() / "prototype" / "scripts" / "execute_altfreezing_native_facetrack_manifest_probe.py"
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{split}_offset{offset:06d}_limit{limit:04d}.out.log"
    stderr_path = log_dir / f"{split}_offset{offset:06d}_limit{limit:04d}.err.log"
    command = [
        sys.executable,
        str(script),
        "--manifest",
        str(manifest),
        "--output",
        str(shard_csv),
        "--offset",
        str(offset),
        "--limit",
        str(limit),
        "--max-frame",
        str(args.max_frame),
        "--max-clips-per-video",
        str(args.max_clips_per_video),
        "--protect-mode",
        args.protect_mode,
        "--progress-every",
        str(args.progress_every),
    ]
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    print(f"[run] {split} offset={offset} limit={limit}", flush=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        subprocess.run(command, cwd=project_root(), env=env, stdout=stdout, stderr=stderr, check=True)
    if not complete_csv(shard_csv, limit):
        raise RuntimeError(f"Shard did not complete with {limit} rows: {shard_csv}")
    return shard_csv


def combine(split: str, output_dir: Path, shard_paths: list[Path], combined_path: Path, expected_total: int) -> None:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    seen: set[str] = set()
    for shard_path in shard_paths:
        for row in read_csv(shard_path):
            sample_id = row.get("sample_id", "")
            if sample_id in seen:
                raise RuntimeError(f"Duplicate sample_id in {split}: {sample_id}")
            seen.add(sample_id)
            rows.append(row)
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    if len(rows) != expected_total:
        raise RuntimeError(f"{split} combined rows {len(rows)} != expected {expected_total}")
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with combined_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[combine] {split}: {len(rows)} rows -> {combined_path}", flush=True)


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Run resumable full-manifest AltFreezing native face-track shards.")
    parser.add_argument("--calib-manifest", default=str(root / "prototype" / "manifests" / "vcf_c23_calib_all_seed20260421_sw010.csv"))
    parser.add_argument("--heldout-manifest", default=str(root / "prototype" / "manifests" / "vcf_c23_heldout_all_seed20260421_sw010.csv"))
    parser.add_argument("--output-dir", default=str(root / "prototype" / "reports" / "altfreezing_native_fullscale_mf64_mc4_2026-04-25"))
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--max-frame", type=int, default=64)
    parser.add_argument("--max-clips-per-video", type=int, default=4)
    parser.add_argument("--protect-mode", choices=["source", "videoseal_aligned"], default="videoseal_aligned")
    parser.add_argument("--progress-every", type=int, default=128)
    parser.add_argument("--splits", nargs="+", choices=["calib", "heldout"], default=["calib", "heldout"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests = {
        "calib": Path(args.calib_manifest),
        "heldout": Path(args.heldout_manifest),
    }
    combined = {
        "calib": output_dir / "vcf_c23_calib_altfreezing_native_fullscale_mf64_mc4_2026-04-25.csv",
        "heldout": output_dir / "vcf_c23_heldout_altfreezing_native_fullscale_mf64_mc4_2026-04-25.csv",
    }

    for split in args.splits:
        manifest = manifests[split]
        total = count_rows(manifest)
        print(f"[split] {split}: total={total}, shard_size={args.shard_size}, shards={math.ceil(total / args.shard_size)}", flush=True)
        shard_paths: list[Path] = []
        for offset, limit in shard_offsets(total, args.shard_size):
            shard_paths.append(run_shard(args, split, manifest, output_dir, offset, limit))
        combine(split, output_dir, shard_paths, combined[split], total)
    print("[done] AltFreezing full-manifest shards complete.", flush=True)


if __name__ == "__main__":
    main()
