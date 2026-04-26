#!/usr/bin/env python3
"""Download the official AltFreezing RecDrive checkpoint with ranged GETs."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import requests


API_ROOT = "https://recapi.ustc.edu.cn/api/v2"
SHARE_NUMBER = "e87360b0-7b2e-11ef-aeef-a9fd0832d537"
PASSWORD = "altf"
EXPECTED_BYTES = 109_241_199


def _headers(content_type: bool = False) -> dict[str, str]:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": f"https://rec.ustc.edu.cn/share/{SHARE_NUMBER}",
        "Origin": "https://rec.ustc.edu.cn",
    }
    if content_type:
        headers["Content-Type"] = "application/json"
    return headers


def get_download_url(session: requests.Session) -> tuple[str, int]:
    base_payload = {
        "share_number": SHARE_NUMBER,
        "share_constraint": {"password": PASSWORD},
    }
    listing = session.post(
        f"{API_ROOT}/share/target/resource/list",
        headers=_headers(content_type=True),
        json={**base_payload, "share_resource_number": None, "is_rec": "false"},
        timeout=(15, 60),
    )
    listing.raise_for_status()
    listing_obj = json.loads(listing.content.decode("utf-8-sig"))
    if listing_obj.get("status_code") != 200:
        raise RuntimeError(f"RecDrive listing failed: {listing_obj}")
    candidates = [
        item
        for item in listing_obj["entity"]
        if item.get("type") == "file" and item.get("file_ext") == "pth"
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one .pth file, got {len(candidates)}")
    resource = candidates[0]
    resource_number = resource["number"]
    expected_bytes = int(resource.get("bytes") or EXPECTED_BYTES)

    dl = session.post(
        f"{API_ROOT}/share/download",
        headers=_headers(content_type=True),
        json={**base_payload, "share_resources_list": [resource_number]},
        timeout=(15, 60),
    )
    dl.raise_for_status()
    dl_obj = json.loads(dl.content.decode("utf-8-sig"))
    if dl_obj.get("status_code") != 200:
        raise RuntimeError(f"RecDrive download-url request failed: {dl_obj}")
    return dl_obj["entity"][resource_number] + "&download=download", expected_bytes


def download_ranges(url: str, out_path: Path, expected_bytes: int, chunk_bytes: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size == expected_bytes:
        print(f"exists complete: {out_path} ({expected_bytes} bytes)")
        return

    part_path = out_path.with_suffix(out_path.suffix + ".part")
    with requests.Session() as session, part_path.open("r+b" if part_path.exists() else "w+b") as fp:
        current = part_path.stat().st_size
        if current > expected_bytes:
            part_path.unlink()
            current = 0
            fp = part_path.open("w+b")
        fp.seek(current)
        while current < expected_bytes:
            end = min(current + chunk_bytes - 1, expected_bytes - 1)
            headers = _headers()
            headers["Range"] = f"bytes={current}-{end}"
            for attempt in range(1, 6):
                try:
                    response = session.get(
                        url,
                        headers=headers,
                        stream=True,
                        timeout=(20, 120),
                    )
                    if response.status_code not in (200, 206):
                        raise RuntimeError(
                            f"unexpected HTTP {response.status_code}: {response.text[:200]}"
                        )
                    written = 0
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            fp.write(block)
                            written += len(block)
                    fp.flush()
                    current += written
                    pct = 100.0 * current / expected_bytes
                    print(f"downloaded {current}/{expected_bytes} bytes ({pct:.1f}%)")
                    break
                except Exception as exc:
                    if attempt == 5:
                        raise
                    wait = 3 * attempt
                    print(f"range {current}-{end} failed on attempt {attempt}: {exc}; retrying in {wait}s")
                    time.sleep(wait)
            else:
                raise RuntimeError("unreachable retry state")
    if part_path.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"download size mismatch: got {part_path.stat().st_size}, expected {expected_bytes}"
        )
    shutil.move(str(part_path), str(out_path))
    print(f"saved: {out_path} ({out_path.stat().st_size} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="workspace/repos/AltFreezing/checkpoints/model.pth",
        help="Output checkpoint path.",
    )
    parser.add_argument("--chunk-mb", type=int, default=8)
    args = parser.parse_args()

    with requests.Session() as session:
        url, expected_bytes = get_download_url(session)
    download_ranges(
        url=url,
        out_path=Path(args.out),
        expected_bytes=expected_bytes,
        chunk_bytes=args.chunk_mb * 1024 * 1024,
    )


if __name__ == "__main__":
    main()
