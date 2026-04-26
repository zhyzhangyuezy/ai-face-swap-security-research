from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

try:
    from gdown.download import get_url_from_gdrive_confirmation
except Exception:  # pragma: no cover
    get_url_from_gdrive_confirmation = None


CHUNK_SIZE = 8 * 1024 * 1024


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def append_attempt(root: Path, event: dict[str, Any]) -> None:
    output = root / "prototype" / "reports" / "google_drive_download_attempts.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def resolve_confirm_url(session: requests.Session, file_id: str, timeout: int) -> tuple[str, dict[str, Any]]:
    url = f"https://drive.google.com/uc?id={file_id}"
    response = session.get(url, stream=True, timeout=timeout)
    info = {
        "initial_status": response.status_code,
        "initial_content_type": response.headers.get("Content-Type", ""),
        "initial_content_length": response.headers.get("Content-Length", ""),
    }
    if response.headers.get("Content-Type", "").startswith("application/octet-stream"):
        return url, info

    text = response.text
    response.close()
    if get_url_from_gdrive_confirmation is None:
        raise RuntimeError("gdown is required to parse Google Drive confirmation pages.")
    confirm_url = get_url_from_gdrive_confirmation(text)
    info["confirm_url_prefix"] = confirm_url[:120]
    return confirm_url, info


def find_part_file(output: Path) -> Path:
    candidates = sorted(output.parent.glob(f"{output.name}*.part"))
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple partial files found for {output}: {candidates}")
    if candidates:
        return candidates[0]
    return output.with_name(output.name + ".part")


def parse_total_size(response: requests.Response, start_size: int) -> int | None:
    content_range = response.headers.get("Content-Range")
    if content_range and "/" in content_range:
        tail = content_range.rsplit("/", 1)[-1]
        if tail.isdigit():
            return int(tail)
    content_length = response.headers.get("Content-Length")
    if content_length and content_length.isdigit():
        return int(content_length) + start_size
    return None


def download_file(file_id: str, output: Path, timeout: int, force: bool) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0 and not force:
        return {
            "status": "skipped_exists",
            "output": str(output),
            "size_bytes": output.stat().st_size,
        }

    part_file = find_part_file(output)
    if force and part_file.exists():
        part_file.unlink()
    start_size = part_file.stat().st_size if part_file.exists() else 0

    session = requests.Session()
    confirm_url, info = resolve_confirm_url(session, file_id, timeout)
    headers = {"Range": f"bytes={start_size}-"} if start_size > 0 else {}
    response = session.get(confirm_url, headers=headers, stream=True, timeout=timeout)
    total_size = parse_total_size(response, start_size)

    if start_size > 0 and response.status_code == 200:
        # Server ignored Range. Restart to avoid corrupting the partial file.
        part_file.unlink(missing_ok=True)
        start_size = 0
        response.close()
        response = session.get(confirm_url, stream=True, timeout=timeout)
        total_size = parse_total_size(response, start_size)

    if response.status_code not in {200, 206}:
        raise RuntimeError(f"Unexpected HTTP status {response.status_code}: {response.text[:200]}")

    content_type = response.headers.get("Content-Type", "")
    content_disposition = response.headers.get("Content-Disposition", "")
    if "text/html" in content_type.lower() and "attachment" not in content_disposition.lower():
        text = response.text[:500]
        response.close()
        session.close()
        raise RuntimeError(f"Google Drive returned HTML instead of a file, likely quota or permission blocking: {text}")

    mode = "ab" if start_size > 0 else "wb"
    downloaded = start_size
    last_report = time.time()
    with part_file.open(mode + "") as handle:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if not chunk:
                continue
            handle.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            if now - last_report >= 10:
                total_display = total_size if total_size is not None else "unknown"
                print(f"downloaded={downloaded} total={total_display}", file=sys.stderr)
                last_report = now

    response.close()
    session.close()

    if total_size is not None and downloaded < total_size:
        return {
            "status": "partial",
            "output": str(output),
            "part_file": str(part_file),
            "downloaded_bytes": downloaded,
            "total_bytes": total_size,
            **info,
        }

    os.replace(part_file, output)
    return {
        "status": "ok",
        "output": str(output),
        "size_bytes": output.stat().st_size,
        "total_bytes": total_size,
        **info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Robust Google Drive file downloader with simple resume support.")
    parser.add_argument("--id", required=True, help="Google Drive file id.")
    parser.add_argument("--output", required=True, help="Output path inside the project.")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = project_root()
    output = Path(args.output)
    if output.is_absolute():
        try:
            output.resolve().relative_to(root.resolve())
        except ValueError as exc:
            raise SystemExit(f"Refusing to write outside project root: {output}") from exc
    else:
        output = root / output

    event = {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "action": "download_google_drive_file",
        "file_id": args.id,
        "output": str(output),
        "status": "started",
    }
    try:
        result = download_file(args.id, output, timeout=args.timeout, force=args.force)
        event.update(result)
    except Exception as exc:
        event.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        append_attempt(root, event)
        print(json.dumps(event, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
