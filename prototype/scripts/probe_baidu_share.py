from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests


APP_ID = "250528"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_share_url(url: str, pwd: str | None) -> tuple[str, str | None]:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    password = pwd or (query.get("pwd", [""])[0] or None)
    token = parsed.path.rstrip("/").rsplit("/", 1)[-1]
    if token.startswith("1"):
        token = token[1:]
    if not token:
        raise SystemExit(f"Could not parse Baidu share token from URL: {url}")
    return token, password


def headers(referer: str) -> dict[str, str]:
    return {
        "User-Agent": USER_AGENT,
        "Referer": referer,
    }


def verify_share(session: requests.Session, surl: str, pwd: str | None) -> dict[str, Any]:
    init_url = f"https://pan.baidu.com/share/init?surl={surl}"
    if pwd:
        init_url += f"&pwd={pwd}"
    session.get(init_url, headers=headers(init_url), timeout=30).raise_for_status()

    verify_url = "https://pan.baidu.com/share/verify"
    params = {
        "surl": surl,
        "t": int(time.time() * 1000),
        "channel": "chunlei",
        "web": "1",
        "app_id": APP_ID,
        "bdstoken": "null",
        "clienttype": "0",
    }
    response = session.post(
        verify_url,
        params=params,
        data={"pwd": pwd or "", "vcode": "", "vcode_str": ""},
        headers={
            **headers(init_url),
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "X-Requested-With": "XMLHttpRequest",
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return {key: data.get(key) for key in ("errno", "err_msg", "request_id")}


def load_share_page(session: requests.Session, surl: str) -> dict[str, Any]:
    page_url = f"https://pan.baidu.com/s/1{surl}"
    response = session.get(page_url, headers=headers(page_url), timeout=30)
    response.raise_for_status()
    match = re.search(r"locals\.mset\((.*?)\);", response.text, flags=re.S)
    if not match:
        title_match = re.search(r"<title>(.*?)</title>", response.text, flags=re.S)
        title = title_match.group(1).strip() if title_match else ""
        raise RuntimeError(f"Could not find locals.mset data in share page; title={title!r}")
    return json.loads(match.group(1))


def summarize_file(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "server_filename": row.get("server_filename", ""),
        "isdir": int(row.get("isdir", 0)),
        "size": int(row.get("size", 0) or 0),
        "fs_id": row.get("fs_id"),
        "path": row.get("path", ""),
        "server_filename_suffix": row.get("server_filename_suffix", ""),
    }


def list_shared_dir(
    session: requests.Session,
    share_uk: str,
    shareid: int | str,
    dir_path: str,
    page_size: int,
) -> dict[str, Any]:
    page_url = "https://pan.baidu.com/share/list"
    response = session.get(
        page_url,
        params={
            "uk": share_uk,
            "shareid": shareid,
            "order": "other",
            "desc": "1",
            "showempty": "0",
            "web": "1",
            "page": "1",
            "num": str(page_size),
            "dir": dir_path,
            "t": int(time.time() * 1000),
            "channel": "chunlei",
            "app_id": APP_ID,
            "clienttype": "0",
        },
        headers=headers(f"https://pan.baidu.com/share/list?dir={dir_path}"),
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return {
        "errno": data.get("errno"),
        "request_id": data.get("request_id"),
        "dir": dir_path,
        "list": [summarize_file(item) for item in data.get("list", [])],
    }


def resolve_output(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    root = project_root().resolve()
    path = Path(path_value)
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"Refusing to write outside project root: {resolved}") from exc
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe a public Baidu Pan share and list visible files.")
    parser.add_argument("--url", required=True, help="Baidu share URL, optionally including ?pwd=xxxx.")
    parser.add_argument("--pwd", default=None, help="Extraction code if not included in the URL.")
    parser.add_argument("--expand-root-dirs", action="store_true", help="List one level inside root directories.")
    parser.add_argument("--page-size", type=int, default=200)
    parser.add_argument("--output", default=None, help="Optional JSON report path inside the project.")
    args = parser.parse_args()

    surl, pwd = parse_share_url(args.url, args.pwd)
    session = requests.Session()
    verify = verify_share(session, surl, pwd)
    share = load_share_page(session, surl)

    root_files = [summarize_file(item) for item in share.get("file_list", [])]
    expanded_dirs = []
    if args.expand_root_dirs:
        for item in root_files:
            if item["isdir"]:
                expanded_dirs.append(
                    list_shared_dir(
                        session,
                        str(share.get("share_uk", "")),
                        share.get("shareid", ""),
                        item["path"],
                        args.page_size,
                    )
                )

    report = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "url": args.url,
        "surl": surl,
        "pwd_provided": bool(pwd),
        "verify": verify,
        "share": {
            "shareid": share.get("shareid"),
            "share_uk": share.get("share_uk"),
            "errno": share.get("errno"),
            "share_page_type": share.get("share_page_type"),
            "linkusername": share.get("linkusername", ""),
            "loginstate": share.get("loginstate"),
        },
        "root_files": root_files,
        "expanded_dirs": expanded_dirs,
    }

    output = resolve_output(args.output)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
