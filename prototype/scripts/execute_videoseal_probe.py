from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from torchvision.transforms import functional as TF


def _bootstrap_src() -> None:
    script_dir = Path(__file__).resolve().parent
    src_root = script_dir.parent / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


_bootstrap_src()

from hybrid_router.config_io import load_optional_yaml_config, load_yaml_config
from hybrid_router.manifest_schema import load_manifest


RESULT_FIELDS = [
    "sample_id",
    "status",
    "protection",
    "degradation_type",
    "degradation_level",
    "source_path",
    "output_path",
    "provenance_json",
    "detect_logit",
    "detect_prob",
    "bit_accuracy",
    "message_len",
    "watermark_l1_mean",
]


def default_paths_file(config_path: Path) -> Path:
    return config_path.parent / "paths.local.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute the VideoSeal provenance probe on a smoke-test manifest.")
    parser.add_argument("--config", required=True, help="Path to the smoke-test YAML config.")
    parser.add_argument("--manifest", help="Optional explicit manifest path.")
    parser.add_argument("--paths", help="Optional local paths YAML config.")
    parser.add_argument("--output", help="Optional explicit CSV output path.")
    parser.add_argument("--model-card", default="videoseal", help="VideoSeal model card to load.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on the number of protected rows to execute.")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing protected outputs and provenance JSON payloads.")
    return parser.parse_args()


def apply_degradation(image: Image.Image, degradation_type: str, degradation_level: str) -> Image.Image:
    if degradation_type == "none":
        return image
    if degradation_type == "jpeg":
        quality = 75
        digits = "".join(ch for ch in degradation_level if ch.isdigit())
        if digits:
            quality = int(digits)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    if degradation_type == "resize":
        ratio = 0.5
        level = degradation_level.lower().replace("x", "")
        try:
            ratio = float(level)
        except ValueError:
            ratio = 0.5
        width, height = image.size
        down = image.resize((max(1, int(width * ratio)), max(1, int(height * ratio))), Image.BILINEAR)
        return down.resize((width, height), Image.BILINEAR)
    raise ValueError(f"Unsupported degradation type: {degradation_type}")


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def execute_rows(
    manifest_rows: List[Dict[str, str]],
    repo_root: Path,
    output_csv: Path,
    model_card: str,
    device: str,
    limit: int,
    skip_existing: bool,
) -> None:
    old_cwd = Path.cwd()
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))
    import videoseal

    model = videoseal.load(model_card).to(device)
    model.eval()

    protected_budget = limit if limit > 0 else None
    protected_count = 0
    results: List[Dict[str, str]] = []

    try:
        with torch.no_grad():
            for row in manifest_rows:
                base = {
                    "sample_id": row["sample_id"],
                    "protection": row["protection"],
                    "degradation_type": row["degradation_type"],
                    "degradation_level": row["degradation_level"],
                    "source_path": row["source_path"],
                    "output_path": row["planned_output_path"],
                    "provenance_json": row["provenance_key"],
                }
                if row["protection"] != "protected":
                    results.append({
                        **base,
                        "status": "skipped_unprotected",
                        "detect_logit": "",
                        "detect_prob": "",
                        "bit_accuracy": "",
                        "message_len": "",
                        "watermark_l1_mean": "",
                    })
                    continue

                source_path = Path(row["source_path"])
                output_path = Path(row["planned_output_path"])
                provenance_json = Path(row["provenance_key"])
                if skip_existing and output_path.exists() and provenance_json.exists():
                    try:
                        payload = json.loads(provenance_json.read_text(encoding="utf-8"))
                    except Exception:
                        payload = {}
                    results.append({
                        **base,
                        "status": "skipped_existing",
                        "detect_logit": f"{float(payload.get('detect_logit', 0.0)):.6f}" if "detect_logit" in payload else "",
                        "detect_prob": f"{float(payload.get('detect_prob', 0.0)):.6f}" if "detect_prob" in payload else "",
                        "bit_accuracy": f"{float(payload.get('bit_accuracy', 0.0)):.6f}" if "bit_accuracy" in payload else "",
                        "message_len": str(payload.get("message_len", "")),
                        "watermark_l1_mean": f"{float(payload.get('watermark_l1_mean', 0.0)):.6f}" if "watermark_l1_mean" in payload else "",
                    })
                    continue

                if protected_budget is not None and protected_count >= protected_budget:
                    results.append({
                        **base,
                        "status": "skipped_limit",
                        "detect_logit": "",
                        "detect_prob": "",
                        "bit_accuracy": "",
                        "message_len": "",
                        "watermark_l1_mean": "",
                    })
                    continue

                protected_count += 1

                image = Image.open(source_path).convert("RGB")
                source_tensor = TF.to_tensor(image).unsqueeze(0).to(device)
                embedded = model.embed(source_tensor)
                watermarked = embedded["imgs_w"][0].detach().cpu().clamp(0, 1)
                watermark_l1_mean = float(torch.mean(torch.abs(watermarked - source_tensor[0].detach().cpu())).item())

                watermarked_pil = TF.to_pil_image(watermarked)
                degraded_pil = apply_degradation(watermarked_pil, row["degradation_type"], row["degradation_level"])
                output_path.parent.mkdir(parents=True, exist_ok=True)
                degraded_pil.save(output_path)

                detect_tensor = TF.to_tensor(degraded_pil).unsqueeze(0).to(device)
                detected = model.detect(detect_tensor)
                detect_logit = float(detected["preds"][0, 0].item())
                detect_prob = float(torch.sigmoid(detected["preds"][0, 0]).item())

                embedded_bits = embedded["msgs"][0].detach().cpu().to(torch.int64)
                recovered_bits = (detected["preds"][0, 1:].detach().cpu() > 0).to(torch.int64)
                bit_accuracy = float((embedded_bits == recovered_bits).float().mean().item())

                payload = {
                    "sample_id": row["sample_id"],
                    "model_card": model_card,
                    "device": device,
                    "source_path": str(source_path),
                    "output_path": str(output_path),
                    "degradation_type": row["degradation_type"],
                    "degradation_level": row["degradation_level"],
                    "detect_logit": detect_logit,
                    "detect_prob": detect_prob,
                    "bit_accuracy": bit_accuracy,
                    "message_len": int(embedded_bits.numel()),
                    "watermark_l1_mean": watermark_l1_mean,
                }
                provenance_json.parent.mkdir(parents=True, exist_ok=True)
                provenance_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

                results.append({
                    **base,
                    "status": "ok",
                    "detect_logit": f"{detect_logit:.6f}",
                    "detect_prob": f"{detect_prob:.6f}",
                    "bit_accuracy": f"{bit_accuracy:.6f}",
                    "message_len": str(int(embedded_bits.numel())),
                    "watermark_l1_mean": f"{watermark_l1_mean:.6f}",
                })
    finally:
        os.chdir(old_cwd)

    write_csv(output_csv, results)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_yaml_config(config_path)
    manifest_path = Path(args.manifest or config["paths"]["manifest_path"])
    output_csv = Path(args.output or str(Path(config["paths"]["provenance_jobs_path"]).with_name(f"{Path(config['paths']['provenance_jobs_path']).stem.replace('_jobs', '')}_results.csv")))
    paths_cfg = load_optional_yaml_config(args.paths or default_paths_file(config_path))
    repo_root = Path(paths_cfg.get("repos", {}).get("videoseal", ""))
    if not repo_root.exists():
        raise FileNotFoundError(f"VideoSeal repo root does not exist: {repo_root}")

    manifest_rows = load_manifest(manifest_path)
    execute_rows(manifest_rows, repo_root, output_csv, args.model_card, args.device, args.limit, args.skip_existing)
    print(f"Wrote VideoSeal provenance results to {output_csv}")


if __name__ == "__main__":
    main()
