from __future__ import annotations

import argparse
import csv
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image


RESULT_FIELDS = [
    "sample_id",
    "status",
    "input_path",
    "protection",
    "prob_fake",
    "pred_label",
    "logit_real",
    "logit_fake",
    "feature_dim",
    "feature_norm",
    "feature_path",
]


def _bootstrap_src() -> None:
    script_dir = Path(__file__).resolve().parent
    src_root = script_dir.parent / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


_bootstrap_src()

from hybrid_router.config_io import load_yaml_config
from hybrid_router.manifest_schema import load_manifest


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@contextmanager
def temporary_working_directory(path: Path):
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


def parse_notes(notes: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for part in notes.split(";"):
        token = part.strip()
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def resolve_input(row: Dict[str, str]) -> Path:
    if row["protection"] == "protected":
        return Path(row["planned_output_path"])
    return Path(row["source_path"])


def temporal_groups(rows: List[Dict[str, str]]) -> List[dict]:
    grouped: Dict[str, List[dict]] = {}
    for index, row in enumerate(rows):
        notes = parse_notes(row.get("notes", ""))
        temporal_sample_id = notes.get("temporal_sample_id") or row["sample_id"]
        slot = int(notes.get("temporal_frame_slot", str(index)))
        grouped.setdefault(temporal_sample_id, []).append({"slot": slot, "row": row, "input_path": resolve_input(row)})

    output: List[dict] = []
    for temporal_sample_id, items in grouped.items():
        items.sort(key=lambda item: item["slot"])
        output.append({"temporal_sample_id": temporal_sample_id, "items": items})
    return output


def padded_clip_indices(frame_count: int, clip_size: int) -> List[int]:
    if frame_count <= 0:
        return []
    if frame_count == 1:
        return [0] * clip_size
    if frame_count >= clip_size:
        return [int(round(value)) for value in np.linspace(0, frame_count - 1, clip_size)]

    inner = list(range(frame_count))
    pad_length = clip_size - 1
    post_module = inner[1:-1][::-1] + inner
    if not post_module:
        post_module = inner
    post_module = (post_module * (pad_length // len(post_module) + 1))[:pad_length]

    pre_module = inner + inner[1:-1][::-1]
    if not pre_module:
        pre_module = inner
    pre_module = (pre_module * (pad_length // len(pre_module) + 1))[-pad_length:]

    padded = pre_module + inner + post_module
    start = max(0, (len(padded) - clip_size) // 2)
    return padded[start : start + clip_size]


def read_rgb(path: Path, image_size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    return np.asarray(image, dtype=np.float32)


def group_tensor(items: List[dict], image_size: int, clip_size: int) -> torch.Tensor:
    frames = [read_rgb(item["input_path"], image_size) for item in items]
    indices = padded_clip_indices(len(frames), clip_size)
    clip = np.stack([frames[index] for index in indices], axis=0)
    tensor = torch.as_tensor(clip, dtype=torch.float32).permute(3, 0, 1, 2)
    return tensor


def install_ftcn_path(ftcn_root: Path) -> None:
    if str(ftcn_root) not in sys.path:
        sys.path.insert(0, str(ftcn_root))


def load_ftcn(ftcn_root: Path, checkpoint_path: Path, device: str):
    install_ftcn_path(ftcn_root)
    from config import config as cfg
    from utils.plugin_loader import PluginLoader

    cfg.init_with_yaml()
    cfg.update_with_yaml("ftcn_tt.yaml")
    cfg.freeze()
    with temporary_working_directory(ftcn_root):
        classifier = PluginLoader.get_classifier(cfg.classifier_type)()
        classifier.to(device).eval()
        loaded, _ = classifier.load(str(checkpoint_path))
    if not loaded:
        raise RuntimeError(f"Failed to load FTCN checkpoint: {checkpoint_path}")
    return classifier, int(cfg.clip_size), int(cfg.imsize)


def empty_result(row: Dict[str, str], input_path: Path, status: str) -> Dict[str, str]:
    return {
        "sample_id": row["sample_id"],
        "status": status,
        "input_path": str(input_path),
        "protection": row["protection"],
        "prob_fake": "",
        "pred_label": "",
        "logit_real": "",
        "logit_fake": "",
        "feature_dim": "",
        "feature_norm": "",
        "feature_path": "",
    }


def scored_result(row: Dict[str, str], input_path: Path, prob_fake: float, threshold: float) -> Dict[str, str]:
    pred_label = 1 if prob_fake >= threshold else 0
    return {
        "sample_id": row["sample_id"],
        "status": "ok",
        "input_path": str(input_path),
        "protection": row["protection"],
        "prob_fake": f"{prob_fake:.6f}",
        "pred_label": str(pred_label),
        "logit_real": f"{1.0 - prob_fake:.6f}",
        "logit_fake": f"{prob_fake:.6f}",
        "feature_dim": "0",
        "feature_norm": "0.000000",
        "feature_path": "",
    }


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Run FTCN+TT on a grouped VCF temporal manifest.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", help="Optional explicit manifest path.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--ftcn-root", default=str(root / "workspace" / "repos" / "FTCN"))
    parser.add_argument("--checkpoint", default=str(root / "workspace" / "checkpoints" / "ftcn" / "ftcn_tt.pth"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit-groups", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="Use CUDA autocast for faster inference.")
    parser.add_argument("--progress-every", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
    config = load_yaml_config(Path(args.config))
    manifest_path = Path(args.manifest or config["paths"]["manifest_path"])
    rows = load_manifest(manifest_path)
    groups = temporal_groups(rows)
    if args.limit_groups > 0:
        budget = args.limit_groups
    else:
        budget = None

    model, clip_size, image_size = load_ftcn(Path(args.ftcn_root), Path(args.checkpoint), args.device)
    mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255], device=args.device).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255], device=args.device).view(1, 3, 1, 1, 1)

    batch_groups: List[dict] = []
    batch_tensors: List[torch.Tensor] = []
    processed_groups = 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_handle = output_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(output_handle, fieldnames=RESULT_FIELDS)
    writer.writeheader()

    def emit(rows_to_write: List[Dict[str, str]]) -> None:
        writer.writerows(rows_to_write)
        output_handle.flush()

    def flush_batch() -> None:
        nonlocal processed_groups
        if not batch_groups:
            return
        images = torch.stack(batch_tensors, dim=0).to(args.device).sub(mean).div(std)
        use_amp = args.amp and args.device.startswith("cuda")
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                outputs = model(images)["final_output"].reshape(-1)
        probs = outputs.detach().cpu().numpy().astype(float).tolist()
        rows_to_write: List[Dict[str, str]] = []
        for group, prob_fake in zip(batch_groups, probs):
            for item in group["items"]:
                row = item["row"]
                input_path = item["input_path"]
                rows_to_write.append(scored_result(row, input_path, float(prob_fake), args.threshold))
        emit(rows_to_write)
        processed_groups += len(batch_groups)
        if args.progress_every > 0 and processed_groups % args.progress_every < len(batch_groups):
            print(f"processed_groups={processed_groups}/{len(groups)}", flush=True)
        batch_groups.clear()
        batch_tensors.clear()

    try:
        for group_index, group in enumerate(groups):
            if budget is not None and group_index >= budget:
                rows_to_write = []
                for item in group["items"]:
                    rows_to_write.append(empty_result(item["row"], item["input_path"], "skipped_limit"))
                emit(rows_to_write)
                continue
            missing = [item for item in group["items"] if not item["input_path"].exists()]
            if missing:
                rows_to_write = []
                for item in group["items"]:
                    status = "missing_input" if item in missing else "skipped_group_missing"
                    rows_to_write.append(empty_result(item["row"], item["input_path"], status))
                emit(rows_to_write)
                continue
            try:
                tensor = group_tensor(group["items"], image_size=image_size, clip_size=clip_size)
            except Exception as exc:
                rows_to_write = []
                for item in group["items"]:
                    rows_to_write.append(empty_result(item["row"], item["input_path"], f"error:{type(exc).__name__}"))
                emit(rows_to_write)
                continue
            batch_groups.append(group)
            batch_tensors.append(tensor)
            if len(batch_groups) >= args.batch_size:
                flush_batch()
        flush_batch()
    finally:
        output_handle.close()

    print(f"Wrote FTCN+TT passive results to {args.output}")


if __name__ == "__main__":
    main()
