from __future__ import annotations

import argparse
import csv
import io
import os
import sys
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List

import imageio.v3 as iio
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
    "raw_video_path",
    "clip_frames",
    "protect_mode",
]


def _bootstrap_src() -> None:
    script_dir = Path(__file__).resolve().parent
    src_root = script_dir.parent / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


_bootstrap_src()

from hybrid_router.config_io import load_yaml_config
from hybrid_router.manifest_schema import load_manifest


@contextmanager
def temporary_working_directory(path: Path):
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_notes(notes: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for part in notes.split(";"):
        token = part.strip()
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def raw_video_from_row(row: Dict[str, str]) -> Path:
    notes = parse_notes(row.get("notes", ""))
    if notes.get("raw_video_path"):
        return Path(notes["raw_video_path"])
    return Path(row["source_path"])


def metadata_frame_count(video_path: Path) -> int:
    try:
        meta = iio.immeta(video_path)
    except Exception:
        return 0
    fps = float(meta.get("fps") or 0.0)
    duration = float(meta.get("duration") or 0.0)
    if fps > 0 and duration > 0:
        return max(1, int(round(fps * duration)))
    nframes = meta.get("nframes") or meta.get("n_frames")
    try:
        return max(1, int(nframes))
    except Exception:
        return 0


def clip_indices(video_path: Path, clip_size: int) -> List[int]:
    total = metadata_frame_count(video_path)
    if total <= 1:
        return [0] * clip_size
    return [min(total - 1, max(0, int(round(v)))) for v in np.linspace(0, total - 1, clip_size)]


def read_rgb(path: Path, index: int, image_size: int) -> Image.Image:
    try:
        frame = iio.imread(path, index=index)
    except Exception:
        frame = iio.imread(path, index=0)
    image = Image.fromarray(np.asarray(frame).astype(np.uint8)).convert("RGB")
    return image.resize((image_size, image_size), Image.BILINEAR)


def read_clip_frames(video_path: Path, image_size: int, clip_size: int) -> List[Image.Image]:
    indices = clip_indices(video_path, clip_size)
    needed = {int(index) for index in indices}
    captured: Dict[int, Image.Image] = {}
    max_needed = max(needed) if needed else 0
    try:
        for frame_index, frame in enumerate(iio.imiter(video_path)):
            if frame_index in needed:
                image = Image.fromarray(np.asarray(frame).astype(np.uint8)).convert("RGB")
                captured[frame_index] = image.resize((image_size, image_size), Image.BILINEAR)
            if frame_index >= max_needed and len(captured) == len(needed):
                break
    except Exception:
        captured = {}
    if len(captured) != len(needed):
        return [read_rgb(video_path, index, image_size) for index in indices]
    return [captured[int(index)].copy() for index in indices]


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


def load_videoseal(videoseal_root: Path, model_card: str, device: str):
    if str(videoseal_root) not in sys.path:
        sys.path.insert(0, str(videoseal_root))
    old_cwd = Path.cwd()
    os.chdir(videoseal_root)
    try:
        import videoseal

        model = videoseal.load(model_card).to(device)
        model.eval()
    finally:
        os.chdir(old_cwd)
    return model


def protect_frames(
    frames: List[Image.Image],
    videoseal_model,
    device: str,
    degradation_type: str,
    degradation_level: str,
) -> List[Image.Image]:
    from torchvision.transforms import functional as TF

    tensors = torch.stack([TF.to_tensor(frame) for frame in frames], dim=0).to(device)
    with torch.inference_mode():
        embedded = videoseal_model.embed(tensors)["imgs_w"].detach().cpu().clamp(0, 1)
    protected: List[Image.Image] = []
    for tensor in embedded:
        image = TF.to_pil_image(tensor)
        protected.append(apply_degradation(image, degradation_type, degradation_level))
    return protected


def cached_put(cache: OrderedDict[str, List[Image.Image]], key: str, value: List[Image.Image], max_items: int) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_items:
        cache.popitem(last=False)


def cached_read_frames(
    cache: OrderedDict[str, List[Image.Image]],
    video_path: Path,
    image_size: int,
    clip_size: int,
    max_items: int,
) -> List[Image.Image]:
    key = f"{video_path}|{image_size}|{clip_size}"
    if key in cache:
        cache.move_to_end(key)
        return [frame.copy() for frame in cache[key]]
    frames = read_clip_frames(video_path, image_size, clip_size)
    cached_put(cache, key, [frame.copy() for frame in frames], max_items)
    return frames


def cached_protected_frames(
    cache: OrderedDict[str, List[Image.Image]],
    frames: List[Image.Image],
    video_path: Path,
    videoseal_model,
    device: str,
    max_items: int,
) -> List[Image.Image]:
    key = f"{video_path}|videoseal"
    if key in cache:
        cache.move_to_end(key)
        return [frame.copy() for frame in cache[key]]
    from torchvision.transforms import functional as TF

    tensors = torch.stack([TF.to_tensor(frame) for frame in frames], dim=0).to(device)
    with torch.inference_mode():
        embedded = videoseal_model.embed(tensors)["imgs_w"].detach().cpu().clamp(0, 1)
    protected = [TF.to_pil_image(tensor) for tensor in embedded]
    cached_put(cache, key, [frame.copy() for frame in protected], max_items)
    return protected


def row_clip_tensor(
    row: Dict[str, str],
    image_size: int,
    clip_size: int,
    protect_mode: str,
    videoseal_model,
    device: str,
    frame_cache: OrderedDict[str, List[Image.Image]],
    protected_cache: OrderedDict[str, List[Image.Image]],
    cache_items: int,
) -> tuple[torch.Tensor, Path, int]:
    video_path = raw_video_from_row(row)
    frames = cached_read_frames(frame_cache, video_path, image_size, clip_size, cache_items)
    if row["protection"] == "protected" and protect_mode == "videoseal":
        frames = cached_protected_frames(protected_cache, frames, video_path, videoseal_model, device, cache_items)
        frames = [apply_degradation(frame, row["degradation_type"], row["degradation_level"]) for frame in frames]
    arrays = [np.asarray(frame, dtype=np.float32) for frame in frames]
    clip = np.stack(arrays, axis=0)
    tensor = torch.as_tensor(clip, dtype=torch.float32).permute(3, 0, 1, 2)
    return tensor, video_path, len(frames)


def scored_result(row: Dict[str, str], video_path: Path, prob_fake: float, threshold: float, clip_size: int, protect_mode: str) -> Dict[str, str]:
    pred_label = 1 if prob_fake >= threshold else 0
    return {
        "sample_id": row["sample_id"],
        "status": "ok",
        "input_path": str(video_path),
        "protection": row["protection"],
        "prob_fake": f"{prob_fake:.6f}",
        "pred_label": str(pred_label),
        "logit_real": f"{1.0 - prob_fake:.6f}",
        "logit_fake": f"{prob_fake:.6f}",
        "feature_dim": "0",
        "feature_norm": "0.000000",
        "feature_path": "",
        "raw_video_path": str(video_path),
        "clip_frames": str(clip_size),
        "protect_mode": protect_mode,
    }


def empty_result(row: Dict[str, str], video_path: Path, status: str, protect_mode: str) -> Dict[str, str]:
    return {
        "sample_id": row["sample_id"],
        "status": status,
        "input_path": str(video_path),
        "protection": row["protection"],
        "prob_fake": "",
        "pred_label": "",
        "logit_real": "",
        "logit_fake": "",
        "feature_dim": "",
        "feature_norm": "",
        "feature_path": "",
        "raw_video_path": str(video_path),
        "clip_frames": "",
        "protect_mode": protect_mode,
    }


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Run FTCN+TT on raw 32-frame VCF clips with optional in-memory protection.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", help="Optional explicit manifest path.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--ftcn-root", default=str(root / "workspace" / "repos" / "FTCN"))
    parser.add_argument("--checkpoint", default=str(root / "workspace" / "checkpoints" / "ftcn" / "ftcn_tt.pth"))
    parser.add_argument("--videoseal-root", default=str(root / "workspace" / "repos" / "videoseal"))
    parser.add_argument("--videoseal-card", default="videoseal")
    parser.add_argument("--protect-mode", choices=["source", "videoseal"], default="source")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cache-items", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
    config = load_yaml_config(Path(args.config))
    manifest_path = Path(args.manifest or config["paths"]["manifest_path"])
    rows = load_manifest(manifest_path)
    if args.limit > 0:
        rows = rows[: args.limit]

    model, clip_size, image_size = load_ftcn(Path(args.ftcn_root), Path(args.checkpoint), args.device)
    videoseal_model = None
    if args.protect_mode == "videoseal":
        videoseal_model = load_videoseal(Path(args.videoseal_root), args.videoseal_card, args.device)

    mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255], device=args.device).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255], device=args.device).view(1, 3, 1, 1, 1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    handle = output_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
    writer.writeheader()

    batch_rows: List[Dict[str, str]] = []
    batch_paths: List[Path] = []
    batch_tensors: List[torch.Tensor] = []
    frame_cache: OrderedDict[str, List[Image.Image]] = OrderedDict()
    protected_cache: OrderedDict[str, List[Image.Image]] = OrderedDict()

    def flush() -> None:
        if not batch_rows:
            return
        images = torch.stack(batch_tensors, dim=0).to(args.device).sub(mean).div(std)
        with torch.inference_mode():
            probs = model(images)["final_output"].reshape(-1).detach().cpu().numpy().astype(float).tolist()
        for row, path, prob in zip(batch_rows, batch_paths, probs):
            writer.writerow(scored_result(row, path, float(prob), args.threshold, clip_size, args.protect_mode))
        handle.flush()
        batch_rows.clear()
        batch_paths.clear()
        batch_tensors.clear()

    try:
        for index, row in enumerate(rows, start=1):
            video_path = raw_video_from_row(row)
            if not video_path.exists():
                writer.writerow(empty_result(row, video_path, "missing_raw_video", args.protect_mode))
                continue
            try:
                tensor, resolved_path, _ = row_clip_tensor(
                    row,
                    image_size=image_size,
                    clip_size=clip_size,
                    protect_mode=args.protect_mode,
                    videoseal_model=videoseal_model,
                    device=args.device,
                    frame_cache=frame_cache,
                    protected_cache=protected_cache,
                    cache_items=args.cache_items,
                )
            except Exception as exc:
                writer.writerow(empty_result(row, video_path, f"error:{type(exc).__name__}", args.protect_mode))
                continue
            batch_rows.append(row)
            batch_paths.append(resolved_path)
            batch_tensors.append(tensor)
            if len(batch_rows) >= args.batch_size:
                flush()
            if args.progress_every > 0 and index % args.progress_every == 0:
                print(f"processed_rows={index}/{len(rows)}", flush=True)
        flush()
    finally:
        handle.close()
    print(f"Wrote FTCN+TT clip32 results to {args.output}")


if __name__ == "__main__":
    main()
