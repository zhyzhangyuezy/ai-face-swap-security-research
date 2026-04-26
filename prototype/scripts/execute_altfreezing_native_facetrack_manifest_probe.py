from __future__ import annotations

import argparse
import builtins
import csv
import hashlib
import io
import os
import sys
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# The released FTCN/AltFreezing face-track code predates NumPy 1.24.
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


RESULT_FIELDS = [
    "sample_id",
    "status",
    "input_path",
    "protection",
    "label",
    "prob_fake",
    "pred_label",
    "raw_video_path",
    "native_frames",
    "native_tracks",
    "native_clips",
    "protect_mode",
    "degradation_type",
    "degradation_level",
]


def _bootstrap_src() -> None:
    script_dir = Path(__file__).resolve().parent
    src_root = script_dir.parent / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


_bootstrap_src()

from hybrid_router.manifest_schema import load_manifest


@contextmanager
def temporary_working_directory(path: Path):
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


@contextmanager
def yaml_open_defaults_to_utf8():
    original_open = builtins.open

    def open_with_utf8(file, mode="r", *args, **kwargs):
        if "b" not in mode and "encoding" not in kwargs and str(file).lower().endswith((".yaml", ".yml")):
            kwargs["encoding"] = "utf-8"
        return original_open(file, mode, *args, **kwargs)

    builtins.open = open_with_utf8
    try:
        yield
    finally:
        builtins.open = original_open


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


def raw_video_from_row(row: Dict[str, str], raw_root: Path) -> Path:
    notes = parse_notes(row.get("notes", ""))
    if notes.get("raw_video_path"):
        return Path(notes["raw_video_path"])
    if {"resolution", "bucket", "stem"}.issubset(notes):
        domain = notes.get("domain", "c23")
        method = notes.get("method") or "targets"
        return (raw_root / domain / method / notes["resolution"] / "mp4" / notes["bucket"] / f"{notes['stem']}.mp4").resolve()
    return Path(row["source_path"]).resolve()


def install_altfreezing_path(altfreezing_root: Path) -> None:
    if str(altfreezing_root) not in sys.path:
        sys.path.insert(0, str(altfreezing_root))


def load_altfreezing(altfreezing_root: Path, checkpoint_path: Path, device: str):
    install_altfreezing_path(altfreezing_root)
    from config import config as cfg
    from utils.plugin_loader import PluginLoader

    with temporary_working_directory(altfreezing_root), yaml_open_defaults_to_utf8():
        cfg.init_with_yaml()
        cfg.update_with_yaml("i3d_ori.yaml")
        cfg.freeze()
        classifier = PluginLoader.get_classifier(cfg.classifier_type)()
        classifier.to(device).eval()
        loaded, _ = classifier.load(str(checkpoint_path))
    if not loaded:
        raise RuntimeError(f"Failed to load AltFreezing checkpoint: {checkpoint_path}")
    return classifier, int(cfg.clip_size), int(cfg.imsize)


def load_videoseal(videoseal_root: Path, model_card: str, device: str):
    if str(videoseal_root) not in sys.path:
        sys.path.insert(0, str(videoseal_root))
    with temporary_working_directory(videoseal_root):
        import videoseal

        model = videoseal.load(model_card).to(device)
        model.eval()
    return model


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


def protect_aligned_clip(images: np.ndarray, videoseal_model, device: str, degradation_type: str, degradation_level: str) -> np.ndarray:
    from torchvision.transforms import functional as TF

    pil_frames = [Image.fromarray(np.asarray(frame).astype(np.uint8)).convert("RGB") for frame in images]
    tensors = torch.stack([TF.to_tensor(frame) for frame in pil_frames], dim=0).to(device)
    with torch.inference_mode():
        protected = videoseal_model.embed(tensors)["imgs_w"].detach().cpu().clamp(0, 1)
    out = []
    for tensor in protected:
        image = TF.to_pil_image(tensor)
        image = apply_degradation(image, degradation_type, degradation_level)
        out.append(np.asarray(image, dtype=np.uint8))
    return np.stack(out, axis=0)


def load_native_video_state(video_path: Path, altfreezing_root: Path, max_frame: int):
    install_altfreezing_path(altfreezing_root)
    from test_tools.common import detect_all, grab_all_frames
    from test_tools.ct.operations import find_longest, multiple_tracking
    from test_tools.utils import get_crop_box

    cache_dir = project_root() / "workspace" / "cache" / "altfreezing_native_facetrack"
    cache_dir.mkdir(parents=True, exist_ok=True)
    video_hash = hashlib.sha1(str(video_path.resolve()).encode("utf-8")).hexdigest()[:12]
    cache_file = cache_dir / f"{video_hash}_{video_path.stem}_{max_frame}.pth"
    if cache_file.exists():
        detect_res, all_lm68 = torch.load(cache_file, map_location="cpu", weights_only=False)
        frames = grab_all_frames(str(video_path), max_size=max_frame, cvt=True)
    else:
        detect_res, all_lm68, frames = detect_all(str(video_path), return_frames=True, max_size=max_frame)
        torch.save((detect_res, all_lm68), cache_file)
    if not frames:
        return None
    shape = frames[0].shape[:2]
    all_detect_res = []
    for faces, faces_lm68 in zip(detect_res, all_lm68):
        new_faces = []
        for (box, lm5, score), face_lm68 in zip(faces, faces_lm68):
            new_faces.append((box, lm5, face_lm68, score))
        all_detect_res.append(new_faces)
    detect_res = all_detect_res
    tracks = multiple_tracking(detect_res)
    tuples = [(0, len(detect_res))] * len(tracks)
    if len(tracks) == 0:
        tuples, tracks = find_longest(detect_res)
    if len(tracks) == 0:
        return {
            "frames": frames,
            "data_storage": {},
            "super_clips": [],
            "tracks": 0,
        }
    data_storage = {}
    super_clips = []
    for track_i, ((start, end), track) in enumerate(zip(tuples, tracks)):
        if len(detect_res[start:end]) != len(track):
            continue
        super_clips.append(len(track))
        for face, frame_idx, j in zip(track, range(start, end), range(len(track))):
            box, lm5, lm68 = face[:3]
            big_box = get_crop_box(shape, box, scale=0.5)
            top_left = big_box[:2][None, :]
            new_lm5 = lm5 - top_left
            new_lm68 = lm68 - top_left
            new_box = (box.reshape(2, 2) - top_left).reshape(-1)
            info = (new_box, new_lm5, new_lm68, big_box)
            x1, y1, x2, y2 = big_box
            cropped = frames[frame_idx][y1:y2, x1:x2]
            base_key = f"{track_i}_{j}_"
            data_storage[f"{base_key}img"] = cropped
            data_storage[f"{base_key}ldm"] = info
            data_storage[f"{base_key}idx"] = frame_idx
    return {
        "frames": frames,
        "data_storage": data_storage,
        "super_clips": super_clips,
        "tracks": len(tracks),
    }


def clip_index_windows(super_clips: List[int], clip_size: int) -> List[List[Tuple[int, int]]]:
    windows: List[List[Tuple[int, int]]] = []
    pad_length = clip_size - 1
    for super_clip_idx, super_clip_size in enumerate(super_clips):
        inner_index = list(range(super_clip_size))
        if not inner_index:
            continue
        if super_clip_size < clip_size:
            if super_clip_size == 1:
                inner_index = inner_index * (clip_size * 2)
            else:
                post_module = inner_index[1:-1][::-1] + inner_index
                post_module = (post_module * (pad_length // max(1, len(post_module)) + 1))[:pad_length]
                pre_module = inner_index + inner_index[1:-1][::-1]
                pre_module = (pre_module * (pad_length // max(1, len(pre_module)) + 1))[-pad_length:]
                inner_index = pre_module + inner_index + post_module
        super_clip_size = len(inner_index)
        frame_range = [inner_index[i : i + clip_size] for i in range(super_clip_size) if i + clip_size <= super_clip_size]
        for indices in frame_range:
            windows.append([(super_clip_idx, t) for t in indices])
    return windows


def select_windows(windows: List[List[Tuple[int, int]]], max_clips: int) -> List[List[Tuple[int, int]]]:
    if max_clips <= 0 or len(windows) <= max_clips:
        return windows
    picks = np.linspace(0, len(windows) - 1, max_clips)
    return [windows[int(round(index))] for index in picks]


def score_video_variant(
    *,
    state,
    classifier,
    crop_align_func,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: str,
    clip_size: int,
    protect: bool,
    protect_mode: str,
    videoseal_model,
    degradation_type: str,
    degradation_level: str,
    max_clips: int,
) -> tuple[str, float | None, int]:
    windows = select_windows(clip_index_windows(state["super_clips"], clip_size), max_clips)
    if not windows:
        return "no_native_clips", None, 0
    preds: List[float] = []
    data_storage = state["data_storage"]
    for clip in windows:
        try:
            images = [data_storage[f"{i}_{j}_img"] for i, j in clip]
            landmarks = [data_storage[f"{i}_{j}_ldm"] for i, j in clip]
            _, aligned = crop_align_func(landmarks, images)
            if protect and protect_mode == "videoseal_aligned":
                aligned = protect_aligned_clip(aligned, videoseal_model, device, degradation_type, degradation_level)
            tensor = torch.as_tensor(aligned, dtype=torch.float32, device=device).permute(3, 0, 1, 2)
            tensor = tensor.unsqueeze(0).sub(mean).div(std)
            with torch.inference_mode():
                output = classifier(tensor)
            pred = F.sigmoid(output["final_output"].reshape(-1))[0]
            preds.append(float(pred.detach().cpu()))
        except Exception:
            continue
    if not preds:
        return "clip_errors", None, len(windows)
    return "ok", float(np.mean(preds)), len(windows)


def result_row(row: Dict[str, str], status: str, video_path: Path, prob_fake: float | None, threshold: float, meta: dict, protect_mode: str) -> Dict[str, str]:
    pred = "" if prob_fake is None else str(int(prob_fake >= threshold))
    return {
        "sample_id": row["sample_id"],
        "status": status,
        "input_path": row.get("source_path", ""),
        "protection": row.get("protection", ""),
        "label": row.get("label", ""),
        "prob_fake": "" if prob_fake is None else f"{prob_fake:.6f}",
        "pred_label": pred,
        "raw_video_path": str(video_path),
        "native_frames": str(meta.get("frames", "")),
        "native_tracks": str(meta.get("tracks", "")),
        "native_clips": str(meta.get("clips", "")),
        "protect_mode": protect_mode,
        "degradation_type": row.get("degradation_type", ""),
        "degradation_level": row.get("degradation_level", ""),
    }


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Run official AltFreezing native face-track video inference on manifest rows.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--raw-root", default=str(root / "workspace" / "datasets" / "raw" / "vcf"))
    parser.add_argument("--altfreezing-root", default=str(root / "workspace" / "repos" / "AltFreezing"))
    parser.add_argument("--checkpoint", default=str(root / "workspace" / "repos" / "AltFreezing" / "checkpoints" / "model.pth"))
    parser.add_argument("--videoseal-root", default=str(root / "workspace" / "repos" / "videoseal"))
    parser.add_argument("--videoseal-card", default="videoseal")
    parser.add_argument("--protect-mode", choices=["source", "videoseal_aligned"], default="source")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--max-frame", type=int, default=128)
    parser.add_argument("--max-clips-per-video", type=int, default=24)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    altfreezing_root = Path(args.altfreezing_root).resolve()
    classifier, clip_size, imsize = load_altfreezing(altfreezing_root, Path(args.checkpoint).resolve(), args.device)
    install_altfreezing_path(altfreezing_root)
    from test_tools.faster_crop_align_xray import FasterCropAlignXRay

    crop_align_func = FasterCropAlignXRay(imsize)
    videoseal_model = None
    if args.protect_mode == "videoseal_aligned":
        videoseal_model = load_videoseal(Path(args.videoseal_root).resolve(), args.videoseal_card, args.device)
    mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255], device=args.device).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255], device=args.device).view(1, 3, 1, 1, 1)

    rows = load_manifest(Path(args.manifest))
    if args.offset > 0:
        rows = rows[args.offset :]
    if args.limit > 0:
        rows = rows[: args.limit]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_cache: OrderedDict[str, object] = OrderedDict()
    score_cache: Dict[tuple[str, str, str, str], tuple[str, float | None, int]] = {}

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for index, row in enumerate(tqdm(rows, desc="altfreezing-native-rows"), start=1):
            video_path = raw_video_from_row(row, Path(args.raw_root).resolve())
            if not video_path.exists():
                writer.writerow(result_row(row, "missing_raw_video", video_path, None, args.threshold, {}, args.protect_mode))
                continue
            video_key = str(video_path)
            if video_key not in state_cache:
                if args.max_videos > 0 and len(state_cache) >= args.max_videos:
                    writer.writerow(result_row(row, "skipped_max_videos", video_path, None, args.threshold, {}, args.protect_mode))
                    continue
                state_cache[video_key] = load_native_video_state(video_path, altfreezing_root, args.max_frame)
            state = state_cache[video_key]
            if state is None:
                writer.writerow(result_row(row, "video_read_error", video_path, None, args.threshold, {}, args.protect_mode))
                continue
            protect = row.get("protection") == "protected"
            score_key = (
                video_key,
                row.get("protection", ""),
                row.get("degradation_type", ""),
                row.get("degradation_level", ""),
            )
            if score_key not in score_cache:
                status, prob, clip_count = score_video_variant(
                    state=state,
                    classifier=classifier,
                    crop_align_func=crop_align_func,
                    mean=mean,
                    std=std,
                    device=args.device,
                    clip_size=clip_size,
                    protect=protect,
                    protect_mode=args.protect_mode,
                    videoseal_model=videoseal_model,
                    degradation_type=row.get("degradation_type", "none"),
                    degradation_level=row.get("degradation_level", "clean"),
                    max_clips=args.max_clips_per_video,
                )
                score_cache[score_key] = (status, prob, clip_count)
            status, prob, clip_count = score_cache[score_key]
            meta = {
                "frames": len(state.get("frames", [])),
                "tracks": state.get("tracks", 0),
                "clips": clip_count,
            }
            writer.writerow(result_row(row, status, video_path, prob, args.threshold, meta, args.protect_mode))
            if args.progress_every > 0 and index % args.progress_every == 0:
                handle.flush()
                print(f"processed_rows={index}/{len(rows)} unique_videos={len(state_cache)}", flush=True)
    print(f"Wrote AltFreezing native face-track results to {output_path}")


if __name__ == "__main__":
    main()
