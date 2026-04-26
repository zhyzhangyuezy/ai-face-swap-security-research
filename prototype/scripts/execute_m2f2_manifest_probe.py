#!/usr/bin/env python3
"""Run the official M2F2-Det stage-1 detector on paper manifests."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import types
from importlib.machinery import ModuleSpec
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


RESULT_FIELDS = [
    "sample_id",
    "status",
    "input_path",
    "protection",
    "label",
    "prob_fake",
    "pred_label",
    "model",
    "degradation_type",
    "degradation_level",
    "source_group_id",
    "notes",
]


@contextmanager
def temporary_working_directory(path: Path):
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


def read_manifest(path: Path, offset: int, limit: int | None) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = rows[offset:]
    if limit is not None:
        rows = rows[:limit]
    return rows


def manifest_input_path(row: Dict[str, str]) -> Path:
    if row.get("protection") == "protected" and row.get("planned_output_path"):
        return Path(row["planned_output_path"])
    return Path(row["source_path"])


class _TorchMHA(nn.Module):
    """flash-attn compatible fallback used only for inference on Windows."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, causal: bool = False, **_: object):
        super().__init__()
        self.causal = causal
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,
        )

    def forward(self, x: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        attn_mask = None
        if self.causal:
            seq_len = x.shape[0]
            attn_mask = torch.triu(
                torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool),
                diagonal=1,
            )
        out, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        return out


def install_flash_attn_fallback() -> None:
    flash_attn = types.ModuleType("flash_attn")
    modules = types.ModuleType("flash_attn.modules")
    mha_mod = types.ModuleType("flash_attn.modules.mha")
    flash_attn.__spec__ = ModuleSpec("flash_attn", loader=None)
    modules.__spec__ = ModuleSpec("flash_attn.modules", loader=None)
    mha_mod.__spec__ = ModuleSpec("flash_attn.modules.mha", loader=None)
    mha_mod.MHA = _TorchMHA
    flash_attn.modules = modules  # type: ignore[attr-defined]
    modules.mha = mha_mod  # type: ignore[attr-defined]
    sys.modules.setdefault("flash_attn", flash_attn)
    sys.modules.setdefault("flash_attn.modules", modules)
    sys.modules.setdefault("flash_attn.modules.mha", mha_mod)


def install_transformers_lightweight_constructors() -> None:
    from transformers import (
        AutoTokenizer,
        CLIPImageProcessor,
        CLIPTextConfig,
        CLIPTextModel,
        CLIPVisionConfig,
        CLIPVisionModel,
    )

    def vision_config() -> CLIPVisionConfig:
        return CLIPVisionConfig(
            hidden_size=1024,
            intermediate_size=4096,
            projection_dim=768,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_channels=3,
            image_size=336,
            patch_size=14,
            hidden_act="quick_gelu",
            layer_norm_eps=1e-5,
        )

    def text_config() -> CLIPTextConfig:
        return CLIPTextConfig(
            vocab_size=49408,
            hidden_size=768,
            intermediate_size=3072,
            projection_dim=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            max_position_embeddings=77,
            hidden_act="quick_gelu",
            layer_norm_eps=1e-5,
            bos_token_id=49406,
            eos_token_id=49407,
            pad_token_id=1,
        )

    def vision_from_pretrained(cls, *args: object, **kwargs: object) -> CLIPVisionModel:
        return cls(vision_config())

    def text_from_pretrained(cls, *args: object, **kwargs: object) -> CLIPTextModel:
        return cls(text_config())

    def processor_from_pretrained(cls, *args: object, **kwargs: object) -> CLIPImageProcessor:
        return cls(
            do_resize=True,
            size={"shortest_edge": 336},
            do_center_crop=True,
            crop_size={"height": 336, "width": 336},
            do_rescale=True,
            rescale_factor=1 / 255,
            do_normalize=True,
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711],
        )

    class _UnusedTokenizer:
        pass

    CLIPVisionModel.from_pretrained = classmethod(vision_from_pretrained)  # type: ignore[method-assign]
    CLIPTextModel.from_pretrained = classmethod(text_from_pretrained)  # type: ignore[method-assign]
    CLIPImageProcessor.from_pretrained = classmethod(processor_from_pretrained)  # type: ignore[method-assign]
    AutoTokenizer.from_pretrained = classmethod(lambda cls, *args, **kwargs: _UnusedTokenizer())  # type: ignore[method-assign]


def install_torchvision_no_download_patch() -> None:
    from torchvision.models import efficientnet

    original_b4 = efficientnet.efficientnet_b4

    def efficientnet_b4_no_download(*args: object, **kwargs: object):
        kwargs["weights"] = None
        return original_b4(*args, **kwargs)

    efficientnet.efficientnet_b4 = efficientnet_b4_no_download  # type: ignore[assignment]


def load_m2f2(
    repo_root: Path,
    stage1_checkpoint: Path,
    vision_tower_path: Path,
    device: torch.device,
) -> nn.Module:
    install_transformers_lightweight_constructors()
    install_flash_attn_fallback()
    install_torchvision_no_download_patch()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from sequence.models.M2F2_Det.models.model import M2F2Det

    with temporary_working_directory(repo_root):
        model = M2F2Det(
            clip_text_encoder_name="openai/clip-vit-large-patch14-336",
            clip_vision_encoder_name="openai/clip-vit-large-patch14-336",
            deepfake_encoder_name="efficientnet_b4",
            hidden_size=1792,
        )
        llava_vision_tower = torch.load(vision_tower_path, map_location="cpu", weights_only=True)
        vision_tower_dict = {
            key.replace("vision_tower.", ""): value
            for key, value in llava_vision_tower.items()
        }
        if model.clip_vision_encoder is not None:
            model.clip_vision_encoder.model.load_state_dict(vision_tower_dict, strict=True)

        wrapped = torch.nn.DataParallel(model).to(device)
        checkpoint = torch.load(stage1_checkpoint, map_location="cpu", weights_only=False)
        wrapped.load_state_dict(checkpoint["model_state_dict"], strict=True)
        wrapped.eval()
        return wrapped


def batched(rows: List[Dict[str, str]], batch_size: int) -> Iterable[List[Dict[str, str]]]:
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def load_image_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return transforms.ToTensor()(image)


def run_manifest(
    rows: List[Dict[str, str]],
    model: nn.Module,
    output_csv: Path,
    device: torch.device,
    batch_size: int,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for batch_rows in tqdm(list(batched(rows, batch_size)), desc="m2f2"):
            ok_rows: List[Dict[str, str]] = []
            tensors: List[torch.Tensor] = []
            for row in batch_rows:
                input_path = manifest_input_path(row)
                if not input_path.exists():
                    writer.writerow(
                        {
                            "sample_id": row.get("sample_id", ""),
                            "status": "missing_input",
                            "input_path": str(input_path),
                            "protection": row.get("protection", ""),
                            "label": row.get("label", ""),
                            "prob_fake": "",
                            "pred_label": "",
                            "model": "M2F2-Det stage-1",
                            "degradation_type": row.get("degradation_type", ""),
                            "degradation_level": row.get("degradation_level", ""),
                            "source_group_id": row.get("source_group_id", ""),
                            "notes": row.get("notes", ""),
                        }
                    )
                    continue
                try:
                    tensors.append(load_image_tensor(input_path))
                    ok_rows.append(row)
                except Exception as exc:
                    writer.writerow(
                        {
                            "sample_id": row.get("sample_id", ""),
                            "status": f"load_error:{type(exc).__name__}",
                            "input_path": str(input_path),
                            "protection": row.get("protection", ""),
                            "label": row.get("label", ""),
                            "prob_fake": "",
                            "pred_label": "",
                            "model": "M2F2-Det stage-1",
                            "degradation_type": row.get("degradation_type", ""),
                            "degradation_level": row.get("degradation_level", ""),
                            "source_group_id": row.get("source_group_id", ""),
                            "notes": row.get("notes", ""),
                        }
                    )
            if not ok_rows:
                continue
            if len({tuple(t.shape) for t in tensors}) != 1:
                # Preserve the official in-model preprocessing while keeping batching simple.
                # Mixed image sizes are handled one at a time.
                for row, tensor in zip(ok_rows, tensors):
                    score_rows = [(row, tensor.unsqueeze(0))]
                    write_predictions(model, writer, score_rows, device)
            else:
                write_predictions(
                    model,
                    writer,
                    [(row, tensor) for row, tensor in zip(ok_rows, tensors)],
                    device,
                )


def write_predictions(
    model: nn.Module,
    writer: csv.DictWriter,
    row_tensors: List[tuple[Dict[str, str], torch.Tensor]],
    device: torch.device,
) -> None:
    rows = [row for row, _ in row_tensors]
    tensors = [tensor for _, tensor in row_tensors]
    if tensors[0].ndim == 3:
        images = torch.stack(tensors, dim=0)
    else:
        images = torch.cat(tensors, dim=0)
    images = images.to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        out = model(images, return_dict=True)
        prob_fake = F.softmax(out["pred"], dim=-1)[:, 0].detach().cpu().numpy()
    for row, score in zip(rows, prob_fake):
        input_path = manifest_input_path(row)
        writer.writerow(
            {
                "sample_id": row.get("sample_id", ""),
                "status": "ok",
                "input_path": str(input_path),
                "protection": row.get("protection", ""),
                "label": row.get("label", ""),
                "prob_fake": f"{float(score):.8f}",
                "pred_label": str(int(float(score) >= 0.5)),
                "model": "M2F2-Det stage-1",
                "degradation_type": row.get("degradation_type", ""),
                "degradation_level": row.get("degradation_level", ""),
                "source_group_id": row.get("source_group_id", ""),
                "notes": row.get("notes", ""),
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M2F2-Det detector-only checkpoint on a manifest CSV.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--repo-root", default="workspace/repos/M2F2_Det")
    parser.add_argument("--stage1-checkpoint", default="workspace/repos/M2F2_Det/checkpoints/stage_1/current_model_180.pth")
    parser.add_argument("--vision-tower", default="workspace/repos/M2F2_Det/utils/weights/vision_tower.pth")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    stage1_checkpoint = Path(args.stage1_checkpoint).resolve()
    vision_tower_path = Path(args.vision_tower).resolve()
    if not stage1_checkpoint.exists():
        raise FileNotFoundError(stage1_checkpoint)
    if not vision_tower_path.exists():
        raise FileNotFoundError(vision_tower_path)

    device = torch.device(args.device)
    rows = read_manifest(Path(args.manifest), offset=args.offset, limit=args.limit)
    model = load_m2f2(repo_root, stage1_checkpoint, vision_tower_path, device)
    run_manifest(rows, model, Path(args.output_csv), device, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
