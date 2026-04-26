from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def _bootstrap_paths() -> None:
    script_dir = Path(__file__).resolve().parent
    src_root = script_dir.parent / "src"
    passive_root = script_dir.parents[1] / "pilot" / "passive_baseline"
    for path in [src_root, passive_root]:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


_bootstrap_paths()

from hybrid_router.config_io import load_yaml_config
from hybrid_router.manifest_schema import load_manifest
from train_clip_adapter import CLIP_MEAN, CLIP_STD
from train_effort_manifest import EffortStyleDetector, HighFrequencyEmphasis


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute an Effort-style CLIP SVD residual detector on a hybrid manifest.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", help="Optional explicit manifest path.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def make_transform(image_size: int, frequency_emphasis: float = 0.0) -> transforms.Compose:
    tensor_steps = [transforms.ToTensor()]
    if frequency_emphasis > 0:
        tensor_steps.append(HighFrequencyEmphasis(frequency_emphasis))
    tensor_steps.append(transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD))
    return transforms.Compose([transforms.Resize((image_size, image_size)), *tensor_steps])


def load_model(checkpoint_path: Path, cache_dir: str, device: str) -> tuple[EffortStyleDetector, int, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = checkpoint["model_name"]
    residual_rank = int(checkpoint["residual_rank"])
    dropout = float(checkpoint.get("dropout", 0.1))
    image_size = int(checkpoint.get("image_size", 224))
    pooling = checkpoint.get("pooling", "cls")
    attn_hidden = int(checkpoint.get("attn_hidden", 256))
    mean_gate_init = float(checkpoint.get("mean_gate_init", 0.2))
    pool_topk_frac = float(checkpoint.get("pool_topk_frac", 0.25))
    frequency_emphasis = float(checkpoint.get("frequency_emphasis", 0.0))
    use_patch_mil = bool(checkpoint.get("use_patch_mil", float(checkpoint.get("patch_mil_weight", 0.0)) > 0))
    head_type = checkpoint.get("head_type", "linear")
    cosine_scale_init = float(checkpoint.get("cosine_scale_init", 16.0))
    real_concept_prototypes = int(checkpoint.get("real_concept_prototypes", 16))
    real_concept_scale_init = float(checkpoint.get("real_concept_scale_init", 16.0))
    model = EffortStyleDetector(
        model_name=model_name,
        cache_dir=cache_dir or str(Path(__file__).resolve().parents[2] / "workspace" / "checkpoints" / "huggingface"),
        residual_rank=residual_rank,
        dropout=dropout,
        pooling=pooling,
        attn_hidden=attn_hidden,
        mean_gate_init=mean_gate_init,
        use_patch_mil=use_patch_mil,
        head_type=head_type,
        cosine_scale_init=cosine_scale_init,
        pool_topk_frac=pool_topk_frac,
        real_concept_prototypes=real_concept_prototypes,
        real_concept_scale_init=real_concept_scale_init,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, image_size, frequency_emphasis


def resolve_input(row: Dict[str, str]) -> Path:
    if row["protection"] == "protected":
        return Path(row["planned_output_path"])
    return Path(row["source_path"])


def feature_output_path(row: Dict[str, str]) -> Path | None:
    passive_key = row.get("passive_key", "")
    if not passive_key or not passive_key.lower().endswith(".npz"):
        return None
    original = Path(passive_key)
    return original.with_name(f"{original.stem}_effort{original.suffix}")


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


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def flush_batch(
    model: EffortStyleDetector,
    batch_rows: List[Dict[str, str]],
    batch_tensors: List[torch.Tensor],
    batch_paths: List[Path],
    output_rows: List[Dict[str, str]],
    device: str,
    threshold: float,
) -> None:
    if not batch_rows:
        return
    images = torch.stack(batch_tensors, dim=0).to(device)
    with torch.no_grad():
        logits, features = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1]
    for row, input_path, logit, feature, prob in zip(batch_rows, batch_paths, logits, features, probs):
        prob_fake = float(prob.detach().cpu().item())
        pred_label = 1 if prob_fake >= threshold else 0
        feature_vector = feature.detach().cpu().numpy().astype(np.float32)
        feature_path = feature_output_path(row)
        feature_path_text = ""
        if feature_path is not None:
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                feature_path,
                sample_id=row["sample_id"],
                input_path=str(input_path),
                arch="effort_clip_svd",
                prob_fake=prob_fake,
                pred_label=pred_label,
                feature=feature_vector,
            )
            feature_path_text = str(feature_path)
        output_rows.append(
            {
                "sample_id": row["sample_id"],
                "status": "ok",
                "input_path": str(input_path),
                "protection": row["protection"],
                "prob_fake": f"{prob_fake:.6f}",
                "pred_label": str(pred_label),
                "logit_real": f"{float(logit[0].detach().cpu().item()):.6f}",
                "logit_fake": f"{float(logit[1].detach().cpu().item()):.6f}",
                "feature_dim": str(int(feature_vector.shape[0])),
                "feature_norm": f"{float(np.linalg.norm(feature_vector)):.6f}",
                "feature_path": feature_path_text,
            }
        )


def main() -> None:
    args = parse_args()
    config = load_yaml_config(Path(args.config))
    manifest_path = Path(args.manifest or config["paths"]["manifest_path"])
    model, image_size, frequency_emphasis = load_model(Path(args.checkpoint), args.cache_dir, args.device)
    transform = make_transform(image_size, frequency_emphasis)
    rows = load_manifest(manifest_path)
    output_rows: List[Dict[str, str]] = []
    batch_rows: List[Dict[str, str]] = []
    batch_tensors: List[torch.Tensor] = []
    batch_paths: List[Path] = []
    budget = args.limit if args.limit > 0 else None

    for index, row in enumerate(rows):
        input_path = resolve_input(row)
        if budget is not None and index >= budget:
            output_rows.append(empty_result(row, input_path, "skipped_limit"))
            continue
        if not input_path.exists():
            output_rows.append(empty_result(row, input_path, "missing_input"))
            continue
        image = Image.open(input_path).convert("RGB")
        batch_rows.append(row)
        batch_tensors.append(transform(image))
        batch_paths.append(input_path)
        if len(batch_rows) >= args.batch_size:
            flush_batch(model, batch_rows, batch_tensors, batch_paths, output_rows, args.device, args.threshold)
            batch_rows, batch_tensors, batch_paths = [], [], []
    flush_batch(model, batch_rows, batch_tensors, batch_paths, output_rows, args.device, args.threshold)
    write_csv(Path(args.output), output_rows)
    print(f"Wrote Effort-style passive results to {args.output}")


if __name__ == "__main__":
    main()
