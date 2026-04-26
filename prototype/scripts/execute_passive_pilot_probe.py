from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


def _bootstrap_src() -> None:
    script_dir = Path(__file__).resolve().parent
    src_root = script_dir.parent / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


_bootstrap_src()

from hybrid_router.config_io import load_yaml_config
from hybrid_router.manifest_schema import load_manifest


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
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Execute the current passive pilot baseline on a smoke-test manifest.")
    parser.add_argument("--config", required=True, help="Path to the smoke-test YAML config.")
    parser.add_argument("--manifest", help="Optional explicit manifest path.")
    parser.add_argument("--output", help="Optional explicit CSV output path.")
    parser.add_argument(
        "--checkpoint",
        default=str(project_root / "pilot" / "passive_baseline" / "runs" / "uadfv_celebdf_resnet18_joint_v5" / "best.pt"),
        help="Path to the passive baseline checkpoint.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on executed rows.")
    return parser.parse_args()


def build_model() -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: str) -> None:
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)


def extract_logits_and_features(model: nn.Module, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = model.conv1(images)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    features = torch.flatten(x, 1)
    logits = model.fc(features)
    return logits, features


def make_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    manifest_path = Path(args.manifest or config["paths"]["manifest_path"])
    output_csv = Path(args.output or str(Path(config["paths"]["passive_jobs_path"]).with_name(f"{Path(config['paths']['passive_jobs_path']).stem.replace('_jobs', '')}_results.csv")))
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Passive checkpoint not found: {checkpoint_path}")

    manifest_rows = load_manifest(manifest_path)
    model = build_model().to(args.device)
    load_checkpoint(model, checkpoint_path, args.device)
    model.eval()
    transform = make_transform()

    rows: List[Dict[str, str]] = []
    budget = args.limit if args.limit > 0 else None

    with torch.no_grad():
        for index, row in enumerate(manifest_rows):
            input_path = Path(row["planned_output_path"] if row["protection"] == "protected" else row["source_path"])
            result = {
                "sample_id": row["sample_id"],
                "input_path": str(input_path),
                "protection": row["protection"],
            }

            if budget is not None and index >= budget:
                rows.append({
                    **result,
                    "status": "skipped_limit",
                    "prob_fake": "",
                    "pred_label": "",
                    "logit_real": "",
                    "logit_fake": "",
                    "feature_dim": "",
                    "feature_norm": "",
                    "feature_path": "",
                })
                continue

            if not input_path.exists():
                rows.append({
                    **result,
                    "status": "missing_input",
                    "prob_fake": "",
                    "pred_label": "",
                    "logit_real": "",
                    "logit_fake": "",
                    "feature_dim": "",
                    "feature_norm": "",
                    "feature_path": "",
                })
                continue

            image = Image.open(input_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(args.device)
            logits, features = extract_logits_and_features(model, tensor)
            probs = torch.softmax(logits, dim=1)[0]
            pred_label = int(torch.argmax(probs).item())
            prob_fake = float(probs[1].item())
            feature_vector = features[0].detach().cpu().numpy().astype(np.float32)

            feature_path = ""
            passive_key = row.get("passive_key", "")
            if passive_key and passive_key.lower().endswith(".npz"):
                feature_output = Path(passive_key)
                feature_output.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    feature_output,
                    sample_id=row["sample_id"],
                    input_path=str(input_path),
                    prob_fake=prob_fake,
                    pred_label=pred_label,
                    feature=feature_vector,
                )
                feature_path = str(feature_output)

            rows.append({
                **result,
                "status": "ok",
                "prob_fake": f"{prob_fake:.6f}",
                "pred_label": str(pred_label),
                "logit_real": f"{float(logits[0, 0].item()):.6f}",
                "logit_fake": f"{float(logits[0, 1].item()):.6f}",
                "feature_dim": str(int(feature_vector.shape[0])),
                "feature_norm": f"{float(np.linalg.norm(feature_vector)):.6f}",
                "feature_path": feature_path,
            })

    write_csv(output_csv, rows)
    print(f"Wrote passive pilot results to {output_csv}")


if __name__ == "__main__":
    main()
