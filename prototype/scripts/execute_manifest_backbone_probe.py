from __future__ import annotations

import argparse
import csv
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
    parser = argparse.ArgumentParser(description="Execute a manifest-trained passive backbone on a hybrid manifest.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", help="Optional explicit manifest path.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--arch", choices=["resnet18", "efficientnet_b0", "convnext_tiny", "vit_b_16", "swin_t"], required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--threshold", type=float, default=0.5, help="Only affects pred_label; prob_fake is unchanged.")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def build_model(arch: str) -> nn.Module:
    if arch == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model
    if arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
        return model
    if arch == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
        return model
    if arch == "vit_b_16":
        model = models.vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, 2)
        return model
    if arch == "swin_t":
        model = models.swin_t(weights=None)
        model.head = nn.Linear(model.head.in_features, 2)
        return model
    raise ValueError(f"Unsupported architecture: {arch}")


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: str) -> None:
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)


def make_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def extract_features(model: nn.Module, arch: str, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if arch == "resnet18":
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
        return model.fc(features), features
    if arch == "efficientnet_b0":
        x = model.features(images)
        x = model.avgpool(x)
        features = torch.flatten(x, 1)
        logits = model.classifier(features)
        return logits, features
    if arch == "convnext_tiny":
        x = model.features(images)
        x = model.avgpool(x)
        features = torch.flatten(x, 1)
        logits = model.classifier(features)
        return logits, features
    logits = model(images)
    return logits, logits


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = load_yaml_config(Path(args.config))
    manifest_path = Path(args.manifest or config["paths"]["manifest_path"])
    output_csv = Path(args.output)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    rows = load_manifest(manifest_path)
    model = build_model(args.arch).to(args.device)
    load_checkpoint(model, checkpoint_path, args.device)
    model.eval()
    transform = make_transform(args.image_size)

    output_rows: List[Dict[str, str]] = []
    budget = args.limit if args.limit > 0 else None
    with torch.no_grad():
        for index, row in enumerate(rows):
            input_path = Path(row["planned_output_path"] if row["protection"] == "protected" else row["source_path"])
            base = {
                "sample_id": row["sample_id"],
                "input_path": str(input_path),
                "protection": row["protection"],
            }
            if budget is not None and index >= budget:
                output_rows.append(
                    {
                        **base,
                        "status": "skipped_limit",
                        "prob_fake": "",
                        "pred_label": "",
                        "logit_real": "",
                        "logit_fake": "",
                        "feature_dim": "",
                        "feature_norm": "",
                        "feature_path": "",
                    }
                )
                continue
            if not input_path.exists():
                output_rows.append(
                    {
                        **base,
                        "status": "missing_input",
                        "prob_fake": "",
                        "pred_label": "",
                        "logit_real": "",
                        "logit_fake": "",
                        "feature_dim": "",
                        "feature_norm": "",
                        "feature_path": "",
                    }
                )
                continue

            image = Image.open(input_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(args.device)
            logits, features = extract_features(model, args.arch, tensor)
            probs = torch.softmax(logits, dim=1)[0]
            prob_fake = float(probs[1].item())
            pred_label = 1 if prob_fake >= args.threshold else 0
            feature_vector = features[0].detach().cpu().numpy().astype(np.float32)

            feature_path = ""
            passive_key = row.get("passive_key", "")
            if passive_key and passive_key.lower().endswith(".npz"):
                original = Path(passive_key)
                feature_output = original.with_name(f"{original.stem}_{args.arch}{original.suffix}")
                feature_output.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    feature_output,
                    sample_id=row["sample_id"],
                    input_path=str(input_path),
                    arch=args.arch,
                    prob_fake=prob_fake,
                    pred_label=pred_label,
                    feature=feature_vector,
                )
                feature_path = str(feature_output)

            output_rows.append(
                {
                    **base,
                    "status": "ok",
                    "prob_fake": f"{prob_fake:.6f}",
                    "pred_label": str(pred_label),
                    "logit_real": f"{float(logits[0, 0].item()):.6f}",
                    "logit_fake": f"{float(logits[0, 1].item()):.6f}",
                    "feature_dim": str(int(feature_vector.shape[0])),
                    "feature_norm": f"{float(np.linalg.norm(feature_vector)):.6f}",
                    "feature_path": feature_path,
                }
            )

    write_csv(output_csv, output_rows)
    print(f"Wrote {args.arch} passive results to {output_csv}")


if __name__ == "__main__":
    main()
