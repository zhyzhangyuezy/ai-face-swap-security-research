from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoModel

from train_clip_adapter import (
    ManifestFrameDataset,
    append_results,
    collect_probabilities,
    evaluate,
    load_manifest_records,
    parse_eval_manifest,
    project_root,
    resolve_path,
    select_threshold,
    split_train_records,
)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FoundationAdapterDetector(nn.Module):
    def __init__(
        self,
        model_name: str,
        cache_dir: str,
        adapter_dim: int,
        dropout: float,
        train_last_layers: int,
    ) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        hidden_dim = int(self.backbone.config.hidden_size)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if train_last_layers > 0 and hasattr(self.backbone, "encoder"):
            layers = getattr(self.backbone.encoder, "layer", None)
            if layers is not None:
                for layer in list(layers)[-train_last_layers:]:
                    for parameter in layer.parameters():
                        parameter.requires_grad = True

        self.adapter = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, adapter_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_dim, hidden_dim),
        )
        self.adapter_scale = nn.Parameter(torch.tensor(0.2))
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, 2))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        train_backbone = any(parameter.requires_grad for parameter in self.backbone.parameters())
        if train_backbone:
            outputs = self.backbone(pixel_values=images)
        else:
            with torch.no_grad():
                outputs = self.backbone(pixel_values=images)
        features = outputs.last_hidden_state[:, 0]
        adapted = features + self.adapter_scale * self.adapter(features)
        return self.head(adapted), adapted


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Train a frozen foundation-vision residual adapter detector from manifests.")
    parser.add_argument("--train-manifest", action="append", required=True)
    parser.add_argument("--eval-manifest", action="append", default=[], help="NAME=manifest.csv. Repeat for cross eval.")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--description", default="")
    parser.add_argument("--model-name", default="facebook/dinov2-base")
    parser.add_argument("--cache-dir", default=str(root / "workspace" / "checkpoints" / "huggingface"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--adapter-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--train-last-layers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aug-profile", choices=["basic", "robust"], default="robust")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default=str(root / "pilot" / "passive_baseline" / "runs"))
    parser.add_argument("--results-tsv", default=str(root / "pilot" / "passive_baseline" / "foundation_adapter_results.tsv"))
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_transforms(image_size: int, aug_profile: str) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if aug_profile == "robust":
        train_tf = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.70, 1.0), ratio=(0.85, 1.15)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.16, contrast=0.16, saturation=0.10, hue=0.02),
                transforms.RandomGrayscale(p=0.05),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(p=0.12, scale=(0.02, 0.08), ratio=(0.3, 3.3)),
            ]
        )
    else:
        train_tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08),
                transforms.ToTensor(),
                normalize,
            ]
        )
    eval_tf = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize])
    return train_tf, eval_tf


def make_loader(records, transform, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        ManifestFrameDataset(records, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def trainable_parameters(model: nn.Module):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def run_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: str) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits, _features = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        total_items += labels.size(0)
    return total_loss / max(1, total_items)


def ensure_results_tsv(path: Path) -> None:
    header = "\t".join(
        [
            "timestamp",
            "run_name",
            "model_name",
            "adapter_dim",
            "train_last_layers",
            "epochs",
            "best_epoch",
            "val_auc",
            "val_balanced_acc",
            "test_auc",
            "test_balanced_acc",
            "min_cross_balanced_acc",
            "mean_cross_balanced_acc",
            "status",
            "description",
        ]
    )
    if not path.exists() or path.read_text(encoding="utf-8").splitlines()[0] != header:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(header + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = project_root()
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    run_name = args.run_name or f"foundation_adapter_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_records = []
    for manifest_value in args.train_manifest:
        manifest_path = resolve_path(manifest_value, root)
        train_records.extend(load_manifest_records(manifest_path, manifest_path.stem, root, max_rows=args.max_train_rows))
    split_records = split_train_records(train_records, args.seed)
    eval_specs = [parse_eval_manifest(value) for value in args.eval_manifest]
    eval_records = {
        name: load_manifest_records(resolve_path(path, root), name, root, max_rows=args.max_eval_rows)
        for name, path in eval_specs
    }

    train_tf, eval_tf = make_transforms(args.image_size, args.aug_profile)
    train_loader = make_loader(split_records["train"], train_tf, args.batch_size, args.workers, shuffle=True)
    val_loader = make_loader(split_records["val"], eval_tf, args.batch_size, args.workers, shuffle=False)
    test_loader = make_loader(split_records["test"], eval_tf, args.batch_size, args.workers, shuffle=False)
    eval_loaders = {
        name: make_loader(records, eval_tf, args.batch_size, args.workers, shuffle=False)
        for name, records in eval_records.items()
    }

    model = FoundationAdapterDetector(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        adapter_dim=args.adapter_dim,
        dropout=args.dropout,
        train_last_layers=args.train_last_layers,
    ).to(args.device)
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_epoch = -1
    best_val_auc = -1.0
    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, args.device)
        val_half = evaluate(model, val_loader, args.device, threshold=0.5)
        if val_half["auc"] > best_val_auc:
            best_val_auc = val_half["auc"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        history.append({"epoch": epoch, "train_loss": train_loss, "val_at_0_5": val_half})
        print(json.dumps(history[-1], ensure_ascii=False), flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    val_labels, val_probs = collect_probabilities(model, val_loader, args.device)
    threshold, threshold_balanced_acc = select_threshold(val_labels, val_probs)
    cross = {name: evaluate(model, loader, args.device, threshold) for name, loader in eval_loaders.items()}
    cross_ba = [value["balanced_acc"] for value in cross.values() if not math.isnan(value["balanced_acc"])]
    metrics = {
        "run_name": run_name,
        "model_name": args.model_name,
        "adapter_dim": args.adapter_dim,
        "train_last_layers": args.train_last_layers,
        "description": args.description,
        "device": args.device,
        "best_epoch": best_epoch,
        "selected_threshold": threshold,
        "selected_threshold_val_balanced_acc": threshold_balanced_acc,
        "counts": {
            "train": len(split_records["train"]),
            "val": len(split_records["val"]),
            "test": len(split_records["test"]),
            **{f"eval_{name}": len(records) for name, records in eval_records.items()},
        },
        "trainable_parameters": int(sum(parameter.numel() for parameter in trainable_parameters(model))),
        "val": evaluate(model, val_loader, args.device, threshold),
        "test": evaluate(model, test_loader, args.device, threshold),
        "cross": cross,
        "cross_summary": {
            "min_balanced_acc": min(cross_ba) if cross_ba else float("nan"),
            "mean_balanced_acc": float(np.mean(cross_ba)) if cross_ba else float("nan"),
        },
        "elapsed_sec": round(time.time() - start, 2),
        "history": history,
    }

    checkpoint_path = run_dir / "best.pt"
    if best_state is not None:
        torch.save(
            {
                "state_dict": best_state,
                "model_name": args.model_name,
                "adapter_dim": args.adapter_dim,
                "dropout": args.dropout,
                "train_last_layers": args.train_last_layers,
                "image_size": args.image_size,
                "mean": IMAGENET_MEAN,
                "std": IMAGENET_STD,
            },
            checkpoint_path,
        )
    metrics["checkpoint"] = str(checkpoint_path)
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    ensure_results_tsv(Path(args.results_tsv))
    append_results(
        Path(args.results_tsv),
        [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            run_name,
            args.model_name,
            str(args.adapter_dim),
            str(args.train_last_layers),
            str(args.epochs),
            str(best_epoch),
            f"{metrics['val']['auc']:.6f}",
            f"{metrics['val']['balanced_acc']:.6f}",
            f"{metrics['test']['auc']:.6f}",
            f"{metrics['test']['balanced_acc']:.6f}",
            f"{metrics['cross_summary']['min_balanced_acc']:.6f}",
            f"{metrics['cross_summary']['mean_balanced_acc']:.6f}",
            "complete",
            args.description,
        ],
    )
    print(json.dumps({"metrics": str(metrics_path), **metrics}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)
