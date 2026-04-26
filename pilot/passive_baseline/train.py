from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass
class FrameRecord:
    path: str
    label: int
    video_id: str
    source: str


class FrameDataset(Dataset):
    def __init__(self, records: list[FrameRecord], transform: transforms.Compose) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        image = Image.open(record.path).convert("RGB")
        return self.transform(image), record.label


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Train a passive pilot baseline on UADFV and cross-evaluate on Celeb-DF-v1.")
    parser.add_argument("--train-real-root", default=str(project_root / "workspace" / "datasets" / "raw" / "UADFV" / "real" / "frames"))
    parser.add_argument("--train-fake-root", default=str(project_root / "workspace" / "datasets" / "raw" / "UADFV" / "fake" / "frames"))
    parser.add_argument("--cross-real-root", nargs="+", default=[
        str(project_root / "workspace" / "datasets" / "raw" / "celebdf" / "Celeb-real" / "frames"),
        str(project_root / "workspace" / "datasets" / "raw" / "celebdf" / "YouTube-real" / "frames"),
    ])
    parser.add_argument("--cross-fake-root", nargs="+", default=[
        str(project_root / "workspace" / "datasets" / "raw" / "celebdf" / "Celeb-synthesis" / "frames"),
    ])
    parser.add_argument("--output-dir", default=str(project_root / "pilot" / "passive_baseline" / "runs"))
    parser.add_argument("--results-tsv", default=str(project_root / "pilot" / "passive_baseline" / "results.tsv"))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--description", default="baseline")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-max-frames-per-video", type=int, default=12)
    parser.add_argument("--cross-max-frames-per-video", type=int, default=6)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--selection-metric", choices=["val_auc", "cross_auc", "joint_auc"], default="joint_auc")
    parser.add_argument("--joint-cross-weight", type=float, default=0.7)
    parser.add_argument("--min-val-auc", type=float, default=0.9)
    parser.add_argument("--aug-profile", choices=["basic", "robust"], default="basic")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evenly_subsample(items: list[Path], max_items: int) -> list[Path]:
    if max_items <= 0 or len(items) <= max_items:
        return items
    indices = np.linspace(0, len(items) - 1, num=max_items, dtype=int)
    return [items[index] for index in indices]


def collect_video_records(root: Path, label: int, source: str, max_frames_per_video: int) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    if not root.exists():
        return records
    for video_dir in sorted(child for child in root.iterdir() if child.is_dir()):
        frame_paths = [path for path in sorted(video_dir.iterdir()) if path.suffix.lower() in IMAGE_EXTS]
        for frame_path in evenly_subsample(frame_paths, max_frames_per_video):
            records.append(
                FrameRecord(
                    path=str(frame_path),
                    label=label,
                    video_id=video_dir.name,
                    source=source,
                )
            )
    return records


def split_by_video(real_root: Path, fake_root: Path, max_frames_per_video: int, seed: int) -> dict[str, list[FrameRecord]]:
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for root, label, source in [
        (real_root, 0, "uadfv_real"),
        (fake_root, 1, "uadfv_fake"),
    ]:
        video_dirs = sorted(child for child in root.iterdir() if child.is_dir()) if root.exists() else []
        rng.shuffle(video_dirs)
        n_total = len(video_dirs)
        n_train = max(1, int(n_total * 0.6))
        n_val = max(1, int(n_total * 0.2))
        split_map = {
            "train": video_dirs[:n_train],
            "val": video_dirs[n_train:n_train + n_val],
            "test": video_dirs[n_train + n_val:],
        }
        if not split_map["test"]:
            split_map["test"] = video_dirs[-n_val:]
        for split_name, dirs in split_map.items():
            for video_dir in dirs:
                frame_paths = [path for path in sorted(video_dir.iterdir()) if path.suffix.lower() in IMAGE_EXTS]
                for frame_path in evenly_subsample(frame_paths, max_frames_per_video):
                    splits[split_name].append(
                        FrameRecord(
                            path=str(frame_path),
                            label=label,
                            video_id=video_dir.name,
                            source=source,
                        )
                    )
    return splits


def build_cross_records(real_roots: Iterable[Path], fake_roots: Iterable[Path], max_frames_per_video: int) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    for root in real_roots:
        records.extend(collect_video_records(root, label=0, source=root.parent.name, max_frames_per_video=max_frames_per_video))
    for root in fake_roots:
        records.extend(collect_video_records(root, label=1, source=root.parent.name, max_frames_per_video=max_frames_per_video))
    return records


def make_transforms(image_size: int, aug_profile: str) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if aug_profile == "robust":
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


def make_loader(records: list[FrameRecord], transform: transforms.Compose, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    dataset = FrameDataset(records, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_model(freeze_backbone: bool) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    if freeze_backbone:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("fc.")
    return model


def run_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: str) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size
    return total_loss / max(1, total_items)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    probabilities: list[float] = []
    labels_all: list[int] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        probabilities.extend(probs.tolist())
        labels_all.extend(labels.numpy().tolist())

    labels_np = np.asarray(labels_all, dtype=np.int64)
    probs_np = np.asarray(probabilities, dtype=np.float32)
    preds_np = (probs_np >= 0.5).astype(np.int64)

    metrics = {
        "acc": float(accuracy_score(labels_np, preds_np)),
        "auc": float(roc_auc_score(labels_np, probs_np)) if len(np.unique(labels_np)) > 1 else float("nan"),
        "ap": float(average_precision_score(labels_np, probs_np)) if len(np.unique(labels_np)) > 1 else float("nan"),
    }
    return metrics


def ensure_results_file(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "timestamp\trun_name\tselection_metric\tselected_epoch\tval_auc\ttest_auc\tcross_auc\tcross_ap\tstatus\tdescription\n",
        encoding="utf-8",
    )


def append_results(path: Path, row: list[str]) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(row) + "\n")


def selection_score(epoch_row: dict, selection_metric: str, joint_cross_weight: float, min_val_auc: float) -> float:
    val_auc = epoch_row["val"]["auc"]
    cross_auc = epoch_row["cross"]["auc"]
    guardrail_penalty = 0.0 if math.isnan(val_auc) or val_auc >= min_val_auc else (min_val_auc - val_auc) * 10.0

    if selection_metric == "val_auc":
        return val_auc
    if selection_metric == "cross_auc":
        return cross_auc - guardrail_penalty
    joint_score = joint_cross_weight * cross_auc + (1.0 - joint_cross_weight) * val_auc
    return joint_score - guardrail_penalty


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_real_root = Path(args.train_real_root)
    train_fake_root = Path(args.train_fake_root)
    cross_real_roots = [Path(path) for path in args.cross_real_root]
    cross_fake_roots = [Path(path) for path in args.cross_fake_root]

    train_splits = split_by_video(train_real_root, train_fake_root, args.train_max_frames_per_video, args.seed)
    cross_records = build_cross_records(cross_real_roots, cross_fake_roots, args.cross_max_frames_per_video)

    run_name = args.run_name or f"resnet18_e{args.epochs}_lr{args.lr}_freeze{int(args.freeze_backbone)}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_tf, eval_tf = make_transforms(args.image_size, args.aug_profile)
    train_loader = make_loader(train_splits["train"], train_tf, args.batch_size, args.workers, shuffle=True)
    val_loader = make_loader(train_splits["val"], eval_tf, args.batch_size, args.workers, shuffle=False)
    test_loader = make_loader(train_splits["test"], eval_tf, args.batch_size, args.workers, shuffle=False)
    cross_loader = make_loader(cross_records, eval_tf, args.batch_size, args.workers, shuffle=False)

    model = build_model(args.freeze_backbone).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_state = None
    best_metrics = None
    best_score = None
    history = []
    started_at = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, args.device)
        val_metrics = evaluate(model, val_loader, args.device)
        test_metrics = evaluate(model, test_loader, args.device)
        cross_metrics = evaluate(model, cross_loader, args.device)
        epoch_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val": val_metrics,
            "test": test_metrics,
            "cross": cross_metrics,
        }
        history.append(epoch_row)

        current_score = selection_score(
            epoch_row,
            selection_metric=args.selection_metric,
            joint_cross_weight=args.joint_cross_weight,
            min_val_auc=args.min_val_auc,
        )

        if best_metrics is None or best_score is None or current_score > best_score:
            best_metrics = epoch_row
            best_score = current_score
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}

    duration = time.time() - started_at
    checkpoint_path = output_dir / "best.pt"
    if best_state is not None:
        torch.save(best_state, checkpoint_path)

    summary = {
        "run_name": run_name,
        "description": args.description,
        "device": args.device,
        "freeze_backbone": args.freeze_backbone,
        "aug_profile": args.aug_profile,
        "selection_metric": args.selection_metric,
        "joint_cross_weight": args.joint_cross_weight,
        "min_val_auc": args.min_val_auc,
        "duration_seconds": round(duration, 2),
        "train_count": len(train_splits["train"]),
        "val_count": len(train_splits["val"]),
        "test_count": len(train_splits["test"]),
        "cross_count": len(cross_records),
        "best_score": best_score,
        "best": best_metrics,
        "history": history,
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    ensure_results_file(Path(args.results_tsv))
    append_results(
        Path(args.results_tsv),
        [
            time.strftime("%Y-%m-%d %H:%M:%S"),
            run_name,
            args.selection_metric,
            str(best_metrics["epoch"]),
            f"{best_metrics['val']['auc']:.6f}",
            f"{best_metrics['test']['auc']:.6f}",
            f"{best_metrics['cross']['auc']:.6f}",
            f"{best_metrics['cross']['ap']:.6f}",
            "keep",
            args.description,
        ],
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
