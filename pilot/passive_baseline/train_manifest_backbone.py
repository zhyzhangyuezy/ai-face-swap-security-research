from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


@dataclass(frozen=True)
class ManifestRecord:
    sample_id: str
    path: str
    label: int
    split: str
    dataset: str
    source_manifest: str


class ManifestFrameDataset(Dataset):
    def __init__(self, records: list[ManifestRecord], transform: transforms.Compose) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        image = Image.open(record.path).convert("RGB")
        return self.transform(image), record.label


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Train a passive detector from hybrid manifests and evaluate cross-dataset generalization."
    )
    parser.add_argument("--train-manifest", action="append", required=True, help="Manifest CSV. Repeat for multi-source training.")
    parser.add_argument("--eval-manifest", action="append", default=[], help="NAME=manifest.csv. Repeat for cross evaluation.")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--description", default="")
    parser.add_argument("--arch", choices=["resnet18", "efficientnet_b0", "convnext_tiny", "vit_b_16", "swin_t"], default="resnet18")
    parser.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aug-profile", choices=["basic", "robust"], default="basic")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default=str(root / "pilot" / "passive_baseline" / "runs"))
    parser.add_argument("--results-tsv", default=str(root / "pilot" / "passive_baseline" / "manifest_backbone_results.tsv"))
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_path(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def parse_eval_manifest(value: str) -> tuple[str, str]:
    if "=" not in value:
        path = Path(value)
        return path.stem, value
    name, path = value.split("=", 1)
    return name.strip(), path.strip()


def balanced_limit(records: list[ManifestRecord], max_rows: int) -> list[ManifestRecord]:
    if max_rows <= 0 or len(records) <= max_rows:
        return records
    real_rows = [record for record in records if record.label == 0]
    fake_rows = [record for record in records if record.label == 1]
    if not real_rows or not fake_rows:
        return records[:max_rows]
    half = max_rows // 2
    limited = real_rows[:half] + fake_rows[: max_rows - half]
    limited.sort(key=lambda record: (record.split, record.dataset, record.sample_id))
    return limited


def load_manifest_records(manifest_path: Path, dataset_name: str, root: Path, max_rows: int = 0) -> list[ManifestRecord]:
    rows: list[ManifestRecord] = []
    seen_pairs: set[str] = set()
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("protection") != "unprotected":
                continue
            label_text = row.get("label", "")
            if label_text not in {"real", "fake"}:
                continue
            source_path = resolve_path(row["source_path"], root)
            if not source_path.exists():
                continue
            dedup_key = row.get("pair_group_id") or row.get("sample_id") or str(source_path)
            if dedup_key in seen_pairs:
                continue
            seen_pairs.add(dedup_key)
            rows.append(
                ManifestRecord(
                    sample_id=row.get("sample_id", ""),
                    path=str(source_path),
                    label=1 if label_text == "fake" else 0,
                    split=row.get("split", "train") or "train",
                    dataset=dataset_name,
                    source_manifest=str(manifest_path),
                )
            )
    return balanced_limit(rows, max_rows)


def split_train_records(records: list[ManifestRecord], seed: int) -> dict[str, list[ManifestRecord]]:
    split_map = {"train": [], "val": [], "test": []}
    leftovers: list[ManifestRecord] = []
    for record in records:
        split = record.split.lower()
        if split in {"train", "val", "test"}:
            split_map[split].append(record)
        elif split in {"valid", "validation"}:
            split_map["val"].append(record)
        else:
            leftovers.append(record)

    if leftovers or not split_map["train"] or not split_map["val"]:
        all_rows = records[:]
        rng = random.Random(seed)
        rng.shuffle(all_rows)
        n = len(all_rows)
        n_train = max(1, math.floor(n * 0.70))
        n_val = max(1, math.floor(n * 0.15))
        split_map = {
            "train": all_rows[:n_train],
            "val": all_rows[n_train : n_train + n_val],
            "test": all_rows[n_train + n_val :],
        }
    if not split_map["test"]:
        split_map["test"] = split_map["val"]
    return split_map


def make_transforms(image_size: int, aug_profile: str) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if aug_profile == "robust":
        train_tf = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.85, 1.15)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12, hue=0.02),
                transforms.RandomGrayscale(p=0.08),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(p=0.20, scale=(0.02, 0.08), ratio=(0.3, 3.3)),
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


def make_loader(records: list[ManifestRecord], transform: transforms.Compose, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        ManifestFrameDataset(records, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_model(arch: str, weights: str) -> nn.Module:
    use_weights = weights == "imagenet"
    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if use_weights else None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model
    if arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if use_weights else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
        return model
    if arch == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if use_weights else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
        return model
    if arch == "vit_b_16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT if use_weights else None)
        model.heads.head = nn.Linear(model.heads.head.in_features, 2)
        return model
    if arch == "swin_t":
        model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT if use_weights else None)
        model.head = nn.Linear(model.head.in_features, 2)
        return model
    raise ValueError(f"Unsupported architecture: {arch}")


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
        total_loss += float(loss.item()) * labels.size(0)
        total_items += labels.size(0)
    return total_loss / max(1, total_items)


def threshold_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (probabilities >= threshold).astype(np.int64)
    real_mask = labels == 0
    fake_mask = labels == 1
    return {
        "acc": float(accuracy_score(labels, preds)),
        "balanced_acc": float(balanced_accuracy_score(labels, preds)) if len(np.unique(labels)) > 1 else float("nan"),
        "real_acc": float(np.mean(preds[real_mask] == 0)) if np.any(real_mask) else float("nan"),
        "fake_recall": float(np.mean(preds[fake_mask] == 1)) if np.any(fake_mask) else float("nan"),
    }


def select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.01, 0.99, 99):
        metrics = threshold_metrics(labels, probabilities, float(threshold))
        score = metrics["balanced_acc"]
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, best_score


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, threshold: float) -> dict[str, float]:
    model.eval()
    labels_all: list[int] = []
    probs_all: list[float] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        probs_all.extend(probs.tolist())
        labels_all.extend(labels.numpy().tolist())

    labels_np = np.asarray(labels_all, dtype=np.int64)
    probs_np = np.asarray(probs_all, dtype=np.float32)
    out = threshold_metrics(labels_np, probs_np, threshold)
    out["auc"] = float(roc_auc_score(labels_np, probs_np)) if len(np.unique(labels_np)) > 1 else float("nan")
    out["ap"] = float(average_precision_score(labels_np, probs_np)) if len(np.unique(labels_np)) > 1 else float("nan")
    out["n"] = int(labels_np.size)
    out["threshold"] = float(threshold)
    out["prob_fake_real_mean"] = float(np.mean(probs_np[labels_np == 0])) if np.any(labels_np == 0) else float("nan")
    out["prob_fake_fake_mean"] = float(np.mean(probs_np[labels_np == 1])) if np.any(labels_np == 1) else float("nan")
    return out


def ensure_results_tsv(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\t".join(
            [
                "timestamp",
                "run_name",
                "arch",
                "weights",
                "epochs",
                "best_epoch",
                "val_auc",
                "val_balanced_acc",
                "test_auc",
                "test_balanced_acc",
                "best_cross_auc",
                "best_cross_balanced_acc",
                "status",
                "description",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def append_results(path: Path, values: list[str]) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(values) + "\n")


def main() -> None:
    args = parse_args()
    root = project_root()
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    run_name = args.run_name or f"manifest_{args.arch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_records: list[ManifestRecord] = []
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

    model = build_model(args.arch, args.weights).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val_auc = -1.0
    history: list[dict[str, object]] = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        loss = run_epoch(model, train_loader, optimizer, criterion, args.device)
        val_half = evaluate(model, val_loader, args.device, threshold=0.5)
        if val_half["auc"] > best_val_auc:
            best_val_auc = val_half["auc"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        history.append({"epoch": epoch, "train_loss": loss, "val_at_0_5": val_half})
        print(json.dumps(history[-1], ensure_ascii=False))

    if best_state is not None:
        model.load_state_dict(best_state)

    # Select the operating threshold only on the validation split.
    val_probs_loader = make_loader(split_records["val"], eval_tf, args.batch_size, args.workers, shuffle=False)
    model.eval()
    labels_all: list[int] = []
    probs_all: list[float] = []
    with torch.no_grad():
        for images, labels in val_probs_loader:
            logits = model(images.to(args.device, non_blocking=True))
            probs_all.extend(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().tolist())
            labels_all.extend(labels.numpy().tolist())
    threshold, threshold_balanced_acc = select_threshold(
        np.asarray(labels_all, dtype=np.int64),
        np.asarray(probs_all, dtype=np.float32),
    )

    metrics = {
        "run_name": run_name,
        "arch": args.arch,
        "weights": args.weights,
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
        "val": evaluate(model, val_loader, args.device, threshold),
        "test": evaluate(model, test_loader, args.device, threshold),
        "cross": {
            name: evaluate(model, loader, args.device, threshold)
            for name, loader in eval_loaders.items()
        },
        "elapsed_sec": round(time.time() - start, 2),
        "history": history,
    }

    checkpoint_path = run_dir / "best.pt"
    if best_state is not None:
        torch.save(best_state, checkpoint_path)
    metrics["checkpoint"] = str(checkpoint_path)

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    ensure_results_tsv(Path(args.results_tsv))
    cross_values = list(metrics["cross"].values())
    best_cross_auc = max((item["auc"] for item in cross_values), default=float("nan"))
    best_cross_ba = max((item["balanced_acc"] for item in cross_values), default=float("nan"))
    append_results(
        Path(args.results_tsv),
        [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            run_name,
            args.arch,
            args.weights,
            str(args.epochs),
            str(best_epoch),
            f"{metrics['val']['auc']:.6f}",
            f"{metrics['val']['balanced_acc']:.6f}",
            f"{metrics['test']['auc']:.6f}",
            f"{metrics['test']['balanced_acc']:.6f}",
            f"{best_cross_auc:.6f}",
            f"{best_cross_ba:.6f}",
            "complete",
            args.description,
        ],
    )
    print(json.dumps({"metrics": str(metrics_path), **metrics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
