from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import CLIPVisionModel


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


@dataclass(frozen=True)
class ManifestRecord:
    sample_id: str
    path: str
    label: int
    split: str
    dataset: str
    source_manifest: str
    source_group: str = ""
    protection: str = "unprotected"
    sample_weight: float = 1.0
    notes: str = ""


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


class CLIPAdapterDetector(nn.Module):
    def __init__(
        self,
        model_name: str,
        cache_dir: str,
        adapter_dim: int = 256,
        dropout: float = 0.1,
        train_vision_last_layer: bool = False,
    ) -> None:
        super().__init__()
        self.vision = CLIPVisionModel.from_pretrained(model_name, cache_dir=cache_dir)
        hidden_dim = int(self.vision.config.hidden_size)
        for param in self.vision.parameters():
            param.requires_grad = False
        if train_vision_last_layer:
            for param in self.vision.vision_model.encoder.layers[-1].parameters():
                param.requires_grad = True

        self.adapter = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, adapter_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_dim, hidden_dim),
        )
        self.adapter_scale = nn.Parameter(torch.tensor(0.2))
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        train_vision = any(param.requires_grad for param in self.vision.parameters())
        if train_vision:
            pooled = self.vision(pixel_values=images).pooler_output
        else:
            with torch.no_grad():
                pooled = self.vision(pixel_values=images).pooler_output
        adapted = pooled + self.adapter_scale * self.adapter(pooled)
        logits = self.head(adapted)
        return logits, adapted


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Train a frozen-CLIP residual adapter detector from manifests.")
    parser.add_argument("--train-manifest", action="append", required=True)
    parser.add_argument("--eval-manifest", action="append", default=[], help="NAME=manifest.csv. Repeat for cross eval.")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--description", default="")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch16")
    parser.add_argument("--cache-dir", default=str(root / "workspace" / "checkpoints" / "huggingface"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--adapter-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--train-vision-last-layer", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aug-profile", choices=["basic", "robust"], default="robust")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default=str(root / "pilot" / "passive_baseline" / "runs"))
    parser.add_argument("--results-tsv", default=str(root / "pilot" / "passive_baseline" / "clip_adapter_results.tsv"))
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


def manifest_source_group(row: dict[str, str]) -> str:
    for key in ("pair_source_group", "source_group_id", "sourceaware_group", "pair_group_id"):
        value = row.get(key, "")
        if value:
            return value
    for note in row.get("notes", "").split(";"):
        note = note.strip()
        if note.startswith("sourceaware_group="):
            return note.split("=", 1)[1].strip()
    return row.get("sample_id", "")


def manifest_sample_weight(row: dict[str, str]) -> float:
    value = (row.get("sample_weight", "") or "").strip()
    if value:
        try:
            return max(0.0, float(value))
        except ValueError:
            pass
    for note in row.get("notes", "").split(";"):
        note = note.strip()
        if note.startswith("sample_weight="):
            try:
                return max(0.0, float(note.split("=", 1)[1].strip()))
            except ValueError:
                break
    return 1.0


def load_manifest_records(
    manifest_path: Path,
    dataset_name: str,
    root: Path,
    max_rows: int = 0,
    include_protected_real: bool = False,
) -> list[ManifestRecord]:
    rows: list[ManifestRecord] = []
    seen_pairs: set[str] = set()
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            protection = row.get("protection", "")
            label_text = row.get("label", "")
            if protection != "unprotected" and not (
                include_protected_real and protection == "protected" and label_text == "real"
            ):
                continue
            if label_text not in {"real", "fake"}:
                continue
            path_value = row.get("planned_output_path", "") if protection == "protected" else row.get("source_path", "")
            source_path = resolve_path(path_value or row["source_path"], root)
            if not source_path.exists():
                continue
            if protection == "protected":
                dedup_key = f"protected:{row.get('pair_group_id') or row.get('sample_id') or str(source_path)}"
            else:
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
                    source_group=manifest_source_group(row),
                    protection=protection,
                    sample_weight=manifest_sample_weight(row),
                    notes=row.get("notes", "") or "",
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
    normalize = transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
    if aug_profile == "robust":
        train_tf = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.70, 1.0), ratio=(0.85, 1.15)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.16, contrast=0.16, saturation=0.10, hue=0.02),
                transforms.RandomGrayscale(p=0.05),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(p=0.15, scale=(0.02, 0.08), ratio=(0.3, 3.3)),
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


def trainable_parameters(model: nn.Module):
    return [param for param in model.parameters() if param.requires_grad]


def run_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: str) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(images)
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
        score = threshold_metrics(labels, probabilities, float(threshold))["balanced_acc"]
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, best_score


@torch.no_grad()
def collect_probabilities(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels_all: list[int] = []
    probs_all: list[float] = []
    for images, labels in loader:
        logits, _ = model(images.to(device, non_blocking=True))
        probs_all.extend(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().tolist())
        labels_all.extend(labels.numpy().tolist())
    return np.asarray(labels_all, dtype=np.int64), np.asarray(probs_all, dtype=np.float32)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, threshold: float) -> dict[str, float]:
    labels_np, probs_np = collect_probabilities(model, loader, device)
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
                "model_name",
                "adapter_dim",
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

    run_name = args.run_name or f"clip_adapter_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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

    model = CLIPAdapterDetector(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        adapter_dim=args.adapter_dim,
        dropout=args.dropout,
        train_vision_last_layer=args.train_vision_last_layer,
    ).to(args.device)
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=args.lr, weight_decay=args.weight_decay)
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

    val_labels, val_probs = collect_probabilities(model, val_loader, args.device)
    threshold, threshold_balanced_acc = select_threshold(val_labels, val_probs)

    metrics = {
        "run_name": run_name,
        "model_name": args.model_name,
        "adapter_dim": args.adapter_dim,
        "dropout": args.dropout,
        "train_vision_last_layer": args.train_vision_last_layer,
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
        "cross": {name: evaluate(model, loader, args.device, threshold) for name, loader in eval_loaders.items()},
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
                "train_vision_last_layer": args.train_vision_last_layer,
                "image_size": args.image_size,
                "clip_mean": CLIP_MEAN,
                "clip_std": CLIP_STD,
            },
            checkpoint_path,
        )
    metrics["checkpoint"] = checkpoint_path.as_posix()

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
            args.model_name,
            str(args.adapter_dim),
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
    print(json.dumps({"metrics": str(metrics_path), **metrics}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)
