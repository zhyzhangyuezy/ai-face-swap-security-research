from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from train_manifest_backbone import (
    ManifestRecord,
    append_results,
    balanced_limit,
    build_model,
    ensure_results_tsv,
    parse_eval_manifest,
    project_root,
    resolve_path,
    select_threshold,
    split_train_records,
    threshold_metrics,
)


@dataclass(frozen=True)
class DomainRecord:
    base: ManifestRecord
    domain: str


class DomainFrameDataset(Dataset):
    def __init__(
        self,
        records: list[DomainRecord],
        transform: transforms.Compose,
        domain_to_index: dict[str, int],
    ) -> None:
        self.records = records
        self.transform = transform
        self.domain_to_index = domain_to_index

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        record = self.records[index]
        image = Image.open(record.base.path).convert("RGB")
        domain_index = self.domain_to_index.get(record.domain, 0)
        return self.transform(image), record.base.label, domain_index


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = strength
        return value.view_as(value)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.strength * grad_output, None


def reverse_gradient(value: torch.Tensor, strength: float) -> torch.Tensor:
    return GradientReversal.apply(value, strength)


class DomainAdversarialDetector(nn.Module):
    def __init__(
        self,
        arch: str,
        weights: str,
        num_domains: int,
        domain_hidden_dim: int,
        domain_dropout: float,
    ) -> None:
        super().__init__()
        self.arch = arch
        self.detector = build_model(arch, weights)
        feature_dim = feature_dimension(self.detector, arch)
        self.domain_head = nn.Sequential(
            nn.Linear(feature_dim, domain_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(domain_dropout),
            nn.Linear(domain_hidden_dim, num_domains),
        )

    def forward(self, images: torch.Tensor, grl_strength: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        logits, features = extract_features(self.detector, self.arch, images)
        domain_logits = self.domain_head(reverse_gradient(features, grl_strength))
        return logits, domain_logits


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Train one passive detector with domain-adversarial regularization for cross-dataset generalization."
    )
    parser.add_argument("--train-manifest", action="append", required=True, help="Manifest CSV. Repeat for multi-source training.")
    parser.add_argument("--eval-manifest", action="append", default=[], help="NAME=manifest.csv. Repeat for cross evaluation.")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--description", default="")
    parser.add_argument("--arch", choices=["resnet18", "efficientnet_b0", "convnext_tiny"], default="efficientnet_b0")
    parser.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aug-profile", choices=["basic", "robust"], default="robust")
    parser.add_argument("--domain-mode", choices=["dataset", "dataset_label"], default="dataset")
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--domain-hidden-dim", type=int, default=256)
    parser.add_argument("--domain-dropout", type=float, default=0.20)
    parser.add_argument("--domain-loss-weight", type=float, default=0.30)
    parser.add_argument("--grl-max-lambda", type=float, default=0.50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default=str(root / "pilot" / "passive_baseline" / "runs"))
    parser.add_argument(
        "--results-tsv",
        default=str(root / "pilot" / "passive_baseline" / "domain_adversarial_results.tsv"),
    )
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def source_domain(manifest_path: Path) -> str:
    stem = manifest_path.stem.lower()
    if "ffpp" in stem or "faceforensics" in stem:
        return "ffpp"
    if "uadfv" in stem:
        return "uadfv"
    if "celebdfv1" in stem or "celebdf_v1" in stem:
        return "celebdfv1"
    if "celebdfv2" in stem or "celeb-df-v2" in stem:
        return "celebdfv2"
    if "dfdcp" in stem:
        return "dfdcp"
    if "df40" in stem:
        return "df40"
    return stem


def make_domain(record: ManifestRecord, dataset_domain: str, mode: str) -> str:
    if mode == "dataset_label":
        return f"{dataset_domain}_{record.label}"
    return dataset_domain


def load_domain_records(manifest_path: Path, root: Path, max_rows: int, domain_mode: str) -> list[DomainRecord]:
    dataset_domain = source_domain(manifest_path)
    base_records = load_manifest_records_local(manifest_path, dataset_domain, root, max_rows)
    return [DomainRecord(base=record, domain=make_domain(record, dataset_domain, domain_mode)) for record in base_records]


def load_manifest_records_local(manifest_path: Path, dataset_name: str, root: Path, max_rows: int = 0) -> list[ManifestRecord]:
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


def split_domain_records(records: list[DomainRecord], seed: int) -> dict[str, list[DomainRecord]]:
    base_to_domain = {id(record.base): record.domain for record in records}
    base_split = split_train_records([record.base for record in records], seed)
    return {
        split: [DomainRecord(base=record, domain=base_to_domain[id(record)]) for record in split_records]
        for split, split_records in base_split.items()
    }


def make_transforms(image_size: int, aug_profile: str) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if aug_profile == "robust":
        train_tf = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.70, 1.0), ratio=(0.80, 1.20)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.22, contrast=0.22, saturation=0.16, hue=0.025),
                transforms.RandomGrayscale(p=0.08),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.12),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(p=0.18, scale=(0.02, 0.08), ratio=(0.3, 3.3)),
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


def make_loader(
    records: list[DomainRecord],
    transform: transforms.Compose,
    domain_to_index: dict[str, int],
    batch_size: int,
    workers: int,
    shuffle: bool,
    balanced_sampler: bool = False,
) -> DataLoader:
    sampler = None
    if balanced_sampler:
        buckets: dict[tuple[str, int], int] = {}
        for record in records:
            key = (record.domain, record.base.label)
            buckets[key] = buckets.get(key, 0) + 1
        weights = [1.0 / max(1, buckets[(record.domain, record.base.label)]) for record in records]
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=True)
        shuffle = False
    return DataLoader(
        DomainFrameDataset(records, transform, domain_to_index),
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def feature_dimension(model: nn.Module, arch: str) -> int:
    if arch == "resnet18":
        return int(model.fc.in_features)
    if arch in {"efficientnet_b0", "convnext_tiny"}:
        return int(model.classifier[-1].in_features)
    raise ValueError(f"Unsupported architecture: {arch}")


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
        return model.classifier(features), features
    if arch == "convnext_tiny":
        x = model.features(images)
        x = model.avgpool(x)
        features = torch.flatten(x, 1)
        return model.classifier(features), features
    raise ValueError(f"Unsupported architecture: {arch}")


def grl_schedule(step: int, total_steps: int, max_lambda: float) -> float:
    progress = min(1.0, max(0.0, step / max(1, total_steps)))
    return float(max_lambda * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0))


def run_epoch(
    model: DomainAdversarialDetector,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    fake_criterion: nn.Module,
    domain_criterion: nn.Module,
    device: str,
    domain_loss_weight: float,
    grl_max_lambda: float,
    start_step: int,
    total_steps: int,
) -> tuple[dict[str, float], int]:
    model.train()
    total_loss = 0.0
    total_fake_loss = 0.0
    total_domain_loss = 0.0
    total_domain_correct = 0
    total_items = 0
    step = start_step
    for images, labels, domains in loader:
        step += 1
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        domains = domains.to(device, non_blocking=True)
        strength = grl_schedule(step, total_steps, grl_max_lambda)

        optimizer.zero_grad(set_to_none=True)
        fake_logits, domain_logits = model(images, grl_strength=strength)
        fake_loss = fake_criterion(fake_logits, labels)
        domain_loss = domain_criterion(domain_logits, domains)
        loss = fake_loss + domain_loss_weight * domain_loss
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_fake_loss += float(fake_loss.item()) * batch_size
        total_domain_loss += float(domain_loss.item()) * batch_size
        total_domain_correct += int((domain_logits.argmax(dim=1) == domains).sum().item())
        total_items += batch_size

    return (
        {
            "loss": total_loss / max(1, total_items),
            "fake_loss": total_fake_loss / max(1, total_items),
            "domain_loss": total_domain_loss / max(1, total_items),
            "domain_acc": total_domain_correct / max(1, total_items),
        },
        step,
    )


@torch.no_grad()
def evaluate(model: DomainAdversarialDetector, loader: DataLoader, device: str, threshold: float) -> dict[str, float]:
    model.eval()
    labels_all: list[int] = []
    probs_all: list[float] = []
    for images, labels, _domains in loader:
        images = images.to(device, non_blocking=True)
        logits, _domain_logits = model(images, grl_strength=0.0)
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


@torch.no_grad()
def collect_probs(model: DomainAdversarialDetector, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels_all: list[int] = []
    probs_all: list[float] = []
    for images, labels, _domains in loader:
        logits, _domain_logits = model(images.to(device, non_blocking=True), grl_strength=0.0)
        probs_all.extend(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().tolist())
        labels_all.extend(labels.numpy().tolist())
    return np.asarray(labels_all, dtype=np.int64), np.asarray(probs_all, dtype=np.float32)


def empty_metrics() -> dict[str, float]:
    return {
        "acc": float("nan"),
        "balanced_acc": float("nan"),
        "real_acc": float("nan"),
        "fake_recall": float("nan"),
        "auc": float("nan"),
        "ap": float("nan"),
        "n": 0,
        "threshold": float("nan"),
        "prob_fake_real_mean": float("nan"),
        "prob_fake_fake_mean": float("nan"),
    }


def summarize_domains(records: list[DomainRecord]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for record in records:
        item = summary.setdefault(record.domain, {"real": 0, "fake": 0})
        item["fake" if record.base.label else "real"] += 1
    return summary


def main() -> None:
    args = parse_args()
    root = project_root()
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    run_name = args.run_name or f"dann_{args.arch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    records: list[DomainRecord] = []
    for manifest_value in args.train_manifest:
        manifest_path = resolve_path(manifest_value, root)
        records.extend(load_domain_records(manifest_path, root, args.max_train_rows, args.domain_mode))
    split_records = split_domain_records(records, args.seed)
    domains = sorted({record.domain for record in split_records["train"]})
    domain_to_index = {domain: index for index, domain in enumerate(domains)}

    eval_specs = [parse_eval_manifest(value) for value in args.eval_manifest]
    eval_records = {
        name: load_domain_records(resolve_path(path, root), root, args.max_eval_rows, args.domain_mode)
        for name, path in eval_specs
    }

    train_tf, eval_tf = make_transforms(args.image_size, args.aug_profile)
    train_loader = make_loader(
        split_records["train"],
        train_tf,
        domain_to_index,
        args.batch_size,
        args.workers,
        shuffle=True,
        balanced_sampler=args.balanced_sampler,
    )
    val_loader = make_loader(split_records["val"], eval_tf, domain_to_index, args.batch_size, args.workers, shuffle=False)
    test_loader = make_loader(split_records["test"], eval_tf, domain_to_index, args.batch_size, args.workers, shuffle=False)
    eval_loaders = {
        name: make_loader(rows, eval_tf, domain_to_index, args.batch_size, args.workers, shuffle=False)
        for name, rows in eval_records.items()
    }

    model = DomainAdversarialDetector(
        arch=args.arch,
        weights=args.weights,
        num_domains=max(1, len(domain_to_index)),
        domain_hidden_dim=args.domain_hidden_dim,
        domain_dropout=args.domain_dropout,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    fake_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()

    best_state: dict[str, torch.Tensor] | None = None
    best_detector_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val_auc = -1.0
    history: list[dict[str, object]] = []
    start = time.time()
    step = 0
    total_steps = max(1, args.epochs * len(train_loader))

    for epoch in range(1, args.epochs + 1):
        train_summary, step = run_epoch(
            model,
            train_loader,
            optimizer,
            fake_criterion,
            domain_criterion,
            args.device,
            args.domain_loss_weight,
            args.grl_max_lambda,
            step,
            total_steps,
        )
        val_half = evaluate(model, val_loader, args.device, threshold=0.5)
        if val_half["auc"] > best_val_auc:
            best_val_auc = val_half["auc"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            best_detector_state = {key: value.detach().cpu() for key, value in model.detector.state_dict().items()}
        history.append({"epoch": epoch, "train": train_summary, "val_at_0_5": val_half})
        print(json.dumps(history[-1], ensure_ascii=False))

    if best_state is not None:
        model.load_state_dict(best_state)

    val_labels, val_probs = collect_probs(model, val_loader, args.device)
    threshold, threshold_balanced_acc = select_threshold(val_labels, val_probs)

    cross_metrics = {
        name: evaluate(model, loader, args.device, threshold) if len(eval_records[name]) else empty_metrics()
        for name, loader in eval_loaders.items()
    }
    cross_ba = [value["balanced_acc"] for value in cross_metrics.values() if not math.isnan(value["balanced_acc"])]
    cross_auc = [value["auc"] for value in cross_metrics.values() if not math.isnan(value["auc"])]
    metrics = {
        "run_name": run_name,
        "arch": args.arch,
        "weights": args.weights,
        "description": args.description,
        "domain_mode": args.domain_mode,
        "balanced_sampler": args.balanced_sampler,
        "domain_loss_weight": args.domain_loss_weight,
        "grl_max_lambda": args.grl_max_lambda,
        "device": args.device,
        "best_epoch": best_epoch,
        "selected_threshold": threshold,
        "selected_threshold_val_balanced_acc": threshold_balanced_acc,
        "domains": domain_to_index,
        "domain_counts": {
            "train": summarize_domains(split_records["train"]),
            "val": summarize_domains(split_records["val"]),
            "test": summarize_domains(split_records["test"]),
        },
        "counts": {
            "train": len(split_records["train"]),
            "val": len(split_records["val"]),
            "test": len(split_records["test"]),
            **{f"eval_{name}": len(rows) for name, rows in eval_records.items()},
        },
        "val": evaluate(model, val_loader, args.device, threshold),
        "test": evaluate(model, test_loader, args.device, threshold),
        "cross": cross_metrics,
        "cross_summary": {
            "min_balanced_acc": min(cross_ba) if cross_ba else float("nan"),
            "mean_balanced_acc": float(np.mean(cross_ba)) if cross_ba else float("nan"),
            "max_auc": max(cross_auc) if cross_auc else float("nan"),
        },
        "elapsed_sec": round(time.time() - start, 2),
        "history": history,
    }

    detector_checkpoint_path = run_dir / "best.pt"
    full_checkpoint_path = run_dir / "full_with_domain_head.pt"
    if best_detector_state is not None:
        torch.save(best_detector_state, detector_checkpoint_path)
    if best_state is not None:
        torch.save(best_state, full_checkpoint_path)
    metrics["checkpoint"] = str(detector_checkpoint_path)
    metrics["full_checkpoint"] = str(full_checkpoint_path)

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    ensure_results_tsv(Path(args.results_tsv))
    header = Path(args.results_tsv).read_text(encoding="utf-8").splitlines()[0] if Path(args.results_tsv).exists() else ""
    desired_header = "\t".join(
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
            "min_cross_balanced_acc",
            "mean_cross_balanced_acc",
            "max_cross_auc",
            "domain_mode",
            "balanced_sampler",
            "domain_loss_weight",
            "grl_max_lambda",
            "status",
            "description",
        ]
    )
    if header != desired_header:
        Path(args.results_tsv).write_text(desired_header + "\n", encoding="utf-8")
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
            f"{metrics['cross_summary']['min_balanced_acc']:.6f}",
            f"{metrics['cross_summary']['mean_balanced_acc']:.6f}",
            f"{metrics['cross_summary']['max_auc']:.6f}",
            args.domain_mode,
            str(args.balanced_sampler),
            f"{args.domain_loss_weight:.6f}",
            f"{args.grl_max_lambda:.6f}",
            "complete",
            args.description,
        ],
    )
    print(json.dumps({"metrics": str(metrics_path), **metrics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
