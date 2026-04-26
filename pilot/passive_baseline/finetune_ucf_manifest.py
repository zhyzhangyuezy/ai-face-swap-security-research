from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from train_manifest_backbone import (
    ManifestRecord,
    append_results,
    balanced_limit,
    parse_eval_manifest,
    project_root,
    resolve_path,
    select_threshold,
    split_train_records,
    threshold_metrics,
)
from train_domain_adversarial_backbone import source_domain


def bootstrap_probe_helpers(root: Path) -> None:
    scripts_dir = root / "prototype" / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))


@dataclass(frozen=True)
class UCFRecord:
    base: ManifestRecord
    domain: str


class UCFPairDataset(Dataset):
    def __init__(self, records: list[UCFRecord], transform: transforms.Compose, spe_label_map: dict[str, int]) -> None:
        self.transform = transform
        self.spe_label_map = spe_label_map
        self.pairs = make_pairs(records)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        real, fake = self.pairs[index]
        real_img = Image.open(real.base.path).convert("RGB")
        fake_img = Image.open(fake.base.path).convert("RGB")
        return self.transform(real_img), self.transform(fake_img), self.spe_label_map[fake.domain]


class UCFFrameDataset(Dataset):
    def __init__(self, records: list[UCFRecord], transform: transforms.Compose) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        image = Image.open(record.base.path).convert("RGB")
        return self.transform(image), record.base.label


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Fine-tune DeepfakeBench UCF on project manifests without changing its inference form.")
    parser.add_argument("--train-manifest", action="append", required=True, help="Manifest CSV. Repeat for multi-source training.")
    parser.add_argument("--eval-manifest", action="append", default=[], help="NAME=manifest.csv. Repeat for cross evaluation.")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--description", default="")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-pairs", type=int, default=4, help="Each pair contributes one real and one fake image.")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--freeze-encoders", action="store_true")
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    parser.add_argument("--deepfakebench-root", default=str(root / "workspace" / "repos" / "DeepfakeBench"))
    parser.add_argument(
        "--detector-config",
        default=str(root / "workspace" / "repos" / "DeepfakeBench" / "training" / "config" / "detector" / "ucf.yaml"),
    )
    parser.add_argument(
        "--pretrained",
        default=str(root / "workspace" / "checkpoints" / "deepfakebench" / "pretrained" / "pretrained" / "xception-b5690688.pth"),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(root / "workspace" / "checkpoints" / "deepfakebench" / "weights" / "ucf_best.pth"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default=str(root / "pilot" / "passive_baseline" / "runs"))
    parser.add_argument("--results-tsv", default=str(root / "pilot" / "passive_baseline" / "ucf_finetune_results.tsv"))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_transform(image_size: int, train: bool) -> transforms.Compose:
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.80, 1.0), ratio=(0.90, 1.10)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.06, hue=0.01),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize])


def load_records(manifest_path: Path, root: Path, max_rows: int = 0) -> list[UCFRecord]:
    rows: list[ManifestRecord] = []
    seen_pairs: set[str] = set()
    domain = source_domain(manifest_path)
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
                    dataset=domain,
                    source_manifest=str(manifest_path),
                )
            )
    return [UCFRecord(base=record, domain=domain) for record in balanced_limit(rows, max_rows)]


def split_ucf_records(records: list[UCFRecord], seed: int) -> dict[str, list[UCFRecord]]:
    base_to_domain = {id(record.base): record.domain for record in records}
    split = split_train_records([record.base for record in records], seed)
    return {
        name: [UCFRecord(base=record, domain=base_to_domain[id(record)]) for record in split_records]
        for name, split_records in split.items()
    }


def make_pairs(records: list[UCFRecord]) -> list[tuple[UCFRecord, UCFRecord]]:
    by_domain: dict[str, dict[int, list[UCFRecord]]] = {}
    for record in records:
        by_domain.setdefault(record.domain, {0: [], 1: []})[record.base.label].append(record)
    pairs: list[tuple[UCFRecord, UCFRecord]] = []
    rng = random.Random(42)
    for domain, buckets in by_domain.items():
        real_rows = buckets[0][:]
        fake_rows = buckets[1][:]
        rng.shuffle(real_rows)
        rng.shuffle(fake_rows)
        count = min(len(real_rows), len(fake_rows))
        pairs.extend((real_rows[index], fake_rows[index]) for index in range(count))
    rng.shuffle(pairs)
    return pairs


def collate_pairs(batch: list[tuple[torch.Tensor, torch.Tensor, int]]) -> dict[str, torch.Tensor]:
    real_images = torch.stack([item[0] for item in batch], dim=0)
    fake_images = torch.stack([item[1] for item in batch], dim=0)
    labels = torch.tensor([0] * len(batch) + [1] * len(batch), dtype=torch.long)
    label_spe = torch.tensor([0] * len(batch) + [item[2] for item in batch], dtype=torch.long)
    return {"image": torch.cat([real_images, fake_images], dim=0), "label": labels, "label_spe": label_spe}


def make_frame_loader(records: list[UCFRecord], transform: transforms.Compose, batch_size: int, workers: int) -> DataLoader:
    return DataLoader(
        UCFFrameDataset(records, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def maybe_freeze_encoders(model: torch.nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if name.startswith("encoder_f.") or name.startswith("encoder_c."):
            parameter.requires_grad = False


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> dict[str, float]:
    model.train()
    total: dict[str, float] = {"overall": 0.0, "common": 0.0, "specific": 0.0, "reconstruction": 0.0, "contrastive": 0.0}
    total_items = 0
    for batch in loader:
        data_dict = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        pred_dict = model(data_dict, inference=False)
        loss_dict = model.get_losses(data_dict, pred_dict)
        loss = loss_dict["overall"]
        loss.backward()
        optimizer.step()
        batch_items = int(data_dict["label"].size(0))
        for key in total:
            if key in loss_dict:
                total[key] += float(loss_dict[key].item()) * batch_items
        total_items += batch_items
    return {key: value / max(1, total_items) for key, value in total.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: str, threshold: float) -> dict[str, float]:
    model.eval()
    labels_all: list[int] = []
    probs_all: list[float] = []
    for images, labels in loader:
        labels = labels.to(device, non_blocking=True)
        outputs = model({"image": images.to(device, non_blocking=True), "label": labels}, inference=True)
        probs = torch.softmax(outputs["cls"], dim=1)[:, 1].detach().cpu().numpy()
        probs_all.extend(probs.tolist())
        labels_all.extend(labels.detach().cpu().numpy().tolist())
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
def collect_probs(model: torch.nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels_all: list[int] = []
    probs_all: list[float] = []
    for images, labels in loader:
        labels = labels.to(device, non_blocking=True)
        outputs = model({"image": images.to(device, non_blocking=True), "label": labels}, inference=True)
        probs_all.extend(torch.softmax(outputs["cls"], dim=1)[:, 1].detach().cpu().numpy().tolist())
        labels_all.extend(labels.detach().cpu().numpy().tolist())
    return np.asarray(labels_all, dtype=np.int64), np.asarray(probs_all, dtype=np.float32)


def ensure_ucf_results_tsv(path: Path) -> None:
    header = "\t".join(
        [
            "timestamp",
            "run_name",
            "epochs",
            "best_epoch",
            "freeze_encoders",
            "val_auc",
            "val_balanced_acc",
            "test_auc",
            "test_balanced_acc",
            "min_cross_balanced_acc",
            "mean_cross_balanced_acc",
            "max_cross_auc",
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
    bootstrap_probe_helpers(root)
    from execute_deepfakebench_ucf_probe import load_ucf_model

    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    run_name = args.run_name or f"ucf_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    records: list[UCFRecord] = []
    for manifest_value in args.train_manifest:
        records.extend(load_records(resolve_path(manifest_value, root), root, args.max_train_rows))
    split_records = split_ucf_records(records, args.seed)
    spe_domains = [domain for domain in ["ffpp", "uadfv", "celebdfv1"] if any(row.domain == domain for row in split_records["train"])]
    spe_label_map = {domain: index + 1 for index, domain in enumerate(spe_domains)}

    train_tf = make_transform(args.image_size, train=True)
    eval_tf = make_transform(args.image_size, train=False)
    train_loader = DataLoader(
        UCFPairDataset(split_records["train"], train_tf, spe_label_map),
        batch_size=args.batch_pairs,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_pairs,
    )
    val_loader = make_frame_loader(split_records["val"], eval_tf, args.eval_batch_size, args.workers)
    test_loader = make_frame_loader(split_records["test"], eval_tf, args.eval_batch_size, args.workers)

    eval_specs = [parse_eval_manifest(value) for value in args.eval_manifest]
    eval_records = {name: load_records(resolve_path(path, root), root, args.max_eval_rows) for name, path in eval_specs}
    eval_loaders = {name: make_frame_loader(rows, eval_tf, args.eval_batch_size, args.workers) for name, rows in eval_records.items()}

    model = load_ucf_model(
        deepfakebench_root=Path(args.deepfakebench_root),
        detector_config_path=Path(args.detector_config),
        pretrained_path=Path(args.pretrained),
        checkpoint_path=Path(args.checkpoint),
        device=args.device,
    )
    if args.freeze_encoders:
        maybe_freeze_encoders(model)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val_auc = -1.0
    history: list[dict[str, object]] = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_summary = train_epoch(model, train_loader, optimizer, args.device)
        val_half = evaluate(model, val_loader, args.device, threshold=0.5)
        if val_half["auc"] > best_val_auc:
            best_val_auc = val_half["auc"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        history.append({"epoch": epoch, "train": train_summary, "val_at_0_5": val_half})
        print(json.dumps(history[-1], ensure_ascii=False))

    if best_state is not None:
        model.load_state_dict(best_state)

    val_labels, val_probs = collect_probs(model, val_loader, args.device)
    threshold, threshold_ba = select_threshold(val_labels, val_probs)
    cross = {name: evaluate(model, loader, args.device, threshold) for name, loader in eval_loaders.items()}
    cross_ba = [value["balanced_acc"] for value in cross.values() if not math.isnan(value["balanced_acc"])]
    cross_auc = [value["auc"] for value in cross.values() if not math.isnan(value["auc"])]
    metrics = {
        "run_name": run_name,
        "description": args.description,
        "freeze_encoders": args.freeze_encoders,
        "spe_label_map": spe_label_map,
        "best_epoch": best_epoch,
        "selected_threshold": threshold,
        "selected_threshold_val_balanced_acc": threshold_ba,
        "counts": {
            "train_pairs": len(train_loader.dataset),
            "val": len(split_records["val"]),
            "test": len(split_records["test"]),
            **{f"eval_{name}": len(rows) for name, rows in eval_records.items()},
        },
        "val": evaluate(model, val_loader, args.device, threshold),
        "test": evaluate(model, test_loader, args.device, threshold),
        "cross": cross,
        "cross_summary": {
            "min_balanced_acc": min(cross_ba) if cross_ba else float("nan"),
            "mean_balanced_acc": float(np.mean(cross_ba)) if cross_ba else float("nan"),
            "max_auc": max(cross_auc) if cross_auc else float("nan"),
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

    ensure_ucf_results_tsv(Path(args.results_tsv))
    append_results(
        Path(args.results_tsv),
        [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            run_name,
            str(args.epochs),
            str(best_epoch),
            str(args.freeze_encoders),
            f"{metrics['val']['auc']:.6f}",
            f"{metrics['val']['balanced_acc']:.6f}",
            f"{metrics['test']['auc']:.6f}",
            f"{metrics['test']['balanced_acc']:.6f}",
            f"{metrics['cross_summary']['min_balanced_acc']:.6f}",
            f"{metrics['cross_summary']['mean_balanced_acc']:.6f}",
            f"{metrics['cross_summary']['max_auc']:.6f}",
            "complete",
            args.description,
        ],
    )
    print(json.dumps({"metrics": str(metrics_path), **metrics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
