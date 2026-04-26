from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from calibrate_vcf_temporal_feature_fusion import group_rows
from calibrate_vcf_temporal_fusion import compute_metrics, labels, masks, row_with_prefix, thresholds


class MlpHead(nn.Module):
    def __init__(self, input_dim: int, frames: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim * frames, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class GruHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        pooled = torch.cat([h[-2], h[-1]], dim=1)
        return self.out(pooled).squeeze(-1)


class TcnHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(1),
        )
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x.transpose(1, 2)).squeeze(-1)
        return self.out(z).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def build_sequence_tensor(groups: List[dict]) -> np.ndarray:
    rows: List[np.ndarray] = []
    for group in groups:
        feats = np.asarray(group["features"], dtype=np.float32)
        probs = np.asarray(group["values"], dtype=np.float32).reshape(-1, 1)
        rows.append(np.concatenate([feats, probs], axis=1))
    return np.stack(rows, axis=0).astype(np.float32)


def standardize(train_x: np.ndarray, eval_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean_vec = train_x.reshape(-1, train_x.shape[-1]).mean(axis=0)
    std_vec = train_x.reshape(-1, train_x.shape[-1]).std(axis=0)
    std_vec = np.where(std_vec < 1e-6, 1.0, std_vec)
    return ((train_x - mean_vec) / std_vec).astype(np.float32), ((eval_x - mean_vec) / std_vec).astype(np.float32)


def sample_weights(groups: List[dict], protected_real_weight: float) -> np.ndarray:
    y = labels(groups)
    mask = masks(groups)
    weights = np.ones_like(y, dtype=np.float32)
    for cls in [0, 1]:
        cls_count = max(1, int((y == cls).sum()))
        weights[y == cls] *= len(y) / (2.0 * cls_count)
    weights[mask["protected_real"]] *= protected_real_weight
    return weights.astype(np.float32)


def make_model(head: str, input_dim: int, frames: int, hidden_dim: int, dropout: float) -> nn.Module:
    if head == "mlp":
        return MlpHead(input_dim=input_dim, frames=frames, hidden_dim=hidden_dim, dropout=dropout)
    if head == "gru":
        return GruHead(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    if head == "tcn":
        return TcnHead(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    raise ValueError(f"Unknown head: {head}")


def train_head(
    head: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    weights: np.ndarray,
    seed: int,
    args: argparse.Namespace,
) -> nn.Module:
    set_seed(seed)
    device = torch.device(args.device)
    model = make_model(head, train_x.shape[-1], train_x.shape[1], args.hidden_dim, args.dropout).to(device)
    dataset = TensorDataset(
        torch.from_numpy(train_x),
        torch.from_numpy(train_y.astype(np.float32)),
        torch.from_numpy(weights.astype(np.float32)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    model.train()
    for _ in range(args.epochs):
        for batch_x, batch_y, batch_w in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_w = batch_w.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = (criterion(logits, batch_y) * batch_w).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
    return model


def predict_scores(model: nn.Module, x: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    model.eval()
    outputs: List[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).to(device)
            outputs.append(torch.sigmoid(model(batch)).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def select_constrained(rows: List[dict], max_calib_pr_fpr: float) -> dict:
    candidates = [row for row in rows if float(row["calib_protected_real_fpr"]) <= max_calib_pr_fpr + 1e-12]
    if not candidates:
        candidates = rows
    selected = max(
        candidates,
        key=lambda row: (
            float(row["calib_balanced_accuracy"]),
            float(row["calib_protected_fake_recall"]),
            float(row["calib_unprotected_fake_recall"]),
            -float(row["calib_protected_real_fpr"]),
            float(row["threshold"]),
        ),
    )
    selected["threshold_rule_satisfied"] = "1" if float(selected["calib_protected_real_fpr"]) <= max_calib_pr_fpr + 1e-12 else "0"
    return selected


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: List[dict]) -> dict:
    keys = [
        "heldout_balanced_accuracy",
        "heldout_protected_real_fpr",
        "heldout_protected_fake_recall",
        "heldout_unprotected_fake_recall",
        "heldout_real_accuracy",
        "heldout_fake_accuracy",
        "threshold",
    ]
    out: dict = {}
    for key in keys:
        values = [float(row[key]) for row in rows]
        out[f"{key}_mean"] = mean(values)
        out[f"{key}_std"] = pstdev(values) if len(values) > 1 else 0.0
        out[f"{key}_min"] = min(values)
        out[f"{key}_max"] = max(values)
    out["all_threshold_rules_satisfied"] = all(row["threshold_rule_satisfied"] == "1" for row in rows)
    out["all_strict_fpr08"] = all(float(row["heldout_protected_real_fpr"]) <= 0.08 for row in rows)
    out["all_fake_recall_90"] = all(
        float(row["heldout_protected_fake_recall"]) >= 0.90 and float(row["heldout_unprotected_fake_recall"]) >= 0.90
        for row in rows
    )
    out["all_ba_90"] = all(float(row["heldout_balanced_accuracy"]) >= 0.90 for row in rows)
    out["high_level_vcf_gate_pass"] = (
        out["all_threshold_rules_satisfied"] and out["all_strict_fpr08"] and out["all_fake_recall_90"] and out["all_ba_90"]
    )
    return out


def write_summary_tsv(path: Path, selected_rows: List[dict], summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for head in summary["heads"]:
        agg = summary["selected_aggregate"][head]
        rows.append(
            {
                "head": head,
                "seeds": ",".join(str(seed) for seed in summary["seeds"]),
                "calib_groups": summary["calib_groups"],
                "heldout_groups": summary["heldout_groups"],
                "heldout_ba_mean": f"{agg['heldout_balanced_accuracy_mean']:.6f}",
                "heldout_ba_std": f"{agg['heldout_balanced_accuracy_std']:.6f}",
                "protected_real_fpr_mean": f"{agg['heldout_protected_real_fpr_mean']:.6f}",
                "protected_real_fpr_std": f"{agg['heldout_protected_real_fpr_std']:.6f}",
                "protected_fake_recall_mean": f"{agg['heldout_protected_fake_recall_mean']:.6f}",
                "unprotected_fake_recall_mean": f"{agg['heldout_unprotected_fake_recall_mean']:.6f}",
                "all_strict_fpr08": str(agg["all_strict_fpr08"]).lower(),
                "all_fake_recall_90": str(agg["all_fake_recall_90"]).lower(),
                "high_level_vcf_gate_pass": str(agg["high_level_vcf_gate_pass"]).lower(),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate stronger neural temporal heads on VCF sequence features.")
    parser.add_argument("--calib-manifest", required=True)
    parser.add_argument("--calib-passive", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--eval-passive", required=True)
    parser.add_argument("--selected-output", required=True)
    parser.add_argument("--sweep-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--summary-tsv", required=True)
    parser.add_argument("--heads", nargs="+", default=["mlp", "gru", "tcn"], choices=["mlp", "gru", "tcn"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 21, 42, 84, 20260422])
    parser.add_argument("--protected-real-weight", type=float, default=4.0)
    parser.add_argument("--max-calib-pr-fpr", type=float, default=0.08)
    parser.add_argument("--threshold-step", type=float, default=0.005)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calib_groups = group_rows(Path(args.calib_manifest), Path(args.calib_passive))
    eval_groups = group_rows(Path(args.eval_manifest), Path(args.eval_passive))
    train_x_raw = build_sequence_tensor(calib_groups)
    eval_x_raw = build_sequence_tensor(eval_groups)
    train_x, eval_x = standardize(train_x_raw, eval_x_raw)
    train_y = labels(calib_groups)
    weights = sample_weights(calib_groups, args.protected_real_weight)

    sweep_rows: List[dict] = []
    selected_rows: List[dict] = []
    for head in args.heads:
        for seed in args.seeds:
            model = train_head(head, train_x, train_y, weights, seed, args)
            calib_scores = predict_scores(model, train_x, args.device, args.batch_size)
            eval_scores = predict_scores(model, eval_x, args.device, args.batch_size)
            seed_rows: List[dict] = []
            model_name = f"{head}_pr{args.protected_real_weight:g}_h{args.hidden_dim}_e{args.epochs}"
            for threshold in thresholds(args.threshold_step):
                calib_metrics = compute_metrics(calib_groups, calib_scores, threshold)
                eval_metrics = compute_metrics(eval_groups, eval_scores, threshold)
                row = {
                    "model": model_name,
                    "head": head,
                    "seed": str(seed),
                    "threshold": f"{threshold:.6f}",
                    **row_with_prefix("calib", calib_metrics),
                    **row_with_prefix("heldout", eval_metrics),
                    "calib_strict_fpr08": "1" if calib_metrics["protected_real_fpr"] <= 0.08 else "0",
                    "heldout_strict_fpr08": "1" if eval_metrics["protected_real_fpr"] <= 0.08 else "0",
                }
                seed_rows.append(row)
                sweep_rows.append(row)
            selected_rows.append(select_constrained(seed_rows, args.max_calib_pr_fpr))

    aggregate_by_head = {
        head: aggregate([row for row in selected_rows if row["head"] == head])
        for head in args.heads
    }
    summary = {
        "protocol": {
            "selection": "maximize calibration BA subject to protected-real FPR constraint",
            "max_calib_pr_fpr": args.max_calib_pr_fpr,
            "protected_real_weight": args.protected_real_weight,
            "threshold_step": args.threshold_step,
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "device": args.device,
        },
        "heads": args.heads,
        "seeds": args.seeds,
        "calib_groups": len(calib_groups),
        "heldout_groups": len(eval_groups),
        "feature_dim_per_frame": int(train_x.shape[-1]),
        "frames": int(train_x.shape[1]),
        "selected": selected_rows,
        "selected_aggregate": aggregate_by_head,
    }
    write_csv(Path(args.sweep_output), sweep_rows)
    write_csv(Path(args.selected_output), selected_rows)
    write_summary_tsv(Path(args.summary_tsv), selected_rows, summary)
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["selected_aggregate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
