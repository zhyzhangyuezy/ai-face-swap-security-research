from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def parse_notes(notes: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for part in notes.split(";"):
        token = part.strip()
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def score_key(row: Dict[str, str]) -> str:
    return row.get("sample_id", "")


def join_rows(manifest_path: Path, scores_path: Path) -> List[dict]:
    scores = {score_key(row): row for row in load_csv(scores_path) if score_key(row)}
    rows: List[dict] = []
    for row in load_csv(manifest_path):
        score = scores.get(row["sample_id"])
        if not score or score.get("status") != "ok" or not score.get("prob_fake"):
            continue
        rows.append(
            {
                "sample_id": row["sample_id"],
                "label": row["label"],
                "protection": row["protection"],
                "degradation_type": row["degradation_type"],
                "degradation_level": row["degradation_level"],
                "score": float(score["prob_fake"]),
            }
        )
    return rows


def labels(rows: List[dict]) -> np.ndarray:
    return np.asarray([1 if row["label"] == "fake" else 0 for row in rows], dtype=np.int64)


def masks(rows: List[dict]) -> dict:
    y = labels(rows)
    protection = np.asarray([row["protection"] for row in rows])
    return {
        "real": y == 0,
        "fake": y == 1,
        "protected_real": (y == 0) & (protection == "protected"),
        "protected_fake": (y == 1) & (protection == "protected"),
        "unprotected_real": (y == 0) & (protection == "unprotected"),
        "unprotected_fake": (y == 1) & (protection == "unprotected"),
    }


def rate(values: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) == 0:
        return 0.0
    return float(values[mask].mean())


def compute_metrics(rows: List[dict], threshold: float) -> dict:
    y = labels(rows)
    score = np.asarray([row["score"] for row in rows], dtype=np.float32)
    mask = masks(rows)
    pred_fake = score >= threshold
    correct = pred_fake == (y == 1)
    real_acc = rate(correct, mask["real"])
    fake_acc = rate(correct, mask["fake"])
    pr_acc = rate(correct, mask["protected_real"])
    return {
        "groups": int(len(rows)),
        "accuracy": float(correct.mean()) if len(rows) else 0.0,
        "balanced_accuracy": 0.5 * (real_acc + fake_acc),
        "real_accuracy": real_acc,
        "fake_accuracy": fake_acc,
        "protected_real_accuracy": pr_acc,
        "protected_real_fpr": 1.0 - pr_acc,
        "protected_fake_recall": rate(correct, mask["protected_fake"]),
        "unprotected_real_accuracy": rate(correct, mask["unprotected_real"]),
        "unprotected_fake_recall": rate(correct, mask["unprotected_fake"]),
    }


def thresholds(step: float) -> List[float]:
    count = int(round(1.0 / step))
    values = [round(idx * step, 6) for idx in range(count + 1)]
    for value in [0.95, 0.96, 0.97, 0.98, 0.99, 0.995]:
        if value not in values:
            values.append(value)
    return sorted(values)


def row_with_prefix(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}_{key}": (f"{value:.6f}" if isinstance(value, float) else value) for key, value in metrics.items()}


def selection_key(row: dict) -> tuple[float, float, float, float, float]:
    return (
        float(row["calib_balanced_accuracy"]),
        float(row["calib_protected_fake_recall"]),
        float(row["calib_unprotected_fake_recall"]),
        -float(row["calib_protected_real_fpr"]),
        float(row["threshold"]),
    )


def select(rows: List[dict], max_calib_pr_fpr: float | None) -> dict:
    candidates = rows if max_calib_pr_fpr is None else [
        row for row in rows if float(row["calib_protected_real_fpr"]) <= max_calib_pr_fpr + 1e-12
    ]
    if not candidates:
        candidates = rows
    selected = max(candidates, key=selection_key)
    selected["threshold_rule_satisfied"] = (
        "1"
        if max_calib_pr_fpr is None or float(selected["calib_protected_real_fpr"]) <= max_calib_pr_fpr + 1e-12
        else "0"
    )
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate flat manifest-level fake scores under protected-real FPR control.")
    parser.add_argument("--calib-manifest", required=True)
    parser.add_argument("--calib-scores", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--eval-scores", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--candidates-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--threshold-step", type=float, default=0.005)
    parser.add_argument("--max-calib-pr-fpr", type=float, default=0.08)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calib_rows = join_rows(Path(args.calib_manifest), Path(args.calib_scores))
    eval_rows = join_rows(Path(args.eval_manifest), Path(args.eval_scores))
    candidate_rows: List[dict] = []
    for threshold in thresholds(args.threshold_step):
        calib_metrics = compute_metrics(calib_rows, threshold)
        eval_metrics = compute_metrics(eval_rows, threshold)
        candidate_rows.append(
            {
                "model": args.model_name,
                "threshold": f"{threshold:.6f}",
                **row_with_prefix("calib", calib_metrics),
                **row_with_prefix("heldout", eval_metrics),
                "calib_strict_fpr08": "1" if calib_metrics["protected_real_fpr"] <= 0.08 else "0",
                "heldout_strict_fpr08": "1" if eval_metrics["protected_real_fpr"] <= 0.08 else "0",
            }
        )
    selected = select(candidate_rows, args.max_calib_pr_fpr)
    summary = {
        "model": args.model_name,
        "selection": "maximize calibration BA subject to calibration protected-real FPR constraint",
        "max_calib_pr_fpr": args.max_calib_pr_fpr,
        "calib_rows": len(calib_rows),
        "heldout_rows": len(eval_rows),
        "selected": selected,
    }
    write_csv(Path(args.candidates_output), candidate_rows)
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
