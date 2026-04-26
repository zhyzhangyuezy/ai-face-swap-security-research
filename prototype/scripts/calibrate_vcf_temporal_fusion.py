from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median
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


def group_temporal_rows(manifest_path: Path, passive_path: Path) -> List[dict]:
    manifest_rows = load_csv(manifest_path)
    passive_rows = {row["sample_id"]: row for row in load_csv(passive_path) if row.get("sample_id")}
    grouped: Dict[str, List[dict]] = {}
    for row in manifest_rows:
        passive = passive_rows.get(row["sample_id"], {})
        if passive.get("status") != "ok" or not passive.get("prob_fake"):
            continue
        notes = parse_notes(row.get("notes", ""))
        temporal_sample_id = notes.get("temporal_sample_id") or row["sample_id"]
        slot = int(notes.get("temporal_frame_slot", "0"))
        grouped.setdefault(temporal_sample_id, []).append(
            {
                "slot": slot,
                "prob_fake": float(passive["prob_fake"]),
                "label": row["label"],
                "protection": row["protection"],
                "degradation_type": row["degradation_type"],
                "degradation_level": row["degradation_level"],
            }
        )

    output: List[dict] = []
    for temporal_sample_id, rows in grouped.items():
        rows.sort(key=lambda item: item["slot"])
        values = [row["prob_fake"] for row in rows]
        if len(values) < 3:
            continue
        output.append(
            {
                "temporal_sample_id": temporal_sample_id,
                "label": rows[0]["label"],
                "protection": rows[0]["protection"],
                "degradation_type": rows[0]["degradation_type"],
                "degradation_level": rows[0]["degradation_level"],
                "values": values[:3],
            }
        )
    return output


def feature_matrix(groups: List[dict]) -> np.ndarray:
    rows: List[List[float]] = []
    for group in groups:
        p0, p1, p2 = group["values"]
        vals = [p0, p1, p2]
        ordered = sorted(vals, reverse=True)
        rows.append(
            [
                p0,
                p1,
                p2,
                mean(vals),
                median(vals),
                max(vals),
                min(vals),
                float(np.std(vals)),
                max(vals) - min(vals),
                mean(ordered[:2]),
                mean(vals[1:]),
                p2 - p0,
                abs(p2 - p0),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def labels(groups: List[dict]) -> np.ndarray:
    return np.asarray([1 if group["label"] == "fake" else 0 for group in groups], dtype=np.int64)


def masks(groups: List[dict]) -> dict:
    y = labels(groups)
    protection = np.asarray([group["protection"] for group in groups])
    return {
        "real": y == 0,
        "fake": y == 1,
        "protected": protection == "protected",
        "unprotected": protection == "unprotected",
        "protected_real": (y == 0) & (protection == "protected"),
        "protected_fake": (y == 1) & (protection == "protected"),
        "unprotected_real": (y == 0) & (protection == "unprotected"),
        "unprotected_fake": (y == 1) & (protection == "unprotected"),
    }


def rate(correct: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) == 0:
        return 0.0
    return float(correct[mask].mean())


def compute_metrics(groups: List[dict], scores: np.ndarray, threshold: float) -> dict:
    y = labels(groups)
    mask = masks(groups)
    pred_fake = scores >= threshold
    correct = pred_fake == (y == 1)
    real_accuracy = rate(correct, mask["real"])
    fake_accuracy = rate(correct, mask["fake"])
    protected_real_accuracy = rate(correct, mask["protected_real"])
    return {
        "groups": int(len(groups)),
        "accuracy": float(correct.mean()) if len(groups) else 0.0,
        "balanced_accuracy": 0.5 * (real_accuracy + fake_accuracy),
        "real_accuracy": real_accuracy,
        "fake_accuracy": fake_accuracy,
        "protected_real_accuracy": protected_real_accuracy,
        "protected_real_fpr": 1.0 - protected_real_accuracy,
        "protected_fake_recall": rate(correct, mask["protected_fake"]),
        "unprotected_real_accuracy": rate(correct, mask["unprotected_real"]),
        "unprotected_fake_recall": rate(correct, mask["unprotected_fake"]),
    }


def aggregate_score(groups: List[dict], strategy: str) -> np.ndarray:
    scores: List[float] = []
    for group in groups:
        vals = group["values"]
        if strategy == "mean":
            score = mean(vals)
        elif strategy == "median":
            score = median(vals)
        elif strategy == "max":
            score = max(vals)
        elif strategy == "top2_mean":
            score = mean(sorted(vals, reverse=True)[:2])
        elif strategy == "midlate_mean":
            score = mean(vals[1:])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        scores.append(float(score))
    return np.asarray(scores, dtype=np.float32)


def logistic_scores(train_groups: List[dict], eval_groups: List[dict]) -> Dict[str, tuple[np.ndarray, np.ndarray]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    x_train = feature_matrix(train_groups)
    y_train = labels(train_groups)
    x_eval = feature_matrix(eval_groups)
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_eval = scaler.transform(x_eval)

    sample_weight_base = np.ones_like(y_train, dtype=np.float32)
    train_masks = masks(train_groups)
    variants: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for c_value in [0.1, 1.0, 10.0]:
        for weight_name, protected_real_weight in [("bal", 1.0), ("pr2", 2.0), ("pr4", 4.0)]:
            sample_weight = sample_weight_base.copy()
            sample_weight[train_masks["protected_real"]] *= protected_real_weight
            model = LogisticRegression(C=c_value, max_iter=2000, class_weight="balanced", solver="lbfgs")
            model.fit(x_train, y_train, sample_weight=sample_weight)
            name = f"logreg_{weight_name}_c{c_value:g}"
            variants[name] = (
                model.predict_proba(x_train)[:, 1].astype(np.float32),
                model.predict_proba(x_eval)[:, 1].astype(np.float32),
            )
    return variants


def thresholds(step: float) -> List[float]:
    count = int(round(1.0 / step))
    values = [round(idx * step, 6) for idx in range(count + 1)]
    for value in [0.95, 0.96, 0.97, 0.98, 0.99, 0.995]:
        rounded = round(value, 6)
        if rounded not in values:
            values.append(rounded)
    return sorted(values)


def row_with_prefix(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}_{key}": (f"{value:.6f}" if isinstance(value, float) else value) for key, value in metrics.items()}


def selection_key(row: dict) -> tuple[float, float, float, float]:
    return (
        float(row["calib_balanced_accuracy"]),
        float(row["calib_protected_fake_recall"]),
        float(row["calib_unprotected_fake_recall"]),
        -float(row["calib_protected_real_fpr"]),
    )


def best_under(rows: List[dict], max_fpr: float | None) -> dict | None:
    candidates = rows if max_fpr is None else [row for row in rows if float(row["calib_protected_real_fpr"]) <= max_fpr]
    if not candidates:
        return None
    return max(candidates, key=selection_key)


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
    parser = argparse.ArgumentParser(description="Calibrate temporal VCF fusion on calib and evaluate once on heldout.")
    parser.add_argument("--calib-manifest", required=True)
    parser.add_argument("--calib-passive", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--eval-passive", required=True)
    parser.add_argument("--candidates-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--threshold-step", type=float, default=0.005)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calib_groups = group_temporal_rows(Path(args.calib_manifest), Path(args.calib_passive))
    eval_groups = group_temporal_rows(Path(args.eval_manifest), Path(args.eval_passive))

    score_sets: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for strategy in ["mean", "median", "top2_mean", "midlate_mean", "max"]:
        score_sets[strategy] = (aggregate_score(calib_groups, strategy), aggregate_score(eval_groups, strategy))
    score_sets.update(logistic_scores(calib_groups, eval_groups))

    candidate_rows: List[dict] = []
    for model_name, (calib_scores, eval_scores) in score_sets.items():
        for threshold in thresholds(args.threshold_step):
            calib_metrics = compute_metrics(calib_groups, calib_scores, threshold)
            eval_metrics = compute_metrics(eval_groups, eval_scores, threshold)
            candidate_rows.append(
                {
                    "score_model": model_name,
                    "threshold": f"{threshold:.6f}",
                    **row_with_prefix("calib", calib_metrics),
                    **row_with_prefix("heldout", eval_metrics),
                    "calib_strict_fpr08": "1" if calib_metrics["protected_real_fpr"] <= 0.08 else "0",
                    "heldout_strict_fpr08": "1" if eval_metrics["protected_real_fpr"] <= 0.08 else "0",
                }
            )

    selections = {
        "best_by_calib_any": best_under(candidate_rows, None),
        "best_by_calib_fpr08": best_under(candidate_rows, 0.08),
        "best_by_calib_fpr12": best_under(candidate_rows, 0.12),
        "best_by_calib_fpr20": best_under(candidate_rows, 0.20),
    }
    summary = {
        "calib_manifest": str(Path(args.calib_manifest).resolve()),
        "eval_manifest": str(Path(args.eval_manifest).resolve()),
        "calib_groups": len(calib_groups),
        "eval_groups": len(eval_groups),
        "score_models": sorted(score_sets),
        "selections": selections,
    }

    write_csv(Path(args.candidates_output), candidate_rows)
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
