from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _bootstrap_scripts() -> None:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))


_bootstrap_scripts()

from run_routing_sanity import compute_unprotected_map, load_csv, route_row
from search_routing_thresholds import dataset_name_from_path, validate_bridge_rows


FEATURE_NAMES = [
    "passive_prob_fake",
    "passive_logit_margin",
    "passive_delta",
    "passive_delta_abs",
    "is_protected",
    "prov_bit_accuracy",
    "prov_detect_prob",
    "prov_watermark_l1_mean",
    "deg_jpeg",
    "deg_resize",
]


@dataclass(frozen=True)
class Sample:
    dataset: str
    sample_id: str
    label: int
    protection: str
    features: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a tiny learned routing calibrator with leave-one-dataset-out splits.")
    parser.add_argument("--bridge-csv", action="append", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--max-protected-real-fpr", type=float, default=0.40)
    parser.add_argument("--min-protected-fake-recall", type=float, default=0.55)
    parser.add_argument("--min-unprotected-fake-recall", type=float, default=0.375)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def safe_float(value: str, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def base_rules(passive_threshold: float, anomaly_threshold: float) -> dict:
    return {
        "route_labels": {
            "trust": {"min_bit_accuracy": 0.975, "max_passive_delta_abs": 0.05},
            "suspicious": {"min_bit_accuracy": 0.94, "max_passive_delta_abs": 0.05},
            "abstain": {"require_protected": True},
        },
        "fallback": {
            "passive_fake_threshold": passive_threshold,
            "protected_passive_fake_threshold": passive_threshold,
            "protected_delta_abs_anomaly_threshold": anomaly_threshold,
            "min_passive_confidence_gap": 0.45,
        },
    }


def rows_to_samples(path: Path) -> list[Sample]:
    rows = load_csv(path)
    validate_bridge_rows(path, rows)
    routed_rows = [route_row(row, compute_unprotected_map(rows), base_rules(0.01, 0.004)) for row in rows]
    dataset = dataset_name_from_path(path)

    samples: list[Sample] = []
    for row in routed_rows:
        if row.get("passive_status") != "ok":
            continue
        label = 1 if row.get("label") == "fake" else 0
        features = [
            safe_float(row.get("passive_prob_fake", "")),
            safe_float(row.get("passive_logit_fake", "")) - safe_float(row.get("passive_logit_real", "")),
            safe_float(row.get("passive_delta", "")),
            safe_float(row.get("passive_delta_abs", "")),
            1.0 if row.get("protection") == "protected" else 0.0,
            safe_float(row.get("prov_bit_accuracy", "")),
            safe_float(row.get("prov_detect_prob", "")),
            safe_float(row.get("prov_watermark_l1_mean", "")),
            1.0 if row.get("degradation_type") == "jpeg" else 0.0,
            1.0 if row.get("degradation_type") == "resize" else 0.0,
        ]
        samples.append(
            Sample(
                dataset=dataset,
                sample_id=row.get("sample_id", ""),
                label=label,
                protection=row.get("protection", ""),
                features=features,
            )
        )
    return samples


def metrics_from_predictions(samples: list[Sample], predictions: np.ndarray) -> dict:
    labels = np.asarray([sample.label for sample in samples], dtype=np.int64)
    predictions = predictions.astype(np.int64)
    correct = (labels == predictions).astype(np.float32)

    def subset_accuracy(label: int | None = None, protection: str | None = None) -> float:
        mask = np.ones(len(samples), dtype=bool)
        if label is not None:
            mask &= labels == label
        if protection is not None:
            mask &= np.asarray([sample.protection == protection for sample in samples], dtype=bool)
        if not mask.any():
            return 0.0
        return float(correct[mask].mean())

    real_accuracy = subset_accuracy(label=0)
    fake_accuracy = subset_accuracy(label=1)
    protected_real_accuracy = subset_accuracy(label=0, protection="protected")
    return {
        "accuracy": float(correct.mean()) if len(correct) else 0.0,
        "balanced_accuracy": 0.5 * (real_accuracy + fake_accuracy),
        "real_accuracy": real_accuracy,
        "fake_accuracy": fake_accuracy,
        "protected_real_fpr": 1.0 - protected_real_accuracy,
        "protected_fake_recall": subset_accuracy(label=1, protection="protected"),
        "unprotected_fake_recall": subset_accuracy(label=1, protection="unprotected"),
    }


def rule_predictions(rows: list[Dict[str, str]], rules: dict) -> np.ndarray:
    routed = [route_row(row, compute_unprotected_map(rows), rules) for row in rows]
    return np.asarray([1 if row.get("effective_pred_label") == "fake" else 0 for row in routed], dtype=np.int64)


def samples_from_rows(path: Path) -> tuple[list[Sample], list[Dict[str, str]]]:
    rows = load_csv(path)
    validate_bridge_rows(path, rows)
    return rows_to_samples(path), rows


def choose_threshold(train_samples: list[Sample], train_probs: np.ndarray, args: argparse.Namespace) -> tuple[float, dict]:
    best_threshold = 0.5
    best_metrics: dict | None = None
    best_key: tuple[float, float, float, float] | None = None

    for threshold in np.linspace(0.01, 0.99, 99):
        predictions = (train_probs >= threshold).astype(np.int64)
        metrics = metrics_from_predictions(train_samples, predictions)
        feasible = (
            metrics["protected_real_fpr"] <= args.max_protected_real_fpr
            and metrics["protected_fake_recall"] >= args.min_protected_fake_recall
            and metrics["unprotected_fake_recall"] >= args.min_unprotected_fake_recall
        )
        key = (
            1.0 if feasible else 0.0,
            metrics["balanced_accuracy"],
            -metrics["protected_real_fpr"],
            metrics["fake_accuracy"],
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics
    assert best_metrics is not None
    return best_threshold, best_metrics


def model_for_seed(seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed),
    )


def row_from_metrics(holdout: str, method: str, metrics: dict, threshold: float | str = "") -> dict:
    return {
        "holdout_dataset": holdout,
        "method": method,
        "threshold": "" if threshold == "" else f"{float(threshold):.6f}",
        "accuracy": f"{metrics['accuracy']:.6f}",
        "balanced_accuracy": f"{metrics['balanced_accuracy']:.6f}",
        "real_accuracy": f"{metrics['real_accuracy']:.6f}",
        "fake_accuracy": f"{metrics['fake_accuracy']:.6f}",
        "protected_real_fpr": f"{metrics['protected_real_fpr']:.6f}",
        "protected_fake_recall": f"{metrics['protected_fake_recall']:.6f}",
        "unprotected_fake_recall": f"{metrics['unprotected_fake_recall']:.6f}",
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    bridge_paths = [Path(path) for path in args.bridge_csv]

    by_dataset: dict[str, tuple[list[Sample], list[Dict[str, str]]]] = {}
    all_samples: list[Sample] = []
    for path in bridge_paths:
        samples, rows = samples_from_rows(path)
        dataset = dataset_name_from_path(path)
        by_dataset[dataset] = (samples, rows)
        all_samples.extend(samples)

    output_rows: list[dict] = []
    summary: dict[str, dict] = {"feature_names": FEATURE_NAMES, "holdouts": {}}

    for holdout, (test_samples, test_rows) in by_dataset.items():
        train_samples = [
            sample
            for dataset, (samples, _) in by_dataset.items()
            if dataset != holdout
            for sample in samples
        ]
        x_train = np.asarray([sample.features for sample in train_samples], dtype=np.float32)
        y_train = np.asarray([sample.label for sample in train_samples], dtype=np.int64)
        x_test = np.asarray([sample.features for sample in test_samples], dtype=np.float32)

        model = model_for_seed(args.seed)
        model.fit(x_train, y_train)
        train_probs = model.predict_proba(x_train)[:, 1]
        threshold, train_metrics = choose_threshold(train_samples, train_probs, args)

        learned_probs = model.predict_proba(x_test)[:, 1]
        learned_predictions = (learned_probs >= threshold).astype(np.int64)
        learned_metrics = metrics_from_predictions(test_samples, learned_predictions)

        default_predictions = rule_predictions(test_rows, base_rules(0.01, 0.004))
        constrained_predictions = rule_predictions(test_rows, base_rules(0.02, 0.010))
        default_metrics = metrics_from_predictions(test_samples, default_predictions)
        constrained_metrics = metrics_from_predictions(test_samples, constrained_predictions)

        output_rows.extend(
            [
                row_from_metrics(holdout, "default_rule", default_metrics),
                row_from_metrics(holdout, "constrained_rule", constrained_metrics),
                row_from_metrics(holdout, "learned_calibrator", learned_metrics, threshold=threshold),
            ]
        )
        summary["holdouts"][holdout] = {
            "train_count": len(train_samples),
            "test_count": len(test_samples),
            "selected_threshold": threshold,
            "train_threshold_metrics": train_metrics,
            "default_rule": default_metrics,
            "constrained_rule": constrained_metrics,
            "learned_calibrator": learned_metrics,
        }

    write_csv(Path(args.output_csv), output_rows)
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
