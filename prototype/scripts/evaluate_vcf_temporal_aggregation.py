from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List


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
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aggregate(values: List[float], strategy: str) -> float:
    if strategy == "mean":
        return mean(values)
    if strategy == "median":
        return median(values)
    if strategy == "max":
        return max(values)
    if strategy == "top2_mean":
        ordered = sorted(values, reverse=True)
        return mean(ordered[: min(2, len(ordered))])
    if strategy == "midlate_mean":
        return mean(values[1:]) if len(values) > 1 else values[0]
    raise ValueError(f"Unknown aggregation strategy: {strategy}")


def subset_accuracy(groups: List[Dict[str, object]], label: str | None = None, protection: str | None = None) -> float:
    subset = [
        group
        for group in groups
        if (label is None or group["label"] == label) and (protection is None or group["protection"] == protection)
    ]
    if not subset:
        return 0.0
    return sum(int(group["correct"]) for group in subset) / len(subset)


def metrics(groups: List[Dict[str, object]]) -> dict:
    real_accuracy = subset_accuracy(groups, label="real")
    fake_accuracy = subset_accuracy(groups, label="fake")
    protected_real_accuracy = subset_accuracy(groups, label="real", protection="protected")
    return {
        "groups": len(groups),
        "accuracy": 0.0 if not groups else sum(int(group["correct"]) for group in groups) / len(groups),
        "balanced_accuracy": 0.5 * (real_accuracy + fake_accuracy),
        "real_accuracy": real_accuracy,
        "fake_accuracy": fake_accuracy,
        "protected_real_accuracy": protected_real_accuracy,
        "protected_real_fpr": 1.0 - protected_real_accuracy,
        "protected_fake_recall": subset_accuracy(groups, label="fake", protection="protected"),
        "unprotected_real_accuracy": subset_accuracy(groups, label="real", protection="unprotected"),
        "unprotected_fake_recall": subset_accuracy(groups, label="fake", protection="unprotected"),
    }


def thresholds(step: float) -> Iterable[float]:
    count = int(math.floor(1.0 / step))
    yielded = set()
    for idx in range(count + 1):
        value = round(idx * step, 6)
        yielded.add(value)
        yield value
    for value in [0.95, 0.96, 0.97, 0.98, 0.99]:
        value = round(value, 6)
        if value not in yielded:
            yield value


def build_groups(
    manifest_rows: List[Dict[str, str]],
    passive_rows: Dict[str, Dict[str, str]],
    strategy: str,
    threshold: float,
) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in manifest_rows:
        passive = passive_rows.get(row["sample_id"], {})
        if passive.get("status") != "ok" or not passive.get("prob_fake"):
            continue
        notes = parse_notes(row.get("notes", ""))
        temporal_sample_id = notes.get("temporal_sample_id") or row["sample_id"]
        grouped.setdefault(temporal_sample_id, []).append({**row, "prob_fake": passive["prob_fake"]})

    output: List[Dict[str, object]] = []
    for temporal_sample_id, rows in grouped.items():
        rows.sort(key=lambda item: int(parse_notes(item.get("notes", "")).get("temporal_frame_slot", "0")))
        values = [float(row["prob_fake"]) for row in rows]
        score = aggregate(values, strategy)
        label = rows[0]["label"]
        protection = rows[0]["protection"]
        pred_fake = score >= threshold
        correct = pred_fake if label == "fake" else not pred_fake
        output.append(
            {
                "temporal_sample_id": temporal_sample_id,
                "label": label,
                "protection": protection,
                "degradation_type": rows[0]["degradation_type"],
                "degradation_level": rows[0]["degradation_level"],
                "score": score,
                "frame_count": len(values),
                "correct": int(correct),
                "pred_label": "fake" if pred_fake else "real",
            }
        )
    return output


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def best_row(rows: List[dict], *, max_fpr: float | None = None) -> dict | None:
    candidates = rows
    if max_fpr is not None:
        candidates = [row for row in rows if float(row["protected_real_fpr"]) <= max_fpr]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            float(row["balanced_accuracy"]),
            float(row["protected_fake_recall"]),
            float(row["unprotected_fake_recall"]),
            -float(row["protected_real_fpr"]),
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate video-level temporal aggregation on VCF multi-frame passive results.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--passive-csv", required=True)
    parser.add_argument("--sweep-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--prediction-output", default="")
    parser.add_argument("--threshold-step", type=float, default=0.005)
    parser.add_argument("--strategies", nargs="+", default=["mean", "median", "top2_mean", "midlate_mean", "max"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_rows = load_csv(Path(args.manifest))
    passive_rows = {row["sample_id"]: row for row in load_csv(Path(args.passive_csv)) if row.get("sample_id")}

    sweep_rows: List[dict] = []
    selected_predictions: List[dict] = []
    for strategy in args.strategies:
        for threshold in thresholds(args.threshold_step):
            groups = build_groups(manifest_rows, passive_rows, strategy, threshold)
            row_metrics = metrics(groups)
            sweep_rows.append(
                {
                    "strategy": strategy,
                    "threshold": f"{threshold:.6f}",
                    **{key: f"{value:.6f}" if isinstance(value, float) else value for key, value in row_metrics.items()},
                    "strict_fpr08": "1" if row_metrics["protected_real_fpr"] <= 0.08 else "0",
                }
            )

    by_strategy = {}
    for strategy in args.strategies:
        strategy_rows = [row for row in sweep_rows if row["strategy"] == strategy]
        by_strategy[strategy] = {
            "best_any": best_row(strategy_rows),
            "best_fpr08": best_row(strategy_rows, max_fpr=0.08),
            "best_fpr12": best_row(strategy_rows, max_fpr=0.12),
            "best_fpr20": best_row(strategy_rows, max_fpr=0.20),
        }

    overall_best = best_row(sweep_rows)
    overall_fpr08 = best_row(sweep_rows, max_fpr=0.08)
    if args.prediction_output and overall_best is not None:
        selected_predictions = build_groups(
            manifest_rows,
            passive_rows,
            str(overall_best["strategy"]),
            float(overall_best["threshold"]),
        )
        write_csv(Path(args.prediction_output), selected_predictions)

    write_csv(Path(args.sweep_output), sweep_rows)
    summary = {
        "manifest": str(Path(args.manifest).resolve()),
        "passive_csv": str(Path(args.passive_csv).resolve()),
        "strategies": args.strategies,
        "best_by_strategy": by_strategy,
        "overall_best": overall_best,
        "overall_best_fpr08": overall_fpr08,
        "prediction_output": str(Path(args.prediction_output).resolve()) if args.prediction_output else "",
    }
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
