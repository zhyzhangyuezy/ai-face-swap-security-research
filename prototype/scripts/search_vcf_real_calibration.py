from __future__ import annotations

import argparse
import csv
import math
import re
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate VCF detector-side real-anchor calibration. Thresholds are estimated only from "
            "unprotected VCF target-real rows, then applied slice-wise to all non-anomaly rows."
        )
    )
    parser.add_argument("--routing-csv", required=True)
    parser.add_argument("--results-tsv", required=True)
    parser.add_argument("--best-output-csv", required=True)
    parser.add_argument("--best-summary", required=True)
    parser.add_argument("--max-fake-accuracy-drop", type=float, default=0.10)
    parser.add_argument("--min-prfpr-improvement", type=float, default=0.10)
    return parser.parse_args()


def safe_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_tsv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def extract_note_field(notes: str, key: str) -> str:
    match = re.search(rf"(?:^|; )({re.escape(key)})=([^;]+)", notes)
    return match.group(2) if match else "unknown"


def slice_key(row: Dict[str, str], mode: str) -> str:
    notes = row.get("notes", "")
    resolution = extract_note_field(notes, "resolution")
    bucket = extract_note_field(notes, "bucket")
    degradation = row.get("degradation_type", "unknown")
    if mode == "global":
        return "global"
    if mode == "resolution":
        return resolution
    if mode == "bucket":
        return bucket
    if mode == "resolution_bucket":
        return f"{resolution}::{bucket}"
    if mode == "bucket_degradation":
        return f"{bucket}::{degradation}"
    if mode == "resolution_bucket_degradation":
        return f"{resolution}::{bucket}::{degradation}"
    raise ValueError(f"Unknown slice mode: {mode}")


def quantile(values: List[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return ordered[int(pos)]
    weight = pos - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def build_thresholds(rows: List[Dict[str, str]], mode: str, q: float, margin: float, min_group_count: int) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = {}
    global_values: List[float] = []
    for row in rows:
        if row.get("label") != "real" or row.get("protection") != "unprotected":
            continue
        prob = safe_float(row.get("passive_prob_fake"))
        if prob is None:
            continue
        global_values.append(prob)
        grouped.setdefault(slice_key(row, mode), []).append(prob)

    global_threshold = min(0.999999, quantile(global_values, q) + margin)
    thresholds = {"__global__": global_threshold}
    for key, values in grouped.items():
        if len(values) < min_group_count:
            thresholds[key] = global_threshold
        else:
            thresholds[key] = min(0.999999, quantile(values, q) + margin)
    return thresholds


def apply_calibration(rows: List[Dict[str, str]], mode: str, thresholds: Dict[str, float], *, apply_to: str) -> List[Dict[str, str]]:
    calibrated: List[Dict[str, str]] = []
    for row in rows:
        updated = deepcopy(row)
        key = slice_key(updated, mode)
        calibrated_threshold = thresholds.get(key, thresholds["__global__"])
        base_threshold = safe_float(updated.get("passive_fake_threshold")) or 0.5
        threshold = max(base_threshold, calibrated_threshold)
        updated["vcf_calibration_key"] = key
        updated["vcf_calibrated_threshold"] = f"{threshold:.6f}"
        updated["vcf_calibrated_threshold_source"] = mode
        updated["vcf_calibration_applied"] = "0"

        prob = safe_float(updated.get("passive_prob_fake"))
        if prob is None:
            calibrated.append(updated)
            continue

        if updated.get("anomaly_flag") == "1":
            calibrated.append(updated)
            continue

        if apply_to == "protected" and updated.get("protection") != "protected":
            calibrated.append(updated)
            continue
        if apply_to == "realish" and updated.get("passive_baseline_prob", "") != "":
            base_prob = safe_float(updated.get("passive_baseline_prob"))
            if base_prob is not None and base_prob > threshold:
                calibrated.append(updated)
                continue

        pred = "fake" if prob >= threshold else "real"
        updated["vcf_calibration_applied"] = "1"
        updated["passive_pred_label"] = pred
        updated["passive_fake_threshold"] = f"{threshold:.6f}"
        updated["effective_pred_label"] = pred
        if updated.get("label") in {"real", "fake"}:
            updated["effective_correct"] = "1" if pred == updated.get("label") else "0"
        if pred == "real":
            updated["final_decision"] = "vcf_real_calibrated_accept_real"
        else:
            updated["final_decision"] = "vcf_real_calibrated_flag_fake"
        calibrated.append(updated)
    return calibrated


def fast_metrics(rows: List[Dict[str, str]], mode: str, thresholds: Dict[str, float], *, apply_to: str) -> Dict[str, float]:
    counts = {
        "real_total": 0,
        "real_correct": 0,
        "fake_total": 0,
        "fake_correct": 0,
        "protected_real_total": 0,
        "protected_real_correct": 0,
        "protected_fake_total": 0,
        "protected_fake_correct": 0,
        "unprotected_real_total": 0,
        "unprotected_real_correct": 0,
        "unprotected_fake_total": 0,
        "unprotected_fake_correct": 0,
        "calibrated_count": 0,
    }
    for row in rows:
        label = row.get("label")
        protection = row.get("protection")
        if label not in {"real", "fake"}:
            continue

        pred = row.get("effective_pred_label")
        prob = safe_float(row.get("passive_prob_fake"))
        if prob is not None and row.get("anomaly_flag") != "1":
            should_apply = True
            if apply_to == "protected" and protection != "protected":
                should_apply = False
            if should_apply:
                key = slice_key(row, mode)
                threshold = max(safe_float(row.get("passive_fake_threshold")) or 0.5, thresholds.get(key, thresholds["__global__"]))
                pred = "fake" if prob >= threshold else "real"
                counts["calibrated_count"] += 1

        correct = 1 if pred == label else 0
        if label == "real":
            counts["real_total"] += 1
            counts["real_correct"] += correct
            if protection == "protected":
                counts["protected_real_total"] += 1
                counts["protected_real_correct"] += correct
            elif protection == "unprotected":
                counts["unprotected_real_total"] += 1
                counts["unprotected_real_correct"] += correct
        else:
            counts["fake_total"] += 1
            counts["fake_correct"] += correct
            if protection == "protected":
                counts["protected_fake_total"] += 1
                counts["protected_fake_correct"] += correct
            elif protection == "unprotected":
                counts["unprotected_fake_total"] += 1
                counts["unprotected_fake_correct"] += correct

    def acc(name: str) -> float:
        total = counts[f"{name}_total"]
        return 0.0 if total == 0 else counts[f"{name}_correct"] / total

    real_accuracy = acc("real")
    fake_accuracy = acc("fake")
    protected_real_accuracy = acc("protected_real")
    return {
        "balanced_accuracy": 0.5 * (real_accuracy + fake_accuracy),
        "real_accuracy": real_accuracy,
        "fake_accuracy": fake_accuracy,
        "protected_real_fpr": 1.0 - protected_real_accuracy,
        "protected_fake_recall": acc("protected_fake"),
        "unprotected_real_accuracy": acc("unprotected_real"),
        "unprotected_fake_recall": acc("unprotected_fake"),
        "calibrated_rate": counts["calibrated_count"] / len(rows) if rows else 0.0,
    }


def class_accuracy(rows: List[Dict[str, str]], label: str, protection: str | None = None) -> float:
    subset = [
        row
        for row in rows
        if row.get("effective_correct") in {"0", "1"}
        and row.get("label") == label
        and (protection is None or row.get("protection") == protection)
    ]
    if not subset:
        return 0.0
    return sum(int(row["effective_correct"]) for row in subset) / len(subset)


def metrics(rows: List[Dict[str, str]]) -> Dict[str, float]:
    real_accuracy = class_accuracy(rows, "real")
    fake_accuracy = class_accuracy(rows, "fake")
    protected_real_accuracy = class_accuracy(rows, "real", "protected")
    protected_fake_recall = class_accuracy(rows, "fake", "protected")
    unprotected_real_accuracy = class_accuracy(rows, "real", "unprotected")
    unprotected_fake_recall = class_accuracy(rows, "fake", "unprotected")
    calibrated_count = sum(1 for row in rows if row.get("vcf_calibration_applied") == "1")
    return {
        "balanced_accuracy": 0.5 * (real_accuracy + fake_accuracy),
        "real_accuracy": real_accuracy,
        "fake_accuracy": fake_accuracy,
        "protected_real_fpr": 1.0 - protected_real_accuracy,
        "protected_fake_recall": protected_fake_recall,
        "unprotected_real_accuracy": unprotected_real_accuracy,
        "unprotected_fake_recall": unprotected_fake_recall,
        "calibrated_rate": calibrated_count / len(rows) if rows else 0.0,
    }


def candidate_configs() -> Iterable[Tuple[str, float, float, int, str]]:
    modes = [
        "global",
        "resolution",
        "bucket",
        "resolution_bucket",
        "bucket_degradation",
        "resolution_bucket_degradation",
    ]
    quantiles = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    margins = [0.00, 0.02, 0.05, 0.08]
    min_counts = [5, 10]
    apply_to_values = ["all", "protected"]
    for mode in modes:
        for q in quantiles:
            for margin in margins:
                for min_count in min_counts:
                    for apply_to in apply_to_values:
                        yield mode, q, margin, min_count, apply_to


def row_from_metrics(
    config_id: int,
    config: Tuple[str, float, float, int, str],
    current: Dict[str, float],
    baseline: Dict[str, float],
    feasible: bool,
) -> Dict[str, str]:
    mode, q, margin, min_count, apply_to = config
    return {
        "config_id": str(config_id),
        "feasible": "1" if feasible else "0",
        "mode": mode,
        "quantile": f"{q:.2f}",
        "margin": f"{margin:.2f}",
        "min_group_count": str(min_count),
        "apply_to": apply_to,
        "balanced_accuracy": f"{current['balanced_accuracy']:.6f}",
        "fake_accuracy": f"{current['fake_accuracy']:.6f}",
        "real_accuracy": f"{current['real_accuracy']:.6f}",
        "protected_real_fpr": f"{current['protected_real_fpr']:.6f}",
        "protected_fake_recall": f"{current['protected_fake_recall']:.6f}",
        "unprotected_real_accuracy": f"{current['unprotected_real_accuracy']:.6f}",
        "unprotected_fake_recall": f"{current['unprotected_fake_recall']:.6f}",
        "calibrated_rate": f"{current['calibrated_rate']:.6f}",
        "ba_delta": f"{current['balanced_accuracy'] - baseline['balanced_accuracy']:.6f}",
        "fake_accuracy_delta": f"{current['fake_accuracy'] - baseline['fake_accuracy']:.6f}",
        "protected_real_fpr_improvement": f"{baseline['protected_real_fpr'] - current['protected_real_fpr']:.6f}",
    }


def sort_key(item: Tuple[Dict[str, str], List[Dict[str, str]], Dict[str, float], bool]) -> Tuple[float, float, float, float]:
    row, _, current, feasible = item
    return (
        1.0 if feasible else 0.0,
        current["balanced_accuracy"],
        1.0 - current["protected_real_fpr"],
        current["fake_accuracy"],
    )


def main() -> None:
    args = parse_args()
    rows = read_csv(Path(args.routing_csv))
    baseline_metrics = metrics(rows)
    scored: List[Tuple[Dict[str, str], Tuple[str, float, float, int, str], Dict[str, float], bool]] = []

    baseline_row = row_from_metrics(
        0,
        ("reference", 0.0, 0.0, 0, "none"),
        baseline_metrics,
        baseline_metrics,
        False,
    )
    scored.append((baseline_row, ("reference", 0.0, 0.0, 0, "none"), baseline_metrics, False))

    for config_id, config in enumerate(candidate_configs(), start=1):
        mode, q, margin, min_count, apply_to = config
        thresholds = build_thresholds(rows, mode, q, margin, min_count)
        current_metrics = fast_metrics(rows, mode, thresholds, apply_to=apply_to)
        feasible = (
            baseline_metrics["protected_real_fpr"] - current_metrics["protected_real_fpr"] >= args.min_prfpr_improvement
            and current_metrics["fake_accuracy"] >= baseline_metrics["fake_accuracy"] - args.max_fake_accuracy_drop
        )
        scored.append((row_from_metrics(config_id, config, current_metrics, baseline_metrics, feasible), config, current_metrics, feasible))

    scored.sort(key=sort_key, reverse=True)
    write_tsv(Path(args.results_tsv), [row for row, _, _, _ in scored])
    best_row, best_config, _, _ = scored[0]
    if best_config[0] == "reference":
        best_rows = rows
    else:
        mode, q, margin, min_count, apply_to = best_config
        thresholds = build_thresholds(rows, mode, q, margin, min_count)
        best_rows = apply_calibration(rows, mode, thresholds, apply_to=apply_to)
    write_csv(Path(args.best_output_csv), best_rows)
    Path(args.best_summary).parent.mkdir(parents=True, exist_ok=True)
    summary_lines = [
        "# VCF Real-Anchor Calibration Best Summary",
        "",
        "| field | value |",
        "| --- | --- |",
    ]
    for key, value in best_row.items():
        summary_lines.append(f"| {key} | {value} |")
    Path(args.best_summary).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
