from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import yaml


def _bootstrap_scripts() -> None:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))


_bootstrap_scripts()

from run_routing_sanity import compute_unprotected_map, load_csv, route_row
from search_routing_thresholds import dataset_name_from_path, validate_bridge_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search routing thresholds while constraining protected-real false positives "
            "and fake recall on protected/unprotected content."
        )
    )
    parser.add_argument("--bridge-csv", action="append", required=True, help="Hybrid bridge CSV. Repeat for multiple datasets.")
    parser.add_argument("--results-tsv", required=True)
    parser.add_argument("--best-rules-yaml", required=True)
    parser.add_argument("--max-protected-real-fpr", type=float, default=0.40)
    parser.add_argument("--min-protected-fake-recall", type=float, default=0.55)
    parser.add_argument("--min-unprotected-fake-recall", type=float, default=0.375)
    parser.add_argument("--min-dataset-balanced-accuracy", type=float, default=0.0)
    return parser.parse_args()


def candidate_rules() -> Iterable[dict]:
    passive_thresholds = [
        0.005,
        0.010,
        0.015,
        0.020,
        0.030,
        0.050,
        0.080,
        0.100,
        0.150,
        0.200,
        0.250,
        0.300,
        0.310,
        0.350,
        0.400,
        0.450,
        0.500,
        0.550,
        0.600,
        0.700,
        0.800,
        0.900,
    ]
    anomaly_thresholds = [
        0.002,
        0.004,
        0.006,
        0.008,
        0.010,
        0.015,
        0.020,
        0.030,
        0.050,
        0.080,
        0.120,
        0.180,
        0.250,
        0.350,
        0.450,
        0.600,
        0.800,
        1.000,
    ]

    for passive_thr in passive_thresholds:
        for anomaly_thr in anomaly_thresholds:
            yield {
                "route_labels": {
                    "trust": {
                        "min_bit_accuracy": 0.975,
                        "max_passive_delta_abs": 0.05,
                    },
                    "suspicious": {
                        "min_bit_accuracy": 0.94,
                        "max_passive_delta_abs": 0.05,
                    },
                    "abstain": {"require_protected": True},
                },
                "fallback": {
                    "passive_fake_threshold": passive_thr,
                    "protected_passive_fake_threshold": passive_thr,
                    "protected_delta_abs_anomaly_threshold": anomaly_thr,
                    "min_passive_confidence_gap": 0.45,
                },
                "notes": [
                    "Constrained search on existing bridge tables.",
                    "Trust/suspicious thresholds are fixed; this search varies passive fake threshold and protected-delta anomaly threshold.",
                ],
            }


def subset_accuracy(rows: List[Dict[str, str]], label: str | None = None, protection: str | None = None) -> float:
    subset = [
        row
        for row in rows
        if row.get("effective_correct") in {"0", "1"}
        and (label is None or row.get("label") == label)
        and (protection is None or row.get("protection") == protection)
    ]
    if not subset:
        return 0.0
    return sum(int(row["effective_correct"]) for row in subset) / len(subset)


def accuracy_metrics(rows: List[Dict[str, str]]) -> dict:
    acc_rows = [row for row in rows if row.get("effective_correct") in {"0", "1"}]
    real_accuracy = subset_accuracy(acc_rows, label="real")
    fake_accuracy = subset_accuracy(acc_rows, label="fake")
    protected_real_accuracy = subset_accuracy(acc_rows, label="real", protection="protected")
    protected_fake_recall = subset_accuracy(acc_rows, label="fake", protection="protected")
    unprotected_real_accuracy = subset_accuracy(acc_rows, label="real", protection="unprotected")
    unprotected_fake_recall = subset_accuracy(acc_rows, label="fake", protection="unprotected")

    return {
        "accuracy": 0.0 if not acc_rows else sum(int(row["effective_correct"]) for row in acc_rows) / len(acc_rows),
        "balanced_accuracy": 0.5 * (real_accuracy + fake_accuracy),
        "real_accuracy": real_accuracy,
        "fake_accuracy": fake_accuracy,
        "protected_real_accuracy": protected_real_accuracy,
        "protected_real_fpr": 1.0 - protected_real_accuracy,
        "protected_fake_recall": protected_fake_recall,
        "unprotected_real_accuracy": unprotected_real_accuracy,
        "unprotected_fake_recall": unprotected_fake_recall,
        "trust_rate": sum(1 for row in rows if row.get("route_label") == "trust") / len(rows),
        "abstain_rate": sum(1 for row in rows if row.get("route_label") == "abstain") / len(rows),
        "anomaly_rate": sum(1 for row in rows if row.get("anomaly_flag") == "1") / len(rows),
    }


def evaluate_rules(bridge_paths: List[Path], rules: dict) -> dict:
    dataset_metrics: Dict[str, dict] = {}
    combined_rows: List[Dict[str, str]] = []

    for bridge_path in bridge_paths:
        bridge_rows = load_csv(bridge_path)
        validate_bridge_rows(bridge_path, bridge_rows)
        unprotected_map = compute_unprotected_map(bridge_rows)
        routed_rows = [route_row(row, unprotected_map, rules) for row in bridge_rows]
        dataset_metrics[dataset_name_from_path(bridge_path)] = accuracy_metrics(routed_rows)
        combined_rows.extend(routed_rows)

    combined = accuracy_metrics(combined_rows)
    return {
        "combined": combined,
        "datasets": dataset_metrics,
        "min_dataset_balanced_accuracy": min(metrics["balanced_accuracy"] for metrics in dataset_metrics.values()),
        "max_protected_real_fpr": max(metrics["protected_real_fpr"] for metrics in dataset_metrics.values()),
        "min_protected_fake_recall": min(metrics["protected_fake_recall"] for metrics in dataset_metrics.values()),
        "min_unprotected_fake_recall": min(metrics["unprotected_fake_recall"] for metrics in dataset_metrics.values()),
    }


def is_feasible(metrics: dict, args: argparse.Namespace) -> bool:
    return (
        metrics["max_protected_real_fpr"] <= args.max_protected_real_fpr
        and metrics["min_protected_fake_recall"] >= args.min_protected_fake_recall
        and metrics["min_unprotected_fake_recall"] >= args.min_unprotected_fake_recall
        and metrics["min_dataset_balanced_accuracy"] >= args.min_dataset_balanced_accuracy
    )


def results_row(config_id: int, rules: dict, metrics: dict, feasible: bool) -> dict:
    row = {
        "config_id": str(config_id),
        "feasible": "1" if feasible else "0",
        "passive_fake_threshold": f"{rules['fallback']['passive_fake_threshold']:.6f}",
        "protected_delta_abs_anomaly_threshold": f"{rules['fallback']['protected_delta_abs_anomaly_threshold']:.6f}",
        "overall_balanced_accuracy": f"{metrics['combined']['balanced_accuracy']:.6f}",
        "overall_accuracy": f"{metrics['combined']['accuracy']:.6f}",
        "min_dataset_balanced_accuracy": f"{metrics['min_dataset_balanced_accuracy']:.6f}",
        "max_protected_real_fpr": f"{metrics['max_protected_real_fpr']:.6f}",
        "min_protected_fake_recall": f"{metrics['min_protected_fake_recall']:.6f}",
        "min_unprotected_fake_recall": f"{metrics['min_unprotected_fake_recall']:.6f}",
    }
    for dataset_name, dataset_metrics in metrics["datasets"].items():
        row[f"{dataset_name}_balanced_accuracy"] = f"{dataset_metrics['balanced_accuracy']:.6f}"
        row[f"{dataset_name}_real_accuracy"] = f"{dataset_metrics['real_accuracy']:.6f}"
        row[f"{dataset_name}_fake_accuracy"] = f"{dataset_metrics['fake_accuracy']:.6f}"
        row[f"{dataset_name}_protected_real_fpr"] = f"{dataset_metrics['protected_real_fpr']:.6f}"
        row[f"{dataset_name}_protected_fake_recall"] = f"{dataset_metrics['protected_fake_recall']:.6f}"
        row[f"{dataset_name}_unprotected_fake_recall"] = f"{dataset_metrics['unprotected_fake_recall']:.6f}"
    return row


def sort_key(result: tuple[dict, dict, dict, bool]) -> tuple[float, float, float, float, float, float]:
    _, rules, metrics, feasible = result
    return (
        1.0 if feasible else 0.0,
        metrics["min_dataset_balanced_accuracy"],
        metrics["combined"]["balanced_accuracy"],
        -metrics["max_protected_real_fpr"],
        metrics["min_unprotected_fake_recall"],
        -float(rules["fallback"]["passive_fake_threshold"]),
    )


def write_tsv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    bridge_paths = [Path(path) for path in args.bridge_csv]

    scored: List[tuple[dict, dict, dict, bool]] = []
    for config_id, rules in enumerate(candidate_rules(), start=1):
        metrics = evaluate_rules(bridge_paths, rules)
        feasible = is_feasible(metrics, args)
        row = results_row(config_id, rules, metrics, feasible)
        scored.append((row, rules, metrics, feasible))

    scored.sort(key=sort_key, reverse=True)
    write_tsv(Path(args.results_tsv), [row for row, _, _, _ in scored])

    best_row, best_rules, best_metrics, best_feasible = scored[0]
    best_payload = {
        **best_rules,
        "constraints": {
            "max_protected_real_fpr": args.max_protected_real_fpr,
            "min_protected_fake_recall": args.min_protected_fake_recall,
            "min_unprotected_fake_recall": args.min_unprotected_fake_recall,
            "min_dataset_balanced_accuracy": args.min_dataset_balanced_accuracy,
        },
        "search_summary": {
            "selected_config_id": int(best_row["config_id"]),
            "feasible": best_feasible,
            "overall_accuracy": float(best_row["overall_accuracy"]),
            "overall_balanced_accuracy": float(best_row["overall_balanced_accuracy"]),
            "min_dataset_balanced_accuracy": float(best_row["min_dataset_balanced_accuracy"]),
            "max_protected_real_fpr": float(best_row["max_protected_real_fpr"]),
            "min_protected_fake_recall": float(best_row["min_protected_fake_recall"]),
            "min_unprotected_fake_recall": float(best_row["min_unprotected_fake_recall"]),
            "datasets": best_metrics["datasets"],
        },
    }
    write_yaml(Path(args.best_rules_yaml), best_payload)
    print(json.dumps(best_payload["search_summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
