from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml


def _bootstrap_scripts() -> None:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))


_bootstrap_scripts()

from run_routing_sanity import compute_unprotected_map, load_csv, route_row
from search_routing_thresholds import dataset_name_from_path, validate_bridge_rows
from search_routing_thresholds_constrained import accuracy_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Leave-one-dataset-out validation for one fixed explicit protected-transfer ruleset, "
            "so paper-friendly literal or distilled policies can be checked as single configurations."
        )
    )
    parser.add_argument("--rules-yaml", required=True)
    parser.add_argument("--canonical-bridge-csv", action="append", required=True)
    parser.add_argument("--paired-bridge-csv", action="append", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--tag", required=True)
    return parser.parse_args()


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


def evaluate_paths(paths: list[Path], rules: dict) -> dict:
    dataset_metrics: dict[str, dict] = {}
    combined_rows: list[dict] = []
    for bridge_path in paths:
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
        "min_dataset_balanced_accuracy": min(item["balanced_accuracy"] for item in dataset_metrics.values()),
        "max_protected_real_fpr": max(item["protected_real_fpr"] for item in dataset_metrics.values()),
        "min_protected_fake_recall": min(item["protected_fake_recall"] for item in dataset_metrics.values()),
        "min_unprotected_fake_recall": min(item["unprotected_fake_recall"] for item in dataset_metrics.values()),
    }


def paired_objective(paths: list[Path], rules: dict) -> dict:
    dataset_metrics: dict[str, dict] = {}
    for bridge_path in paths:
        bridge_rows = load_csv(bridge_path)
        validate_bridge_rows(bridge_path, bridge_rows)
        unprotected_map = compute_unprotected_map(bridge_rows)
        routed_rows = [route_row(row, unprotected_map, rules) for row in bridge_rows]
        dataset_metrics[dataset_name_from_path(bridge_path)] = accuracy_metrics(routed_rows)

    fake_accuracies = [item["fake_accuracy"] for item in dataset_metrics.values()]
    bas = [item["balanced_accuracy"] for item in dataset_metrics.values()]
    return {
        "datasets": dataset_metrics,
        "mean_fake_accuracy": sum(fake_accuracies) / len(fake_accuracies),
        "min_fake_accuracy": min(fake_accuracies),
        "mean_balanced_accuracy": sum(bas) / len(bas),
    }


def main() -> None:
    args = parse_args()
    canonical_paths = [Path(path) for path in args.canonical_bridge_csv]
    paired_paths = [Path(path) for path in args.paired_bridge_csv]
    rules = yaml.safe_load(Path(args.rules_yaml).read_text(encoding="utf-8"))

    output_rows: list[dict] = []
    summary = {
        "tag": args.tag,
        "selection": vars(args),
        "all_data_canonical": evaluate_paths(canonical_paths, rules),
        "all_data_paired": paired_objective(paired_paths, rules),
        "holdouts": {},
    }

    for holdout in canonical_paths:
        holdout_eval = evaluate_paths([holdout], rules)
        paired_eval = paired_objective(paired_paths, rules)
        holdout_name = dataset_name_from_path(holdout)
        holdout_metrics = holdout_eval["datasets"][holdout_name]
        output_rows.append(
            {
                "holdout_dataset": holdout_name,
                "tag": args.tag,
                "paired_mean_fake_accuracy": f"{paired_eval['mean_fake_accuracy']:.6f}",
                "paired_min_fake_accuracy": f"{paired_eval['min_fake_accuracy']:.6f}",
                "holdout_balanced_accuracy": f"{holdout_metrics['balanced_accuracy']:.6f}",
                "holdout_fake_accuracy": f"{holdout_metrics['fake_accuracy']:.6f}",
                "holdout_protected_real_fpr": f"{holdout_metrics['protected_real_fpr']:.6f}",
                "holdout_protected_fake_recall": f"{holdout_metrics['protected_fake_recall']:.6f}",
                "holdout_unprotected_fake_recall": f"{holdout_metrics['unprotected_fake_recall']:.6f}",
            }
        )
        summary["holdouts"][holdout_name] = {
            "paired_objective": paired_eval,
            "holdout_metrics": holdout_metrics,
        }

    holdout_bas = [float(row["holdout_balanced_accuracy"]) for row in output_rows]
    holdout_fprs = [float(row["holdout_protected_real_fpr"]) for row in output_rows]
    paired_means = [float(row["paired_mean_fake_accuracy"]) for row in output_rows]
    summary["aggregate"] = {
        "mean_holdout_balanced_accuracy": sum(holdout_bas) / len(holdout_bas),
        "min_holdout_balanced_accuracy": min(holdout_bas),
        "max_holdout_protected_real_fpr": max(holdout_fprs),
        "mean_paired_mean_fake_accuracy": sum(paired_means) / len(paired_means),
    }

    write_csv(Path(args.output_csv), output_rows)
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
