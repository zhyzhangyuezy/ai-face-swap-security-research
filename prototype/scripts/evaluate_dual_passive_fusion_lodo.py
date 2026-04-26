from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def _bootstrap_scripts() -> None:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))


_bootstrap_scripts()

from evaluate_dual_passive_fusion import (
    candidate_rules,
    evaluate_strategy,
    feasible,
    load_pair,
    materialize_strategy_rows,
    parse_pair,
    result_row,
    sort_key,
)
from run_routing_sanity import compute_unprotected_map, route_row
from search_routing_thresholds_constrained import accuracy_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leave-one-dataset-out validation for dual-passive fusion strategy selection."
    )
    parser.add_argument(
        "--bridge-pair",
        action="append",
        required=True,
        help="DATASET=EFFB0_BRIDGE_CSV,UCF_BRIDGE_CSV. Repeat for each dataset.",
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--max-protected-real-fpr", type=float, default=0.40)
    parser.add_argument("--min-protected-fake-recall", type=float, default=0.50)
    parser.add_argument("--min-unprotected-fake-recall", type=float, default=0.375)
    return parser.parse_args()


def strategies() -> list[str]:
    return [
        "effb0",
        "ucf",
        "mean",
        "max",
        "min",
        "confidence_select",
        "weighted_ucf_0.25",
        "weighted_ucf_0.50",
        "weighted_ucf_0.75",
    ]


def evaluate_on_holdout(rows: list[dict], strategy: str, rules: dict) -> dict:
    fused_rows = materialize_strategy_rows(rows, strategy)
    routed_rows = [route_row(row, compute_unprotected_map(fused_rows), rules) for row in fused_rows]
    return accuracy_metrics(routed_rows)


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


def metric_row(holdout: str, selected: dict, holdout_metrics: dict) -> dict:
    rules = selected["rules"]
    return {
        "holdout_dataset": holdout,
        "selected_strategy": selected["strategy"],
        "selected_config_id": str(selected["config_id"]),
        "selected_passive_fake_threshold": f"{rules['fallback']['passive_fake_threshold']:.6f}",
        "selected_anomaly_threshold": f"{rules['fallback']['protected_delta_abs_anomaly_threshold']:.6f}",
        "train_overall_ba": f"{selected['train_metrics']['combined']['balanced_accuracy']:.6f}",
        "train_min_dataset_ba": f"{selected['train_metrics']['min_dataset_balanced_accuracy']:.6f}",
        "holdout_accuracy": f"{holdout_metrics['accuracy']:.6f}",
        "holdout_balanced_accuracy": f"{holdout_metrics['balanced_accuracy']:.6f}",
        "holdout_real_accuracy": f"{holdout_metrics['real_accuracy']:.6f}",
        "holdout_fake_accuracy": f"{holdout_metrics['fake_accuracy']:.6f}",
        "holdout_protected_real_fpr": f"{holdout_metrics['protected_real_fpr']:.6f}",
        "holdout_protected_fake_recall": f"{holdout_metrics['protected_fake_recall']:.6f}",
        "holdout_unprotected_fake_recall": f"{holdout_metrics['unprotected_fake_recall']:.6f}",
    }


def main() -> None:
    args = parse_args()
    dataset_rows = {}
    for value in args.bridge_pair:
        dataset, effb0_path, ucf_path = parse_pair(value)
        dataset_rows[dataset] = load_pair(dataset, effb0_path, ucf_path)

    output_rows: list[dict] = []
    summary = {"holdouts": {}}

    for holdout in sorted(dataset_rows):
        train_rows = {name: rows for name, rows in dataset_rows.items() if name != holdout}
        scored = []
        for strategy in strategies():
            for config_id, rules in enumerate(candidate_rules(), start=1):
                metrics = evaluate_strategy(train_rows, strategy, rules)
                row = result_row(strategy, config_id, rules, metrics, feasible(metrics, args))
                scored.append((row, rules, metrics, feasible(metrics, args)))

        scored.sort(key=sort_key, reverse=True)
        best_row, best_rules, best_train_metrics, best_feasible = scored[0]
        holdout_metrics = evaluate_on_holdout(dataset_rows[holdout], best_row["strategy"], best_rules)
        selected = {
            "strategy": best_row["strategy"],
            "config_id": int(best_row["config_id"]),
            "rules": best_rules,
            "train_feasible": best_feasible,
            "train_metrics": best_train_metrics,
            "holdout_metrics": holdout_metrics,
        }
        summary["holdouts"][holdout] = selected
        output_rows.append(metric_row(holdout, selected, holdout_metrics))

    if output_rows:
        mean_ba = sum(float(row["holdout_balanced_accuracy"]) for row in output_rows) / len(output_rows)
        min_ba = min(float(row["holdout_balanced_accuracy"]) for row in output_rows)
    else:
        mean_ba = 0.0
        min_ba = 0.0
    summary["aggregate"] = {
        "mean_holdout_balanced_accuracy": mean_ba,
        "min_holdout_balanced_accuracy": min_ba,
    }

    write_csv(Path(args.output_csv), output_rows)
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
