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

from evaluate_dual_passive_fusion import candidate_rules, feasible, result_row, sort_key
from run_routing_sanity import compute_unprotected_map, load_csv, route_row
from search_routing_thresholds import validate_bridge_rows
from search_routing_thresholds_constrained import accuracy_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leave-one-dataset-out validation for one passive bridge family.")
    parser.add_argument("--bridge", action="append", required=True, help="DATASET=BRIDGE_CSV. Repeat for each dataset.")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--max-protected-real-fpr", type=float, default=0.40)
    parser.add_argument("--min-protected-fake-recall", type=float, default=0.50)
    parser.add_argument("--min-unprotected-fake-recall", type=float, default=0.375)
    parser.add_argument("--selection-mode", choices=["default", "fpr_guarded", "fpr_first", "route_floor"], default="default")
    parser.add_argument("--selection-train-fpr-target", type=float, default=0.04)
    parser.add_argument("--selection-min-ba-drop", type=float, default=0.025)
    parser.add_argument("--selection-min-passive-threshold", type=float, default=0.30)
    parser.add_argument("--selection-min-anomaly-threshold", type=float, default=0.25)
    return parser.parse_args()


def parse_bridge(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected DATASET=CSV, got: {value}")
    dataset, path = value.split("=", 1)
    return dataset.strip(), Path(path.strip())


def load_dataset_rows(values: list[str]) -> dict[str, list[dict]]:
    out = {}
    for value in values:
        dataset, path = parse_bridge(value)
        rows = load_csv(path)
        validate_bridge_rows(path, rows)
        for row in rows:
            row["dataset_name"] = dataset
        out[dataset] = rows
    return out


def route_dataset(rows: list[dict], rules: dict) -> list[dict]:
    unprotected_map = compute_unprotected_map(rows)
    return [route_row(row, unprotected_map, rules) for row in rows]


def evaluate_rules(dataset_rows: dict[str, list[dict]], rules: dict) -> dict:
    dataset_metrics = {}
    combined_rows = []
    for dataset, rows in dataset_rows.items():
        routed = route_dataset(rows, rules)
        dataset_metrics[dataset] = accuracy_metrics(routed)
        combined_rows.extend(routed)
    combined = accuracy_metrics(combined_rows)
    return {
        "combined": combined,
        "datasets": dataset_metrics,
        "min_dataset_balanced_accuracy": min(item["balanced_accuracy"] for item in dataset_metrics.values()),
        "max_protected_real_fpr": max(item["protected_real_fpr"] for item in dataset_metrics.values()),
        "min_protected_fake_recall": min(item["protected_fake_recall"] for item in dataset_metrics.values()),
        "min_unprotected_fake_recall": min(item["unprotected_fake_recall"] for item in dataset_metrics.values()),
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


def metric_row(holdout: str, selected: dict, holdout_metrics: dict) -> dict:
    rules = selected["rules"]
    return {
        "holdout_dataset": holdout,
        "selected_config_id": str(selected["config_id"]),
        "selected_passive_fake_threshold": f"{rules['fallback']['passive_fake_threshold']:.6f}",
        "selected_anomaly_threshold": f"{rules['fallback']['protected_delta_abs_anomaly_threshold']:.6f}",
        "train_feasible": "1" if selected["train_feasible"] else "0",
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


def select_scored(scored: list[tuple[dict, dict, dict, bool]], args: argparse.Namespace) -> tuple[dict, dict, dict, bool]:
    scored.sort(key=sort_key, reverse=True)
    if args.selection_mode == "default":
        return scored[0]
    best_min_ba = scored[0][2]["min_dataset_balanced_accuracy"]
    candidates = []
    for item in scored:
        row, _, metrics, is_feasible = item
        if not is_feasible or metrics["min_dataset_balanced_accuracy"] < best_min_ba - args.selection_min_ba_drop:
            continue
        if args.selection_mode in {"fpr_guarded", "fpr_first"} and metrics["max_protected_real_fpr"] > args.selection_train_fpr_target:
            continue
        if args.selection_mode == "route_floor":
            if float(row["passive_fake_threshold"]) < args.selection_min_passive_threshold:
                continue
            if float(row["protected_delta_abs_anomaly_threshold"]) < args.selection_min_anomaly_threshold:
                continue
        candidates.append(item)
    if not candidates:
        return scored[0]
    if args.selection_mode == "fpr_first":
        candidates.sort(
            key=lambda item: (
                -item[2]["max_protected_real_fpr"],
                item[2]["min_dataset_balanced_accuracy"],
                item[2]["combined"]["balanced_accuracy"],
                float(item[0]["passive_fake_threshold"]),
            ),
            reverse=True,
        )
    else:
        candidates.sort(
            key=lambda item: (
                item[2]["min_dataset_balanced_accuracy"],
                item[2]["combined"]["balanced_accuracy"],
                -item[2]["max_protected_real_fpr"],
                -float(item[0]["passive_fake_threshold"]),
            ),
            reverse=True,
        )
    return candidates[0]


def main() -> None:
    args = parse_args()
    dataset_rows = load_dataset_rows(args.bridge)
    output_rows: list[dict] = []
    summary = {"holdouts": {}}

    for holdout in sorted(dataset_rows):
        train_rows = {name: rows for name, rows in dataset_rows.items() if name != holdout}
        scored = []
        for config_id, rules in enumerate(candidate_rules(), start=1):
            metrics = evaluate_rules(train_rows, rules)
            row = result_row("single", config_id, rules, metrics, feasible(metrics, args))
            scored.append((row, rules, metrics, feasible(metrics, args)))
        best_row, best_rules, best_metrics, best_feasible = select_scored(scored, args)
        holdout_metrics = accuracy_metrics(route_dataset(dataset_rows[holdout], best_rules))
        selected = {
            "config_id": int(best_row["config_id"]),
            "rules": best_rules,
            "train_feasible": best_feasible,
            "train_metrics": best_metrics,
            "holdout_metrics": holdout_metrics,
        }
        summary["holdouts"][holdout] = selected
        output_rows.append(metric_row(holdout, selected, holdout_metrics))

    summary["selection"] = {
        "mode": args.selection_mode,
        "train_fpr_target": args.selection_train_fpr_target,
        "min_ba_drop": args.selection_min_ba_drop,
        "min_passive_threshold": args.selection_min_passive_threshold,
        "min_anomaly_threshold": args.selection_min_anomaly_threshold,
    }
    if output_rows:
        holdout_bas = [float(row["holdout_balanced_accuracy"]) for row in output_rows]
        summary["aggregate"] = {
            "mean_holdout_balanced_accuracy": sum(holdout_bas) / len(holdout_bas),
            "min_holdout_balanced_accuracy": min(holdout_bas),
        }
    else:
        summary["aggregate"] = {"mean_holdout_balanced_accuracy": 0.0, "min_holdout_balanced_accuracy": 0.0}

    write_csv(Path(args.output_csv), output_rows)
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
