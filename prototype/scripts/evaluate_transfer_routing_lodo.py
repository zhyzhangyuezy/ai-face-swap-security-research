from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict

import yaml


def _bootstrap_scripts() -> None:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))


_bootstrap_scripts()

from run_routing_sanity import compute_unprotected_map, load_csv, route_row
from search_routing_thresholds import dataset_name_from_path, validate_bridge_rows
from search_routing_thresholds_constrained import accuracy_metrics, is_feasible
from search_transfer_routing_rules import candidate_rules, sort_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Leave-one-dataset-out validation for the transfer-aware routing family, "
            "using canonical datasets as safety guards and paired datasets as the optimization target."
        )
    )
    parser.add_argument("--base-rules-yaml", required=True)
    parser.add_argument("--canonical-bridge-csv", action="append", required=True)
    parser.add_argument("--paired-bridge-csv", action="append", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--max-protected-real-fpr", type=float, default=0.08)
    parser.add_argument("--min-protected-fake-recall", type=float, default=0.60)
    parser.add_argument("--min-unprotected-fake-recall", type=float, default=0.60)
    parser.add_argument("--min-dataset-balanced-accuracy", type=float, default=0.0)
    parser.add_argument("--selection-mode", choices=["default", "route_floor"], default="default")
    parser.add_argument("--selection-min-ba-drop", type=float, default=0.025)
    parser.add_argument("--selection-min-transfer-threshold", type=float, default=0.45)
    parser.add_argument("--selection-min-transfer-anomaly-threshold", type=float, default=0.35)
    return parser.parse_args()


def load_cached_rows(paths: list[Path]) -> dict[Path, tuple[list[dict], dict[str, float]]]:
    cache: dict[Path, tuple[list[dict], dict[str, float]]] = {}
    for path in paths:
        rows = load_csv(path)
        validate_bridge_rows(path, rows)
        cache[path] = (rows, compute_unprotected_map(rows))
    return cache


def evaluate_paths(cache: dict[Path, tuple[list[dict], dict[str, float]]], paths: list[Path], rules: dict) -> dict:
    dataset_metrics: Dict[str, dict] = {}
    combined_rows: list[dict] = []
    for path in paths:
        rows, unprotected_map = cache[path]
        routed_rows = [route_row(row, unprotected_map, rules) for row in rows]
        dataset_metrics[dataset_name_from_path(path)] = accuracy_metrics(routed_rows)
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


def paired_objective(cache: dict[Path, tuple[list[dict], dict[str, float]]], paths: list[Path], rules: dict) -> dict:
    dataset_metrics: Dict[str, dict] = {}
    for path in paths:
        rows, unprotected_map = cache[path]
        routed_rows = [route_row(row, unprotected_map, rules) for row in rows]
        dataset_metrics[dataset_name_from_path(path)] = accuracy_metrics(routed_rows)
    fake_accuracies = [item["fake_accuracy"] for item in dataset_metrics.values()]
    bas = [item["balanced_accuracy"] for item in dataset_metrics.values()]
    return {
        "datasets": dataset_metrics,
        "mean_fake_accuracy": sum(fake_accuracies) / len(fake_accuracies),
        "min_fake_accuracy": min(fake_accuracies),
        "mean_balanced_accuracy": sum(bas) / len(bas),
    }


def transfer_thresholds(rules: dict) -> tuple[float | None, float | None]:
    transfer_cfg = rules.get("protected_transfer")
    if not transfer_cfg:
        return None, None
    return (
        float(transfer_cfg.get("protected_passive_fake_threshold", rules["fallback"]["protected_passive_fake_threshold"])),
        float(
            transfer_cfg.get(
                "protected_delta_abs_anomaly_threshold",
                rules["fallback"]["protected_delta_abs_anomaly_threshold"],
            )
        ),
    )


def select_scored(scored: list[tuple[dict, dict, dict, dict, bool]], args: argparse.Namespace) -> tuple[dict, dict, dict, dict, bool]:
    scored.sort(key=sort_key, reverse=True)
    if args.selection_mode == "default":
        return scored[0]

    best_min_ba = scored[0][2]["min_dataset_balanced_accuracy"]
    candidates: list[tuple[dict, dict, dict, dict, bool]] = []
    for item in scored:
        _, rules, canonical_metrics, _, feasible = item
        if not feasible:
            continue
        if canonical_metrics["min_dataset_balanced_accuracy"] < best_min_ba - args.selection_min_ba_drop:
            continue
        transfer_threshold, transfer_anomaly = transfer_thresholds(rules)
        if transfer_threshold is not None and transfer_threshold < args.selection_min_transfer_threshold:
            continue
        if transfer_anomaly is not None and transfer_anomaly < args.selection_min_transfer_anomaly_threshold:
            continue
        candidates.append(item)

    if not candidates:
        return scored[0]

    candidates.sort(key=sort_key, reverse=True)
    return candidates[0]


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
    canonical_paths = [Path(path) for path in args.canonical_bridge_csv]
    paired_paths = [Path(path) for path in args.paired_bridge_csv]
    cache = load_cached_rows(canonical_paths + paired_paths)
    base_rules = yaml.safe_load(Path(args.base_rules_yaml).read_text(encoding="utf-8"))

    output_rows: list[dict] = []
    summary = {"holdouts": {}, "selection": vars(args)}

    for holdout in canonical_paths:
        train_paths = [path for path in canonical_paths if path != holdout]
        scored: list[tuple[dict, dict, dict, dict, bool]] = []
        for config_id, rules in enumerate(candidate_rules(base_rules), start=1):
            canonical_metrics = evaluate_paths(cache, train_paths, rules)
            feasible = is_feasible(canonical_metrics, args)
            pair_eval = paired_objective(cache, paired_paths, rules)
            scored.append(({"config_id": config_id}, rules, canonical_metrics, pair_eval, feasible))

        best_row, best_rules, best_train_canonical, best_pair_eval, best_feasible = select_scored(scored, args)
        holdout_eval = evaluate_paths(cache, [holdout], best_rules)
        holdout_name = dataset_name_from_path(holdout)
        holdout_metrics = holdout_eval["datasets"][holdout_name]
        transfer_cfg = best_rules.get("protected_transfer") or {}
        output_row = {
            "holdout_dataset": holdout_name,
            "selected_config_id": str(best_row["config_id"]),
            "selected_transfer": "1" if transfer_cfg else "0",
            "selected_transfer_min_pair_prob": (
                "" if "min_unprotected_pair_fake_prob" not in transfer_cfg else f"{float(transfer_cfg['min_unprotected_pair_fake_prob']):.6f}"
            ),
            "selected_transfer_threshold": (
                "" if "protected_passive_fake_threshold" not in transfer_cfg else f"{float(transfer_cfg['protected_passive_fake_threshold']):.6f}"
            ),
            "selected_transfer_anomaly_threshold": (
                "" if "protected_delta_abs_anomaly_threshold" not in transfer_cfg else f"{float(transfer_cfg['protected_delta_abs_anomaly_threshold']):.6f}"
            ),
            "train_feasible": "1" if best_feasible else "0",
            "train_min_dataset_ba": f"{best_train_canonical['min_dataset_balanced_accuracy']:.6f}",
            "train_max_protected_real_fpr": f"{best_train_canonical['max_protected_real_fpr']:.6f}",
            "paired_mean_fake_accuracy": f"{best_pair_eval['mean_fake_accuracy']:.6f}",
            "holdout_balanced_accuracy": f"{holdout_metrics['balanced_accuracy']:.6f}",
            "holdout_fake_accuracy": f"{holdout_metrics['fake_accuracy']:.6f}",
            "holdout_protected_real_fpr": f"{holdout_metrics['protected_real_fpr']:.6f}",
            "holdout_protected_fake_recall": f"{holdout_metrics['protected_fake_recall']:.6f}",
            "holdout_unprotected_fake_recall": f"{holdout_metrics['unprotected_fake_recall']:.6f}",
        }
        output_rows.append(output_row)
        summary["holdouts"][holdout_name] = {
            "selected_config_id": int(best_row["config_id"]),
            "selected_rules": best_rules,
            "train_feasible": best_feasible,
            "train_canonical": best_train_canonical,
            "paired_objective": best_pair_eval,
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
