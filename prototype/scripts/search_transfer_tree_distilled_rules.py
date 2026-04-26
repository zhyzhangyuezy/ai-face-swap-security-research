from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List

import yaml


def _bootstrap_scripts() -> None:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))


_bootstrap_scripts()

from run_routing_sanity import compute_unprotected_map, load_csv, route_row
from search_routing_thresholds import dataset_name_from_path, validate_bridge_rows
from search_routing_thresholds_constrained import accuracy_metrics, is_feasible


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search a sparse hand-written rule family distilled from the current shallow transfer tree, "
            "to test whether the tree frontier can be reproduced by a paper-friendlier explicit policy."
        )
    )
    parser.add_argument("--base-rules-yaml", required=True)
    parser.add_argument("--canonical-bridge-csv", action="append", required=True)
    parser.add_argument("--paired-bridge-csv", action="append", required=True)
    parser.add_argument("--results-tsv", required=True)
    parser.add_argument("--best-rules-yaml", required=True)
    parser.add_argument("--max-protected-real-fpr", type=float, default=0.08)
    parser.add_argument("--min-protected-fake-recall", type=float, default=0.60)
    parser.add_argument("--min-unprotected-fake-recall", type=float, default=0.60)
    parser.add_argument("--min-dataset-balanced-accuracy", type=float, default=0.0)
    return parser.parse_args()


def stripped_base_rules(base_rules: dict) -> dict:
    rules = deepcopy(base_rules)
    for key in ["protected_transfer", "protected_transfer_patterns", "search_summary", "constraints"]:
        rules.pop(key, None)
    return rules


def candidate_rules(base_rules: dict) -> Iterable[dict]:
    current_safe = deepcopy(base_rules)
    current_safe.setdefault("notes", []).append(
        "Tree-distillation search keeps the current pattern-safe routing rule as the explicit-rule reference."
    )
    yield current_safe

    base = stripped_base_rules(base_rules)
    base_notes = list(base.get("notes", []))
    base_notes.append(
        "Tree-distillation search hand-writes a sparse protected-transfer family around the winning shallow tree splits."
    )
    base_notes.append(
        "The target is not to invent a new detector, but to see whether a simpler explicit rule family can match the learned tree's frontier."
    )

    route_label_sets = [
        ["trust", "abstain"],
        ["abstain"],
    ]
    max_protected_probs = [0.495, 0.50]
    neg_delta_thresholds = [-0.015, -0.01, -0.005]
    conf_gap_thresholds = [0.035, 0.04, 0.045]
    neg_delta_abs_caps = [0.06, 0.07, 0.08]
    feature_norm_caps = [14.0, 14.25, 14.5]
    min_bit_accuracy_options = [None, 0.98]

    for route_labels in route_label_sets:
        for max_protected_prob in max_protected_probs:
            for neg_delta_threshold in neg_delta_thresholds:
                for conf_gap_threshold in conf_gap_thresholds:
                    for neg_delta_abs_cap in neg_delta_abs_caps:
                        for feature_norm_cap in feature_norm_caps:
                            for min_bit_accuracy in min_bit_accuracy_options:
                                rules = deepcopy(base)
                                patterns = []

                                common_cfg = {
                                    "allowed_route_labels": route_labels,
                                    "protected_passive_fake_threshold": 0.45,
                                    "protected_delta_abs_anomaly_threshold": 0.35,
                                    "min_protected_passive_prob": 0.45,
                                    "max_protected_passive_prob": max_protected_prob,
                                }
                                if min_bit_accuracy is not None:
                                    common_cfg["min_bit_accuracy"] = min_bit_accuracy

                                neg_small_gap = {
                                    "name": "tree_neg_small_gap",
                                    **common_cfg,
                                    "max_passive_delta": neg_delta_threshold,
                                    "max_passive_conf_gap": conf_gap_threshold,
                                }
                                patterns.append(neg_small_gap)

                                neg_wide_gap = {
                                    "name": "tree_neg_wide_gap",
                                    **common_cfg,
                                    "max_passive_delta": neg_delta_threshold,
                                    "min_passive_conf_gap": conf_gap_threshold,
                                    "max_passive_delta_abs": neg_delta_abs_cap,
                                }
                                patterns.append(neg_wide_gap)

                                low_feature = {
                                    "name": "tree_low_feature",
                                    **common_cfg,
                                    "min_passive_delta": neg_delta_threshold,
                                    "max_passive_feature_norm": feature_norm_cap,
                                }
                                patterns.append(low_feature)

                                rules["protected_transfer_patterns"] = patterns
                                notes = list(base_notes)
                                notes.append(
                                    "This candidate distills the tree into three explicit leaves: two negative-delta bands and one low-feature-norm band."
                                )
                                rules["notes"] = notes
                                yield rules


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


def paired_metrics(bridge_paths: List[Path], rules: dict) -> dict:
    dataset_metrics: Dict[str, dict] = {}
    for bridge_path in bridge_paths:
        bridge_rows = load_csv(bridge_path)
        validate_bridge_rows(bridge_path, bridge_rows)
        unprotected_map = compute_unprotected_map(bridge_rows)
        routed_rows = [route_row(row, unprotected_map, rules) for row in bridge_rows]
        dataset_metrics[dataset_name_from_path(bridge_path)] = accuracy_metrics(routed_rows)

    fake_accuracies = [metrics["fake_accuracy"] for metrics in dataset_metrics.values()]
    bas = [metrics["balanced_accuracy"] for metrics in dataset_metrics.values()]
    return {
        "datasets": dataset_metrics,
        "mean_fake_accuracy": mean(fake_accuracies),
        "min_fake_accuracy": min(fake_accuracies),
        "mean_balanced_accuracy": mean(bas),
    }


def distilled_rule_summary(rules: dict) -> dict:
    summary = {
        "mode": "base_reference",
        "allowed_route_labels": "",
        "max_protected_prob": "",
        "neg_delta_threshold": "",
        "conf_gap_threshold": "",
        "neg_delta_abs_cap": "",
        "feature_norm_cap": "",
        "min_bit_accuracy": "",
    }
    patterns = rules.get("protected_transfer_patterns") or []
    if not patterns:
        return summary

    if not all(str(pattern.get("name", "")).startswith("tree_") for pattern in patterns):
        summary["mode"] = "pattern_reference"
        return summary

    first = patterns[0]
    summary["mode"] = "tree_distilled"
    summary["allowed_route_labels"] = ",".join(first.get("allowed_route_labels", []))
    if "max_protected_passive_prob" in first:
        summary["max_protected_prob"] = f"{float(first['max_protected_passive_prob']):.6f}"
    if "max_passive_delta" in first:
        summary["neg_delta_threshold"] = f"{float(first['max_passive_delta']):.6f}"
    if "max_passive_conf_gap" in first:
        summary["conf_gap_threshold"] = f"{float(first['max_passive_conf_gap']):.6f}"
    if "min_bit_accuracy" in first:
        summary["min_bit_accuracy"] = f"{float(first['min_bit_accuracy']):.6f}"

    for pattern in patterns:
        if pattern.get("name") == "tree_neg_wide_gap" and "max_passive_delta_abs" in pattern:
            summary["neg_delta_abs_cap"] = f"{float(pattern['max_passive_delta_abs']):.6f}"
        if pattern.get("name") == "tree_low_feature" and "max_passive_feature_norm" in pattern:
            summary["feature_norm_cap"] = f"{float(pattern['max_passive_feature_norm']):.6f}"
    return summary


def results_row(config_id: int, rules: dict, canonical_metrics: dict, paired_eval: dict, feasible: bool) -> dict:
    row = {
        "config_id": str(config_id),
        "feasible": "1" if feasible else "0",
        "canonical_overall_balanced_accuracy": f"{canonical_metrics['combined']['balanced_accuracy']:.6f}",
        "canonical_min_dataset_balanced_accuracy": f"{canonical_metrics['min_dataset_balanced_accuracy']:.6f}",
        "canonical_max_protected_real_fpr": f"{canonical_metrics['max_protected_real_fpr']:.6f}",
        "canonical_min_protected_fake_recall": f"{canonical_metrics['min_protected_fake_recall']:.6f}",
        "canonical_min_unprotected_fake_recall": f"{canonical_metrics['min_unprotected_fake_recall']:.6f}",
        "paired_mean_fake_accuracy": f"{paired_eval['mean_fake_accuracy']:.6f}",
        "paired_min_fake_accuracy": f"{paired_eval['min_fake_accuracy']:.6f}",
        "paired_mean_balanced_accuracy": f"{paired_eval['mean_balanced_accuracy']:.6f}",
        **distilled_rule_summary(rules),
    }
    for dataset_name, dataset_metrics in paired_eval["datasets"].items():
        row[f"{dataset_name}_fake_accuracy"] = f"{dataset_metrics['fake_accuracy']:.6f}"
        row[f"{dataset_name}_balanced_accuracy"] = f"{dataset_metrics['balanced_accuracy']:.6f}"
        row[f"{dataset_name}_protected_real_fpr"] = f"{dataset_metrics['protected_real_fpr']:.6f}"
    return row


def sort_key(result: tuple[dict, dict, dict, dict, bool]) -> tuple[float, float, float, float, float, float]:
    _, _, canonical_metrics, paired_eval, feasible = result
    return (
        1.0 if feasible else 0.0,
        paired_eval["mean_fake_accuracy"],
        paired_eval["min_fake_accuracy"],
        canonical_metrics["min_dataset_balanced_accuracy"],
        canonical_metrics["combined"]["balanced_accuracy"],
        -canonical_metrics["max_protected_real_fpr"],
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
    canonical_paths = [Path(path) for path in args.canonical_bridge_csv]
    paired_paths = [Path(path) for path in args.paired_bridge_csv]
    base_rules = yaml.safe_load(Path(args.base_rules_yaml).read_text(encoding="utf-8"))

    scored: List[tuple[dict, dict, dict, dict, bool]] = []
    for config_id, rules in enumerate(candidate_rules(base_rules), start=1):
        canonical_metrics = evaluate_rules(canonical_paths, rules)
        feasible = is_feasible(canonical_metrics, args)
        paired_eval = paired_metrics(paired_paths, rules)
        row = results_row(config_id, rules, canonical_metrics, paired_eval, feasible)
        scored.append((row, rules, canonical_metrics, paired_eval, feasible))

    scored.sort(key=sort_key, reverse=True)
    write_tsv(Path(args.results_tsv), [row for row, _, _, _, _ in scored])

    best_row, best_rules, best_canonical_metrics, best_paired_eval, best_feasible = scored[0]
    payload = deepcopy(best_rules)
    payload["constraints"] = {
        "max_protected_real_fpr": args.max_protected_real_fpr,
        "min_protected_fake_recall": args.min_protected_fake_recall,
        "min_unprotected_fake_recall": args.min_unprotected_fake_recall,
        "min_dataset_balanced_accuracy": args.min_dataset_balanced_accuracy,
    }
    payload["search_summary"] = {
        "selected_config_id": int(best_row["config_id"]),
        "feasible": bool(best_feasible),
        "canonical_overall_balanced_accuracy": round(best_canonical_metrics["combined"]["balanced_accuracy"], 6),
        "canonical_min_dataset_balanced_accuracy": round(best_canonical_metrics["min_dataset_balanced_accuracy"], 6),
        "canonical_max_protected_real_fpr": round(best_canonical_metrics["max_protected_real_fpr"], 6),
        "canonical_min_protected_fake_recall": round(best_canonical_metrics["min_protected_fake_recall"], 6),
        "canonical_min_unprotected_fake_recall": round(best_canonical_metrics["min_unprotected_fake_recall"], 6),
        "paired_mean_fake_accuracy": round(best_paired_eval["mean_fake_accuracy"], 6),
        "paired_min_fake_accuracy": round(best_paired_eval["min_fake_accuracy"], 6),
        "paired_mean_balanced_accuracy": round(best_paired_eval["mean_balanced_accuracy"], 6),
        "distilled_rule_summary": distilled_rule_summary(best_rules),
        "canonical_datasets": best_canonical_metrics["datasets"],
        "paired_datasets": best_paired_eval["datasets"],
    }
    write_yaml(Path(args.best_rules_yaml), payload)
    print(json.dumps(payload["search_summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
