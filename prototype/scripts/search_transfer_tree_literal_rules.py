from __future__ import annotations

import argparse
import csv
import json
import math
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
from search_transfer_tree_policy import FEATURE_NAMES, collect_train_rows, fit_tree, stripped_base_rules


FIXED_TREE_CFG = {
    "max_depth": 3,
    "min_samples_leaf": 1,
    "criterion": "gini",
    "paired_positive_weight": 2.0,
    "negative_weight": 1.5,
}

POSITIVE_FLOAT_RULE_KEYS = {
    "passive_baseline_prob": ("min_unprotected_pair_fake_prob", "max_unprotected_pair_fake_prob"),
    "passive_prob_fake": ("min_protected_passive_prob", "max_protected_passive_prob"),
    "passive_delta": ("min_passive_delta", "max_passive_delta"),
    "passive_delta_abs": ("min_passive_delta_abs", "max_passive_delta_abs"),
    "passive_conf_gap": ("min_passive_conf_gap", "max_passive_conf_gap"),
    "prov_bit_accuracy": ("min_bit_accuracy", "max_bit_accuracy"),
    "prov_watermark_l1_mean": ("min_prov_watermark_l1_mean", "max_prov_watermark_l1_mean"),
    "passive_feature_norm": ("min_passive_feature_norm", "max_passive_feature_norm"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Distill the fixed transfer-tree frontier into a literal explicit rule family that mirrors the learned "
            "positive leaves, including per-leaf anomaly behavior."
        )
    )
    parser.add_argument("--base-rules-yaml", required=True)
    parser.add_argument("--canonical-bridge-csv", action="append", required=True)
    parser.add_argument("--paired-bridge-csv", action="append", required=True)
    parser.add_argument("--results-tsv", required=True)
    parser.add_argument("--best-rules-yaml", required=True)
    parser.add_argument("--leaf-summary-json", required=True)
    parser.add_argument("--max-protected-real-fpr", type=float, default=0.08)
    parser.add_argument("--min-protected-fake-recall", type=float, default=0.60)
    parser.add_argument("--min-unprotected-fake-recall", type=float, default=0.60)
    parser.add_argument("--min-dataset-balanced-accuracy", type=float, default=0.0)
    return parser.parse_args()


def _set_upper(bounds: dict, feature: str, threshold: float) -> None:
    current = bounds.setdefault(feature, {})
    current["max"] = threshold if "max" not in current else min(current["max"], threshold)


def _set_lower(bounds: dict, feature: str, threshold: float) -> None:
    current = bounds.setdefault(feature, {})
    current["min"] = threshold if "min" not in current else max(current["min"], threshold)


def extract_positive_leaf_specs(clf) -> list[dict]:
    tree = clf.tree_
    positive_specs: list[dict] = []

    def walk(node_id: int, bounds: dict, allowed_route_labels: set[str]) -> None:
        feature_id = tree.feature[node_id]
        if feature_id < 0:
            value = tree.value[node_id][0]
            pred = int(value[1] > value[0])
            if pred == 1 and allowed_route_labels:
                positive_specs.append(
                    {
                        "bounds": deepcopy(bounds),
                        "allowed_route_labels": sorted(allowed_route_labels),
                        "leaf_value": value.tolist(),
                    }
                )
            return

        feature_name = FEATURE_NAMES[feature_id]
        threshold = float(tree.threshold[node_id])
        left_node = int(tree.children_left[node_id])
        right_node = int(tree.children_right[node_id])

        left_bounds = deepcopy(bounds)
        left_labels = set(allowed_route_labels)
        if feature_name == "route_is_trust":
            left_labels.discard("trust")
        elif feature_name == "route_is_abstain":
            left_labels.discard("abstain")
        else:
            _set_upper(left_bounds, feature_name, threshold)
        walk(left_node, left_bounds, left_labels)

        right_bounds = deepcopy(bounds)
        right_labels = set(allowed_route_labels)
        if feature_name == "route_is_trust":
            right_labels &= {"trust"}
        elif feature_name == "route_is_abstain":
            right_labels &= {"abstain"}
        else:
            _set_lower(right_bounds, feature_name, math.nextafter(threshold, math.inf))
        walk(right_node, right_bounds, right_labels)

    walk(0, {}, {"trust", "abstain"})
    return positive_specs


def leaf_pattern(
    spec: dict,
    *,
    leaf_index: int,
    protected_prob_cap: float,
    anomaly_threshold: float,
    rounding_digits: int | None,
) -> dict:
    pattern = {
        "name": f"literal_leaf_{leaf_index}",
        "allowed_route_labels": spec["allowed_route_labels"],
        "protected_passive_fake_threshold": 0.45,
        "protected_delta_abs_anomaly_threshold": anomaly_threshold,
        "min_protected_passive_prob": 0.45,
        "max_protected_passive_prob": protected_prob_cap,
    }

    for feature_name, bounds in spec["bounds"].items():
        if feature_name not in POSITIVE_FLOAT_RULE_KEYS:
            continue
        min_key, max_key = POSITIVE_FLOAT_RULE_KEYS[feature_name]
        if "min" in bounds:
            value = float(bounds["min"])
            pattern[min_key] = round(value, rounding_digits) if rounding_digits is not None else value
        if "max" in bounds:
            value = float(bounds["max"])
            pattern[max_key] = round(value, rounding_digits) if rounding_digits is not None else value

    return pattern


def candidate_rules(base_rules: dict, positive_specs: list[dict]) -> Iterable[dict]:
    reference = deepcopy(base_rules)
    reference.setdefault("notes", []).append(
        "Literal tree-distillation search keeps the current transfer-pattern routing rule as the explicit-rule reference."
    )
    yield reference

    base = stripped_base_rules(base_rules)
    base_notes = list(base.get("notes", []))
    base_notes.append(
        "Literal tree-distillation search transcribes the fixed78 learned tree into four explicit protected-transfer leaves."
    )
    base_notes.append(
        "Unlike the earlier sparse three-leaf test, this family preserves every positive tree leaf and also aligns the per-leaf anomaly behavior."
    )

    for protected_prob_cap in [0.499999, 0.50]:
        for anomaly_threshold in [0.35, 0.60, 1.00]:
            for rounding_digits in [None, 6, 4, 3]:
                rules = deepcopy(base)
                rules["protected_transfer_patterns"] = [
                    leaf_pattern(
                        spec,
                        leaf_index=index,
                        protected_prob_cap=protected_prob_cap,
                        anomaly_threshold=anomaly_threshold,
                        rounding_digits=rounding_digits,
                    )
                    for index, spec in enumerate(positive_specs, start=1)
                ]
                notes = list(base_notes)
                round_note = "exact thresholds" if rounding_digits is None else f"thresholds rounded to {rounding_digits} decimals"
                notes.append(
                    f"This candidate uses {round_note}, protected max passive prob {protected_prob_cap:.6f}, and per-leaf anomaly threshold {anomaly_threshold:.2f}."
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


def literal_rule_summary(rules: dict) -> dict:
    summary = {
        "mode": "reference",
        "leaf_count": "0",
        "rounding_digits": "",
        "protected_prob_cap": "",
        "anomaly_threshold": "",
    }
    patterns = rules.get("protected_transfer_patterns") or []
    if not patterns:
        return summary

    if not all(str(pattern.get("name", "")).startswith("literal_leaf_") for pattern in patterns):
        summary["mode"] = "other_pattern"
        return summary

    summary["mode"] = "literal_tree"
    summary["leaf_count"] = str(len(patterns))
    protected_prob_caps = [pattern.get("max_protected_passive_prob") for pattern in patterns if "max_protected_passive_prob" in pattern]
    anomaly_thresholds = [
        pattern.get("protected_delta_abs_anomaly_threshold")
        for pattern in patterns
        if "protected_delta_abs_anomaly_threshold" in pattern
    ]
    if protected_prob_caps:
        summary["protected_prob_cap"] = f"{float(protected_prob_caps[0]):.6f}"
    if anomaly_thresholds:
        summary["anomaly_threshold"] = f"{float(anomaly_thresholds[0]):.6f}"

    sample_pattern = patterns[0]
    precision = ""
    for key, value in sample_pattern.items():
        if key.startswith("min_") or key.startswith("max_") or key.endswith("_threshold"):
            text = str(value)
            if "." in text:
                precision = str(len(text.split(".")[1]))
                break
    summary["rounding_digits"] = precision
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
        **literal_rule_summary(rules),
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


def main() -> None:
    args = parse_args()
    canonical_paths = [Path(path) for path in args.canonical_bridge_csv]
    paired_paths = [Path(path) for path in args.paired_bridge_csv]
    base_rules = yaml.safe_load(Path(args.base_rules_yaml).read_text(encoding="utf-8"))

    tree_train_rows = collect_train_rows(canonical_paths, paired_paths, stripped_base_rules(base_rules))
    clf, info = fit_tree(tree_train_rows, **FIXED_TREE_CFG)
    positive_specs = extract_positive_leaf_specs(clf)

    results: list[tuple[dict, dict, dict, dict, bool]] = []
    tsv_rows: list[dict] = []
    for config_id, rules in enumerate(candidate_rules(base_rules, positive_specs), start=1):
        canonical_metrics = evaluate_rules(canonical_paths, rules)
        feasible = is_feasible(canonical_metrics, args)
        paired_eval = paired_metrics(paired_paths, rules)
        results.append(({"config_id": config_id}, rules, canonical_metrics, paired_eval, feasible))
        tsv_rows.append(results_row(config_id, rules, canonical_metrics, paired_eval, feasible))

    results.sort(key=sort_key, reverse=True)
    best_meta, best_rules, best_canonical, best_paired, best_feasible = results[0]

    write_tsv(Path(args.results_tsv), tsv_rows)
    Path(args.best_rules_yaml).parent.mkdir(parents=True, exist_ok=True)
    Path(args.best_rules_yaml).write_text(yaml.safe_dump(best_rules, sort_keys=False, allow_unicode=True), encoding="utf-8")

    leaf_summary = {
        "fixed_tree_cfg": FIXED_TREE_CFG,
        "tree_info": info,
        "positive_leaf_specs": positive_specs,
        "best_config_id": int(best_meta["config_id"]),
        "best_feasible": best_feasible,
        "best_canonical": best_canonical,
        "best_paired": best_paired,
    }
    Path(args.leaf_summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.leaf_summary_json).write_text(json.dumps(leaf_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "positive_leaf_count": len(positive_specs),
                "best_config_id": int(best_meta["config_id"]),
                "best_feasible": best_feasible,
                "canonical_overall_balanced_accuracy": best_canonical["combined"]["balanced_accuracy"],
                "canonical_min_dataset_balanced_accuracy": best_canonical["min_dataset_balanced_accuracy"],
                "canonical_max_protected_real_fpr": best_canonical["max_protected_real_fpr"],
                "paired_mean_fake_accuracy": best_paired["mean_fake_accuracy"],
                "paired_min_fake_accuracy": best_paired["min_fake_accuracy"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
