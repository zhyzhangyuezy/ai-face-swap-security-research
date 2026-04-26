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
            "Search a small family of pattern-based protected-transfer rules on top of a fixed routing backbone "
            "while preserving canonical protected-real constraints."
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
    rules.pop("protected_transfer", None)
    rules.pop("protected_transfer_patterns", None)
    return rules


def candidate_rules(base_rules: dict) -> Iterable[dict]:
    current_safe = deepcopy(base_rules)
    current_safe.setdefault("notes", []).append(
        "Pattern search keeps the current scalar transferpair rule as a reference frontier."
    )
    yield current_safe

    base = stripped_base_rules(base_rules)
    base_notes = list(base.get("notes", []))
    base_notes.append(
        "Pattern search uses route-specific protected-transfer bands instead of one shared scalar trigger."
    )
    base_notes.append(
        "Trust patterns are intentionally stricter on source-real family evidence; abstain patterns focus on the borderline band."
    )

    trust_min_pair_probs = [None, 0.46, 0.48, 0.50]
    trust_min_current_probs = [None, 0.45, 0.46]
    abstain_min_pair_probs = [0.20, 0.25, 0.30, 0.35, 0.40]
    abstain_min_current_probs = [0.45, 0.46, 0.47]
    abstain_max_current_probs = [None, 0.495]
    abstain_max_conf_gaps = [None, 0.045]
    abstain_min_bit_accuracies = [0.98, 0.99]

    for trust_min_pair_prob in trust_min_pair_probs:
        for trust_min_current_prob in trust_min_current_probs:
            for abstain_min_pair_prob in abstain_min_pair_probs:
                for abstain_min_current_prob in abstain_min_current_probs:
                    for abstain_max_current_prob in abstain_max_current_probs:
                        for abstain_max_conf_gap in abstain_max_conf_gaps:
                            for abstain_min_bit_accuracy in abstain_min_bit_accuracies:
                                rules = deepcopy(base)
                                patterns = []

                                trust_pattern = {
                                    "name": "trust_band",
                                    "allowed_route_labels": ["trust"],
                                    "protected_passive_fake_threshold": 0.45,
                                    "protected_delta_abs_anomaly_threshold": 0.35,
                                    "min_bit_accuracy": 0.98,
                                }
                                if trust_min_pair_prob is not None:
                                    trust_pattern["min_unprotected_pair_fake_prob"] = trust_min_pair_prob
                                if trust_min_current_prob is not None:
                                    trust_pattern["min_protected_passive_prob"] = trust_min_current_prob
                                if trust_min_pair_prob is not None or trust_min_current_prob is not None:
                                    patterns.append(trust_pattern)

                                abstain_pattern = {
                                    "name": "abstain_band",
                                    "allowed_route_labels": ["abstain"],
                                    "protected_passive_fake_threshold": 0.45,
                                    "protected_delta_abs_anomaly_threshold": 0.35,
                                    "min_unprotected_pair_fake_prob": abstain_min_pair_prob,
                                    "min_protected_passive_prob": abstain_min_current_prob,
                                    "min_bit_accuracy": abstain_min_bit_accuracy,
                                }
                                if abstain_max_current_prob is not None:
                                    abstain_pattern["max_protected_passive_prob"] = abstain_max_current_prob
                                if abstain_max_conf_gap is not None:
                                    abstain_pattern["max_passive_conf_gap"] = abstain_max_conf_gap
                                patterns.append(abstain_pattern)

                                rules["protected_transfer_patterns"] = patterns
                                notes = list(base_notes)
                                notes.append(
                                    "This candidate probes whether a tighter trust gate plus an abstain-only borderline band "
                                    "can keep paired gains while shaving off canonical protected-real damage."
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


def pattern_summary(rules: dict) -> dict:
    summary = {
        "mode": "scalar_reference" if rules.get("protected_transfer") else "pattern_family",
        "trust_min_pair_prob": "",
        "trust_min_protected_prob": "",
        "abstain_min_pair_prob": "",
        "abstain_min_protected_prob": "",
        "abstain_max_protected_prob": "",
        "abstain_max_conf_gap": "",
        "abstain_min_bit_accuracy": "",
    }
    patterns = rules.get("protected_transfer_patterns") or []
    for pattern in patterns:
        name = pattern.get("name")
        if name == "trust_band":
            if "min_unprotected_pair_fake_prob" in pattern:
                summary["trust_min_pair_prob"] = f"{float(pattern['min_unprotected_pair_fake_prob']):.6f}"
            if "min_protected_passive_prob" in pattern:
                summary["trust_min_protected_prob"] = f"{float(pattern['min_protected_passive_prob']):.6f}"
        elif name == "abstain_band":
            if "min_unprotected_pair_fake_prob" in pattern:
                summary["abstain_min_pair_prob"] = f"{float(pattern['min_unprotected_pair_fake_prob']):.6f}"
            if "min_protected_passive_prob" in pattern:
                summary["abstain_min_protected_prob"] = f"{float(pattern['min_protected_passive_prob']):.6f}"
            if "max_protected_passive_prob" in pattern:
                summary["abstain_max_protected_prob"] = f"{float(pattern['max_protected_passive_prob']):.6f}"
            if "max_passive_conf_gap" in pattern:
                summary["abstain_max_conf_gap"] = f"{float(pattern['max_passive_conf_gap']):.6f}"
            if "min_bit_accuracy" in pattern:
                summary["abstain_min_bit_accuracy"] = f"{float(pattern['min_bit_accuracy']):.6f}"
    return summary


def results_row(config_id: int, rules: dict, canonical_metrics: dict, paired_eval: dict, feasible: bool) -> dict:
    row = {
        "config_id": str(config_id),
        "feasible": "1" if feasible else "0",
        "base_passive_fake_threshold": f"{float(rules['fallback']['passive_fake_threshold']):.6f}",
        "base_protected_delta_abs_anomaly_threshold": f"{float(rules['fallback']['protected_delta_abs_anomaly_threshold']):.6f}",
        "canonical_overall_balanced_accuracy": f"{canonical_metrics['combined']['balanced_accuracy']:.6f}",
        "canonical_min_dataset_balanced_accuracy": f"{canonical_metrics['min_dataset_balanced_accuracy']:.6f}",
        "canonical_max_protected_real_fpr": f"{canonical_metrics['max_protected_real_fpr']:.6f}",
        "canonical_min_protected_fake_recall": f"{canonical_metrics['min_protected_fake_recall']:.6f}",
        "canonical_min_unprotected_fake_recall": f"{canonical_metrics['min_unprotected_fake_recall']:.6f}",
        "paired_mean_fake_accuracy": f"{paired_eval['mean_fake_accuracy']:.6f}",
        "paired_min_fake_accuracy": f"{paired_eval['min_fake_accuracy']:.6f}",
        "paired_mean_balanced_accuracy": f"{paired_eval['mean_balanced_accuracy']:.6f}",
        **pattern_summary(rules),
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
            "mode": best_row["mode"],
            "canonical_overall_balanced_accuracy": float(best_row["canonical_overall_balanced_accuracy"]),
            "canonical_min_dataset_balanced_accuracy": float(best_row["canonical_min_dataset_balanced_accuracy"]),
            "canonical_max_protected_real_fpr": float(best_row["canonical_max_protected_real_fpr"]),
            "canonical_min_protected_fake_recall": float(best_row["canonical_min_protected_fake_recall"]),
            "canonical_min_unprotected_fake_recall": float(best_row["canonical_min_unprotected_fake_recall"]),
            "paired_mean_fake_accuracy": float(best_row["paired_mean_fake_accuracy"]),
            "paired_min_fake_accuracy": float(best_row["paired_min_fake_accuracy"]),
            "paired_mean_balanced_accuracy": float(best_row["paired_mean_balanced_accuracy"]),
            "pattern_summary": pattern_summary(best_rules),
            "canonical_datasets": best_canonical_metrics["datasets"],
            "paired_datasets": best_paired_eval["datasets"],
        },
    }
    write_yaml(Path(args.best_rules_yaml), best_payload)
    print(json.dumps(best_payload["search_summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
