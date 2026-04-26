from __future__ import annotations

import argparse
import csv
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

from run_routing_sanity import compute_unprotected_map, load_csv, route_row, safe_float, transfer_override_match
from search_routing_thresholds import dataset_name_from_path, validate_bridge_rows
from search_routing_thresholds_constrained import accuracy_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search a minimal VCF-oriented protected-real guard on top of a fixed routing backbone. "
            "The guard only vetoes protected fake calls under strong-provenance / low-shift conditions."
        )
    )
    parser.add_argument("--base-rules-yaml", required=True)
    parser.add_argument("--vcf-bridge-csv", action="append", required=True)
    parser.add_argument("--canonical-bridge-csv", action="append", required=True)
    parser.add_argument("--deeper-bridge-csv", action="append", required=True)
    parser.add_argument("--paired-bridge-csv", action="append", required=True)
    parser.add_argument("--results-tsv", required=True)
    parser.add_argument("--best-rules-yaml", required=True)
    parser.add_argument("--min-vcf-protected-real-fpr-improvement", type=float, default=0.02)
    parser.add_argument("--max-vcf-fake-accuracy-drop", type=float, default=0.02)
    parser.add_argument("--max-canonical-min-ba-drop", type=float, default=0.002)
    parser.add_argument("--max-canonical-prfpr-increase", type=float, default=0.0)
    parser.add_argument("--max-deeper-ba-drop", type=float, default=0.01)
    parser.add_argument("--max-paired-mean-fake-drop", type=float, default=0.01)
    parser.add_argument("--max-paired-min-fake-drop", type=float, default=0.02)
    return parser.parse_args()


def stripped_base_rules(base_rules: dict) -> dict:
    rules = deepcopy(base_rules)
    for key in [
        "protected_real_guard",
        "protected_real_guard_patterns",
        "search_summary",
        "constraints",
    ]:
        rules.pop(key, None)
    return rules


def candidate_rules(base_rules: dict) -> Iterable[dict]:
    reference = deepcopy(base_rules)
    reference.setdefault("notes", []).append(
        "VCF real-guard search keeps the current no-guard routing backbone as a reference row."
    )
    yield reference

    base = stripped_base_rules(base_rules)
    base_notes = list(base.get("notes", []))
    base_notes.append(
        "VCF real-guard search adds a protected-only veto that flips fake back to real under strong provenance and low pair shift."
    )
    base_notes.append(
        "This family is intentionally small and conservative; it does not touch unprotected routing or anomaly handling."
    )

    allowed_route_labels_options = [["trust"], ["trust", "abstain"]]
    min_bit_accuracies = [0.975, 0.98, 0.99]
    max_unprotected_pair_probs = [0.20, 0.25, 0.30, 0.35, 0.40]
    max_protected_probs = [0.32, 0.35, 0.40, 0.45]
    max_passive_deltas = [0.005, 0.015, 0.03]
    max_passive_conf_gaps = [0.05, 0.10, 0.15]

    for allowed_route_labels in allowed_route_labels_options:
        for min_bit_accuracy in min_bit_accuracies:
            for max_unprotected_pair_prob in max_unprotected_pair_probs:
                for max_protected_prob in max_protected_probs:
                    for max_passive_delta in max_passive_deltas:
                        for max_passive_conf_gap in max_passive_conf_gaps:
                            rules = deepcopy(base)
                            rules["protected_real_guard_patterns"] = [
                                {
                                    "name": "vcf_real_guard",
                                    "allowed_route_labels": allowed_route_labels,
                                    "min_bit_accuracy": min_bit_accuracy,
                                    "max_unprotected_pair_fake_prob": max_unprotected_pair_prob,
                                    "max_protected_passive_prob": max_protected_prob,
                                    "max_passive_delta": max_passive_delta,
                                    "max_passive_conf_gap": max_passive_conf_gap,
                                }
                            ]
                            notes = list(base_notes)
                            notes.append(
                                "This candidate only accepts protected samples back to real when the unprotected mate is not too fake-like, "
                                "the protected passive score stays in a moderate band, and provenance remains strong."
                            )
                            rules["notes"] = notes
                            yield rules


def build_base_routed_group(bridge_paths: List[Path], base_rules: dict) -> Dict[str, List[Dict[str, str]]]:
    dataset_rows: Dict[str, List[Dict[str, str]]] = {}
    for bridge_path in bridge_paths:
        bridge_rows = load_csv(bridge_path)
        validate_bridge_rows(bridge_path, bridge_rows)
        unprotected_map = compute_unprotected_map(bridge_rows)
        routed_rows = [route_row(row, unprotected_map, base_rules) for row in bridge_rows]
        dataset_rows[dataset_name_from_path(bridge_path)] = routed_rows
    return dataset_rows


def apply_guard_to_routed_rows(rows: List[Dict[str, str]], rules: dict) -> List[Dict[str, str]]:
    guard_cfg = rules.get("protected_real_guard") or {}
    guard_patterns = rules.get("protected_real_guard_patterns") or []
    if not guard_cfg and not guard_patterns:
        return [dict(row) for row in rows]

    guarded_rows: List[Dict[str, str]] = []
    for row in rows:
        updated = dict(row)
        updated["protected_real_guard_flag"] = "0"
        updated["protected_real_guard_reason"] = ""
        updated["protected_real_guard_pattern"] = ""

        protection = updated.get("protection", "")
        passive_pred = updated.get("passive_pred_label", "")
        anomaly_flag = updated.get("anomaly_flag", "0") == "1"
        if protection != "protected" or passive_pred != "fake" or anomaly_flag:
            guarded_rows.append(updated)
            continue

        route_label = updated.get("route_label", "")
        base_prob = safe_float(updated.get("passive_baseline_prob", ""))
        passive_prob = safe_float(updated.get("passive_prob_fake", ""))
        passive_conf_gap = safe_float(updated.get("passive_conf_gap", ""))
        passive_delta = safe_float(updated.get("passive_delta", ""))
        passive_delta_abs = safe_float(updated.get("passive_delta_abs", ""))
        bit_accuracy = safe_float(updated.get("prov_bit_accuracy", ""))
        prov_watermark_l1_mean = safe_float(updated.get("prov_watermark_l1_mean", ""))
        passive_feature_norm = safe_float(updated.get("passive_feature_norm", ""))

        guard_candidates = []
        for index, pattern_cfg in enumerate(guard_patterns, start=1):
            pattern_name = str(pattern_cfg.get("name", f"pattern_{index}"))
            guard_candidates.append((f"protected_real_guard_pattern:{pattern_name}", pattern_name, pattern_cfg))
        if guard_cfg:
            guard_candidates.append(("protected_real_guard_override", "", guard_cfg))

        matched = False
        for reason, pattern_name, candidate_cfg in guard_candidates:
            if transfer_override_match(
                base_prob=base_prob,
                passive_prob=passive_prob,
                passive_conf_gap=passive_conf_gap,
                passive_delta=passive_delta,
                passive_delta_abs=passive_delta_abs,
                bit_accuracy=bit_accuracy,
                prov_watermark_l1_mean=prov_watermark_l1_mean,
                passive_feature_norm=passive_feature_norm,
                route_label=route_label,
                cfg=candidate_cfg,
            ):
                updated["protected_real_guard_flag"] = "1"
                updated["protected_real_guard_reason"] = reason
                updated["protected_real_guard_pattern"] = pattern_name
                updated["final_decision"] = "protected_real_guard_accept_real"
                updated["effective_pred_label"] = "real"
                updated["effective_correct"] = "1" if updated.get("label") == "real" else "0"
                matched = True
                break

        if not matched:
            updated["protected_real_guard_flag"] = "0"
            updated["protected_real_guard_reason"] = ""
            updated["protected_real_guard_pattern"] = ""
        guarded_rows.append(updated)
    return guarded_rows


def evaluate_group(dataset_rows: Dict[str, List[Dict[str, str]]], rules: dict) -> dict:
    dataset_metrics: Dict[str, dict] = {}
    combined_rows: List[Dict[str, str]] = []

    for dataset_name, rows in dataset_rows.items():
        guarded_rows = apply_guard_to_routed_rows(rows, rules)
        dataset_metrics[dataset_name] = accuracy_metrics(guarded_rows)
        combined_rows.extend(guarded_rows)

    combined = accuracy_metrics(combined_rows)
    guard_rows = [row for row in combined_rows if row.get("protected_real_guard_flag") == "1"]
    guard_real_rows = [row for row in guard_rows if row.get("label") == "real"]
    guard_fake_rows = [row for row in guard_rows if row.get("label") == "fake"]
    return {
        "combined": combined,
        "datasets": dataset_metrics,
        "row_count": len(combined_rows),
        "guard_count": len(guard_rows),
        "guard_real_count": len(guard_real_rows),
        "guard_fake_count": len(guard_fake_rows),
        "min_dataset_balanced_accuracy": min(metrics["balanced_accuracy"] for metrics in dataset_metrics.values()),
        "max_protected_real_fpr": max(metrics["protected_real_fpr"] for metrics in dataset_metrics.values()),
    }


def paired_metrics_from_rows(dataset_rows: Dict[str, List[Dict[str, str]]], rules: dict) -> dict:
    dataset_metrics: Dict[str, dict] = {}
    for dataset_name, rows in dataset_rows.items():
        guarded_rows = apply_guard_to_routed_rows(rows, rules)
        dataset_metrics[dataset_name] = accuracy_metrics(guarded_rows)

    fake_accuracies = [metrics["fake_accuracy"] for metrics in dataset_metrics.values()]
    bas = [metrics["balanced_accuracy"] for metrics in dataset_metrics.values()]
    return {
        "datasets": dataset_metrics,
        "mean_fake_accuracy": mean(fake_accuracies),
        "min_fake_accuracy": min(fake_accuracies),
        "mean_balanced_accuracy": mean(bas),
    }


def evaluate_all(
    rules: dict,
    *,
    vcf_dataset_rows: Dict[str, List[Dict[str, str]]],
    canonical_dataset_rows: Dict[str, List[Dict[str, str]]],
    deeper_dataset_rows: Dict[str, List[Dict[str, str]]],
    paired_dataset_rows: Dict[str, List[Dict[str, str]]],
) -> dict:
    vcf = evaluate_group(vcf_dataset_rows, rules)
    canonical = evaluate_group(canonical_dataset_rows, rules)
    deeper = evaluate_group(deeper_dataset_rows, rules)
    paired = paired_metrics_from_rows(paired_dataset_rows, rules)
    return {
        "vcf": vcf,
        "canonical": canonical,
        "deeper": deeper,
        "paired": paired,
    }


def is_feasible(metrics: dict, baseline: dict, args: argparse.Namespace) -> bool:
    return (
        baseline["vcf"]["combined"]["protected_real_fpr"] - metrics["vcf"]["combined"]["protected_real_fpr"]
        >= args.min_vcf_protected_real_fpr_improvement
        and metrics["vcf"]["combined"]["fake_accuracy"]
        >= baseline["vcf"]["combined"]["fake_accuracy"] - args.max_vcf_fake_accuracy_drop
        and metrics["canonical"]["min_dataset_balanced_accuracy"]
        >= baseline["canonical"]["min_dataset_balanced_accuracy"] - args.max_canonical_min_ba_drop
        and metrics["canonical"]["max_protected_real_fpr"]
        <= baseline["canonical"]["max_protected_real_fpr"] + args.max_canonical_prfpr_increase
        and metrics["deeper"]["combined"]["balanced_accuracy"]
        >= baseline["deeper"]["combined"]["balanced_accuracy"] - args.max_deeper_ba_drop
        and metrics["paired"]["mean_fake_accuracy"]
        >= baseline["paired"]["mean_fake_accuracy"] - args.max_paired_mean_fake_drop
        and metrics["paired"]["min_fake_accuracy"]
        >= baseline["paired"]["min_fake_accuracy"] - args.max_paired_min_fake_drop
    )


def guard_summary(rules: dict) -> dict:
    summary = {
        "mode": "reference",
        "allowed_route_labels": "",
        "min_bit_accuracy": "",
        "max_unprotected_pair_fake_prob": "",
        "max_protected_passive_prob": "",
        "max_passive_delta": "",
        "max_passive_conf_gap": "",
    }
    patterns = rules.get("protected_real_guard_patterns") or []
    if not patterns:
        return summary
    pattern = patterns[0]
    summary["mode"] = "vcf_real_guard"
    summary["allowed_route_labels"] = ",".join(pattern.get("allowed_route_labels", []))
    for key in [
        "min_bit_accuracy",
        "max_unprotected_pair_fake_prob",
        "max_protected_passive_prob",
        "max_passive_delta",
        "max_passive_conf_gap",
    ]:
        if key in pattern:
            summary[key] = f"{float(pattern[key]):.6f}"
    return summary


def results_row(config_id: int, rules: dict, metrics: dict, baseline: dict, feasible: bool) -> dict:
    vcf = metrics["vcf"]
    canonical = metrics["canonical"]
    deeper = metrics["deeper"]
    paired = metrics["paired"]
    row = {
        "config_id": str(config_id),
        "feasible": "1" if feasible else "0",
        "vcf_overall_balanced_accuracy": f"{vcf['combined']['balanced_accuracy']:.6f}",
        "vcf_fake_accuracy": f"{vcf['combined']['fake_accuracy']:.6f}",
        "vcf_protected_real_fpr": f"{vcf['combined']['protected_real_fpr']:.6f}",
        "vcf_guard_count": str(vcf["guard_count"]),
        "vcf_guard_real_count": str(vcf["guard_real_count"]),
        "vcf_guard_fake_count": str(vcf["guard_fake_count"]),
        "vcf_prfpr_improvement_vs_baseline": f"{baseline['vcf']['combined']['protected_real_fpr'] - vcf['combined']['protected_real_fpr']:.6f}",
        "vcf_fake_accuracy_delta_vs_baseline": f"{vcf['combined']['fake_accuracy'] - baseline['vcf']['combined']['fake_accuracy']:.6f}",
        "canonical_min_dataset_balanced_accuracy": f"{canonical['min_dataset_balanced_accuracy']:.6f}",
        "canonical_max_protected_real_fpr": f"{canonical['max_protected_real_fpr']:.6f}",
        "canonical_min_ba_delta_vs_baseline": f"{canonical['min_dataset_balanced_accuracy'] - baseline['canonical']['min_dataset_balanced_accuracy']:.6f}",
        "canonical_prfpr_delta_vs_baseline": f"{canonical['max_protected_real_fpr'] - baseline['canonical']['max_protected_real_fpr']:.6f}",
        "deeper_balanced_accuracy": f"{deeper['combined']['balanced_accuracy']:.6f}",
        "deeper_balanced_accuracy_delta_vs_baseline": f"{deeper['combined']['balanced_accuracy'] - baseline['deeper']['combined']['balanced_accuracy']:.6f}",
        "paired_mean_fake_accuracy": f"{paired['mean_fake_accuracy']:.6f}",
        "paired_min_fake_accuracy": f"{paired['min_fake_accuracy']:.6f}",
        "paired_mean_fake_delta_vs_baseline": f"{paired['mean_fake_accuracy'] - baseline['paired']['mean_fake_accuracy']:.6f}",
        "paired_min_fake_delta_vs_baseline": f"{paired['min_fake_accuracy'] - baseline['paired']['min_fake_accuracy']:.6f}",
        **guard_summary(rules),
    }
    for dataset_name, dataset_metrics in paired["datasets"].items():
        row[f"{dataset_name}_fake_accuracy"] = f"{dataset_metrics['fake_accuracy']:.6f}"
    return row


def sort_key(result: tuple[dict, dict, dict, bool]) -> tuple[float, float, float, float, float, float]:
    _, _, metrics, feasible = result
    return (
        1.0 if feasible else 0.0,
        1.0 - metrics["vcf"]["combined"]["protected_real_fpr"],
        metrics["vcf"]["combined"]["balanced_accuracy"],
        metrics["deeper"]["combined"]["balanced_accuracy"],
        metrics["canonical"]["min_dataset_balanced_accuracy"],
        metrics["paired"]["mean_fake_accuracy"],
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
    vcf_paths = [Path(path) for path in args.vcf_bridge_csv]
    canonical_paths = [Path(path) for path in args.canonical_bridge_csv]
    deeper_paths = [Path(path) for path in args.deeper_bridge_csv]
    paired_paths = [Path(path) for path in args.paired_bridge_csv]
    base_rules = yaml.safe_load(Path(args.base_rules_yaml).read_text(encoding="utf-8"))
    vcf_dataset_rows = build_base_routed_group(vcf_paths, base_rules)
    canonical_dataset_rows = build_base_routed_group(canonical_paths, base_rules)
    deeper_dataset_rows = build_base_routed_group(deeper_paths, base_rules)
    paired_dataset_rows = build_base_routed_group(paired_paths, base_rules)

    scored: List[tuple[dict, dict, dict, bool]] = []
    baseline_metrics = None
    for config_id, rules in enumerate(candidate_rules(base_rules), start=1):
        metrics = evaluate_all(
            rules,
            vcf_dataset_rows=vcf_dataset_rows,
            canonical_dataset_rows=canonical_dataset_rows,
            deeper_dataset_rows=deeper_dataset_rows,
            paired_dataset_rows=paired_dataset_rows,
        )
        if baseline_metrics is None:
            baseline_metrics = metrics
        feasible = is_feasible(metrics, baseline_metrics, args)
        row = results_row(config_id, rules, metrics, baseline_metrics, feasible)
        scored.append((row, rules, metrics, feasible))

    scored.sort(key=sort_key, reverse=True)
    write_tsv(Path(args.results_tsv), [row for row, _, _, _ in scored])

    best_row, best_rules, best_metrics, best_feasible = scored[0]
    best_payload = {
        **best_rules,
        "constraints": {
            "min_vcf_protected_real_fpr_improvement": args.min_vcf_protected_real_fpr_improvement,
            "max_vcf_fake_accuracy_drop": args.max_vcf_fake_accuracy_drop,
            "max_canonical_min_ba_drop": args.max_canonical_min_ba_drop,
            "max_canonical_prfpr_increase": args.max_canonical_prfpr_increase,
            "max_deeper_ba_drop": args.max_deeper_ba_drop,
            "max_paired_mean_fake_drop": args.max_paired_mean_fake_drop,
            "max_paired_min_fake_drop": args.max_paired_min_fake_drop,
        },
        "search_summary": {
            "selected_config_id": int(best_row["config_id"]),
            "feasible": best_feasible,
            "vcf_overall_balanced_accuracy": float(best_row["vcf_overall_balanced_accuracy"]),
            "vcf_fake_accuracy": float(best_row["vcf_fake_accuracy"]),
            "vcf_protected_real_fpr": float(best_row["vcf_protected_real_fpr"]),
            "vcf_prfpr_improvement_vs_baseline": float(best_row["vcf_prfpr_improvement_vs_baseline"]),
            "canonical_min_dataset_balanced_accuracy": float(best_row["canonical_min_dataset_balanced_accuracy"]),
            "canonical_max_protected_real_fpr": float(best_row["canonical_max_protected_real_fpr"]),
            "deeper_balanced_accuracy": float(best_row["deeper_balanced_accuracy"]),
            "paired_mean_fake_accuracy": float(best_row["paired_mean_fake_accuracy"]),
            "paired_min_fake_accuracy": float(best_row["paired_min_fake_accuracy"]),
            "vcf_guard_count": int(best_row["vcf_guard_count"]),
            "vcf_guard_real_count": int(best_row["vcf_guard_real_count"]),
            "vcf_guard_fake_count": int(best_row["vcf_guard_fake_count"]),
            "baseline": {
                "vcf_protected_real_fpr": baseline_metrics["vcf"]["combined"]["protected_real_fpr"],
                "vcf_fake_accuracy": baseline_metrics["vcf"]["combined"]["fake_accuracy"],
                "canonical_min_dataset_balanced_accuracy": baseline_metrics["canonical"]["min_dataset_balanced_accuracy"],
                "canonical_max_protected_real_fpr": baseline_metrics["canonical"]["max_protected_real_fpr"],
                "deeper_balanced_accuracy": baseline_metrics["deeper"]["combined"]["balanced_accuracy"],
                "paired_mean_fake_accuracy": baseline_metrics["paired"]["mean_fake_accuracy"],
                "paired_min_fake_accuracy": baseline_metrics["paired"]["min_fake_accuracy"],
            },
        },
    }
    write_yaml(Path(args.best_rules_yaml), best_payload)

    print(yaml.safe_dump(best_payload["search_summary"], sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    main()
