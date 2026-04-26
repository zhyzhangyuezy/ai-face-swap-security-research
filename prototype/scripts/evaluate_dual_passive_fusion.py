from __future__ import annotations

import argparse
import csv
import json
import math
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
from search_routing_thresholds_constrained import accuracy_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate simple dual-passive fusion strategies over EfficientNet-B0 and UCF bridge tables."
    )
    parser.add_argument(
        "--bridge-pair",
        action="append",
        required=True,
        help="DATASET=EFFB0_BRIDGE_CSV,UCF_BRIDGE_CSV. Repeat for each dataset.",
    )
    parser.add_argument("--results-tsv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--best-rules-yaml", required=True)
    parser.add_argument("--materialize-dir", default="")
    parser.add_argument("--max-protected-real-fpr", type=float, default=0.40)
    parser.add_argument("--min-protected-fake-recall", type=float, default=0.50)
    parser.add_argument("--min-unprotected-fake-recall", type=float, default=0.375)
    return parser.parse_args()


def parse_pair(value: str) -> tuple[str, Path, Path]:
    if "=" not in value or "," not in value:
        raise ValueError(f"Expected DATASET=EFFB0.csv,UCF.csv, got: {value}")
    dataset, paths = value.split("=", 1)
    effb0_path, ucf_path = paths.split(",", 1)
    return dataset.strip(), Path(effb0_path.strip()), Path(ucf_path.strip())


def safe_float(value: str, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def prob_to_margin(prob: float) -> float:
    eps = 1e-6
    prob = min(max(prob, eps), 1.0 - eps)
    return math.log(prob / (1.0 - prob))


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
        0.300,
        0.400,
        0.500,
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
    ]

    for passive_thr in passive_thresholds:
        for anomaly_thr in anomaly_thresholds:
            yield {
                "route_labels": {
                    "trust": {"min_bit_accuracy": 0.975, "max_passive_delta_abs": 0.05},
                    "suspicious": {"min_bit_accuracy": 0.94, "max_passive_delta_abs": 0.05},
                    "abstain": {"require_protected": True},
                },
                "fallback": {
                    "passive_fake_threshold": passive_thr,
                    "protected_passive_fake_threshold": passive_thr,
                    "protected_delta_abs_anomaly_threshold": anomaly_thr,
                    "min_passive_confidence_gap": 0.45,
                },
            }


def load_pair(dataset: str, effb0_path: Path, ucf_path: Path) -> list[dict]:
    eff_rows = load_csv(effb0_path)
    ucf_rows = load_csv(ucf_path)
    validate_bridge_rows(effb0_path, eff_rows)
    validate_bridge_rows(ucf_path, ucf_rows)
    ucf_by_id = {row["sample_id"]: row for row in ucf_rows}

    paired: list[dict] = []
    for eff in eff_rows:
        sample_id = eff["sample_id"]
        if sample_id not in ucf_by_id:
            raise ValueError(f"{dataset}: sample_id missing from UCF bridge: {sample_id}")
        ucf = ucf_by_id[sample_id]
        if eff.get("label") != ucf.get("label") or eff.get("protection") != ucf.get("protection"):
            raise ValueError(f"{dataset}: bridge row mismatch for sample_id={sample_id}")

        eff_prob = safe_float(eff.get("passive_prob_fake", ""))
        ucf_prob = safe_float(ucf.get("passive_prob_fake", ""))
        row = dict(eff)
        row["dataset_name"] = dataset
        row["effb0_prob_fake"] = f"{eff_prob:.6f}"
        row["ucf_prob_fake"] = f"{ucf_prob:.6f}"
        row["effb0_logit_real"] = eff.get("passive_logit_real", "")
        row["effb0_logit_fake"] = eff.get("passive_logit_fake", "")
        row["ucf_logit_real"] = ucf.get("passive_logit_real", "")
        row["ucf_logit_fake"] = ucf.get("passive_logit_fake", "")
        paired.append(row)
    return paired


def fused_prob(strategy: str, eff_prob: float, ucf_prob: float) -> tuple[float, str]:
    if strategy == "effb0":
        return eff_prob, "effb0"
    if strategy == "ucf":
        return ucf_prob, "ucf"
    if strategy == "mean":
        return 0.5 * eff_prob + 0.5 * ucf_prob, "mean"
    if strategy == "max":
        return max(eff_prob, ucf_prob), "effb0" if eff_prob >= ucf_prob else "ucf"
    if strategy == "min":
        return min(eff_prob, ucf_prob), "effb0" if eff_prob <= ucf_prob else "ucf"
    if strategy == "confidence_select":
        eff_conf = abs(eff_prob - 0.5)
        ucf_conf = abs(ucf_prob - 0.5)
        return (eff_prob, "effb0") if eff_conf >= ucf_conf else (ucf_prob, "ucf")
    if strategy.startswith("weighted_ucf_"):
        weight = float(strategy.rsplit("_", 1)[1])
        return (1.0 - weight) * eff_prob + weight * ucf_prob, f"weighted_ucf_{weight:.2f}"
    raise ValueError(f"Unsupported fusion strategy: {strategy}")


def materialize_strategy_rows(rows: list[dict], strategy: str) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        eff_prob = safe_float(row.get("effb0_prob_fake", ""))
        ucf_prob = safe_float(row.get("ucf_prob_fake", ""))
        prob, source = fused_prob(strategy, eff_prob, ucf_prob)
        margin = prob_to_margin(prob)
        fused = dict(row)
        fused["passive_prob_fake"] = f"{prob:.6f}"
        fused["passive_pred_label"] = "1" if prob >= 0.5 else "0"
        fused["passive_logit_real"] = "0.000000"
        fused["passive_logit_fake"] = f"{margin:.6f}"
        fused["passive_fusion_strategy"] = strategy
        fused["passive_fusion_source"] = source
        out.append(fused)
    return out


def evaluate_strategy(dataset_rows: dict[str, list[dict]], strategy: str, rules: dict) -> dict:
    dataset_metrics: Dict[str, dict] = {}
    combined_rows: list[dict] = []
    for dataset, rows in dataset_rows.items():
        fused_rows = materialize_strategy_rows(rows, strategy)
        unprotected_map = compute_unprotected_map(fused_rows)
        routed_rows = [route_row(row, unprotected_map, rules) for row in fused_rows]
        dataset_metrics[dataset] = accuracy_metrics(routed_rows)
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


def feasible(metrics: dict, args: argparse.Namespace) -> bool:
    return (
        metrics["max_protected_real_fpr"] <= args.max_protected_real_fpr
        and metrics["min_protected_fake_recall"] >= args.min_protected_fake_recall
        and metrics["min_unprotected_fake_recall"] >= args.min_unprotected_fake_recall
    )


def sort_key(item: tuple[dict, dict, dict, bool]) -> tuple[float, float, float, float, float]:
    row, _, metrics, is_feasible = item
    return (
        1.0 if is_feasible else 0.0,
        metrics["min_dataset_balanced_accuracy"],
        metrics["combined"]["balanced_accuracy"],
        -metrics["max_protected_real_fpr"],
        -float(row["passive_fake_threshold"]),
    )


def result_row(strategy: str, config_id: int, rules: dict, metrics: dict, is_feasible: bool) -> dict:
    row = {
        "strategy": strategy,
        "config_id": str(config_id),
        "feasible": "1" if is_feasible else "0",
        "passive_fake_threshold": f"{rules['fallback']['passive_fake_threshold']:.6f}",
        "protected_delta_abs_anomaly_threshold": f"{rules['fallback']['protected_delta_abs_anomaly_threshold']:.6f}",
        "overall_balanced_accuracy": f"{metrics['combined']['balanced_accuracy']:.6f}",
        "overall_accuracy": f"{metrics['combined']['accuracy']:.6f}",
        "min_dataset_balanced_accuracy": f"{metrics['min_dataset_balanced_accuracy']:.6f}",
        "max_protected_real_fpr": f"{metrics['max_protected_real_fpr']:.6f}",
        "min_protected_fake_recall": f"{metrics['min_protected_fake_recall']:.6f}",
        "min_unprotected_fake_recall": f"{metrics['min_unprotected_fake_recall']:.6f}",
    }
    for dataset, item in metrics["datasets"].items():
        row[f"{dataset}_balanced_accuracy"] = f"{item['balanced_accuracy']:.6f}"
        row[f"{dataset}_real_accuracy"] = f"{item['real_accuracy']:.6f}"
        row[f"{dataset}_fake_accuracy"] = f"{item['fake_accuracy']:.6f}"
        row[f"{dataset}_protected_real_fpr"] = f"{item['protected_real_fpr']:.6f}"
        row[f"{dataset}_protected_fake_recall"] = f"{item['protected_fake_recall']:.6f}"
        row[f"{dataset}_unprotected_fake_recall"] = f"{item['unprotected_fake_recall']:.6f}"
    return row


def write_csv(path: Path, rows: list[dict], delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def materialize_best(materialize_dir: Path, dataset_rows: dict[str, list[dict]], strategy: str, rules: dict) -> dict:
    materialize_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict] = {}
    for dataset, rows in dataset_rows.items():
        fused_rows = materialize_strategy_rows(rows, strategy)
        routed_rows = [route_row(row, compute_unprotected_map(fused_rows), rules) for row in fused_rows]
        bridge_path = materialize_dir / f"{dataset}_merged_dual_{strategy}.csv"
        routing_path = materialize_dir / f"{dataset}_routing_dual_{strategy}.csv"
        write_csv(bridge_path, fused_rows)
        write_csv(routing_path, routed_rows)
        outputs[dataset] = {"bridge": str(bridge_path), "routing": str(routing_path)}
    return outputs


def main() -> None:
    args = parse_args()
    dataset_rows: dict[str, list[dict]] = {}
    for value in args.bridge_pair:
        dataset, effb0_path, ucf_path = parse_pair(value)
        dataset_rows[dataset] = load_pair(dataset, effb0_path, ucf_path)

    strategies = [
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

    scored: list[tuple[dict, dict, dict, bool]] = []
    for strategy in strategies:
        for config_id, rules in enumerate(candidate_rules(), start=1):
            metrics = evaluate_strategy(dataset_rows, strategy, rules)
            is_feasible = feasible(metrics, args)
            scored.append((result_row(strategy, config_id, rules, metrics, is_feasible), rules, metrics, is_feasible))

    scored.sort(key=sort_key, reverse=True)
    write_csv(Path(args.results_tsv), [row for row, _, _, _ in scored], delimiter="\t")

    best_row, best_rules, best_metrics, best_feasible = scored[0]
    summary = {
        "best": {
            "strategy": best_row["strategy"],
            "config_id": int(best_row["config_id"]),
            "feasible": best_feasible,
            "rules": best_rules,
            "metrics": best_metrics,
        },
        "top_rows": [row for row, _, _, _ in scored[:10]],
    }
    if args.materialize_dir:
        summary["materialized"] = materialize_best(Path(args.materialize_dir), dataset_rows, best_row["strategy"], best_rules)
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    best_payload = {
        **best_rules,
        "fusion": {
            "strategy": best_row["strategy"],
            "source_branches": ["manifest_multisource_ffpp_uadfv_celebdfv1_effb0_v1", "DeepfakeBench-UCF"],
        },
        "constraints": {
            "max_protected_real_fpr": args.max_protected_real_fpr,
            "min_protected_fake_recall": args.min_protected_fake_recall,
            "min_unprotected_fake_recall": args.min_unprotected_fake_recall,
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
