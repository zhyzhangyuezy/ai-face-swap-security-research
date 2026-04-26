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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search routing thresholds across one or more hybrid bridge CSVs.")
    parser.add_argument("--bridge-csv", action="append", required=True, help="Hybrid bridge CSV. Repeat for multiple datasets.")
    parser.add_argument("--results-tsv", required=True, help="Output TSV path for the searched configurations.")
    parser.add_argument("--best-rules-yaml", required=True, help="Output YAML path for the selected best rules.")
    return parser.parse_args()


def candidate_rules() -> Iterable[dict]:
    passive_thresholds = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
    anomaly_thresholds = [0.004, 0.005, 0.008, 0.010, 0.020]
    trust_bits = [0.975, 0.980]
    trust_deltas = [0.005, 0.010, 0.020, 0.050]
    suspicious_bits = [0.940, 0.950]
    suspicious_deltas = [0.010, 0.020, 0.050]

    for passive_thr in passive_thresholds:
        for anomaly_thr in anomaly_thresholds:
            for trust_bit in trust_bits:
                for trust_delta in trust_deltas:
                    for suspicious_bit in suspicious_bits:
                        for suspicious_delta in suspicious_deltas:
                            if suspicious_bit > trust_bit:
                                continue
                            if suspicious_delta < trust_delta:
                                continue
                            yield {
                                "route_labels": {
                                    "trust": {
                                        "min_bit_accuracy": trust_bit,
                                        "max_passive_delta_abs": trust_delta,
                                    },
                                    "suspicious": {
                                        "min_bit_accuracy": suspicious_bit,
                                        "max_passive_delta_abs": suspicious_delta,
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
                                    "Autosearched against the available hybrid bridge tables.",
                                    "Primary objective is balanced accuracy, secondary objective is trust coverage.",
                                ],
                            }


def dataset_name_from_path(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_hybrid_bridge"):
        stem = stem[: -len("_hybrid_bridge")]
    if stem.endswith("_merged"):
        stem = stem[: -len("_merged")]
    return stem


def validate_bridge_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        raise SystemExit(f"Bridge CSV is empty: {path}")

    required_columns = {
        "sample_id",
        "pair_group_id",
        "label",
        "protection",
        "passive_status",
        "passive_prob_fake",
        "prov_status",
        "prov_bit_accuracy",
    }
    missing = [column for column in sorted(required_columns) if column not in rows[0]]
    if missing:
        raise SystemExit(f"Bridge CSV is missing result columns {missing}: {path}")

    passive_ok = sum(1 for row in rows if row.get("passive_status") == "ok")
    protected_rows = [row for row in rows if row.get("protection") == "protected"]
    prov_ok = sum(1 for row in protected_rows if row.get("prov_status") == "ok")
    if passive_ok == 0:
        raise SystemExit(f"Bridge CSV has no passive results; did you pass a job sheet instead of a result table? {path}")
    if protected_rows and prov_ok == 0:
        raise SystemExit(f"Bridge CSV has no protected provenance results; did you pass a pre-execution merged/job table? {path}")


def accuracy_metrics(rows: List[Dict[str, str]]) -> dict:
    acc_rows = [row for row in rows if row.get("effective_correct") in {"0", "1"}]
    if not acc_rows:
        return {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "real_accuracy": 0.0,
            "fake_accuracy": 0.0,
            "trust_rate": 0.0,
            "abstain_rate": 0.0,
        }

    total = len(acc_rows)
    accuracy = sum(int(row["effective_correct"]) for row in acc_rows) / total

    def class_acc(label: str) -> float:
        subset = [row for row in acc_rows if row.get("label") == label]
        if not subset:
            return 0.0
        return sum(int(row["effective_correct"]) for row in subset) / len(subset)

    real_acc = class_acc("real")
    fake_acc = class_acc("fake")
    balanced = 0.5 * (real_acc + fake_acc)
    trust_rate = sum(1 for row in rows if row.get("route_label") == "trust") / len(rows)
    abstain_rate = sum(1 for row in rows if row.get("route_label") == "abstain") / len(rows)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "real_accuracy": real_acc,
        "fake_accuracy": fake_acc,
        "trust_rate": trust_rate,
        "abstain_rate": abstain_rate,
    }


def evaluate_rules(bridge_paths: List[Path], rules: dict) -> dict:
    dataset_metrics: Dict[str, dict] = {}
    combined_rows: List[Dict[str, str]] = []

    for bridge_path in bridge_paths:
        bridge_rows = load_csv(bridge_path)
        validate_bridge_rows(bridge_path, bridge_rows)
        unprotected_map = compute_unprotected_map(bridge_rows)
        routed_rows = [route_row(row, unprotected_map, rules) for row in bridge_rows]
        metrics = accuracy_metrics(routed_rows)
        dataset_metrics[dataset_name_from_path(bridge_path)] = metrics
        combined_rows.extend(routed_rows)

    combined = accuracy_metrics(combined_rows)
    min_dataset_balanced = min(metrics["balanced_accuracy"] for metrics in dataset_metrics.values())

    return {
        "combined": combined,
        "datasets": dataset_metrics,
        "min_dataset_balanced_accuracy": min_dataset_balanced,
    }


def results_row(config_id: int, rules: dict, metrics: dict) -> dict:
    row = {
        "config_id": str(config_id),
        "passive_fake_threshold": f"{rules['fallback']['passive_fake_threshold']:.6f}",
        "protected_delta_abs_anomaly_threshold": f"{rules['fallback']['protected_delta_abs_anomaly_threshold']:.6f}",
        "trust_min_bit_accuracy": f"{rules['route_labels']['trust']['min_bit_accuracy']:.6f}",
        "trust_max_passive_delta_abs": f"{rules['route_labels']['trust']['max_passive_delta_abs']:.6f}",
        "suspicious_min_bit_accuracy": f"{rules['route_labels']['suspicious']['min_bit_accuracy']:.6f}",
        "suspicious_max_passive_delta_abs": f"{rules['route_labels']['suspicious']['max_passive_delta_abs']:.6f}",
        "overall_accuracy": f"{metrics['combined']['accuracy']:.6f}",
        "overall_balanced_accuracy": f"{metrics['combined']['balanced_accuracy']:.6f}",
        "overall_real_accuracy": f"{metrics['combined']['real_accuracy']:.6f}",
        "overall_fake_accuracy": f"{metrics['combined']['fake_accuracy']:.6f}",
        "overall_trust_rate": f"{metrics['combined']['trust_rate']:.6f}",
        "overall_abstain_rate": f"{metrics['combined']['abstain_rate']:.6f}",
        "min_dataset_balanced_accuracy": f"{metrics['min_dataset_balanced_accuracy']:.6f}",
    }

    for dataset_name, dataset_metrics in metrics["datasets"].items():
        prefix = dataset_name
        row[f"{prefix}_balanced_accuracy"] = f"{dataset_metrics['balanced_accuracy']:.6f}"
        row[f"{prefix}_accuracy"] = f"{dataset_metrics['accuracy']:.6f}"
        row[f"{prefix}_fake_accuracy"] = f"{dataset_metrics['fake_accuracy']:.6f}"
        row[f"{prefix}_real_accuracy"] = f"{dataset_metrics['real_accuracy']:.6f}"
        row[f"{prefix}_trust_rate"] = f"{dataset_metrics['trust_rate']:.6f}"
    return row


def write_tsv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def sort_key(result: tuple[dict, dict, dict]) -> tuple[float, float, float, float, float]:
    _, rules, metrics = result
    combined = metrics["combined"]
    return (
        combined["balanced_accuracy"],
        metrics["min_dataset_balanced_accuracy"],
        combined["accuracy"],
        combined["trust_rate"],
        -combined["abstain_rate"],
    )


def main() -> None:
    args = parse_args()
    bridge_paths = [Path(path) for path in args.bridge_csv]

    scored: List[tuple[dict, dict, dict]] = []
    for config_id, rules in enumerate(candidate_rules(), start=1):
        metrics = evaluate_rules(bridge_paths, rules)
        row = results_row(config_id, rules, metrics)
        scored.append((row, rules, metrics))

    scored.sort(key=sort_key, reverse=True)
    write_tsv(Path(args.results_tsv), [row for row, _, _ in scored])

    best_row, best_rules, best_metrics = scored[0]
    best_payload = {
        **best_rules,
        "search_summary": {
            "selected_config_id": int(best_row["config_id"]),
            "overall_accuracy": float(best_row["overall_accuracy"]),
            "overall_balanced_accuracy": float(best_row["overall_balanced_accuracy"]),
            "min_dataset_balanced_accuracy": float(best_row["min_dataset_balanced_accuracy"]),
            "datasets": best_metrics["datasets"],
        },
    }
    write_yaml(Path(args.best_rules_yaml), best_payload)
    print(json.dumps(best_payload["search_summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
