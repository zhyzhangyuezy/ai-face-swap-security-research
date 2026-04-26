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
from sklearn.tree import DecisionTreeClassifier, export_text


def _bootstrap_scripts() -> None:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))


_bootstrap_scripts()

from run_routing_sanity import compute_unprotected_map, load_csv, route_row, safe_float
from search_routing_thresholds import dataset_name_from_path, validate_bridge_rows
from search_routing_thresholds_constrained import accuracy_metrics, is_feasible


FEATURE_NAMES = [
    "route_is_trust",
    "route_is_abstain",
    "passive_baseline_prob",
    "passive_prob_fake",
    "passive_delta",
    "passive_delta_abs",
    "passive_conf_gap",
    "prov_bit_accuracy",
    "prov_detect_prob",
    "prov_watermark_l1_mean",
    "passive_feature_norm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search a shallow learned-but-explainable protected-transfer policy on the current borderline band "
            "using existing pair features."
        )
    )
    parser.add_argument("--base-rules-yaml", required=True)
    parser.add_argument("--canonical-bridge-csv", action="append", required=True)
    parser.add_argument("--paired-bridge-csv", action="append", required=True)
    parser.add_argument("--results-tsv", required=True)
    parser.add_argument("--best-policy-json", required=True)
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


def build_feature_vector(row: dict) -> list[float]:
    return [
        1.0 if row.get("route_label") == "trust" else 0.0,
        1.0 if row.get("route_label") == "abstain" else 0.0,
        float(row.get("passive_baseline_prob") or 0.0),
        float(row.get("passive_prob_fake") or 0.0),
        float(row.get("passive_delta") or 0.0),
        float(row.get("passive_delta_abs") or 0.0),
        float(row.get("passive_conf_gap") or 0.0),
        float(row.get("prov_bit_accuracy") or 0.0),
        float(row.get("prov_detect_prob") or 0.0),
        float(row.get("prov_watermark_l1_mean") or 0.0),
        float(row.get("passive_feature_norm") or 0.0),
    ]


def candidate_band_row(row: dict) -> bool:
    prob = safe_float(row.get("passive_prob_fake"))
    if prob is None:
        return False
    return (
        row.get("protection") == "protected"
        and row.get("route_label") in {"trust", "abstain"}
        and prob >= 0.45
        and prob < 0.50
        and row.get("passive_status") == "ok"
        and row.get("prov_status") == "ok"
    )


def candidate_models() -> Iterable[dict]:
    yield {"mode": "reference"}
    for max_depth in [1, 2, 3]:
        for min_samples_leaf in [1, 2]:
            for criterion in ["gini", "entropy"]:
                for paired_positive_weight in [1.0, 2.0, 3.0]:
                    for negative_weight in [1.0, 1.5, 2.0]:
                        yield {
                            "mode": "tree",
                            "max_depth": max_depth,
                            "min_samples_leaf": min_samples_leaf,
                            "criterion": criterion,
                            "paired_positive_weight": paired_positive_weight,
                            "negative_weight": negative_weight,
                        }


def fit_tree(
    train_rows: list[dict],
    *,
    max_depth: int,
    min_samples_leaf: int,
    criterion: str,
    paired_positive_weight: float,
    negative_weight: float,
) -> tuple[DecisionTreeClassifier, dict]:
    xs: list[list[float]] = []
    ys: list[int] = []
    weights: list[float] = []

    for row in train_rows:
        xs.append(build_feature_vector(row))
        label = 1 if row.get("label") == "fake" else 0
        ys.append(label)
        dataset_role = row.get("dataset_role")
        weight = 1.0
        if label == 1 and dataset_role == "paired":
            weight = paired_positive_weight
        elif label == 0:
            weight = negative_weight
        weights.append(weight)

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        random_state=0,
    )
    clf.fit(xs, ys, sample_weight=weights)
    info = {
        "tree_text": export_text(clf, feature_names=FEATURE_NAMES),
        "train_rows": len(train_rows),
        "train_positive": int(sum(ys)),
        "train_negative": int(len(ys) - sum(ys)),
    }
    return clf, info


def route_row_with_tree(base_row: dict, clf: DecisionTreeClassifier | None, reference_rules: dict) -> dict:
    if clf is None:
        return route_row(base_row, {}, reference_rules)  # not used
    raise RuntimeError("unused")


def apply_policy_to_rows(
    source_rows: list[dict],
    unprotected_map: dict[str, float],
    *,
    base_rules: dict,
    reference_rules: dict,
    clf: DecisionTreeClassifier | None,
    model_cfg: dict,
) -> list[dict]:
    if model_cfg["mode"] == "reference":
        return [route_row(row, unprotected_map, reference_rules) for row in source_rows]

    locked_rows = [route_row(row, unprotected_map, base_rules) for row in source_rows]
    output_rows: list[dict] = []
    for row in locked_rows:
        transfer_override = False
        if candidate_band_row(row):
            pred = int(clf.predict([build_feature_vector(row)])[0])
            transfer_override = pred == 1

        if transfer_override:
            updated = dict(row)
            updated["protected_transfer_override_flag"] = "1"
            updated["protected_transfer_override_reason"] = "learned_tree_policy"
            updated["protected_transfer_pattern"] = "tree_policy"
            updated["passive_fake_threshold"] = "0.450000"
            passive_prob = safe_float(row.get("passive_prob_fake"))
            passive_pred = "fake" if passive_prob is not None and passive_prob >= 0.45 else "real"
            updated["passive_pred_label"] = passive_pred
            if row.get("route_label") == "trust":
                final_decision = "trust_and_flag_fake" if passive_pred == "fake" else "trust_and_accept_real"
            else:
                final_decision = "fallback_to_passive_fake" if passive_pred == "fake" else "fallback_to_passive_real"
            updated["final_decision"] = final_decision
            updated["effective_pred_label"] = passive_pred
            if row.get("label") in {"real", "fake"}:
                updated["effective_correct"] = "1" if row.get("label") == passive_pred else "0"
            updated["passive_threshold_source"] = "learned_tree_policy"
            output_rows.append(updated)
        else:
            output_rows.append(row)
    return output_rows


def evaluate_rules(bridge_paths: List[Path], base_rules: dict, reference_rules: dict, clf: DecisionTreeClassifier | None, model_cfg: dict) -> dict:
    dataset_metrics: Dict[str, dict] = {}
    combined_rows: List[Dict[str, str]] = []
    for bridge_path in bridge_paths:
        bridge_rows = load_csv(bridge_path)
        validate_bridge_rows(bridge_path, bridge_rows)
        unprotected_map = compute_unprotected_map(bridge_rows)
        routed_rows = apply_policy_to_rows(
            bridge_rows,
            unprotected_map,
            base_rules=base_rules,
            reference_rules=reference_rules,
            clf=clf,
            model_cfg=model_cfg,
        )
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


def paired_metrics(bridge_paths: List[Path], base_rules: dict, reference_rules: dict, clf: DecisionTreeClassifier | None, model_cfg: dict) -> dict:
    dataset_metrics: Dict[str, dict] = {}
    for bridge_path in bridge_paths:
        bridge_rows = load_csv(bridge_path)
        validate_bridge_rows(bridge_path, bridge_rows)
        unprotected_map = compute_unprotected_map(bridge_rows)
        routed_rows = apply_policy_to_rows(
            bridge_rows,
            unprotected_map,
            base_rules=base_rules,
            reference_rules=reference_rules,
            clf=clf,
            model_cfg=model_cfg,
        )
        dataset_metrics[dataset_name_from_path(bridge_path)] = accuracy_metrics(routed_rows)

    fake_accuracies = [metrics["fake_accuracy"] for metrics in dataset_metrics.values()]
    bas = [metrics["balanced_accuracy"] for metrics in dataset_metrics.values()]
    return {
        "datasets": dataset_metrics,
        "mean_fake_accuracy": mean(fake_accuracies),
        "min_fake_accuracy": min(fake_accuracies),
        "mean_balanced_accuracy": mean(bas),
    }


def collect_train_rows(canonical_paths: List[Path], paired_paths: List[Path], base_rules: dict) -> list[dict]:
    rows_out: list[dict] = []
    for dataset_role, paths in [("canonical", canonical_paths), ("paired", paired_paths)]:
        for bridge_path in paths:
            bridge_rows = load_csv(bridge_path)
            validate_bridge_rows(bridge_path, bridge_rows)
            unprotected_map = compute_unprotected_map(bridge_rows)
            locked_rows = [route_row(row, unprotected_map, base_rules) for row in bridge_rows]
            for row in locked_rows:
                if candidate_band_row(row):
                    row_copy = dict(row)
                    row_copy["dataset_role"] = dataset_role
                    row_copy["dataset_name"] = dataset_name_from_path(bridge_path)
                    rows_out.append(row_copy)
    return rows_out


def model_summary(model_cfg: dict, info: dict | None) -> dict:
    if model_cfg["mode"] == "reference":
        return {"mode": "reference"}
    return {
        "mode": "tree",
        "max_depth": model_cfg["max_depth"],
        "min_samples_leaf": model_cfg["min_samples_leaf"],
        "criterion": model_cfg["criterion"],
        "paired_positive_weight": model_cfg["paired_positive_weight"],
        "negative_weight": model_cfg["negative_weight"],
        "tree_text": info["tree_text"] if info else "",
        "train_rows": info["train_rows"] if info else 0,
        "train_positive": info["train_positive"] if info else 0,
        "train_negative": info["train_negative"] if info else 0,
    }


def results_row(config_id: int, model_cfg: dict, info: dict | None, canonical_metrics: dict, paired_eval: dict, feasible: bool) -> dict:
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
        **{k: str(v) for k, v in model_summary(model_cfg, info).items() if k != "tree_text"},
    }
    for dataset_name, dataset_metrics in paired_eval["datasets"].items():
        row[f"{dataset_name}_fake_accuracy"] = f"{dataset_metrics['fake_accuracy']:.6f}"
        row[f"{dataset_name}_balanced_accuracy"] = f"{dataset_metrics['balanced_accuracy']:.6f}"
        row[f"{dataset_name}_protected_real_fpr"] = f"{dataset_metrics['protected_real_fpr']:.6f}"
    return row


def sort_key(result: tuple[dict, dict, dict | None, dict, dict, bool]) -> tuple[float, float, float, float, float, float]:
    _, _, _, canonical_metrics, paired_eval, feasible = result
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
    reference_rules = yaml.safe_load(Path(args.base_rules_yaml).read_text(encoding="utf-8"))
    base_rules = stripped_base_rules(reference_rules)
    train_rows = collect_train_rows(canonical_paths, paired_paths, base_rules)

    scored: List[tuple[dict, dict, dict | None, dict, dict, bool]] = []
    for config_id, model_cfg in enumerate(candidate_models(), start=1):
        clf = None
        info = None
        if model_cfg["mode"] == "tree":
            clf, info = fit_tree(
                train_rows,
                max_depth=model_cfg["max_depth"],
                min_samples_leaf=model_cfg["min_samples_leaf"],
                criterion=model_cfg["criterion"],
                paired_positive_weight=model_cfg["paired_positive_weight"],
                negative_weight=model_cfg["negative_weight"],
            )
        canonical_metrics = evaluate_rules(canonical_paths, base_rules, reference_rules, clf, model_cfg)
        feasible = is_feasible(canonical_metrics, args)
        paired_eval = paired_metrics(paired_paths, base_rules, reference_rules, clf, model_cfg)
        row = results_row(config_id, model_cfg, info, canonical_metrics, paired_eval, feasible)
        scored.append((row, model_cfg, info, canonical_metrics, paired_eval, feasible))

    scored.sort(key=sort_key, reverse=True)
    write_tsv(Path(args.results_tsv), [row for row, _, _, _, _, _ in scored])

    best_row, best_model_cfg, best_info, best_canonical_metrics, best_paired_eval, best_feasible = scored[0]
    payload = {
        "selection": model_summary(best_model_cfg, best_info),
        "constraints": {
            "max_protected_real_fpr": args.max_protected_real_fpr,
            "min_protected_fake_recall": args.min_protected_fake_recall,
            "min_unprotected_fake_recall": args.min_unprotected_fake_recall,
            "min_dataset_balanced_accuracy": args.min_dataset_balanced_accuracy,
        },
        "search_summary": {
            "selected_config_id": int(best_row["config_id"]),
            "feasible": best_feasible,
            "canonical_overall_balanced_accuracy": float(best_row["canonical_overall_balanced_accuracy"]),
            "canonical_min_dataset_balanced_accuracy": float(best_row["canonical_min_dataset_balanced_accuracy"]),
            "canonical_max_protected_real_fpr": float(best_row["canonical_max_protected_real_fpr"]),
            "canonical_min_protected_fake_recall": float(best_row["canonical_min_protected_fake_recall"]),
            "canonical_min_unprotected_fake_recall": float(best_row["canonical_min_unprotected_fake_recall"]),
            "paired_mean_fake_accuracy": float(best_row["paired_mean_fake_accuracy"]),
            "paired_min_fake_accuracy": float(best_row["paired_min_fake_accuracy"]),
            "paired_mean_balanced_accuracy": float(best_row["paired_mean_balanced_accuracy"]),
            "canonical_datasets": best_canonical_metrics["datasets"],
            "paired_datasets": best_paired_eval["datasets"],
            "train_row_count": len(train_rows),
        },
    }
    Path(args.best_policy_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.best_policy_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
