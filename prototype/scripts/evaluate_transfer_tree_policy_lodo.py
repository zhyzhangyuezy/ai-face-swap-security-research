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

from search_routing_thresholds import dataset_name_from_path
from search_routing_thresholds_constrained import is_feasible
from search_transfer_tree_policy import (
    candidate_models,
    collect_train_rows,
    evaluate_rules,
    fit_tree,
    model_summary,
    paired_metrics,
    sort_key,
    stripped_base_rules,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Leave-one-dataset-out validation for the shallow learned transfer-policy family, "
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
    return parser.parse_args()


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
    reference_rules = yaml.safe_load(Path(args.base_rules_yaml).read_text(encoding="utf-8"))
    base_rules = stripped_base_rules(reference_rules)

    output_rows: list[dict] = []
    summary = {"holdouts": {}, "selection": vars(args)}

    for holdout in canonical_paths:
        train_paths = [path for path in canonical_paths if path != holdout]
        train_rows = collect_train_rows(train_paths, paired_paths, base_rules)
        scored: list[tuple[dict, dict, dict | None, dict, dict, bool]] = []

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
            train_canonical = evaluate_rules(train_paths, base_rules, reference_rules, clf, model_cfg)
            feasible = is_feasible(train_canonical, args)
            pair_eval = paired_metrics(paired_paths, base_rules, reference_rules, clf, model_cfg)
            row = {
                "config_id": str(config_id),
                "train_min_dataset_balanced_accuracy": f"{train_canonical['min_dataset_balanced_accuracy']:.6f}",
                "train_max_protected_real_fpr": f"{train_canonical['max_protected_real_fpr']:.6f}",
                "paired_mean_fake_accuracy": f"{pair_eval['mean_fake_accuracy']:.6f}",
            }
            scored.append((row, model_cfg, info, train_canonical, pair_eval, feasible))

        scored.sort(key=sort_key, reverse=True)
        best_row, best_model_cfg, best_info, best_train_canonical, best_pair_eval, best_feasible = scored[0]
        holdout_eval = evaluate_rules([holdout], base_rules, reference_rules, clf if False else None, {"mode": "reference"})
        # Re-evaluate the selected model on the holdout using its actual classifier.
        selected_clf = None
        if best_model_cfg["mode"] == "tree":
            selected_clf, _ = fit_tree(
                train_rows,
                max_depth=best_model_cfg["max_depth"],
                min_samples_leaf=best_model_cfg["min_samples_leaf"],
                criterion=best_model_cfg["criterion"],
                paired_positive_weight=best_model_cfg["paired_positive_weight"],
                negative_weight=best_model_cfg["negative_weight"],
            )
        holdout_eval = evaluate_rules([holdout], base_rules, reference_rules, selected_clf, best_model_cfg)
        holdout_name = dataset_name_from_path(holdout)
        holdout_metrics = holdout_eval["datasets"][holdout_name]

        output_row = {
            "holdout_dataset": holdout_name,
            "selected_config_id": str(best_row["config_id"]),
            "selected_mode": best_model_cfg["mode"],
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
            "selection": model_summary(best_model_cfg, best_info),
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
