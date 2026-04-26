from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml


def _bootstrap_scripts() -> None:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))


_bootstrap_scripts()

from search_routing_thresholds import dataset_name_from_path
from search_transfer_tree_policy import (
    collect_train_rows,
    evaluate_rules,
    fit_tree,
    model_summary,
    paired_metrics,
    stripped_base_rules,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Leave-one-dataset-out validation for one fixed shallow transfer-tree configuration, "
            "so simpler paper-friendly candidates can be checked without re-running the whole search."
        )
    )
    parser.add_argument("--base-rules-yaml", required=True)
    parser.add_argument("--canonical-bridge-csv", action="append", required=True)
    parser.add_argument("--paired-bridge-csv", action="append", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--max-depth", type=int, required=True)
    parser.add_argument("--min-samples-leaf", type=int, required=True)
    parser.add_argument("--criterion", choices=["gini", "entropy"], required=True)
    parser.add_argument("--paired-positive-weight", type=float, required=True)
    parser.add_argument("--negative-weight", type=float, required=True)
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

    cfg = {
        "mode": "tree",
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "criterion": args.criterion,
        "paired_positive_weight": args.paired_positive_weight,
        "negative_weight": args.negative_weight,
    }

    output_rows: list[dict] = []
    summary = {
        "tag": args.tag,
        "config": cfg,
        "selection": vars(args),
        "holdouts": {},
    }

    all_train_rows = collect_train_rows(canonical_paths, paired_paths, base_rules)
    all_clf, all_info = fit_tree(
        all_train_rows,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        criterion=args.criterion,
        paired_positive_weight=args.paired_positive_weight,
        negative_weight=args.negative_weight,
    )
    summary["all_data_selection"] = model_summary(cfg, all_info)
    summary["all_data_canonical"] = evaluate_rules(canonical_paths, base_rules, reference_rules, all_clf, cfg)
    summary["all_data_paired"] = paired_metrics(paired_paths, base_rules, reference_rules, all_clf, cfg)

    for holdout in canonical_paths:
        train_paths = [path for path in canonical_paths if path != holdout]
        train_rows = collect_train_rows(train_paths, paired_paths, base_rules)
        clf, info = fit_tree(
            train_rows,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            criterion=args.criterion,
            paired_positive_weight=args.paired_positive_weight,
            negative_weight=args.negative_weight,
        )
        train_canonical = evaluate_rules(train_paths, base_rules, reference_rules, clf, cfg)
        pair_eval = paired_metrics(paired_paths, base_rules, reference_rules, clf, cfg)
        holdout_eval = evaluate_rules([holdout], base_rules, reference_rules, clf, cfg)
        holdout_name = dataset_name_from_path(holdout)
        holdout_metrics = holdout_eval["datasets"][holdout_name]

        output_row = {
            "holdout_dataset": holdout_name,
            "tag": args.tag,
            "train_min_dataset_ba": f"{train_canonical['min_dataset_balanced_accuracy']:.6f}",
            "train_max_protected_real_fpr": f"{train_canonical['max_protected_real_fpr']:.6f}",
            "paired_mean_fake_accuracy": f"{pair_eval['mean_fake_accuracy']:.6f}",
            "paired_min_fake_accuracy": f"{pair_eval['min_fake_accuracy']:.6f}",
            "holdout_balanced_accuracy": f"{holdout_metrics['balanced_accuracy']:.6f}",
            "holdout_fake_accuracy": f"{holdout_metrics['fake_accuracy']:.6f}",
            "holdout_protected_real_fpr": f"{holdout_metrics['protected_real_fpr']:.6f}",
            "holdout_protected_fake_recall": f"{holdout_metrics['protected_fake_recall']:.6f}",
            "holdout_unprotected_fake_recall": f"{holdout_metrics['unprotected_fake_recall']:.6f}",
        }
        output_rows.append(output_row)
        summary["holdouts"][holdout_name] = {
            "selection": model_summary(cfg, info),
            "train_canonical": train_canonical,
            "paired_objective": pair_eval,
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
