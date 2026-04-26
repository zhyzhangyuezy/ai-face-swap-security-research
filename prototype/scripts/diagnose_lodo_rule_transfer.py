from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def _bootstrap_scripts() -> None:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))


_bootstrap_scripts()

from evaluate_dual_passive_fusion import candidate_rules, feasible, result_row, sort_key
from evaluate_single_passive_lodo import evaluate_rules, load_dataset_rows, route_dataset
from search_routing_thresholds_constrained import accuracy_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose leave-one-dataset-out rule transfer by evaluating every "
            "training-feasible route rule on the held-out dataset."
        )
    )
    parser.add_argument("--bridge", action="append", required=True, help="DATASET=BRIDGE_CSV. Repeat for each dataset.")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--max-protected-real-fpr", type=float, default=0.08)
    parser.add_argument("--min-protected-fake-recall", type=float, default=0.60)
    parser.add_argument("--min-unprotected-fake-recall", type=float, default=0.60)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def holdout_feasible(metrics: dict, args: argparse.Namespace) -> bool:
    return (
        metrics["protected_real_fpr"] <= args.max_protected_real_fpr
        and metrics["protected_fake_recall"] >= args.min_protected_fake_recall
        and metrics["unprotected_fake_recall"] >= args.min_unprotected_fake_recall
    )


def candidate_row(holdout: str, train_rank: int, row: dict, train_metrics: dict, holdout_metrics: dict, args) -> dict:
    return {
        "holdout_dataset": holdout,
        "train_rank": str(train_rank),
        "config_id": row["config_id"],
        "train_feasible": row["feasible"],
        "holdout_feasible": "1" if holdout_feasible(holdout_metrics, args) else "0",
        "passive_fake_threshold": row["passive_fake_threshold"],
        "protected_delta_abs_anomaly_threshold": row["protected_delta_abs_anomaly_threshold"],
        "train_overall_ba": f"{train_metrics['combined']['balanced_accuracy']:.6f}",
        "train_min_dataset_ba": f"{train_metrics['min_dataset_balanced_accuracy']:.6f}",
        "train_max_protected_real_fpr": f"{train_metrics['max_protected_real_fpr']:.6f}",
        "train_min_protected_fake_recall": f"{train_metrics['min_protected_fake_recall']:.6f}",
        "train_min_unprotected_fake_recall": f"{train_metrics['min_unprotected_fake_recall']:.6f}",
        "holdout_balanced_accuracy": f"{holdout_metrics['balanced_accuracy']:.6f}",
        "holdout_protected_real_fpr": f"{holdout_metrics['protected_real_fpr']:.6f}",
        "holdout_protected_fake_recall": f"{holdout_metrics['protected_fake_recall']:.6f}",
        "holdout_unprotected_fake_recall": f"{holdout_metrics['unprotected_fake_recall']:.6f}",
        "holdout_real_accuracy": f"{holdout_metrics['real_accuracy']:.6f}",
        "holdout_fake_accuracy": f"{holdout_metrics['fake_accuracy']:.6f}",
    }


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
    dataset_rows = load_dataset_rows(args.bridge)
    rows_out: list[dict] = []
    summary: dict[str, dict] = {"holdouts": {}}

    for holdout in sorted(dataset_rows):
        train_rows = {name: rows for name, rows in dataset_rows.items() if name != holdout}
        scored = []
        for config_id, rules in enumerate(candidate_rules(), start=1):
            train_metrics = evaluate_rules(train_rows, rules)
            is_train_feasible = feasible(train_metrics, args)
            row = result_row("single", config_id, rules, train_metrics, is_train_feasible)
            holdout_metrics = accuracy_metrics(route_dataset(dataset_rows[holdout], rules))
            scored.append((row, rules, train_metrics, is_train_feasible, holdout_metrics))

        scored.sort(key=lambda item: sort_key((item[0], item[1], item[2], item[3])), reverse=True)
        for train_rank, (row, _, train_metrics, _, holdout_metrics) in enumerate(scored[: args.top_k], start=1):
            rows_out.append(candidate_row(holdout, train_rank, row, train_metrics, holdout_metrics, args))

        train_selected = scored[0]
        train_feasible_candidates = [item for item in scored if item[3]]
        transfer_safe = [item for item in train_feasible_candidates if holdout_feasible(item[4], args)]
        holdout_oracle = sorted(
            transfer_safe,
            key=lambda item: (
                item[4]["balanced_accuracy"],
                item[2]["min_dataset_balanced_accuracy"],
                item[2]["combined"]["balanced_accuracy"],
                -item[4]["protected_real_fpr"],
            ),
            reverse=True,
        )
        selected_row = candidate_row(holdout, 1, train_selected[0], train_selected[2], train_selected[4], args)
        oracle_row = (
            candidate_row(holdout, -1, holdout_oracle[0][0], holdout_oracle[0][2], holdout_oracle[0][4], args)
            if holdout_oracle
            else None
        )
        summary["holdouts"][holdout] = {
            "train_selected": selected_row,
            "train_feasible_count": len(train_feasible_candidates),
            "transfer_safe_count": len(transfer_safe),
            "best_transfer_safe_oracle": oracle_row,
        }

    write_csv(Path(args.output_csv), rows_out)
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
