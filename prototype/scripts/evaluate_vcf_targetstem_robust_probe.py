from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Dict, List

import numpy as np

from calibrate_vcf_temporal_feature_fusion import group_rows, vectorize
from calibrate_vcf_temporal_fusion import compute_metrics, labels, masks, thresholds


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "prototype" / "reports"
MANIFESTS = ROOT / "prototype" / "manifests"


def write_tsv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def stratified_stem_split(groups: List[dict], val_fraction: float, seed: int) -> tuple[List[dict], List[dict]]:
    rng = random.Random(seed)
    stems = sorted({group["target_stem"] for group in groups})
    rng.shuffle(stems)
    val_count = max(1, round(len(stems) * val_fraction))
    val_stems = set(stems[:val_count])
    train_groups = [group for group in groups if group["target_stem"] not in val_stems]
    val_groups = [group for group in groups if group["target_stem"] in val_stems]
    return train_groups, val_groups


def sample_weights(groups: List[dict], protected_real_weight: float, stem_mode: str) -> np.ndarray:
    y = labels(groups)
    mask = masks(groups)
    weights = np.ones_like(y, dtype=np.float32)
    for cls in [0, 1]:
        cls_mask = y == cls
        weights[cls_mask] *= len(y) / (2.0 * max(1, int(cls_mask.sum())))
    weights[mask["protected_real"]] *= protected_real_weight

    if stem_mode != "none":
        counts = Counter(group["target_stem"] for group in groups)
        max_count = max(counts.values())
        stem_weights = []
        for group in groups:
            ratio = max_count / counts[group["target_stem"]]
            stem_weights.append(np.sqrt(ratio) if stem_mode == "sqrt" else ratio)
        stem_weights = np.asarray(stem_weights, dtype=np.float32)
        stem_weights /= float(stem_weights.mean())
        weights *= stem_weights
    return weights.astype(np.float32)


def fit_sgd(
    train_x: np.ndarray,
    train_y: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    seed: int,
    max_iter: int,
) -> object:
    from sklearn.linear_model import SGDClassifier

    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        max_iter=max_iter,
        tol=1e-3,
        random_state=seed,
        early_stopping=False,
        n_iter_no_change=20,
    )
    model.fit(train_x, train_y, sample_weight=weights)
    return model


def predict_scores(model: object, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1].astype(np.float32)
    decision = model.decision_function(x)
    return (1.0 / (1.0 + np.exp(-decision))).astype(np.float32)


def select_threshold(groups: List[dict], scores: np.ndarray, max_pr_fpr: float) -> tuple[float, dict]:
    rows = []
    for threshold in thresholds(0.005):
        metric = compute_metrics(groups, scores, threshold)
        rows.append((threshold, metric))
    candidates = [row for row in rows if row[1]["protected_real_fpr"] <= max_pr_fpr + 1e-12]
    if not candidates:
        candidates = rows
    return max(
        candidates,
        key=lambda row: (
            row[1]["balanced_accuracy"],
            row[1]["protected_fake_recall"],
            row[1]["unprotected_fake_recall"],
            -row[1]["protected_real_fpr"],
            row[0],
        ),
    )


def row(prefix: str, metric: dict) -> dict:
    return {
        f"{prefix}_ba": f"{metric['balanced_accuracy']:.6f}",
        f"{prefix}_pr_fpr": f"{metric['protected_real_fpr']:.6f}",
        f"{prefix}_pf_recall": f"{metric['protected_fake_recall']:.6f}",
        f"{prefix}_uf_recall": f"{metric['unprotected_fake_recall']:.6f}",
        f"{prefix}_real_acc": f"{metric['real_accuracy']:.6f}",
        f"{prefix}_fake_acc": f"{metric['fake_accuracy']:.6f}",
    }


def summarize(rows: List[dict], key_prefix: str) -> dict:
    values = {}
    for metric in ["ba", "pr_fpr", "pf_recall", "uf_recall"]:
        vals = [float(row[f"{key_prefix}_{metric}"]) for row in rows]
        values[f"{key_prefix}_{metric}_mean"] = mean(vals)
        values[f"{key_prefix}_{metric}_min"] = min(vals)
        values[f"{key_prefix}_{metric}_max"] = max(vals)
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Source-family robust probe for VCF target-stem-disjoint stress.")
    parser.add_argument(
        "--calib-manifest",
        default=str(MANIFESTS / "vcf_c23_targetstem_calib_temporal_f3_seed20260425_grouped.csv"),
    )
    parser.add_argument(
        "--heldout-manifest",
        default=str(MANIFESTS / "vcf_c23_targetstem_heldout_temporal_f3_seed20260425_grouped.csv"),
    )
    parser.add_argument(
        "--calib-passive",
        default=str(REPORTS / "vcf_c23_targetstem_calib_temporal_f3_dualrealc_tailm25_fw010_realm25_passive_results.csv"),
    )
    parser.add_argument(
        "--heldout-passive",
        default=str(REPORTS / "vcf_c23_targetstem_heldout_temporal_f3_dualrealc_tailm25_fw010_realm25_passive_results.csv"),
    )
    parser.add_argument("--output-tsv", default=str(REPORTS / "vcf_c23_targetstem_robust_probe_2026-04-26.tsv"))
    parser.add_argument("--summary-output", default=str(REPORTS / "vcf_c23_targetstem_robust_probe_2026-04-26.json"))
    parser.add_argument("--seed", type=int, default=20260426)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--max-val-pr-fpr", type=float, default=0.08)
    parser.add_argument("--max-iter", type=int, default=700)
    parser.add_argument("--protected-real-weights", nargs="+", type=float, default=[4.0, 8.0, 16.0])
    parser.add_argument("--stem-weights", nargs="+", default=["none", "sqrt", "inv"], choices=["none", "sqrt", "inv"])
    parser.add_argument("--alphas", nargs="+", type=float, default=[1e-3, 3e-3, 1e-2])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from sklearn.preprocessing import StandardScaler

    calib_groups = group_rows(Path(args.calib_manifest), Path(args.calib_passive))
    heldout_groups = group_rows(Path(args.heldout_manifest), Path(args.heldout_passive))
    train_groups, val_groups = stratified_stem_split(calib_groups, args.val_fraction, args.seed)

    scaler = StandardScaler().fit(vectorize(train_groups, "sequence"))
    train_x = scaler.transform(vectorize(train_groups, "sequence"))
    val_x = scaler.transform(vectorize(val_groups, "sequence"))
    heldout_x = scaler.transform(vectorize(heldout_groups, "sequence"))
    train_y = labels(train_groups)

    probe_rows: List[dict] = []
    total_runs = len(args.protected_real_weights) * len(args.stem_weights) * len(args.alphas)
    run_index = 0
    for protected_real_weight in args.protected_real_weights:
        for stem_mode in args.stem_weights:
            for alpha in args.alphas:
                run_index += 1
                print(
                    f"[{run_index}/{total_runs}] prw={protected_real_weight:g} stem={stem_mode} alpha={alpha:g}",
                    flush=True,
                )
                weights = sample_weights(train_groups, protected_real_weight, stem_mode)
                model = fit_sgd(train_x, train_y, weights, alpha, args.seed, args.max_iter)
                val_scores = predict_scores(model, val_x)
                heldout_scores = predict_scores(model, heldout_x)
                val_threshold, val_metric = select_threshold(val_groups, val_scores, args.max_val_pr_fpr)
                heldout_metric = compute_metrics(heldout_groups, heldout_scores, val_threshold)
                oracle_threshold, oracle_metric = select_threshold(heldout_groups, heldout_scores, 0.08)
                probe_rows.append(
                    {
                        "protocol": "inner-stem-val-selected",
                        "protected_real_weight": protected_real_weight,
                        "stem_weight": stem_mode,
                        "alpha": alpha,
                        "threshold": f"{val_threshold:.6f}",
                        **row("val", val_metric),
                        **row("heldout", heldout_metric),
                        "oracle_threshold": f"{oracle_threshold:.6f}",
                        **row("oracle", oracle_metric),
                    }
                )

    selected = max(
        probe_rows,
        key=lambda item: (
            float(item["val_ba"]),
            float(item["val_pf_recall"]),
            float(item["val_uf_recall"]),
            -float(item["val_pr_fpr"]),
        ),
    )
    best_heldout_valid = max(
        probe_rows,
        key=lambda item: (
            float(item["heldout_pr_fpr"]) <= 0.08,
            float(item["heldout_ba"]) if float(item["heldout_pr_fpr"]) <= 0.08 else -1.0,
        ),
    )
    best_oracle = max(probe_rows, key=lambda item: float(item["oracle_ba"]))
    summary = {
        "protocol": "diagnostic only; model/threshold selected on calibration target-stem inner validation, held-out unchanged",
        "max_val_pr_fpr": args.max_val_pr_fpr,
        "calib_groups": len(calib_groups),
        "inner_train_groups": len(train_groups),
        "inner_val_groups": len(val_groups),
        "heldout_groups": len(heldout_groups),
        "calib_stems": len({group["target_stem"] for group in calib_groups}),
        "inner_train_stems": len({group["target_stem"] for group in train_groups}),
        "inner_val_stems": len({group["target_stem"] for group in val_groups}),
        "heldout_stems": len({group["target_stem"] for group in heldout_groups}),
        "selected_by_inner_val": selected,
        "best_heldout_valid_after_inner_selection_grid": best_heldout_valid,
        "best_heldout_oracle_fpr08_ceiling": best_oracle,
        "heldout_grid_summary": summarize(probe_rows, "heldout"),
        "oracle_grid_summary": summarize(probe_rows, "oracle"),
    }
    write_tsv(Path(args.output_tsv), probe_rows)
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
