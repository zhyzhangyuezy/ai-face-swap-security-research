from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from calibrate_vcf_temporal_feature_fusion import group_rows, vectorize
from calibrate_vcf_temporal_fusion import compute_metrics, labels, masks, row_with_prefix, thresholds


def robust_scale(train_x: np.ndarray, real_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    real_x = train_x[real_mask]
    center = np.median(real_x, axis=0)
    q75 = np.percentile(real_x, 75, axis=0)
    q25 = np.percentile(real_x, 25, axis=0)
    scale = (q75 - q25) / 1.349
    scale = np.where(scale < 1e-5, 1.0, scale)
    return center.astype(np.float32), scale.astype(np.float32)


def centroid_score(x: np.ndarray, centroid: np.ndarray, scale: np.ndarray) -> np.ndarray:
    z = (x - centroid) / scale
    return np.mean(np.square(z), axis=1).astype(np.float32)


def nearest_centroid_score(x: np.ndarray, centroids: List[np.ndarray], scale: np.ndarray) -> np.ndarray:
    scores = [centroid_score(x, centroid, scale) for centroid in centroids]
    return np.min(np.stack(scores, axis=1), axis=1).astype(np.float32)


def score_sets(calib_groups: List[dict], eval_groups: List[dict], mode: str) -> Dict[str, tuple[np.ndarray, np.ndarray]]:
    calib_x = vectorize(calib_groups, mode)
    eval_x = vectorize(eval_groups, mode)
    y = labels(calib_groups)
    m = masks(calib_groups)
    _, scale = robust_scale(calib_x, y == 0)

    variants: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
    real_all = np.median(calib_x[y == 0], axis=0).astype(np.float32)
    variants[f"{mode}_real_concept_all"] = (
        centroid_score(calib_x, real_all, scale),
        centroid_score(eval_x, real_all, scale),
    )

    if int(m["protected_real"].sum()) > 0:
        real_pr = np.median(calib_x[m["protected_real"]], axis=0).astype(np.float32)
        variants[f"{mode}_real_concept_protected"] = (
            centroid_score(calib_x, real_pr, scale),
            centroid_score(eval_x, real_pr, scale),
        )
    if int(m["unprotected_real"].sum()) > 0:
        real_ur = np.median(calib_x[m["unprotected_real"]], axis=0).astype(np.float32)
        variants[f"{mode}_real_concept_unprotected"] = (
            centroid_score(calib_x, real_ur, scale),
            centroid_score(eval_x, real_ur, scale),
        )
    if int(m["protected_real"].sum()) > 0 and int(m["unprotected_real"].sum()) > 0:
        variants[f"{mode}_real_concept_nearest_pr_ur"] = (
            nearest_centroid_score(calib_x, [real_pr, real_ur], scale),
            nearest_centroid_score(eval_x, [real_pr, real_ur], scale),
        )
    return variants


def selection_key(row: dict) -> tuple[float, float, float, float, float]:
    return (
        float(row["calib_balanced_accuracy"]),
        float(row["calib_protected_fake_recall"]),
        float(row["calib_unprotected_fake_recall"]),
        -float(row["calib_protected_real_fpr"]),
        float(row["threshold"]),
    )


def best_under(rows: List[dict], max_calib_pr_fpr: float | None) -> dict | None:
    candidates = rows if max_calib_pr_fpr is None else [
        row for row in rows if float(row["calib_protected_real_fpr"]) <= max_calib_pr_fpr + 1e-12
    ]
    if not candidates:
        return None
    selected = max(candidates, key=selection_key)
    selected["threshold_rule_satisfied"] = "1"
    return selected


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RealID-style real-concept VCF baselines.")
    parser.add_argument("--calib-manifest", required=True)
    parser.add_argument("--calib-passive", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--eval-passive", required=True)
    parser.add_argument("--candidates-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--threshold-step", type=float, default=0.005)
    parser.add_argument("--max-calib-pr-fpr", type=float, default=0.08)
    parser.add_argument("--modes", nargs="+", default=["sequence", "meanstd", "delta"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calib_groups = group_rows(Path(args.calib_manifest), Path(args.calib_passive))
    eval_groups = group_rows(Path(args.eval_manifest), Path(args.eval_passive))

    scores: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for mode in args.modes:
        scores.update(score_sets(calib_groups, eval_groups, mode))

    candidate_rows: List[dict] = []
    for model_name, (calib_scores, eval_scores) in scores.items():
        max_score = max(float(np.max(calib_scores)), float(np.max(eval_scores)), 1e-6)
        calib_norm = calib_scores / max_score
        eval_norm = eval_scores / max_score
        for threshold in thresholds(args.threshold_step):
            calib_metrics = compute_metrics(calib_groups, calib_norm, threshold)
            eval_metrics = compute_metrics(eval_groups, eval_norm, threshold)
            candidate_rows.append(
                {
                    "model": model_name,
                    "threshold": f"{threshold:.6f}",
                    **row_with_prefix("calib", calib_metrics),
                    **row_with_prefix("heldout", eval_metrics),
                    "calib_strict_fpr08": "1" if calib_metrics["protected_real_fpr"] <= 0.08 else "0",
                    "heldout_strict_fpr08": "1" if eval_metrics["protected_real_fpr"] <= 0.08 else "0",
                }
            )

    selections = {
        "best_by_calib_any": best_under(candidate_rows, None),
        "best_by_calib_fpr08": best_under(candidate_rows, args.max_calib_pr_fpr),
    }
    summary = {
        "protocol": "RealID-style independent real-concept distance; calibration-only constrained thresholding",
        "max_calib_pr_fpr": args.max_calib_pr_fpr,
        "calib_groups": len(calib_groups),
        "heldout_groups": len(eval_groups),
        "modes": args.modes,
        "models": sorted(scores),
        "selections": selections,
    }
    write_csv(Path(args.candidates_output), candidate_rows)
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
