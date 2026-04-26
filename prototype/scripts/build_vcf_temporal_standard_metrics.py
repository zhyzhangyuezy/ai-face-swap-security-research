from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MANIFESTS = ROOT / "manifests"

CALIB_MANIFEST = MANIFESTS / "vcf_c23_calib_temporal_f3_seed20260421_sw010_grouped.csv"
EVAL_MANIFEST = MANIFESTS / "vcf_c23_heldout_temporal_f3_seed20260421_sw010_grouped.csv"

EXTERNAL_MODELS = [
    ("DeepfakeBench UCF", "ucf"),
    ("DeepfakeBench Xception", "xception"),
    ("DeepfakeBench SPSL", "spsl"),
    ("DeepfakeBench F3Net", "f3net"),
    ("DeepfakeBench CORE", "core"),
    ("DeepfakeBench FFD", "ffd"),
    ("DeepfakeBench SRM", "srm"),
    ("Effort official", "effort_official"),
    ("FTCN+TT official adapter", "ftcn_tt"),
]


def latest_path(pattern: str) -> Path:
    matches = sorted(REPORTS.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched: {pattern}")
    return matches[-1]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(path: Path) -> List[dict]:
    text = path.read_text(encoding="utf-8-sig")
    header = text.splitlines()[0] if text else ""
    delimiter = "\t" if "\t" in header and header.count("\t") >= header.count(",") else ","
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def auc_and_eer_from_curve(points: List[tuple[float, float]]) -> tuple[float, float]:
    merged: Dict[float, float] = {}
    for fpr, tpr in points:
        key = round(float(fpr), 12)
        merged[key] = max(float(tpr), merged.get(key, 0.0))
    ordered = sorted(merged.items())
    if not ordered:
        raise ValueError("Empty ROC point set.")
    if ordered[0][0] > 0.0:
        ordered = [(0.0, ordered[0][1])] + ordered
    if ordered[-1][0] < 1.0:
        ordered = ordered + [(1.0, ordered[-1][1])]
    xs = np.asarray([item[0] for item in ordered], dtype=np.float64)
    ys = np.asarray([item[1] for item in ordered], dtype=np.float64)
    auc = float(np.sum((xs[1:] - xs[:-1]) * (ys[1:] + ys[:-1]) * 0.5))
    fnr = 1.0 - ys
    diff = xs - fnr
    crossing = np.where(diff >= 0)[0]
    if crossing.size == 0:
        idx = int(np.argmin(np.abs(diff)))
        eer = float((xs[idx] + fnr[idx]) / 2.0)
        return auc, eer
    idx = int(crossing[0])
    if idx == 0:
        eer = float((xs[idx] + fnr[idx]) / 2.0)
        return auc, eer
    x0 = diff[idx - 1]
    x1 = diff[idx]
    y0 = xs[idx - 1]
    y1 = xs[idx]
    if x1 == x0:
        eer = float((y0 + y1) / 2.0)
        return auc, eer
    alpha = -x0 / (x1 - x0)
    eer = float(y0 + alpha * (y1 - y0))
    return auc, eer


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


def curve_from_sweep(rows: List[dict], score_model: str, aggregate_seeds: bool) -> tuple[float, float]:
    filtered = [row for row in rows if row.get("score_model", row.get("model")) == score_model]
    if not filtered:
        raise ValueError(f"No sweep rows found for model: {score_model}")
    grouped: Dict[float, List[tuple[float, float]]] = {}
    for row in filtered:
        threshold = float(row["threshold"])
        fpr = 1.0 - float(row["heldout_real_accuracy"])
        tpr = float(row["heldout_fake_accuracy"])
        grouped.setdefault(threshold, []).append((fpr, tpr))
    points: List[tuple[float, float]] = []
    for threshold, threshold_points in grouped.items():
        if aggregate_seeds:
            mean_fpr = float(np.mean([item[0] for item in threshold_points]))
            mean_tpr = float(np.mean([item[1] for item in threshold_points]))
            points.append((mean_fpr, mean_tpr))
        else:
            points.extend(threshold_points)
    return auc_and_eer_from_curve(points)


def ours_row() -> dict:
    sweep_rows = load_rows(latest_path("vcf_c23_temporal_f3_locked_sequence_pr4_seed_sweep_mi2000.csv"))
    roc_auc, eer = curve_from_sweep(sweep_rows, "sequence_logreg_pr4_c0.1", aggregate_seeds=True)
    summary = load_json(latest_path("vcf_c23_temporal_f3_locked_sequence_pr4_seed_summary_mi2000.json"))
    agg = summary["selected_aggregate"]
    return {
        "model": "Ours sequence PR4",
        "selection": "5-seed mean threshold sweep of the locked PR4 sequence head",
        "heldout_ba": f"{agg['heldout_balanced_accuracy_mean']:.6f}",
        "heldout_pr_fpr": f"{agg['heldout_protected_real_fpr_mean']:.6f}",
        "roc_auc": f"{roc_auc:.6f}",
        "eer": f"{eer:.6f}",
    }


def external_row(display_name: str, token: str) -> dict:
    summary = load_json(latest_path(f"vcf_c23_temporal_f3_{token}_calibrated_summary_*.json"))
    selected = summary["selections"]["best_by_calib_fpr08"]
    try:
        sweep_path = latest_path(f"vcf_c23_temporal_f3_{token}_calibrated_candidates_*.csv")
    except FileNotFoundError:
        sweep_path = latest_path(f"vcf_c23_temporal_f3_{token}_calibrated_candidates_*.tsv")
    sweep_rows = load_rows(sweep_path)
    roc_auc, eer = curve_from_sweep(sweep_rows, selected["score_model"], aggregate_seeds=False)
    return {
        "model": display_name,
        "selection": f"best_by_calib_fpr08 ({selected['score_model']})",
        "heldout_ba": selected["heldout_balanced_accuracy"],
        "heldout_pr_fpr": selected["heldout_protected_real_fpr"],
        "roc_auc": f"{roc_auc:.6f}",
        "eer": f"{eer:.6f}",
    }


def main() -> None:
    rows = [ours_row()]
    for display_name, token in EXTERNAL_MODELS:
        rows.append(external_row(display_name, token))
    out_path = REPORTS / "vcf_temporal_standard_metrics_2026-04-23.tsv"
    write_tsv(out_path, rows)
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
