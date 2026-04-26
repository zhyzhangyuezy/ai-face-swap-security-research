from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from calibrate_vcf_temporal_feature_fusion import group_rows, vectorize
from calibrate_vcf_temporal_fusion import compute_metrics, labels, masks, row_with_prefix, thresholds


def parse_notes(notes: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for part in notes.split(";"):
        token = part.strip()
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


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


def manifest_sets(manifest_path: Path, passive_path: Path) -> dict:
    rows = load_csv(manifest_path)
    passive_rows = {row["sample_id"]: row for row in load_csv(passive_path) if row.get("sample_id")}
    target_keys = set()
    temporal_ids = set()
    sample_ids = set()
    raw_videos = set()
    source_paths = set()
    planned_outputs = set()
    feature_paths = set()
    protected_missing = 0
    feature_missing = 0

    for row in rows:
        notes = parse_notes(row.get("notes", ""))
        target_key = notes.get("temporal_split_key") or "|".join(
            [notes.get("domain", "c23"), notes.get("resolution", ""), notes.get("bucket", ""), notes.get("stem", "")]
        )
        target_keys.add(target_key)
        temporal_ids.add(notes.get("temporal_sample_id") or row["sample_id"])
        sample_ids.add(row["sample_id"])
        if notes.get("raw_video_path"):
            raw_videos.add(notes["raw_video_path"])
        source_paths.add(row["source_path"])
        planned_outputs.add(row["planned_output_path"])
        if row["protection"] == "protected" and not Path(row["planned_output_path"]).exists():
            protected_missing += 1
        passive = passive_rows.get(row["sample_id"], {})
        if passive.get("feature_path"):
            feature_paths.add(passive["feature_path"])
            if not Path(passive["feature_path"]).exists():
                feature_missing += 1
    return {
        "rows": len(rows),
        "target_keys": target_keys,
        "temporal_ids": temporal_ids,
        "sample_ids": sample_ids,
        "raw_videos": raw_videos,
        "source_paths": source_paths,
        "planned_outputs": planned_outputs,
        "feature_paths": feature_paths,
        "protected_missing": protected_missing,
        "feature_missing": feature_missing,
    }


def leakage_audit(calib_manifest: Path, calib_passive: Path, eval_manifest: Path, eval_passive: Path) -> dict:
    calib = manifest_sets(calib_manifest, calib_passive)
    heldout = manifest_sets(eval_manifest, eval_passive)
    overlaps = {
        "target_keys": len(calib["target_keys"] & heldout["target_keys"]),
        "temporal_ids": len(calib["temporal_ids"] & heldout["temporal_ids"]),
        "sample_ids": len(calib["sample_ids"] & heldout["sample_ids"]),
        "raw_videos": len(calib["raw_videos"] & heldout["raw_videos"]),
        "source_paths": len(calib["source_paths"] & heldout["source_paths"]),
        "planned_outputs": len(calib["planned_outputs"] & heldout["planned_outputs"]),
        "feature_paths": len(calib["feature_paths"] & heldout["feature_paths"]),
    }
    pass_flag = (
        all(value == 0 for value in overlaps.values())
        and calib["protected_missing"] == 0
        and heldout["protected_missing"] == 0
        and calib["feature_missing"] == 0
        and heldout["feature_missing"] == 0
    )
    return {
        "pass": pass_flag,
        "calib_counts": {key: len(value) if isinstance(value, set) else value for key, value in calib.items()},
        "heldout_counts": {key: len(value) if isinstance(value, set) else value for key, value in heldout.items()},
        "overlaps": overlaps,
    }


def train_scores(
    calib_groups: List[dict],
    eval_groups: List[dict],
    seed: int,
    max_iter: int,
    tol: float,
    feature_mode: str,
    protected_real_weight: float,
    c_value: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    x_train = vectorize(calib_groups, feature_mode)
    x_eval = vectorize(eval_groups, feature_mode)
    y_train = labels(calib_groups)
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_eval = scaler.transform(x_eval)
    sample_weight = np.ones_like(y_train, dtype=np.float32)
    sample_weight[masks(calib_groups)["protected_real"]] *= protected_real_weight
    model = LogisticRegression(
        C=c_value,
        max_iter=max_iter,
        class_weight="balanced",
        solver="saga",
        penalty="l2",
        n_jobs=1,
        random_state=seed,
        tol=tol,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, y_train, sample_weight=sample_weight)
    meta = {
        "seed": seed,
        "max_iter": max_iter,
        "tol": tol,
        "n_iter": int(model.n_iter_[0]) if hasattr(model, "n_iter_") else None,
        "convergence_warnings": sum(1 for item in caught if issubclass(item.category, ConvergenceWarning)),
        "feature_mode": feature_mode,
        "protected_real_weight": protected_real_weight,
        "c": c_value,
    }
    return model.predict_proba(x_train)[:, 1].astype(np.float32), model.predict_proba(x_eval)[:, 1].astype(np.float32), meta


def select_locked_threshold(rows: List[dict], min_calib_ba: float, max_calib_pr_fpr: float) -> dict:
    eps = 1e-12
    candidates = [
        row
        for row in rows
        if float(row["calib_balanced_accuracy"]) + eps >= min_calib_ba
        and float(row["calib_protected_real_fpr"]) <= max_calib_pr_fpr + eps
    ]
    if candidates:
        selected = max(candidates, key=lambda row: (float(row["threshold"]), float(row["calib_balanced_accuracy"])))
        selected["threshold_rule_satisfied"] = "1"
        return selected
    selected = max(
        rows,
        key=lambda row: (
            -max(0.0, float(row["calib_protected_real_fpr"]) - max_calib_pr_fpr),
            -max(0.0, min_calib_ba - float(row["calib_balanced_accuracy"])),
            float(row["calib_balanced_accuracy"]),
            float(row["threshold"]),
        ),
    )
    selected["threshold_rule_satisfied"] = "0"
    return selected


def aggregate_selected(rows: List[dict]) -> dict:
    metric_keys = [
        "heldout_balanced_accuracy",
        "heldout_protected_real_fpr",
        "heldout_protected_fake_recall",
        "heldout_unprotected_fake_recall",
        "heldout_real_accuracy",
        "heldout_fake_accuracy",
        "threshold",
    ]
    summary = {}
    for key in metric_keys:
        values = [float(row[key]) for row in rows]
        summary[f"{key}_mean"] = mean(values)
        summary[f"{key}_std"] = pstdev(values) if len(values) > 1 else 0.0
        summary[f"{key}_min"] = min(values)
        summary[f"{key}_max"] = max(values)
    summary["all_threshold_rules_satisfied"] = all(row["threshold_rule_satisfied"] == "1" for row in rows)
    summary["all_strict_fpr08"] = all(float(row["heldout_protected_real_fpr"]) <= 0.08 for row in rows)
    summary["all_fake_recall_90"] = all(
        float(row["heldout_protected_fake_recall"]) >= 0.90 and float(row["heldout_unprotected_fake_recall"]) >= 0.90
        for row in rows
    )
    summary["all_ba_90"] = all(float(row["heldout_balanced_accuracy"]) >= 0.90 for row in rows)
    summary["high_level_vcf_gate_pass"] = (
        summary["all_threshold_rules_satisfied"]
        and summary["all_strict_fpr08"]
        and summary["all_fake_recall_90"]
        and summary["all_ba_90"]
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locked VCF temporal sequence-head stability and leakage evaluation.")
    parser.add_argument("--calib-manifest", required=True)
    parser.add_argument("--calib-passive", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--eval-passive", required=True)
    parser.add_argument("--selected-output", required=True)
    parser.add_argument("--sweep-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--audit-output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 21, 42, 84, 20260422])
    parser.add_argument("--threshold-step", type=float, default=0.005)
    parser.add_argument("--min-calib-ba", type=float, default=1.0)
    parser.add_argument("--max-calib-pr-fpr", type=float, default=0.0)
    parser.add_argument("--max-iter", type=int, default=800)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--feature-mode", default="sequence", choices=["meanstd", "delta", "sequence", "seq_mean_delta"])
    parser.add_argument("--protected-real-weight", type=float, default=4.0)
    parser.add_argument("--c-value", type=float, default=0.1)
    parser.add_argument("--model-name", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calib_manifest = Path(args.calib_manifest)
    calib_passive = Path(args.calib_passive)
    eval_manifest = Path(args.eval_manifest)
    eval_passive = Path(args.eval_passive)
    audit = leakage_audit(calib_manifest, calib_passive, eval_manifest, eval_passive)

    calib_groups = group_rows(calib_manifest, calib_passive)
    eval_groups = group_rows(eval_manifest, eval_passive)
    sweep_rows: List[dict] = []
    selected_rows: List[dict] = []
    model_meta: List[dict] = []

    for seed in args.seeds:
        calib_scores, eval_scores, meta = train_scores(
            calib_groups,
            eval_groups,
            seed,
            args.max_iter,
            args.tol,
            args.feature_mode,
            args.protected_real_weight,
            args.c_value,
        )
        model_meta.append(meta)
        model_name = args.model_name or f"{args.feature_mode}_logreg_pr{args.protected_real_weight:g}_c{args.c_value:g}"
        seed_rows: List[dict] = []
        for threshold in thresholds(args.threshold_step):
            calib_metrics = compute_metrics(calib_groups, calib_scores, threshold)
            eval_metrics = compute_metrics(eval_groups, eval_scores, threshold)
            row = {
                "model": model_name,
                "seed": str(seed),
                "threshold": f"{threshold:.6f}",
                **row_with_prefix("calib", calib_metrics),
                **row_with_prefix("heldout", eval_metrics),
                "calib_strict_fpr08": "1" if calib_metrics["protected_real_fpr"] <= 0.08 else "0",
                "heldout_strict_fpr08": "1" if eval_metrics["protected_real_fpr"] <= 0.08 else "0",
            }
            seed_rows.append(row)
            sweep_rows.append(row)
        selected_rows.append(select_locked_threshold(seed_rows, args.min_calib_ba, args.max_calib_pr_fpr))

    summary = {
        "protocol": {
            "model": args.model_name or f"{args.feature_mode}_logreg_pr{args.protected_real_weight:g}_c{args.c_value:g}",
            "feature_mode": args.feature_mode,
            "protected_real_weight": args.protected_real_weight,
            "c": args.c_value,
            "threshold_rule": (
                "highest threshold whose calib balanced_accuracy >= min_calib_ba "
                "and calib protected_real_fpr <= max_calib_pr_fpr"
            ),
            "min_calib_ba": args.min_calib_ba,
            "max_calib_pr_fpr": args.max_calib_pr_fpr,
            "threshold_step": args.threshold_step,
            "seeds": args.seeds,
        },
        "leakage_audit": audit,
        "calib_groups": len(calib_groups),
        "heldout_groups": len(eval_groups),
        "model_meta": model_meta,
        "selected": selected_rows,
        "selected_aggregate": aggregate_selected(selected_rows),
    }
    write_csv(Path(args.sweep_output), sweep_rows)
    write_csv(Path(args.selected_output), selected_rows)
    Path(args.audit_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.audit_output).write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["selected_aggregate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
