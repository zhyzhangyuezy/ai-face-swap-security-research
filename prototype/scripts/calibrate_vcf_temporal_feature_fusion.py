from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean, median
from typing import Dict, List

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

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


def load_feature(path: Path) -> np.ndarray:
    with np.load(path) as payload:
        return np.asarray(payload["feature"], dtype=np.float32)


def resolve_workspace_path(raw: str) -> Path:
    path = Path(raw)
    if path.exists():
        return path

    parts = path.parts
    if "ai-face-swap-security-research" in parts:
        suffix = Path(*parts[parts.index("ai-face-swap-security-research") + 1 :])
        candidate = ROOT / suffix
        if candidate.exists():
            return candidate
    if "workspace" in parts:
        suffix = Path(*parts[parts.index("workspace") + 1 :])
        candidate = ROOT / "workspace" / suffix
        if candidate.exists():
            return candidate
    return path


def target_stem(row: Dict[str, str]) -> str:
    notes = parse_notes(row.get("notes", ""))
    stem = notes.get("stem")
    if stem:
        return stem
    source = Path(row.get("source_path", ""))
    return source.stem or row.get("pair_group_id") or row["sample_id"]


def group_rows(manifest_path: Path, passive_path: Path) -> List[dict]:
    manifest_rows = load_csv(manifest_path)
    passive_rows = {row["sample_id"]: row for row in load_csv(passive_path) if row.get("sample_id")}
    grouped: Dict[str, List[dict]] = {}
    for row in manifest_rows:
        passive = passive_rows.get(row["sample_id"], {})
        if passive.get("status") != "ok" or not passive.get("prob_fake") or not passive.get("feature_path"):
            continue
        feature_path = resolve_workspace_path(passive["feature_path"])
        if not feature_path.exists():
            continue
        notes = parse_notes(row.get("notes", ""))
        temporal_sample_id = notes.get("temporal_sample_id") or row["sample_id"]
        grouped.setdefault(temporal_sample_id, []).append(
            {
                "slot": int(notes.get("temporal_frame_slot", "0")),
                "prob_fake": float(passive["prob_fake"]),
                "feature_path": feature_path,
                "label": row["label"],
                "protection": row["protection"],
                "degradation_type": row["degradation_type"],
                "degradation_level": row["degradation_level"],
                "target_stem": target_stem(row),
            }
        )

    output: List[dict] = []
    for temporal_sample_id, rows in grouped.items():
        rows.sort(key=lambda item: item["slot"])
        if len(rows) < 3:
            continue
        values = [row["prob_fake"] for row in rows[:3]]
        features = np.stack([load_feature(row["feature_path"]) for row in rows[:3]], axis=0)
        output.append(
            {
                "temporal_sample_id": temporal_sample_id,
                "label": rows[0]["label"],
                "protection": rows[0]["protection"],
                "degradation_type": rows[0]["degradation_type"],
                "degradation_level": rows[0]["degradation_level"],
                "target_stem": rows[0]["target_stem"],
                "values": values,
                "features": features,
            }
        )
    return output


def prob_stats(values: List[float]) -> List[float]:
    ordered = sorted(values, reverse=True)
    return [
        values[0],
        values[1],
        values[2],
        mean(values),
        median(values),
        max(values),
        min(values),
        float(np.std(values)),
        max(values) - min(values),
        mean(ordered[:2]),
        mean(values[1:]),
        values[2] - values[0],
        abs(values[2] - values[0]),
    ]


def vectorize(groups: List[dict], mode: str) -> np.ndarray:
    vectors: List[np.ndarray] = []
    for group in groups:
        feats = group["features"]
        pstats = np.asarray(prob_stats(group["values"]), dtype=np.float32)
        if mode == "meanstd":
            vec = np.concatenate([feats.mean(axis=0), feats.std(axis=0), pstats], axis=0)
        elif mode == "delta":
            vec = np.concatenate([feats.mean(axis=0), feats.std(axis=0), feats[2] - feats[0], pstats], axis=0)
        elif mode == "sequence":
            vec = np.concatenate([feats.reshape(-1), pstats], axis=0)
        elif mode == "seq_mean_delta":
            vec = np.concatenate([feats.reshape(-1), feats.mean(axis=0), feats[2] - feats[0], pstats], axis=0)
        else:
            raise ValueError(f"Unknown feature mode: {mode}")
        vectors.append(vec.astype(np.float32))
    return np.stack(vectors, axis=0)


def fit_scores(train_groups: List[dict], eval_groups: List[dict], mode: str) -> Dict[str, tuple[np.ndarray, np.ndarray]]:
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.preprocessing import StandardScaler

    x_train = vectorize(train_groups, mode)
    x_eval = vectorize(eval_groups, mode)
    y_train = labels(train_groups)
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_eval = scaler.transform(x_eval)
    train_masks = masks(train_groups)
    base_weight = np.ones_like(y_train, dtype=np.float32)

    results: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for protected_real_weight in [1.0, 2.0, 4.0]:
        weight_name = f"pr{int(protected_real_weight)}"
        sample_weight = base_weight.copy()
        sample_weight[train_masks["protected_real"]] *= protected_real_weight
        for c_value in [0.03, 0.1]:
            model = LogisticRegression(
                C=c_value,
                max_iter=800,
                class_weight="balanced",
                solver="saga",
                penalty="l2",
                n_jobs=1,
                random_state=20260422,
            )
            model.fit(x_train, y_train, sample_weight=sample_weight)
            name = f"{mode}_logreg_{weight_name}_c{c_value:g}"
            results[name] = (
                model.predict_proba(x_train)[:, 1].astype(np.float32),
                model.predict_proba(x_eval)[:, 1].astype(np.float32),
            )
        for alpha in [0.0001, 0.001]:
            model = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=alpha,
                max_iter=2000,
                tol=1e-4,
                class_weight="balanced",
                random_state=20260422,
            )
            model.fit(x_train, y_train, sample_weight=sample_weight)
            name = f"{mode}_sgd_{weight_name}_a{alpha:g}"
            results[name] = (
                model.predict_proba(x_train)[:, 1].astype(np.float32),
                model.predict_proba(x_eval)[:, 1].astype(np.float32),
            )
    return results


def selection_key(row: dict) -> tuple[float, float, float, float]:
    return (
        float(row["calib_balanced_accuracy"]),
        float(row["calib_protected_fake_recall"]),
        float(row["calib_unprotected_fake_recall"]),
        -float(row["calib_protected_real_fpr"]),
    )


def best_under(rows: List[dict], max_fpr: float | None) -> dict | None:
    candidates = rows if max_fpr is None else [row for row in rows if float(row["calib_protected_real_fpr"]) <= max_fpr]
    if not candidates:
        return None
    return max(candidates, key=selection_key)


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
    parser = argparse.ArgumentParser(description="Train and calibrate VCF temporal feature-fusion heads.")
    parser.add_argument("--calib-manifest", required=True)
    parser.add_argument("--calib-passive", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--eval-passive", required=True)
    parser.add_argument("--candidates-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--threshold-step", type=float, default=0.005)
    parser.add_argument("--modes", nargs="+", default=["meanstd", "delta", "sequence"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calib_groups = group_rows(Path(args.calib_manifest), Path(args.calib_passive))
    eval_groups = group_rows(Path(args.eval_manifest), Path(args.eval_passive))

    score_sets: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for mode in args.modes:
        score_sets.update(fit_scores(calib_groups, eval_groups, mode))

    candidate_rows: List[dict] = []
    for model_name, (calib_scores, eval_scores) in score_sets.items():
        for threshold in thresholds(args.threshold_step):
            calib_metrics = compute_metrics(calib_groups, calib_scores, threshold)
            eval_metrics = compute_metrics(eval_groups, eval_scores, threshold)
            candidate_rows.append(
                {
                    "score_model": model_name,
                    "threshold": f"{threshold:.6f}",
                    **row_with_prefix("calib", calib_metrics),
                    **row_with_prefix("heldout", eval_metrics),
                    "calib_strict_fpr08": "1" if calib_metrics["protected_real_fpr"] <= 0.08 else "0",
                    "heldout_strict_fpr08": "1" if eval_metrics["protected_real_fpr"] <= 0.08 else "0",
                }
            )

    selections = {
        "best_by_calib_any": best_under(candidate_rows, None),
        "best_by_calib_fpr065": best_under(candidate_rows, 0.065),
        "best_by_calib_fpr08": best_under(candidate_rows, 0.08),
        "best_by_calib_fpr12": best_under(candidate_rows, 0.12),
        "best_by_calib_fpr20": best_under(candidate_rows, 0.20),
    }
    summary = {
        "calib_groups": len(calib_groups),
        "eval_groups": len(eval_groups),
        "modes": args.modes,
        "score_models": sorted(score_sets),
        "selections": selections,
    }
    write_csv(Path(args.candidates_output), candidate_rows)
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
