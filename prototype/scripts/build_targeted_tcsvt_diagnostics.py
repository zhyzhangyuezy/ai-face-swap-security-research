from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "prototype" / "reports"
MANIFESTS = ROOT / "prototype" / "manifests"

MAIN_CALIB_MANIFEST = MANIFESTS / "vcf_c23_calib_temporal_f3_seed20260421_sw010_grouped.csv"
MAIN_HELDOUT_MANIFEST = MANIFESTS / "vcf_c23_heldout_temporal_f3_seed20260421_sw010_grouped.csv"
MAIN_CALIB_PASSIVE = REPORTS / "vcf_c23_calib_temporal_f3_dualrealc_tailm25_fw010_realm25_passive_results.csv"
MAIN_HELDOUT_PASSIVE = REPORTS / "vcf_c23_heldout_temporal_f3_dualrealc_tailm25_fw010_realm25_passive_results.csv"

STEM_CALIB_MANIFEST = MANIFESTS / "vcf_c23_targetstem_calib_temporal_f3_seed20260425_grouped.csv"
STEM_HELDOUT_MANIFEST = MANIFESTS / "vcf_c23_targetstem_heldout_temporal_f3_seed20260425_grouped.csv"
STEM_CALIB_PASSIVE = REPORTS / "vcf_c23_targetstem_calib_temporal_f3_dualrealc_tailm25_fw010_realm25_passive_results.csv"
STEM_HELDOUT_PASSIVE = REPORTS / "vcf_c23_targetstem_heldout_temporal_f3_dualrealc_tailm25_fw010_realm25_passive_results.csv"

OUT_ERROR = REPORTS / "source_family_error_decomposition_2026-04-26.tsv"
OUT_TEMPORAL = REPORTS / "temporal_ablation_review_table_2026-04-26.tsv"
OUT_ORDER = REPORTS / "temporal_order_control_2026-04-26.tsv"
OUT_CALIB = REPORTS / "calibration_fraction_sensitivity_2026-04-26.tsv"
OUT_SUMMARY = REPORTS / "targeted_tcsvt_diagnostics_2026-04-26.json"

SEEDS = [7, 21, 42]
CALIBRATION_SEEDS = [7]
CI_SAFE_THRESHOLD = 0.88
TARGETSTEM_THRESHOLD = 0.84
FAKE_RECALL_FLOOR = 0.9985


def load_csv(path: Path, delimiter: str = ",") -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


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


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def parse_notes(notes: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for part in notes.split(";"):
        token = part.strip()
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def resolve_workspace_path(raw: str) -> Path:
    path = Path(raw)
    if path.exists():
        return path
    parts = path.parts
    if "workspace" in parts:
        suffix = Path(*parts[parts.index("workspace") + 1 :])
        candidate = ROOT / "workspace" / suffix
        if candidate.exists():
            return candidate
    return path


def load_feature(raw: str) -> np.ndarray:
    path = resolve_workspace_path(raw)
    with np.load(path) as payload:
        return np.asarray(payload["feature"], dtype=np.float32)


def target_stem(row: Dict[str, str]) -> str:
    notes = parse_notes(row.get("notes", ""))
    stem = notes.get("stem")
    if stem:
        return stem
    source = Path(row.get("source_path", ""))
    return source.stem or row.get("pair_group_id") or row["sample_id"]


def temporal_group_id(row: Dict[str, str]) -> str:
    notes = parse_notes(row.get("notes", ""))
    return notes.get("temporal_sample_id") or row["sample_id"]


def group_rows(manifest_path: Path, passive_path: Path) -> List[dict]:
    manifest_rows = load_csv(manifest_path)
    passive_rows = {row["sample_id"]: row for row in load_csv(passive_path) if row.get("sample_id")}
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in manifest_rows:
        passive = passive_rows.get(row["sample_id"], {})
        if passive.get("status") != "ok" or not passive.get("prob_fake") or not passive.get("feature_path"):
            continue
        feature_path = resolve_workspace_path(passive["feature_path"])
        if not feature_path.exists():
            continue
        notes = parse_notes(row.get("notes", ""))
        grouped[temporal_group_id(row)].append(
            {
                "slot": int(notes.get("temporal_frame_slot", "0")),
                "prob_fake": float(passive["prob_fake"]),
                "feature_path": str(feature_path),
                "label": row["label"],
                "protection": row["protection"],
                "degradation_type": row.get("degradation_type") or "none",
                "degradation_level": row.get("degradation_level") or "",
                "target_stem": target_stem(row),
                "temporal_group_id": temporal_group_id(row),
                "sample_id": row["sample_id"],
            }
        )

    output: List[dict] = []
    for group_id, rows in grouped.items():
        rows.sort(key=lambda item: item["slot"])
        if len(rows) < 3:
            continue
        values = [row["prob_fake"] for row in rows[:3]]
        features = np.stack([load_feature(row["feature_path"]) for row in rows[:3]], axis=0)
        first = rows[0]
        output.append(
            {
                "temporal_group_id": group_id,
                "label": first["label"],
                "protection": first["protection"],
                "degradation_type": first["degradation_type"],
                "degradation_level": first["degradation_level"],
                "target_stem": first["target_stem"],
                "values": values,
                "features": features,
            }
        )
    return output


def prob_stats(values: List[float]) -> np.ndarray:
    ordered = sorted(values, reverse=True)
    return np.asarray(
        [
            values[0],
            values[1],
            values[2],
            mean(values),
            sorted(values)[1],
            max(values),
            min(values),
            float(np.std(values)),
            max(values) - min(values),
            mean(ordered[:2]),
            mean(values[1:]),
            values[2] - values[0],
            abs(values[2] - values[0]),
        ],
        dtype=np.float32,
    )


def clone_with_order(groups: List[dict], mode: str, seed: int = 20260426) -> List[dict]:
    rng = random.Random(seed)
    output: List[dict] = []
    for idx, group in enumerate(groups):
        if mode == "ordered":
            order = [0, 1, 2]
        elif mode == "reversed":
            order = [2, 1, 0]
        elif mode == "random":
            local = random.Random(seed + idx)
            order = [0, 1, 2]
            local.shuffle(order)
        else:
            raise ValueError(f"Unknown order mode: {mode}")
        copied = dict(group)
        copied["values"] = [group["values"][i] for i in order]
        copied["features"] = group["features"][order]
        output.append(copied)
    return output


def vectorize(groups: List[dict], mode: str = "sequence") -> np.ndarray:
    vectors: List[np.ndarray] = []
    for group in groups:
        feats = np.asarray(group["features"], dtype=np.float32)
        pstats = prob_stats(group["values"])
        if mode == "meanstd":
            vec = np.concatenate([feats.mean(axis=0), feats.std(axis=0), pstats], axis=0)
        elif mode == "delta":
            vec = np.concatenate([feats.mean(axis=0), feats.std(axis=0), feats[2] - feats[0], pstats], axis=0)
        elif mode == "sequence":
            vec = np.concatenate([feats.reshape(-1), pstats], axis=0)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        vectors.append(vec.astype(np.float32))
    return np.stack(vectors, axis=0)


def labels(groups: List[dict]) -> np.ndarray:
    return np.asarray([1 if group["label"] == "fake" else 0 for group in groups], dtype=np.int64)


def masks(groups: List[dict]) -> Dict[str, np.ndarray]:
    y = labels(groups)
    protection = np.asarray([group["protection"] for group in groups])
    return {
        "real": y == 0,
        "fake": y == 1,
        "protected_real": (y == 0) & (protection == "protected"),
        "protected_fake": (y == 1) & (protection == "protected"),
        "unprotected_fake": (y == 1) & (protection == "unprotected"),
    }


def rate(correct: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) == 0:
        return 0.0
    return float(correct[mask].mean())


def compute_metrics(groups: List[dict], scores: np.ndarray, threshold: float) -> Dict[str, float]:
    y = labels(groups)
    mask = masks(groups)
    pred_fake = scores >= threshold
    correct = pred_fake == (y == 1)
    real_acc = rate(correct, mask["real"])
    fake_acc = rate(correct, mask["fake"])
    return {
        "groups": float(len(groups)),
        "balanced_accuracy": 0.5 * (real_acc + fake_acc),
        "protected_real_fpr": 1.0 - rate(correct, mask["protected_real"]),
        "protected_fake_recall": rate(correct, mask["protected_fake"]),
        "unprotected_fake_recall": rate(correct, mask["unprotected_fake"]),
        "protected_real_fp": float(((pred_fake == 1) & mask["protected_real"]).sum()),
        "protected_real_total": float(mask["protected_real"].sum()),
    }


def wilson_upper(k: int, n: int, z: float = 1.959963984540054) -> float:
    if n <= 0:
        return float("nan")
    p = k / n
    den = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
    return (centre + margin) / den


def fit_logreg(
    calib_groups: List[dict],
    eval_groups_by_mode: Dict[str, List[dict]],
    seed: int,
    train_groups_for_scaler: List[dict] | None = None,
) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import warnings

    train_groups_for_scaler = train_groups_for_scaler or calib_groups
    x_train = vectorize(calib_groups, "sequence")
    y_train = labels(calib_groups)
    scaler = StandardScaler().fit(vectorize(train_groups_for_scaler, "sequence"))
    x_train = scaler.transform(x_train)
    train_masks = masks(calib_groups)
    sample_weight = np.ones_like(y_train, dtype=np.float32)
    sample_weight[train_masks["protected_real"]] *= 4.0
    model = LogisticRegression(
        C=0.1,
        max_iter=500,
        class_weight="balanced",
        solver="liblinear",
        penalty="l2",
        random_state=seed,
        tol=1e-4,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_train, y_train, sample_weight=sample_weight)
    calib_scores = model.predict_proba(x_train)[:, 1].astype(np.float32)
    eval_scores = {
        mode: model.predict_proba(scaler.transform(vectorize(groups, "sequence")))[:, 1].astype(np.float32)
        for mode, groups in eval_groups_by_mode.items()
    }
    return calib_scores, eval_scores


def thresholds(step: float = 0.005) -> List[float]:
    count = int(round(1.0 / step))
    values = [round(idx * step, 6) for idx in range(count + 1)]
    for value in [0.95, 0.96, 0.97, 0.98, 0.99, 0.995]:
        if value not in values:
            values.append(value)
    return sorted(values)


def select_threshold(calib_groups: List[dict], calib_scores: np.ndarray) -> float:
    candidates: List[float] = []
    for threshold in thresholds():
        metrics = compute_metrics(calib_groups, calib_scores, threshold)
        if (
            metrics["protected_real_fpr"] <= 0.0 + 1e-12
            and metrics["protected_fake_recall"] >= FAKE_RECALL_FLOOR
            and metrics["unprotected_fake_recall"] >= FAKE_RECALL_FLOOR
        ):
            candidates.append(threshold)
    if candidates:
        return max(candidates)
    best = max(
        thresholds(),
        key=lambda threshold: (
            -compute_metrics(calib_groups, calib_scores, threshold)["protected_real_fpr"],
            compute_metrics(calib_groups, calib_scores, threshold)["protected_fake_recall"],
            compute_metrics(calib_groups, calib_scores, threshold)["unprotected_fake_recall"],
        ),
    )
    return best


def stratified_subset(groups: List[dict], fraction: float, seed: int) -> List[dict]:
    if fraction >= 0.999:
        return groups
    strata: Dict[tuple[str, str], List[dict]] = defaultdict(list)
    for group in groups:
        strata[(group["label"], group["protection"])].append(group)
    rng = random.Random(seed)
    subset: List[dict] = []
    for rows in strata.values():
        rows = list(rows)
        rng.shuffle(rows)
        keep = max(1, int(round(len(rows) * fraction)))
        subset.extend(rows[:keep])
    subset.sort(key=lambda group: group["temporal_group_id"])
    return subset


def aggregate_metric_rows(rows: List[Dict[str, float]], key: str) -> tuple[float, float]:
    values = [float(row[key]) for row in rows]
    return mean(values), pstdev(values) if len(values) > 1 else 0.0


def build_order_control(calib_groups: List[dict], heldout_groups: List[dict]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for seed in SEEDS:
        modes = {
            "ordered": clone_with_order(heldout_groups, "ordered"),
            "reversed inference": clone_with_order(heldout_groups, "reversed"),
            "random inference": clone_with_order(heldout_groups, "random", seed=20260426 + seed),
        }
        _, eval_scores = fit_logreg(calib_groups, modes, seed)
        for mode_name, groups in modes.items():
            metrics = compute_metrics(groups, eval_scores[mode_name], CI_SAFE_THRESHOLD)
            rows.append({"seed": seed, "temporal_input": mode_name, **metrics})
    output: List[Dict[str, object]] = []
    for mode_name in ["ordered", "reversed inference", "random inference"]:
        mode_rows = [row for row in rows if row["temporal_input"] == mode_name]
        ba, ba_std = aggregate_metric_rows(mode_rows, "balanced_accuracy")
        fpr, _ = aggregate_metric_rows(mode_rows, "protected_real_fpr")
        pf, _ = aggregate_metric_rows(mode_rows, "protected_fake_recall")
        uf, _ = aggregate_metric_rows(mode_rows, "unprotected_fake_recall")
        fp_mean, _ = aggregate_metric_rows(mode_rows, "protected_real_fp")
        total = int(mode_rows[0]["protected_real_total"])
        output.append(
            {
                "temporal_input": mode_name,
                "seeds": ",".join(str(seed) for seed in SEEDS),
                "ba_mean": f"{ba:.6f}",
                "ba_std": f"{ba_std:.6f}",
                "protected_real_fpr": f"{fpr:.6f}",
                "wilson_upper": f"{wilson_upper(round(fp_mean), total):.6f}",
                "protected_fake_recall": f"{pf:.6f}",
                "unprotected_fake_recall": f"{uf:.6f}",
                "reading": "frozen ordered model" if mode_name == "ordered" else "same frozen model with perturbed frame order",
            }
        )
    return output


def build_calibration_fraction(calib_groups: List[dict], heldout_groups: List[dict]) -> List[Dict[str, object]]:
    raw_rows: List[Dict[str, float]] = []
    for fraction in [0.25, 0.50, 0.75, 1.00]:
        for seed in CALIBRATION_SEEDS:
            subset = stratified_subset(calib_groups, fraction, seed)
            calib_scores, eval_scores = fit_logreg(subset, {"heldout": heldout_groups}, seed)
            threshold = select_threshold(subset, calib_scores)
            metrics = compute_metrics(heldout_groups, eval_scores["heldout"], threshold)
            raw_rows.append({"fraction": fraction, "seed": seed, "threshold": threshold, **metrics})
    output: List[Dict[str, object]] = []
    for fraction in [0.25, 0.50, 0.75, 1.00]:
        rows = [row for row in raw_rows if row["fraction"] == fraction]
        ba, ba_std = aggregate_metric_rows(rows, "balanced_accuracy")
        fpr, _ = aggregate_metric_rows(rows, "protected_real_fpr")
        pf, _ = aggregate_metric_rows(rows, "protected_fake_recall")
        uf, _ = aggregate_metric_rows(rows, "unprotected_fake_recall")
        threshold_mean, threshold_std = aggregate_metric_rows(rows, "threshold")
        fp_mean, _ = aggregate_metric_rows(rows, "protected_real_fp")
        total = int(rows[0]["protected_real_total"])
        output.append(
            {
                "calibration_fraction": f"{fraction:.2f}",
                "seeds": ",".join(str(seed) for seed in CALIBRATION_SEEDS),
                "threshold_mean": f"{threshold_mean:.6f}",
                "threshold_std": f"{threshold_std:.6f}",
                "ba_mean": f"{ba:.6f}",
                "ba_std": f"{ba_std:.6f}",
                "protected_real_fpr": f"{fpr:.6f}",
                "wilson_upper": f"{wilson_upper(round(fp_mean), total):.6f}",
                "protected_fake_recall": f"{pf:.6f}",
                "unprotected_fake_recall": f"{uf:.6f}",
            }
        )
    return output


def build_error_decomposition(
    split_name: str,
    calib_manifest: Path,
    heldout_manifest: Path,
    calib_passive: Path,
    heldout_passive: Path,
    threshold: float,
) -> List[Dict[str, object]]:
    calib_groups = group_rows(calib_manifest, calib_passive)
    heldout_groups = group_rows(heldout_manifest, heldout_passive)
    _, eval_scores = fit_logreg(calib_groups, {"heldout": heldout_groups}, 7)
    scores = eval_scores["heldout"]
    y = labels(heldout_groups)
    mask = masks(heldout_groups)
    pred_fake = scores >= threshold
    error_masks = {
        "PR false positives": (y == 0) & mask["protected_real"] & pred_fake,
        "PF misses": (y == 1) & mask["protected_fake"] & (~pred_fake),
        "UF misses": (y == 1) & mask["unprotected_fake"] & (~pred_fake),
    }
    rows: List[Dict[str, object]] = []
    stems = np.asarray([group["target_stem"] for group in heldout_groups])
    degradations = np.asarray([group["degradation_type"] for group in heldout_groups])
    for error_name, err_mask in error_masks.items():
        total = int(err_mask.sum())
        counter = Counter(stems[err_mask])
        for rank, (stem, count) in enumerate(counter.most_common(5), start=1):
            local_degs = Counter(degradations[err_mask & (stems == stem)])
            rows.append(
                {
                    "split": split_name,
                    "error_type": error_name,
                    "rank": rank,
                    "target_stem": stem,
                    "count": count,
                    "total_errors": total,
                    "share": f"{(count / total) if total else 0.0:.3f}",
                    "top_degradation": "; ".join(f"{key}:{value}" for key, value in local_degs.most_common(3)),
                }
            )
    return rows


def build_temporal_table() -> List[Dict[str, object]]:
    ablation_rows = load_csv(REPORTS / "vcf_temporal_ablation_rows_2026-04-22.tsv", delimiter="\t")
    neural_rows = load_csv(REPORTS / "vcf_c23_temporal_f3_neural_heads_2026-04-24.tsv", delimiter="\t")
    order_rows = load_csv(OUT_ORDER, delimiter="\t")
    rows: List[Dict[str, object]] = []

    selected = [
        ("scalar_temporal_median", "Median scalar aggregation", "no", "no", "yes"),
        ("meanstd_feature_pr4", "Mean/std feature fusion", "no", "yes", "yes"),
        ("delta_feature_pr4", "Delta feature fusion", "partial", "yes", "yes"),
        ("sequence_pr4_5seed_ci_safe_main", "Ordered sequence + PR4", "yes", "yes", "yes"),
    ]
    by_name = {row["row"]: row for row in ablation_rows}
    for key, method, ordered, frame_features, score_stats in selected:
        row = by_name[key]
        fpr = float(row["protected_real_fpr"])
        k = round(fpr * 909)
        rows.append(
            {
                "method": method,
                "ordered": ordered,
                "frame_features": frame_features,
                "score_stats": score_stats,
                "ba": f"{float(row['heldout_ba']):.4f}",
                "pr_fpr": f"{fpr:.4f}",
                "wilson_upper": f"{wilson_upper(k, 909):.4f}",
                "pf_recall": f"{float(row['protected_fake_recall']):.4f}",
                "uf_recall": f"{float(row['unprotected_fake_recall']):.4f}",
                "reading": row["reading"],
            }
        )
    for row in order_rows:
        if row["temporal_input"] == "ordered":
            continue
        rows.append(
            {
                "method": row["temporal_input"],
                "ordered": "perturbed",
                "frame_features": "yes",
                "score_stats": "yes",
                "ba": f"{float(row['ba_mean']):.4f}",
                "pr_fpr": f"{float(row['protected_real_fpr']):.4f}",
                "wilson_upper": f"{float(row['wilson_upper']):.4f}",
                "pf_recall": f"{float(row['protected_fake_recall']):.4f}",
                "uf_recall": f"{float(row['unprotected_fake_recall']):.4f}",
                "reading": row["reading"],
            }
        )
    for row in neural_rows:
        rows.append(
            {
                "method": f"{row['head'].upper()} temporal head",
                "ordered": "yes",
                "frame_features": "yes",
                "score_stats": "no",
                "ba": f"{float(row['heldout_ba_mean']):.4f}",
                "pr_fpr": f"{float(row['protected_real_fpr_mean']):.4f}",
                "wilson_upper": "--",
                "pf_recall": f"{float(row['protected_fake_recall_mean']):.4f}",
                "uf_recall": f"{float(row['unprotected_fake_recall_mean']):.4f}",
                "reading": "higher-capacity head; fake recall remains high but protected-real FPR exceeds FPR08",
            }
        )
    return rows


def main() -> None:
    calib_groups = group_rows(MAIN_CALIB_MANIFEST, MAIN_CALIB_PASSIVE)
    heldout_groups = group_rows(MAIN_HELDOUT_MANIFEST, MAIN_HELDOUT_PASSIVE)
    if len(calib_groups) != 6740 or len(heldout_groups) != 6060:
        raise RuntimeError(f"Unexpected group counts: calib={len(calib_groups)} heldout={len(heldout_groups)}")

    if OUT_ORDER.exists():
        order_rows = read_tsv(OUT_ORDER)
    else:
        order_rows = build_order_control(calib_groups, heldout_groups)
        write_tsv(OUT_ORDER, order_rows)
    calib_rows = build_calibration_fraction(calib_groups, heldout_groups)
    write_tsv(OUT_CALIB, calib_rows)
    temporal_rows = build_temporal_table()
    write_tsv(OUT_TEMPORAL, temporal_rows)
    error_rows = []
    error_rows.extend(
        build_error_decomposition(
            "main VCF",
            MAIN_CALIB_MANIFEST,
            MAIN_HELDOUT_MANIFEST,
            MAIN_CALIB_PASSIVE,
            MAIN_HELDOUT_PASSIVE,
            CI_SAFE_THRESHOLD,
        )
    )
    error_rows.extend(
        build_error_decomposition(
            "target-stem-disjoint",
            STEM_CALIB_MANIFEST,
            STEM_HELDOUT_MANIFEST,
            STEM_CALIB_PASSIVE,
            STEM_HELDOUT_PASSIVE,
            TARGETSTEM_THRESHOLD,
        )
    )
    write_tsv(OUT_ERROR, error_rows)

    summary = {
        "calib_groups": len(calib_groups),
        "heldout_groups": len(heldout_groups),
        "outputs": {
            "source_family_error_decomposition": str(OUT_ERROR.relative_to(ROOT)),
            "temporal_ablation_review_table": str(OUT_TEMPORAL.relative_to(ROOT)),
            "temporal_order_control": str(OUT_ORDER.relative_to(ROOT)),
            "calibration_fraction_sensitivity": str(OUT_CALIB.relative_to(ROOT)),
        },
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
