from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List


import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the first reliability-aware routing sanity pass on a merged hybrid bridge CSV.")
    parser.add_argument("--bridge-csv", required=True, help="Merged bridge CSV with provenance and passive outputs.")
    parser.add_argument("--rules", required=True, help="Routing rules YAML.")
    parser.add_argument("--output-csv", required=True, help="Per-sample routing output CSV.")
    parser.add_argument("--summary-json", required=True, help="Summary JSON output path.")
    parser.add_argument("--backbone-name", required=True, help="User-facing provenance backbone name.")
    return parser.parse_args()


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_rules(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def safe_float(value: str) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def transfer_override_match(
    *,
    base_prob: float | None,
    passive_prob: float | None,
    passive_conf_gap: float | None,
    passive_delta: float | None,
    passive_delta_abs: float | None,
    bit_accuracy: float | None,
    prov_watermark_l1_mean: float | None,
    passive_feature_norm: float | None,
    route_label: str,
    cfg: dict,
) -> bool:
    allowed_route_labels = set(cfg.get("allowed_route_labels", ["trust", "suspicious", "abstain"]))
    min_baseline_prob = safe_float(cfg.get("min_unprotected_pair_fake_prob"))
    max_baseline_prob = safe_float(cfg.get("max_unprotected_pair_fake_prob"))
    min_protected_prob = safe_float(cfg.get("min_protected_passive_prob"))
    max_protected_prob = safe_float(cfg.get("max_protected_passive_prob"))
    min_bit_accuracy = safe_float(cfg.get("min_bit_accuracy"))
    max_bit_accuracy = safe_float(cfg.get("max_bit_accuracy"))
    min_passive_delta = safe_float(cfg.get("min_passive_delta"))
    max_passive_delta = safe_float(cfg.get("max_passive_delta"))
    min_passive_delta_abs = safe_float(cfg.get("min_passive_delta_abs"))
    max_passive_delta_abs = safe_float(cfg.get("max_passive_delta_abs"))
    min_passive_conf_gap = safe_float(cfg.get("min_passive_conf_gap"))
    max_passive_conf_gap = safe_float(cfg.get("max_passive_conf_gap"))
    min_prov_watermark_l1_mean = safe_float(cfg.get("min_prov_watermark_l1_mean"))
    max_prov_watermark_l1_mean = safe_float(cfg.get("max_prov_watermark_l1_mean"))
    min_passive_feature_norm = safe_float(cfg.get("min_passive_feature_norm"))
    max_passive_feature_norm = safe_float(cfg.get("max_passive_feature_norm"))

    checks = [
        base_prob is not None,
        route_label in allowed_route_labels,
    ]
    if min_baseline_prob is not None:
        checks.append(base_prob is not None and base_prob >= min_baseline_prob)
    if max_baseline_prob is not None:
        checks.append(base_prob is not None and base_prob <= max_baseline_prob)
    if min_protected_prob is not None:
        checks.append(passive_prob is not None and passive_prob >= min_protected_prob)
    if max_protected_prob is not None:
        checks.append(passive_prob is not None and passive_prob <= max_protected_prob)
    if min_bit_accuracy is not None:
        checks.append(bit_accuracy is not None and bit_accuracy >= min_bit_accuracy)
    if max_bit_accuracy is not None:
        checks.append(bit_accuracy is not None and bit_accuracy <= max_bit_accuracy)
    if min_passive_delta is not None:
        checks.append(passive_delta is not None and passive_delta >= min_passive_delta)
    if max_passive_delta is not None:
        checks.append(passive_delta is not None and passive_delta <= max_passive_delta)
    if min_passive_delta_abs is not None:
        checks.append(passive_delta_abs is not None and passive_delta_abs >= min_passive_delta_abs)
    if max_passive_delta_abs is not None:
        checks.append(passive_delta_abs is not None and passive_delta_abs <= max_passive_delta_abs)
    if min_passive_conf_gap is not None:
        checks.append(passive_conf_gap is not None and passive_conf_gap >= min_passive_conf_gap)
    if max_passive_conf_gap is not None:
        checks.append(passive_conf_gap is not None and passive_conf_gap <= max_passive_conf_gap)
    if min_prov_watermark_l1_mean is not None:
        checks.append(prov_watermark_l1_mean is not None and prov_watermark_l1_mean >= min_prov_watermark_l1_mean)
    if max_prov_watermark_l1_mean is not None:
        checks.append(prov_watermark_l1_mean is not None and prov_watermark_l1_mean <= max_prov_watermark_l1_mean)
    if min_passive_feature_norm is not None:
        checks.append(passive_feature_norm is not None and passive_feature_norm >= min_passive_feature_norm)
    if max_passive_feature_norm is not None:
        checks.append(passive_feature_norm is not None and passive_feature_norm <= max_passive_feature_norm)
    return all(checks)


def compute_unprotected_map(rows: List[Dict[str, str]]) -> Dict[str, float]:
    mapping: Dict[str, float] = {}
    for row in rows:
        if row.get("protection") == "unprotected" and row.get("passive_status") == "ok":
            prob = safe_float(row.get("passive_prob_fake", ""))
            if prob is not None:
                mapping[row["pair_group_id"]] = prob
    return mapping


def route_row(
    row: Dict[str, str],
    unprotected_map: Dict[str, float],
    rules: dict,
) -> Dict[str, str]:
    trust_cfg = rules["route_labels"]["trust"]
    suspicious_cfg = rules["route_labels"]["suspicious"]
    fallback_cfg = rules["fallback"]

    protection = row.get("protection", "")
    label = row.get("label", "")
    passive_prob = safe_float(row.get("passive_prob_fake", ""))
    base_fake_threshold = float(fallback_cfg["passive_fake_threshold"])
    protected_fake_threshold = float(fallback_cfg.get("protected_passive_fake_threshold", base_fake_threshold))
    passive_fake_threshold = protected_fake_threshold if protection == "protected" else base_fake_threshold
    passive_pred = "fake" if passive_prob is not None and passive_prob >= passive_fake_threshold else "real"
    passive_conf_gap = abs(passive_prob - 0.5) if passive_prob is not None else None

    base_prob = unprotected_map.get(row.get("pair_group_id", ""))
    passive_delta = None if passive_prob is None or base_prob is None else passive_prob - base_prob
    passive_delta_abs = None if passive_delta is None else abs(passive_delta)

    bit_accuracy = safe_float(row.get("prov_bit_accuracy", ""))
    bit_error_rate = None if bit_accuracy is None else 1.0 - bit_accuracy
    prov_watermark_l1_mean = safe_float(row.get("prov_watermark_l1_mean", ""))
    passive_feature_norm = safe_float(row.get("passive_feature_norm", ""))
    prov_status = row.get("prov_status", "")

    route_label = "abstain"
    route_reason = "unprotected_or_missing"
    if protection == "protected" and prov_status == "ok" and bit_accuracy is not None and passive_delta_abs is not None:
        if bit_accuracy >= float(trust_cfg["min_bit_accuracy"]) and passive_delta_abs <= float(trust_cfg["max_passive_delta_abs"]):
            route_label = "trust"
            route_reason = "strong_provenance_and_low_passive_shift"
        elif bit_accuracy >= float(suspicious_cfg["min_bit_accuracy"]) and passive_delta_abs <= float(suspicious_cfg["max_passive_delta_abs"]):
            route_label = "suspicious"
            route_reason = "usable_provenance_but_not_clean_enough_for_trust"
        else:
            route_label = "abstain"
            route_reason = "weak_provenance_or_large_passive_shift"

    transfer_cfg = rules.get("protected_transfer") or {}
    transfer_patterns = rules.get("protected_transfer_patterns") or []
    real_guard_cfg = rules.get("protected_real_guard") or {}
    real_guard_patterns = rules.get("protected_real_guard_patterns") or []
    real_tail_cfg = rules.get("protected_real_tail_calibration") or {}
    real_tail_patterns = rules.get("protected_real_tail_calibration_patterns") or []
    transfer_override = False
    transfer_reason = ""
    transfer_pattern_name = ""
    passive_threshold_source = "default"
    anomaly_threshold_source = "default"
    real_guard_override = False
    real_guard_reason = ""
    real_guard_pattern_name = ""
    real_tail_override = False
    real_tail_reason = ""
    real_tail_pattern_name = ""

    applied_transfer_cfg = None
    if protection == "protected":
        transfer_candidates = []
        for index, pattern_cfg in enumerate(transfer_patterns, start=1):
            pattern_name = str(pattern_cfg.get("name", f"pattern_{index}"))
            transfer_candidates.append((f"protected_transfer_pattern:{pattern_name}", pattern_name, pattern_cfg))
        if transfer_cfg:
            transfer_candidates.append(("protected_pair_transfer_override", "", transfer_cfg))

        for reason, pattern_name, candidate_cfg in transfer_candidates:
            if transfer_override_match(
                base_prob=base_prob,
                passive_prob=passive_prob,
                passive_conf_gap=passive_conf_gap,
                passive_delta=passive_delta,
                passive_delta_abs=passive_delta_abs,
                bit_accuracy=bit_accuracy,
                prov_watermark_l1_mean=prov_watermark_l1_mean,
                passive_feature_norm=passive_feature_norm,
                route_label=route_label,
                cfg=candidate_cfg,
            ):
                transfer_override = True
                transfer_reason = reason
                transfer_pattern_name = pattern_name
                applied_transfer_cfg = candidate_cfg
                override_fake_threshold = safe_float(candidate_cfg.get("protected_passive_fake_threshold"))
                if override_fake_threshold is not None:
                    passive_fake_threshold = override_fake_threshold
                    passive_pred = "fake" if passive_prob is not None and passive_prob >= passive_fake_threshold else "real"
                    passive_threshold_source = "protected_transfer"
                break

    anomaly_threshold = fallback_cfg.get("protected_delta_abs_anomaly_threshold")
    if transfer_override and applied_transfer_cfg and "protected_delta_abs_anomaly_threshold" in applied_transfer_cfg:
        anomaly_threshold = applied_transfer_cfg.get("protected_delta_abs_anomaly_threshold")
        anomaly_threshold_source = "protected_transfer"
    anomaly_flag = (
        protection == "protected"
        and passive_delta_abs is not None
        and anomaly_threshold is not None
        and passive_delta_abs >= float(anomaly_threshold)
    )

    if protection == "protected" and not anomaly_flag and passive_pred == "fake":
        real_guard_candidates = []
        for index, pattern_cfg in enumerate(real_guard_patterns, start=1):
            pattern_name = str(pattern_cfg.get("name", f"pattern_{index}"))
            real_guard_candidates.append((f"protected_real_guard_pattern:{pattern_name}", pattern_name, pattern_cfg))
        if real_guard_cfg:
            real_guard_candidates.append(("protected_real_guard_override", "", real_guard_cfg))

        for reason, pattern_name, candidate_cfg in real_guard_candidates:
            if transfer_override_match(
                base_prob=base_prob,
                passive_prob=passive_prob,
                passive_conf_gap=passive_conf_gap,
                passive_delta=passive_delta,
                passive_delta_abs=passive_delta_abs,
                bit_accuracy=bit_accuracy,
                prov_watermark_l1_mean=prov_watermark_l1_mean,
                passive_feature_norm=passive_feature_norm,
                route_label=route_label,
                cfg=candidate_cfg,
            ):
                real_guard_override = True
                real_guard_reason = reason
                real_guard_pattern_name = pattern_name
                break

    if anomaly_flag:
        final_decision = "delta_anomaly_flag_fake"
        effective_pred = "fake"
    elif real_guard_override:
        final_decision = "protected_real_guard_accept_real"
        effective_pred = "real"
    elif route_label == "trust":
        final_decision = "trust_and_flag_fake" if passive_pred == "fake" else "trust_and_accept_real"
        effective_pred = passive_pred
    elif route_label == "suspicious":
        final_decision = "flag_fake_review_provenance" if passive_pred == "fake" else "review_protected_real"
        effective_pred = passive_pred
    else:
        if passive_conf_gap is not None and passive_conf_gap >= float(fallback_cfg["min_passive_confidence_gap"]):
            final_decision = "fallback_to_passive_fake" if passive_pred == "fake" else "fallback_to_passive_real"
            effective_pred = passive_pred
        else:
            final_decision = "manual_review"
            effective_pred = passive_pred

    if protection == "protected" and effective_pred == "fake":
        real_tail_candidates = []
        for index, pattern_cfg in enumerate(real_tail_patterns, start=1):
            pattern_name = str(pattern_cfg.get("name", f"pattern_{index}"))
            real_tail_candidates.append((f"protected_real_tail_pattern:{pattern_name}", pattern_name, pattern_cfg))
        if real_tail_cfg:
            real_tail_candidates.append(("protected_real_tail_calibration", "", real_tail_cfg))

        for reason, pattern_name, candidate_cfg in real_tail_candidates:
            allowed_decisions = set(candidate_cfg.get("allowed_final_decisions", []))
            if allowed_decisions and final_decision not in allowed_decisions:
                continue
            if transfer_override_match(
                base_prob=base_prob,
                passive_prob=passive_prob,
                passive_conf_gap=passive_conf_gap,
                passive_delta=passive_delta,
                passive_delta_abs=passive_delta_abs,
                bit_accuracy=bit_accuracy,
                prov_watermark_l1_mean=prov_watermark_l1_mean,
                passive_feature_norm=passive_feature_norm,
                route_label=route_label,
                cfg=candidate_cfg,
            ):
                real_tail_override = True
                real_tail_reason = reason
                real_tail_pattern_name = pattern_name
                final_decision = "protected_real_tail_accept_real"
                effective_pred = "real"
                break

    correctness = ""
    if effective_pred in {"real", "fake"} and label in {"real", "fake"}:
        correctness = "1" if effective_pred == label else "0"

    return {
        **row,
        "route_label": route_label,
        "route_reason": route_reason,
        "passive_baseline_prob": "" if base_prob is None else f"{base_prob:.6f}",
        "passive_delta": "" if passive_delta is None else f"{passive_delta:.6f}",
        "passive_delta_abs": "" if passive_delta_abs is None else f"{passive_delta_abs:.6f}",
        "bit_error_rate": "" if bit_error_rate is None else f"{bit_error_rate:.6f}",
        "passive_pred_label": passive_pred,
        "passive_fake_threshold": f"{passive_fake_threshold:.6f}",
        "passive_conf_gap": "" if passive_conf_gap is None else f"{passive_conf_gap:.6f}",
        "protected_transfer_override_flag": "1" if transfer_override else "0",
        "protected_transfer_override_reason": transfer_reason,
        "protected_transfer_pattern": transfer_pattern_name,
        "protected_real_guard_flag": "1" if real_guard_override else "0",
        "protected_real_guard_reason": real_guard_reason,
        "protected_real_guard_pattern": real_guard_pattern_name,
        "protected_real_tail_calibration_flag": "1" if real_tail_override else "0",
        "protected_real_tail_calibration_reason": real_tail_reason,
        "protected_real_tail_calibration_pattern": real_tail_pattern_name,
        "passive_threshold_source": passive_threshold_source,
        "anomaly_threshold_source": anomaly_threshold_source,
        "anomaly_flag": "1" if anomaly_flag else "0",
        "final_decision": final_decision,
        "effective_pred_label": effective_pred,
        "effective_correct": correctness,
    }


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Dict[str, str]], backbone_name: str) -> dict:
    protected_rows = [row for row in rows if row.get("protection") == "protected"]
    route_counts = Counter(row["route_label"] for row in rows)
    decision_counts = Counter(row["final_decision"] for row in rows)

    trust_by_deg = defaultdict(int)
    protected_by_deg = defaultdict(int)
    for row in protected_rows:
        deg = row["degradation_type"]
        protected_by_deg[deg] += 1
        if row["route_label"] == "trust":
            trust_by_deg[deg] += 1

    accuracy_rows = [row for row in rows if row.get("effective_correct") in {"0", "1"}]
    effective_accuracy = (
        sum(int(row["effective_correct"]) for row in accuracy_rows) / len(accuracy_rows)
        if accuracy_rows
        else None
    )

    return {
        "backbone_name": backbone_name,
        "row_count": len(rows),
        "protected_row_count": len(protected_rows),
        "route_counts": dict(route_counts),
        "decision_counts": dict(decision_counts),
        "anomaly_count": sum(1 for row in rows if row.get("anomaly_flag") == "1"),
        "protected_real_tail_calibration_count": sum(
            1 for row in rows if row.get("protected_real_tail_calibration_flag") == "1"
        ),
        "trust_rate_by_degradation": {
            key: 0.0 if protected_by_deg[key] == 0 else trust_by_deg[key] / protected_by_deg[key]
            for key in sorted(protected_by_deg)
        },
        "effective_accuracy": effective_accuracy,
    }


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    bridge_rows = load_csv(Path(args.bridge_csv))
    rules = load_rules(Path(args.rules))
    unprotected_map = compute_unprotected_map(bridge_rows)
    routed_rows = [route_row(row, unprotected_map, rules) for row in bridge_rows]
    write_csv(Path(args.output_csv), routed_rows)
    summary = summarize(routed_rows, args.backbone_name)
    write_json(Path(args.summary_json), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
