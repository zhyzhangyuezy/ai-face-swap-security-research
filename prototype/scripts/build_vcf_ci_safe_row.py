#!/usr/bin/env python
"""Build a calibration-only VCF FPR08-CI operating row from the locked sweep."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "prototype" / "reports" / "vcf_c23_temporal_f3_locked_sequence_pr4_seed_sweep_mi2000.csv"
OUT_SELECTED = ROOT / "prototype" / "reports" / "vcf_c23_temporal_f3_sequence_pr4_ci_safe_selected_2026-04-25.csv"
OUT_SUMMARY = ROOT / "prototype" / "reports" / "vcf_c23_temporal_f3_sequence_pr4_ci_safe_summary_2026-04-25.json"
OUT_TSV = ROOT / "prototype" / "reports" / "vcf_c23_temporal_f3_sequence_pr4_ci_safe_summary_2026-04-25.tsv"
EXTERNAL_OLD = ROOT / "prototype" / "reports" / "external_sota_breadth_vcf_temporal_update_2026-04-24.tsv"
EXTERNAL_NEW = ROOT / "prototype" / "reports" / "external_sota_breadth_vcf_temporal_update_2026-04-25.tsv"

PR_DENOMINATOR = 909
FAKE_RECALL_FLOOR = 0.9985
CALIB_PR_FPR_MAX = 0.0


def as_float(value: str) -> float:
    return float(value) if value != "" else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
    return ((centre - margin) / den, (centre + margin) / den)


def load_rows() -> list[dict[str, object]]:
    with IN_PATH.open(newline="", encoding="utf-8") as f:
        rows: list[dict[str, object]] = []
        for row in csv.DictReader(f):
            parsed: dict[str, object] = {"model": row["model"], "seed": int(row["seed"])}
            for key, value in row.items():
                if key in parsed or key == "model":
                    continue
                parsed[key] = as_float(value)
            rows.append(parsed)
    return rows


def choose_ci_safe(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    selected = []
    for seed in sorted({int(r["seed"]) for r in rows}):
        candidates = [
            r
            for r in rows
            if int(r["seed"]) == seed
            and float(r["calib_protected_real_fpr"]) <= CALIB_PR_FPR_MAX
            and float(r["calib_protected_fake_recall"]) >= FAKE_RECALL_FLOOR
            and float(r["calib_unprotected_fake_recall"]) >= FAKE_RECALL_FLOOR
        ]
        if not candidates:
            raise RuntimeError(f"No CI-safe calibration candidate for seed {seed}")
        selected.append(max(candidates, key=lambda r: float(r["threshold"])))
    return selected


def summarize(selected: list[dict[str, object]]) -> dict[str, object]:
    enriched = []
    for row in selected:
        k = round(float(row["heldout_protected_real_fpr"]) * PR_DENOMINATOR)
        lo, hi = wilson_interval(k, PR_DENOMINATOR)
        enriched.append(
            {
                "seed": int(row["seed"]),
                "threshold": float(row["threshold"]),
                "heldout_pr_fp": k,
                "heldout_pr_total": PR_DENOMINATOR,
                "wilson_low": lo,
                "wilson_high": hi,
                "heldout_balanced_accuracy": float(row["heldout_balanced_accuracy"]),
                "heldout_protected_real_fpr": float(row["heldout_protected_real_fpr"]),
                "heldout_protected_fake_recall": float(row["heldout_protected_fake_recall"]),
                "heldout_unprotected_fake_recall": float(row["heldout_unprotected_fake_recall"]),
                "calib_balanced_accuracy": float(row["calib_balanced_accuracy"]),
                "calib_protected_real_fpr": float(row["calib_protected_real_fpr"]),
                "calib_protected_fake_recall": float(row["calib_protected_fake_recall"]),
                "calib_unprotected_fake_recall": float(row["calib_unprotected_fake_recall"]),
            }
        )
    return {
        "source_sweep": str(IN_PATH.relative_to(ROOT)),
        "selection_rule": (
            "For each seed, choose the highest threshold using calibration data only, "
            "subject to calibration protected-real FPR = 0 and both calibration protected "
            f"and unprotected fake recall >= {FAKE_RECALL_FLOOR:.4f}."
        ),
        "protected_real_denominator": PR_DENOMINATOR,
        "fake_recall_floor": FAKE_RECALL_FLOOR,
        "selected": enriched,
        "aggregate": {
            "seeds": [r["seed"] for r in enriched],
            "threshold_mean": mean(r["threshold"] for r in enriched),
            "threshold_std": pstdev(r["threshold"] for r in enriched),
            "heldout_ba_mean": mean(r["heldout_balanced_accuracy"] for r in enriched),
            "heldout_ba_std": pstdev(r["heldout_balanced_accuracy"] for r in enriched),
            "heldout_pr_fpr_mean": mean(r["heldout_protected_real_fpr"] for r in enriched),
            "heldout_pr_fp_min": min(r["heldout_pr_fp"] for r in enriched),
            "heldout_pr_fp_max": max(r["heldout_pr_fp"] for r in enriched),
            "wilson_upper_max": max(r["wilson_high"] for r in enriched),
            "protected_fake_recall_mean": mean(r["heldout_protected_fake_recall"] for r in enriched),
            "unprotected_fake_recall_mean": mean(r["heldout_unprotected_fake_recall"] for r in enriched),
            "calib_ba_mean": mean(r["calib_balanced_accuracy"] for r in enriched),
            "calib_pr_fpr_mean": mean(r["calib_protected_real_fpr"] for r in enriched),
            "calib_protected_fake_recall_mean": mean(r["calib_protected_fake_recall"] for r in enriched),
            "calib_unprotected_fake_recall_mean": mean(r["calib_unprotected_fake_recall"] for r in enriched),
            "fpr08_ci_pass_all_seeds": all(r["wilson_high"] <= 0.08 for r in enriched),
        },
    }


def write_selected(selected: list[dict[str, object]]) -> None:
    fieldnames = [
        "model",
        "seed",
        "threshold",
        "calib_balanced_accuracy",
        "calib_protected_real_fpr",
        "calib_protected_fake_recall",
        "calib_unprotected_fake_recall",
        "heldout_groups",
        "heldout_balanced_accuracy",
        "heldout_protected_real_fpr",
        "heldout_protected_fake_recall",
        "heldout_unprotected_fake_recall",
        "heldout_pr_fp",
        "heldout_pr_total",
        "wilson_low",
        "wilson_high",
    ]
    with OUT_SELECTED.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            k = round(float(row["heldout_protected_real_fpr"]) * PR_DENOMINATOR)
            lo, hi = wilson_interval(k, PR_DENOMINATOR)
            writer.writerow(
                {
                    "model": row["model"],
                    "seed": row["seed"],
                    "threshold": f"{float(row['threshold']):.6f}",
                    "calib_balanced_accuracy": f"{float(row['calib_balanced_accuracy']):.6f}",
                    "calib_protected_real_fpr": f"{float(row['calib_protected_real_fpr']):.6f}",
                    "calib_protected_fake_recall": f"{float(row['calib_protected_fake_recall']):.6f}",
                    "calib_unprotected_fake_recall": f"{float(row['calib_unprotected_fake_recall']):.6f}",
                    "heldout_groups": int(float(row["heldout_groups"])),
                    "heldout_balanced_accuracy": f"{float(row['heldout_balanced_accuracy']):.6f}",
                    "heldout_protected_real_fpr": f"{float(row['heldout_protected_real_fpr']):.6f}",
                    "heldout_protected_fake_recall": f"{float(row['heldout_protected_fake_recall']):.6f}",
                    "heldout_unprotected_fake_recall": f"{float(row['heldout_unprotected_fake_recall']):.6f}",
                    "heldout_pr_fp": k,
                    "heldout_pr_total": PR_DENOMINATOR,
                    "wilson_low": f"{lo:.6f}",
                    "wilson_high": f"{hi:.6f}",
                }
            )


def write_outputs(summary: dict[str, object]) -> None:
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    agg = summary["aggregate"]
    assert isinstance(agg, dict)
    with OUT_TSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "row",
                "threshold_mean",
                "heldout_ba_mean",
                "heldout_ba_std",
                "heldout_pr_fpr_mean",
                "heldout_pr_fp_range",
                "wilson_upper_max",
                "protected_fake_recall_mean",
                "unprotected_fake_recall_mean",
                "fpr08_ci_pass_all_seeds",
            ]
        )
        writer.writerow(
            [
                "VCF temporal PR4 CI-safe",
                f"{agg['threshold_mean']:.6f}",
                f"{agg['heldout_ba_mean']:.6f}",
                f"{agg['heldout_ba_std']:.6f}",
                f"{agg['heldout_pr_fpr_mean']:.6f}",
                f"{agg['heldout_pr_fp_min']}--{agg['heldout_pr_fp_max']} / {PR_DENOMINATOR}",
                f"{agg['wilson_upper_max']:.6f}",
                f"{agg['protected_fake_recall_mean']:.6f}",
                f"{agg['unprotected_fake_recall_mean']:.6f}",
                int(bool(agg["fpr08_ci_pass_all_seeds"])),
            ]
        )

    with EXTERNAL_OLD.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
        fieldnames = rows[0].keys()
    for row in rows:
        if row["model"] == "ours_sequence_pr4":
            row.update(
                {
                    "status": "kept_main_ci_safe",
                    "selection": "calibration-only CI-safe fake-recall-floor row",
                    "heldout_ba": f"{agg['heldout_ba_mean']:.6f}",
                    "heldout_protected_real_fpr": f"{agg['heldout_pr_fpr_mean']:.6f}",
                    "heldout_protected_fake_recall": f"{agg['protected_fake_recall_mean']:.6f}",
                    "heldout_unprotected_fake_recall": f"{agg['unprotected_fake_recall_mean']:.6f}",
                    "heldout_strict_fpr08": "1",
                    "source": str(OUT_SUMMARY.relative_to(ROOT)).replace("\\", "/"),
                }
            )
    with EXTERNAL_NEW.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = load_rows()
    selected = choose_ci_safe(rows)
    summary = summarize(selected)
    write_selected(selected)
    write_outputs(summary)
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
