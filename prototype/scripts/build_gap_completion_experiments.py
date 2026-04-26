from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import date
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[2]
PROTOTYPE = ROOT / "prototype"
REPORTS = PROTOTYPE / "reports"
MANIFESTS = PROTOTYPE / "manifests"


FPR_TARGETS = [0.02, 0.05, 0.08, 0.10, 0.15]
SWEEP_PATH = REPORTS / "vcf_c23_temporal_f3_locked_sequence_pr4_seed_sweep_mi2000.csv"
CALIB_MANIFEST = MANIFESTS / "vcf_c23_calib_temporal_f3_seed20260421_sw010_grouped.csv"
HELDOUT_MANIFEST = MANIFESTS / "vcf_c23_heldout_temporal_f3_seed20260421_sw010_grouped.csv"

PROTECTION_RUNS = [
    ("UADFV", "VideoSeal", REPORTS / "uadfv_smoke_v0_1_routing_sanity.csv"),
    ("UADFV", "PixelSeal", REPORTS / "uadfv_pixelseal_smoke_v0_1_routing_sanity.csv"),
    ("Celeb-DF-v1", "VideoSeal", REPORTS / "celebdf_smoke_v0_1_routing_sanity.csv"),
    ("Celeb-DF-v1", "PixelSeal", REPORTS / "celebdf_pixelseal_smoke_v0_1_routing_sanity.csv"),
]


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
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


def parse_notes(notes: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for part in notes.split(";"):
        token = part.strip()
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def f(value: str | float | int | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def wilson_upper(fp: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return float("nan")
    p = fp / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return min(1.0, (centre + radius) / denom)


def protected_real_group_count(manifest_path: Path) -> int:
    seen = set()
    for row in load_csv(manifest_path):
        if row.get("label") != "real" or row.get("protection") != "protected":
            continue
        notes = parse_notes(row.get("notes", ""))
        temporal_id = notes.get("temporal_sample_id") or row.get("sample_id", "")
        if temporal_id:
            seen.add(temporal_id)
    return len(seen)


def metric_mean(rows: List[Dict[str, str]], key: str) -> float:
    return mean(f(row[key]) for row in rows)


def metric_std(rows: List[Dict[str, str]], key: str) -> float:
    values = [f(row[key]) for row in rows]
    return pstdev(values) if len(values) > 1 else 0.0


def group_by_seed(rows: Iterable[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["seed"]].append(row)
    for seed_rows in grouped.values():
        seed_rows.sort(key=lambda r: f(r["threshold"]))
    return grouped


def count_from_rate(rate: float, n: int) -> int:
    return int(round(rate * n))


def selection_key(row: Dict[str, str]) -> tuple[float, float, float, float]:
    return (
        f(row["calib_balanced_accuracy"]),
        f(row["calib_protected_fake_recall"]),
        f(row["calib_unprotected_fake_recall"]),
        f(row["threshold"]),
    )


def heldout_key(row: Dict[str, str]) -> tuple[float, float, float, float]:
    return (
        f(row["heldout_balanced_accuracy"]),
        f(row["heldout_protected_fake_recall"]),
        f(row["heldout_unprotected_fake_recall"]),
        -f(row["heldout_protected_real_fpr"]),
    )


def aggregate_selected(
    *,
    strategy: str,
    target: float,
    selected: List[Dict[str, str]],
    calib_pr_n: int,
    heldout_pr_n: int,
    diagnostic: bool,
) -> Dict[str, object]:
    heldout_fps = [count_from_rate(f(row["heldout_protected_real_fpr"]), heldout_pr_n) for row in selected]
    calib_fps = [count_from_rate(f(row["calib_protected_real_fpr"]), calib_pr_n) for row in selected]
    return {
        "strategy": strategy,
        "target": f"{target:.2f}",
        "diagnostic_only": int(diagnostic),
        "seeds": ",".join(row["seed"] for row in selected),
        "threshold_mean": f"{metric_mean(selected, 'threshold'):.6f}",
        "threshold_std": f"{metric_std(selected, 'threshold'):.6f}",
        "calib_ba_mean": f"{metric_mean(selected, 'calib_balanced_accuracy'):.6f}",
        "calib_pr_fpr_mean": f"{metric_mean(selected, 'calib_protected_real_fpr'):.6f}",
        "calib_wilson_upper_mean": f"{mean(wilson_upper(fp, calib_pr_n) for fp in calib_fps):.6f}",
        "heldout_ba_mean": f"{metric_mean(selected, 'heldout_balanced_accuracy'):.6f}",
        "heldout_ba_std": f"{metric_std(selected, 'heldout_balanced_accuracy'):.6f}",
        "heldout_pr_fpr_mean": f"{metric_mean(selected, 'heldout_protected_real_fpr'):.6f}",
        "heldout_wilson_upper_mean": f"{mean(wilson_upper(fp, heldout_pr_n) for fp in heldout_fps):.6f}",
        "protected_fake_recall_mean": f"{metric_mean(selected, 'heldout_protected_fake_recall'):.6f}",
        "unprotected_fake_recall_mean": f"{metric_mean(selected, 'heldout_unprotected_fake_recall'):.6f}",
        "observed_target_pass": int(all(f(row["heldout_protected_real_fpr"]) <= target + 1e-12 for row in selected)),
        "heldout_ci_target_pass": int(all(wilson_upper(fp, heldout_pr_n) <= target + 1e-12 for fp in heldout_fps)),
    }


def build_operating_point_rows() -> tuple[List[Dict[str, object]], dict]:
    sweep_rows = load_csv(SWEEP_PATH)
    by_seed = group_by_seed(sweep_rows)
    calib_pr_n = protected_real_group_count(CALIB_MANIFEST)
    heldout_pr_n = protected_real_group_count(HELDOUT_MANIFEST)
    output: List[Dict[str, object]] = []

    for target in FPR_TARGETS:
        observed_selected = []
        ci_selected = []
        frontier_selected = []
        for seed, seed_rows in by_seed.items():
            observed_eligible = [row for row in seed_rows if f(row["calib_protected_real_fpr"]) <= target + 1e-12]
            if observed_eligible:
                observed_selected.append(max(observed_eligible, key=selection_key))

            ci_eligible = []
            for row in seed_rows:
                fp = count_from_rate(f(row["calib_protected_real_fpr"]), calib_pr_n)
                if wilson_upper(fp, calib_pr_n) <= target + 1e-12:
                    ci_eligible.append(row)
            if ci_eligible:
                ci_selected.append(max(ci_eligible, key=selection_key))

            frontier_eligible = [row for row in seed_rows if f(row["heldout_protected_real_fpr"]) <= target + 1e-12]
            if frontier_eligible:
                frontier_selected.append(max(frontier_eligible, key=heldout_key))

        if observed_selected:
            output.append(
                aggregate_selected(
                    strategy="calib_observed_fpr",
                    target=target,
                    selected=observed_selected,
                    calib_pr_n=calib_pr_n,
                    heldout_pr_n=heldout_pr_n,
                    diagnostic=False,
                )
            )
        if ci_selected:
            output.append(
                aggregate_selected(
                    strategy="calib_wilson_upper_fpr",
                    target=target,
                    selected=ci_selected,
                    calib_pr_n=calib_pr_n,
                    heldout_pr_n=heldout_pr_n,
                    diagnostic=False,
                )
            )
        if frontier_selected:
            output.append(
                aggregate_selected(
                    strategy="heldout_frontier_oracle",
                    target=target,
                    selected=frontier_selected,
                    calib_pr_n=calib_pr_n,
                    heldout_pr_n=heldout_pr_n,
                    diagnostic=True,
                )
            )

    meta = {
        "sweep_path": str(SWEEP_PATH.relative_to(ROOT)),
        "calib_manifest": str(CALIB_MANIFEST.relative_to(ROOT)),
        "heldout_manifest": str(HELDOUT_MANIFEST.relative_to(ROOT)),
        "calib_protected_real_groups": calib_pr_n,
        "heldout_protected_real_groups": heldout_pr_n,
        "targets": FPR_TARGETS,
    }
    return output, meta


def is_fake_pred(row: Dict[str, str]) -> bool:
    return row.get("effective_pred_label") == "fake"


def rate(num: int, den: int) -> float:
    return float("nan") if den == 0 else num / den


def fmt_rate(num: int, den: int) -> str:
    return "" if den == 0 else f"{num / den:.6f}"


def protection_metrics(rows: List[Dict[str, str]]) -> Dict[str, object]:
    total = len(rows)
    protected = [row for row in rows if row.get("protection") == "protected"]
    protected_real = [row for row in rows if row.get("protection") == "protected" and row.get("label") == "real"]
    protected_fake = [row for row in rows if row.get("protection") == "protected" and row.get("label") == "fake"]
    unprotected_fake = [row for row in rows if row.get("protection") == "unprotected" and row.get("label") == "fake"]
    correct = sum(1 for row in rows if row.get("effective_correct") == "1")
    protected_real_fp = sum(1 for row in protected_real if is_fake_pred(row))
    protected_fake_tp = sum(1 for row in protected_fake if is_fake_pred(row))
    unprotected_fake_tp = sum(1 for row in unprotected_fake if is_fake_pred(row))
    trust = sum(1 for row in protected if row.get("route_label") == "trust")
    review = sum(
        1
        for row in protected
        if "review" in row.get("final_decision", "") or row.get("final_decision", "") == "manual_review"
    )
    bit_values = [f(row.get("prov_bit_accuracy"), float("nan")) for row in protected if row.get("prov_bit_accuracy")]
    bit_values = [value for value in bit_values if not math.isnan(value)]
    return {
        "rows": total,
        "protected_rows": len(protected),
        "accuracy": "" if total == 0 else f"{correct / total:.6f}",
        "protected_real_fp": protected_real_fp,
        "protected_real_total": len(protected_real),
        "protected_real_fpr": fmt_rate(protected_real_fp, len(protected_real)),
        "protected_fake_tp": protected_fake_tp,
        "protected_fake_total": len(protected_fake),
        "protected_fake_recall": fmt_rate(protected_fake_tp, len(protected_fake)),
        "unprotected_fake_tp": unprotected_fake_tp,
        "unprotected_fake_total": len(unprotected_fake),
        "unprotected_fake_recall": fmt_rate(unprotected_fake_tp, len(unprotected_fake)),
        "trust_rate": fmt_rate(trust, len(protected)),
        "review_rate": fmt_rate(review, len(protected)),
        "mean_bit_accuracy": "" if not bit_values else f"{mean(bit_values):.6f}",
    }


def build_protection_rows() -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for dataset, backbone, path in PROTECTION_RUNS:
        rows = load_csv(path)
        for degradation in ["all"] + sorted({row.get("degradation_type", "") for row in rows if row.get("protection") == "protected"}):
            if degradation == "all":
                subset = rows
            else:
                subset = [row for row in rows if row.get("degradation_type") == degradation or row.get("protection") == "unprotected"]
            metrics = protection_metrics(subset)
            output.append(
                {
                    "dataset": dataset,
                    "backbone": backbone,
                    "degradation": degradation,
                    **metrics,
                    "source": str(path.relative_to(ROOT)),
                }
            )
    return output


def compact_operating_markdown(rows: List[Dict[str, object]]) -> List[str]:
    lines = [
        "| Strategy | Target | Threshold | BA | PR-FPR | Wilson upper | PF recall | UF recall | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    preferred = [
        row
        for row in rows
        if row["strategy"] in {"calib_observed_fpr", "calib_wilson_upper_fpr"}
        and row["target"] in {"0.02", "0.05", "0.08", "0.10", "0.15"}
    ]
    for row in preferred:
        label = "Calib observed" if row["strategy"] == "calib_observed_fpr" else "Calib Wilson"
        pass_text = "yes" if row["observed_target_pass"] else "no"
        lines.append(
            f"| {label} | {row['target']} | {float(row['threshold_mean']):.3f} | "
            f"{float(row['heldout_ba_mean']):.4f} | {float(row['heldout_pr_fpr_mean']):.4f} | "
            f"{float(row['heldout_wilson_upper_mean']):.4f} | {float(row['protected_fake_recall_mean']):.4f} | "
            f"{float(row['unprotected_fake_recall_mean']):.4f} | {pass_text} |"
        )
    return lines


def compact_protection_markdown(rows: List[Dict[str, object]]) -> List[str]:
    lines = [
        "| Dataset | Backbone | Deg. | Rows | PR-FPR | PF recall | UF recall | Trust | Review | Bit acc. |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["degradation"] != "all":
            continue
        lines.append(
            f"| {row['dataset']} | {row['backbone']} | all | {row['rows']} | "
            f"{row['protected_real_fpr'] or '--'} | {row['protected_fake_recall'] or '--'} | "
            f"{row['unprotected_fake_recall'] or '--'} | {row['trust_rate'] or '--'} | "
            f"{row['review_rate'] or '--'} | {row['mean_bit_accuracy'] or '--'} |"
        )
    return lines


def write_report(
    *,
    operating_rows: List[Dict[str, object]],
    protection_rows: List[Dict[str, object]],
    meta: dict,
    report_path: Path,
    json_path: Path,
) -> None:
    payload = {
        "date": date.today().isoformat(),
        "operating_point_meta": meta,
        "operating_point_rows": operating_rows,
        "protection_shift_rows": protection_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Gap-Completion Experiments",
        "",
        f"Date: {date.today().isoformat()}",
        "",
        "## 1. VCF operating-point and confidence-control diagnostic",
        "",
        "This block reuses the saved five-seed VCF PR4 threshold sweep and performs only calibration-side selection unless marked as held-out frontier. It tests whether the main FPR08 result is robust to stricter observed and Wilson-upper constraints.",
        "",
        *compact_operating_markdown(operating_rows),
        "",
        "Key reading: the locked PR4 row is stable for calibration-side observed FPR targets because calibration protected-real FPR is zero in the selected high-threshold region. However, the held-out Wilson upper bound remains above 0.08 for the main VCF row. A confidence-level FPR08 claim would require either more protected-real held-out groups or a more conservative threshold with a recall cost.",
        "",
        "## 2. Protection-pipeline shift smoke diagnostic",
        "",
        "This block compares the existing VideoSeal and PixelSeal smoke probes on UADFV and Celeb-DF-v1 with clean, JPEG-q75, and resize-half protected variants. It is intentionally reported as a smoke diagnostic, not as a main benchmark, because the row counts are small and the PixelSeal branch was not part of the final large-scale VCF protocol.",
        "",
        *compact_protection_markdown(protection_rows),
        "",
        "Key reading: UADFV remains easy for both protection backbones in this small probe, while Celeb-DF-v1 is sensitive to the protection backend. PixelSeal falls back to abstain/manual-review much more often and has lower effective accuracy. This supports adding a bounded cross-protection robustness statement rather than claiming universal provenance-system robustness.",
        "",
        "## Files",
        "",
        f"- Operating TSV: `{(REPORTS / 'vcf_fpr_confidence_operating_points_2026-04-24.tsv').relative_to(ROOT)}`",
        f"- Protection TSV: `{(REPORTS / 'protection_pipeline_shift_smoke_2026-04-24.tsv').relative_to(ROOT)}`",
        f"- JSON payload: `{json_path.relative_to(ROOT)}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    operating_rows, meta = build_operating_point_rows()
    protection_rows = build_protection_rows()

    operating_tsv = REPORTS / "vcf_fpr_confidence_operating_points_2026-04-24.tsv"
    protection_tsv = REPORTS / "protection_pipeline_shift_smoke_2026-04-24.tsv"
    report_path = REPORTS / "gap_completion_experiments_2026-04-24.zh-CN.md"
    json_path = REPORTS / "gap_completion_experiments_2026-04-24.json"

    write_csv(operating_tsv, operating_rows)
    write_csv(protection_tsv, protection_rows)
    write_report(
        operating_rows=operating_rows,
        protection_rows=protection_rows,
        meta=meta,
        report_path=report_path,
        json_path=json_path,
    )

    print(
        json.dumps(
            {
                "operating_rows": len(operating_rows),
                "protection_rows": len(protection_rows),
                "operating_tsv": str(operating_tsv.resolve()),
                "protection_tsv": str(protection_tsv.resolve()),
                "report": str(report_path.resolve()),
                "json": str(json_path.resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
