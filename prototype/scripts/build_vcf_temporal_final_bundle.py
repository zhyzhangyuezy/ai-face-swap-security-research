from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


REPORTS = Path(__file__).resolve().parents[1] / "reports"


def load_json(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def load_tsv(name: str) -> List[Dict[str, str]]:
    with (REPORTS / name).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


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
        for row in rows:
            writer.writerow({key: fmt(value) for key, value in row.items()})


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def agg(summary_name: str) -> dict:
    return load_json(summary_name)["selected_aggregate"]


def candidate_locked_row(candidates: List[Dict[str, str]], model: str) -> dict:
    rows = [
        row
        for row in candidates
        if row["score_model"] == model
        and float(row["calib_balanced_accuracy"]) >= 1.0
        and float(row["calib_protected_real_fpr"]) <= 0.0
    ]
    if not rows:
        raise RuntimeError(f"No locked candidate rows for {model}")
    return max(rows, key=lambda row: float(row["threshold"]))


def build_ablation_rows() -> List[dict]:
    scalar_summary = load_json("vcf_c23_heldout_temporal_f3_dualrealc_tailm25_fw010_realm25_video_summary.json")
    scalar_strict = scalar_summary["overall_best_fpr08"]
    feature_candidates_path = REPORTS / "vcf_c23_temporal_f3_dualrealc_tailm25_fw010_realm25_featurefusion_candidates.csv"
    with feature_candidates_path.open("r", newline="", encoding="utf-8") as handle:
        feature_candidates = list(csv.DictReader(handle))

    pr1 = agg("vcf_c23_temporal_f3_locked_sequence_pr1_seed_summary_mi2000.json")
    pr2 = agg("vcf_c23_temporal_f3_locked_sequence_pr2_seed_summary_mi2000.json")
    pr4 = agg("vcf_c23_temporal_f3_locked_sequence_pr4_seed_summary_mi2000.json")

    meanstd_pr4 = candidate_locked_row(feature_candidates, "meanstd_logreg_pr4_c0.1")
    delta_pr4 = candidate_locked_row(feature_candidates, "delta_logreg_pr4_c0.1")
    sequence_pr4_single = candidate_locked_row(feature_candidates, "sequence_logreg_pr4_c0.1")

    rows = [
        {
            "row": "frame_prev_selective",
            "component": "single-frame selective hard-slice rank",
            "protocol": "heldout best threshold",
            "heldout_ba": 0.749278,
            "protected_real_fpr": 0.348735,
            "protected_fake_recall": 0.839384,
            "unprotected_fake_recall": 0.842409,
            "strict_fpr08": False,
            "reading": "Useful early VCF adaptation signal, but real-side failure remains large.",
        },
        {
            "row": "frame_dualrealc_tail_margin",
            "component": "real concept + independent dual head + hard-tail margins",
            "protocol": "heldout best threshold",
            "heldout_ba": 0.793936,
            "protected_real_fpr": 0.191419,
            "protected_fake_recall": 0.770352,
            "unprotected_fake_recall": 0.769802,
            "strict_fpr08": False,
            "reading": "Strong frame-level improvement, but still misses strict FPR08.",
        },
        {
            "row": "scalar_temporal_median",
            "component": "3-frame scalar probability median",
            "protocol": "best strict heldout diagnostic",
            "heldout_ba": float(scalar_strict["balanced_accuracy"]),
            "protected_real_fpr": float(scalar_strict["protected_real_fpr"]),
            "protected_fake_recall": float(scalar_strict["protected_fake_recall"]),
            "unprotected_fake_recall": float(scalar_strict["unprotected_fake_recall"]),
            "strict_fpr08": scalar_strict["strict_fpr08"] == "1",
            "reading": "Temporal scores alone open the strict gate but recall remains too low.",
        },
        {
            "row": "meanstd_feature_pr4",
            "component": "mean/std feature fusion + protected-real weighting",
            "protocol": "single-seed locked diagnostic",
            "heldout_ba": float(meanstd_pr4["heldout_balanced_accuracy"]),
            "protected_real_fpr": float(meanstd_pr4["heldout_protected_real_fpr"]),
            "protected_fake_recall": float(meanstd_pr4["heldout_protected_fake_recall"]),
            "unprotected_fake_recall": float(meanstd_pr4["heldout_unprotected_fake_recall"]),
            "strict_fpr08": float(meanstd_pr4["heldout_protected_real_fpr"]) <= 0.08,
            "reading": "Feature aggregation helps recall but loses strict FPR safety.",
        },
        {
            "row": "delta_feature_pr4",
            "component": "mean/std + late-minus-early feature delta",
            "protocol": "single-seed locked diagnostic",
            "heldout_ba": float(delta_pr4["heldout_balanced_accuracy"]),
            "protected_real_fpr": float(delta_pr4["heldout_protected_real_fpr"]),
            "protected_fake_recall": float(delta_pr4["heldout_protected_fake_recall"]),
            "unprotected_fake_recall": float(delta_pr4["heldout_unprotected_fake_recall"]),
            "strict_fpr08": float(delta_pr4["heldout_protected_real_fpr"]) <= 0.08,
            "reading": "Feature deltas are close but still miss FPR08; full sequence matters.",
        },
        {
            "row": "sequence_pr4_single_seed",
            "component": "full 3-frame feature sequence + protected-real weighting",
            "protocol": "single-seed locked diagnostic",
            "heldout_ba": float(sequence_pr4_single["heldout_balanced_accuracy"]),
            "protected_real_fpr": float(sequence_pr4_single["heldout_protected_real_fpr"]),
            "protected_fake_recall": float(sequence_pr4_single["heldout_protected_fake_recall"]),
            "unprotected_fake_recall": float(sequence_pr4_single["heldout_unprotected_fake_recall"]),
            "strict_fpr08": float(sequence_pr4_single["heldout_protected_real_fpr"]) <= 0.08,
            "reading": "Full sequence is the first strong strict-FPR solution.",
        },
        {
            "row": "sequence_pr1_5seed",
            "component": "full sequence, no extra protected-real weighting",
            "protocol": "5-seed locked, max_iter=2000",
            "heldout_ba": pr1["heldout_balanced_accuracy_mean"],
            "heldout_ba_std": pr1["heldout_balanced_accuracy_std"],
            "protected_real_fpr": pr1["heldout_protected_real_fpr_mean"],
            "protected_fake_recall": pr1["heldout_protected_fake_recall_mean"],
            "unprotected_fake_recall": pr1["heldout_unprotected_fake_recall_mean"],
            "strict_fpr08": pr1["all_strict_fpr08"],
            "reading": "Best BA/recall, but with smaller FPR margin; useful ablation, not the pre-locked safety row.",
        },
        {
            "row": "sequence_pr2_5seed",
            "component": "full sequence, moderate protected-real weighting",
            "protocol": "5-seed locked, max_iter=2000",
            "heldout_ba": pr2["heldout_balanced_accuracy_mean"],
            "heldout_ba_std": pr2["heldout_balanced_accuracy_std"],
            "protected_real_fpr": pr2["heldout_protected_real_fpr_mean"],
            "protected_fake_recall": pr2["heldout_protected_fake_recall_mean"],
            "unprotected_fake_recall": pr2["heldout_unprotected_fake_recall_mean"],
            "strict_fpr08": pr2["all_strict_fpr08"],
            "reading": "Middle tradeoff; confirms the sequence head is robust across weighting choices.",
        },
        {
            "row": "sequence_pr4_5seed_safety_main",
            "component": "full sequence, safety-oriented protected-real weighting",
            "protocol": "5-seed locked, max_iter=2000",
            "heldout_ba": pr4["heldout_balanced_accuracy_mean"],
            "heldout_ba_std": pr4["heldout_balanced_accuracy_std"],
            "protected_real_fpr": pr4["heldout_protected_real_fpr_mean"],
            "protected_fake_recall": pr4["heldout_protected_fake_recall_mean"],
            "unprotected_fake_recall": pr4["heldout_unprotected_fake_recall_mean"],
            "strict_fpr08": pr4["all_strict_fpr08"],
            "reading": "Pre-locked safety candidate; best FPR margin while preserving >0.92 fake recall.",
        },
    ]
    return rows


def build_sota_rows() -> List[dict]:
    old_rows = load_tsv("journal_post_vcf_sota_rows_2026-04-21.tsv")
    old_by_branch = {row["branch"]: row for row in old_rows}
    pr4 = agg("vcf_c23_temporal_f3_locked_sequence_pr4_seed_summary_mi2000.json")
    pr1 = agg("vcf_c23_temporal_f3_locked_sequence_pr1_seed_summary_mi2000.json")
    rows = []
    for branch in ["effb0_multisource_wide", "ucf_common_feature", "patchattn_mean_fpr08", "sfrg05_route_floor", "df40aux8_route_floor", "df40aux8_transfertree_literal_fixed"]:
        source = old_by_branch[branch]
        rows.append({**source, "vcf_temporal_ba": "", "vcf_temporal_pr_fpr": "", "vcf_temporal_pf_recall": "", "vcf_temporal_uf_recall": ""})
    rows.append(
        {
            "branch": "dualrealc_frame_vcf",
            "family": "vcf_spatial_adaptation",
            "protocol_tier": "VCF frame-level diagnostic",
            "d3_vcf_ba": "0.793936",
            "d3_vcf_fake_acc": "",
            "d3_vcf_pr_fpr": "0.191419",
            "vcf_temporal_ba": "",
            "vcf_temporal_pr_fpr": "",
            "vcf_temporal_pf_recall": "",
            "vcf_temporal_uf_recall": "",
            "note": "Strongest frame-level VCF detector before temporal fusion; still misses strict FPR08.",
        }
    )
    rows.append(
        {
            "branch": "sequence_logreg_pr4_c0.1_locked",
            "family": "vcf_temporal_feature_fusion",
            "protocol_tier": "VCF 3-frame locked temporal, strict FPR08",
            "vcf_temporal_ba": pr4["heldout_balanced_accuracy_mean"],
            "vcf_temporal_ba_std": pr4["heldout_balanced_accuracy_std"],
            "vcf_temporal_pr_fpr": pr4["heldout_protected_real_fpr_mean"],
            "vcf_temporal_pf_recall": pr4["heldout_protected_fake_recall_mean"],
            "vcf_temporal_uf_recall": pr4["heldout_unprotected_fake_recall_mean"],
            "note": "Pre-locked safety candidate; closes the adverse VCF deployment-stress gate under calib-only threshold selection.",
        }
    )
    rows.append(
        {
            "branch": "sequence_logreg_pr1_c0.1_locked_ablation",
            "family": "vcf_temporal_feature_fusion_ablation",
            "protocol_tier": "VCF 3-frame locked temporal ablation",
            "vcf_temporal_ba": pr1["heldout_balanced_accuracy_mean"],
            "vcf_temporal_ba_std": pr1["heldout_balanced_accuracy_std"],
            "vcf_temporal_pr_fpr": pr1["heldout_protected_real_fpr_mean"],
            "vcf_temporal_pf_recall": pr1["heldout_protected_fake_recall_mean"],
            "vcf_temporal_uf_recall": pr1["heldout_unprotected_fake_recall_mean"],
            "note": "BA-optimal ablation after the locked candidate; useful but not the pre-locked safety row.",
        }
    )
    return rows


def write_markdown(path: Path, ablation_rows: List[dict], sota_rows: List[dict]) -> None:
    pr4 = agg("vcf_c23_temporal_f3_locked_sequence_pr4_seed_summary_mi2000.json")
    pr1 = agg("vcf_c23_temporal_f3_locked_sequence_pr1_seed_summary_mi2000.json")
    d2 = load_tsv("deeperforensics_stress_summary_2026-04-20.tsv")
    d2_main = next(row for row in d2 if row["branch"] == "sfrg05_route_floor")

    lines: List[str] = []
    lines.append("# Journal Post-Temporal Unified Bundle")
    lines.append("")
    lines.append("Date: 2026-04-22")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append("- `VCF` is no longer an unresolved deployment-stress failure: the locked 3-frame temporal sequence head clears strict FPR08 with high fake recall.")
    lines.append("- The full high-level-journal package is close, but still needs final external SOTA source polishing and a paper-ready ablation narrative.")
    lines.append("- The final VCF main candidate should remain `sequence_logreg_pr4_c0.1_locked` because it was the pre-locked safety row. The stronger `pr1` result is valuable as an ablation, but should not replace the pre-locked row without a nested calib-only hyperparameter rule.")
    lines.append("")
    lines.append("## Main Evidence Snapshot")
    lines.append("")
    lines.append("| Evidence block | Row | BA | PR-FPR | Protected fake recall | Unprotected fake recall | Reading |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- |")
    lines.append(f"| Current local suite | `sfrg05 + route_floor` | 0.929522 | 0.063151 |  |  | Stable current-asset main row. |")
    lines.append(f"| DeeperForensics-1.0 | `sfrg05 + route_floor` | {d2_main['overall_balanced_accuracy']} | {d2_main['protected_real_fpr']} | {d2_main['protected_fake_recall']} | {d2_main['unprotected_fake_recall']} | First official deployment-stress block remains valid. |")
    lines.append("| FF++ four-method paired | `aux8 + literal_fixed` |  | <=0.028249 per method |  |  | Stronger paired frontier, but hardest simswap/inswap remain moderate. |")
    lines.append("| VCF frame-only | `dualrealc_tailm25_fw010_realm25` | 0.793936 | 0.191419 | 0.770352 | 0.769802 | Strong but not strict-safe. |")
    lines.append(f"| VCF temporal locked | `sequence_logreg_pr4_c0.1` | {pr4['heldout_balanced_accuracy_mean']:.6f} | {pr4['heldout_protected_real_fpr_mean']:.6f} | {pr4['heldout_protected_fake_recall_mean']:.6f} | {pr4['heldout_unprotected_fake_recall_mean']:.6f} | Pre-locked safety candidate; closes VCF strict gate. |")
    lines.append(f"| VCF temporal ablation | `sequence_logreg_pr1_c0.1` | {pr1['heldout_balanced_accuracy_mean']:.6f} | {pr1['heldout_protected_real_fpr_mean']:.6f} | {pr1['heldout_protected_fake_recall_mean']:.6f} | {pr1['heldout_unprotected_fake_recall_mean']:.6f} | BA-optimal ablation; not promoted as final row without nested selection. |")
    lines.append("")
    lines.append("## VCF Ablation Reading")
    lines.append("")
    lines.append("| Row | Component | Protocol | BA | PR-FPR | PF recall | UF recall | Strict FPR08 |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | --- |")
    for row in ablation_rows:
        lines.append(
            f"| `{row['row']}` | {row['component']} | {row['protocol']} | {fmt(row.get('heldout_ba'))} | {fmt(row.get('protected_real_fpr'))} | {fmt(row.get('protected_fake_recall'))} | {fmt(row.get('unprotected_fake_recall'))} | {fmt(row.get('strict_fpr08'))} |"
        )
    lines.append("")
    lines.append("Key interpretation:")
    lines.append("")
    lines.append("1. Frame-level adaptation was necessary but insufficient: even the strongest frame-only row misses strict FPR08.")
    lines.append("2. Scalar temporal aggregation proves video evidence matters, but scalar logits alone lose too much fake recall.")
    lines.append("3. Full frame-feature sequence fusion is the decisive component; mean/std and simple deltas do not reliably meet strict FPR08.")
    lines.append("4. Protected-real weighting is a safety-margin control, not the only source of the gain: `pr1`, `pr2`, and `pr4` all pass, while `pr4` gives the cleanest FPR margin among pre-locked candidates.")
    lines.append("")
    lines.append("## Remaining Paper Gaps")
    lines.append("")
    lines.append("1. Final external SOTA comparison still needs source-cleaned citations and a unified protocol table.")
    lines.append("2. The method section must clearly define when the temporal head is active and how non-video/one-frame datasets fall back to the spatial router.")
    lines.append("3. The ablation package is now strong for VCF, but the paper should still include a compact cross-dataset regression statement showing that the added temporal head does not alter the already validated spatial rows.")
    lines.append("")
    lines.append("## Companion Files")
    lines.append("")
    lines.append("- `prototype/reports/vcf_temporal_ablation_rows_2026-04-22.tsv`")
    lines.append("- `prototype/reports/journal_post_temporal_sota_rows_2026-04-22.tsv`")
    lines.append("- `prototype/reports/vcf_temporal_locked_stability_2026-04-22.zh-CN.md`")
    lines.append("- `prototype/reports/vcf_temporal_evidence_2026-04-22.zh-CN.md`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ablation_rows = build_ablation_rows()
    sota_rows = build_sota_rows()
    write_tsv(REPORTS / "vcf_temporal_ablation_rows_2026-04-22.tsv", ablation_rows)
    write_tsv(REPORTS / "journal_post_temporal_sota_rows_2026-04-22.tsv", sota_rows)
    write_markdown(REPORTS / "journal_post_temporal_sota_bundle_2026-04-22.zh-CN.md", ablation_rows, sota_rows)
    print("Wrote final temporal bundle reports.")


if __name__ == "__main__":
    main()
