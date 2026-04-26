from __future__ import annotations

import csv
import statistics
from pathlib import Path


PROTOTYPE_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROTOTYPE_ROOT / "reports"


def load_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def fmt(value: float | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def mean_std_text(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.6f} +/- 0.000000"
    return f"{statistics.mean(values):.6f} +/- {statistics.stdev(values):.6f}"


def build_stability_summary(progress_rows: list[dict[str, str]], branch: str) -> dict[str, str]:
    branch_rows = [row for row in progress_rows if row["branch"] == branch]
    kept_row = next(row for row in branch_rows if row["status"] == "kept_reference")
    rerun_rows = [row for row in branch_rows if row["status"] == "rerun_completed"]
    return {
        "kept_test_ba": fmt(to_float(kept_row["test_balanced_acc"])),
        "kept_cross_min_ba": fmt(to_float(kept_row["cross_min_balanced_acc"])),
        "kept_cross_mean_ba": fmt(to_float(kept_row["cross_mean_balanced_acc"])),
        "rerun_test_ba_mean_std": mean_std_text([to_float(row["test_balanced_acc"]) for row in rerun_rows if to_float(row["test_balanced_acc"]) is not None]),
        "rerun_cross_min_ba_mean_std": mean_std_text([to_float(row["cross_min_balanced_acc"]) for row in rerun_rows if to_float(row["cross_min_balanced_acc"]) is not None]),
        "rerun_cross_mean_ba_mean_std": mean_std_text([to_float(row["cross_mean_balanced_acc"]) for row in rerun_rows if to_float(row["cross_mean_balanced_acc"]) is not None]),
    }


def main() -> None:
    current_evidence = load_tsv(REPORTS / "current_evidence_table_2026-04-20.tsv")
    stability_progress = load_tsv(REPORTS / "journal_stability_progress_2026-04-20.tsv")
    paired_rows = load_tsv(REPORTS / "df40_ffpp_test_paired_all_methods_2026-04-20.tsv")
    missing_rows = load_tsv(REPORTS / "journal_missing_evidence_tracker_2026-04-20.tsv")

    main_branches = {
        row["branch"]: row
        for row in current_evidence
        if row["branch"] in {"sfrg05_route_floor", "df40aux8_route_floor", "df40aux8_transfertree_literal_fixed"}
    }
    paired_by_variant: dict[str, list[dict[str, str]]] = {}
    for row in paired_rows:
        paired_by_variant.setdefault(row["variant"], []).append(row)

    summary_rows: list[dict[str, str]] = []

    for branch in ["sfrg05_route_floor", "df40aux8_route_floor", "df40aux8_transfertree_literal_fixed"]:
        row = main_branches[branch]
        summary_rows.extend(
            [
                {"section": "main_rows", "item": branch, "metric": "all_data_ba", "value": row["all_data_ba"], "note": row["note"]},
                {"section": "main_rows", "item": branch, "metric": "min_dataset_ba", "value": row["all_data_min_dataset_ba"], "note": row["note"]},
                {"section": "main_rows", "item": branch, "metric": "max_protected_real_fpr", "value": row["all_data_max_protected_real_fpr"], "note": row["note"]},
                {"section": "main_rows", "item": branch, "metric": "lodo_mean_ba", "value": row["lodo_mean_ba"], "note": row["note"]},
                {"section": "main_rows", "item": branch, "metric": "lodo_min_ba", "value": row["lodo_min_ba"], "note": row["note"]},
            ]
        )

    for variant, rows in paired_by_variant.items():
        fake_bas = [to_float(row["fake_accuracy"]) for row in rows if to_float(row["fake_accuracy"]) is not None]
        summary_rows.append(
            {
                "section": "paired_four_method_block",
                "item": variant,
                "metric": "mean_fake_ba",
                "value": fmt(statistics.mean(fake_bas)),
                "note": f"min_fake_ba={min(fake_bas):.6f}; max_fake_ba={max(fake_bas):.6f}",
            }
        )
        for row in rows:
            summary_rows.append(
                {
                    "section": "paired_four_method_block",
                    "item": variant,
                    "metric": f"{row['method']}_fake_ba",
                    "value": row["fake_accuracy"],
                    "note": f"protected_real_fpr={row['protected_real_fpr']}",
                }
            )

    for branch in ["sfrg05_route_floor", "df40aux8_route_floor"]:
        stability = build_stability_summary(stability_progress, branch)
        for metric, value in stability.items():
            summary_rows.append(
                {
                    "section": "stability",
                    "item": branch,
                    "metric": metric,
                    "value": value,
                    "note": "",
                }
            )

    for row in missing_rows:
        summary_rows.append(
            {
                "section": "remaining_gaps",
                "item": row["evidence_gap"],
                "metric": "status",
                "value": row["status"],
                "note": row["note"],
            }
        )

    tsv_path = REPORTS / "journal_unified_summary_rows_2026-04-20.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["section", "item", "metric", "value", "note"], delimiter="\t")
        writer.writeheader()
        writer.writerows(summary_rows)

    sfrg05_stability = build_stability_summary(stability_progress, "sfrg05_route_floor")
    aux8_stability = build_stability_summary(stability_progress, "df40aux8_route_floor")

    lines: list[str] = []
    lines.append("# Journal Unified Package (Pre-D2)")
    lines.append("")
    lines.append("This package merges the closed current-asset evidence (`P0/P1/P2`) with the newly closed `D1` four-method held-out FF++ paired block. It is the strongest paper-facing package currently available before an official deployment-stress dataset lands.")
    lines.append("")
    lines.append("## Current Verdict")
    lines.append("")
    lines.append("- `claim_supported: partial`")
    lines.append("- `confidence: medium`")
    lines.append("- Stable main row on current assets: `sfrg05 + route_floor`")
    lines.append("- Balanced light branch: `df40aux8 + route_floor`")
    lines.append("- Paired-oriented frontier / diagnostic: `df40aux8 + transfertree_literal_fixed`")
    lines.append("- Remaining external-data blocker: `D2` deployment-stress benchmark")
    lines.append("")
    lines.append("## Main Rows")
    lines.append("")
    lines.append("| Branch | All BA | Min BA | Max PR-FPR | LODO mean/min | Current role |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for branch in ["sfrg05_route_floor", "df40aux8_route_floor", "df40aux8_transfertree_literal_fixed"]:
        row = main_branches[branch]
        role = {
            "sfrg05_route_floor": "stable_main_row_current_assets",
            "df40aux8_route_floor": "balanced_light_branch",
            "df40aux8_transfertree_literal_fixed": "paired_oriented_frontier_diagnostic",
        }[branch]
        lines.append(
            f"| `{branch}` | {row['all_data_ba']} | {row['all_data_min_dataset_ba']} | {row['all_data_max_protected_real_fpr']} | {row['lodo_mean_ba']} / {row['lodo_min_ba']} | `{role}` |"
        )
    lines.append("")
    lines.append("## Four-Method Held-Out Paired Block")
    lines.append("")
    lines.append("All four DF40 methods are now evaluated under one aligned FF++ `test.json` paired protocol.")
    lines.append("")
    lines.append("| Method | sfrg05 Fake BA | aux8 Fake BA | aux8+literal Fake BA | literal-aux8 Delta |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    methods = ["simswap", "inswap", "faceswap", "fsgan"]
    for method in methods:
        row_map = {row["variant"]: row for row in paired_rows if row["method"] == method}
        delta = to_float(row_map["aux8_literal_fixed"]["fake_accuracy"]) - to_float(row_map["aux8_plain"]["fake_accuracy"])
        lines.append(
            f"| `{method}` | {row_map['sfrg05_plain']['fake_accuracy']} | {row_map['aux8_plain']['fake_accuracy']} | {row_map['aux8_literal_fixed']['fake_accuracy']} | {delta:+.6f} |"
        )
    lines.append("")
    lines.append("| Variant | Mean Fake BA | Min Fake BA | Mean Protected-real FPR |")
    lines.append("| --- | ---: | ---: | ---: |")
    for variant in ["sfrg05_plain", "aux8_plain", "aux8_literal_fixed"]:
        rows = paired_by_variant[variant]
        fake_values = [to_float(row["fake_accuracy"]) for row in rows if to_float(row["fake_accuracy"]) is not None]
        fpr_values = [to_float(row["protected_real_fpr"]) for row in rows if to_float(row["protected_real_fpr"]) is not None]
        lines.append(
            f"| `{variant}` | {statistics.mean(fake_values):.6f} | {min(fake_values):.6f} | {statistics.mean(fpr_values):.6f} |"
        )
    lines.append("")
    lines.append("## Stability Position")
    lines.append("")
    lines.append("| Branch | Kept cross min BA | Rerun cross min BA | Interpretation |")
    lines.append("| --- | ---: | ---: | --- |")
    lines.append(
        f"| `sfrg05_route_floor` | {sfrg05_stability['kept_cross_min_ba']} | {sfrg05_stability['rerun_cross_min_ba_mean_std']} | Stable enough to anchor the current main row. |"
    )
    lines.append(
        f"| `df40aux8_route_floor` | {aux8_stability['kept_cross_min_ba']} | {aux8_stability['rerun_cross_min_ba_mean_std']} | Useful branch, but cross-domain floor is clearly looser. |"
    )
    lines.append(
        "| `df40aux8_transfertree_literal_fixed` | 0.883545 | replay 0.815023 +/- 0.004301 | Keep as paired-oriented frontier / diagnostic, not as the stable final row. |"
    )
    lines.append("")
    lines.append("## Remaining Gaps")
    lines.append("")
    lines.append("1. One official deployment-stress benchmark in the main table (`VCF` preferred, otherwise `DeeperForensics-1.0`).")
    lines.append("2. Final unified external SOTA comparison under one protocol.")
    lines.append("3. If the final paper still wants a paired-oriented final row, that row needs a stability-safe version rather than only the kept seed-42 frontier.")
    lines.append("")
    lines.append("## Immediate Paper-Facing Use")
    lines.append("")
    lines.append("Use this package to write or revise the main experimental narrative now:")
    lines.append("")
    lines.append("- Main cross-dataset claim row: `sfrg05 + route_floor`")
    lines.append("- Supporting light fake-side row: `df40aux8 + route_floor`")
    lines.append("- Paired-oriented frontier row: `df40aux8 + transfertree_literal_fixed`")
    lines.append("- Four-method unseen-generator paired block: `simswap / inswap / faceswap / fsgan` under FF++ `test.json`")
    lines.append("- Explicit caveat: deployment-stress and paired-oriented stability remain open")
    lines.append("")
    lines.append("Raw machine-readable summary rows are stored in `prototype/reports/journal_unified_summary_rows_2026-04-20.tsv`.")

    md_path = REPORTS / "journal_unified_package_2026-04-20.zh-CN.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {md_path}")
    print(f"Wrote {tsv_path}")


if __name__ == "__main__":
    main()
