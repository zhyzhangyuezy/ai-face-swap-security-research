from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


PROTOTYPE_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROTOTYPE_ROOT / "reports"


def load_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def to_float(value: str | float | None) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, float):
        return value
    return float(value)


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(value) if isinstance(value, float) or value is None else value for key, value in row.items()})


def main() -> None:
    post_d2_rows = load_tsv(REPORTS / "journal_post_d2_sota_rows_2026-04-20.tsv")
    vcf_rows = {row["branch"]: row for row in load_tsv(REPORTS / "vcf_stress_summary_2026-04-21.tsv")}

    final_rows: list[dict[str, Any]] = []
    for row in post_d2_rows:
        branch = row["branch"]
        vcf = vcf_rows.get(branch, {})
        final_rows.append(
            {
                **row,
                "d3_vcf_ba": to_float(vcf.get("overall_balanced_accuracy")),
                "d3_vcf_fake_acc": to_float(vcf.get("overall_fake_accuracy")),
                "d3_vcf_pr_fpr": to_float(vcf.get("protected_real_fpr")),
                "d3_vcf_subset_min_fake_acc": to_float(vcf.get("subset_min_fake_accuracy")),
                "d3_vcf_subset_max_fake_acc": to_float(vcf.get("subset_max_fake_accuracy")),
            }
        )

    out_tsv = REPORTS / "journal_post_vcf_sota_rows_2026-04-21.tsv"
    write_tsv(out_tsv, final_rows)

    by_branch = {row["branch"]: row for row in final_rows}
    lines: list[str] = []
    lines.append("# Journal Post-VCF Unified SOTA Bundle")
    lines.append("")
    lines.append("This bundle extends the earlier post-D2 package with a second official deployment-style benchmark: local `VCF` under the `c23` domain using internal target/fake pairs. The story is now much tighter because the same stable-main-row conclusion survives both `DeeperForensics-1.0` and `VCF`.")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append("- `claim_supported: partial`")
    lines.append("- `confidence: medium`")
    lines.append("- `D2`: completed with `DeeperForensics-1.0` official manipulated test split.")
    lines.append("- `VCF`: completed with internal paired `targets`/fake evaluation under `c23`.")
    lines.append("- `VCF` exposes a genuine deployment-stress failure mode: protected-real FPR rises sharply for every current branch.")
    lines.append("- Stable paper-facing main row remains `sfrg05 + route_floor` on the closed current-asset + DeeperForensics package, but `VCF` no longer supports treating that conclusion as fully stress-invariant.")
    lines.append("- Paired-oriented explicit branch remains: frontier / diagnostic, not stable final row.")
    lines.append("")
    lines.append("## Main Comparator Table")
    lines.append("")
    lines.append("| Branch | Tier | All BA | Min BA | Max PR-FPR | LODO mean/min | Four-method paired mean fake BA | DF BA / Fake Acc / PR-FPR | VCF BA / Fake Acc / PR-FPR |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for branch in [
        "effb0_multisource_wide",
        "ucf_common_feature",
        "patchattn_mean_fpr08",
        "sfrg05_route_floor",
        "df40aux8_route_floor",
        "df40aux8_transfertree_literal_fixed",
    ]:
        row = by_branch[branch]
        d2_text = ""
        if row["d2_deeperforensics_ba"] is not None:
            d2_text = f"{fmt(row['d2_deeperforensics_ba'])} / {fmt(row['d2_deeperforensics_fake_acc'])} / {fmt(row['d2_deeperforensics_pr_fpr'])}"
        d3_text = ""
        if row["d3_vcf_ba"] is not None:
            d3_text = f"{fmt(row['d3_vcf_ba'])} / {fmt(row['d3_vcf_fake_acc'])} / {fmt(row['d3_vcf_pr_fpr'])}"
        lines.append(
            f"| `{branch}` | `{row['protocol_tier']}` | {row['all_data_ba']} | {row['min_dataset_ba']} | {row['max_protected_real_fpr']} | {row['lodo_mean_ba']} / {row['lodo_min_ba']} | {row['paired_four_method_mean_fake_ba']} | {d2_text} | {d3_text} |"
        )
    lines.append("")
    lines.append("## What VCF Changed")
    lines.append("")
    lines.append("1. The project now has two deployment-style checks rather than one, which is valuable, but the second one is not a clean confirmation result.")
    lines.append("2. `VCF` is substantially harsher than `DeeperForensics-1.0`: protected-real FPR jumps from roughly `0.02` to roughly `0.48-0.53` for the current branches.")
    lines.append("3. `df40aux8 + route_floor` becomes the best VCF branch by BA and real-side cost, while `df40aux8 + transfertree_literal_fixed` gets the highest fake accuracy but the worst PR-FPR.")
    lines.append("4. The stable-main-row story still holds on the earlier package, but the overall claim package remains only `partial` because VCF exposes a real generalization gap.")
    lines.append("")
    lines.append("## Stress Highlights")
    lines.append("")
    lines.append(f"- `sfrg05 + route_floor`: DF {fmt(by_branch['sfrg05_route_floor']['d2_deeperforensics_ba'])} BA / {fmt(by_branch['sfrg05_route_floor']['d2_deeperforensics_pr_fpr'])} PR-FPR, VCF {fmt(by_branch['sfrg05_route_floor']['d3_vcf_ba'])} BA / {fmt(by_branch['sfrg05_route_floor']['d3_vcf_pr_fpr'])} PR-FPR")
    lines.append(f"- `df40aux8 + route_floor`: DF {fmt(by_branch['df40aux8_route_floor']['d2_deeperforensics_ba'])} BA / {fmt(by_branch['df40aux8_route_floor']['d2_deeperforensics_pr_fpr'])} PR-FPR, VCF {fmt(by_branch['df40aux8_route_floor']['d3_vcf_ba'])} BA / {fmt(by_branch['df40aux8_route_floor']['d3_vcf_pr_fpr'])} PR-FPR")
    lines.append(f"- `df40aux8 + transfertree_literal_fixed`: DF {fmt(by_branch['df40aux8_transfertree_literal_fixed']['d2_deeperforensics_ba'])} BA / {fmt(by_branch['df40aux8_transfertree_literal_fixed']['d2_deeperforensics_pr_fpr'])} PR-FPR, VCF {fmt(by_branch['df40aux8_transfertree_literal_fixed']['d3_vcf_ba'])} BA / {fmt(by_branch['df40aux8_transfertree_literal_fixed']['d3_vcf_pr_fpr'])} PR-FPR")
    lines.append("")
    lines.append("## Remaining Gaps")
    lines.append("")
    lines.append("1. A VCF-safe adaptation or calibration story is now a genuine method gap, not just a packaging gap.")
    lines.append("2. Final external SOTA comparison packaging under one reviewer-facing protocol.")
    lines.append("3. A stronger claim-safe story for the paired-oriented branch, unless the paper keeps it explicitly as a frontier / diagnostic row.")
    lines.append("")
    lines.append("## Companion Files")
    lines.append("")
    lines.append("- `prototype/reports/deeperforensics_stress_report_2026-04-20.zh-CN.md`")
    lines.append("- `prototype/reports/vcf_stress_report_2026-04-21.zh-CN.md`")
    lines.append("- `prototype/reports/journal_post_d2_sota_bundle_2026-04-20.zh-CN.md`")
    lines.append("- `prototype/reports/df40_ffpp_test_paired_all_methods_2026-04-20.zh-CN.md`")
    lines.append("- `prototype/reports/journal_stability_progress_2026-04-20.zh-CN.md`")
    lines.append("")

    out_md = REPORTS / "journal_post_vcf_sota_bundle_2026-04-21.zh-CN.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_md}")
    print(f"Wrote {out_tsv}")


if __name__ == "__main__":
    main()
