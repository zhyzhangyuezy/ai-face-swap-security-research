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
    pre_d2_rows = load_tsv(REPORTS / "journal_pre_d2_sota_rows_2026-04-20.tsv")
    d2_rows = {row["branch"]: row for row in load_tsv(REPORTS / "deeperforensics_stress_summary_2026-04-20.tsv")}

    final_rows: list[dict[str, Any]] = []
    for row in pre_d2_rows:
        branch = row["branch"]
        d2 = d2_rows.get(branch, {})
        final_rows.append(
            {
                **row,
                "d2_deeperforensics_ba": to_float(d2.get("overall_balanced_accuracy")),
                "d2_deeperforensics_fake_acc": to_float(d2.get("overall_fake_accuracy")),
                "d2_deeperforensics_pr_fpr": to_float(d2.get("protected_real_fpr")),
                "d2_deeperforensics_subset_min_fake_acc": to_float(d2.get("subset_min_fake_accuracy")),
                "d2_deeperforensics_subset_max_fake_acc": to_float(d2.get("subset_max_fake_accuracy")),
            }
        )

    out_tsv = REPORTS / "journal_post_d2_sota_rows_2026-04-20.tsv"
    write_tsv(out_tsv, final_rows)

    by_branch = {row["branch"]: row for row in final_rows}
    lines: list[str] = []
    lines.append("# Journal Post-D2 Unified SOTA Bundle")
    lines.append("")
    lines.append("This bundle extends the earlier protocol-aware current-asset package with one official deployment-stress benchmark: `DeeperForensics-1.0` manipulated test split plus local FF++ real anchors. The key story is now cleaner: the external deployment-stress gap is no longer hypothetical, and it strengthens `sfrg05 + route_floor` rather than overturning the current stable-main-row conclusion.")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append("- `claim_supported: partial`")
    lines.append("- `confidence: medium`")
    lines.append("- `D2: completed` with `DeeperForensics-1.0` official manipulated test split.")
    lines.append("- Stable main row still: `sfrg05 + route_floor`.")
    lines.append("- Paired-oriented explicit branch still: frontier / diagnostic, not stable final row.")
    lines.append("")
    lines.append("## Main Comparator Table")
    lines.append("")
    lines.append("| Branch | Tier | All BA | Min BA | Max PR-FPR | LODO mean/min | Four-method paired mean fake BA | D2 BA / Fake Acc / PR-FPR |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
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
        lines.append(
            f"| `{branch}` | `{row['protocol_tier']}` | {row['all_data_ba']} | {row['min_dataset_ba']} | {row['max_protected_real_fpr']} | {row['lodo_mean_ba']} / {row['lodo_min_ba']} | {row['paired_four_method_mean_fake_ba']} | {d2_text} |"
        )
    lines.append("")
    lines.append("## What D2 Changed")
    lines.append("")
    lines.append("1. The external deployment-stress gap is now materially smaller because one official benchmark is in hand.")
    lines.append("2. `sfrg05 + route_floor` wins that benchmark cleanly, so the stable main-row story is stronger than before.")
    lines.append("3. `df40aux8 + transfertree_literal_fixed` still beats plain `aux8` under D2, which means the paired-oriented routing signal transfers, but it still does not displace `sfrg05` as the main row.")
    lines.append("")
    lines.append("## D2 Highlights")
    lines.append("")
    lines.append(f"- `sfrg05 + route_floor`: BA {fmt(by_branch['sfrg05_route_floor']['d2_deeperforensics_ba'])}, fake acc {fmt(by_branch['sfrg05_route_floor']['d2_deeperforensics_fake_acc'])}, PR-FPR {fmt(by_branch['sfrg05_route_floor']['d2_deeperforensics_pr_fpr'])}")
    lines.append(f"- `df40aux8 + route_floor`: BA {fmt(by_branch['df40aux8_route_floor']['d2_deeperforensics_ba'])}, fake acc {fmt(by_branch['df40aux8_route_floor']['d2_deeperforensics_fake_acc'])}, PR-FPR {fmt(by_branch['df40aux8_route_floor']['d2_deeperforensics_pr_fpr'])}")
    lines.append(f"- `df40aux8 + transfertree_literal_fixed`: BA {fmt(by_branch['df40aux8_transfertree_literal_fixed']['d2_deeperforensics_ba'])}, fake acc {fmt(by_branch['df40aux8_transfertree_literal_fixed']['d2_deeperforensics_fake_acc'])}, PR-FPR {fmt(by_branch['df40aux8_transfertree_literal_fixed']['d2_deeperforensics_pr_fpr'])}")
    lines.append("")
    lines.append("## Remaining Gaps")
    lines.append("")
    lines.append("1. Final external comparison packaging under one clean protocol.")
    lines.append("2. A stronger claim-safe story for the paired-oriented branch, unless the paper keeps it explicitly as a frontier / diagnostic row.")
    lines.append("3. If available, a second deployment-style benchmark such as `VCF` would still improve persuasion further, but it is no longer the only thing standing between the project and a journal-facing evidence package.")
    lines.append("")
    lines.append("## Companion Files")
    lines.append("")
    lines.append("- `prototype/reports/deeperforensics_stress_report_2026-04-20.zh-CN.md`")
    lines.append("- `prototype/reports/journal_pre_d2_sota_bundle_2026-04-20.zh-CN.md`")
    lines.append("- `prototype/reports/df40_ffpp_test_paired_all_methods_2026-04-20.zh-CN.md`")
    lines.append("- `prototype/reports/journal_stability_progress_2026-04-20.zh-CN.md`")
    lines.append("")

    out_md = REPORTS / "journal_post_d2_sota_bundle_2026-04-20.zh-CN.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_md}")
    print(f"Wrote {out_tsv}")


if __name__ == "__main__":
    main()
