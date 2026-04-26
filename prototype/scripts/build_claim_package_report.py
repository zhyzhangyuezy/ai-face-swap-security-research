from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def load_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def delta(value: float | None, base: float | None) -> float | None:
    if value is None or base is None:
        return None
    return value - base


def displayed_role(branch: str) -> str:
    roles = {
        "patchattn_mean_fpr08": "historical_strict_baseline",
        "sfrg05_route_floor": "stable_main_row_current_assets",
        "df40aux8_route_floor": "balanced_light_branch",
        "df40aux8_transferpattern_route_floor": "explicit_fallback",
        "df40aux8_transfertree_fixed78": "learned_precursor",
        "df40aux8_transfertree_literal_fixed": "paired_oriented_frontier_diagnostic",
    }
    return roles.get(branch, branch)


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(value) if isinstance(value, float) or value is None else value for key, value in row.items()})


def main() -> None:
    evidence_rows = load_tsv(REPORTS / "current_evidence_table_2026-04-20.tsv")
    evidence = {row["branch"]: row for row in evidence_rows}
    stability_rows = {row["branch"]: row for row in load_tsv(REPORTS / "journal_stability_rows_2026-04-20.tsv")}

    historical = evidence["patchattn_mean_fpr08"]
    transferpattern = evidence["df40aux8_transferpattern_route_floor"]
    fixed78 = evidence["df40aux8_transfertree_fixed78"]
    literal = evidence["df40aux8_transfertree_literal_fixed"]
    sfrg05 = evidence["sfrg05_route_floor"]
    df40aux8 = evidence["df40aux8_route_floor"]

    delta_rows: list[dict[str, Any]] = []
    for branch in [sfrg05, df40aux8, transferpattern, fixed78, literal]:
        delta_rows.append(
            {
                "branch": branch["branch"],
                "role": displayed_role(branch["branch"]),
                "delta_all_data_ba_vs_patchattn": delta(to_float(branch["all_data_ba"]), to_float(historical["all_data_ba"])),
                "delta_min_dataset_ba_vs_patchattn": delta(
                    to_float(branch["all_data_min_dataset_ba"]),
                    to_float(historical["all_data_min_dataset_ba"]),
                ),
                "delta_max_protected_real_fpr_vs_patchattn": delta(
                    to_float(branch["all_data_max_protected_real_fpr"]),
                    to_float(historical["all_data_max_protected_real_fpr"]),
                ),
                "delta_lodo_mean_ba_vs_patchattn": delta(to_float(branch["lodo_mean_ba"]), to_float(historical["lodo_mean_ba"])),
                "delta_lodo_min_ba_vs_patchattn": delta(to_float(branch["lodo_min_ba"]), to_float(historical["lodo_min_ba"])),
                "delta_paired_mean_fake_ba_vs_transferpattern": delta(
                    to_float(branch["paired_mean_fake_ba"]),
                    to_float(transferpattern["paired_mean_fake_ba"]),
                ),
                "delta_simswap_fake_ba_vs_transferpattern": delta(
                    to_float(branch["paired_simswap_fake_ba"]),
                    to_float(transferpattern["paired_simswap_fake_ba"]),
                ),
                "delta_inswap_fake_ba_vs_transferpattern": delta(
                    to_float(branch["paired_inswap_fake_ba"]),
                    to_float(transferpattern["paired_inswap_fake_ba"]),
                ),
            }
        )

    paired_diag_rows = load_tsv(REPORTS / "df40_ffpp_paired_transferpair_compare_2026-04-19.tsv")
    overlap_rows = {row["method"]: row for row in load_tsv(REPORTS / "df40_ffpp_paired_overlap_audit_2026-04-20.tsv")}

    sfrg05_stability = stability_rows["sfrg05_route_floor"]
    df40aux8_stability = stability_rows["df40aux8_route_floor"]
    literal_stability = stability_rows["df40aux8_transfertree_literal_fixed"]

    md_lines = [
        "# Current Claim Package",
        "",
        "This note reorganizes the strongest current local evidence into a journal-facing claim package: main comparable rows, stability position, routing ablation ladder, paired ceiling diagnostics, and the remaining claim gaps after the current-asset stability closure.",
        "",
        "## Main Comparable Rows",
        "",
        "| Branch | Role | All BA | Min BA | Max PR-FPR | LODO mean/min | Paired mean/min fake BA |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for branch in [historical, sfrg05, df40aux8, literal]:
        paired = ""
        if branch["paired_mean_fake_ba"]:
            paired = f"{branch['paired_mean_fake_ba']} / {branch['paired_min_fake_ba']}"
        md_lines.append(
            f"| {branch['branch']} | {displayed_role(branch['branch'])} | {branch['all_data_ba']} | {branch['all_data_min_dataset_ba']} | "
            f"{branch['all_data_max_protected_real_fpr']} | {branch['lodo_mean_ba']} / {branch['lodo_min_ba']} | {paired} |"
        )

    md_lines.extend(
        [
            "",
            "## Stability Position",
            "",
            "| Branch | Current role | Current signal | Interpretation |",
            "| --- | --- | --- | --- |",
            f"| sfrg05_route_floor | stable main row | rerun test BA {sfrg05_stability['rerun_test_balanced_acc_mean_std']}; rerun cross min BA {sfrg05_stability['rerun_cross_min_balanced_acc_mean_std']}; rerun cross mean BA {sfrg05_stability['rerun_cross_mean_balanced_acc_mean_std']} | This is now the tightest passive family on current assets and should anchor the paper-facing main row. |",
            f"| df40aux8_route_floor | balanced light branch | rerun test BA {df40aux8_stability['rerun_test_balanced_acc_mean_std']}; rerun cross min BA {df40aux8_stability['rerun_cross_min_balanced_acc_mean_std']}; rerun cross mean BA {df40aux8_stability['rerun_cross_mean_balanced_acc_mean_std']} | The DF40 light branch stays useful, but its worst-case cross-domain floor is materially weaker than the kept seed-42 run. |",
            f"| df40aux8_transfertree_literal_fixed | paired-oriented frontier / diagnostic | replay canonical all BA {literal_stability['replay_canonical_all_ba_mean_std']}; replay canonical min BA {literal_stability['replay_canonical_min_ba_mean_std']}; replay paired mean fake BA {literal_stability['replay_paired_mean_fake_ba_mean_std']}; replay LODO mean/min BA {literal_stability['replay_lodo_mean_ba_mean_std']} / {literal_stability['replay_lodo_min_ba_mean_std']} | The kept seed-42 frontier is real, but replay is systematically weaker, so this branch should not be framed as the stable final row. |",
            "",
            "## Delta vs Historical Strict Baseline",
            "",
            "| Branch | dAll BA | dMin BA | dMax PR-FPR | dLODO mean | dLODO min |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in delta_rows:
        md_lines.append(
            f"| {row['branch']} | {fmt(row['delta_all_data_ba_vs_patchattn'])} | "
            f"{fmt(row['delta_min_dataset_ba_vs_patchattn'])} | "
            f"{fmt(row['delta_max_protected_real_fpr_vs_patchattn'])} | "
            f"{fmt(row['delta_lodo_mean_ba_vs_patchattn'])} | "
            f"{fmt(row['delta_lodo_min_ba_vs_patchattn'])} |"
        )

    md_lines.extend(
        [
            "",
            "## Routing Ablation Ladder",
            "",
            "| Branch | All BA | LODO mean/min | Paired mean/min fake BA | Simswap / Inswap | Key Interpretation |",
            "| --- | --- | --- | --- | --- | --- |",
            f"| {transferpattern['branch']} | {transferpattern['all_data_ba']} | {transferpattern['lodo_mean_ba']} / {transferpattern['lodo_min_ba']} | "
            f"{transferpattern['paired_mean_fake_ba']} / {transferpattern['paired_min_fake_ba']} | {transferpattern['paired_simswap_fake_ba']} / {transferpattern['paired_inswap_fake_ba']} | Simpler explicit fallback; safe, but no longer the best paired-oriented rule. |",
            f"| {fixed78['branch']} | {fixed78['all_data_ba']} | {fixed78['lodo_mean_ba']} / {fixed78['lodo_min_ba']} | "
            f"{fixed78['paired_mean_fake_ba']} / {fixed78['paired_min_fake_ba']} | {fixed78['paired_simswap_fake_ba']} / {fixed78['paired_inswap_fake_ba']} | Fixed learned precursor that exposed the useful tree leaves. |",
            f"| {literal['branch']} | {literal['all_data_ba']} | {literal['lodo_mean_ba']} / {literal['lodo_min_ba']} | "
            f"{literal['paired_mean_fake_ba']} / {literal['paired_min_fake_ba']} | {literal['paired_simswap_fake_ba']} / {literal['paired_inswap_fake_ba']} | Exact explicit four-leaf recovery of the kept-seed tree frontier; keep it as the paired-oriented frontier / diagnostic branch, not as the stable final row. |",
            "",
            "## Paired Ceiling Diagnostic",
            "",
            "| Rule | Method | Fake BA | Protected-real FPR | Interpretation |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in paired_diag_rows:
        interpretation = {
            "aux8_locked": "Locked canonical rule; safe but conservative paired starting point.",
            "aux8_transferpair": "Safe transfer-aware diagnostic; modest paired gain.",
            "aux8_sfrg05rules": "Aggressive transferred-rule ceiling; useful upper bound but not the kept safe routing branch.",
        }[row["rule"]]
        md_lines.append(
            f"| {row['rule']} | {row['method']} | {row['fake_accuracy']} | {row['protected_real_fpr']} | {interpretation} |"
        )

    md_lines.extend(
        [
            "",
            "## Ablation Status",
            "",
            "- Done: historical strict baseline -> `sfrg05` main gain.",
            "- Done: `df40aux8` fake-side light branch vs heavier/lighter auxiliary sweeps and selective weighting.",
            "- Done: routing ladder `transferpattern -> transfertree_fixed78 -> transfertree_literal_fixed`.",
            "- Done: current-asset stability closure is now real rather than pending: three-seed rerun variance exists for `sfrg05` and `df40aux8`, and fixed-policy replay exists for `transfertree_literal_fixed`.",
            "- Missing: final claim-aligned external SOTA comparison table under one unified protocol.",
            f"- Missing: held-out paired `faceswap/fsgan` main-table evidence or an equivalent paired unseen-generator block. The local overlap audit shows they are train-only rather than fully unpairable (`faceswap {overlap_rows['faceswap']['ffpp_train_kept_pairs']}/{overlap_rows['faceswap']['ffpp_val_kept_pairs']}/{overlap_rows['faceswap']['ffpp_test_kept_pairs']}`, `fsgan {overlap_rows['fsgan']['ffpp_train_kept_pairs']}/{overlap_rows['fsgan']['ffpp_val_kept_pairs']}/{overlap_rows['fsgan']['ffpp_test_kept_pairs']}` across FF++ train/val/test), so the blocker is held-out coverage rather than bridge tooling.",
            "- Missing: one official deployment-stress benchmark.",
            "- Open: if the paper still wants a paired-oriented final row, it still needs a stability-safe version. The current replay package supports `transfertree_literal_fixed` as a frontier / diagnostic result, not as the stable final branch.",
            "",
            "## Immediate Next Step",
            "",
            "Use this package as the scaffold for the final paper-strength evidence bundle: keep `sfrg05_route_floor` as the stable main cross-dataset row, keep `df40aux8_route_floor` as the balanced light fake-side row, keep `df40aux8_transfertree_literal_fixed` only as the paired-oriented frontier / diagnostic row, and spend the next effort on the missing evidence blocks rather than more current-asset sweeps. The next priority is now clear: official held-out paired coverage for `faceswap/fsgan`, one deployment-stress benchmark, and the final unified SOTA / ablation / stability bundle.",
        ]
    )

    delta_path = REPORTS / "current_claim_package_delta_2026-04-20.tsv"
    md_path = REPORTS / "current_claim_package_2026-04-20.zh-CN.md"
    write_tsv(delta_path, delta_rows)
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "delta_tsv": str(delta_path),
                "report_md": str(md_path),
                "delta_rows": len(delta_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
