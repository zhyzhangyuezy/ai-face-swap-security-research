from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any


PROTOTYPE_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROTOTYPE_ROOT / "reports"


def load_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def to_float(value: str | float | None) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, float):
        return value
    return float(value)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(value) if isinstance(value, float) or value is None else value for key, value in row.items()})


def mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def mean_std_text(values: list[float]) -> str:
    mean_value, std_value = mean_std(values)
    return f"{mean_value:.6f} +/- {std_value:.6f}"


def branch_note(branch: str) -> str:
    notes = {
        "patchattn_mean_fpr08": "Historical strict single-passive baseline before source-family real-guard and DF40 auxiliary routing.",
        "sfrg05_route_floor": "Current stable cross-dataset main row on current assets after rerun validation.",
        "df40aux8_route_floor": "Current balanced light fake-side branch; useful, but clearly less stable than sfrg05 on worst-case cross-domain floor.",
        "df40aux8_transferpattern_route_floor": "Simpler explicit paired-oriented routing fallback that remains strict-safe but no longer defines the frontier.",
        "df40aux8_transfertree_fixed78": "Learned precursor that exposed the useful leaves later recovered as an explicit policy.",
        "df40aux8_transfertree_literal_fixed": "Kept-seed paired-oriented frontier and strong diagnostic, but not a stable final main row under replay.",
    }
    return notes[branch]


def displayed_role(branch: str) -> str:
    roles = {
        "patchattn_mean_fpr08": "historical_strict_baseline",
        "sfrg05_route_floor": "stable_main_row_current_assets",
        "df40aux8_route_floor": "balanced_light_branch",
        "df40aux8_transferpattern_route_floor": "explicit_fallback",
        "df40aux8_transfertree_fixed78": "learned_precursor",
        "df40aux8_transfertree_literal_fixed": "paired_oriented_frontier_diagnostic",
    }
    return roles[branch]


def build_main_rows(evidence: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for branch in [
        "patchattn_mean_fpr08",
        "sfrg05_route_floor",
        "df40aux8_route_floor",
        "df40aux8_transfertree_literal_fixed",
    ]:
        source = evidence[branch]
        rows.append(
            {
                "branch": branch,
                "role": displayed_role(branch),
                "all_data_ba": to_float(source["all_data_ba"]),
                "min_dataset_ba": to_float(source["all_data_min_dataset_ba"]),
                "max_protected_real_fpr": to_float(source["all_data_max_protected_real_fpr"]),
                "lodo_mean_ba": to_float(source["lodo_mean_ba"]),
                "lodo_min_ba": to_float(source["lodo_min_ba"]),
                "paired_mean_fake_ba": to_float(source["paired_mean_fake_ba"]),
                "paired_min_fake_ba": to_float(source["paired_min_fake_ba"]),
                "paired_simswap_fake_ba": to_float(source["paired_simswap_fake_ba"]),
                "paired_inswap_fake_ba": to_float(source["paired_inswap_fake_ba"]),
                "note": branch_note(branch),
            }
        )
    return rows


def build_ablation_rows(evidence: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    patchattn = evidence["patchattn_mean_fpr08"]
    transferpattern = evidence["df40aux8_transferpattern_route_floor"]
    rows: list[dict[str, Any]] = []

    for branch in [
        "sfrg05_route_floor",
        "df40aux8_route_floor",
        "df40aux8_transferpattern_route_floor",
        "df40aux8_transfertree_fixed78",
        "df40aux8_transfertree_literal_fixed",
    ]:
        source = evidence[branch]
        all_ba = to_float(source["all_data_ba"])
        min_ba = to_float(source["all_data_min_dataset_ba"])
        max_pr_fpr = to_float(source["all_data_max_protected_real_fpr"])
        lodo_mean = to_float(source["lodo_mean_ba"])
        lodo_min = to_float(source["lodo_min_ba"])
        paired_mean = to_float(source["paired_mean_fake_ba"])
        paired_simswap = to_float(source["paired_simswap_fake_ba"])
        paired_inswap = to_float(source["paired_inswap_fake_ba"])
        rows.append(
            {
                "branch": branch,
                "role": displayed_role(branch),
                "all_data_ba": all_ba,
                "delta_all_data_ba_vs_patchattn": all_ba - to_float(patchattn["all_data_ba"]),
                "delta_min_dataset_ba_vs_patchattn": min_ba - to_float(patchattn["all_data_min_dataset_ba"]),
                "delta_max_protected_real_fpr_vs_patchattn": max_pr_fpr - to_float(patchattn["all_data_max_protected_real_fpr"]),
                "lodo_mean_ba": lodo_mean,
                "lodo_min_ba": lodo_min,
                "paired_mean_fake_ba": paired_mean,
                "delta_paired_mean_fake_ba_vs_transferpattern": (
                    paired_mean - to_float(transferpattern["paired_mean_fake_ba"]) if paired_mean is not None else None
                ),
                "paired_simswap_fake_ba": paired_simswap,
                "paired_inswap_fake_ba": paired_inswap,
                "note": branch_note(branch),
            }
        )
    return rows


def build_family_rerun_summary(progress_rows: list[dict[str, str]], branch: str) -> dict[str, Any]:
    branch_rows = [row for row in progress_rows if row["branch"] == branch]
    kept = next(row for row in branch_rows if row["status"] == "kept_reference")
    reruns = [row for row in branch_rows if row["status"] == "rerun_completed"]

    rerun_test = [to_float(row["test_balanced_acc"]) for row in reruns]
    rerun_cross_min = [to_float(row["cross_min_balanced_acc"]) for row in reruns]
    rerun_cross_mean = [to_float(row["cross_mean_balanced_acc"]) for row in reruns]

    return {
        "kept_test_balanced_acc": to_float(kept["test_balanced_acc"]),
        "kept_cross_min_balanced_acc": to_float(kept["cross_min_balanced_acc"]),
        "kept_cross_mean_balanced_acc": to_float(kept["cross_mean_balanced_acc"]),
        "kept_lodo_mean_ba": to_float(kept["cross_mean_balanced_acc"]) if branch == "sfrg05_route_floor" else None,
        "kept_selected_threshold": to_float(kept["selected_threshold"]),
        "rerun_test_balanced_acc_mean_std": mean_std_text([value for value in rerun_test if value is not None]),
        "rerun_cross_min_balanced_acc_mean_std": mean_std_text([value for value in rerun_cross_min if value is not None]),
        "rerun_cross_mean_balanced_acc_mean_std": mean_std_text([value for value in rerun_cross_mean if value is not None]),
    }


def build_literal_replay_summary() -> dict[str, Any]:
    kept = load_json(REPORTS / "single_passive_transfertree_literal_fixed_lodo_2026-04-20_summary.json")
    replay_paths = [
        REPORTS / "single_passive_transfertree_literal_fixed_seed7rerun_2026-04-20_lodo_summary.json",
        REPORTS / "single_passive_transfertree_literal_fixed_seed21rerun_2026-04-20_lodo_summary.json",
        REPORTS / "single_passive_transfertree_literal_fixed_seed84rerun_2026-04-20_lodo_summary.json",
    ]
    replays = [load_json(path) for path in replay_paths]

    canonical_all = [float(item["all_data_canonical"]["combined"]["balanced_accuracy"]) for item in replays]
    canonical_min = [float(item["all_data_canonical"]["min_dataset_balanced_accuracy"]) for item in replays]
    canonical_max_pr_fpr = [float(item["all_data_canonical"]["max_protected_real_fpr"]) for item in replays]
    paired_mean_fake = [float(item["all_data_paired"]["mean_fake_accuracy"]) for item in replays]
    lodo_mean = [float(item["aggregate"]["mean_holdout_balanced_accuracy"]) for item in replays]
    lodo_min = [float(item["aggregate"]["min_holdout_balanced_accuracy"]) for item in replays]
    lodo_max_pr_fpr = [float(item["aggregate"]["max_holdout_protected_real_fpr"]) for item in replays]

    return {
        "kept_canonical_all_ba": float(kept["all_data_canonical"]["combined"]["balanced_accuracy"]),
        "kept_canonical_min_ba": float(kept["all_data_canonical"]["min_dataset_balanced_accuracy"]),
        "kept_canonical_max_pr_fpr": float(kept["all_data_canonical"]["max_protected_real_fpr"]),
        "kept_paired_mean_fake_ba": float(kept["all_data_paired"]["mean_fake_accuracy"]),
        "kept_lodo_mean_ba": float(kept["aggregate"]["mean_holdout_balanced_accuracy"]),
        "kept_lodo_min_ba": float(kept["aggregate"]["min_holdout_balanced_accuracy"]),
        "replay_canonical_all_ba_mean_std": mean_std_text(canonical_all),
        "replay_canonical_min_ba_mean_std": mean_std_text(canonical_min),
        "replay_canonical_max_pr_fpr_mean_std": mean_std_text(canonical_max_pr_fpr),
        "replay_paired_mean_fake_ba_mean_std": mean_std_text(paired_mean_fake),
        "replay_lodo_mean_ba_mean_std": mean_std_text(lodo_mean),
        "replay_lodo_min_ba_mean_std": mean_std_text(lodo_min),
        "replay_lodo_max_pr_fpr_mean_std": mean_std_text(lodo_max_pr_fpr),
    }


def build_stability_rows(evidence: dict[str, dict[str, str]], progress_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    patchattn = evidence["patchattn_mean_fpr08"]
    df40aux8 = build_family_rerun_summary(progress_rows, "df40aux8_route_floor")
    sfrg05 = build_family_rerun_summary(progress_rows, "sfrg05_route_floor")
    literal = build_literal_replay_summary()

    rows = [
        {
            "branch": "patchattn_mean_fpr08",
            "stability_position": "historical_reference_only",
            "kept_test_balanced_acc": to_float(patchattn["all_data_ba"]),
            "kept_cross_min_balanced_acc": to_float(patchattn["all_data_min_dataset_ba"]),
            "kept_cross_mean_balanced_acc": to_float(patchattn["lodo_mean_ba"]),
            "kept_lodo_mean_ba": to_float(patchattn["lodo_mean_ba"]),
            "kept_lodo_min_ba": to_float(patchattn["lodo_min_ba"]),
            "rerun_test_balanced_acc_mean_std": "",
            "rerun_cross_min_balanced_acc_mean_std": "",
            "rerun_cross_mean_balanced_acc_mean_std": "",
            "replay_canonical_all_ba_mean_std": "",
            "replay_canonical_min_ba_mean_std": "",
            "replay_paired_mean_fake_ba_mean_std": "",
            "replay_lodo_mean_ba_mean_std": "",
            "replay_lodo_min_ba_mean_std": "",
            "rerun_variance_available": "no",
            "note": "Historical baseline has only the kept run plus route-floor LODO; no multi-seed rerun package was opened.",
        },
        {
            "branch": "sfrg05_route_floor",
            "stability_position": "stable_main_row_current_assets",
            "kept_test_balanced_acc": sfrg05["kept_test_balanced_acc"],
            "kept_cross_min_balanced_acc": sfrg05["kept_cross_min_balanced_acc"],
            "kept_cross_mean_balanced_acc": sfrg05["kept_cross_mean_balanced_acc"],
            "kept_lodo_mean_ba": to_float(evidence["sfrg05_route_floor"]["lodo_mean_ba"]),
            "kept_lodo_min_ba": to_float(evidence["sfrg05_route_floor"]["lodo_min_ba"]),
            "rerun_test_balanced_acc_mean_std": sfrg05["rerun_test_balanced_acc_mean_std"],
            "rerun_cross_min_balanced_acc_mean_std": sfrg05["rerun_cross_min_balanced_acc_mean_std"],
            "rerun_cross_mean_balanced_acc_mean_std": sfrg05["rerun_cross_mean_balanced_acc_mean_std"],
            "replay_canonical_all_ba_mean_std": "",
            "replay_canonical_min_ba_mean_std": "",
            "replay_paired_mean_fake_ba_mean_std": "",
            "replay_lodo_mean_ba_mean_std": "",
            "replay_lodo_min_ba_mean_std": "",
            "rerun_variance_available": "yes",
            "note": "Three reruns are complete and remain reasonably tight, making sfrg05 the best stable main row on current assets.",
        },
        {
            "branch": "df40aux8_route_floor",
            "stability_position": "balanced_light_branch_not_main_stability_anchor",
            "kept_test_balanced_acc": df40aux8["kept_test_balanced_acc"],
            "kept_cross_min_balanced_acc": df40aux8["kept_cross_min_balanced_acc"],
            "kept_cross_mean_balanced_acc": df40aux8["kept_cross_mean_balanced_acc"],
            "kept_lodo_mean_ba": to_float(evidence["df40aux8_route_floor"]["lodo_mean_ba"]),
            "kept_lodo_min_ba": to_float(evidence["df40aux8_route_floor"]["lodo_min_ba"]),
            "rerun_test_balanced_acc_mean_std": df40aux8["rerun_test_balanced_acc_mean_std"],
            "rerun_cross_min_balanced_acc_mean_std": df40aux8["rerun_cross_min_balanced_acc_mean_std"],
            "rerun_cross_mean_balanced_acc_mean_std": df40aux8["rerun_cross_mean_balanced_acc_mean_std"],
            "replay_canonical_all_ba_mean_std": "",
            "replay_canonical_min_ba_mean_std": "",
            "replay_paired_mean_fake_ba_mean_std": "",
            "replay_lodo_mean_ba_mean_std": "",
            "replay_lodo_min_ba_mean_std": "",
            "rerun_variance_available": "yes",
            "note": "In-domain metrics stay strong, but the three-seed rerun package shows a materially weaker worst-case cross-domain floor.",
        },
        {
            "branch": "df40aux8_transfertree_literal_fixed",
            "stability_position": "paired_oriented_frontier_diagnostic_only",
            "kept_test_balanced_acc": df40aux8["kept_test_balanced_acc"],
            "kept_cross_min_balanced_acc": literal["kept_canonical_min_ba"],
            "kept_cross_mean_balanced_acc": literal["kept_canonical_all_ba"],
            "kept_lodo_mean_ba": literal["kept_lodo_mean_ba"],
            "kept_lodo_min_ba": literal["kept_lodo_min_ba"],
            "rerun_test_balanced_acc_mean_std": "",
            "rerun_cross_min_balanced_acc_mean_std": "",
            "rerun_cross_mean_balanced_acc_mean_std": "",
            "replay_canonical_all_ba_mean_std": literal["replay_canonical_all_ba_mean_std"],
            "replay_canonical_min_ba_mean_std": literal["replay_canonical_min_ba_mean_std"],
            "replay_paired_mean_fake_ba_mean_std": literal["replay_paired_mean_fake_ba_mean_std"],
            "replay_lodo_mean_ba_mean_std": literal["replay_lodo_mean_ba_mean_std"],
            "replay_lodo_min_ba_mean_std": literal["replay_lodo_min_ba_mean_std"],
            "rerun_variance_available": "yes_negative_policy_replay",
            "note": "The kept seed-42 explicit frontier is real, but fixed-policy replay on rerun seeds is systematically weaker, so this branch should stay diagnostic for now.",
        },
    ]
    return rows


def build_missing_tracker(overlap_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    overlap = {row["method"]: row for row in overlap_rows}
    return [
        {
            "evidence_gap": "heldout_paired_faceswap",
            "status": "blocked",
            "current_local_status": overlap["faceswap"]["local_status"],
            "current_overlap_counts": f"{overlap['faceswap']['ffpp_train_kept_pairs']}/{overlap['faceswap']['ffpp_val_kept_pairs']}/{overlap['faceswap']['ffpp_test_kept_pairs']}",
            "next_requirement": "official DF40 held-out testing zip (currently Google Drive quota-blocked) or an equivalent held-out paired unseen-generator benchmark",
            "priority": "critical",
            "note": "Local faceswap overlap exists only in FF++ train, not in held-out val/test; the official held-out testing zip is now individually identified but not yet locally downloadable because of quota.",
        },
        {
            "evidence_gap": "heldout_paired_fsgan",
            "status": "blocked",
            "current_local_status": overlap["fsgan"]["local_status"],
            "current_overlap_counts": f"{overlap['fsgan']['ffpp_train_kept_pairs']}/{overlap['fsgan']['ffpp_val_kept_pairs']}/{overlap['fsgan']['ffpp_test_kept_pairs']}",
            "next_requirement": "official DF40 held-out testing zip (currently Google Drive quota-blocked) or an equivalent held-out paired unseen-generator benchmark",
            "priority": "critical",
            "note": "Local fsgan overlap also stops at FF++ train; the official held-out testing zip is now individually identified but not yet locally downloadable because of quota.",
        },
        {
            "evidence_gap": "deployment_stress_benchmark",
            "status": "blocked",
            "current_local_status": "official_data_missing",
            "current_overlap_counts": "",
            "next_requirement": "one approved official benchmark block such as VCF or DeeperForensics-1.0",
            "priority": "critical",
            "note": "This is now the only major non-paired evidence block still missing on current assets.",
        },
        {
            "evidence_gap": "paired_oriented_final_row_stability",
            "status": "open",
            "current_local_status": "negative_fixed_policy_replay",
            "current_overlap_counts": "seed42 kept + replay seeds 7/21/84",
            "next_requirement": "either improve paired-oriented stability or keep the literal policy as a frontier / diagnostic result",
            "priority": "high",
            "note": "The replay package is complete and currently argues for claim narrowing, not for promotion to the stable main row.",
        },
        {
            "evidence_gap": "final_unified_sota_package",
            "status": "pending",
            "current_local_status": "current_asset_bundle_ready",
            "current_overlap_counts": "",
            "next_requirement": "merge the current-asset package with D1 and D2 once the blocked evidence lands",
            "priority": "critical",
            "note": "P0/P1/P2 are done; the last paper-facing bundle now depends mostly on the blocked external evidence.",
        },
    ]


def build_markdown(
    main_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
    stability_rows: list[dict[str, Any]],
    missing_rows: list[dict[str, Any]],
) -> str:
    sfrg05 = next(row for row in stability_rows if row["branch"] == "sfrg05_route_floor")
    df40aux8 = next(row for row in stability_rows if row["branch"] == "df40aux8_route_floor")
    literal = next(row for row in stability_rows if row["branch"] == "df40aux8_transfertree_literal_fixed")

    lines = [
        "# Journal Evidence Bundle",
        "",
        "This bundle now reflects the closed current-asset package: `P0` comparison, `P1` ablation, and `P2` stability are complete. The remaining blockers are no longer local rerun mechanics; they are the held-out paired `faceswap/fsgan` gap, one deployment-stress benchmark, and the final unified comparison package.",
        "",
        "## Raw Tables",
        "",
        "- `prototype/reports/journal_main_rows_2026-04-20.tsv`",
        "- `prototype/reports/journal_ablation_rows_2026-04-20.tsv`",
        "- `prototype/reports/journal_stability_rows_2026-04-20.tsv`",
        "- `prototype/reports/journal_missing_evidence_tracker_2026-04-20.tsv`",
        "",
        "## Main Rows",
        "",
        "| Branch | Role | All BA | Min BA | Max PR-FPR | LODO mean/min | Paired mean/min fake BA |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in main_rows:
        paired = ""
        if row["paired_mean_fake_ba"] is not None:
            paired = f"{fmt(row['paired_mean_fake_ba'])} / {fmt(row['paired_min_fake_ba'])}"
        lines.append(
            f"| {row['branch']} | {row['role']} | {fmt(row['all_data_ba'])} | {fmt(row['min_dataset_ba'])} | "
            f"{fmt(row['max_protected_real_fpr'])} | {fmt(row['lodo_mean_ba'])} / {fmt(row['lodo_min_ba'])} | {paired} |"
        )

    lines.extend(
        [
            "",
            "## Ablation Snapshot",
            "",
            "| Branch | Role | dAll BA vs patchattn | dMin BA vs patchattn | dMax PR-FPR vs patchattn | Paired mean fake BA | dPaired mean vs transferpattern |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in ablation_rows:
        lines.append(
            f"| {row['branch']} | {row['role']} | {fmt(row['delta_all_data_ba_vs_patchattn'])} | "
            f"{fmt(row['delta_min_dataset_ba_vs_patchattn'])} | {fmt(row['delta_max_protected_real_fpr_vs_patchattn'])} | "
            f"{fmt(row['paired_mean_fake_ba'])} | {fmt(row['delta_paired_mean_fake_ba_vs_transferpattern'])} |"
        )

    lines.extend(
        [
            "",
            "## Stability Snapshot",
            "",
            "| Branch | Position | Current signal | Interpretation |",
            "| --- | --- | --- | --- |",
            f"| sfrg05_route_floor | stable main row | rerun test BA {sfrg05['rerun_test_balanced_acc_mean_std']}; rerun cross min BA {sfrg05['rerun_cross_min_balanced_acc_mean_std']}; rerun cross mean BA {sfrg05['rerun_cross_mean_balanced_acc_mean_std']} | This is now the tightest passive family on current assets and should anchor the paper-facing main row. |",
            f"| df40aux8_route_floor | balanced light branch | rerun test BA {df40aux8['rerun_test_balanced_acc_mean_std']}; rerun cross min BA {df40aux8['rerun_cross_min_balanced_acc_mean_std']}; rerun cross mean BA {df40aux8['rerun_cross_mean_balanced_acc_mean_std']} | In-domain behavior is stable, but the worst-case cross-domain floor is materially weaker than the kept seed-42 run. |",
            f"| df40aux8_transfertree_literal_fixed | paired-oriented frontier / diagnostic | replay canonical all BA {literal['replay_canonical_all_ba_mean_std']}; replay canonical min BA {literal['replay_canonical_min_ba_mean_std']}; replay paired mean fake BA {literal['replay_paired_mean_fake_ba_mean_std']}; replay LODO mean/min BA {literal['replay_lodo_mean_ba_mean_std']} / {literal['replay_lodo_min_ba_mean_std']} | The kept seed-42 frontier is real, but fixed-policy replay is systematically weaker, so this should stay diagnostic for now. |",
            "",
            "## Missing Evidence Tracker",
            "",
            "| Gap | Status | Current local status | Counts | Next requirement |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in missing_rows:
        lines.append(
            f"| {row['evidence_gap']} | {row['status']} | {row['current_local_status']} | {row['current_overlap_counts']} | {row['next_requirement']} |"
        )

    lines.extend(
        [
            "",
            "## Key Findings",
            "",
            "1. `sfrg05 + route_floor` should now be treated as the stable paper-facing main row on current local assets. The rerun package is complete and materially tighter than `df40aux8` on worst-case cross-domain behavior.",
            "2. `df40aux8 + route_floor` still matters: it remains the balanced light fake-side branch and continues to support the DF40 diversity story, but it should not anchor the main stability claim.",
            "3. `df40aux8 + transfertree_literal_fixed` remains valuable, but its role is now narrower: it is the strongest paired-oriented kept-seed frontier and a strong diagnostic, not the stable final row.",
            "4. The remaining hard blockers are now external evidence blocks rather than more current-asset sweeps: held-out paired `faceswap/fsgan`, one official deployment-stress benchmark, and the final unified comparison package that absorbs those blocks.",
            "",
            "## Suggested Next Experiments",
            "",
            "1. Keep `sfrg05 + route_floor` as the main row, keep plain `df40aux8 + route_floor` as the balanced light row, and keep `df40aux8 + transfertree_literal_fixed` only as the paired-oriented frontier / diagnostic branch in paper-facing summaries.",
            "2. Resume `D1`: obtain held-out paired `faceswap/fsgan` coverage through official DF40 original/test assets or an equivalent paired unseen-generator benchmark, rather than spending more time on bridge plumbing.",
            "3. Resume `D2` and `P3`: add one official deployment-stress benchmark, then merge the current-asset package into the final unified SOTA / ablation / stability bundle.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    evidence_rows = load_tsv(REPORTS / "current_evidence_table_2026-04-20.tsv")
    evidence = {row["branch"]: row for row in evidence_rows}
    progress_rows = load_tsv(REPORTS / "journal_stability_progress_2026-04-20.tsv")
    overlap_rows = load_tsv(REPORTS / "df40_ffpp_paired_overlap_audit_2026-04-20.tsv")

    main_rows = build_main_rows(evidence)
    ablation_rows = build_ablation_rows(evidence)
    stability_rows = build_stability_rows(evidence, progress_rows)
    missing_rows = build_missing_tracker(overlap_rows)

    main_path = REPORTS / "journal_main_rows_2026-04-20.tsv"
    ablation_path = REPORTS / "journal_ablation_rows_2026-04-20.tsv"
    stability_path = REPORTS / "journal_stability_rows_2026-04-20.tsv"
    missing_path = REPORTS / "journal_missing_evidence_tracker_2026-04-20.tsv"
    md_path = REPORTS / "journal_evidence_bundle_2026-04-20.zh-CN.md"

    write_tsv(main_path, main_rows)
    write_tsv(ablation_path, ablation_rows)
    write_tsv(stability_path, stability_rows)
    write_tsv(missing_path, missing_rows)
    md_path.write_text(build_markdown(main_rows, ablation_rows, stability_rows, missing_rows), encoding="utf-8")

    print(
        json.dumps(
            {
                "journal_main_rows": str(main_path),
                "journal_ablation_rows": str(ablation_path),
                "journal_stability_rows": str(stability_path),
                "journal_missing_tracker": str(missing_path),
                "journal_bundle_md": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
