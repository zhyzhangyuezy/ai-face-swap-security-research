from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any


PROTOTYPE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PROTOTYPE_ROOT.parent
REPORTS = PROTOTYPE_ROOT / "reports"
RUNS = PROJECT_ROOT / "pilot" / "passive_baseline" / "runs"

OUTPUT_TSV = REPORTS / "journal_stability_progress_2026-04-20.tsv"
OUTPUT_MD = REPORTS / "journal_stability_progress_2026-04-20.zh-CN.md"

LITERAL_KEPT_SUMMARY = REPORTS / "single_passive_transfertree_literal_fixed_lodo_2026-04-20_summary.json"
LITERAL_RULES = REPORTS / "route_label_rules_sfrank_df40aux8_transfertree_literal_fpr08.yaml"
PIPELINE_SUMMARY = REPORTS / "effort_sfrank_df40aux8_2026-04-19_pipeline_summary.json"
FIXED_POLICY_REPLAY_SUMMARIES = {
    "7": REPORTS / "single_passive_transfertree_literal_fixed_seed7rerun_2026-04-20_lodo_summary.json",
    "21": REPORTS / "single_passive_transfertree_literal_fixed_seed21rerun_2026-04-20_lodo_summary.json",
    "84": REPORTS / "single_passive_transfertree_literal_fixed_seed84rerun_2026-04-20_lodo_summary.json",
}

FAMILY_SPECS: dict[str, dict[str, Any]] = {
    "df40aux8_route_floor": {
        "label": "df40aux8 + route_floor",
        "split_recipe": "effort_multisource_sourceaware_curriculum_v0_2 + df40 train8",
        "runs": [
            {
                "branch": "df40aux8_route_floor",
                "run_name": "effort_clipl_rank4_patchattn_mean_sfrank_df40aux8_w10_m15_e5_v1",
                "seed": "42",
                "status": "kept_reference",
                "note": "Original kept df40aux8 passive branch.",
            },
            {
                "branch": "df40aux8_route_floor",
                "run_name": "effort_clipl_rank4_patchattn_mean_sfrank_df40aux8_w10_m15_e5_seed7rerun_v1",
                "seed": "7",
                "status": "rerun_completed",
                "note": "First rerun; in-domain metrics stay strong, but the cross-domain floor drops materially.",
            },
            {
                "branch": "df40aux8_route_floor",
                "run_name": "effort_clipl_rank4_patchattn_mean_sfrank_df40aux8_w10_m15_e5_seed21rerun_v1",
                "seed": "21",
                "status": "rerun_completed",
                "note": "Second rerun; same pattern, with a weaker cross-domain floor than the kept seed-42 run.",
            },
            {
                "branch": "df40aux8_route_floor",
                "run_name": "effort_clipl_rank4_patchattn_mean_sfrank_df40aux8_w10_m15_e5_seed84rerun_v1",
                "seed": "84",
                "status": "rerun_completed",
                "note": "Third rerun; still stable in-domain, but worst-case cross-domain performance is weakest here.",
            },
        ],
    },
    "sfrg05_route_floor": {
        "label": "sfrg05 + route_floor",
        "split_recipe": "effort_multisource_sourceaware_curriculum_protreal_v0_4 + include_protected_real_train",
        "runs": [
            {
                "branch": "sfrg05_route_floor",
                "run_name": "effort_clipl_rank4_patchattn_mean_sfrank_realguard_w10_m15_g05_e5_v1",
                "seed": "42",
                "status": "kept_reference",
                "note": "Original kept sfrg05 passive branch.",
            },
            {
                "branch": "sfrg05_route_floor",
                "run_name": "effort_clipl_rank4_patchattn_mean_sfrank_realguard_w10_m15_g05_e5_seed7rerun_v1",
                "seed": "7",
                "status": "rerun_completed",
                "note": "First rerun; slightly weaker than kept seed-42, but the cross-domain floor remains close.",
            },
            {
                "branch": "sfrg05_route_floor",
                "run_name": "effort_clipl_rank4_patchattn_mean_sfrank_realguard_w10_m15_g05_e5_seed21rerun_v1",
                "seed": "21",
                "status": "rerun_completed",
                "note": "Second rerun; lower threshold and weaker DFDCP/Celeb-DF-v2 floor than the other seeds.",
            },
            {
                "branch": "sfrg05_route_floor",
                "run_name": "effort_clipl_rank4_patchattn_mean_sfrank_realguard_w10_m15_g05_e5_seed84rerun_v1",
                "seed": "84",
                "status": "rerun_completed",
                "note": "Third rerun; close to seed7, again much tighter than the df40aux8 family.",
            },
        ],
    },
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan")}
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0}
    return {"mean": statistics.mean(values), "std": statistics.stdev(values)}


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(value) if isinstance(value, float) or value is None else value for key, value in row.items()})


def collect_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, family_spec in FAMILY_SPECS.items():
        kept_run = family_spec["runs"][0]["run_name"]
        kept_metrics = load_json(RUNS / kept_run / "metrics.json")
        kept_test_ba = float(kept_metrics["test"]["balanced_acc"])
        kept_cross_min = float(kept_metrics["cross_summary"]["min_balanced_acc"])
        kept_cross_mean = float(kept_metrics["cross_summary"]["mean_balanced_acc"])
        for run_spec in family_spec["runs"]:
            metrics = load_json(RUNS / run_spec["run_name"] / "metrics.json")
            rows.append(
                {
                    "family": family,
                    "family_label": family_spec["label"],
                    "branch": run_spec["branch"],
                    "run_name": run_spec["run_name"],
                    "seed": run_spec["seed"],
                    "split_recipe": family_spec["split_recipe"],
                    "train_count": metrics["counts"]["train"],
                    "val_count": metrics["counts"]["val"],
                    "test_count": metrics["counts"]["test"],
                    "selected_threshold": metrics["selected_threshold"],
                    "val_balanced_acc": metrics["val"]["balanced_acc"],
                    "test_balanced_acc": metrics["test"]["balanced_acc"],
                    "cross_min_balanced_acc": metrics["cross_summary"]["min_balanced_acc"],
                    "cross_mean_balanced_acc": metrics["cross_summary"]["mean_balanced_acc"],
                    "elapsed_sec": metrics["elapsed_sec"],
                    "delta_test_ba_vs_kept": metrics["test"]["balanced_acc"] - kept_test_ba,
                    "delta_cross_min_ba_vs_kept": metrics["cross_summary"]["min_balanced_acc"] - kept_cross_min,
                    "delta_cross_mean_ba_vs_kept": metrics["cross_summary"]["mean_balanced_acc"] - kept_cross_mean,
                    "status": run_spec["status"],
                    "note": run_spec["note"],
                }
            )
    return rows


def summarize_family(rows: list[dict[str, Any]], family: str) -> dict[str, Any]:
    family_rows = [row for row in rows if row["family"] == family]
    kept = next(row for row in family_rows if row["seed"] == "42")
    reruns = [row for row in family_rows if row["seed"] != "42"]
    return {
        "kept": kept,
        "rerun_threshold": mean_std([float(row["selected_threshold"]) for row in reruns]),
        "rerun_val_ba": mean_std([float(row["val_balanced_acc"]) for row in reruns]),
        "rerun_test_ba": mean_std([float(row["test_balanced_acc"]) for row in reruns]),
        "rerun_cross_min_ba": mean_std([float(row["cross_min_balanced_acc"]) for row in reruns]),
        "rerun_cross_mean_ba": mean_std([float(row["cross_mean_balanced_acc"]) for row in reruns]),
        "rerun_elapsed_sec": mean_std([float(row["elapsed_sec"]) for row in reruns]),
        "rerun_delta_vs_kept": {
            "test_ba_mean_delta": statistics.mean(float(row["test_balanced_acc"]) for row in reruns)
            - float(kept["test_balanced_acc"]),
            "cross_min_ba_mean_delta": statistics.mean(float(row["cross_min_balanced_acc"]) for row in reruns)
            - float(kept["cross_min_balanced_acc"]),
            "cross_mean_ba_mean_delta": statistics.mean(float(row["cross_mean_balanced_acc"]) for row in reruns)
            - float(kept["cross_mean_balanced_acc"]),
        },
    }


def build_replay_rows() -> list[dict[str, Any]]:
    replay_rows: list[dict[str, Any]] = []
    for seed, path in FIXED_POLICY_REPLAY_SUMMARIES.items():
        replay_summary = load_json(path)
        replay_rows.append(
            {
                "seed": seed,
                "canonical_all_ba": replay_summary["all_data_canonical"]["combined"]["balanced_accuracy"],
                "canonical_min_ba": replay_summary["all_data_canonical"]["min_dataset_balanced_accuracy"],
                "canonical_max_pr_fpr": replay_summary["all_data_canonical"]["max_protected_real_fpr"],
                "paired_mean_fake_ba": replay_summary["all_data_paired"]["mean_fake_accuracy"],
                "lodo_mean_ba": replay_summary["aggregate"]["mean_holdout_balanced_accuracy"],
                "lodo_min_ba": replay_summary["aggregate"]["min_holdout_balanced_accuracy"],
            }
        )
    return replay_rows


def build_markdown(
    rows: list[dict[str, Any]],
    family_summaries: dict[str, dict[str, Any]],
    replay_rows: list[dict[str, Any]],
) -> str:
    literal_summary = load_json(LITERAL_KEPT_SUMMARY)
    literal_aggregate = literal_summary["aggregate"]
    df40_summary = family_summaries["df40aux8_route_floor"]
    sfrg05_summary = family_summaries["sfrg05_route_floor"]

    replay_mean_all = statistics.mean(row["canonical_all_ba"] for row in replay_rows)
    replay_std_all = statistics.stdev(row["canonical_all_ba"] for row in replay_rows)
    replay_mean_min = statistics.mean(row["canonical_min_ba"] for row in replay_rows)
    replay_std_min = statistics.stdev(row["canonical_min_ba"] for row in replay_rows)
    replay_mean_pair = statistics.mean(row["paired_mean_fake_ba"] for row in replay_rows)
    replay_std_pair = statistics.stdev(row["paired_mean_fake_ba"] for row in replay_rows)
    replay_mean_lodo = statistics.mean(row["lodo_mean_ba"] for row in replay_rows)
    replay_std_lodo = statistics.stdev(row["lodo_mean_ba"] for row in replay_rows)
    replay_mean_lodo_min = statistics.mean(row["lodo_min_ba"] for row in replay_rows)
    replay_std_lodo_min = statistics.stdev(row["lodo_min_ba"] for row in replay_rows)

    lines = [
        "# Journal Stability Progress",
        "",
        "This update closes the current-asset rerun matrix for both passive families. We now have: the three-seed `df40aux8` rerun package, the three fixed-policy replays for `transfertree_literal_fixed`, and the three-seed `sfrg05` rerun package.",
        "",
        "## Raw Table",
        "",
        "| family | seed | threshold | val BA | test BA | cross min BA | cross mean BA | elapsed sec |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        label = f"{row['family_label']} (kept)" if row["seed"] == "42" else f"{row['family_label']} (rerun)"
        lines.append(
            f"| {label} | {row['seed']} | {fmt(row['selected_threshold'])} | {fmt(row['val_balanced_acc'])} | "
            f"{fmt(row['test_balanced_acc'])} | {fmt(row['cross_min_balanced_acc'])} | "
            f"{fmt(row['cross_mean_balanced_acc'])} | {fmt(row['elapsed_sec'])} |"
        )

    lines.extend(
        [
            "",
            "## Family Aggregates",
            "",
            "| family | kept test BA | rerun test BA | kept cross min BA | rerun cross min BA | kept cross mean BA | rerun cross mean BA |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            (
                f"| df40aux8 + route_floor | {fmt(df40_summary['kept']['test_balanced_acc'])} | "
                f"{fmt(df40_summary['rerun_test_ba']['mean'])} +/- {fmt(df40_summary['rerun_test_ba']['std'])} | "
                f"{fmt(df40_summary['kept']['cross_min_balanced_acc'])} | "
                f"{fmt(df40_summary['rerun_cross_min_ba']['mean'])} +/- {fmt(df40_summary['rerun_cross_min_ba']['std'])} | "
                f"{fmt(df40_summary['kept']['cross_mean_balanced_acc'])} | "
                f"{fmt(df40_summary['rerun_cross_mean_ba']['mean'])} +/- {fmt(df40_summary['rerun_cross_mean_ba']['std'])} |"
            ),
            (
                f"| sfrg05 + route_floor | {fmt(sfrg05_summary['kept']['test_balanced_acc'])} | "
                f"{fmt(sfrg05_summary['rerun_test_ba']['mean'])} +/- {fmt(sfrg05_summary['rerun_test_ba']['std'])} | "
                f"{fmt(sfrg05_summary['kept']['cross_min_balanced_acc'])} | "
                f"{fmt(sfrg05_summary['rerun_cross_min_ba']['mean'])} +/- {fmt(sfrg05_summary['rerun_cross_min_ba']['std'])} | "
                f"{fmt(sfrg05_summary['kept']['cross_mean_balanced_acc'])} | "
                f"{fmt(sfrg05_summary['rerun_cross_mean_ba']['mean'])} +/- {fmt(sfrg05_summary['rerun_cross_mean_ba']['std'])} |"
            ),
            "",
            "## Fixed-Policy Replay",
            "",
            "| branch | canonical all BA | canonical min BA | max PR-FPR | paired mean fake BA | LODO mean/min BA |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            (
                f"| kept `transfertree_literal_fixed` (seed42 passive) | "
                f"{fmt(literal_summary['all_data_canonical']['combined']['balanced_accuracy'])} | "
                f"{fmt(literal_summary['all_data_canonical']['min_dataset_balanced_accuracy'])} | "
                f"{fmt(literal_summary['all_data_canonical']['max_protected_real_fpr'])} | "
                f"{fmt(literal_summary['all_data_paired']['mean_fake_accuracy'])} | "
                f"{fmt(literal_aggregate['mean_holdout_balanced_accuracy'])} / {fmt(literal_aggregate['min_holdout_balanced_accuracy'])} |"
            ),
        ]
    )
    for row in replay_rows:
        lines.append(
            f"| replay seed{row['seed']} | {fmt(row['canonical_all_ba'])} | {fmt(row['canonical_min_ba'])} | "
            f"{fmt(row['canonical_max_pr_fpr'])} | {fmt(row['paired_mean_fake_ba'])} | "
            f"{fmt(row['lodo_mean_ba'])} / {fmt(row['lodo_min_ba'])} |"
        )
    lines.extend(
        [
            "",
            "| replay aggregate | canonical all BA | canonical min BA | max PR-FPR | paired mean fake BA | LODO mean/min BA |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            (
                f"| rerun seeds `7/21/84` | {fmt(replay_mean_all)} +/- {fmt(replay_std_all)} | "
                f"{fmt(replay_mean_min)} +/- {fmt(replay_std_min)} | "
                f"{fmt(statistics.mean(row['canonical_max_pr_fpr'] for row in replay_rows))} +/- "
                f"{fmt(statistics.stdev(row['canonical_max_pr_fpr'] for row in replay_rows))} | "
                f"{fmt(replay_mean_pair)} +/- {fmt(replay_std_pair)} | "
                f"{fmt(replay_mean_lodo)} +/- {fmt(replay_std_lodo)} / "
                f"{fmt(replay_mean_lodo_min)} +/- {fmt(replay_std_lodo_min)} |"
            ),
            "",
            "## Key Findings",
            "",
            (
                "1. **Observation**: `df40aux8` reruns remain stable in-domain, but the cross-domain floor is still materially weaker than the kept seed-42 run. "
                f"The rerun mean is `test BA {fmt(df40_summary['rerun_test_ba']['mean'])} +/- {fmt(df40_summary['rerun_test_ba']['std'])}`, "
                f"yet `cross min BA` is only `{fmt(df40_summary['rerun_cross_min_ba']['mean'])} +/- {fmt(df40_summary['rerun_cross_min_ba']['std'])}` "
                f"versus kept `{fmt(df40_summary['kept']['cross_min_balanced_acc'])}`. "
                "**Interpretation**: the fake-side light family still has a real worst-case generalization stability gap. "
                "**Implication**: plain `df40aux8` remains useful, but not as the most stable paper-facing main row. "
                "**Next step**: keep it as the canonical balanced light branch and avoid centering the stability claim on it."
            ),
            "",
            (
                "2. **Observation**: the full three-seed fixed-policy replay is systematically weaker than the kept explicit frontier. "
                f"Replay means are `canonical all BA {fmt(replay_mean_all)} +/- {fmt(replay_std_all)}`, "
                f"`canonical min BA {fmt(replay_mean_min)} +/- {fmt(replay_std_min)}`, "
                f"`paired mean fake BA {fmt(replay_mean_pair)} +/- {fmt(replay_std_pair)}`, and "
                f"`LODO mean/min BA {fmt(replay_mean_lodo)} +/- {fmt(replay_std_lodo)} / {fmt(replay_mean_lodo_min)} +/- {fmt(replay_std_lodo_min)}`. "
                "**Interpretation**: `transfertree_literal_fixed` is not currently a stable final row; it is a kept-seed frontier and a strong paired-oriented diagnostic. "
                "**Implication**: paired-oriented gains remain publishable as frontier evidence, but should not anchor the main stability claim. "
                f"**Next step**: keep `{LITERAL_RULES.name}` and `{PIPELINE_SUMMARY.name}` for reproducible diagnostics, but shift the paper-facing main-row emphasis away from this branch."
            ),
            "",
            (
                "3. **Observation**: `sfrg05` is materially tighter than `df40aux8` under reruns. "
                f"The rerun mean is `test BA {fmt(sfrg05_summary['rerun_test_ba']['mean'])} +/- {fmt(sfrg05_summary['rerun_test_ba']['std'])}`, "
                f"`cross min BA {fmt(sfrg05_summary['rerun_cross_min_ba']['mean'])} +/- {fmt(sfrg05_summary['rerun_cross_min_ba']['std'])}`, and "
                f"`cross mean BA {fmt(sfrg05_summary['rerun_cross_mean_ba']['mean'])} +/- {fmt(sfrg05_summary['rerun_cross_mean_ba']['std'])}`; "
                f"the mean drop from the kept seed-42 run is only `{fmt(sfrg05_summary['rerun_delta_vs_kept']['cross_min_ba_mean_delta'])}` on cross min BA and "
                f"`{fmt(sfrg05_summary['rerun_delta_vs_kept']['cross_mean_ba_mean_delta'])}` on cross mean BA. "
                "**Interpretation**: this is now the strongest candidate for the stable cross-dataset main row on current assets. "
                "**Implication**: the paper-facing stability story should now center on `sfrg05 + route_floor`, not on the paired-oriented explicit frontier. "
                "**Next step**: update the claim package so `sfrg05` becomes the stable main row and the literal paired policy moves to frontier / appendix status."
            ),
            "",
            (
                "4. **Observation**: the current-asset rerun matrix is now complete for `P2`: `df40aux8` reruns, literal-policy replays, and `sfrg05` reruns are all in hand. "
                "**Interpretation**: the remaining evidence gaps are no longer current-asset stability mechanics. "
                "**Implication**: the project can now move from stability collection back to missing-evidence closure. "
                "**Next step**: focus on held-out paired `faceswap/fsgan`, one deployment-stress benchmark, and the final unified SOTA / ablation / stability bundle."
            ),
            "",
            "## Recommended Next Experiments",
            "",
            "1. Promote `sfrg05 + route_floor` to the main stable cross-dataset row in the claim package and move `df40aux8 + transfertree_literal_fixed` to a frontier / paired-oriented diagnostic role.",
            "2. Update the paper-facing evidence bundle so the stability section reflects both passive families and the negative fixed-policy replay result.",
            "3. Resume the blocked evidence tasks: held-out paired `faceswap/fsgan` and one official deployment-stress benchmark.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = collect_rows()
    family_summaries = {family: summarize_family(rows, family) for family in FAMILY_SPECS}
    replay_rows = build_replay_rows()
    write_tsv(OUTPUT_TSV, rows)
    OUTPUT_MD.write_text(build_markdown(rows, family_summaries, replay_rows), encoding="utf-8")
    print(str(OUTPUT_TSV))
    print(str(OUTPUT_MD))


if __name__ == "__main__":
    main()
