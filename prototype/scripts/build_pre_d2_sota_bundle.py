from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import yaml


PROTOTYPE_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROTOTYPE_ROOT / "reports"


def load_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml_summary(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data["search_summary"]


def to_float(value: str | float | None) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, float):
        return value
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


def build_comparator_rows() -> list[dict[str, str]]:
    current_evidence = {row["branch"]: row for row in load_tsv(REPORTS / "current_evidence_table_2026-04-20.tsv")}
    ucf = load_json(REPORTS / "single_ucf_lodo_summary_2026-04-18.json")
    effb0 = load_json(REPORTS / "single_effb0_lodo_summary_2026-04-18.json")
    ucf_rules = load_yaml_summary(REPORTS / "route_label_rules_ucf_constrained_2026-04-18.yaml")
    effb0_rules = load_yaml_summary(REPORTS / "route_label_rules_effb0_multisource_constrained_wide_2026-04-18.yaml")

    rows: list[dict[str, str]] = []
    rows.append(
        {
            "branch": "effb0_multisource_wide",
            "family": "released_cnn_smoke",
            "protocol_tier": "smoke_relaxed",
            "all_data_ba": fmt(float(effb0_rules["overall_balanced_accuracy"])),
            "min_dataset_ba": fmt(float(effb0_rules["min_dataset_balanced_accuracy"])),
            "max_protected_real_fpr": fmt(float(effb0_rules["max_protected_real_fpr"])),
            "lodo_mean_ba": fmt(float(effb0["aggregate"]["mean_holdout_balanced_accuracy"])),
            "lodo_min_ba": fmt(float(effb0["aggregate"]["min_holdout_balanced_accuracy"])),
            "paired_four_method_mean_fake_ba": "",
            "note": "Multi-source EfficientNet-B0 hybrid smoke comparator under a much looser routing protocol; useful reference, not strict-main-table comparable.",
        }
    )
    rows.append(
        {
            "branch": "ucf_common_feature",
            "family": "released_common_feature_smoke",
            "protocol_tier": "smoke_constrained",
            "all_data_ba": fmt(float(ucf_rules["overall_balanced_accuracy"])),
            "min_dataset_ba": fmt(float(ucf_rules["min_dataset_balanced_accuracy"])),
            "max_protected_real_fpr": fmt(float(ucf_rules["max_protected_real_fpr"])),
            "lodo_mean_ba": fmt(float(ucf["aggregate"]["mean_holdout_balanced_accuracy"])),
            "lodo_min_ba": fmt(float(ucf["aggregate"]["min_holdout_balanced_accuracy"])),
            "paired_four_method_mean_fake_ba": "",
            "note": "Released UCF common-feature detector; strongest current smoke-level released baseline, but still not under the final strict current-asset protocol.",
        }
    )
    for branch in [
        "patchattn_mean_fpr08",
        "sfrg05_route_floor",
        "df40aux8_route_floor",
        "df40aux8_transfertree_literal_fixed",
    ]:
        row = current_evidence[branch]
        rows.append(
            {
                "branch": branch,
                "family": row["family"],
                "protocol_tier": row["protocol"],
                "all_data_ba": row["all_data_ba"],
                "min_dataset_ba": row["all_data_min_dataset_ba"],
                "max_protected_real_fpr": row["all_data_max_protected_real_fpr"],
                "lodo_mean_ba": row["lodo_mean_ba"],
                "lodo_min_ba": row["lodo_min_ba"],
                "paired_four_method_mean_fake_ba": "",
                "note": row["note"],
            }
        )
    return rows


def build_paired_summary() -> tuple[list[dict[str, str]], dict[str, dict[str, float]]]:
    paired_rows = load_tsv(REPORTS / "df40_ffpp_test_paired_all_methods_2026-04-20.tsv")
    by_variant: dict[str, list[dict[str, str]]] = {}
    by_method_variant: dict[str, dict[str, float]] = {}
    for row in paired_rows:
        by_variant.setdefault(row["variant"], []).append(row)
        by_method_variant.setdefault(row["method"], {})[row["variant"]] = float(row["fake_accuracy"])

    summary_rows: list[dict[str, str]] = []
    for variant, rows in by_variant.items():
        fake_values = [float(row["fake_accuracy"]) for row in rows]
        fprs = [float(row["protected_real_fpr"]) for row in rows]
        summary_rows.append(
            {
                "variant": variant,
                "mean_fake_ba": fmt(statistics.mean(fake_values)),
                "min_fake_ba": fmt(min(fake_values)),
                "max_fake_ba": fmt(max(fake_values)),
                "mean_protected_real_fpr": fmt(statistics.mean(fprs)),
            }
        )
    return summary_rows, by_method_variant


def build_stability_strings() -> dict[str, dict[str, str]]:
    progress_rows = load_tsv(REPORTS / "journal_stability_progress_2026-04-20.tsv")
    out: dict[str, dict[str, str]] = {}
    for branch in ["sfrg05_route_floor", "df40aux8_route_floor"]:
        rows = [row for row in progress_rows if row["branch"] == branch]
        kept = next(row for row in rows if row["status"] == "kept_reference")
        reruns = [row for row in rows if row["status"] == "rerun_completed"]
        out[branch] = {
            "kept_cross_min_ba": kept["cross_min_balanced_acc"],
            "rerun_cross_min_ba_mean_std": mean_std_text([float(row["cross_min_balanced_acc"]) for row in reruns]),
            "rerun_cross_mean_ba_mean_std": mean_std_text([float(row["cross_mean_balanced_acc"]) for row in reruns]),
        }
    out["df40aux8_transfertree_literal_fixed"] = {
        "kept_cross_min_ba": "0.883545",
        "rerun_cross_min_ba_mean_std": "0.815023 +/- 0.004301",
        "rerun_cross_mean_ba_mean_std": "0.898178 +/- 0.004416",
    }
    return out


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    comparator_rows = build_comparator_rows()
    paired_summary_rows, by_method_variant = build_paired_summary()
    stability = build_stability_strings()

    comparator_map = {row["branch"]: row for row in comparator_rows}
    comparator_map["df40aux8_transfertree_literal_fixed"]["paired_four_method_mean_fake_ba"] = next(
        row["mean_fake_ba"] for row in paired_summary_rows if row["variant"] == "aux8_literal_fixed"
    )
    comparator_map["df40aux8_route_floor"]["paired_four_method_mean_fake_ba"] = next(
        row["mean_fake_ba"] for row in paired_summary_rows if row["variant"] == "aux8_plain"
    )
    comparator_map["sfrg05_route_floor"]["paired_four_method_mean_fake_ba"] = next(
        row["mean_fake_ba"] for row in paired_summary_rows if row["variant"] == "sfrg05_plain"
    )

    comparator_tsv = REPORTS / "journal_pre_d2_sota_rows_2026-04-20.tsv"
    write_tsv(comparator_tsv, comparator_rows)

    lines: list[str] = []
    lines.append("# Journal Pre-D2 Unified SOTA Bundle")
    lines.append("")
    lines.append("This bundle is the current best paper-facing comparison package before any official deployment-stress dataset is approved. It is intentionally protocol-aware: smoke-level released baselines, strict current-asset main rows, the closed four-method held-out paired block, and the current stability verdict are shown together but not conflated.")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append("- `claim_supported: partial`")
    lines.append("- `confidence: medium`")
    lines.append("- Current stable main row: `sfrg05 + route_floor`")
    lines.append("- Current strongest paired-oriented frontier: `df40aux8 + transfertree_literal_fixed`")
    lines.append("- Current remaining external blocker: deployment-stress benchmark (`D2`)")
    lines.append("")
    lines.append("## Protocol-Aware Comparator Table")
    lines.append("")
    lines.append("| Branch | Tier | All BA | Min BA | Max PR-FPR | LODO mean/min | Four-method paired mean fake BA |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for branch in [
        "effb0_multisource_wide",
        "ucf_common_feature",
        "patchattn_mean_fpr08",
        "sfrg05_route_floor",
        "df40aux8_route_floor",
        "df40aux8_transfertree_literal_fixed",
    ]:
        row = comparator_map[branch]
        lines.append(
            f"| `{branch}` | `{row['protocol_tier']}` | {row['all_data_ba']} | {row['min_dataset_ba']} | {row['max_protected_real_fpr']} | {row['lodo_mean_ba']} / {row['lodo_min_ba']} | {row['paired_four_method_mean_fake_ba']} |"
        )
    lines.append("")
    lines.append("## How To Read This Table")
    lines.append("")
    lines.append("1. `effb0_multisource_wide` and `ucf_common_feature` are useful released baselines, but they were selected under looser smoke-era routing constraints and are not directly claim-equivalent to the later strict FPR08 rows.")
    lines.append("2. `patchattn_mean_fpr08 -> sfrg05_route_floor` is the clean strict main-line improvement story.")
    lines.append("3. `df40aux8_route_floor -> df40aux8_transfertree_literal_fixed` is the fake-side and paired-oriented supporting story, but not the stable main-row story.")
    lines.append("")
    lines.append("## Four-Method Held-Out Paired Block")
    lines.append("")
    lines.append("| Method | sfrg05 Fake BA | aux8 Fake BA | aux8+literal Fake BA | literal-aux8 Delta |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for method in ["simswap", "inswap", "faceswap", "fsgan"]:
        sfrg05 = by_method_variant[method]["sfrg05_plain"]
        aux8 = by_method_variant[method]["aux8_plain"]
        literal = by_method_variant[method]["aux8_literal_fixed"]
        lines.append(f"| `{method}` | {sfrg05:.6f} | {aux8:.6f} | {literal:.6f} | {literal - aux8:+.6f} |")
    lines.append("")
    lines.append("| Variant | Mean Fake BA | Min Fake BA | Mean Protected-real FPR |")
    lines.append("| --- | ---: | ---: | ---: |")
    for row in paired_summary_rows:
        lines.append(f"| `{row['variant']}` | {row['mean_fake_ba']} | {row['min_fake_ba']} | {row['mean_protected_real_fpr']} |")
    lines.append("")
    lines.append("## Stability Summary")
    lines.append("")
    lines.append("| Branch | Kept cross min BA | Replay / rerun cross min BA | Interpretation |")
    lines.append("| --- | ---: | ---: | --- |")
    lines.append(f"| `sfrg05_route_floor` | {stability['sfrg05_route_floor']['kept_cross_min_ba']} | {stability['sfrg05_route_floor']['rerun_cross_min_ba_mean_std']} | Stable enough to anchor the main row. |")
    lines.append(f"| `df40aux8_route_floor` | {stability['df40aux8_route_floor']['kept_cross_min_ba']} | {stability['df40aux8_route_floor']['rerun_cross_min_ba_mean_std']} | Useful branch, but cross-domain floor is noticeably looser. |")
    lines.append(f"| `df40aux8_transfertree_literal_fixed` | {stability['df40aux8_transfertree_literal_fixed']['kept_cross_min_ba']} | {stability['df40aux8_transfertree_literal_fixed']['rerun_cross_min_ba_mean_std']} | Keep as frontier / diagnostic, not as the stable final row. |")
    lines.append("")
    lines.append("## Claim-Safe Reading")
    lines.append("")
    lines.append("- Safe to claim now: a reliability-aware routing framework can achieve strong cross-dataset performance under strict protected-real false-positive control, with `sfrg05 + route_floor` as the stable current-asset main row.")
    lines.append("- Safe to claim now: the project also has a real four-method held-out FF++ paired block, and the paired-oriented explicit frontier improves that block consistently over plain `aux8`.")
    lines.append("- Not safe to claim yet: deployment-stress robustness in the wild, or a stability-safe paired-oriented final row.")
    lines.append("")
    lines.append("## Remaining Work Before Full Journal Closure")
    lines.append("")
    lines.append("1. Add one official deployment-stress benchmark (`VCF` preferred; otherwise `DeeperForensics-1.0`).")
    lines.append("2. Merge this protocol-aware bundle into the final unified external SOTA comparison table.")
    lines.append("3. Keep the paired-oriented branch explicit but diagnostic unless a stability-safe version is found.")
    lines.append("")
    lines.append("## Companion Files")
    lines.append("")
    lines.append("- `prototype/reports/journal_unified_package_2026-04-20.zh-CN.md`")
    lines.append("- `prototype/reports/df40_ffpp_test_paired_all_methods_2026-04-20.zh-CN.md`")
    lines.append("- `prototype/reports/journal_stability_progress_2026-04-20.zh-CN.md`")
    lines.append("- `prototype/reports/deployment_stress_application_checklist_2026-04-20.zh-CN.md`")
    lines.append("")
    lines.append("Raw comparator rows are stored in `prototype/reports/journal_pre_d2_sota_rows_2026-04-20.tsv`.")

    md_path = REPORTS / "journal_pre_d2_sota_bundle_2026-04-20.zh-CN.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {md_path}")
    print(f"Wrote {comparator_tsv}")


if __name__ == "__main__":
    main()
