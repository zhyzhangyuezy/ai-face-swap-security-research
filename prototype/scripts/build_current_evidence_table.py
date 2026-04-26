from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def get_nested(mapping: dict[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def build_rows() -> list[dict[str, Any]]:
    patchattn = load_yaml(REPORTS / "route_label_rules_effort_scaled_512_patchattn_mean_2026-04-18_fpr08.yaml")
    patchattn_lodo = load_json(REPORTS / "single_effort_lodo_scaled_512_patchattn_mean_2026-04-18_fpr08_summary.json")

    sfrg05 = load_yaml(REPORTS / "route_label_rules_sfrg05_fpr08.yaml")
    sfrg05_lodo = load_json(REPORTS / "single_effort_lodo_sfrg05_fpr08_routefloor_summary.json")

    df40aux8 = load_yaml(REPORTS / "route_label_rules_sfrank_df40aux8_fpr08.yaml")
    df40aux8_lodo = load_json(REPORTS / "single_effort_lodo_sfrank_df40aux8_fpr08_routefloor_summary.json")

    transferpattern = load_yaml(REPORTS / "route_label_rules_sfrank_df40aux8_transferpattern_fpr08.yaml")
    transferpattern_lodo = load_json(REPORTS / "single_passive_transferpattern_routefloor_lodo_2026-04-20_summary.json")

    transfertree = load_json(REPORTS / "route_label_rules_sfrank_df40aux8_transfertree_fpr08.json")
    transfertree_fixed78_lodo = load_json(REPORTS / "single_passive_transfertree_fixed78_lodo_2026-04-20_summary.json")

    transfertree_literal = load_json(REPORTS / "transfertree_literal_leaf_summary_2026-04-20.json")
    transfertree_literal_lodo = load_json(REPORTS / "single_passive_transfertree_literal_fixed_lodo_2026-04-20_summary.json")

    rows = [
        {
            "branch": "patchattn_mean_fpr08",
            "family": "historical_single_passive",
            "role": "historical_strict_baseline",
            "protocol": "canonical_fpr08 + lodo",
            "all_data_ba": get_nested(patchattn, "search_summary", "overall_balanced_accuracy"),
            "all_data_min_dataset_ba": get_nested(patchattn, "search_summary", "min_dataset_balanced_accuracy"),
            "all_data_max_protected_real_fpr": get_nested(patchattn, "search_summary", "max_protected_real_fpr"),
            "lodo_mean_ba": get_nested(patchattn_lodo, "aggregate", "mean_holdout_balanced_accuracy"),
            "lodo_min_ba": get_nested(patchattn_lodo, "aggregate", "min_holdout_balanced_accuracy"),
            "lodo_max_protected_real_fpr": None,
            "paired_mean_fake_ba": None,
            "paired_min_fake_ba": None,
            "paired_simswap_fake_ba": None,
            "paired_inswap_fake_ba": None,
            "note": "Previous strict single-passive baseline before source-family real-guard and DF40 auxiliary routing.",
        },
        {
            "branch": "sfrg05_route_floor",
            "family": "paired_routing_main",
            "role": "strongest_cross_dataset_ba",
            "protocol": "canonical_fpr08 + route_floor_lodo",
            "all_data_ba": get_nested(sfrg05, "search_summary", "overall_balanced_accuracy"),
            "all_data_min_dataset_ba": get_nested(sfrg05, "search_summary", "min_dataset_balanced_accuracy"),
            "all_data_max_protected_real_fpr": get_nested(sfrg05, "search_summary", "max_protected_real_fpr"),
            "lodo_mean_ba": get_nested(sfrg05_lodo, "aggregate", "mean_holdout_balanced_accuracy"),
            "lodo_min_ba": get_nested(sfrg05_lodo, "aggregate", "min_holdout_balanced_accuracy"),
            "lodo_max_protected_real_fpr": None,
            "paired_mean_fake_ba": None,
            "paired_min_fake_ba": None,
            "paired_simswap_fake_ba": None,
            "paired_inswap_fake_ba": None,
            "note": "Current strongest overall paired/routing BA branch under strict protected-real control.",
        },
        {
            "branch": "df40aux8_route_floor",
            "family": "fake_side_light",
            "role": "strongest_canonical_light_branch",
            "protocol": "canonical_fpr08 + route_floor_lodo",
            "all_data_ba": get_nested(df40aux8, "search_summary", "overall_balanced_accuracy"),
            "all_data_min_dataset_ba": get_nested(df40aux8, "search_summary", "min_dataset_balanced_accuracy"),
            "all_data_max_protected_real_fpr": get_nested(df40aux8, "search_summary", "max_protected_real_fpr"),
            "lodo_mean_ba": get_nested(df40aux8_lodo, "aggregate", "mean_holdout_balanced_accuracy"),
            "lodo_min_ba": get_nested(df40aux8_lodo, "aggregate", "min_holdout_balanced_accuracy"),
            "lodo_max_protected_real_fpr": None,
            "paired_mean_fake_ba": None,
            "paired_min_fake_ba": None,
            "paired_simswap_fake_ba": None,
            "paired_inswap_fake_ba": None,
            "note": "Current strongest admissible fake-side light branch before transfer-aware routing refinement.",
        },
        {
            "branch": "df40aux8_transferpattern_route_floor",
            "family": "paired_oriented_routing_explicit",
            "role": "explicit_fallback",
            "protocol": "canonical_fpr08 + paired_route_floor_lodo",
            "all_data_ba": get_nested(transferpattern, "search_summary", "canonical_overall_balanced_accuracy"),
            "all_data_min_dataset_ba": get_nested(transferpattern, "search_summary", "canonical_min_dataset_balanced_accuracy"),
            "all_data_max_protected_real_fpr": get_nested(transferpattern, "search_summary", "canonical_max_protected_real_fpr"),
            "lodo_mean_ba": get_nested(transferpattern_lodo, "aggregate", "mean_holdout_balanced_accuracy"),
            "lodo_min_ba": get_nested(transferpattern_lodo, "aggregate", "min_holdout_balanced_accuracy"),
            "lodo_max_protected_real_fpr": get_nested(transferpattern_lodo, "aggregate", "max_holdout_protected_real_fpr"),
            "paired_mean_fake_ba": get_nested(transferpattern, "search_summary", "paired_mean_fake_accuracy"),
            "paired_min_fake_ba": get_nested(transferpattern, "search_summary", "paired_min_fake_accuracy"),
            "paired_simswap_fake_ba": get_nested(transferpattern, "search_summary", "paired_datasets", "df40_simswap_ffpp_test_paired_aux8_v0_1", "fake_accuracy"),
            "paired_inswap_fake_ba": get_nested(transferpattern, "search_summary", "paired_datasets", "df40_inswap_ffpp_test_paired_aux8_v0_1", "fake_accuracy"),
            "note": "Best simpler explicit fallback before tree-based routing and literal recovery.",
        },
        {
            "branch": "df40aux8_transfertree_fixed78",
            "family": "paired_oriented_routing_learned",
            "role": "paper_friendly_learned_precursor",
            "protocol": "canonical_fpr08 + fixed_config_lodo",
            "all_data_ba": get_nested(transfertree, "search_summary", "canonical_overall_balanced_accuracy"),
            "all_data_min_dataset_ba": get_nested(transfertree, "search_summary", "canonical_min_dataset_balanced_accuracy"),
            "all_data_max_protected_real_fpr": get_nested(transfertree, "search_summary", "canonical_max_protected_real_fpr"),
            "lodo_mean_ba": get_nested(transfertree_fixed78_lodo, "aggregate", "mean_holdout_balanced_accuracy"),
            "lodo_min_ba": get_nested(transfertree_fixed78_lodo, "aggregate", "min_holdout_balanced_accuracy"),
            "lodo_max_protected_real_fpr": get_nested(transfertree_fixed78_lodo, "aggregate", "max_holdout_protected_real_fpr"),
            "paired_mean_fake_ba": get_nested(transfertree_fixed78_lodo, "aggregate", "mean_paired_mean_fake_accuracy"),
            "paired_min_fake_ba": get_nested(transfertree, "search_summary", "paired_min_fake_accuracy"),
            "paired_simswap_fake_ba": get_nested(transfertree, "search_summary", "paired_datasets", "df40_simswap_ffpp_test_paired_aux8_v0_1", "fake_accuracy"),
            "paired_inswap_fake_ba": get_nested(transfertree, "search_summary", "paired_datasets", "df40_inswap_ffpp_test_paired_aux8_v0_1", "fake_accuracy"),
            "note": "Best fixed learned routing branch that exposed the useful leaves for later explicit recovery.",
        },
        {
            "branch": "df40aux8_transfertree_literal_fixed",
            "family": "paired_oriented_routing_explicit",
            "role": "current_paper_friendly_frontier",
            "protocol": "canonical_fpr08 + fixed_explicit_lodo",
            "all_data_ba": get_nested(transfertree_literal, "best_canonical", "combined", "balanced_accuracy"),
            "all_data_min_dataset_ba": get_nested(transfertree_literal, "best_canonical", "min_dataset_balanced_accuracy"),
            "all_data_max_protected_real_fpr": get_nested(transfertree_literal, "best_canonical", "max_protected_real_fpr"),
            "lodo_mean_ba": get_nested(transfertree_literal_lodo, "aggregate", "mean_holdout_balanced_accuracy"),
            "lodo_min_ba": get_nested(transfertree_literal_lodo, "aggregate", "min_holdout_balanced_accuracy"),
            "lodo_max_protected_real_fpr": get_nested(transfertree_literal_lodo, "aggregate", "max_holdout_protected_real_fpr"),
            "paired_mean_fake_ba": get_nested(transfertree_literal_lodo, "aggregate", "mean_paired_mean_fake_accuracy"),
            "paired_min_fake_ba": get_nested(transfertree_literal, "best_paired", "min_fake_accuracy"),
            "paired_simswap_fake_ba": get_nested(transfertree_literal, "best_paired", "datasets", "df40_simswap_ffpp_test_paired_aux8_v0_1", "fake_accuracy"),
            "paired_inswap_fake_ba": get_nested(transfertree_literal, "best_paired", "datasets", "df40_inswap_ffpp_test_paired_aux8_v0_1", "fake_accuracy"),
            "note": "Exact explicit four-leaf recovery of the learned tree frontier; now the strongest paper-friendly paired-oriented routing branch.",
        },
    ]
    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(value) for key, value in row.items()})


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "Branch",
        "Role",
        "All BA",
        "Min BA",
        "Max PR-FPR",
        "LODO mean/min",
        "Paired mean/min fake BA",
        "Simswap / Inswap",
        "Note",
    ]
    lines = [
        "# Current Evidence Table",
        "",
        "This table tracks the current strict-FPR evidence package for the main historical baseline, the current strongest cross-dataset main, the kept fake-side light branch, and the paired-oriented routing frontiers.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lodo = ""
        if row["lodo_mean_ba"] is not None or row["lodo_min_ba"] is not None:
            lodo = f"{fmt(row['lodo_mean_ba'])} / {fmt(row['lodo_min_ba'])}"
        paired = ""
        if row["paired_mean_fake_ba"] is not None or row["paired_min_fake_ba"] is not None:
            paired = f"{fmt(row['paired_mean_fake_ba'])} / {fmt(row['paired_min_fake_ba'])}"
        methods = ""
        if row["paired_simswap_fake_ba"] is not None or row["paired_inswap_fake_ba"] is not None:
            methods = f"{fmt(row['paired_simswap_fake_ba'])} / {fmt(row['paired_inswap_fake_ba'])}"
        lines.append(
            "| "
            + " | ".join(
                [
                    row["branch"],
                    row["role"],
                    fmt(row["all_data_ba"]),
                    fmt(row["all_data_min_dataset_ba"]),
                    fmt(row["all_data_max_protected_real_fpr"]),
                    lodo,
                    paired,
                    methods,
                    row["note"],
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Key Findings",
            "",
            "1. `sfrg05_route_floor` is still the strongest branch for overall cross-dataset BA under the strict protected-real constraint.",
            "2. `df40aux8_route_floor` remains the strongest admissible light fake-side branch and the anchor for all paired-oriented routing refinements.",
            "3. `df40aux8_transfertree_literal_fixed` now dominates the previous paper-friendly learned routing branch and replaces it as the preferred paired-oriented explicit frontier.",
            "",
            "## Next Step",
            "",
            "Shift the next effort away from routing distillation and toward evidence completion: paired `faceswap/fsgan`, deployment-stress data, and final claim-aligned SOTA/ablation/stability tables built around these kept rows.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = build_rows()
    tsv_path = REPORTS / "current_evidence_table_2026-04-20.tsv"
    md_path = REPORTS / "current_evidence_table_2026-04-20.zh-CN.md"
    write_tsv(tsv_path, rows)
    write_markdown(md_path, rows)
    print(json.dumps({"tsv": str(tsv_path), "md": str(md_path), "row_count": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
