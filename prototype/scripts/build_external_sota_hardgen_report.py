from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def f(value: object) -> float:
    if value in (None, ""):
        return math.nan
    return float(value)


def subset_accuracy(rows: List[Dict[str, str]], label: str | None = None, protection: str | None = None) -> float:
    subset = [
        row
        for row in rows
        if row.get("effective_correct") in {"0", "1"}
        and (label is None or row.get("label") == label)
        and (protection is None or row.get("protection") == protection)
    ]
    if not subset:
        return math.nan
    return sum(int(row["effective_correct"]) for row in subset) / len(subset)


def routed_metrics(path: Path) -> dict:
    rows = load_csv(path)
    real = subset_accuracy(rows, label="real")
    fake = subset_accuracy(rows, label="fake")
    protected_real = subset_accuracy(rows, label="real", protection="protected")
    return {
        "balanced_accuracy": 0.5 * (real + fake),
        "fake_accuracy": fake,
        "real_accuracy": real,
        "protected_real_fpr": 1.0 - protected_real,
        "protected_fake_recall": subset_accuracy(rows, label="fake", protection="protected"),
        "unprotected_fake_recall": subset_accuracy(rows, label="fake", protection="unprotected"),
    }


def raw_passive_metrics(path: Path, threshold: float) -> dict:
    rows = [row for row in load_csv(path) if row.get("passive_prob_fake")]

    def acc(label: str | None = None, protection: str | None = None) -> float:
        subset = [
            row
            for row in rows
            if (label is None or row.get("label") == label)
            and (protection is None or row.get("protection") == protection)
        ]
        if not subset:
            return math.nan
        ok = 0
        for row in subset:
            pred = "fake" if float(row["passive_prob_fake"]) >= threshold else "real"
            ok += pred == row["label"]
        return ok / len(subset)

    real = acc("real")
    fake = acc("fake")
    protected_real = acc("real", "protected")
    return {
        "balanced_accuracy": 0.5 * (real + fake),
        "fake_accuracy": fake,
        "real_accuracy": real,
        "protected_real_fpr": 1.0 - protected_real,
        "protected_fake_recall": acc("fake", "protected"),
        "unprotected_fake_recall": acc("fake", "unprotected"),
    }


def fmt(value: float) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.6f}"


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
        writer.writerows(rows)


def external_rows(root: Path) -> List[dict]:
    ours = load_json(root / "prototype/reports/vcf_c23_temporal_f3_locked_sequence_pr4_seed_summary_mi2000.json")
    selected = ours["selected"]
    ba = [f(row["heldout_balanced_accuracy"]) for row in selected]
    pr_fpr = [f(row["heldout_protected_real_fpr"]) for row in selected]
    pf = [f(row["heldout_protected_fake_recall"]) for row in selected]
    uf = [f(row["heldout_unprotected_fake_recall"]) for row in selected]
    rows = [
        {
            "family": "ours",
            "model": "VCF temporal sequence head",
            "selection": "strict calib rule, 5 seeds",
            "heldout_balanced_accuracy": fmt(mean(ba)),
            "heldout_ba_std": fmt(pstdev(ba)),
            "heldout_protected_real_fpr": fmt(mean(pr_fpr)),
            "heldout_protected_fake_recall": fmt(mean(pf)),
            "heldout_unprotected_fake_recall": fmt(mean(uf)),
            "heldout_strict_fpr08": "1",
            "source": "vcf_c23_temporal_f3_locked_sequence_pr4_seed_summary_mi2000.json",
        }
    ]
    for model, filename in [
        ("DeepfakeBench UCF", "vcf_c23_temporal_f3_ucf_calibrated_summary_2026-04-22.json"),
        ("DeepfakeBench Xception", "vcf_c23_temporal_f3_xception_calibrated_summary_2026-04-22.json"),
        ("DeepfakeBench SPSL", "vcf_c23_temporal_f3_spsl_calibrated_summary_2026-04-22.json"),
    ]:
        summary = load_json(root / "prototype/reports" / filename)
        strict = summary["selections"]["best_by_calib_fpr08"]
        any_row = summary["selections"]["best_by_calib_any"]
        rows.append(
            {
                "family": "external_deepfakebench",
                "model": model,
                "selection": "best calib FPR08 row",
                "heldout_balanced_accuracy": strict["heldout_balanced_accuracy"],
                "heldout_ba_std": "",
                "heldout_protected_real_fpr": strict["heldout_protected_real_fpr"],
                "heldout_protected_fake_recall": strict["heldout_protected_fake_recall"],
                "heldout_unprotected_fake_recall": strict["heldout_unprotected_fake_recall"],
                "heldout_strict_fpr08": strict["heldout_strict_fpr08"],
                "diagnostic_best_any_ba": any_row["heldout_balanced_accuracy"],
                "diagnostic_best_any_pr_fpr": any_row["heldout_protected_real_fpr"],
                "source": filename,
            }
        )
    return rows


def paired_rows(root: Path) -> List[dict]:
    items = [
        (
            "simswap",
            "aux8_plain",
            "kept_reference",
            "df40_simswap_heldoutfull_ffpp_test_paired_sfrank_df40aux8_v0_1_routing.csv",
            None,
        ),
        (
            "simswap",
            "transfertree_literal_fixed",
            "kept_seed42_frontier",
            "df40_simswap_heldoutfull_ffpp_test_paired_transfertree_literal_fixed_v0_1_routing.csv",
            None,
        ),
        (
            "simswap",
            "hardgen_val_literal",
            "new_val_selected_routing",
            "df40_simswap_heldoutfull_ffpp_test_paired_hardgen_val_literal_v0_1_routing.csv",
            None,
        ),
        (
            "simswap",
            "hardgenpair16_default_rules",
            "new_model_branch_rejected",
            "df40_simswap_heldoutfull_ffpp_test_paired_hardgenpair16_v0_1_routing.csv",
            None,
        ),
        (
            "simswap",
            "hardgenpair16_raw_thr025",
            "diagnostic_not_strict",
            "df40_simswap_heldoutfull_ffpp_test_paired_hardgenpair16_v0_1_merged.csv",
            0.25,
        ),
        (
            "inswap",
            "aux8_plain",
            "kept_reference",
            "df40_inswap_heldoutfull_ffpp_test_paired_sfrank_df40aux8_v0_1_routing.csv",
            None,
        ),
        (
            "inswap",
            "transfertree_literal_fixed",
            "kept_seed42_frontier",
            "df40_inswap_heldoutfull_ffpp_test_paired_transfertree_literal_fixed_v0_1_routing.csv",
            None,
        ),
        (
            "inswap",
            "hardgen_val_literal",
            "new_val_selected_routing",
            "df40_inswap_heldoutfull_ffpp_test_paired_hardgen_val_literal_v0_1_routing.csv",
            None,
        ),
        (
            "inswap",
            "hardgenpair16_default_rules",
            "new_model_branch_rejected",
            "df40_inswap_heldoutfull_ffpp_test_paired_hardgenpair16_v0_1_routing.csv",
            None,
        ),
        (
            "inswap",
            "hardgenpair16_raw_thr025",
            "diagnostic_not_strict",
            "df40_inswap_heldoutfull_ffpp_test_paired_hardgenpair16_v0_1_merged.csv",
            0.25,
        ),
    ]
    rows = []
    for method, variant, status, filename, raw_threshold in items:
        path = root / "prototype/reports" / filename
        metrics = raw_passive_metrics(path, raw_threshold) if raw_threshold is not None else routed_metrics(path)
        rows.append(
            {
                "method": method,
                "variant": variant,
                "status": status,
                "balanced_accuracy": fmt(metrics["balanced_accuracy"]),
                "fake_accuracy": fmt(metrics["fake_accuracy"]),
                "real_accuracy": fmt(metrics["real_accuracy"]),
                "protected_real_fpr": fmt(metrics["protected_real_fpr"]),
                "protected_fake_recall": fmt(metrics["protected_fake_recall"]),
                "unprotected_fake_recall": fmt(metrics["unprotected_fake_recall"]),
                "raw_threshold": "" if raw_threshold is None else f"{raw_threshold:.2f}",
                "source": filename,
            }
        )
    return rows


def main() -> None:
    root = project_root()
    report_dir = root / "prototype/reports"
    external = external_rows(root)
    paired = paired_rows(root)
    hardgen_search = yaml.safe_load((root / "prototype/reports/route_label_rules_effort_hardgenpair16_fpr08_2026-04-22.yaml").read_text(encoding="utf-8"))
    hardgen_summary = hardgen_search["search_summary"]

    external_tsv = report_dir / "external_sota_breadth_vcf_temporal_2026-04-22.tsv"
    paired_tsv = report_dir / "hardgen_paired_strengthening_2026-04-22.tsv"
    write_tsv(external_tsv, external)
    write_tsv(paired_tsv, paired)

    md = report_dir / "external_sota_and_hardgen_paired_report_2026-04-22.zh-CN.md"
    lines = [
        "# External SOTA Breadth + Hard-Generator Paired Strengthening (2026-04-22)",
        "",
        "## 结论",
        "",
        "- 外部 breadth 已从 UCF 扩到 `UCF + Xception + SPSL` 三个官方 DeepfakeBench 检测器，并全部使用同一 VCF temporal calib/heldout 协议。",
        "- 三个外部检测器在 VCF heldout 上都没有达到 strict FPR08；最好的 strict-calib 行 BA 仍只有约 `0.508-0.532`，明显低于 locked sequence head 的 `0.931601 +/- 0.000220`。",
        "- simswap/inswap hard-generator paired 强化方面，val 选择的 literal routing 只带来小幅正增益；train-only hardgenpair16 表征分支在 raw `0.25` 阈值有 fake headroom，但 canonical strict FPR08 搜索无可行配置，因此不能作为 kept 主分支。",
        "",
        "## VCF Temporal External Breadth",
        "",
        "| model | selection | heldout BA | PR-FPR | PF recall | UF recall | strict |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in external:
        lines.append(
            f"| {row['model']} | {row['selection']} | {row['heldout_balanced_accuracy']} | "
            f"{row['heldout_protected_real_fpr']} | {row['heldout_protected_fake_recall']} | "
            f"{row['heldout_unprotected_fake_recall']} | {row['heldout_strict_fpr08']} |"
        )
    lines.extend(
        [
            "",
            "## simswap / inswap Paired Strengthening",
            "",
            "| method | variant | status | BA | fake | real | PR-FPR | PF | UF |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in paired:
        lines.append(
            f"| {row['method']} | {row['variant']} | {row['status']} | {row['balanced_accuracy']} | "
            f"{row['fake_accuracy']} | {row['real_accuracy']} | {row['protected_real_fpr']} | "
            f"{row['protected_fake_recall']} | {row['unprotected_fake_recall']} |"
        )
    lines.extend(
        [
            "",
            "## Hardgenpair16 Strict Gate",
            "",
            f"- selected config: `{hardgen_summary['selected_config_id']}`",
            f"- feasible: `{str(hardgen_summary['feasible']).lower()}`",
            f"- all-data BA: `{hardgen_summary['overall_balanced_accuracy']:.6f}`",
            f"- min dataset BA: `{hardgen_summary['min_dataset_balanced_accuracy']:.6f}`",
            f"- max protected-real FPR: `{hardgen_summary['max_protected_real_fpr']:.6f}`",
            f"- min protected fake recall: `{hardgen_summary['min_protected_fake_recall']:.6f}`",
            f"- min unprotected fake recall: `{hardgen_summary['min_unprotected_fake_recall']:.6f}`",
            "",
            "Interpretation: the paired source-family auxiliary objective did move simswap/inswap fake scores upward, but it also raised Celeb-DF-v2 protected-real FPR to `0.129557`. Under the user's strict main-result standard, this is a useful diagnostic and a mechanism clue, not a paper-facing main row.",
            "",
            "## Outputs",
            "",
            f"- `{external_tsv.relative_to(root)}`",
            f"- `{paired_tsv.relative_to(root)}`",
            "- `prototype/reports/vcf_c23_temporal_f3_xception_calibrated_summary_2026-04-22.json`",
            "- `prototype/reports/vcf_c23_temporal_f3_spsl_calibrated_summary_2026-04-22.json`",
            "- `prototype/reports/route_label_rules_sfrank_df40aux8_hardgen_val_literal_fpr08.yaml`",
            "- `prototype/reports/route_label_rules_effort_hardgenpair16_fpr08_2026-04-22.yaml`",
        ]
    )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(md), "external_tsv": str(external_tsv), "paired_tsv": str(paired_tsv)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
