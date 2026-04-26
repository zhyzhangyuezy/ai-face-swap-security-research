from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean
from typing import Any


PROTOTYPE_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROTOTYPE_ROOT / "reports"

import sys

if str(PROTOTYPE_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROTOTYPE_ROOT / "scripts"))

from search_routing_thresholds_constrained import accuracy_metrics


BRANCHES = [
    {
        "name": "sfrg05_route_floor",
        "label": "sfrg05 + route_floor",
        "routing_csv": REPORTS / "deeperforensics_ffpp_test_full_v0_1_sfrg05_routing.csv",
        "position": "stable_main_row",
    },
    {
        "name": "df40aux8_route_floor",
        "label": "df40aux8 + route_floor",
        "routing_csv": REPORTS / "deeperforensics_ffpp_test_full_v0_1_sfrank_df40aux8_routing.csv",
        "position": "balanced_light_branch",
    },
    {
        "name": "df40aux8_transfertree_literal_fixed",
        "label": "df40aux8 + transfertree_literal_fixed",
        "routing_csv": REPORTS / "deeperforensics_ffpp_test_full_v0_1_sfrank_df40aux8_literalfixed_routing.csv",
        "position": "paired_oriented_frontier",
    },
]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def subset_name(row: dict[str, str]) -> str | None:
    if row.get("label") != "fake":
        return None
    return Path(row["source_path"]).parent.name


def build_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    subset_rows: list[dict[str, Any]] = []

    for branch in BRANCHES:
        rows = load_csv(branch["routing_csv"])
        metrics = accuracy_metrics(rows)
        subsets = sorted({subset_name(row) for row in rows if subset_name(row)})

        subset_fake_accs: list[float] = []
        for subset in subsets:
            subset_eval_rows = [row for row in rows if row.get("label") == "real" or subset_name(row) == subset]
            subset_metrics = accuracy_metrics(subset_eval_rows)
            subset_fake_accs.append(subset_metrics["fake_accuracy"])
            subset_rows.append(
                {
                    "branch": branch["name"],
                    "label": branch["label"],
                    "subset": subset,
                    "balanced_accuracy": subset_metrics["balanced_accuracy"],
                    "fake_accuracy": subset_metrics["fake_accuracy"],
                    "protected_real_fpr": subset_metrics["protected_real_fpr"],
                    "protected_fake_recall": subset_metrics["protected_fake_recall"],
                    "unprotected_fake_recall": subset_metrics["unprotected_fake_recall"],
                }
            )

        summary_rows.append(
            {
                "branch": branch["name"],
                "label": branch["label"],
                "position": branch["position"],
                "overall_balanced_accuracy": metrics["balanced_accuracy"],
                "overall_accuracy": metrics["accuracy"],
                "overall_fake_accuracy": metrics["fake_accuracy"],
                "protected_real_fpr": metrics["protected_real_fpr"],
                "protected_fake_recall": metrics["protected_fake_recall"],
                "unprotected_fake_recall": metrics["unprotected_fake_recall"],
                "subset_mean_fake_accuracy": mean(subset_fake_accs),
                "subset_min_fake_accuracy": min(subset_fake_accs),
                "subset_max_fake_accuracy": max(subset_fake_accs),
            }
        )

    return summary_rows, subset_rows


def main() -> None:
    summary_rows, subset_rows = build_rows()
    summary_path = REPORTS / "deeperforensics_stress_summary_2026-04-20.tsv"
    subset_path = REPORTS / "deeperforensics_stress_subset_rows_2026-04-20.tsv"
    write_tsv(summary_path, summary_rows)
    write_tsv(subset_path, subset_rows)

    summary_map = {row["branch"]: row for row in summary_rows}
    subset_map: dict[str, list[dict[str, Any]]] = {}
    for row in subset_rows:
        subset_map.setdefault(row["branch"], []).append(row)

    lines: list[str] = []
    lines.append("# DeeperForensics-1.0 Deployment-Stress Report")
    lines.append("")
    lines.append("This report evaluates the current kept branches on local `DeeperForensics-1.0` manipulated videos plus local `FaceForensics++` real anchors. Concretely, each manipulated video contributes one extracted frame, the official `test.txt` split defines the held-out target-video ids, and the real side reuses the matching FF++ `c23` target-video frames as recommended by the dataset's detection setup.")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append("- `D2: completed` on current assets with `DeeperForensics-1.0` official manipulated test split.")
    lines.append("- `sfrg05 + route_floor` remains the strongest stable main row under deployment stress.")
    lines.append("- `df40aux8 + transfertree_literal_fixed` still improves over plain `aux8`, but it does not beat `sfrg05` on this deployment-stress benchmark.")
    lines.append("")
    lines.append("## Overall Summary")
    lines.append("")
    lines.append("| Branch | Role | Overall BA | Fake Acc | PR-FPR | Protected fake recall | Unprotected fake recall | Mean/min/max subset fake acc |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in summary_rows:
        lines.append(
            f"| `{row['label']}` | `{row['position']}` | {fmt(row['overall_balanced_accuracy'])} | {fmt(row['overall_fake_accuracy'])} | {fmt(row['protected_real_fpr'])} | {fmt(row['protected_fake_recall'])} | {fmt(row['unprotected_fake_recall'])} | {fmt(row['subset_mean_fake_accuracy'])} / {fmt(row['subset_min_fake_accuracy'])} / {fmt(row['subset_max_fake_accuracy'])} |"
        )
    lines.append("")
    lines.append("## Distortion-Subset Breakdown")
    lines.append("")
    lines.append("| Subset | sfrg05 Fake Acc | aux8 Fake Acc | aux8+literal Fake Acc | literal-sfrg05 Delta |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    sfrg05_rows = {row["subset"]: row for row in subset_map["sfrg05_route_floor"]}
    aux8_rows = {row["subset"]: row for row in subset_map["df40aux8_route_floor"]}
    literal_rows = {row["subset"]: row for row in subset_map["df40aux8_transfertree_literal_fixed"]}
    for subset in sorted(sfrg05_rows):
        sfrg05_fake = float(sfrg05_rows[subset]["fake_accuracy"])
        aux8_fake = float(aux8_rows[subset]["fake_accuracy"])
        literal_fake = float(literal_rows[subset]["fake_accuracy"])
        lines.append(
            f"| `{subset}` | {sfrg05_fake:.6f} | {aux8_fake:.6f} | {literal_fake:.6f} | {literal_fake - sfrg05_fake:+.6f} |"
        )
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    lines.append("1. `sfrg05 + route_floor` is the clean winner on this benchmark: it has the best overall BA, the best fake accuracy, and the lowest protected-real FPR.")
    lines.append("2. `df40aux8 + transfertree_literal_fixed` still beats plain `aux8`, so the paired-oriented routing signal is not fake; it just does not dominate the stable main line under DeeperForensics stress.")
    lines.append("3. The hardest subsets remain the mixed-distortion buckets, especially `end_to_end_mix_2_distortions` and `end_to_end_mix_4_distortions`.")
    lines.append("")
    lines.append("## Companion Files")
    lines.append("")
    lines.append("- `prototype/reports/deeperforensics_stress_summary_2026-04-20.tsv`")
    lines.append("- `prototype/reports/deeperforensics_stress_subset_rows_2026-04-20.tsv`")
    lines.append("- `prototype/reports/deeperforensics_ffpp_test_full_v0_1_manifest_summary.json`")
    lines.append("")

    md_path = REPORTS / "deeperforensics_stress_report_2026-04-20.zh-CN.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {md_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {subset_path}")


if __name__ == "__main__":
    main()
