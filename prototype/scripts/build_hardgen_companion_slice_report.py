from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def safe_float(value: object) -> float:
    if value in (None, ""):
        return math.nan
    return float(value)


def fmt(value: float) -> str:
    return "" if math.isnan(value) else f"{value:.6f}"


def subset_accuracy(rows: Iterable[Dict[str, str]], label: str, protection: str | None = None) -> float:
    subset = [
        row
        for row in rows
        if row.get("effective_correct") in {"0", "1"}
        and row.get("label") == label
        and (protection is None or row.get("protection") == protection)
    ]
    if not subset:
        return math.nan
    return sum(int(row["effective_correct"]) for row in subset) / len(subset)


def routed_metrics(path: Path) -> dict:
    rows = load_csv(path)
    real = subset_accuracy(rows, "real")
    fake = subset_accuracy(rows, "fake")
    protected_real = subset_accuracy(rows, "real", "protected")
    return {
        "balanced_accuracy": 0.5 * (real + fake),
        "fake_recall": fake,
        "real_accuracy": real,
        "protected_real_fpr": 1.0 - protected_real,
        "source": path.name,
    }


def raw_metrics(path: Path, threshold: float) -> dict:
    rows = [row for row in load_csv(path) if row.get("passive_prob_fake")]

    def acc(label: str, protection: str | None = None) -> float:
        subset = [
            row
            for row in rows
            if row.get("label") == label and (protection is None or row.get("protection") == protection)
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
        "fake_recall": fake,
        "real_accuracy": real,
        "protected_real_fpr": 1.0 - protected_real,
        "source": path.name,
    }


def strict_summary(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    summary = data["search_summary"]
    return {
        "strict_gate": path.stem.replace("route_label_rules_effort_", ""),
        "feasible": str(summary["feasible"]).lower(),
        "overall_balanced_accuracy": f"{summary['overall_balanced_accuracy']:.6f}",
        "min_dataset_balanced_accuracy": f"{summary['min_dataset_balanced_accuracy']:.6f}",
        "max_protected_real_fpr": f"{summary['max_protected_real_fpr']:.6f}",
        "min_protected_fake_recall": f"{summary['min_protected_fake_recall']:.6f}",
        "min_unprotected_fake_recall": f"{summary['min_unprotected_fake_recall']:.6f}",
        "selected_config_id": str(summary["selected_config_id"]),
        "source": path.name,
    }


def write_tsv(path: Path, rows: List[dict]) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def paired_rows(root: Path) -> List[dict]:
    report_dir = root / "prototype" / "reports"
    specs = [
        ("simswap", "hardgenpair16_no_guard", "routing", "df40_simswap_heldoutfull_ffpp_test_paired_hardgenpair16_v0_1_routing.csv", None),
        ("simswap", "hardgenpair16_no_guard", "raw025", "df40_simswap_heldoutfull_ffpp_test_paired_hardgenpair16_v0_1_merged.csv", 0.25),
        ("simswap", "hardgenpair16_protpatchg005", "routing", "df40_simswap_heldoutfull_ffpp_test_paired_hardgenpair16_protpatchg005_v0_1_routing.csv", None),
        ("simswap", "hardgenpair16_protpatchg005", "raw025", "df40_simswap_heldoutfull_ffpp_test_paired_hardgenpair16_protpatchg005_v0_1_merged.csv", 0.25),
        ("simswap", "hardgenpair32_protpatchg005", "routing", "df40_simswap_heldoutfull_ffpp_test_paired_hardgenpair32_protpatchg005_v0_1_routing.csv", None),
        ("simswap", "hardgenpair32_protpatchg005", "raw025", "df40_simswap_heldoutfull_ffpp_test_paired_hardgenpair32_protpatchg005_v0_1_merged.csv", 0.25),
        ("simswap", "hardgenpair32_protpatchg010", "routing", "df40_simswap_heldoutfull_ffpp_test_paired_hardgenpair32_protpatchg010_v0_1_routing.csv", None),
        ("simswap", "hardgenpair32_protpatchg010", "raw025", "df40_simswap_heldoutfull_ffpp_test_paired_hardgenpair32_protpatchg010_v0_1_merged.csv", 0.25),
        ("inswap", "hardgenpair16_no_guard", "routing", "df40_inswap_heldoutfull_ffpp_test_paired_hardgenpair16_v0_1_routing.csv", None),
        ("inswap", "hardgenpair16_no_guard", "raw025", "df40_inswap_heldoutfull_ffpp_test_paired_hardgenpair16_v0_1_merged.csv", 0.25),
        ("inswap", "hardgenpair16_protpatchg005", "routing", "df40_inswap_heldoutfull_ffpp_test_paired_hardgenpair16_protpatchg005_v0_1_routing.csv", None),
        ("inswap", "hardgenpair16_protpatchg005", "raw025", "df40_inswap_heldoutfull_ffpp_test_paired_hardgenpair16_protpatchg005_v0_1_merged.csv", 0.25),
        ("inswap", "hardgenpair32_protpatchg005", "routing", "df40_inswap_heldoutfull_ffpp_test_paired_hardgenpair32_protpatchg005_v0_1_routing.csv", None),
        ("inswap", "hardgenpair32_protpatchg005", "raw025", "df40_inswap_heldoutfull_ffpp_test_paired_hardgenpair32_protpatchg005_v0_1_merged.csv", 0.25),
        ("inswap", "hardgenpair32_protpatchg010", "routing", "df40_inswap_heldoutfull_ffpp_test_paired_hardgenpair32_protpatchg010_v0_1_routing.csv", None),
        ("inswap", "hardgenpair32_protpatchg010", "raw025", "df40_inswap_heldoutfull_ffpp_test_paired_hardgenpair32_protpatchg010_v0_1_merged.csv", 0.25),
    ]
    rows = []
    for method, variant, view, filename, threshold in specs:
        path = report_dir / filename
        metrics = raw_metrics(path, threshold) if threshold is not None else routed_metrics(path)
        rows.append(
            {
                "method": method,
                "variant": variant,
                "view": view,
                "balanced_accuracy": fmt(metrics["balanced_accuracy"]),
                "fake_recall": fmt(metrics["fake_recall"]),
                "real_accuracy": fmt(metrics["real_accuracy"]),
                "protected_real_fpr": fmt(metrics["protected_real_fpr"]),
                "source": metrics["source"],
            }
        )
    return rows


def main() -> None:
    root = project_root()
    report_dir = root / "prototype" / "reports"
    paired = paired_rows(root)
    strict = [
        strict_summary(report_dir / "route_label_rules_effort_hardgenpair16_fpr08_2026-04-22.yaml"),
        strict_summary(report_dir / "route_label_rules_effort_hardgenpair16_protpatchg005_fpr08_2026-04-22.yaml"),
        strict_summary(report_dir / "route_label_rules_effort_hardgenpair32_protpatchg005_fpr08_2026-04-22.yaml"),
    ]

    paired_tsv = report_dir / "hardgen_companion_slice_paired_2026-04-22.tsv"
    strict_tsv = report_dir / "hardgen_companion_slice_strict_gate_2026-04-22.tsv"
    md_path = report_dir / "hardgen_companion_slice_report_2026-04-22.zh-CN.md"
    write_tsv(paired_tsv, paired)
    write_tsv(strict_tsv, strict)

    lines = [
        "# Hard-Generator Companion Slice Report (2026-04-22)",
        "",
        "## Main Takeaways",
        "",
        "- `hardgenpair16` without protected-real stabilizer has useful hard-fake headroom, but canonical strict FPR08 has no feasible configuration.",
        "- Adding `protpatchg005` to `hardgenpair16` restores strict FPR08 feasibility, but weakens simswap/inswap paired recall too much for a final main row.",
        "- Expanding to `hardgenpair32 + protpatchg005` is the best fake-side movement so far: simswap routing BA reaches `0.804464`, inswap routing BA reaches `0.727754`.",
        "- However, `hardgenpair32 + protpatchg005` fails global strict FPR08 because Celeb-DF-v2 protected-real FPR rises to `0.146484`.",
        "- Increasing the same guard to `0.010` does not improve paired heldout performance, so the next method step should be a more selective protected-real stabilizer, not a global guard-weight sweep.",
        "",
        "## Paired Heldout Summary",
        "",
        "| method | variant | view | BA | fake | real | PR-FPR |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in paired:
        lines.append(
            f"| {row['method']} | {row['variant']} | {row['view']} | {row['balanced_accuracy']} | "
            f"{row['fake_recall']} | {row['real_accuracy']} | {row['protected_real_fpr']} |"
        )
    lines.extend(
        [
            "",
            "## Canonical Strict FPR08 Gate",
            "",
            "| branch | feasible | BA | min dataset BA | max PR-FPR | min PF | min UF |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in strict:
        lines.append(
            f"| {row['strict_gate']} | {row['feasible']} | {row['overall_balanced_accuracy']} | "
            f"{row['min_dataset_balanced_accuracy']} | {row['max_protected_real_fpr']} | "
            f"{row['min_protected_fake_recall']} | {row['min_unprotected_fake_recall']} |"
        )
    lines.extend(
        [
            "",
            "## Next Method Step",
            "",
            "The evidence points away from larger fake-side slices alone and away from global real-side pressure. The next credible move is a selective protected-real stabilizer: keep `hardgenpair32` for hard fake separation, but restrict the stabilizer to the protected-real subfamilies that drive the strict-gate failure, especially Celeb-DF-v2 and DFDCP protected real, without applying broad CE pressure to all protected real.",
            "",
            "## Outputs",
            "",
            f"- `{paired_tsv.relative_to(root)}`",
            f"- `{strict_tsv.relative_to(root)}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(md_path), "paired_tsv": str(paired_tsv), "strict_tsv": str(strict_tsv)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
