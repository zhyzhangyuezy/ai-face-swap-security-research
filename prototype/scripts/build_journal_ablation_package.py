from __future__ import annotations

import csv
import json
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


def build_fake_side_rows() -> list[dict[str, Any]]:
    return [
        {
            "family": "fake_side_light_regime",
            "variant": "df40aux8",
            "status": "kept",
            "strict_fpr08": "feasible",
            "all_data_ba": 0.921171,
            "min_dataset_ba": 0.883545,
            "max_protected_real_fpr": 0.079427,
            "lodo_mean_ba": 0.929243,
            "lodo_min_ba": 0.883545,
            "heldout508_all_recall": 0.674213,
            "heldout508_simswap": 0.551181,
            "heldout508_inswap": 0.385827,
            "paired_simswap_fake_ba": 0.421875,
            "paired_inswap_fake_ba": 0.304688,
            "interpretation": "Current kept fake-side balanced branch.",
        },
        {
            "family": "fake_side_light_regime",
            "variant": "df40aux4",
            "status": "rejected",
            "strict_fpr08": "infeasible",
            "all_data_ba": None,
            "min_dataset_ba": None,
            "max_protected_real_fpr": None,
            "lodo_mean_ba": None,
            "lodo_min_ba": None,
            "heldout508_all_recall": 0.663878,
            "heldout508_simswap": 0.531496,
            "heldout508_inswap": 0.374016,
            "paired_simswap_fake_ba": None,
            "paired_inswap_fake_ba": None,
            "interpretation": "Too light: loses held-out fake recall and still fails strict FPR08.",
        },
        {
            "family": "fake_side_light_regime",
            "variant": "df40aux8w05",
            "status": "rejected",
            "strict_fpr08": "rejected_before_scaled",
            "all_data_ba": None,
            "min_dataset_ba": None,
            "max_protected_real_fpr": None,
            "lodo_mean_ba": None,
            "lodo_min_ba": None,
            "heldout508_all_recall": 0.645177,
            "heldout508_simswap": 0.496063,
            "heldout508_inswap": 0.346457,
            "paired_simswap_fake_ba": None,
            "paired_inswap_fake_ba": None,
            "interpretation": "Weight too small: worse than aux8 and even trails aux4.",
        },
        {
            "family": "fake_side_light_regime",
            "variant": "df40aux8w075",
            "status": "rejected",
            "strict_fpr08": "feasible_but_dominated",
            "all_data_ba": 0.911529,
            "min_dataset_ba": 0.867188,
            "max_protected_real_fpr": 0.073568,
            "lodo_mean_ba": None,
            "lodo_min_ba": None,
            "heldout508_all_recall": 0.662894,
            "heldout508_simswap": 0.523622,
            "heldout508_inswap": 0.375984,
            "paired_simswap_fake_ba": None,
            "paired_inswap_fake_ba": None,
            "interpretation": "Admissible, but strictly weaker than plain aux8 on both fake-side and paired/routing axes.",
        },
        {
            "family": "fake_side_reweighting",
            "variant": "df40aux8method",
            "status": "rejected",
            "strict_fpr08": "infeasible",
            "all_data_ba": None,
            "min_dataset_ba": None,
            "max_protected_real_fpr": 0.097005,
            "lodo_mean_ba": None,
            "lodo_min_ba": None,
            "heldout508_all_recall": 0.685039,
            "heldout508_simswap": 0.568898,
            "heldout508_inswap": 0.401575,
            "paired_simswap_fake_ba": None,
            "paired_inswap_fake_ba": None,
            "interpretation": "Improves fake-only recall as intended, but overshoots protected-real FPR.",
        },
        {
            "family": "fake_side_reweighting",
            "variant": "df40aux8hard",
            "status": "rejected",
            "strict_fpr08": "feasible_but_dominated",
            "all_data_ba": None,
            "min_dataset_ba": None,
            "max_protected_real_fpr": None,
            "lodo_mean_ba": None,
            "lodo_min_ba": None,
            "heldout508_all_recall": 0.682087,
            "heldout508_simswap": 0.568898,
            "heldout508_inswap": 0.393701,
            "paired_simswap_fake_ba": None,
            "paired_inswap_fake_ba": None,
            "interpretation": "Better behaved than method-aware weighting, but still weaker than plain aux8 on the kept evidence package.",
        },
        {
            "family": "fake_side_local_token",
            "variant": "df40aux8milaux005",
            "status": "diagnostic_only",
            "strict_fpr08": "infeasible",
            "all_data_ba": 0.918794,
            "min_dataset_ba": 0.883301,
            "max_protected_real_fpr": 0.113932,
            "lodo_mean_ba": None,
            "lodo_min_ba": None,
            "heldout508_all_recall": 0.691437,
            "heldout508_simswap": 0.574803,
            "heldout508_inswap": 0.413386,
            "paired_simswap_fake_ba": None,
            "paired_inswap_fake_ba": None,
            "interpretation": "Positive structural diagnostic: improves fake-only diversity, but fails strict FPR08.",
        },
        {
            "family": "fake_side_local_token",
            "variant": "df40aux8milaux0025",
            "status": "diagnostic_only",
            "strict_fpr08": "infeasible",
            "all_data_ba": 0.918115,
            "min_dataset_ba": 0.877441,
            "max_protected_real_fpr": 0.129557,
            "lodo_mean_ba": None,
            "lodo_min_ba": None,
            "heldout508_all_recall": 0.691929,
            "heldout508_simswap": 0.572835,
            "heldout508_inswap": 0.413386,
            "paired_simswap_fake_ba": None,
            "paired_inswap_fake_ba": None,
            "interpretation": "Shrinking patch-MIL weight preserves fake-only gain, but does not recover the strict bound.",
        },
        {
            "family": "fake_side_local_token",
            "variant": "df40aux8milaux0025_protpatchg001_auxonly",
            "status": "diagnostic_only",
            "strict_fpr08": "feasible_but_not_frontier",
            "all_data_ba": 0.916554,
            "min_dataset_ba": 0.870117,
            "max_protected_real_fpr": 0.059896,
            "lodo_mean_ba": 0.918969,
            "lodo_min_ba": 0.870117,
            "heldout508_all_recall": 0.656496,
            "heldout508_simswap": 0.513780,
            "heldout508_inswap": 0.370079,
            "paired_simswap_fake_ba": 0.429688,
            "paired_inswap_fake_ba": 0.335938,
            "interpretation": "Protected-real guard is causally helpful, but the family still underperforms plain aux8 on the bottleneck evidence.",
        },
        {
            "family": "fake_side_local_token",
            "variant": "df40aux8milaux0025_protauxonly_noguard",
            "status": "diagnostic_only",
            "strict_fpr08": "feasible_but_not_frontier",
            "all_data_ba": 0.913905,
            "min_dataset_ba": 0.868896,
            "max_protected_real_fpr": 0.067708,
            "lodo_mean_ba": 0.916546,
            "lodo_min_ba": 0.868896,
            "heldout508_all_recall": None,
            "heldout508_simswap": None,
            "heldout508_inswap": None,
            "paired_simswap_fake_ba": None,
            "paired_inswap_fake_ba": None,
            "interpretation": "Matched no-guard control shows the guard helps, but neither branch beats the kept aux8 line.",
        },
    ]


def build_rows() -> list[dict[str, Any]]:
    journal_ablation = {row["branch"]: row for row in load_tsv(REPORTS / "journal_ablation_rows_2026-04-20.tsv")}
    paired_rows = load_tsv(REPORTS / "df40_ffpp_paired_transferpair_compare_2026-04-19.tsv")

    rows: list[dict[str, Any]] = []

    # Main-claim and routing rows.
    for branch in [
        "sfrg05_route_floor",
        "df40aux8_route_floor",
        "df40aux8_transferpattern_route_floor",
        "df40aux8_transfertree_fixed78",
        "df40aux8_transfertree_literal_fixed",
    ]:
        row = journal_ablation[branch]
        rows.append(
            {
                "family": "main_and_routing",
                "variant": branch,
                "status": "kept" if branch in {"sfrg05_route_floor", "df40aux8_route_floor", "df40aux8_transfertree_literal_fixed"} else "ablation",
                "strict_fpr08": "feasible",
                "all_data_ba": to_float(row["all_data_ba"]),
                "min_dataset_ba": None,
                "max_protected_real_fpr": None,
                "lodo_mean_ba": to_float(row["lodo_mean_ba"]),
                "lodo_min_ba": to_float(row["lodo_min_ba"]),
                "heldout508_all_recall": None,
                "heldout508_simswap": None,
                "heldout508_inswap": None,
                "paired_simswap_fake_ba": to_float(row["paired_simswap_fake_ba"]),
                "paired_inswap_fake_ba": to_float(row["paired_inswap_fake_ba"]),
                "interpretation": {
                    "sfrg05_route_floor": "Strongest cross-dataset strict main branch.",
                    "df40aux8_route_floor": "Kept fake-side balanced branch.",
                    "df40aux8_transferpattern_route_floor": "Simpler explicit routing fallback.",
                    "df40aux8_transfertree_fixed78": "Learned precursor that exposed the useful leaves.",
                    "df40aux8_transfertree_literal_fixed": "Current paired-oriented explicit frontier.",
                }[branch],
            }
        )

    rows.extend(build_fake_side_rows())

    # Paired ceiling diagnostics.
    for row in paired_rows:
        rows.append(
            {
                "family": "paired_ceiling_diagnostic",
                "variant": f"{row['rule']}::{row['method']}",
                "status": "diagnostic",
                "strict_fpr08": "safe" if row["rule"] != "aux8_sfrg05rules" else "ceiling_only",
                "all_data_ba": None,
                "min_dataset_ba": None,
                "max_protected_real_fpr": to_float(row["protected_real_fpr"]),
                "lodo_mean_ba": None,
                "lodo_min_ba": None,
                "heldout508_all_recall": None,
                "heldout508_simswap": None,
                "heldout508_inswap": None,
                "paired_simswap_fake_ba": to_float(row["fake_accuracy"]) if row["method"] == "simswap" else None,
                "paired_inswap_fake_ba": to_float(row["fake_accuracy"]) if row["method"] == "inswap" else None,
                "interpretation": {
                    "aux8_locked": "Safe paired starting point.",
                    "aux8_transferpair": "Modest safe routing gain.",
                    "aux8_sfrg05rules": "Aggressive upper bound, not the kept safe branch.",
                }[row["rule"]],
            }
        )

    return rows


def build_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Journal Ablation Package",
        "",
        "这份包把当前资产下已经完成的 main-line、fake-side、routing 和 paired-ceiling 结果，重排成 reviewer-facing 的 ablation 证据。重点不是再证明“我们有一个最好结果”，而是证明：为什么保留这些分支，为什么相邻分支没有被保留。",
        "",
        "## Raw Data Table",
        "",
        "- `prototype/reports/journal_ablation_package_rows_2026-04-20.tsv`",
        "",
        "## Snapshot Table",
        "",
        "| Family | Variant | Status | Strict FPR08 | All BA | LODO mean/min | Heldout508 all | Paired simswap/inswap |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        paired = ""
        if row["paired_simswap_fake_ba"] is not None or row["paired_inswap_fake_ba"] is not None:
            paired = f"{fmt(row['paired_simswap_fake_ba'])} / {fmt(row['paired_inswap_fake_ba'])}"
        lines.append(
            f"| {row['family']} | {row['variant']} | {row['status']} | {row['strict_fpr08']} | "
            f"{fmt(row['all_data_ba'])} | {fmt(row['lodo_mean_ba'])} / {fmt(row['lodo_min_ba'])} | "
            f"{fmt(row['heldout508_all_recall'])} | {paired} |"
        )

    lines.extend(
        [
            "",
            "## Key Findings",
            "",
            "1. **Observation:** `sfrg05 + route_floor` still gives the strongest strict cross-dataset main result (`All BA 0.929522`, `LODO mean/min 0.935558 / 0.871582`). **Interpretation:** the real-guarded paired/routing mechanism is the best choice when the reviewer asks for the strongest overall admissible row. **Implication:** this remains the main cross-dataset branch. **Next step:** keep it fixed and stop reopening nearby routing-threshold families.",
            "2. **Observation:** plain `df40aux8` dominates the whole light fake-side regime among admissible branches. `aux4` and `aux8w05` are too light; `aux8w075` is feasible but weaker; method-aware improves fake-only but is FPR-infeasible; hard-aware is feasible but still weaker than plain `aux8`. **Interpretation:** the kept fake-side gain is not coming from arbitrary DF40 injection size tuning. **Implication:** plain `df40aux8 + route_floor` is the right fake-side admissible row. **Next step:** if we revisit fake-side learning, it should be via a stronger guarded structural idea, not more weight shrinking or scalar reweighting.",
            "3. **Observation:** aux-only patch-MIL branches (`0.05`, `0.025`) push held-out DF40 fake-only recall highest (`0.691437` and `0.691929`), but both break strict FPR08. The protected-real guard branch restores feasibility (`All BA 0.916554`, `max PR-FPR 0.059896`) and beats the no-guard control, yet still loses to plain `aux8` on the bottleneck evidence. **Interpretation:** protected-real local-token guarding is causally helpful, but this family is not yet a kept main-method replacement. **Implication:** these runs are mechanism evidence, not final-method evidence. **Next step:** keep them in the ablation appendix, not in the main claimed branch.",
            "4. **Observation:** the routing ladder is now clean and monotonic enough for paper use: `transferpattern` -> `transfertree_fixed78` -> `transfertree_literal_fixed` raises paired mean fake BA from `0.375000` to `0.376627` to `0.376953`, while keeping strict-safe LODO. **Interpretation:** the paired-oriented gain now comes from a compact explicit policy, not from an opaque search artifact. **Implication:** `transfertree_literal_fixed` should be the paired-oriented explicit ablation endpoint in the paper. **Next step:** no more manual threshold sweeps; move to stability and missing-data closure.",
            "5. **Observation:** paired ceiling diagnostics show a real remaining gap between the safe kept branch and the aggressive upper bound: safe `aux8_transferpair` reaches `0.433594 / 0.316406` on `simswap/inswap`, while aggressive `aux8_sfrg05rules` reaches `0.535156 / 0.410156`. **Interpretation:** the paired bottleneck is still real and not fully solved by the current safe explicit routing policy. **Implication:** the paper can honestly claim safe paired gains, but not that the paired problem is closed. **Next step:** prioritize held-out paired `faceswap/fsgan` data and final stability, not yet another routing micro-sweep.",
            "",
            "## Suggested Next Experiments",
            "",
            "1. Move to `P2` and build the rerun/stability package for `sfrg05`, `df40aux8`, and the final claimed family.",
            "2. Keep this ablation package as the current reviewer-facing appendix backbone; do not promote the fake-side diagnostic branches into the claimed method.",
            "3. As soon as data arrives, attach `faceswap/fsgan` held-out paired rows and one deployment-stress benchmark to close the biggest journal-level gaps.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = build_rows()
    tsv_path = REPORTS / "journal_ablation_package_rows_2026-04-20.tsv"
    md_path = REPORTS / "journal_ablation_package_2026-04-20.zh-CN.md"
    write_tsv(tsv_path, rows)
    md_path.write_text(build_markdown(rows), encoding="utf-8")
    print(json.dumps({"tsv": str(tsv_path), "md": str(md_path), "row_count": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
