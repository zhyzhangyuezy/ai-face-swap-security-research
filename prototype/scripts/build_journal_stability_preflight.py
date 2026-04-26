from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


PROTOTYPE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PROTOTYPE_ROOT.parent
REPORTS = PROTOTYPE_ROOT / "reports"
RUNS = PROJECT_ROOT / "pilot" / "passive_baseline" / "runs"


def load_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def to_float(value: str | float | int | None) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (float, int)):
        return float(value)
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


def build_rows() -> list[dict[str, Any]]:
    evidence_rows = {row["branch"]: row for row in load_tsv(REPORTS / "current_evidence_table_2026-04-20.tsv")}
    stability_rows = {row["branch"]: row for row in load_tsv(REPORTS / "journal_stability_rows_2026-04-20.tsv")}
    plan_rows = {row["phase"]: row for row in load_tsv(REPORTS / "journal_evidence_execution_plan_2026-04-20.tsv")}

    metrics = {
        "sfrg05_route_floor": load_json(
            RUNS / "effort_clipl_rank4_patchattn_mean_sfrank_realguard_w10_m15_g05_e5_v1" / "metrics.json"
        ),
        "df40aux8_route_floor": load_json(
            RUNS / "effort_clipl_rank4_patchattn_mean_sfrank_df40aux8_w10_m15_e5_v1" / "metrics.json"
        ),
    }

    rows: list[dict[str, Any]] = []
    seed_list = "7,21,84"
    total_runs = "4 total (existing seed42 + 3 reruns)"

    sfrg05_metric = metrics["sfrg05_route_floor"]
    sfrg05_evidence = evidence_rows["sfrg05_route_floor"]
    rows.append(
        {
            "priority": 1,
            "branch": "sfrg05_route_floor",
            "paper_role": sfrg05_evidence["role"],
            "training_run_name": sfrg05_metric["run_name"],
            "needs_new_training": "yes",
            "inherits_passive_from": "",
            "eval_scope": "canonical FPR08 + route-floor LODO",
            "single_run_elapsed_sec": sfrg05_metric.get("elapsed_sec"),
            "current_val_balanced_acc": sfrg05_metric.get("val", {}).get("balanced_acc"),
            "current_test_balanced_acc": sfrg05_metric.get("test", {}).get("balanced_acc"),
            "current_cross_min_balanced_acc": sfrg05_metric.get("cross_summary", {}).get("min_balanced_acc"),
            "current_lodo_mean_ba": to_float(sfrg05_evidence["lodo_mean_ba"]),
            "current_lodo_min_ba": to_float(sfrg05_evidence["lodo_min_ba"]),
            "current_max_protected_real_fpr": to_float(sfrg05_evidence["all_data_max_protected_real_fpr"]),
            "current_paired_mean_fake_ba": to_float(sfrg05_evidence["paired_mean_fake_ba"]),
            "recommended_additional_seeds": seed_list,
            "target_total_runs": total_runs,
            "command_recovery_status": "eval pipeline recovered from effort_sfrg05_pipeline_summary.json; train args recoverable from metrics.json but exact manifest invocation still needs canonical re-materialization",
            "readiness": "preflight_ready",
            "blocker": plan_rows["P2"]["blocker"],
            "next_action": "recover canonical train-manifest list, then launch 3 additional reruns and aggregate mean/std",
        }
    )

    df40aux8_metric = metrics["df40aux8_route_floor"]
    df40aux8_evidence = evidence_rows["df40aux8_route_floor"]
    rows.append(
        {
            "priority": 2,
            "branch": "df40aux8_route_floor",
            "paper_role": df40aux8_evidence["role"],
            "training_run_name": df40aux8_metric["run_name"],
            "needs_new_training": "yes",
            "inherits_passive_from": "",
            "eval_scope": "canonical FPR08 + route-floor LODO + heldout DF40 fake-only + FF-paired simswap/inswap",
            "single_run_elapsed_sec": df40aux8_metric.get("elapsed_sec"),
            "current_val_balanced_acc": df40aux8_metric.get("val", {}).get("balanced_acc"),
            "current_test_balanced_acc": df40aux8_metric.get("test", {}).get("balanced_acc"),
            "current_cross_min_balanced_acc": df40aux8_metric.get("cross_summary", {}).get("min_balanced_acc"),
            "current_lodo_mean_ba": to_float(df40aux8_evidence["lodo_mean_ba"]),
            "current_lodo_min_ba": to_float(df40aux8_evidence["lodo_min_ba"]),
            "current_max_protected_real_fpr": to_float(df40aux8_evidence["all_data_max_protected_real_fpr"]),
            "current_paired_mean_fake_ba": to_float(df40aux8_evidence["paired_mean_fake_ba"]),
            "recommended_additional_seeds": seed_list,
            "target_total_runs": total_runs,
            "command_recovery_status": "eval pipeline recovered from effort_sfrank_df40aux8_2026-04-19_pipeline_summary.json; train args recoverable from metrics.json but exact manifest invocation still needs canonical re-materialization",
            "readiness": "preflight_ready",
            "blocker": plan_rows["P2"]["blocker"],
            "next_action": "make df40aux8 the first rerun family because it is the cheapest kept branch and also feeds the final explicit paired policy",
        }
    )

    literal_evidence = evidence_rows["df40aux8_transfertree_literal_fixed"]
    literal_stability = stability_rows["df40aux8_transfertree_literal_fixed"]
    rows.append(
        {
            "priority": 3,
            "branch": "df40aux8_transfertree_literal_fixed",
            "paper_role": literal_evidence["role"],
            "training_run_name": literal_stability["training_run_name"],
            "needs_new_training": "no",
            "inherits_passive_from": "df40aux8_route_floor",
            "eval_scope": "fixed explicit policy + fixed LODO + FF-paired simswap/inswap",
            "single_run_elapsed_sec": to_float(literal_stability["elapsed_sec"]),
            "current_val_balanced_acc": to_float(literal_stability["val_balanced_acc"]),
            "current_test_balanced_acc": to_float(literal_stability["test_balanced_acc"]),
            "current_cross_min_balanced_acc": to_float(literal_stability["cross_min_balanced_acc"]),
            "current_lodo_mean_ba": to_float(literal_evidence["lodo_mean_ba"]),
            "current_lodo_min_ba": to_float(literal_evidence["lodo_min_ba"]),
            "current_max_protected_real_fpr": to_float(literal_evidence["all_data_max_protected_real_fpr"]),
            "current_paired_mean_fake_ba": to_float(literal_evidence["paired_mean_fake_ba"]),
            "recommended_additional_seeds": "inherit df40aux8 rerun seeds",
            "target_total_runs": total_runs,
            "command_recovery_status": "no separate training command needed; rerun package depends on replaying the fixed explicit policy over each df40aux8 rerun checkpoint/output set",
            "readiness": "depends_on_df40aux8_reruns",
            "blocker": "inherits df40aux8 rerun outputs",
            "next_action": "after each df40aux8 rerun, replay literal-fixed routing and aggregate canonical/LODO/paired mean-std",
        }
    )
    return rows


def build_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Journal Stability Preflight",
        "",
        "这份预案把 `P2` 从“下一步应该补稳定性”推进成“现在就知道该补哪几条、先跑哪条、为什么能跑”。它不假装 multi-seed 已经做完，而是把当前证据、rerun 价值和真正剩下的阻塞拆清楚。",
        "",
        "## Raw Data Table",
        "",
        "| Priority | Branch | Need retrain | Single-run sec | Current LODO mean/min | Current paired mean fake BA | Readiness |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['priority']} | {row['branch']} | {row['needs_new_training']} | "
            f"{fmt(row['single_run_elapsed_sec'])} | {fmt(row['current_lodo_mean_ba'])} / {fmt(row['current_lodo_min_ba'])} | "
            f"{fmt(row['current_paired_mean_fake_ba'])} | {row['readiness']} |"
        )

    lines.extend(
        [
            "",
            "## Key Findings",
            "",
            "1. **Observation**: 当前 journal-facing stability 仍然是 single-kept-run 证据，`journal_stability_rows_2026-04-20.tsv` 里的 kept branches 全部还是 `rerun_variance_available = no`。  ",
            "**Interpretation**: 稳定性缺口还是真缺口，不是文档组织问题。  ",
            "**Implication**: `P2` 仍然是当前资产下最值得优先推进的 ready-now 任务。  ",
            "**Next step**: 先把 `df40aux8` 和 `sfrg05` 的 rerun family 补出来，再把 literal fixed policy 挂接到 `df40aux8` rerun 上。",
            "",
            "2. **Observation**: kept passive branches 的训练成本其实不高，`df40aux8` 单次约 `230.94s`，`sfrg05` 单次约 `554.60s`。  ",
            "**Interpretation**: 这已经是分钟级 rerun，而不是需要重新规划半天算力的大作业。  ",
            "**Implication**: 从实验资源角度看，multi-seed variance 是现实可补的。  ",
            "**Next step**: 按 `df40aux8 -> literal_fixed replay -> sfrg05` 的顺序推进，优先拿到最接近最终 claimed family 的稳定性证据。",
            "",
            "3. **Observation**: 最终 paired-oriented explicit branch `df40aux8 + transfertree_literal_fixed` 不需要单独训练，它继承 `df40aux8` passive weights。  ",
            "**Interpretation**: `P2` 的真正训练 rerun 对象只有两个 passive families，而不是三个。  ",
            "**Implication**: final claimed branch 的稳定性可以通过 `df40aux8` rerun + fixed explicit policy replay 来补，证据路径更短。  ",
            "**Next step**: 把 literal branch 作为 `df40aux8` rerun 的下游固定评估，而不是额外开一条训练线。",
            "",
            "## Suggested Next Experiments",
            "",
            "1. 先恢复 `df40aux8` 与 `sfrg05` 的 canonical train-manifest 调用，补齐训练命令闭环。",
            "2. 先起 `df40aux8` 的 additional seeds：`7, 21, 84`，汇总 canonical / LODO / heldout paired 的 mean/std。",
            "3. 对每个 `df40aux8` rerun 直接复用 fixed explicit literal policy，补齐最终 claimed paired branch 的稳定性统计。",
            "4. 再起 `sfrg05` 的 additional seeds：`7, 21, 84`，补 strongest cross-dataset main line 的 variance。",
            "",
            "## Current Verdict",
            "",
            "- `P2` 现在应视为 **in progress**，而不是只停留在 ready。",
            "- 当前真正的阻塞已经缩小成两类：`train-manifest` 调用闭环，以及实际启动 rerun 的时间窗口。",
            "- 一旦 `df40aux8` rerun 起跑，`transfertree_literal_fixed` 的稳定性证据也会同步开始闭环。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = build_rows()
    tsv_path = REPORTS / "journal_stability_preflight_2026-04-20.tsv"
    md_path = REPORTS / "journal_stability_preflight_2026-04-20.zh-CN.md"
    write_tsv(tsv_path, rows)
    md_path.write_text(build_markdown(rows), encoding="utf-8")
    print(str(tsv_path))
    print(str(md_path))


if __name__ == "__main__":
    main()
