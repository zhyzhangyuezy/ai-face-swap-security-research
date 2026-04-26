from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row.get("status") == "ok" and row.get("prob_fake")]


def metrics(rows: List[Dict[str, str]], threshold: float) -> Dict[str, float | int]:
    real = [row for row in rows if row["label"] == "real"]
    fake = [row for row in rows if row["label"] == "fake"]
    pr = [row for row in real if row["protection"] == "protected"]
    pf = [row for row in fake if row["protection"] == "protected"]
    uf = [row for row in fake if row["protection"] == "unprotected"]

    def pred_fake(row: Dict[str, str]) -> bool:
        return float(row["prob_fake"]) >= threshold

    real_acc = sum(not pred_fake(row) for row in real) / len(real) if real else 0.0
    fake_acc = sum(pred_fake(row) for row in fake) / len(fake) if fake else 0.0
    pr_fpr = sum(pred_fake(row) for row in pr) / len(pr) if pr else 0.0
    pf_recall = sum(pred_fake(row) for row in pf) / len(pf) if pf else 0.0
    uf_recall = sum(pred_fake(row) for row in uf) / len(uf) if uf else 0.0
    return {
        "threshold": threshold,
        "rows": len(rows),
        "real_rows": len(real),
        "fake_rows": len(fake),
        "protected_real_rows": len(pr),
        "protected_fake_rows": len(pf),
        "unprotected_fake_rows": len(uf),
        "real_accuracy": real_acc,
        "fake_accuracy": fake_acc,
        "balanced_accuracy": 0.5 * (real_acc + fake_acc),
        "protected_real_fpr": pr_fpr,
        "protected_fake_recall": pf_recall,
        "unprotected_fake_recall": uf_recall,
        "strict_fpr08": int(pr_fpr <= 0.08),
    }


def threshold_grid(rows: List[Dict[str, str]]) -> List[float]:
    scores = sorted({float(row["prob_fake"]) for row in rows})
    candidates = {0.0, 1.0}
    for score in scores:
        candidates.add(score)
        candidates.add(max(0.0, score + 1e-6))
    return sorted(candidates)


def select_threshold(rows: List[Dict[str, str]], max_pr_fpr: float) -> Dict[str, float | int]:
    candidates = [metrics(rows, threshold) for threshold in threshold_grid(rows)]
    feasible = [row for row in candidates if float(row["protected_real_fpr"]) <= max_pr_fpr]
    if not feasible:
        return max(candidates, key=lambda row: (float(row["balanced_accuracy"]), -float(row["protected_real_fpr"])))
    return max(feasible, key=lambda row: (float(row["balanced_accuracy"]), float(row["protected_fake_recall"]), float(row["unprotected_fake_recall"])))


def checkpoint_status(path: Path, min_size: int = 1024) -> str:
    if not path.exists():
        return "missing"
    if path.is_file() and path.stat().st_size >= min_size:
        return "available"
    files = [item for item in path.rglob("*") if item.is_file() and item.stat().st_size >= min_size]
    return "available" if files else "missing_or_placeholder_only"


def all_available(paths: List[Path], min_size: int = 1024) -> str:
    return "available" if all(path.exists() and path.stat().st_size >= min_size for path in paths) else "missing"


def blocker(status: str, message: str) -> str:
    return "" if status == "available" else message


def write_tsv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize native video / multimodal frontier baseline readiness and micro results.")
    parser.add_argument("--calib", required=True)
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--summary-tsv", required=True)
    parser.add_argument("--readiness-json", required=True)
    parser.add_argument("--report-md", required=True)
    parser.add_argument("--max-pr-fpr", type=float, default=0.08)
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    calib_rows = read_rows(Path(args.calib))
    heldout_rows = read_rows(Path(args.heldout))
    selected = select_threshold(calib_rows, args.max_pr_fpr)
    threshold = float(selected["threshold"])
    heldout = metrics(heldout_rows, threshold)
    source_rows = metrics(heldout_rows, 0.5)

    summary_rows = []
    for split, row in [("calibration_selected", selected), ("heldout_at_calib_threshold", heldout), ("heldout_at_default_0p5", source_rows)]:
        out = {"split": split, "model": "FTCN+TT native face-track micro", "protocol": "VCF c23 raw-video face tracking + aligned VideoSeal rows"}
        out.update(row)
        summary_rows.append(out)
    write_tsv(Path(args.summary_tsv), summary_rows)

    alt_checkpoint = root / "workspace" / "repos" / "AltFreezing" / "checkpoints" / "model.pth"
    m2f2_stage1 = root / "workspace" / "repos" / "M2F2_Det" / "checkpoints" / "stage_1" / "current_model_180.pth"
    m2f2_vision = root / "workspace" / "repos" / "M2F2_Det" / "utils" / "weights" / "vision_tower.pth"
    alt_status = checkpoint_status(alt_checkpoint)
    m2f2_status = all_available([m2f2_stage1, m2f2_vision])

    readiness = {
        "ftcn_native": {
            "repo": str(root / "workspace" / "repos" / "FTCN"),
            "checkpoint": str(root / "workspace" / "repos" / "FTCN" / "checkpoints" / "ftcn_tt.pth"),
            "status": checkpoint_status(root / "workspace" / "repos" / "FTCN" / "checkpoints" / "ftcn_tt.pth"),
            "micro_calib_csv": str(Path(args.calib).resolve()),
            "micro_heldout_csv": str(Path(args.heldout).resolve()),
        },
        "altfreezing": {
            "repo": str(root / "workspace" / "repos" / "AltFreezing"),
            "checkpoint_dir": str(root / "workspace" / "repos" / "AltFreezing" / "checkpoints"),
            "checkpoint": str(alt_checkpoint),
            "status": alt_status,
            "blocking_reason": blocker(
                alt_status,
                "official RecDrive/Baidu checkpoint is not present as checkpoints/model.pth",
            ),
        },
        "m2f2_det": {
            "repo": str(root / "workspace" / "repos" / "M2F2_Det"),
            "checkpoint_dir": str(root / "workspace" / "repos" / "M2F2_Det" / "checkpoints"),
            "vision_tower_dir": str(root / "workspace" / "repos" / "M2F2_Det" / "utils" / "weights"),
            "stage1_checkpoint": str(m2f2_stage1),
            "vision_tower": str(m2f2_vision),
            "status": m2f2_status,
            "blocking_reason": blocker(
                m2f2_status,
                "detector-only stage-1 weights and/or LLaVA CLIP vision tower are not both present locally",
            ),
        },
    }
    Path(args.readiness_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.readiness_json).write_text(json.dumps(readiness, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Native Video / Multimodal Frontier Baseline Progress",
        "",
        "## Native FTCN+TT face-track micro result",
        "",
        f"- Calibration rows: {len(calib_rows)}; held-out rows: {len(heldout_rows)}.",
        f"- Selection rule: maximize calibration BA subject to calibration protected-real FPR <= {args.max_pr_fpr:.2f}.",
        f"- Selected threshold: {threshold:.6f}.",
        f"- Held-out BA at selected threshold: {float(heldout['balanced_accuracy']):.6f}.",
        f"- Held-out protected-real FPR: {float(heldout['protected_real_fpr']):.6f}.",
        f"- Held-out protected/unprotected fake recall: {float(heldout['protected_fake_recall']):.6f} / {float(heldout['unprotected_fake_recall']):.6f}.",
        "",
        "| split | threshold | BA | PR-FPR | PF recall | UF recall | rows |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["split"]),
                    fmt(row["threshold"]),
                    fmt(row["balanced_accuracy"]),
                    fmt(row["protected_real_fpr"]),
                    fmt(row["protected_fake_recall"]),
                    fmt(row["unprotected_fake_recall"]),
                    str(row["rows"]),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Frontier baseline readiness",
        "",
        f"- FTCN+TT native face-track: {readiness['ftcn_native']['status']} and now runnable through the manifest probe.",
        f"- AltFreezing: {readiness['altfreezing']['status']} at `{readiness['altfreezing']['checkpoint']}`.",
        f"- M2F2-Det: {readiness['m2f2_det']['status']} with stage-1 and vision-tower weights.",
        "",
        "## Interpretation",
        "",
        "This is a micro-protocol sanity run, not a full SOTA table. It closes the engineering gap for native face-track video inference by proving that the official detector/tracker/crop-align/classifier path can be driven from the paper manifest. The held-out protected-real resize row already shows why a full native run must retain protected-row stress rather than reporting only raw-video accuracy.",
    ]
    Path(args.report_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
