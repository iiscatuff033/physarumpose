import argparse
from pathlib import Path

import pandas as pd


def fmt(v):
    try:
        return f"{float(v):.2f}"
    except Exception:
        return str(v)


def pct(v):
    try:
        return f"{100 * float(v):.2f}%"
    except Exception:
        return str(v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_csv", type=str, required=True)
    parser.add_argument("--by_target_csv", type=str, required=True)
    parser.add_argument("--out_txt", type=str, default="outputs/stage6a_unified_physarum_pose/eval/stage6a_report_text.txt")
    args = parser.parse_args()

    s = pd.read_csv(args.summary_csv).iloc[0]
    by_target = pd.read_csv(args.by_target_csv)

    lines = []
    lines.append("Stage 6A: Unified Differentiable PhysarumPose")
    lines.append("")
    lines.append("Main unified model result:")
    lines.append(f"- Unified soft output mean error: {fmt(s.get('unified_soft_mean_px'))} px")
    lines.append(f"- Unified soft output median error: {fmt(s.get('unified_soft_median_px'))} px")
    lines.append(f"- Unified Top-1 path mean error: {fmt(s.get('unified_top1_mean_px'))} px")
    lines.append(f"- Unified region head mean error: {fmt(s.get('unified_region_mean_px'))} px")
    lines.append(f"- Unified anchor-pair mean error: {fmt(s.get('unified_anchor_pair_mean_px'))} px")
    lines.append(f"- Top-5 oracle mean error: {fmt(s.get('top5_oracle_mean_px'))} px")
    lines.append("")

    if "yolo_mean_px" in s.index:
        lines.append("Baseline comparison:")
        lines.append(f"- YOLO mean error: {fmt(s.get('yolo_mean_px'))} px")
        lines.append(f"- Soft win rate vs YOLO: {pct(s.get('soft_win_vs_yolo_rate'))}")
        lines.append(f"- Top-1 win rate vs YOLO: {pct(s.get('top1_win_vs_yolo_rate'))}")
        lines.append("")

    if "previous_gated_mean_px" in s.index:
        lines.append("Comparison to previous staged best:")
        lines.append(f"- Previous gated mean error: {fmt(s.get('previous_gated_mean_px'))} px")
        lines.append(f"- Soft win rate vs previous gated: {pct(s.get('soft_win_vs_previous_gated_rate'))}")
        lines.append(f"- Top-1 win rate vs previous gated: {pct(s.get('top1_win_vs_previous_gated_rate'))}")
        lines.append("")

    lines.append("Per-target result:")
    for _, r in by_target.iterrows():
        lines.append(
            f"- {r['target_name']}: soft {fmt(r.get('unified_soft_mean_px'))} px, "
            f"top-1 {fmt(r.get('unified_top1_mean_px'))} px, "
            f"region {fmt(r.get('unified_region_mean_px'))} px, "
            f"anchor pair {fmt(r.get('unified_anchor_pair_mean_px'))} px"
        )

    lines.append("")
    lines.append("Interpretation:")
    lines.append(
        "Stage 6A is the first clean one-model version: a shared backbone predicts target-conditioned anchors and a target-region heatmap, "
        "then a differentiable soft-slime layer generates and scores candidate paths inside the same PyTorch graph. "
        "This is the direction needed for a top-tier paper, even if the first unified result is slightly worse than the staged pipeline."
    )

    out_path = Path(args.out_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\\n".join(lines), encoding="utf-8")

    print("\\n".join(lines))
    print("\\nSaved:", out_path)


if __name__ == "__main__":
    main()
