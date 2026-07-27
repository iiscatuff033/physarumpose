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
    parser.add_argument("--anchor_summary_csv", type=str, required=True)
    parser.add_argument("--out_txt", type=str, default="outputs/stage5h_instance_refined_soft_slime/stage5h_report_text.txt")
    args = parser.parse_args()

    s = pd.read_csv(args.summary_csv).iloc[0]
    a = pd.read_csv(args.anchor_summary_csv).iloc[0]

    lines = []
    lines.append("Stage 5H: Instance-Aware Anchor Refinement")
    lines.append("")
    lines.append("Anchor refinement result:")
    lines.append(f"- Refined anchor-pair mean error: {fmt(a.get('refined_anchor_pair_mean_err_px'))} px")
    lines.append(f"- Teacher anchor-pair mean error: {fmt(a.get('teacher_anchor_pair_mean_err_px'))} px")
    lines.append(f"- Refined better than teacher rate: {pct(a.get('refined_better_than_teacher_rate'))}")
    lines.append(f"- Refined target-region mean error: {fmt(a.get('refined_region_mean_err_px'))} px")
    lines.append("")
    lines.append("Instance-refined slime result:")
    lines.append(f"- Instance-refined gated mean error: {fmt(s.get('instance_refined_gated_mean_px'))} px")
    lines.append(f"- Previous gated mean error: {fmt(s.get('previous_gated_mean_px'))} px")
    lines.append(f"- Instance-refined Top-1 mean error: {fmt(s.get('instance_refined_top1_mean_px'))} px")
    lines.append(f"- Instance-refined soft mean error: {fmt(s.get('instance_refined_soft_mean_px'))} px")
    lines.append(f"- Instance-refined region mean error: {fmt(s.get('instance_refined_region_mean_px'))} px")
    if "yolo_mean_px" in s.index:
        lines.append(f"- YOLO mean error: {fmt(s.get('yolo_mean_px'))} px")
        lines.append(f"- Win rate vs YOLO: {pct(s.get('refined_win_vs_yolo'))}")
    lines.append(f"- Win rate vs previous gated output: {pct(s.get('refined_win_vs_previous_gated'))}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append(
        "This stage tests whether crop-level instance-aware anchor refinement can reduce wrong-person or long-anchor-path failures. "
        "If it improves suspicious cases or lowers the final gated error, it supports the claim that instance consistency is important for robust slime-based occluded keypoint recovery."
    )

    out_path = Path(args.out_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\\n".join(lines), encoding="utf-8")

    print("\\n".join(lines))
    print("\\nSaved:", out_path)


if __name__ == "__main__":
    main()
