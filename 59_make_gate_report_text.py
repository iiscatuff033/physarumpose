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
    parser.add_argument("--out_txt", type=str, default="outputs/stage5fc_anchor_quality_gate/stage5fc_report_text.txt")
    args = parser.parse_args()

    summary = pd.read_csv(args.summary_csv)
    by_target = pd.read_csv(args.by_target_csv)

    # Prefer all split
    if "split" in summary.columns and (summary["split"] == "all").any():
        s = summary[summary["split"] == "all"].iloc[0]
    else:
        s = summary.iloc[0]

    lines = []
    lines.append("Stage 5F-C: Anchor-Quality-Gated Inference")
    lines.append("")
    lines.append("Main result:")
    lines.append(f"- Gated final mean error: {fmt(s.get('gated_mean_px'))} px")
    lines.append(f"- Gated final median error: {fmt(s.get('gated_median_px'))} px")
    lines.append(f"- Top-1 slime path mean error: {fmt(s.get('top1_mean_px'))} px")
    lines.append(f"- Region head mean error: {fmt(s.get('region_mean_px'))} px")
    if "soft_mean_px" in s.index:
        lines.append(f"- Soft weighted output mean error: {fmt(s.get('soft_mean_px'))} px")
    if "yolo_mean_px" in s.index:
        lines.append(f"- YOLO mean error: {fmt(s.get('yolo_mean_px'))} px")
    if "top5_oracle_mean_px" in s.index:
        lines.append(f"- Top-5 oracle mean error: {fmt(s.get('top5_oracle_mean_px'))} px")
    lines.append(f"- Gate used Top-1 slime path on: {pct(s.get('gate_use_top1_rate'))} of samples")
    if "gated_win_vs_yolo" in s.index:
        lines.append(f"- Gated win rate vs YOLO: {pct(s.get('gated_win_vs_yolo'))}")
    lines.append("")

    lines.append("Interpretation:")
    lines.append(
        "The gate decides whether to trust the anchor-based slime path or fall back to the target-region head. "
        "This reduces the impact of visibly wrong anchor paths in multi-person or uncertain scenes."
    )

    lines.append("")
    lines.append("Per-joint summary:")
    for _, r in by_target.iterrows():
        lines.append(
            f"- {r['target_name']}: gated {fmt(r.get('gated_mean_px'))} px, "
            f"top-1 {fmt(r.get('top1_mean_px'))} px, "
            f"region {fmt(r.get('region_mean_px'))} px"
        )

    out_path = Path(args.out_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
