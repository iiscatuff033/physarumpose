import argparse
from pathlib import Path

import pandas as pd


def fmt(v):
    try:
        return f"{float(v):.2f}"
    except Exception:
        return str(v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_csv", type=str, required=True)
    parser.add_argument("--by_target_csv", type=str, required=True)
    parser.add_argument("--out_txt", type=str, default="outputs/stage5e_multihypothesis_slime/stage5e_report_text.txt")
    args = parser.parse_args()

    summary = pd.read_csv(args.summary_csv)
    by_target = pd.read_csv(args.by_target_csv)

    s = summary.iloc[0]

    lines = []
    lines.append("Stage 5E: Multi-Hypothesis Slime Head")
    lines.append("")
    lines.append("Main result:")
    lines.append(f"- Top-1 mean error: {fmt(s.get('top1_mean_px'))} px")
    lines.append(f"- Top-3 oracle mean error: {fmt(s.get('top3_oracle_mean_px'))} px")
    lines.append(f"- Top-5 oracle mean error: {fmt(s.get('top5_oracle_mean_px'))} px")
    lines.append(f"- Top-3 hit rate within 20 px: {fmt(100 * s.get('top3_hit_20px', 0))}%")
    lines.append(f"- Top-5 hit rate within 20 px: {fmt(100 * s.get('top5_hit_20px', 0))}%")
    lines.append(f"- Top-5 path diversity: {fmt(s.get('top5_diversity_mean_px'))} px")
    lines.append("")

    if "yolo_target_mean_px" in s.index:
        lines.append(f"- YOLO mean error: {fmt(s.get('yolo_target_mean_px'))} px")
    if "coarse_region_mean_px" in s.index:
        lines.append(f"- Coarse region mean error: {fmt(s.get('coarse_region_mean_px'))} px")

    lines.append("")
    lines.append("Interpretation:")
    lines.append(
        "The multi-hypothesis slime head keeps several plausible anatomical paths instead of forcing "
        "only one coordinate under occlusion. Top-K oracle error and hit-rate quantify whether the "
        "correct hidden joint is present among the retained path hypotheses."
    )
    lines.append("")
    lines.append("Per-joint summary:")
    for _, r in by_target.iterrows():
        lines.append(
            f"- {r['target_name']}: Top-1 {fmt(r.get('top1_mean_px'))} px, "
            f"Top-3 oracle {fmt(r.get('top3_oracle_mean_px'))} px, "
            f"Top-5 oracle {fmt(r.get('top5_oracle_mean_px'))} px, "
            f"Top-5 hit@20 {fmt(100 * r.get('top5_hit_20px', 0))}%"
        )

    out_path = Path(args.out_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
