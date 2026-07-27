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
    parser.add_argument("--crop_summary_csv", type=str, required=True)
    parser.add_argument("--crop_csv", type=str, required=True)
    parser.add_argument("--out_txt", type=str, default="outputs/stage5g_target_person_crop_filter/stage5g_report_text.txt")
    args = parser.parse_args()

    summary = pd.read_csv(args.crop_summary_csv).iloc[0]
    df = pd.read_csv(args.crop_csv)

    lines = []
    lines.append("Stage 5G: Target-Person Crop + Wrong-Instance Filtering")
    lines.append("")
    lines.append("Main crop-quality result:")
    lines.append(f"- Total samples: {fmt(summary.get('count'))}")
    lines.append(f"- Suspicious visual/path cases: {pct(summary.get('suspicious_rate'))}")
    lines.append(f"- Good paper/example cases: {pct(summary.get('good_paper_example_rate'))}")
    lines.append(f"- Mean anchor distance: {fmt(summary.get('mean_anchor_dist_px'))} px")
    lines.append(f"- Median anchor distance: {fmt(summary.get('median_anchor_dist_px'))} px")
    lines.append(f"- Mean Top-1 vs region distance: {fmt(summary.get('mean_top1_region_dist_px'))} px")
    lines.append("")

    if "gated_mean_px_all" in summary.index:
        lines.append("Accuracy split:")
        lines.append(f"- Gated mean error, all samples: {fmt(summary.get('gated_mean_px_all'))} px")
        lines.append(f"- Gated mean error, good examples: {fmt(summary.get('gated_mean_px_good_examples'))} px")
        lines.append(f"- Gated mean error, suspicious cases: {fmt(summary.get('gated_mean_px_suspicious'))} px")
        lines.append("")

    # Counts for failure flags
    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    if flag_cols:
        lines.append("Suspicion flag rates:")
        for c in flag_cols:
            lines.append(f"- {c}: {pct(df[c].mean())}")
        lines.append("")

    lines.append("Interpretation:")
    lines.append(
        "This stage does not retrain the model. It crops around the target anchor/path region and flags cases where "
        "the path geometry suggests possible wrong-person or wrong-anchor behavior. These crops should be used for "
        "clean paper/presentation figures and for diagnosing failure cases."
    )

    out_path = Path(args.out_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\\n".join(lines), encoding="utf-8")

    print("\\n".join(lines))
    print("\\nSaved:", out_path)


if __name__ == "__main__":
    main()
