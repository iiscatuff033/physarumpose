import argparse
from pathlib import Path
import pandas as pd

def fmt(v):
    try: return f"{float(v):.2f}"
    except Exception: return str(v)
def pct(v):
    try: return f"{100*float(v):.2f}%"
    except Exception: return str(v)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--summary_csv",required=True); ap.add_argument("--by_target_csv",required=True); ap.add_argument("--out_txt",required=True)
    args=ap.parse_args()
    s=pd.read_csv(args.summary_csv).iloc[0]; by=pd.read_csv(args.by_target_csv)
    lines=["Stage 6C: GT-Anchor Scheduled Unified Differentiable PhysarumPose","",
           "Main result:",
           f"- GT-scheduled soft output mean error: {fmt(s.get('gt_scheduled_soft_mean_px'))} px",
           f"- GT-scheduled soft output median error: {fmt(s.get('gt_scheduled_soft_median_px'))} px",
           f"- GT-scheduled Top-1 path mean error: {fmt(s.get('gt_scheduled_top1_mean_px'))} px",
           f"- GT-scheduled region head mean error: {fmt(s.get('gt_scheduled_region_mean_px'))} px",
           f"- GT-scheduled anchor-pair mean error: {fmt(s.get('gt_scheduled_anchor_pair_mean_px'))} px",
           f"- Top-5 oracle mean error: {fmt(s.get('top5_oracle_mean_px'))} px",""]
    if "yolo_mean_px" in s.index:
        lines += ["Baseline comparison:", f"- YOLO mean error: {fmt(s.get('yolo_mean_px'))} px", f"- Soft win rate vs YOLO: {pct(s.get('soft_win_vs_yolo_rate'))}", ""]
    if "previous_gated_mean_px" in s.index:
        lines += ["Comparison to previous staged best:", f"- Previous gated mean error: {fmt(s.get('previous_gated_mean_px'))} px", f"- Soft win rate vs previous gated: {pct(s.get('soft_win_vs_previous_gated_rate'))}", ""]
    lines.append("Per-target result:")
    for _,r in by.iterrows():
        lines.append(f"- {r['target_name']}: soft {fmt(r.get('gt_scheduled_soft_mean_px'))} px, top-1 {fmt(r.get('gt_scheduled_top1_mean_px'))} px, region {fmt(r.get('gt_scheduled_region_mean_px'))} px, anchor pair {fmt(r.get('gt_scheduled_anchor_pair_mean_px'))} px")
    lines += ["","Interpretation:",
              "Stage 6C uses GT anchors only during training to warm-start the slime path module, then gradually transitions to predicted anchors. Inference remains a single differentiable model using predicted anchors only."]
    out=Path(args.out_txt); out.parent.mkdir(parents=True,exist_ok=True); out.write_text("\n".join(lines),encoding="utf-8")
    print("\n".join(lines)); print("\nSaved:",out)
if __name__=="__main__": main()
