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
    p=argparse.ArgumentParser()
    p.add_argument("--summary_csv",required=True); p.add_argument("--by_target_csv",required=True)
    p.add_argument("--out_txt",default="outputs/stage6c_r1_cvpr_anchor_curriculum/eval/stage6c_r1_report_text.txt")
    args=p.parse_args()
    s=pd.read_csv(args.summary_csv).iloc[0]; by=pd.read_csv(args.by_target_csv)
    lines=[
        "Stage 6C-R1: CVPR Anchor-Curriculum Differentiable PhysarumPose",
        "",
        "Main result:",
        f"- Top-1 slime path mean error: {fmt(s.get('r1_top1_mean_px'))} px",
        f"- Soft output mean error: {fmt(s.get('r1_soft_mean_px'))} px",
        f"- Region head mean error: {fmt(s.get('r1_region_mean_px'))} px",
        f"- Anchor-pair mean error: {fmt(s.get('r1_anchor_pair_mean_px'))} px",
        f"- Top-5 oracle mean error: {fmt(s.get('top5_oracle_mean_px'))} px",
        "",
    ]
    if "yolo_mean_px" in s.index:
        lines += [f"- YOLO mean error: {fmt(s.get('yolo_mean_px'))} px", f"- Top-1 win rate vs YOLO: {pct(s.get('top1_win_vs_yolo_rate'))}", ""]
    if "previous_gated_mean_px" in s.index:
        lines += [f"- Previous staged gated mean error: {fmt(s.get('previous_gated_mean_px'))} px", f"- Top-1 win rate vs previous staged: {pct(s.get('top1_win_vs_previous_gated_rate'))}", ""]
    lines += ["Per-target result:"]
    for _,r in by.iterrows():
        lines.append(f"- {r['target_name']}: top1 {fmt(r.get('r1_top1_mean_px'))} px, soft {fmt(r.get('r1_soft_mean_px'))} px, anchor pair {fmt(r.get('r1_anchor_pair_mean_px'))} px")
    lines += ["", "Interpretation:", "Stage 6C-R1 saves checkpoints only after predicted-anchor-only training begins and treats Top-1 slime path as the main prediction. This is designed to make the anatomical anchor path, not the region shortcut, carry the novelty."]
    out=Path(args.out_txt); out.parent.mkdir(parents=True,exist_ok=True); out.write_text("\\n".join(lines),encoding="utf-8")
    print("\\n".join(lines)); print("\\nSaved:",out)
if __name__=="__main__": main()
