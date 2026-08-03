import argparse, pandas as pd
from pathlib import Path
def fmt(v):
    try: return f'{float(v):.2f}'
    except Exception: return str(v)
def pct(v):
    try: return f'{100*float(v):.2f}%'
    except Exception: return str(v)
def main():
    p=argparse.ArgumentParser(); p.add_argument('--summary_csv',required=True); p.add_argument('--by_target_csv',required=True); p.add_argument('--out_txt',default='outputs/stage6d_semantic_anchor_physarum/eval/stage6d_report_text.txt'); args=p.parse_args(); s=pd.read_csv(args.summary_csv).iloc[0]; by=pd.read_csv(args.by_target_csv)
    lines=['Stage 6D: Fixed Semantic Anchor Head + Differentiable PhysarumPose','',f"- Top-1 slime path mean error: {fmt(s.get('stage6d_top1_mean_px'))} px",f"- Soft output mean error: {fmt(s.get('stage6d_soft_mean_px'))} px",f"- Region head mean error: {fmt(s.get('stage6d_region_mean_px'))} px",f"- Anchor-pair mean error: {fmt(s.get('stage6d_anchor_pair_mean_px'))} px",f"- Anchor-pair median error: {fmt(s.get('stage6d_anchor_pair_median_px'))} px",f"- Top-5 oracle mean error: {fmt(s.get('top5_oracle_mean_px'))} px",'', 'Anchor-quality rates:',f"- Anchor good @20 px: {pct(s.get('anchor_good_20_rate'))}",f"- Anchor good @30 px: {pct(s.get('anchor_good_30_rate'))}",f"- Anchor good @35 px: {pct(s.get('anchor_good_35_rate'))}",f"- Anchor good @40 px: {pct(s.get('anchor_good_40_rate'))}",f"- Paper-good rate, Top-1 <=15 px and anchor <=35 px: {pct(s.get('paper_good_rate'))}",'']
    if 'yolo_mean_px' in s.index: lines += [f"- YOLO mean error: {fmt(s.get('yolo_mean_px'))} px",f"- Top-1 win rate vs YOLO: {pct(s.get('top1_win_vs_yolo_rate'))}",'']
    if 'previous_gated_mean_px' in s.index: lines += [f"- Previous staged gated mean error: {fmt(s.get('previous_gated_mean_px'))} px",f"- Top-1 win rate vs previous staged: {pct(s.get('top1_win_vs_previous_gated_rate'))}",'']
    lines.append('Per-target result:')
    for _,r in by.iterrows(): lines.append(f"- {r['target_name']}: top1 {fmt(r.get('stage6d_top1_mean_px'))} px, soft {fmt(r.get('stage6d_soft_mean_px'))} px, anchor pair {fmt(r.get('stage6d_anchor_pair_mean_px'))} px, anchor@35 {pct(r.get('anchor_good_35_rate'))}")
    lines += ['', 'Interpretation:', 'Stage 6D fixes generic anchor A/B by using fixed semantic anchor channels. The key check is whether semantic anchors reduce anchor-pair error while retaining Top-1 slime-path accuracy.']
    out=Path(args.out_txt); out.parent.mkdir(parents=True,exist_ok=True); out.write_text('\n'.join(lines),encoding='utf-8'); print('\n'.join(lines)); print('\nSaved:',out)
if __name__=='__main__': main()
