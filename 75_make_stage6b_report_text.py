import argparse
from pathlib import Path
import pandas as pd

def fmt(v):
    try: return f'{float(v):.2f}'
    except Exception: return str(v)
def pct(v):
    try: return f'{100*float(v):.2f}%'
    except Exception: return str(v)
def main():
    p=argparse.ArgumentParser(); p.add_argument('--summary_csv',required=True); p.add_argument('--by_target_csv',required=True); p.add_argument('--out_txt',default='outputs/stage6b_anchor_faithful_physarum_pose/eval/stage6b_report_text.txt'); args=p.parse_args()
    s=pd.read_csv(args.summary_csv).iloc[0]; by=pd.read_csv(args.by_target_csv); lines=[]
    lines.append('Stage 6B: Anchor-Faithful Unified Differentiable PhysarumPose'); lines.append(''); lines.append('Main anchor-faithful model result:')
    lines.append(f"- Anchor-faithful soft output mean error: {fmt(s.get('anchor_faithful_soft_mean_px'))} px"); lines.append(f"- Anchor-faithful soft output median error: {fmt(s.get('anchor_faithful_soft_median_px'))} px"); lines.append(f"- Anchor-faithful Top-1 path mean error: {fmt(s.get('anchor_faithful_top1_mean_px'))} px"); lines.append(f"- Anchor-faithful region head mean error: {fmt(s.get('anchor_faithful_region_mean_px'))} px"); lines.append(f"- Anchor-faithful anchor-pair mean error: {fmt(s.get('anchor_faithful_anchor_pair_mean_px'))} px"); lines.append(f"- Top-5 oracle mean error: {fmt(s.get('top5_oracle_mean_px'))} px"); lines.append('')
    if 'yolo_mean_px' in s.index:
        lines.append('Baseline comparison:'); lines.append(f"- YOLO mean error: {fmt(s.get('yolo_mean_px'))} px"); lines.append(f"- Soft win rate vs YOLO: {pct(s.get('soft_win_vs_yolo_rate'))}"); lines.append(f"- Top-1 win rate vs YOLO: {pct(s.get('top1_win_vs_yolo_rate'))}"); lines.append('')
    if 'previous_gated_mean_px' in s.index:
        lines.append('Comparison to previous staged best:'); lines.append(f"- Previous gated mean error: {fmt(s.get('previous_gated_mean_px'))} px"); lines.append(f"- Soft win rate vs previous gated: {pct(s.get('soft_win_vs_previous_gated_rate'))}"); lines.append(f"- Top-1 win rate vs previous gated: {pct(s.get('top1_win_vs_previous_gated_rate'))}"); lines.append('')
    lines.append('Per-target result:')
    for _,r in by.iterrows(): lines.append(f"- {r['target_name']}: soft {fmt(r.get('anchor_faithful_soft_mean_px'))} px, top-1 {fmt(r.get('anchor_faithful_top1_mean_px'))} px, region {fmt(r.get('anchor_faithful_region_mean_px'))} px, anchor pair {fmt(r.get('anchor_faithful_anchor_pair_mean_px'))} px")
    lines.append(''); lines.append('Interpretation:'); lines.append('Stage 6B removes the direct region-candidate shortcut and forces predictions to come from anchor-to-anchor slime paths. The model remains one differentiable architecture, but now the key publication question is whether anchor-pair error decreases while hidden-joint error stays competitive.')
    out=Path(args.out_txt); out.parent.mkdir(parents=True,exist_ok=True); out.write_text('\n'.join(lines),encoding='utf-8'); print('\n'.join(lines)); print('\nSaved:',out)
if __name__=='__main__': main()
