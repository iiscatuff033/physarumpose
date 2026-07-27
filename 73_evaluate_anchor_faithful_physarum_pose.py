import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.stage6b_common import merge_crop_and_annotations, filter_valid_gt_inside_crop, dist2d
from src.anchor_faithful_dataset import AnchorFaithfulDataset
from src.anchor_faithful_model import AnchorFaithfulPhysarumPoseModel
from src.anchor_faithful_eval_utils import crop_norm_to_full, px_error_from_norm, topk_rows_from_output, summarize_predictions, make_error_bins

def collate_fn(batch): return {'image':torch.stack([b['image'] for b in batch]),'target_id':torch.stack([b['target_id'] for b in batch]),'xy':torch.stack([b['xy'] for b in batch]),'heatmaps':torch.stack([b['heatmaps'] for b in batch]),'meta':[b['meta'] for b in batch]}
def move_batch(batch,device): return {k:(v.to(device) if k!='meta' else v) for k,v in batch.items()}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--crop_csv',required=True); p.add_argument('--annotation_csv',required=True); p.add_argument('--image_dir',required=True); p.add_argument('--model_dir',required=True); p.add_argument('--out_dir',default='outputs/stage6b_anchor_faithful_physarum_pose/eval'); p.add_argument('--batch_size',type=int,default=16); p.add_argument('--input_size',type=int,default=256); p.add_argument('--heatmap_size',type=int,default=64); p.add_argument('--cpu',action='store_true'); p.add_argument('--max_samples',type=int,default=-1); args=p.parse_args()
    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True,exist_ok=True); df=filter_valid_gt_inside_crop(merge_crop_and_annotations(args.crop_csv,args.annotation_csv),margin=0.05)
    if args.max_samples and args.max_samples>0: df=df.head(args.max_samples).reset_index(drop=True)
    ds=AnchorFaithfulDataset(df,args.image_dir,args.input_size,args.heatmap_size,augment=False); loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=0,collate_fn=collate_fn)
    device='cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu'); print('Using device:',device); print('Eval samples:',len(ds)); model=AnchorFaithfulPhysarumPoseModel().to(device); model.load_state_dict(torch.load(Path(args.model_dir)/'best_model.pt',map_location=device)); model.eval()
    preds=[]; topks=[]
    with torch.no_grad():
        for batch in tqdm(loader,desc='Evaluating anchor-faithful PhysarumPose'):
            batch=move_batch(batch,device); out=model(batch['image'],batch['target_id']); soft_xy=out['soft_pred_xy'].cpu().numpy(); heat_xy=out['heat_xy'].cpu().numpy(); heat_conf=out['heat_conf'].cpu().numpy(); btop=topk_rows_from_output(out,batch,k=5); topks.extend(btop); by={}
            for r in btop: by.setdefault(r['sample_id'],[]).append(r)
            for b,meta in enumerate(batch['meta']):
                soft_err,soft_x,soft_y=px_error_from_norm(soft_xy[b],meta); ax,ay=crop_norm_to_full(heat_xy[b,0,0],heat_xy[b,0,1],meta); bx,byy=crop_norm_to_full(heat_xy[b,1,0],heat_xy[b,1,1],meta); rx,ry=crop_norm_to_full(heat_xy[b,2,0],heat_xy[b,2,1],meta)
                ae=dist2d(ax,ay,meta['gt_anchor_a_x'],meta['gt_anchor_a_y']); be=dist2d(bx,byy,meta['gt_anchor_b_x'],meta['gt_anchor_b_y']); re=dist2d(rx,ry,meta['target_orig_x'],meta['target_orig_y']); one=sorted(by[meta['sample_id']],key=lambda r:r['rank']); top1=one[0]
                row=dict(meta); row.update({'anchor_faithful_soft_x':soft_x,'anchor_faithful_soft_y':soft_y,'anchor_faithful_soft_error_px':soft_err,'anchor_faithful_anchor_a_x':ax,'anchor_faithful_anchor_a_y':ay,'anchor_faithful_anchor_a_conf':float(heat_conf[b,0]),'anchor_faithful_anchor_a_error_px':ae,'anchor_faithful_anchor_b_x':bx,'anchor_faithful_anchor_b_y':byy,'anchor_faithful_anchor_b_conf':float(heat_conf[b,1]),'anchor_faithful_anchor_b_error_px':be,'anchor_faithful_anchor_pair_mean_error_px':np.nanmean([ae,be]),'anchor_faithful_region_x':rx,'anchor_faithful_region_y':ry,'anchor_faithful_region_conf':float(heat_conf[b,2]),'anchor_faithful_region_error_px':re,'anchor_faithful_top1_x':top1['candidate_x'],'anchor_faithful_top1_y':top1['candidate_y'],'anchor_faithful_top1_error_px':top1['candidate_error_px'],'anchor_faithful_top1_prob':top1['candidate_prob'],'anchor_faithful_top3_oracle_error_px':min(r['candidate_error_px'] for r in one[:3]),'anchor_faithful_top5_oracle_error_px':min(r['candidate_error_px'] for r in one[:5]),'anchor_faithful_anchor_dist_px':dist2d(top1['anchor1_x'],top1['anchor1_y'],top1['anchor2_x'],top1['anchor2_y'])})
                if 'yolo_target_error_px' in row: row['anchor_faithful_soft_better_than_yolo']=row['anchor_faithful_soft_error_px']<row['yolo_target_error_px']; row['anchor_faithful_top1_better_than_yolo']=row['anchor_faithful_top1_error_px']<row['yolo_target_error_px']
                if 'gated_error_px' in row: row['anchor_faithful_soft_better_than_previous_gated']=row['anchor_faithful_soft_error_px']<row['gated_error_px']; row['anchor_faithful_top1_better_than_previous_gated']=row['anchor_faithful_top1_error_px']<row['gated_error_px']
                preds.append(row)
    pred_df=pd.DataFrame(preds); topk_df=pd.DataFrame(topks); pred_df.to_csv(out_dir/'anchor_faithful_predictions.csv',index=False); topk_df.to_csv(out_dir/'anchor_faithful_topk_predictions.csv',index=False); summary=pd.DataFrame([summarize_predictions(pred_df)]); summary.to_csv(out_dir/'anchor_faithful_summary.csv',index=False); by_target=[]
    for t,g in pred_df.groupby('target_name'):
        s=summarize_predictions(g); s['target_name']=t; by_target.append(s)
    pd.DataFrame(by_target).to_csv(out_dir/'anchor_faithful_summary_by_target.csv',index=False); bins=make_error_bins(pred_df)
    if len(bins): bins.to_csv(out_dir/'anchor_faithful_error_bins.csv',index=False)
    print('\nSummary:'); print(summary.T); print('\nSaved to:',out_dir)
if __name__=='__main__': main()
