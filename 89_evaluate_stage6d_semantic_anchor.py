import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.common import *
from src.dataset import SemanticAnchorDataset
from src.model import SemanticAnchorPhysarumPose
from src.eval_utils import crop_norm_to_full,err_norm,topk_rows,summarize,error_bins

def collate(batch):
    out={}
    for k in ['image','target_id','anchor_xy','anchor_mask','region_xy','region_mask','heatmaps','heatmap_mask']: out[k]=torch.stack([b[k] for b in batch],0)
    out['meta']=[b['meta'] for b in batch]; return out
def move(batch,device): return {k:(v.to(device) if k!='meta' else v) for k,v in batch.items()}
def build_pair_indices(target_id,device): return torch.tensor([TARGET_PAIR_INDEX_BY_ID[int(t.item())] for t in target_id],dtype=torch.long,device=device)
def selected_gt_anchor_pair(anchor_xy,pair):
    B=anchor_xy.shape[0]; idx=torch.arange(B,device=anchor_xy.device); return torch.stack([anchor_xy[idx,pair[:,0]],anchor_xy[idx,pair[:,1]]],1)
def selected_target_xy(region_xy,target_id):
    B=region_xy.shape[0]; idx=torch.arange(B,device=region_xy.device); return region_xy[idx,target_id]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--crop_csv',required=True); p.add_argument('--annotation_csv',required=True); p.add_argument('--image_dir',required=True); p.add_argument('--model_dir',required=True); p.add_argument('--out_dir',default='outputs/stage6d_semantic_anchor_physarum/eval'); p.add_argument('--batch_size',type=int,default=16); p.add_argument('--input_size',type=int,default=256); p.add_argument('--heatmap_size',type=int,default=64); p.add_argument('--cpu',action='store_true'); p.add_argument('--max_samples',type=int,default=-1); args=p.parse_args(); out_dir=Path(args.out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    df=merge_crop_and_annotations(args.crop_csv,args.annotation_csv); df=filter_valid_relevant_pair_inside_crop(df,.05)
    if args.max_samples>0: df=df.head(args.max_samples).reset_index(drop=True)
    ds=SemanticAnchorDataset(df,args.image_dir,args.input_size,args.heatmap_size,augment=False); loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=0,collate_fn=collate); device='cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu'); print('Using device:',device,'Eval samples:',len(ds)); model=SemanticAnchorPhysarumPose().to(device); model.load_state_dict(torch.load(Path(args.model_dir)/'best_model.pt',map_location=device)); model.eval(); rows=[]; topks=[]
    with torch.no_grad():
        for batch in tqdm(loader,desc='Evaluating Stage 6D'):
            batch=move(batch,device); pair=build_pair_indices(batch['target_id'],device); out=model(batch['image'],batch['target_id'],pair_indices=pair,gt_anchor_pair=None,gt_anchor_lambda=1.0); soft=out['soft_pred_xy'].detach().cpu().numpy(); axy=out['anchor_xy'].detach().cpu().numpy(); aconf=out['anchor_conf'].detach().cpu().numpy(); rxy=out['region_xy'].detach().cpu().numpy(); rconf=out['region_conf'].detach().cpu().numpy(); tk=topk_rows(out,batch,k=5); topks.extend(tk); by={}
            for r in tk: by.setdefault(r['sample_id'],[]).append(r)
            for b,meta in enumerate(batch['meta']):
                tid=int(batch['target_id'][b].detach().cpu().item()); ai,bi=TARGET_PAIR_INDEX_BY_ID[tid]; se,sx,sy=err_norm(soft[b],meta); ax,ay=crop_norm_to_full(axy[b,ai,0],axy[b,ai,1],meta); bx,byy=crop_norm_to_full(axy[b,bi,0],axy[b,bi,1],meta); rx,ry=crop_norm_to_full(rxy[b,tid,0],rxy[b,tid,1],meta); ae=dist2d(ax,ay,meta['gt_anchor_a_x'],meta['gt_anchor_a_y']); be=dist2d(bx,byy,meta['gt_anchor_b_x'],meta['gt_anchor_b_y']); re=dist2d(rx,ry,meta['target_orig_x'],meta['target_orig_y']); one=sorted(by[meta['sample_id']],key=lambda z:z['rank']); top1=one[0]; row=dict(meta); row.update({'stage6d_soft_x':sx,'stage6d_soft_y':sy,'stage6d_soft_error_px':se,'stage6d_anchor_a_x':ax,'stage6d_anchor_a_y':ay,'stage6d_anchor_a_conf':float(aconf[b,ai]),'stage6d_anchor_a_error_px':ae,'stage6d_anchor_b_x':bx,'stage6d_anchor_b_y':byy,'stage6d_anchor_b_conf':float(aconf[b,bi]),'stage6d_anchor_b_error_px':be,'stage6d_anchor_pair_mean_error_px':np.nanmean([ae,be]),'stage6d_region_x':rx,'stage6d_region_y':ry,'stage6d_region_conf':float(rconf[b,tid]),'stage6d_region_error_px':re,'stage6d_top1_x':top1['candidate_x'],'stage6d_top1_y':top1['candidate_y'],'stage6d_top1_error_px':top1['candidate_error_px'],'stage6d_top1_prob':top1['candidate_prob'],'stage6d_top3_oracle_error_px':min(z['candidate_error_px'] for z in one[:3]),'stage6d_top5_oracle_error_px':min(z['candidate_error_px'] for z in one[:5]),'stage6d_anchor_dist_px':dist2d(top1['anchor1_x'],top1['anchor1_y'],top1['anchor2_x'],top1['anchor2_y']),'stage6d_paper_good':int((top1['candidate_error_px']<=15) and (np.nanmean([ae,be])<=35))}); rows.append(row)
    pred=pd.DataFrame(rows); topk=pd.DataFrame(topks); pred.to_csv(out_dir/'stage6d_predictions.csv',index=False); topk.to_csv(out_dir/'stage6d_topk_predictions.csv',index=False); summary=pd.DataFrame([summarize(pred)]); summary.to_csv(out_dir/'stage6d_summary.csv',index=False); bytar=[]
    for t,g in pred.groupby('target_name'):
        s=summarize(g); s['target_name']=t; bytar.append(s)
    pd.DataFrame(bytar).to_csv(out_dir/'stage6d_summary_by_target.csv',index=False); eb=error_bins(pred)
    if len(eb): eb.to_csv(out_dir/'stage6d_error_bins.csv',index=False)
    print(summary.T)
if __name__=='__main__': main()
