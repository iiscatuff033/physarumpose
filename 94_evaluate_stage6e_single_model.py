import argparse, numpy as np, pandas as pd, torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.common import *
from src.dataset import SemanticOffsetDataset
from src.model import Stage6ESingleModel
from src.eval_utils import crop_norm_to_full,err_norm,topk_rows,summarize,error_bins

def collate(batch):
    out={}
    for k in ['image','target_id','anchor_xy','anchor_mask','region_xy','region_mask','heatmaps','heatmap_mask']: out[k]=torch.stack([b[k] for b in batch],0)
    out['meta']=[b['meta'] for b in batch]; return out
def move(batch,device): return {k:(v.to(device) if k!='meta' else v) for k,v in batch.items()}
def pair_idx(tid,device): return torch.tensor([TARGET_PAIR_INDEX_BY_ID[int(t.item())] for t in tid],dtype=torch.long,device=device)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--crop_csv',required=True); p.add_argument('--annotation_csv',required=True); p.add_argument('--image_dir',required=True); p.add_argument('--model_dir',required=True); p.add_argument('--out_dir',default='outputs/stage6e_single_model_semantic_offset/eval'); p.add_argument('--batch_size',type=int,default=8); p.add_argument('--input_size',type=int,default=384); p.add_argument('--heatmap_size',type=int,default=0); p.add_argument('--cpu',action='store_true'); p.add_argument('--max_samples',type=int,default=-1); args=p.parse_args()
    if args.heatmap_size<=0: args.heatmap_size=args.input_size//4
    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True,exist_ok=True); df=filter_valid_relevant_pair_inside_crop(merge_crop_and_annotations(args.crop_csv,args.annotation_csv),.05)
    if args.max_samples>0: df=df.head(args.max_samples).reset_index(drop=True)
    ds=SemanticOffsetDataset(df,args.image_dir,args.input_size,args.heatmap_size,augment=False); loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=0,collate_fn=collate); device='cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device',device,'Eval samples',len(ds)); model=Stage6ESingleModel().to(device); model.load_state_dict(torch.load(Path(args.model_dir)/'best_model.pt',map_location=device)); model.eval(); rows=[]; topks=[]
    with torch.no_grad():
        for batch in tqdm(loader,desc='Evaluating Stage 6E'):
            batch=move(batch,device); pair=pair_idx(batch['target_id'],device); out=model(batch['image'],batch['target_id'],pair,None,1.0); soft=out['soft_pred_xy'].detach().cpu().numpy(); axys=out['anchor_xy'].detach().cpu().numpy(); raw=out['anchor_xy_raw'].detach().cpu().numpy(); conf=out['anchor_conf'].detach().cpu().numpy(); rxy=out['region_xy'].detach().cpu().numpy(); rconf=out['region_conf'].detach().cpu().numpy(); tk=topk_rows(out,batch,5); topks.extend(tk); by={}
            for r in tk: by.setdefault(r['sample_id'],[]).append(r)
            for b,meta in enumerate(batch['meta']):
                tid=int(batch['target_id'][b].cpu().item()); ai,bi=TARGET_PAIR_INDEX_BY_ID[tid]; se,sx,sy=err_norm(soft[b],meta); ax,ay=crop_norm_to_full(axys[b,ai,0],axys[b,ai,1],meta); bx,byy=crop_norm_to_full(axys[b,bi,0],axys[b,bi,1],meta); arx,ary=crop_norm_to_full(raw[b,ai,0],raw[b,ai,1],meta); brx,bry=crop_norm_to_full(raw[b,bi,0],raw[b,bi,1],meta); rx,ry=crop_norm_to_full(rxy[b,tid,0],rxy[b,tid,1],meta)
                ae=dist2d(ax,ay,meta['gt_anchor_a_x'],meta['gt_anchor_a_y']); be=dist2d(bx,byy,meta['gt_anchor_b_x'],meta['gt_anchor_b_y']); rawae=dist2d(arx,ary,meta['gt_anchor_a_x'],meta['gt_anchor_a_y']); rawbe=dist2d(brx,bry,meta['gt_anchor_b_x'],meta['gt_anchor_b_y']); re=dist2d(rx,ry,meta['target_orig_x'],meta['target_orig_y']); one=sorted(by[meta['sample_id']],key=lambda z:z['rank']); top1=one[0]
                row=dict(meta); row.update({'stage6e_soft_x':sx,'stage6e_soft_y':sy,'stage6e_soft_error_px':se,'stage6e_anchor_a_x':ax,'stage6e_anchor_a_y':ay,'stage6e_anchor_a_conf':float(conf[b,ai]),'stage6e_anchor_a_error_px':ae,'stage6e_anchor_b_x':bx,'stage6e_anchor_b_y':byy,'stage6e_anchor_b_conf':float(conf[b,bi]),'stage6e_anchor_b_error_px':be,'stage6e_raw_anchor_pair_mean_error_px':np.nanmean([rawae,rawbe]),'stage6e_anchor_pair_mean_error_px':np.nanmean([ae,be]),'stage6e_region_x':rx,'stage6e_region_y':ry,'stage6e_region_conf':float(rconf[b,tid]),'stage6e_region_error_px':re,'stage6e_top1_x':top1['candidate_x'],'stage6e_top1_y':top1['candidate_y'],'stage6e_top1_error_px':top1['candidate_error_px'],'stage6e_top1_prob':top1['candidate_prob'],'stage6e_top3_oracle_error_px':min(z['candidate_error_px'] for z in one[:3]),'stage6e_top5_oracle_error_px':min(z['candidate_error_px'] for z in one[:5]),'stage6e_paper_good':int(top1['candidate_error_px']<=15 and np.nanmean([ae,be])<=35)})
                rows.append(row)
    pred=pd.DataFrame(rows); topk=pd.DataFrame(topks); pred.to_csv(out_dir/'stage6e_predictions.csv',index=False); topk.to_csv(out_dir/'stage6e_topk_predictions.csv',index=False); pd.DataFrame([summarize(pred)]).to_csv(out_dir/'stage6e_summary.csv',index=False); bytar=[]
    for t,g in pred.groupby('target_name'):
        s=summarize(g); s['target_name']=t; bytar.append(s)
    pd.DataFrame(bytar).to_csv(out_dir/'stage6e_summary_by_target.csv',index=False); eb=error_bins(pred)
    if len(eb): eb.to_csv(out_dir/'stage6e_error_bins.csv',index=False)
    print(pd.DataFrame([summarize(pred)]).T)
if __name__=='__main__': main()
