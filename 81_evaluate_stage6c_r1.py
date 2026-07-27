import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.common import merge_crop_and_annotations, filter_valid_gt_inside_crop, dist2d
from src.dataset import PoseCropDataset
from src.model import PhysarumPoseR1
from src.eval_utils import crop_norm_to_full, err_norm, topk_rows, summarize, error_bins

def collate(batch):
    return {k: torch.stack([b[k] for b in batch],0) for k in ["image","target_id","xy","heatmaps"]} | {"meta":[b["meta"] for b in batch]}
def move(batch,device):
    return {k:(v.to(device) if k!="meta" else v) for k,v in batch.items()}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--crop_csv",required=True); p.add_argument("--annotation_csv",required=True); p.add_argument("--image_dir",required=True)
    p.add_argument("--model_dir",required=True); p.add_argument("--out_dir",default="outputs/stage6c_r1_cvpr_anchor_curriculum/eval")
    p.add_argument("--batch_size",type=int,default=16); p.add_argument("--input_size",type=int,default=256); p.add_argument("--heatmap_size",type=int,default=64)
    p.add_argument("--cpu",action="store_true"); p.add_argument("--max_samples",type=int,default=-1)
    args=p.parse_args()
    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    df=merge_crop_and_annotations(args.crop_csv,args.annotation_csv); df=filter_valid_gt_inside_crop(df,margin=0.05)
    if args.max_samples>0: df=df.head(args.max_samples).reset_index(drop=True)
    ds=PoseCropDataset(df,args.image_dir,args.input_size,args.heatmap_size,augment=False)
    loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=0,collate_fn=collate)
    device="cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:",device,"Eval samples:",len(ds))
    model=PhysarumPoseR1().to(device); model.load_state_dict(torch.load(Path(args.model_dir)/"best_model.pt",map_location=device)); model.eval()
    rows=[]; topks=[]
    with torch.no_grad():
        for batch in tqdm(loader,desc="Evaluating Stage 6C-R1"):
            batch=move(batch,device)
            out=model(batch["image"],batch["target_id"],gt_xy=None,gt_anchor_lambda=1.0)
            soft=out["soft_pred_xy"].detach().cpu().numpy()
            heat=out["heat_xy"].detach().cpu().numpy()
            conf=out["heat_conf"].detach().cpu().numpy()
            tk=topk_rows(out,batch,k=5); topks.extend(tk)
            by={}
            for r in tk: by.setdefault(r["sample_id"],[]).append(r)
            for b,meta in enumerate(batch["meta"]):
                se,sx,sy=err_norm(soft[b],meta)
                ax,ay=crop_norm_to_full(heat[b,0,0],heat[b,0,1],meta)
                bx,byy=crop_norm_to_full(heat[b,1,0],heat[b,1,1],meta)
                rx,ry=crop_norm_to_full(heat[b,2,0],heat[b,2,1],meta)
                ae=dist2d(ax,ay,meta["gt_anchor_a_x"],meta["gt_anchor_a_y"])
                be=dist2d(bx,byy,meta["gt_anchor_b_x"],meta["gt_anchor_b_y"])
                re=dist2d(rx,ry,meta["target_orig_x"],meta["target_orig_y"])
                one=sorted(by[meta["sample_id"]],key=lambda z:z["rank"])
                top1=one[0]
                row=dict(meta)
                row.update({
                    "r1_soft_x":sx,"r1_soft_y":sy,"r1_soft_error_px":se,
                    "r1_anchor_a_x":ax,"r1_anchor_a_y":ay,"r1_anchor_a_conf":float(conf[b,0]),"r1_anchor_a_error_px":ae,
                    "r1_anchor_b_x":bx,"r1_anchor_b_y":byy,"r1_anchor_b_conf":float(conf[b,1]),"r1_anchor_b_error_px":be,
                    "r1_anchor_pair_mean_error_px":np.nanmean([ae,be]),
                    "r1_region_x":rx,"r1_region_y":ry,"r1_region_conf":float(conf[b,2]),"r1_region_error_px":re,
                    "r1_top1_x":top1["candidate_x"],"r1_top1_y":top1["candidate_y"],"r1_top1_error_px":top1["candidate_error_px"],"r1_top1_prob":top1["candidate_prob"],
                    "r1_top3_oracle_error_px":min(z["candidate_error_px"] for z in one[:3]),
                    "r1_top5_oracle_error_px":min(z["candidate_error_px"] for z in one[:5]),
                    "r1_anchor_dist_px":dist2d(top1["anchor1_x"],top1["anchor1_y"],top1["anchor2_x"],top1["anchor2_y"]),
                })
                rows.append(row)
    pred=pd.DataFrame(rows); topk=pd.DataFrame(topks)
    pred.to_csv(out_dir/"stage6c_r1_predictions.csv",index=False)
    topk.to_csv(out_dir/"stage6c_r1_topk_predictions.csv",index=False)
    pd.DataFrame([summarize(pred)]).to_csv(out_dir/"stage6c_r1_summary.csv",index=False)
    bytar=[]
    for t,g in pred.groupby("target_name"):
        s=summarize(g); s["target_name"]=t; bytar.append(s)
    pd.DataFrame(bytar).to_csv(out_dir/"stage6c_r1_summary_by_target.csv",index=False)
    eb=error_bins(pred)
    if len(eb): eb.to_csv(out_dir/"stage6c_r1_error_bins.csv",index=False)
    print(pd.DataFrame([summarize(pred)]).T)
if __name__=="__main__": main()
