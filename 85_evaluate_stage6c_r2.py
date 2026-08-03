import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.common import merge_crop_and_annotations, filter_valid_gt_inside_crop, dist2d
from src.dataset import PoseCropDataset
from src.model import PhysarumPoseR1
from src.eval_utils import crop_norm_to_full, err_norm, topk_rows

def collate(batch):
    return {k: torch.stack([b[k] for b in batch],0) for k in ["image","target_id","xy","heatmaps"]} | {"meta":[b["meta"] for b in batch]}
def move(batch,device):
    return {k:(v.to(device) if k!="meta" else v) for k,v in batch.items()}

def summarize_r2(df):
    out=dict(count=len(df), r2_top1_mean_px=df["r2_top1_error_px"].mean(), r2_top1_median_px=df["r2_top1_error_px"].median(),
             r2_soft_mean_px=df["r2_soft_error_px"].mean(), r2_soft_median_px=df["r2_soft_error_px"].median(),
             r2_region_mean_px=df["r2_region_error_px"].mean(), r2_anchor_pair_mean_px=df["r2_anchor_pair_mean_error_px"].mean(),
             r2_anchor_pair_median_px=df["r2_anchor_pair_mean_error_px"].median(), top3_oracle_mean_px=df["r2_top3_oracle_error_px"].mean(),
             top5_oracle_mean_px=df["r2_top5_oracle_error_px"].mean(), top1_win_vs_soft_rate=(df["r2_top1_error_px"]<df["r2_soft_error_px"]).mean(),
             anchor_good_20_rate=(df["r2_anchor_pair_mean_error_px"]<=20).mean(), anchor_good_30_rate=(df["r2_anchor_pair_mean_error_px"]<=30).mean(),
             anchor_good_35_rate=(df["r2_anchor_pair_mean_error_px"]<=35).mean(), anchor_good_40_rate=(df["r2_anchor_pair_mean_error_px"]<=40).mean(),
             paper_good_rate=((df["r2_top1_error_px"]<=15)&(df["r2_anchor_pair_mean_error_px"]<=35)).mean())
    if "yolo_target_error_px" in df.columns:
        out["yolo_mean_px"]=df["yolo_target_error_px"].mean(); out["top1_win_vs_yolo_rate"]=(df["r2_top1_error_px"]<df["yolo_target_error_px"]).mean()
    if "gated_error_px" in df.columns:
        out["previous_gated_mean_px"]=df["gated_error_px"].mean(); out["top1_win_vs_previous_gated_rate"]=(df["r2_top1_error_px"]<df["gated_error_px"]).mean()
    return out

def make_r2_error_bins(df):
    if "yolo_target_error_px" not in df.columns: return pd.DataFrame()
    bins=[0,20,40,60,80,100,150,999999]; labels=["0-20","20-40","40-60","60-80","80-100","100-150","150+"]
    tmp=df.copy(); tmp["yolo_error_bin"]=pd.cut(tmp["yolo_target_error_px"],bins=bins,labels=labels,include_lowest=True,right=False)
    return tmp.groupby("yolo_error_bin", observed=False).agg(count=("r2_top1_error_px","count"), yolo_mean_px=("yolo_target_error_px","mean"),
        r2_top1_mean_px=("r2_top1_error_px","mean"), r2_soft_mean_px=("r2_soft_error_px","mean"),
        r2_anchor_pair_mean_px=("r2_anchor_pair_mean_error_px","mean"), anchor_good_35_rate=("r2_anchor_pair_mean_error_px", lambda s:(s<=35).mean()),
        top5_oracle_mean_px=("r2_top5_oracle_error_px","mean")).reset_index()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--crop_csv",required=True); p.add_argument("--annotation_csv",required=True); p.add_argument("--image_dir",required=True)
    p.add_argument("--model_dir",required=True); p.add_argument("--out_dir",default="outputs/stage6c_r2_cvpr_anchor_hardened/eval")
    p.add_argument("--batch_size",type=int,default=16); p.add_argument("--input_size",type=int,default=256); p.add_argument("--heatmap_size",type=int,default=64)
    p.add_argument("--cpu",action="store_true"); p.add_argument("--max_samples",type=int,default=-1); args=p.parse_args()
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
        for batch in tqdm(loader,desc="Evaluating Stage 6C-R2"):
            batch=move(batch,device); out=model(batch["image"],batch["target_id"],gt_xy=None,gt_anchor_lambda=1.0)
            soft=out["soft_pred_xy"].detach().cpu().numpy(); heat=out["heat_xy"].detach().cpu().numpy(); conf=out["heat_conf"].detach().cpu().numpy()
            tk=topk_rows(out,batch,k=5); topks.extend(tk); by={}
            for r in tk: by.setdefault(r["sample_id"],[]).append(r)
            for b,meta in enumerate(batch["meta"]):
                se,sx,sy=err_norm(soft[b],meta)
                ax,ay=crop_norm_to_full(heat[b,0,0],heat[b,0,1],meta); bx,byy=crop_norm_to_full(heat[b,1,0],heat[b,1,1],meta); rx,ry=crop_norm_to_full(heat[b,2,0],heat[b,2,1],meta)
                ae=dist2d(ax,ay,meta["gt_anchor_a_x"],meta["gt_anchor_a_y"]); be=dist2d(bx,byy,meta["gt_anchor_b_x"],meta["gt_anchor_b_y"]); re=dist2d(rx,ry,meta["target_orig_x"],meta["target_orig_y"])
                one=sorted(by[meta["sample_id"]],key=lambda z:z["rank"]); top1=one[0]
                row=dict(meta); pair=np.nanmean([ae,be])
                row.update({"r2_soft_x":sx,"r2_soft_y":sy,"r2_soft_error_px":se,"r2_anchor_a_x":ax,"r2_anchor_a_y":ay,"r2_anchor_a_conf":float(conf[b,0]),"r2_anchor_a_error_px":ae,
                    "r2_anchor_b_x":bx,"r2_anchor_b_y":byy,"r2_anchor_b_conf":float(conf[b,1]),"r2_anchor_b_error_px":be,"r2_anchor_pair_mean_error_px":pair,
                    "r2_region_x":rx,"r2_region_y":ry,"r2_region_conf":float(conf[b,2]),"r2_region_error_px":re,
                    "r2_top1_x":top1["candidate_x"],"r2_top1_y":top1["candidate_y"],"r2_top1_error_px":top1["candidate_error_px"],"r2_top1_prob":top1["candidate_prob"],
                    "r2_top3_oracle_error_px":min(z["candidate_error_px"] for z in one[:3]),"r2_top5_oracle_error_px":min(z["candidate_error_px"] for z in one[:5]),
                    "r2_anchor_dist_px":dist2d(top1["anchor1_x"],top1["anchor1_y"],top1["anchor2_x"],top1["anchor2_y"]),"r2_paper_good":int((top1["candidate_error_px"]<=15) and (pair<=35))})
                rows.append(row)
    pred=pd.DataFrame(rows); topk=pd.DataFrame(topks)
    pred.to_csv(out_dir/"stage6c_r2_predictions.csv",index=False); topk.to_csv(out_dir/"stage6c_r2_topk_predictions.csv",index=False)
    summary=pd.DataFrame([summarize_r2(pred)]); summary.to_csv(out_dir/"stage6c_r2_summary.csv",index=False)
    bytar=[]
    for t,g in pred.groupby("target_name"):
        s=summarize_r2(g); s["target_name"]=t; bytar.append(s)
    pd.DataFrame(bytar).to_csv(out_dir/"stage6c_r2_summary_by_target.csv",index=False)
    eb=make_r2_error_bins(pred)
    if len(eb): eb.to_csv(out_dir/"stage6c_r2_error_bins.csv",index=False)
    print(summary.T)
if __name__=="__main__": main()
