import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.common import *

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--crop_csv",required=True); ap.add_argument("--annotation_csv",required=True); ap.add_argument("--image_dir",required=True); ap.add_argument("--model_dir",required=True)
    ap.add_argument("--out_dir",default="outputs/stage6c_gt_anchor_scheduled_physarum_pose/eval")
    ap.add_argument("--batch_size",type=int,default=16); ap.add_argument("--input_size",type=int,default=256); ap.add_argument("--heatmap_size",type=int,default=64)
    ap.add_argument("--cpu",action="store_true"); ap.add_argument("--max_samples",type=int,default=-1)
    args=ap.parse_args()
    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    df=filter_valid_gt_inside_crop(merge_crop_and_annotations(args.crop_csv,args.annotation_csv), margin=0.05)
    if args.max_samples>0: df=df.head(args.max_samples).reset_index(drop=True)
    ds=PoseCropDataset(df,args.image_dir,args.input_size,args.heatmap_size,augment=False)
    loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=0,collate_fn=collate)
    device="cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    model=GTAnchorScheduledModel().to(device); model.load_state_dict(torch.load(Path(args.model_dir)/"best_model.pt",map_location=device)); model.eval()
    rows=[]; topk=[]
    with torch.no_grad():
        for batch in tqdm(loader,desc="Evaluating Stage 6C"):
            for k in ["image","target_id","xy","heatmaps"]: batch[k]=batch[k].to(device)
            out=model(batch["image"],batch["target_id"],gt_xy=None,gt_anchor_mix_lambda=1.0)
            sx=out["soft_pred_xy"].detach().cpu().numpy(); hx=out["heat_xy"].detach().cpu().numpy(); hc=out["heat_conf"].detach().cpu().numpy()
            tk=topk_rows_from_output(out,batch,k=5); topk.extend(tk)
            by={}
            for r in tk: by.setdefault(r["sample_id"],[]).append(r)
            for i,meta in enumerate(batch["meta"]):
                soft_x,soft_y=crop_norm_to_full(sx[i,0],sx[i,1],meta)
                ax,ay=crop_norm_to_full(hx[i,0,0],hx[i,0,1],meta); bx,byy=crop_norm_to_full(hx[i,1,0],hx[i,1,1],meta)
                rx,ry=crop_norm_to_full(hx[i,2,0],hx[i,2,1],meta)
                one=sorted(by[meta["sample_id"]],key=lambda r:r["rank"])
                row=dict(meta)
                row.update({
                    "gt_scheduled_soft_x":soft_x,"gt_scheduled_soft_y":soft_y,"gt_scheduled_soft_error_px":dist2d(soft_x,soft_y,meta["target_orig_x"],meta["target_orig_y"]),
                    "gt_scheduled_anchor_a_x":ax,"gt_scheduled_anchor_a_y":ay,"gt_scheduled_anchor_a_conf":float(hc[i,0]),"gt_scheduled_anchor_a_error_px":dist2d(ax,ay,meta["gt_anchor_a_x"],meta["gt_anchor_a_y"]),
                    "gt_scheduled_anchor_b_x":bx,"gt_scheduled_anchor_b_y":byy,"gt_scheduled_anchor_b_conf":float(hc[i,1]),"gt_scheduled_anchor_b_error_px":dist2d(bx,byy,meta["gt_anchor_b_x"],meta["gt_anchor_b_y"]),
                    "gt_scheduled_region_x":rx,"gt_scheduled_region_y":ry,"gt_scheduled_region_conf":float(hc[i,2]),"gt_scheduled_region_error_px":dist2d(rx,ry,meta["target_orig_x"],meta["target_orig_y"]),
                    "gt_scheduled_top1_x":one[0]["candidate_x"],"gt_scheduled_top1_y":one[0]["candidate_y"],"gt_scheduled_top1_error_px":one[0]["candidate_error_px"],"gt_scheduled_top1_prob":one[0]["candidate_prob"],
                    "gt_scheduled_top3_oracle_error_px":min(r["candidate_error_px"] for r in one[:3]),
                    "gt_scheduled_top5_oracle_error_px":min(r["candidate_error_px"] for r in one[:5]),
                    "gt_scheduled_anchor_dist_px":dist2d(one[0]["anchor1_x"],one[0]["anchor1_y"],one[0]["anchor2_x"],one[0]["anchor2_y"]),
                })
                row["gt_scheduled_anchor_pair_mean_error_px"]=np.nanmean([row["gt_scheduled_anchor_a_error_px"],row["gt_scheduled_anchor_b_error_px"]])
                rows.append(row)
    pred=pd.DataFrame(rows); tkdf=pd.DataFrame(topk)
    pred.to_csv(out_dir/"gt_scheduled_predictions.csv",index=False); tkdf.to_csv(out_dir/"gt_scheduled_topk_predictions.csv",index=False)
    pd.DataFrame([summarize_predictions(pred)]).to_csv(out_dir/"gt_scheduled_summary.csv",index=False)
    by_target=[]
    for t,g in pred.groupby("target_name"):
        s=summarize_predictions(g); s["target_name"]=t; by_target.append(s)
    pd.DataFrame(by_target).to_csv(out_dir/"gt_scheduled_summary_by_target.csv",index=False)
    bins=error_bins(pred)
    if len(bins): bins.to_csv(out_dir/"gt_scheduled_error_bins.csv",index=False)
    print("Summary:"); print(pd.DataFrame([summarize_predictions(pred)]).T)

if __name__=="__main__": main()
