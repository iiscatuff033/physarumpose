import numpy as np, pandas as pd
from .common import dist2d

def crop_norm_to_full(xn,yn,meta):
    x=float(meta["crop_x1"])+float(xn)*max(float(meta["crop_x2"])-float(meta["crop_x1"]),1.0)
    y=float(meta["crop_y1"])+float(yn)*max(float(meta["crop_y2"])-float(meta["crop_y1"]),1.0)
    return x,y

def err_norm(xy,meta):
    x,y=crop_norm_to_full(xy[0],xy[1],meta)
    return dist2d(x,y,meta["target_orig_x"],meta["target_orig_y"]),x,y

def topk_rows(out,batch,k=5):
    cand=out["candidate_xy"].detach().cpu().numpy()
    probs=out["candidate_probs"].detach().cpu().numpy()
    scores=out["candidate_scores"].detach().cpu().numpy()
    rows=[]
    for b,meta in enumerate(batch["meta"]):
        idx=np.argsort(-probs[b])[:k]
        ax,ay=crop_norm_to_full(out["path_anchor_a"][b,0].detach().cpu().numpy(),out["path_anchor_a"][b,1].detach().cpu().numpy(),meta)
        bx,by=crop_norm_to_full(out["path_anchor_b"][b,0].detach().cpu().numpy(),out["path_anchor_b"][b,1].detach().cpu().numpy(),meta)
        for rank,j in enumerate(idx,start=1):
            x,y=crop_norm_to_full(cand[b,j,0],cand[b,j,1],meta)
            rows.append(dict(sample_id=meta["sample_id"], image_name=meta.get("image_name",""),
                             occluded_image_name=meta["occluded_image_name"], target_name=meta["target_name"],
                             rank=rank, candidate_index=int(j), candidate_x=x, candidate_y=y,
                             candidate_error_px=dist2d(x,y,meta["target_orig_x"],meta["target_orig_y"]),
                             candidate_prob=float(probs[b,j]), candidate_score=float(scores[b,j]),
                             anchor1_x=ax, anchor1_y=ay, anchor2_x=bx, anchor2_y=by))
    return rows

def summarize(df):
    out=dict(
        count=len(df),
        r1_soft_mean_px=df["r1_soft_error_px"].mean(),
        r1_soft_median_px=df["r1_soft_error_px"].median(),
        r1_top1_mean_px=df["r1_top1_error_px"].mean(),
        r1_top1_median_px=df["r1_top1_error_px"].median(),
        r1_region_mean_px=df["r1_region_error_px"].mean(),
        r1_anchor_pair_mean_px=df["r1_anchor_pair_mean_error_px"].mean(),
        r1_anchor_pair_median_px=df["r1_anchor_pair_mean_error_px"].median(),
        top3_oracle_mean_px=df["r1_top3_oracle_error_px"].mean(),
        top5_oracle_mean_px=df["r1_top5_oracle_error_px"].mean(),
        top1_win_vs_soft_rate=(df["r1_top1_error_px"]<df["r1_soft_error_px"]).mean(),
    )
    if "yolo_target_error_px" in df.columns:
        out["yolo_mean_px"]=df["yolo_target_error_px"].mean()
        out["top1_win_vs_yolo_rate"]=(df["r1_top1_error_px"]<df["yolo_target_error_px"]).mean()
    if "gated_error_px" in df.columns:
        out["previous_gated_mean_px"]=df["gated_error_px"].mean()
        out["top1_win_vs_previous_gated_rate"]=(df["r1_top1_error_px"]<df["gated_error_px"]).mean()
    return out

def error_bins(df):
    if "yolo_target_error_px" not in df.columns: return pd.DataFrame()
    bins=[0,20,40,60,80,100,150,999999]
    labels=["0-20","20-40","40-60","60-80","80-100","100-150","150+"]
    tmp=df.copy()
    tmp["yolo_error_bin"]=pd.cut(tmp["yolo_target_error_px"],bins=bins,labels=labels,include_lowest=True,right=False)
    return tmp.groupby("yolo_error_bin", observed=False).agg(
        count=("r1_top1_error_px","count"),
        yolo_mean_px=("yolo_target_error_px","mean"),
        r1_top1_mean_px=("r1_top1_error_px","mean"),
        r1_soft_mean_px=("r1_soft_error_px","mean"),
        r1_anchor_pair_mean_px=("r1_anchor_pair_mean_error_px","mean"),
        top5_oracle_mean_px=("r1_top5_oracle_error_px","mean"),
    ).reset_index()
