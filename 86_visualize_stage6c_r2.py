import argparse
from pathlib import Path
import cv2, pandas as pd
from PIL import Image, ImageDraw
COLORS=[(255,215,0),(0,200,255),(255,100,180),(120,255,120),(180,130,255)]
def marker(draw,x,y,kind,color,r=7,w=4):
    if pd.isna(x) or pd.isna(y): return
    x=int(round(float(x))); y=int(round(float(y)))
    if kind=="x": draw.line([x-r,y-r,x+r,y+r],fill=color,width=w); draw.line([x-r,y+r,x+r,y-r],fill=color,width=w)
    elif kind=="circle": draw.ellipse([x-r,y-r,x+r,y+r],outline=color,width=w)
    elif kind=="filled": draw.ellipse([x-r,y-r,x+r,y+r],fill=color)
    elif kind=="diamond": draw.polygon([(x,y-r),(x-r,y),(x,y+r),(x+r,y)],outline=color,fill=color)
def visualize(img_path,row,topk,out_path,show_gt_anchors=False,show_title=False):
    bgr=cv2.imread(str(img_path))
    if bgr is None: return
    pil=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)); x1,y1,x2,y2=[int(row[c]) for c in ["crop_x1","crop_y1","crop_x2","crop_y2"]]
    crop=pil.crop((x1,y1,x2,y2)); draw=ImageDraw.Draw(crop)
    def sx(x): return float(x)-x1
    def sy(y): return float(y)-y1
    topk=topk.copy()
    for c in ["candidate_x","anchor1_x","anchor2_x"]:
        if c in topk.columns: topk[c]=topk[c]-x1
    for c in ["candidate_y","anchor1_y","anchor2_y"]:
        if c in topk.columns: topk[c]=topk[c]-y1
    if len(topk):
        topk=topk.sort_values("rank"); first=topk.iloc[0]; ax,ay,bx,by=first["anchor1_x"],first["anchor1_y"],first["anchor2_x"],first["anchor2_y"]
        for _,r in topk.iloc[::-1].iterrows():
            rank=int(r["rank"]); col=COLORS[(rank-1)%len(COLORS)]; width=5 if rank==1 else 3
            draw.line([ax,ay,r["candidate_x"],r["candidate_y"],bx,by],fill=col,width=width); marker(draw,r["candidate_x"],r["candidate_y"],"filled",col,r=5)
        marker(draw,ax,ay,"circle",(0,100,255),r=9); marker(draw,bx,by,"circle",(0,100,255),r=9)
    if show_gt_anchors:
        marker(draw,sx(row["gt_anchor_a_x"]),sy(row["gt_anchor_a_y"]),"circle",(255,255,255),r=7,w=3); marker(draw,sx(row["gt_anchor_b_x"]),sy(row["gt_anchor_b_y"]),"circle",(255,255,255),r=7,w=3)
    marker(draw,sx(row["r2_region_x"]),sy(row["r2_region_y"]),"circle",(0,220,120),r=8); marker(draw,sx(row["r2_top1_x"]),sy(row["r2_top1_y"]),"diamond",(255,215,0),r=10,w=5); marker(draw,sx(row["target_orig_x"]),sy(row["target_orig_y"]),"x",(255,90,0),r=10,w=5)
    if show_title:
        title=f"{row['target_name']} top1={row['r2_top1_error_px']:.1f}px anchor={row['r2_anchor_pair_mean_error_px']:.1f}px"; draw.rectangle([8,8,min(crop.size[0]-8,820),42],fill=(255,255,255),outline=(170,170,170)); draw.text((18,17),title,fill=(0,0,0))
    out_path.parent.mkdir(parents=True,exist_ok=True); crop.save(out_path)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--pred_csv",required=True); p.add_argument("--topk_csv",required=True); p.add_argument("--image_dir",required=True); p.add_argument("--out_dir",default="outputs/stage6c_r2_cvpr_anchor_hardened/eval/visualizations"); p.add_argument("--max_images",type=int,default=150); p.add_argument("--only_good",action="store_true"); p.add_argument("--only_failures",action="store_true"); p.add_argument("--only_good_anchors",action="store_true"); p.add_argument("--show_gt_anchors",action="store_true"); p.add_argument("--show_title",action="store_true"); args=p.parse_args()
    pred=pd.read_csv(args.pred_csv); topk=pd.read_csv(args.topk_csv)
    if args.only_good: pred=pred[(pred["r2_top1_error_px"]<=15)&(pred["r2_anchor_pair_mean_error_px"]<=35)].reset_index(drop=True)
    if args.only_good_anchors: pred=pred[pred["r2_anchor_pair_mean_error_px"]<=35].reset_index(drop=True)
    if args.only_failures: pred=pred[(pred["r2_top1_error_px"]>40)|(pred["r2_anchor_pair_mean_error_px"]>80)].reset_index(drop=True)
    if len(pred)==0: print("No rows after filtering."); return
    sample=pred.sort_values(["r2_anchor_pair_mean_error_px","r2_top1_error_px"]).head(args.max_images) if (args.only_good or args.only_good_anchors) else pred.sample(n=min(args.max_images,len(pred)),random_state=628)
    out_dir=Path(args.out_dir); img_dir=Path(args.image_dir)
    for _,row in sample.iterrows():
        sid=row["sample_id"]; one=topk[topk["sample_id"]==sid].sort_values("rank"); visualize(img_dir/row["occluded_image_name"],row,one,out_dir/(Path(row["occluded_image_name"]).stem+"_r2.png"),args.show_gt_anchors,args.show_title)
    print("Saved visualizations to:",out_dir)
if __name__=="__main__": main()
