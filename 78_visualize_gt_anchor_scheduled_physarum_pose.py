import argparse
from pathlib import Path
import cv2, pandas as pd
from PIL import Image, ImageDraw

COLORS=[(255,215,0),(0,200,255),(255,100,180),(120,255,120),(180,130,255)]
def marker(draw,x,y,kind,color,r=7,width=4):
    if pd.isna(x) or pd.isna(y): return
    x=int(round(float(x))); y=int(round(float(y)))
    if kind=="x":
        draw.line([x-r,y-r,x+r,y+r],fill=color,width=width); draw.line([x-r,y+r,x+r,y-r],fill=color,width=width)
    elif kind=="circle": draw.ellipse([x-r,y-r,x+r,y+r],outline=color,width=width)
    elif kind=="filled": draw.ellipse([x-r,y-r,x+r,y+r],fill=color)
    elif kind=="diamond": draw.polygon([(x,y-r),(x-r,y),(x,y+r),(x+r,y)],fill=color)

def visualize(img_path,row,topk,out_path,show_gt_anchors=False,show_title=False):
    bgr=cv2.imread(str(img_path))
    if bgr is None: return
    pil=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    x1,y1,x2,y2=[int(row[c]) for c in ["crop_x1","crop_y1","crop_x2","crop_y2"]]
    pil=pil.crop((x1,y1,x2,y2)); draw=ImageDraw.Draw(pil)
    sx=lambda x: float(x)-x1; sy=lambda y: float(y)-y1
    topk=topk.copy()
    for c in ["candidate_x","anchor1_x","anchor2_x"]: topk[c]=topk[c]-x1
    for c in ["candidate_y","anchor1_y","anchor2_y"]: topk[c]=topk[c]-y1
    if len(topk):
        topk=topk.sort_values("rank"); first=topk.iloc[0]
        ax,ay,bx,by=first.anchor1_x,first.anchor1_y,first.anchor2_x,first.anchor2_y
        for _,r in topk.iloc[::-1].iterrows():
            rank=int(r["rank"]); color=COLORS[(rank-1)%len(COLORS)]; width=5 if rank==1 else 3
            draw.line([ax,ay,r.candidate_x,r.candidate_y,bx,by],fill=color,width=width)
            marker(draw,r.candidate_x,r.candidate_y,"filled",color,r=5)
        marker(draw,ax,ay,"circle",(0,100,255),r=9); marker(draw,bx,by,"circle",(0,100,255),r=9)
    if show_gt_anchors:
        marker(draw,sx(row.gt_anchor_a_x),sy(row.gt_anchor_a_y),"circle",(255,255,255),r=7,width=3)
        marker(draw,sx(row.gt_anchor_b_x),sy(row.gt_anchor_b_y),"circle",(255,255,255),r=7,width=3)
    marker(draw,sx(row.gt_scheduled_region_x),sy(row.gt_scheduled_region_y),"circle",(0,220,120),r=8)
    marker(draw,sx(row.gt_scheduled_soft_x),sy(row.gt_scheduled_soft_y),"diamond",(255,215,0),r=10,width=5)
    marker(draw,sx(row.target_orig_x),sy(row.target_orig_y),"x",(255,90,0),r=10,width=5)
    if show_title:
        title=f"{row.target_name} soft={row.gt_scheduled_soft_error_px:.1f}px anchor={row.gt_scheduled_anchor_pair_mean_error_px:.1f}px"
        draw.rectangle([8,8,min(pil.size[0]-8,760),42],fill=(255,255,255),outline=(170,170,170)); draw.text((18,17),title,fill=(0,0,0))
    out_path.parent.mkdir(parents=True,exist_ok=True); pil.save(out_path)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pred_csv",required=True); ap.add_argument("--topk_csv",required=True); ap.add_argument("--image_dir",required=True); ap.add_argument("--out_dir",required=True)
    ap.add_argument("--max_images",type=int,default=150); ap.add_argument("--only_good",action="store_true"); ap.add_argument("--only_failures",action="store_true"); ap.add_argument("--only_good_anchors",action="store_true")
    ap.add_argument("--show_gt_anchors",action="store_true"); ap.add_argument("--show_title",action="store_true")
    args=ap.parse_args()
    pred=pd.read_csv(args.pred_csv); topk=pd.read_csv(args.topk_csv)
    if args.only_good:
        pred=pred[(pred.gt_scheduled_soft_error_px<=20)&(pred.gt_scheduled_anchor_pair_mean_error_px<=35)].reset_index(drop=True)
    if args.only_good_anchors:
        pred=pred[pred.gt_scheduled_anchor_pair_mean_error_px<=35].reset_index(drop=True)
    if args.only_failures:
        pred=pred[(pred.gt_scheduled_soft_error_px>40)|(pred.gt_scheduled_anchor_pair_mean_error_px>80)].reset_index(drop=True)
    if len(pred)==0:
        print("No rows to visualize after filtering."); return
    sample=pred.sort_values(["gt_scheduled_anchor_pair_mean_error_px","gt_scheduled_soft_error_px"]).head(args.max_images) if (args.only_good or args.only_good_anchors) else pred.sample(n=min(args.max_images,len(pred)),random_state=667)
    out_dir=Path(args.out_dir); img_dir=Path(args.image_dir)
    for _,row in sample.iterrows():
        one=topk[topk.sample_id==row.sample_id].sort_values("rank")
        stem=Path(row.occluded_image_name).stem
        visualize(img_dir/row.occluded_image_name,row,one,out_dir/f"{stem}_gt_scheduled.png",args.show_gt_anchors,args.show_title)
    print("Saved visualizations to:",out_dir)

if __name__=="__main__": main()
