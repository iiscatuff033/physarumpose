import argparse, cv2, numpy as np, pandas as pd
from pathlib import Path
from src.common import *

def main():
    p=argparse.ArgumentParser(); p.add_argument('--crop_csv',required=True); p.add_argument('--annotation_csv',required=True); p.add_argument('--image_dir',required=True); p.add_argument('--out_csv',default='outputs/stage6e_large_context_crops/crop_metadata.csv'); p.add_argument('--scale',type=float,default=1.8); p.add_argument('--use_all_semantic_anchors',action='store_true'); p.add_argument('--min_size',type=int,default=160); args=p.parse_args()
    df=merge_crop_and_annotations(args.crop_csv,args.annotation_csv); out=[]; image_dir=Path(args.image_dir)
    for _,r in df.iterrows():
        img=cv2.imread(str(image_dir/str(r['occluded_image_name']))); 
        if img is None: continue
        h,w=img.shape[:2]; pts=[(float(r.crop_x1),float(r.crop_y1)),(float(r.crop_x2),float(r.crop_y2))]
        try:
            _,_,ax,ay,bx,by,tx,ty=get_relevant_gt_triplet(r); pts += [(ax,ay),(bx,by),(tx,ty)]
        except Exception: pass
        if args.use_all_semantic_anchors:
            for n in SEMANTIC_ANCHORS:
                x,y=get_joint_xy(r,n)
                if np.isfinite(x) and np.isfinite(y): pts.append((x,y))
        pts=[p for p in pts if np.isfinite(p[0]) and np.isfinite(p[1])]
        if len(pts)<2: continue
        xs=[p[0] for p in pts]; ys=[p[1] for p in pts]; x1,y1,x2,y2=min(xs),min(ys),max(xs),max(ys); size=max(max(x2-x1,y2-y1),args.min_size)*args.scale; cx=(x1+x2)/2; cy=(y1+y2)/2; nx1,ny1,nx2,ny2=clamp_box(cx-size/2,cy-size/2,cx+size/2,cy+size/2,w,h)
        rr=r.copy(); rr['orig_crop_x1']=r.crop_x1; rr['orig_crop_y1']=r.crop_y1; rr['orig_crop_x2']=r.crop_x2; rr['orig_crop_y2']=r.crop_y2; rr['crop_x1']=nx1; rr['crop_y1']=ny1; rr['crop_x2']=nx2; rr['crop_y2']=ny2; rr['stage6e_crop_scale']=args.scale; out.append(rr)
    out=pd.DataFrame(out); Path(args.out_csv).parent.mkdir(parents=True,exist_ok=True); out.to_csv(args.out_csv,index=False); print('Saved',args.out_csv,'rows',len(out))
if __name__=='__main__': main()
