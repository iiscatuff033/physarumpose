import argparse
from pathlib import Path
import cv2, pandas as pd
from PIL import Image, ImageDraw
PATH_COLORS=[(255,215,0),(0,200,255),(255,100,180),(120,255,120),(180,130,255)]
def draw_marker(draw,x,y,kind,color,r=7,width=4):
    if pd.isna(x) or pd.isna(y): return
    x=int(round(float(x))); y=int(round(float(y)))
    if kind=='x': draw.line([x-r,y-r,x+r,y+r],fill=color,width=width); draw.line([x-r,y+r,x+r,y-r],fill=color,width=width)
    elif kind=='circle': draw.ellipse([x-r,y-r,x+r,y+r],outline=color,width=width)
    elif kind=='filled': draw.ellipse([x-r,y-r,x+r,y+r],fill=color)
    elif kind=='diamond': draw.polygon([(x,y-r),(x-r,y),(x,y+r),(x+r,y)],outline=color,fill=color)
def visualize_one(image_path,row,topk_rows,out_path,show_title=False,hide_previous=True):
    bgr=cv2.imread(str(image_path))
    if bgr is None: return
    pil_full=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)); x1,y1,x2,y2=int(row['crop_x1']),int(row['crop_y1']),int(row['crop_x2']),int(row['crop_y2']); pil=pil_full.crop((x1,y1,x2,y2)); draw=ImageDraw.Draw(pil)
    def sx(x): return float(x)-x1
    def sy(y): return float(y)-y1
    topk_rows=topk_rows.copy()
    for c in ['candidate_x','anchor1_x','anchor2_x']:
        if c in topk_rows.columns: topk_rows[c]=topk_rows[c]-x1
    for c in ['candidate_y','anchor1_y','anchor2_y']:
        if c in topk_rows.columns: topk_rows[c]=topk_rows[c]-y1
    if len(topk_rows)>0:
        topk_rows=topk_rows.sort_values('rank'); first=topk_rows.iloc[0]; ax,ay=first['anchor1_x'],first['anchor1_y']; bx,by=first['anchor2_x'],first['anchor2_y']
        for _,r in topk_rows.iloc[::-1].iterrows():
            color=PATH_COLORS[(int(r['rank'])-1)%len(PATH_COLORS)]; width=5 if int(r['rank'])==1 else 3; draw.line([ax,ay,r['candidate_x'],r['candidate_y'],bx,by],fill=color,width=width); draw_marker(draw,r['candidate_x'],r['candidate_y'],'filled',color,r=5)
        draw_marker(draw,ax,ay,'circle',(0,100,255),r=9); draw_marker(draw,bx,by,'circle',(0,100,255),r=9)
    draw_marker(draw,sx(row['anchor_faithful_region_x']),sy(row['anchor_faithful_region_y']),'circle',(0,220,120),r=8); draw_marker(draw,sx(row['anchor_faithful_soft_x']),sy(row['anchor_faithful_soft_y']),'diamond',(255,215,0),r=10,width=5); draw_marker(draw,sx(row['target_orig_x']),sy(row['target_orig_y']),'x',(255,90,0),r=10,width=5)
    if show_title:
        title=f"{row['target_name']} | soft={row['anchor_faithful_soft_error_px']:.1f}px | anchor={row['anchor_faithful_anchor_pair_mean_error_px']:.1f}px"; draw.rectangle([8,8,min(pil.size[0]-8,820),42],fill=(255,255,255),outline=(170,170,170)); draw.text((18,17),title,fill=(0,0,0))
    out_path.parent.mkdir(parents=True,exist_ok=True); pil.save(out_path)
def main():
    p=argparse.ArgumentParser(); p.add_argument('--pred_csv',required=True); p.add_argument('--topk_csv',required=True); p.add_argument('--image_dir',required=True); p.add_argument('--out_dir',default='outputs/stage6b_anchor_faithful_physarum_pose/eval/visualizations'); p.add_argument('--max_images',type=int,default=150); p.add_argument('--only_good',action='store_true'); p.add_argument('--only_failures',action='store_true'); p.add_argument('--only_good_anchors',action='store_true'); p.add_argument('--only_improved_vs_yolo',action='store_true'); p.add_argument('--show_title',action='store_true'); args=p.parse_args()
    pred=pd.read_csv(args.pred_csv); topk=pd.read_csv(args.topk_csv)
    if args.only_good: pred=pred[(pred['anchor_faithful_soft_error_px']<=20)&(pred['anchor_faithful_anchor_pair_mean_error_px']<=35)].reset_index(drop=True)
    if args.only_good_anchors: pred=pred[pred['anchor_faithful_anchor_pair_mean_error_px']<=35].reset_index(drop=True)
    if args.only_failures: pred=pred[(pred['anchor_faithful_soft_error_px']>40)|(pred['anchor_faithful_anchor_pair_mean_error_px']>80)].reset_index(drop=True)
    if args.only_improved_vs_yolo and 'yolo_target_error_px' in pred.columns: pred=pred[pred['anchor_faithful_soft_error_px']<pred['yolo_target_error_px']].reset_index(drop=True)
    if len(pred)==0: print('No rows to visualize after filtering.'); return
    sample=pred.sort_values(['anchor_faithful_anchor_pair_mean_error_px','anchor_faithful_soft_error_px']).head(args.max_images) if (args.only_good or args.only_good_anchors) else pred.sample(n=min(args.max_images,len(pred)),random_state=666)
    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True,exist_ok=True); image_dir=Path(args.image_dir)
    for _,row in sample.iterrows():
        sid=row['sample_id']; one_topk=topk[topk['sample_id']==sid].sort_values('rank'); stem=Path(row['occluded_image_name']).stem; visualize_one(image_dir/row['occluded_image_name'],row,one_topk,out_dir/f'{stem}_anchor_faithful.png',show_title=args.show_title)
    print('Saved visualizations to:',out_dir)
if __name__=='__main__': main()
