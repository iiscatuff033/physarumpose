import argparse, random
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.io import loadmat
from tqdm import tqdm
from src.utils_full import MPII_ID_TO_NAME, TARGET_JOINTS

def _as_list(x):
    if x is None: return []
    if isinstance(x, np.ndarray): return [i for i in x.flatten() if i is not None]
    return [x]
def _get_field(obj,name,default=None): return getattr(obj,name,default) if hasattr(obj,name) else default
def _to_float(x, default=None):
    try:
        if isinstance(x,np.ndarray):
            if x.size==0: return default
            return float(x.flatten()[0])
        return float(x)
    except Exception: return default
def _to_int(x, default=None):
    try:
        if isinstance(x,np.ndarray):
            if x.size==0: return default
            return int(x.flatten()[0])
        return int(x)
    except Exception: return default
def parse_visible_flag(point):
    vis=_get_field(point,'is_visible',None)
    if vis is None: return 1
    if isinstance(vis,np.ndarray):
        if vis.size==0: return 1
        vis=vis.flatten()[0]
    try: return int(vis)
    except Exception: return 1
def parse_person_points(rect):
    pts={}
    annopoints=_get_field(rect,'annopoints',None)
    if annopoints is None: return pts
    for pc in _as_list(annopoints):
        for p in _as_list(_get_field(pc,'point',None)):
            jid=_to_int(_get_field(p,'id',None),None)
            if jid is None or jid not in MPII_ID_TO_NAME: continue
            x=_to_float(_get_field(p,'x',None),None); y=_to_float(_get_field(p,'y',None),None)
            if x is None or y is None: continue
            pts[jid]={'x':x,'y':y,'visible':parse_visible_flag(p)}
    return pts
def get_image_name(ann):
    im=_get_field(ann,'image',None)
    if im is None: return ''
    name=_get_field(im,'name','')
    if isinstance(name,np.ndarray):
        if name.size==0: return ''
        name=name.flatten()[0]
    return str(name)
def compute_norm(points):
    xy=np.array([[p['x'],p['y']] for p in points.values()],dtype=np.float32)
    if len(xy)<3: return None,None
    center=xy.mean(axis=0); min_xy=xy.min(axis=0); max_xy=xy.max(axis=0)
    scale=max(float((max_xy-min_xy)[0]), float((max_xy-min_xy)[1]), 1.0)
    return center,scale
def norm_xy(x,y,center,scale):
    a=(np.array([x,y],dtype=np.float32)-center)/scale
    return float(a[0]),float(a[1])
def make_box(x,y,w,h,patch,jitter):
    jx=random.randint(-jitter,jitter) if jitter>0 else 0
    jy=random.randint(-jitter,jitter) if jitter>0 else 0
    cx=int(round(x+jx)); cy=int(round(y+jy)); half=patch//2
    return max(0,cx-half), max(0,cy-half), min(w-1,cx+half), min(h-1,cy+half)
def apply_occ(img,box,mode):
    x1,y1,x2,y2=box; out=img.copy()
    if x2<=x1 or y2<=y1: return out
    if mode=='black': out[y1:y2,x1:x2]=(0,0,0)
    elif mode=='blur': out[y1:y2,x1:x2]=cv2.GaussianBlur(out[y1:y2,x1:x2],(31,31),0)
    elif mode=='noise': out[y1:y2,x1:x2]=np.random.randint(0,255,out[y1:y2,x1:x2].shape,dtype=np.uint8)
    else: out[y1:y2,x1:x2]=(127,127,127)
    return out
def vis_image(img,box,tx,ty,out_path):
    pil=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)); draw=ImageDraw.Draw(pil)
    x1,y1,x2,y2=box; draw.rectangle([x1,y1,x2,y2], outline=(255,0,0), width=3)
    r=6; tx=int(round(tx)); ty=int(round(ty))
    draw.line([tx-r,ty-r,tx+r,ty+r], fill=(255,165,0), width=3)
    draw.line([tx-r,ty+r,tx+r,ty-r], fill=(255,165,0), width=3)
    out_path.parent.mkdir(parents=True,exist_ok=True); pil.save(out_path)
def make_row(points,image_name,pidx,aidx,task_id,tinfo,center,scale,out_name,box):
    tj=tinfo['mpii_id']; tx,ty=norm_xy(points[tj]['x'],points[tj]['y'],center,scale)
    row={'image_name':image_name,'occluded_image_name':out_name,'ann_index':aidx,'person_index':pidx,
         'target_task_id':task_id,'target_name':tinfo['name'],'target_mpii_id':tj,
         'target_x':tx,'target_y':ty,'target_orig_x':float(points[tj]['x']),'target_orig_y':float(points[tj]['y']),
         'center_x':float(center[0]),'center_y':float(center[1]),'scale':float(scale),
         'occ_x1':box[0],'occ_y1':box[1],'occ_x2':box[2],'occ_y2':box[3]}
    for j in range(16):
        if j in points:
            x,y=norm_xy(points[j]['x'],points[j]['y'],center,scale); m=1.0 if points[j]['visible']==1 else 0.0
            ox=float(points[j]['x']); oy=float(points[j]['y'])
        else:
            x=y=m=ox=oy=0.0
        if j==tj: x=y=m=0.0
        row[f'j{j}_x']=x; row[f'j{j}_y']=y; row[f'j{j}_mask']=m; row[f'j{j}_orig_x']=ox; row[f'j{j}_orig_y']=oy
    return row
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mat_path',required=True); ap.add_argument('--image_dir',required=True)
    ap.add_argument('--out_dir',default='data/processed/synth_occlusion_mpii')
    ap.add_argument('--max_samples',type=int,default=2000); ap.add_argument('--patch_size',type=int,default=70)
    ap.add_argument('--jitter',type=int,default=10); ap.add_argument('--patch_mode',default='gray',choices=['gray','black','blur','noise'])
    ap.add_argument('--min_visible',type=int,default=8); ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--include_test_split',action='store_true')
    args=ap.parse_args(); random.seed(args.seed); np.random.seed(args.seed)
    mat=loadmat(args.mat_path,struct_as_record=False,squeeze_me=True); release=mat['RELEASE']
    annolist=_as_list(_get_field(release,'annolist')); img_train=_get_field(release,'img_train',None)
    if img_train is not None: img_train=np.array(img_train).flatten()
    out_dir=Path(args.out_dir); out_img=out_dir/'images'; vis_dir=out_dir/'sample_visualizations'; out_img.mkdir(parents=True,exist_ok=True); vis_dir.mkdir(parents=True,exist_ok=True)
    image_dir=Path(args.image_dir); rows=[]; saved=0
    for aidx,ann in enumerate(tqdm(annolist,desc='Creating synthetic occlusions')):
        if not args.include_test_split and img_train is not None:
            try:
                if int(img_train[aidx])!=1: continue
            except Exception: pass
        image_name=get_image_name(ann); img_path=image_dir/image_name
        if not image_name or not img_path.exists(): continue
        img=cv2.imread(str(img_path));
        if img is None: continue
        h,w=img.shape[:2]
        for pidx,rect in enumerate(_as_list(_get_field(ann,'annorect',None))):
            points=parse_person_points(rect)
            if not points: continue
            if sum(1 for p in points.values() if p['visible']==1)<args.min_visible: continue
            center,scale=compute_norm(points)
            if center is None: continue
            for task_id,tinfo in TARGET_JOINTS.items():
                tj=tinfo['mpii_id']
                if tj not in points or points[tj]['visible']!=1: continue
                tx,ty=points[tj]['x'],points[tj]['y']
                if tx<0 or tx>=w or ty<0 or ty>=h: continue
                box=make_box(tx,ty,w,h,args.patch_size,args.jitter); occ=apply_occ(img,box,args.patch_mode)
                stem=Path(image_name).stem; out_name=f'{stem}_ann{aidx}_p{pidx}_{tinfo["name"]}.jpg'
                cv2.imwrite(str(out_img/out_name),occ)
                rows.append(make_row(points,image_name,pidx,aidx,task_id,tinfo,center,scale,out_name,box))
                if saved<40: vis_image(occ,box,tx,ty,vis_dir/out_name)
                saved+=1
                if saved>=args.max_samples: break
            if saved>=args.max_samples: break
        if saved>=args.max_samples: break
    df=pd.DataFrame(rows); csv=out_dir/'synth_occlusion_metadata.csv'; df.to_csv(csv,index=False)
    print('\nSaved images:',out_img); print('Saved metadata:',csv); print('Total samples:',len(df))
    if len(df)>0: print(df['target_name'].value_counts())
if __name__=='__main__': main()
