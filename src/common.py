from pathlib import Path
import json, cv2
import numpy as np
import pandas as pd
import torch

TARGET_TO_ID={"right_elbow":0,"left_elbow":1,"right_knee":2,"left_knee":3}
ID_TO_TARGET={v:k for k,v in TARGET_TO_ID.items()}
SEMANTIC_ANCHORS=["right_shoulder","right_wrist","right_hip","right_ankle","left_shoulder","left_wrist","left_hip","left_ankle"]
ANCHOR_TO_ID={n:i for i,n in enumerate(SEMANTIC_ANCHORS)}
MPII_ID={"right_ankle":0,"right_knee":1,"right_hip":2,"left_hip":3,"left_knee":4,"left_ankle":5,"right_wrist":10,"right_elbow":11,"right_shoulder":12,"left_shoulder":13,"left_elbow":14,"left_wrist":15}
TARGET_TO_ANCHORS={"right_elbow":("right_shoulder","right_wrist"),"left_elbow":("left_shoulder","left_wrist"),"right_knee":("right_hip","right_ankle"),"left_knee":("left_hip","left_ankle")}
TARGET_PAIR_INDEX={k:(ANCHOR_TO_ID[a],ANCHOR_TO_ID[b]) for k,(a,b) in TARGET_TO_ANCHORS.items()}
TARGET_PAIR_INDEX_BY_ID={TARGET_TO_ID[k]:v for k,v in TARGET_PAIR_INDEX.items()}

def save_json(obj,path):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    with open(p,'w',encoding='utf-8') as f: json.dump(obj,f,indent=2)

def safe_float(v,default=np.nan):
    try:
        if v is None: return default
        if isinstance(v,float) and np.isnan(v): return default
        return float(v)
    except Exception: return default

def dist2d(x1,y1,x2,y2):
    vals=[x1,y1,x2,y2]
    if any(pd.isna(v) for v in vals): return np.nan
    return float(np.sqrt((float(x1)-float(x2))**2+(float(y1)-float(y2))**2))

def make_heatmap(xn,yn,h,w,sigma=2.2):
    heat=np.zeros((h,w),dtype=np.float32)
    if not np.isfinite(xn) or not np.isfinite(yn) or xn<0 or xn>1 or yn<0 or yn>1: return heat
    cx=xn*(w-1); cy=yn*(h-1); yy,xx=np.mgrid[0:h,0:w]
    return np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*sigma*sigma)).astype(np.float32)

def crop_and_resize(rgb,x1,y1,x2,y2,input_size):
    crop=rgb[int(y1):int(y2),int(x1):int(x2)]
    if crop.size==0: raise ValueError('Empty crop')
    return cv2.resize(crop,(input_size,input_size),interpolation=cv2.INTER_LINEAR)

def image_to_tensor(rgb): return torch.from_numpy(rgb.transpose(2,0,1)).float()/255.0

def full_to_crop_norm(x,y,x1,y1,x2,y2):
    return (safe_float(x)-float(x1))/max(float(x2)-float(x1),1.0),(safe_float(y)-float(y1))/max(float(y2)-float(y1),1.0)

def crop_norm_to_full(xn,yn,x1,y1,x2,y2):
    return float(x1)+float(xn)*max(float(x2)-float(x1),1.0), float(y1)+float(yn)*max(float(y2)-float(y1),1.0)

def get_joint_xy(row,name):
    j=MPII_ID[name]
    return safe_float(row.get(f'j{j}_orig_x',np.nan)), safe_float(row.get(f'j{j}_orig_y',np.nan))

def get_target_xy(row):
    tx,ty=get_joint_xy(row,row['target_name'])
    if not np.isfinite(tx) or not np.isfinite(ty): tx,ty=safe_float(row.get('target_orig_x',np.nan)),safe_float(row.get('target_orig_y',np.nan))
    return tx,ty

def get_relevant_gt_triplet(row):
    a,b=TARGET_TO_ANCHORS[row['target_name']]
    ax,ay=get_joint_xy(row,a); bx,by=get_joint_xy(row,b); tx,ty=get_target_xy(row)
    return a,b,ax,ay,bx,by,tx,ty

def merge_crop_and_annotations(crop_csv,ann_csv):
    crop=pd.read_csv(crop_csv); ann=pd.read_csv(ann_csv)
    keep=['occluded_image_name','target_name']
    for j in range(16):
        for s in ['orig_x','orig_y','mask']:
            c=f'j{j}_{s}'
            if c in ann.columns: keep.append(c)
    ann=ann[[c for c in keep if c in ann.columns]].drop_duplicates(subset=['occluded_image_name','target_name'])
    return crop.merge(ann,on=['occluded_image_name','target_name'],how='left')

def filter_valid_relevant_pair_inside_crop(df,margin=0.05):
    rows=[]
    for _,r in df.iterrows():
        try:
            _,_,ax,ay,bx,by,tx,ty=get_relevant_gt_triplet(r)
            ok=True
            for x,y in [(ax,ay),(bx,by),(tx,ty)]:
                if not np.isfinite(x) or not np.isfinite(y): ok=False; break
                xn,yn=full_to_crop_norm(x,y,r.crop_x1,r.crop_y1,r.crop_x2,r.crop_y2)
                if xn<-margin or xn>1+margin or yn<-margin or yn>1+margin: ok=False; break
            if ok: rows.append(r)
        except Exception: pass
    return pd.DataFrame(rows).reset_index(drop=True) if rows else df.reset_index(drop=True)

def split_by_image(df,val_ratio=0.2,seed=42):
    rng=np.random.default_rng(seed); key=df['image_name'].fillna(df['occluded_image_name']) if 'image_name' in df.columns else df['occluded_image_name']
    names=key.unique(); rng.shuffle(names); val=set(names[:max(1,int(len(names)*val_ratio))])
    return df[~key.isin(val)].reset_index(drop=True),df[key.isin(val)].reset_index(drop=True)

def clamp_box(x1,y1,x2,y2,w,h):
    x1=max(0,min(int(round(x1)),w-1)); y1=max(0,min(int(round(y1)),h-1)); x2=max(x1+2,min(int(round(x2)),w)); y2=max(y1+2,min(int(round(y2)),h)); return x1,y1,x2,y2
