from pathlib import Path
import cv2, numpy as np, torch
from torch.utils.data import Dataset
from .common import *

class SemanticOffsetDataset(Dataset):
    def __init__(self,df,image_dir,input_size=384,heatmap_size=96,sigma=2.2,augment=False):
        self.df=df.reset_index(drop=True); self.image_dir=Path(image_dir); self.input_size=int(input_size); self.heatmap_size=int(heatmap_size); self.sigma=float(sigma); self.augment=bool(augment)
    def __len__(self): return len(self.df)
    def _load_crop(self,row):
        path=self.image_dir/str(row['occluded_image_name']); bgr=cv2.imread(str(path))
        if bgr is None: raise FileNotFoundError(path)
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB); ih,iw=rgb.shape[:2]
        x1,y1,x2,y2=[int(row[c]) for c in ['crop_x1','crop_y1','crop_x2','crop_y2']]
        crop=crop_and_resize(rgb,x1,y1,x2,y2,self.input_size)
        if self.augment:
            if np.random.rand()<0.5: crop=np.clip(crop.astype(np.float32)*np.random.uniform(0.85,1.15),0,255).astype(np.uint8)
            if np.random.rand()<0.25: crop=np.clip(crop.astype(np.float32)+np.random.normal(0,3,crop.shape),0,255).astype(np.uint8)
        return crop,iw,ih,x1,y1,x2,y2
    def _inside(self,x,y,x1,y1,x2,y2):
        if not np.isfinite(x) or not np.isfinite(y): return False
        xn,yn=full_to_crop_norm(x,y,x1,y1,x2,y2); return 0<=xn<=1 and 0<=yn<=1
    def __getitem__(self,idx):
        row=self.df.iloc[idx]; crop,iw,ih,x1,y1,x2,y2=self._load_crop(row)
        target=str(row['target_name']); tid=TARGET_TO_ID[target]
        anchor_xy=np.zeros((8,2),np.float32); anchor_mask=np.zeros(8,np.float32); ah=np.zeros((8,self.heatmap_size,self.heatmap_size),np.float32)
        for i,name in enumerate(SEMANTIC_ANCHORS):
            x,y=get_joint_xy(row,name)
            if self._inside(x,y,x1,y1,x2,y2):
                xn,yn=full_to_crop_norm(x,y,x1,y1,x2,y2); xn,yn=np.clip(xn,0,1),np.clip(yn,0,1)
                anchor_xy[i]=[xn,yn]; anchor_mask[i]=1; ah[i]=make_heatmap(xn,yn,self.heatmap_size,self.heatmap_size,self.sigma)
            else: anchor_xy[i]=[0.5,0.5]
        region_xy=np.zeros((4,2),np.float32); region_mask=np.zeros(4,np.float32); rh=np.zeros((4,self.heatmap_size,self.heatmap_size),np.float32)
        tx,ty=get_target_xy(row)
        if self._inside(tx,ty,x1,y1,x2,y2):
            xn,yn=full_to_crop_norm(tx,ty,x1,y1,x2,y2); xn,yn=np.clip(xn,0,1),np.clip(yn,0,1)
            region_xy[tid]=[xn,yn]; region_mask[tid]=1; rh[tid]=make_heatmap(xn,yn,self.heatmap_size,self.heatmap_size,self.sigma)
        a,b=TARGET_TO_ANCHORS[target]
        meta=dict(sample_id=int(row.get('sample_id',idx)),image_name=str(row.get('image_name','')),occluded_image_name=str(row['occluded_image_name']),target_name=target,target_id=int(tid),img_w=float(iw),img_h=float(ih),crop_x1=float(x1),crop_y1=float(y1),crop_x2=float(x2),crop_y2=float(y2),target_orig_x=float(tx),target_orig_y=float(ty),anchor_a_name=a,anchor_b_name=b)
        for name in SEMANTIC_ANCHORS:
            gx,gy=get_joint_xy(row,name); meta[f'gt_{name}_x']=float(gx); meta[f'gt_{name}_y']=float(gy)
        meta['gt_anchor_a_x']=meta[f'gt_{a}_x']; meta['gt_anchor_a_y']=meta[f'gt_{a}_y']; meta['gt_anchor_b_x']=meta[f'gt_{b}_x']; meta['gt_anchor_b_y']=meta[f'gt_{b}_y']
        for col in ['yolo_target_error_px','gated_error_px','top1_error_px','suspicious_case','good_paper_example']:
            if col in row.index: meta[col]=safe_float(row.get(col),np.nan)
        return {'image':image_to_tensor(crop),'target_id':torch.tensor(tid,dtype=torch.long),'anchor_xy':torch.tensor(anchor_xy),'anchor_mask':torch.tensor(anchor_mask),'region_xy':torch.tensor(region_xy),'region_mask':torch.tensor(region_mask),'heatmaps':torch.tensor(np.concatenate([ah,rh],0)),'heatmap_mask':torch.tensor(np.concatenate([anchor_mask,region_mask],0).astype(np.float32)),'meta':meta}
