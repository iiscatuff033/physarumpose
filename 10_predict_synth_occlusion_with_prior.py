import argparse
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
from src.full_context_model import FullSkeletonPriorMLP
from src.utils_full import build_feature_from_row, load_json
class SynthDataset(Dataset):
    def __init__(self,df,num_joints=16,num_targets=4): self.df=df.reset_index(drop=True); self.nj=num_joints; self.nt=num_targets
    def __len__(self): return len(self.df)
    def __getitem__(self,idx):
        row=self.df.iloc[idx]; x=build_feature_from_row(row,self.nj,self.nt)
        y=np.array([row['target_x'],row['target_y']],dtype=np.float32)
        return torch.tensor(x,dtype=torch.float32), torch.tensor(y,dtype=torch.float32), idx
def denorm(nx,ny,cx,cy,s): return float(nx*s+cx), float(ny*s+cy)
def predict(model,df,nj,nt,device):
    ds=SynthDataset(df,nj,nt); dl=DataLoader(ds,batch_size=512,shuffle=False); preds=np.zeros((len(df),2),dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for x,y,idx in dl:
            preds[idx.numpy()]=model(x.to(device)).cpu().numpy()
    out=df.copy(); out['pred_x']=preds[:,0]; out['pred_y']=preds[:,1]
    px=[]; py=[]
    for _,r in out.iterrows():
        x,y=denorm(r['pred_x'],r['pred_y'],r['center_x'],r['center_y'],r['scale']); px.append(x); py.append(y)
    out['pred_orig_x']=px; out['pred_orig_y']=py
    out['pixel_error']=np.sqrt((out['pred_orig_x']-out['target_orig_x'])**2+(out['pred_orig_y']-out['target_orig_y'])**2)
    out['normalized_error']=np.sqrt((out['pred_x']-out['target_x'])**2+(out['pred_y']-out['target_y'])**2)
    return out
def draw_pred(image_path,row,out_path):
    bgr=cv2.imread(str(image_path));
    if bgr is None: return
    pil=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)); draw=ImageDraw.Draw(pil)
    draw.rectangle([int(row['occ_x1']),int(row['occ_y1']),int(row['occ_x2']),int(row['occ_y2'])], outline=(255,0,0), width=3)
    r=7; tx=int(round(row['target_orig_x'])); ty=int(round(row['target_orig_y']))
    draw.line([tx-r,ty-r,tx+r,ty+r], fill=(255,165,0), width=4); draw.line([tx-r,ty+r,tx+r,ty-r], fill=(255,165,0), width=4)
    px=int(round(row['pred_orig_x'])); py=int(round(row['pred_orig_y'])); tri=[(px,py-r),(px-r,py+r),(px+r,py+r)]
    draw.polygon(tri, outline=(0,255,0), fill=(0,255,0))
    text=f"{row['target_name']} | err={row['pixel_error']:.1f}px"; draw.rectangle([8,8,390,36], fill=(0,0,0)); draw.text((12,12),text,fill=(255,255,255))
    out_path.parent.mkdir(parents=True,exist_ok=True); pil.save(out_path)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--csv_path',required=True); ap.add_argument('--image_dir',required=True); ap.add_argument('--model_dir',required=True)
    ap.add_argument('--out_dir',default='outputs/stage2_synth_prior_predictions'); ap.add_argument('--max_visualizations',type=int,default=100); ap.add_argument('--cpu',action='store_true')
    args=ap.parse_args(); out_dir=Path(args.out_dir); vis_dir=out_dir/'visualizations'; out_dir.mkdir(parents=True,exist_ok=True); vis_dir.mkdir(parents=True,exist_ok=True)
    cfg=load_json(Path(args.model_dir)/'model_config.json'); nj=int(cfg['num_joints']); nt=int(cfg['num_targets']); hd=int(cfg['hidden_dim']); do=float(cfg['dropout'])
    device='cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu'); print('Using device:',device)
    model=FullSkeletonPriorMLP(nj,nt,hd,do).to(device); model.load_state_dict(torch.load(Path(args.model_dir)/'best_model.pt',map_location=device))
    df=pd.read_csv(args.csv_path); res=predict(model,df,nj,nt,device)
    res.to_csv(out_dir/'predictions_synth_occlusion.csv',index=False)
    summary=res.groupby('target_name').agg(count=('pixel_error','count'),mean_pixel_error=('pixel_error','mean'),median_pixel_error=('pixel_error','median'),std_pixel_error=('pixel_error','std'),mean_normalized_error=('normalized_error','mean'),median_normalized_error=('normalized_error','median')).reset_index()
    summary.to_csv(out_dir/'summary_by_target.csv',index=False); print(summary)
    sample=res.sample(n=min(args.max_visualizations,len(res)),random_state=10)
    image_dir=Path(args.image_dir)
    for _,row in sample.iterrows(): draw_pred(image_dir/row['occluded_image_name'],row,vis_dir/row['occluded_image_name'])
    print('Saved outputs to:',out_dir)
if __name__=='__main__': main()
