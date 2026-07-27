import argparse
from pathlib import Path
import numpy as np, pandas as pd
ENDPOINTS={'right_elbow':(12,10),'left_elbow':(13,15),'right_knee':(2,0),'left_knee':(3,5)}
def midpoint(row):
    a,b=ENDPOINTS[row['target_name']]; ax,ay=row[f'j{a}_orig_x'],row[f'j{a}_orig_y']; bx,by=row[f'j{b}_orig_x'],row[f'j{b}_orig_y']
    if (ax==0 and ay==0) or (bx==0 and by==0): return np.nan,np.nan
    return (ax+bx)/2,(ay+by)/2
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--pred_csv',required=True); ap.add_argument('--out_dir',required=True); args=ap.parse_args()
    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True,exist_ok=True); df=pd.read_csv(args.pred_csv)
    xs=[]; ys=[]
    for _,r in df.iterrows():
        x,y=midpoint(r); xs.append(x); ys.append(y)
    df['geometry_pred_x']=xs; df['geometry_pred_y']=ys
    df['geometry_pixel_error']=np.sqrt((df['geometry_pred_x']-df['target_orig_x'])**2+(df['geometry_pred_y']-df['target_orig_y'])**2)
    df['prior_better_than_geometry']=df['pixel_error']<df['geometry_pixel_error']
    df.to_csv(out_dir/'predictions_with_geometry_baseline.csv',index=False)
    summary=df.groupby('target_name').agg(count=('pixel_error','count'),prior_mean_px=('pixel_error','mean'),geometry_mean_px=('geometry_pixel_error','mean'),prior_median_px=('pixel_error','median'),geometry_median_px=('geometry_pixel_error','median'),prior_win_rate=('prior_better_than_geometry','mean')).reset_index()
    summary.to_csv(out_dir/'geometry_vs_prior_summary.csv',index=False); print(summary)
if __name__=='__main__': main()
