import argparse, numpy as np, pandas as pd, torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.common import *
from src.dataset import SemanticOffsetDataset
from src.model import Stage6ESingleModel
from src.losses import total_loss,get_weights,schedule_lambda,phase_name,selected_target_xy

def collate(batch):
    out={}
    for k in ['image','target_id','anchor_xy','anchor_mask','region_xy','region_mask','heatmaps','heatmap_mask']: out[k]=torch.stack([b[k] for b in batch],0)
    out['meta']=[b['meta'] for b in batch]; return out
def move(batch,device): return {k:(v.to(device) if k!='meta' else v) for k,v in batch.items()}
def pair_idx(tid,device): return torch.tensor([TARGET_PAIR_INDEX_BY_ID[int(t.item())] for t in tid],dtype=torch.long,device=device)
def gt_pair(anchor_xy,pair):
    idx=torch.arange(anchor_xy.shape[0],device=anchor_xy.device); return torch.stack([anchor_xy[idx,pair[:,0]],anchor_xy[idx,pair[:,1]]],1)
def run_epoch(model,loader,opt,device,args,epoch,train=True):
    model.train() if train else model.eval(); lam=schedule_lambda(epoch,args.anchor_pretrain_epochs,args.gt_anchor_warmup_epochs,args.gt_anchor_ramp_end_epoch); w=get_weights(args,epoch); totals={}; count=0
    for batch in tqdm(loader,leave=False):
        batch=move(batch,device); pair=pair_idx(batch['target_id'],device); gp=gt_pair(batch['anchor_xy'],pair); target=selected_target_xy(batch['region_xy'],batch['target_id'])
        with torch.set_grad_enabled(train):
            out=model(batch['image'],batch['target_id'],pair,gp,lam); loss,parts=total_loss(out,batch,pair,gp,target,w)
            if train: opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        bs=batch['image'].shape[0]; count+=bs
        for k,v in parts.items(): totals[k]=totals.get(k,0)+v*bs
    return {k:v/max(count,1) for k,v in totals.items()},lam

def main():
    p=argparse.ArgumentParser(); p.add_argument('--crop_csv',required=True); p.add_argument('--annotation_csv',required=True); p.add_argument('--image_dir',required=True); p.add_argument('--out_dir',default='outputs/stage6e_single_model_semantic_offset'); p.add_argument('--epochs',type=int,default=380); p.add_argument('--batch_size',type=int,default=8); p.add_argument('--input_size',type=int,default=384); p.add_argument('--heatmap_size',type=int,default=0); p.add_argument('--lr',type=float,default=6e-4); p.add_argument('--val_ratio',type=float,default=.2); p.add_argument('--seed',type=int,default=42); p.add_argument('--cpu',action='store_true'); p.add_argument('--num_workers',type=int,default=0); p.add_argument('--max_samples',type=int,default=-1)
    p.add_argument('--anchor_pretrain_epochs',type=int,default=100); p.add_argument('--gt_anchor_warmup_epochs',type=int,default=150); p.add_argument('--gt_anchor_ramp_end_epoch',type=int,default=240); p.add_argument('--save_best_after_epoch',type=int,default=240)
    p.add_argument('--w_heatmap',type=float,default=1.0); p.add_argument('--anchor_heatmap_weight',type=float,default=20.0); p.add_argument('--region_heatmap_weight',type=float,default=1.4); p.add_argument('--w_all_anchor_coord',type=float,default=12.0); p.add_argument('--w_raw_anchor_coord',type=float,default=5.0); p.add_argument('--w_selected_anchor_coord',type=float,default=38.0); p.add_argument('--w_region_coord',type=float,default=2.0); p.add_argument('--w_soft_joint',type=float,default=8.0); p.add_argument('--w_candidate_kl',type=float,default=.8); p.add_argument('--w_anatomy',type=float,default=1.5); p.add_argument('--w_anchor_separation',type=float,default=1.0); p.add_argument('--w_offset_reg',type=float,default=.02); p.add_argument('--w_entropy',type=float,default=.0005); p.add_argument('--candidate_sigma',type=float,default=.045); p.add_argument('--min_anchor_dist_norm',type=float,default=.08); p.add_argument('--select_anchor_weight',type=float,default=4.5); p.add_argument('--select_soft_weight',type=float,default=1.0); args=p.parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    if args.heatmap_size<=0: args.heatmap_size=args.input_size//4
    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True,exist_ok=True); df=filter_valid_relevant_pair_inside_crop(merge_crop_and_annotations(args.crop_csv,args.annotation_csv),.05)
    if args.max_samples>0: df=df.head(args.max_samples).reset_index(drop=True)
    df.to_csv(out_dir/'merged_valid_dataset.csv',index=False); trdf,vadf=split_by_image(df,args.val_ratio,args.seed); trdf.to_csv(out_dir/'train_split.csv',index=False); vadf.to_csv(out_dir/'val_split.csv',index=False)
    tr=SemanticOffsetDataset(trdf,args.image_dir,args.input_size,args.heatmap_size,augment=True); va=SemanticOffsetDataset(vadf,args.image_dir,args.input_size,args.heatmap_size,augment=False)
    trl=DataLoader(tr,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,collate_fn=collate,pin_memory=not args.cpu); val=DataLoader(va,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,collate_fn=collate,pin_memory=not args.cpu)
    device='cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu'); print('Device',device,'Valid',len(df),'Train',len(tr),'Val',len(va),'input',args.input_size,'hm',args.heatmap_size)
    model=Stage6ESingleModel().to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4); save_json(vars(args),out_dir/'config.json'); best=float('inf'); hist=[]; saved=False
    for epoch in range(1,args.epochs+1):
        trp,lam=run_epoch(model,trl,opt,device,args,epoch,True); vap,_=run_epoch(model,val,opt,device,args,epoch,False); phase=phase_name(epoch,args.anchor_pretrain_epochs,args.gt_anchor_warmup_epochs,args.gt_anchor_ramp_end_epoch); select=args.select_soft_weight*vap['loss_soft_joint']+args.select_anchor_weight*vap['loss_selected_anchor_coord']
        row={'epoch':epoch,'phase':phase,'gt_anchor_lambda':lam,'selection_metric':select}; row.update({f'train_{k}':v for k,v in trp.items()}); row.update({f'val_{k}':v for k,v in vap.items()}); hist.append(row); pd.DataFrame(hist).to_csv(out_dir/'training_history.csv',index=False)
        print(f"Epoch {epoch:03d} [{phase} λ={lam:.3f}] val_anchor={vap['loss_selected_anchor_coord']:.6f} val_soft={vap['loss_soft_joint']:.6f} select={select:.6f}")
        if epoch>=args.save_best_after_epoch and select<best:
            best=select; saved=True; torch.save(model.state_dict(),out_dir/'best_model.pt'); save_json({'best_epoch':epoch,'best_selection_metric':best,'best_phase':phase,'best_gt_anchor_lambda':lam,'note':'Stage 6E best saved after predicted-anchor phase begins.'},out_dir/'best_model_info.json')
    if not saved: torch.save(model.state_dict(),out_dir/'best_model.pt'); save_json({'best_epoch':args.epochs,'warning':'No epoch met save condition; saved final.'},out_dir/'best_model_info.json')
if __name__=='__main__': main()
