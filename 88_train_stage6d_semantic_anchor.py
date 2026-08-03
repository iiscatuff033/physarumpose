import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.common import *
from src.dataset import SemanticAnchorDataset
from src.model import SemanticAnchorPhysarumPose
from src.losses import total_loss,get_weights,schedule_lambda,phase_name

def collate(batch):
    out={}
    for k in ['image','target_id','anchor_xy','anchor_mask','region_xy','region_mask','heatmaps','heatmap_mask']: out[k]=torch.stack([b[k] for b in batch],0)
    out['meta']=[b['meta'] for b in batch]; return out
def move(batch,device): return {k:(v.to(device) if k!='meta' else v) for k,v in batch.items()}
def build_pair_indices(target_id,device): return torch.tensor([TARGET_PAIR_INDEX_BY_ID[int(t.item())] for t in target_id],dtype=torch.long,device=device)
def selected_gt_anchor_pair(anchor_xy,pair):
    B=anchor_xy.shape[0]; idx=torch.arange(B,device=anchor_xy.device); return torch.stack([anchor_xy[idx,pair[:,0]],anchor_xy[idx,pair[:,1]]],1)
def selected_target_xy(region_xy,target_id):
    B=region_xy.shape[0]; idx=torch.arange(B,device=region_xy.device); return region_xy[idx,target_id]

def run_epoch(model,loader,opt,device,args,epoch,train=True):
    model.train() if train else model.eval(); lam=schedule_lambda(epoch,args.anchor_pretrain_epochs,args.gt_anchor_warmup_epochs,args.gt_anchor_ramp_end_epoch); weights=get_weights(args,epoch); totals={}; count=0
    for batch in tqdm(loader,leave=False):
        batch=move(batch,device); pair=build_pair_indices(batch['target_id'],device); gt_pair=selected_gt_anchor_pair(batch['anchor_xy'],pair); target_xy=selected_target_xy(batch['region_xy'],batch['target_id'])
        with torch.set_grad_enabled(train):
            out=model(batch['image'],batch['target_id'],pair_indices=pair,gt_anchor_pair=gt_pair,gt_anchor_lambda=lam); loss,parts=total_loss(out,batch,pair,gt_pair,target_xy,weights)
            if train: opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        bs=batch['image'].shape[0]; count+=bs
        for k,v in parts.items(): totals[k]=totals.get(k,0)+v*bs
    return {k:v/max(count,1) for k,v in totals.items()},lam
def set_seed(s): np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def main():
    p=argparse.ArgumentParser(); p.add_argument('--crop_csv',required=True); p.add_argument('--annotation_csv',required=True); p.add_argument('--image_dir',required=True); p.add_argument('--out_dir',default='outputs/stage6d_semantic_anchor_physarum'); p.add_argument('--epochs',type=int,default=320); p.add_argument('--batch_size',type=int,default=16); p.add_argument('--input_size',type=int,default=256); p.add_argument('--heatmap_size',type=int,default=64); p.add_argument('--lr',type=float,default=7e-4); p.add_argument('--val_ratio',type=float,default=.2); p.add_argument('--seed',type=int,default=42); p.add_argument('--cpu',action='store_true'); p.add_argument('--num_workers',type=int,default=0); p.add_argument('--max_samples',type=int,default=-1); p.add_argument('--anchor_pretrain_epochs',type=int,default=90); p.add_argument('--gt_anchor_warmup_epochs',type=int,default=130); p.add_argument('--gt_anchor_ramp_end_epoch',type=int,default=210); p.add_argument('--save_best_after_epoch',type=int,default=210); p.add_argument('--w_heatmap',type=float,default=1.0); p.add_argument('--anchor_heatmap_weight',type=float,default=18.0); p.add_argument('--region_heatmap_weight',type=float,default=1.5); p.add_argument('--w_all_anchor_coord',type=float,default=10.0); p.add_argument('--w_selected_anchor_coord',type=float,default=34.0); p.add_argument('--w_region_coord',type=float,default=2.0); p.add_argument('--w_soft_joint',type=float,default=8.0); p.add_argument('--w_candidate_kl',type=float,default=.7); p.add_argument('--w_anatomy',type=float,default=1.4); p.add_argument('--w_anchor_separation',type=float,default=1.0); p.add_argument('--w_entropy',type=float,default=.0005); p.add_argument('--candidate_sigma',type=float,default=.05); p.add_argument('--min_anchor_dist_norm',type=float,default=.08); p.add_argument('--select_anchor_weight',type=float,default=4.0); p.add_argument('--select_soft_weight',type=float,default=1.0); args=p.parse_args(); set_seed(args.seed)
    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True,exist_ok=True); df=merge_crop_and_annotations(args.crop_csv,args.annotation_csv); df=filter_valid_relevant_pair_inside_crop(df,.05)
    if args.max_samples>0: df=df.head(args.max_samples).reset_index(drop=True)
    df.to_csv(out_dir/'merged_valid_dataset.csv',index=False); trdf,vadf=split_by_image(df,args.val_ratio,args.seed); trdf.to_csv(out_dir/'train_split.csv',index=False); vadf.to_csv(out_dir/'val_split.csv',index=False)
    trds=SemanticAnchorDataset(trdf,args.image_dir,args.input_size,args.heatmap_size,augment=True); vads=SemanticAnchorDataset(vadf,args.image_dir,args.input_size,args.heatmap_size,augment=False); trlo=DataLoader(trds,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,collate_fn=collate,pin_memory=not args.cpu); valo=DataLoader(vads,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,collate_fn=collate,pin_memory=not args.cpu)
    device='cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu'); print('Using device:',device,'Valid:',len(df),'Train:',len(trds),'Val:',len(vads)); model=SemanticAnchorPhysarumPose().to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4); save_json(vars(args),out_dir/'config.json')
    best=float('inf'); hist=[]; saved=False
    for epoch in range(1,args.epochs+1):
        tr,lam=run_epoch(model,trlo,opt,device,args,epoch,True); va,_=run_epoch(model,valo,opt,device,args,epoch,False); phase=phase_name(epoch,args.anchor_pretrain_epochs,args.gt_anchor_warmup_epochs,args.gt_anchor_ramp_end_epoch); select=args.select_soft_weight*va['loss_soft_joint']+args.select_anchor_weight*va['loss_selected_anchor_coord']; row={'epoch':epoch,'phase':phase,'gt_anchor_lambda':lam,'selection_metric':select}; row.update({f'train_{k}':v for k,v in tr.items()}); row.update({f'val_{k}':v for k,v in va.items()}); hist.append(row); pd.DataFrame(hist).to_csv(out_dir/'training_history.csv',index=False); print(f"Epoch {epoch:03d} [{phase} λ={lam:.3f}] val_sel_anchor={va['loss_selected_anchor_coord']:.6f} val_soft={va['loss_soft_joint']:.6f} select={select:.6f}")
        if epoch>=args.save_best_after_epoch and select<best: best=select; saved=True; torch.save(model.state_dict(),out_dir/'best_model.pt'); save_json({'best_epoch':epoch,'best_selection_metric':best,'best_phase':phase,'best_gt_anchor_lambda':lam,'note':'Stage 6D best saved only after predicted semantic-anchor phase begins.'},out_dir/'best_model_info.json')
    if not saved: torch.save(model.state_dict(),out_dir/'best_model.pt'); save_json({'best_epoch':args.epochs,'warning':'saved final model'},out_dir/'best_model_info.json')
if __name__=='__main__': main()
