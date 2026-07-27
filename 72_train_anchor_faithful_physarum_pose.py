import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.stage6b_common import merge_crop_and_annotations, split_by_image, filter_valid_gt_inside_crop, save_json
from src.anchor_faithful_dataset import AnchorFaithfulDataset
from src.anchor_faithful_model import AnchorFaithfulPhysarumPoseModel
from src.anchor_faithful_losses import anchor_faithful_loss, get_curriculum_weights

def collate_fn(batch): return {'image':torch.stack([b['image'] for b in batch]),'target_id':torch.stack([b['target_id'] for b in batch]),'xy':torch.stack([b['xy'] for b in batch]),'heatmaps':torch.stack([b['heatmaps'] for b in batch]),'meta':[b['meta'] for b in batch]}
def move_batch(batch,device): return {k:(v.to(device) if k!='meta' else v) for k,v in batch.items()}
def run_epoch(model,loader,opt,device,weights,train=True):
    model.train() if train else model.eval(); totals={}; count=0
    for batch in tqdm(loader,leave=False):
        batch=move_batch(batch,device)
        with torch.set_grad_enabled(train):
            out=model(batch['image'],batch['target_id']); loss,parts=anchor_faithful_loss(out,batch,weights)
            if train:
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        bs=batch['image'].shape[0]; count+=bs
        for k,v in parts.items(): totals[k]=totals.get(k,0.0)+v*bs
    return {k:v/max(count,1) for k,v in totals.items()}
def set_seed(seed): np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
def main():
    p=argparse.ArgumentParser(); p.add_argument('--crop_csv',required=True); p.add_argument('--annotation_csv',required=True); p.add_argument('--image_dir',required=True); p.add_argument('--out_dir',default='outputs/stage6b_anchor_faithful_physarum_pose'); p.add_argument('--epochs',type=int,default=120); p.add_argument('--batch_size',type=int,default=16); p.add_argument('--input_size',type=int,default=256); p.add_argument('--heatmap_size',type=int,default=64); p.add_argument('--lr',type=float,default=1e-3); p.add_argument('--val_ratio',type=float,default=0.20); p.add_argument('--seed',type=int,default=42); p.add_argument('--cpu',action='store_true'); p.add_argument('--num_workers',type=int,default=0); p.add_argument('--max_samples',type=int,default=-1); p.add_argument('--anchor_warmup_epochs',type=int,default=35)
    p.add_argument('--w_heatmap',type=float,default=1.0); p.add_argument('--anchor_heatmap_weight',type=float,default=10.0); p.add_argument('--region_heatmap_weight',type=float,default=2.0); p.add_argument('--w_anchor_coord',type=float,default=18.0); p.add_argument('--w_region_coord',type=float,default=3.0); p.add_argument('--w_all_coord',type=float,default=1.0); p.add_argument('--w_soft_joint',type=float,default=10.0); p.add_argument('--w_candidate_kl',type=float,default=0.60); p.add_argument('--w_anatomy',type=float,default=1.0); p.add_argument('--w_anchor_separation',type=float,default=1.0); p.add_argument('--w_entropy',type=float,default=0.0005); p.add_argument('--candidate_sigma',type=float,default=0.05); p.add_argument('--min_anchor_dist_norm',type=float,default=0.08)
    args=p.parse_args(); set_seed(args.seed); out_dir=Path(args.out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    df=filter_valid_gt_inside_crop(merge_crop_and_annotations(args.crop_csv,args.annotation_csv),margin=0.05)
    if args.max_samples and args.max_samples>0: df=df.head(args.max_samples).reset_index(drop=True)
    df.to_csv(out_dir/'merged_valid_dataset.csv',index=False); train_df,val_df=split_by_image(df,args.val_ratio,args.seed); train_df.to_csv(out_dir/'train_split.csv',index=False); val_df.to_csv(out_dir/'val_split.csv',index=False)
    train_ds=AnchorFaithfulDataset(train_df,args.image_dir,args.input_size,args.heatmap_size,augment=True); val_ds=AnchorFaithfulDataset(val_df,args.image_dir,args.input_size,args.heatmap_size,augment=False)
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,collate_fn=collate_fn,pin_memory=not args.cpu); val_loader=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,collate_fn=collate_fn,pin_memory=not args.cpu)
    device='cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu'); print('Using device:',device); print('Valid samples:',len(df)); print('Train samples:',len(train_ds)); print('Val samples:',len(val_ds))
    model=AnchorFaithfulPhysarumPoseModel().to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4); save_json(vars(args),out_dir/'config.json')
    best=float('inf'); hist=[]
    for epoch in range(1,args.epochs+1):
        weights=get_curriculum_weights(args,epoch); train=run_epoch(model,train_loader,opt,device,weights,True); val=run_epoch(model,val_loader,opt,device,weights,False)
        row={'epoch':epoch,'curriculum_phase':'anchor_warmup' if epoch<=args.anchor_warmup_epochs else 'slime_training'}; row.update({f'train_{k}':v for k,v in train.items()}); row.update({f'val_{k}':v for k,v in val.items()}); hist.append(row); pd.DataFrame(hist).to_csv(out_dir/'training_history.csv',index=False)
        print(f"Epoch {epoch:03d} [{row['curriculum_phase']}] | train_total={train['loss_total']:.6f} | val_total={val['loss_total']:.6f} | val_anchor={val['loss_anchor_coord']:.6f} | val_soft={val['loss_soft_joint']:.6f}")
        if val['loss_total']<best:
            best=val['loss_total']; torch.save(model.state_dict(),out_dir/'best_model.pt'); save_json({'best_epoch':epoch,'best_val_loss':best},out_dir/'best_model_info.json')
    print('Training finished. Best val:',best)
if __name__=='__main__': main()
