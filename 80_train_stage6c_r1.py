import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.common import merge_crop_and_annotations, filter_valid_gt_inside_crop, split_by_image, save_json
from src.dataset import PoseCropDataset
from src.model import PhysarumPoseR1
from src.losses import total_loss, get_weights, schedule_lambda, phase_name

def collate(batch):
    return {k: torch.stack([b[k] for b in batch],0) for k in ["image","target_id","xy","heatmaps"]} | {"meta":[b["meta"] for b in batch]}

def move(batch,device):
    return {k:(v.to(device) if k!="meta" else v) for k,v in batch.items()}

def run_epoch(model,loader,opt,device,args,epoch,train=True):
    model.train() if train else model.eval()
    lam=schedule_lambda(epoch,args.anchor_pretrain_epochs,args.gt_anchor_warmup_epochs,args.gt_anchor_ramp_end_epoch)
    weights=get_weights(args,epoch)
    totals={}; count=0
    for batch in tqdm(loader, leave=False):
        batch=move(batch,device)
        with torch.set_grad_enabled(train):
            out=model(batch["image"],batch["target_id"],gt_xy=batch["xy"],gt_anchor_lambda=lam)
            loss,parts=total_loss(out,batch,weights)
            if train:
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        bs=batch["image"].shape[0]; count+=bs
        for k,v in parts.items(): totals[k]=totals.get(k,0.0)+v*bs
    return {k:v/max(count,1) for k,v in totals.items()}, lam

def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--crop_csv",required=True); p.add_argument("--annotation_csv",required=True); p.add_argument("--image_dir",required=True)
    p.add_argument("--out_dir",default="outputs/stage6c_r1_cvpr_anchor_curriculum")
    p.add_argument("--epochs",type=int,default=220); p.add_argument("--batch_size",type=int,default=16)
    p.add_argument("--input_size",type=int,default=256); p.add_argument("--heatmap_size",type=int,default=64)
    p.add_argument("--lr",type=float,default=1e-3); p.add_argument("--val_ratio",type=float,default=0.2)
    p.add_argument("--seed",type=int,default=42); p.add_argument("--cpu",action="store_true"); p.add_argument("--num_workers",type=int,default=0)
    p.add_argument("--max_samples",type=int,default=-1)
    p.add_argument("--anchor_pretrain_epochs",type=int,default=50); p.add_argument("--gt_anchor_warmup_epochs",type=int,default=80)
    p.add_argument("--gt_anchor_ramp_end_epoch",type=int,default=140); p.add_argument("--save_best_after_epoch",type=int,default=140)
    p.add_argument("--w_heatmap",type=float,default=1.0); p.add_argument("--anchor_heatmap_weight",type=float,default=10.0)
    p.add_argument("--region_heatmap_weight",type=float,default=2.0); p.add_argument("--w_anchor_coord",type=float,default=24.0)
    p.add_argument("--w_region_coord",type=float,default=3.0); p.add_argument("--w_all_coord",type=float,default=1.0)
    p.add_argument("--w_soft_joint",type=float,default=10.0); p.add_argument("--w_candidate_kl",type=float,default=0.6)
    p.add_argument("--w_anatomy",type=float,default=1.0); p.add_argument("--w_anchor_separation",type=float,default=1.0)
    p.add_argument("--w_entropy",type=float,default=0.0005); p.add_argument("--candidate_sigma",type=float,default=0.05)
    p.add_argument("--min_anchor_dist_norm",type=float,default=0.08)
    p.add_argument("--select_anchor_weight",type=float,default=2.0); p.add_argument("--select_soft_weight",type=float,default=1.0)
    args=p.parse_args(); set_seed(args.seed)
    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    df=merge_crop_and_annotations(args.crop_csv,args.annotation_csv)
    df=filter_valid_gt_inside_crop(df,margin=0.05)
    if args.max_samples>0: df=df.head(args.max_samples).reset_index(drop=True)
    df.to_csv(out_dir/"merged_valid_dataset.csv",index=False)
    train_df,val_df=split_by_image(df,args.val_ratio,args.seed)
    train_df.to_csv(out_dir/"train_split.csv",index=False); val_df.to_csv(out_dir/"val_split.csv",index=False)
    train_ds=PoseCropDataset(train_df,args.image_dir,args.input_size,args.heatmap_size,augment=True)
    val_ds=PoseCropDataset(val_df,args.image_dir,args.input_size,args.heatmap_size,augment=False)
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,collate_fn=collate,pin_memory=not args.cpu)
    val_loader=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,collate_fn=collate,pin_memory=not args.cpu)
    device="cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:",device); print("Valid samples:",len(df)); print("Train:",len(train_ds),"Val:",len(val_ds))
    model=PhysarumPoseR1().to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    save_json(vars(args),out_dir/"config.json")
    best=float("inf"); hist=[]; saved_any=False
    for epoch in range(1,args.epochs+1):
        tr,lam=run_epoch(model,train_loader,opt,device,args,epoch,True)
        va,_=run_epoch(model,val_loader,opt,device,args,epoch,False)
        phase=phase_name(epoch,args.anchor_pretrain_epochs,args.gt_anchor_warmup_epochs,args.gt_anchor_ramp_end_epoch)
        select=args.select_soft_weight*va["loss_soft_joint"] + args.select_anchor_weight*va["loss_anchor_coord"]
        row={"epoch":epoch,"phase":phase,"gt_anchor_lambda":lam,"selection_metric":select}
        row.update({f"train_{k}":v for k,v in tr.items()}); row.update({f"val_{k}":v for k,v in va.items()}); hist.append(row)
        pd.DataFrame(hist).to_csv(out_dir/"training_history.csv",index=False)
        print(f"Epoch {epoch:03d} [{phase} λ={lam:.3f}] val_anchor={va['loss_anchor_coord']:.6f} val_soft={va['loss_soft_joint']:.6f} select={select:.6f}")
        if epoch >= args.save_best_after_epoch and select < best:
            best=select; saved_any=True; torch.save(model.state_dict(),out_dir/"best_model.pt")
            save_json({"best_epoch":epoch,"best_selection_metric":best,"best_phase":phase,"best_gt_anchor_lambda":lam,
                       "note":"best saved only after save_best_after_epoch"}, out_dir/"best_model_info.json")
    if not saved_any:
        torch.save(model.state_dict(),out_dir/"best_model.pt")
        save_json({"best_epoch":args.epochs,"warning":"No epoch met save condition; saved final model."}, out_dir/"best_model_info.json")
    print("Training finished. Saved:",out_dir/"best_model.pt")
if __name__=="__main__": main()
