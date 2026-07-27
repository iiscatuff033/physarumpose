import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.common import *

def run_epoch(model, loader, opt, device, args, epoch, train=True):
    model.train() if train else model.eval()
    totals={}; count=0; lam=schedule_lambda(epoch,args.gt_anchor_warmup_epochs,args.gt_anchor_ramp_end_epoch)
    for batch in tqdm(loader, leave=False):
        for k in ["image","target_id","xy","heatmaps"]: batch[k]=batch[k].to(device)
        with torch.set_grad_enabled(train):
            out=model(batch["image"], batch["target_id"], gt_xy=batch["xy"], gt_anchor_mix_lambda=lam)
            loss,parts=loss_fn(out,batch,args,epoch)
            if train:
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        bs=batch["image"].shape[0]; count+=bs
        for k,v in parts.items(): totals[k]=totals.get(k,0)+v*bs
    return {k:v/max(count,1) for k,v in totals.items()}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--crop_csv",required=True); ap.add_argument("--annotation_csv",required=True); ap.add_argument("--image_dir",required=True)
    ap.add_argument("--out_dir",default="outputs/stage6c_gt_anchor_scheduled_physarum_pose")
    ap.add_argument("--epochs",type=int,default=140); ap.add_argument("--batch_size",type=int,default=16); ap.add_argument("--input_size",type=int,default=256)
    ap.add_argument("--heatmap_size",type=int,default=64); ap.add_argument("--lr",type=float,default=1e-3); ap.add_argument("--val_ratio",type=float,default=0.2)
    ap.add_argument("--seed",type=int,default=42); ap.add_argument("--cpu",action="store_true"); ap.add_argument("--num_workers",type=int,default=0); ap.add_argument("--max_samples",type=int,default=-1)
    ap.add_argument("--gt_anchor_warmup_epochs",type=int,default=35); ap.add_argument("--gt_anchor_ramp_end_epoch",type=int,default=95)
    ap.add_argument("--w_heatmap",type=float,default=1.0); ap.add_argument("--anchor_heatmap_weight",type=float,default=10.0); ap.add_argument("--region_heatmap_weight",type=float,default=2.0)
    ap.add_argument("--w_anchor_coord",type=float,default=20.0); ap.add_argument("--w_region_coord",type=float,default=3.0); ap.add_argument("--w_soft_joint",type=float,default=10.0)
    ap.add_argument("--w_candidate_kl",type=float,default=0.60); ap.add_argument("--w_anatomy",type=float,default=1.0); ap.add_argument("--w_anchor_separation",type=float,default=1.0); ap.add_argument("--w_entropy",type=float,default=0.0005)
    ap.add_argument("--candidate_sigma",type=float,default=0.05); ap.add_argument("--min_anchor_dist_norm",type=float,default=0.08)
    ap.add_argument("--select_anchor_weight",type=float,default=2.0); ap.add_argument("--select_soft_weight",type=float,default=1.0)
    args=ap.parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    df=filter_valid_gt_inside_crop(merge_crop_and_annotations(args.crop_csv,args.annotation_csv), margin=0.05)
    if args.max_samples>0: df=df.head(args.max_samples).reset_index(drop=True)
    df.to_csv(out_dir/"merged_valid_dataset.csv",index=False)
    tr,va=split_by_image(df,args.val_ratio,args.seed)
    tr.to_csv(out_dir/"train_split.csv",index=False); va.to_csv(out_dir/"val_split.csv",index=False)
    train_ds=PoseCropDataset(tr,args.image_dir,args.input_size,args.heatmap_size,augment=True)
    val_ds=PoseCropDataset(va,args.image_dir,args.input_size,args.heatmap_size,augment=False)
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,collate_fn=collate,pin_memory=not args.cpu)
    val_loader=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,collate_fn=collate,pin_memory=not args.cpu)
    device="cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:",device); print("Valid samples:",len(df)); print("Train samples:",len(train_ds)); print("Val samples:",len(val_ds))
    model=GTAnchorScheduledModel().to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    save_json(vars(args),out_dir/"config.json")
    best=1e9; hist=[]
    for epoch in range(1,args.epochs+1):
        lam=schedule_lambda(epoch,args.gt_anchor_warmup_epochs,args.gt_anchor_ramp_end_epoch)
        train=run_epoch(model,train_loader,opt,device,args,epoch,True)
        val=run_epoch(model,val_loader,opt,device,args,epoch,False)
        metric=args.select_soft_weight*val["loss_soft_joint"] + args.select_anchor_weight*val["loss_anchor_coord"]
        phase="gt_anchor_warmup" if lam==0 else ("ramp_to_predicted" if lam<1 else "predicted_anchor_only")
        row={"epoch":epoch,"phase":phase,"gt_anchor_lambda":lam,"selection_metric":metric}
        row.update({f"train_{k}":v for k,v in train.items()}); row.update({f"val_{k}":v for k,v in val.items()}); hist.append(row)
        pd.DataFrame(hist).to_csv(out_dir/"training_history.csv",index=False)
        print(f"Epoch {epoch:03d} [{phase} λ={lam:.3f}] train={train['loss_total']:.6f} val={val['loss_total']:.6f} val_anchor={val['loss_anchor_coord']:.6f} val_soft={val['loss_soft_joint']:.6f} select={metric:.6f}")
        if metric<best:
            best=metric; torch.save(model.state_dict(),out_dir/"best_model.pt")
            save_json({"best_epoch":epoch,"best_selection_metric":best,"best_phase":phase,"best_gt_anchor_lambda":lam},out_dir/"best_model_info.json")
    print("Training finished. Best selection metric:",best)

if __name__=="__main__": main()
