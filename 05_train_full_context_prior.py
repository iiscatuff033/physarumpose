import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src.full_context_model import FullSkeletonPriorMLP
from src.utils_full import TARGET_JOINTS, build_feature_from_row, build_target_from_row, save_json, set_seed

class FullContextDataset(Dataset):
    def __init__(self, df, num_joints=16, num_targets=4):
        self.df = df.reset_index(drop=True)
        self.num_joints = num_joints
        self.num_targets = num_targets
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = build_feature_from_row(row, self.num_joints, self.num_targets)
        y = build_target_from_row(row)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def split_by_image(df, val_ratio=0.15, seed=42):
    rng = np.random.default_rng(seed)
    names = df["image_name"].dropna().unique()
    rng.shuffle(names)
    n_val = max(1, int(len(names) * val_ratio))
    val_names = set(names[:n_val])
    val = df[df["image_name"].isin(val_names)].reset_index(drop=True)
    train = df[~df["image_name"].isin(val_names)].reset_index(drop=True)
    return train, val

def metrics(pred, target):
    err = torch.linalg.norm(pred-target, dim=1)
    return {"mean_l2": float(err.mean()), "median_l2": float(err.median())}

def run_epoch(model, loader, opt, device, train=True):
    model.train() if train else model.eval()
    loss_fn = torch.nn.SmoothL1Loss()
    total, ps, ys = 0.0, [], []
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            pred = model(x)
            loss = loss_fn(pred, y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
        total += loss.item() * x.size(0)
        ps.append(pred.detach().cpu())
        ys.append(y.detach().cpu())
    ps, ys = torch.cat(ps), torch.cat(ys)
    m = metrics(ps, ys)
    m["loss"] = total / len(loader.dataset)
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", required=True)
    ap.add_argument("--out_dir", default="outputs/full_context_prior_mpii")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv_path)
    if len(df) == 0: raise ValueError("CSV has 0 rows. Try lower --min_visible.")

    train_df, val_df = split_by_image(df, args.val_ratio, args.seed)
    print(f"Total samples: {len(df)}")
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    print(train_df["target_name"].value_counts())

    num_joints, num_targets = 16, len(TARGET_JOINTS)
    train_loader = DataLoader(FullContextDataset(train_df, num_joints, num_targets), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(FullContextDataset(val_df, num_joints, num_targets), batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = FullSkeletonPriorMLP(num_joints, num_targets, args.hidden_dim, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best = float("inf")
    hist = []
    for ep in range(1, args.epochs+1):
        tr = run_epoch(model, train_loader, opt, device, True)
        va = run_epoch(model, val_loader, opt, device, False)
        row = {"epoch": ep, "train_loss": tr["loss"], "train_mean_l2": tr["mean_l2"], "val_loss": va["loss"], "val_mean_l2": va["mean_l2"], "val_median_l2": va["median_l2"]}
        hist.append(row)
        print(f"Epoch {ep:03d} | train_loss={row['train_loss']:.5f} | train_l2={row['train_mean_l2']:.5f} | val_loss={row['val_loss']:.5f} | val_l2={row['val_mean_l2']:.5f}")
        if va["mean_l2"] < best:
            best = va["mean_l2"]
            torch.save(model.state_dict(), out/"best_model.pt")
            save_json({"num_joints": num_joints, "num_targets": num_targets, "target_joints": TARGET_JOINTS, "hidden_dim": args.hidden_dim, "dropout": args.dropout, "best_val_mean_l2": best}, out/"model_config.json")

    pd.DataFrame(hist).to_csv(out/"training_history.csv", index=False)
    train_df.to_csv(out/"train_split.csv", index=False)
    val_df.to_csv(out/"val_split.csv", index=False)
    print(f"Done. Best val L2: {best:.5f}")
    print(f"Saved: {out/'best_model.pt'}")

if __name__ == "__main__":
    main()
