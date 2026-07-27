import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from src.full_context_model import FullSkeletonPriorMLP
from src.utils_full import build_feature_from_row, load_json

class EvalDataset(Dataset):
    def __init__(self, df, num_joints=16, num_targets=4):
        self.df = df.reset_index(drop=True)
        self.num_joints = num_joints
        self.num_targets = num_targets
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = build_feature_from_row(row, self.num_joints, self.num_targets)
        return torch.tensor(x, dtype=torch.float32), idx

def evaluate(model, df, num_joints, num_targets, device):
    loader = DataLoader(EvalDataset(df, num_joints, num_targets), batch_size=512, shuffle=False)
    preds = np.zeros((len(df), 2), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for x, idx in loader:
            pred = model(x.to(device)).cpu().numpy()
            preds[idx.numpy()] = pred
    targets = df[["target_x","target_y"]].values.astype(np.float32)
    errs = np.linalg.norm(preds-targets, axis=1)
    out = df.copy()
    out["pred_x"], out["pred_y"], out["error_l2"] = preds[:,0], preds[:,1], errs
    return out

def plot_examples(df, out_path, n=24):
    if len(df) == 0: return
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    sdf = df.sample(n=min(n, len(df)), random_state=8).reset_index(drop=True)
    cols, rows = 4, int(np.ceil(len(sdf)/4))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    axes = np.array(axes).flatten()
    for i, ax in enumerate(axes):
        ax.axis("equal"); ax.invert_yaxis(); ax.grid(True, alpha=.3)
        if i >= len(sdf):
            ax.axis("off"); continue
        row = sdf.iloc[i]
        xs, ys = [], []
        for j in range(16):
            if row[f"j{j}_mask"] > 0.5:
                xs.append(row[f"j{j}_x"]); ys.append(row[f"j{j}_y"])
        ax.scatter(xs, ys, s=25, label="context")
        ax.scatter([row["target_x"]], [row["target_y"]], marker="x", s=100, label="true")
        ax.scatter([row["pred_x"]], [row["pred_y"]], marker="^", s=100, label="pred")
        ax.set_title(f"{row['target_name']} | err={row['error_l2']:.3f}")
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--out_dir", default="outputs/eval_plots_full_context")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    cfg = load_json(Path(args.model_dir)/"model_config.json")
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    model = FullSkeletonPriorMLP(cfg["num_joints"], cfg["num_targets"], cfg["hidden_dim"], cfg["dropout"]).to(device)
    model.load_state_dict(torch.load(Path(args.model_dir)/"best_model.pt", map_location=device))

    df = pd.read_csv(args.csv_path)
    res = evaluate(model, df, cfg["num_joints"], cfg["num_targets"], device)
    res.to_csv(out/"predictions_full_context.csv", index=False)
    plot_examples(res, out/"prediction_examples_full_context.png")

    print("Overall normalized L2 error:")
    print(res["error_l2"].describe())
    print("\nError by target:")
    print(res.groupby("target_name")["error_l2"].agg(["count","mean","median","std"]))
    print(f"\nSaved: {out/'predictions_full_context.csv'}")
    print(f"Saved: {out/'prediction_examples_full_context.png'}")

if __name__ == "__main__":
    main()
