import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.model import SkeletonPriorMLP
from src.utils import build_feature, build_target, load_json


class EvalDataset(Dataset):
    def __init__(self, df, num_tasks):
        self.df = df.reset_index(drop=True)
        self.num_tasks = num_tasks

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = build_feature(row, self.num_tasks)
        y = build_target(row)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), idx


def evaluate(model, df, num_tasks, device):
    ds = EvalDataset(df, num_tasks=num_tasks)
    loader = DataLoader(ds, batch_size=512, shuffle=False)

    preds = np.zeros((len(df), 2), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for x, _y, idx in loader:
            x = x.to(device)
            pred = model(x).cpu().numpy()
            preds[idx.numpy()] = pred

    targets = df[["target_x", "target_y"]].values.astype(np.float32)
    errors = np.linalg.norm(preds - targets, axis=1)

    out_df = df.copy()
    out_df["pred_x"] = preds[:, 0]
    out_df["pred_y"] = preds[:, 1]
    out_df["error_l2"] = errors

    return out_df


def plot_examples(result_df, out_path, max_examples=24):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sample_df = result_df.sample(n=min(max_examples, len(result_df)), random_state=7).reset_index(drop=True)

    cols = 4
    rows = int(np.ceil(len(sample_df) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if rows == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for ax_idx, ax in enumerate(axes):
        ax.axis("equal")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)

        if ax_idx >= len(sample_df):
            ax.axis("off")
            continue

        row = sample_df.iloc[ax_idx]

        x1, y1 = row["x1"], row["y1"]
        x2, y2 = row["x2"], row["y2"]
        tx, ty = row["target_x"], row["target_y"]
        px, py = row["pred_x"], row["pred_y"]

        ax.plot([x1, x2], [y1, y2], marker="o", label="endpoints")
        ax.scatter([tx], [ty], marker="x", s=80, label="true")
        ax.scatter([px], [py], marker="^", s=80, label="pred")

        ax.set_title(f"{row['task_name']} | err={row['error_l2']:.3f}")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/eval_plots")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_json(model_dir / "model_config.json")
    num_tasks = int(config["num_tasks"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SkeletonPriorMLP(
        num_tasks=num_tasks,
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))

    df = pd.read_csv(args.csv_path)
    result_df = evaluate(model, df, num_tasks, device)

    result_csv = out_dir / "predictions.csv"
    result_df.to_csv(result_csv, index=False)

    print("\nOverall error:")
    print(result_df["error_l2"].describe())

    print("\nError by task:")
    print(result_df.groupby("task_name")["error_l2"].agg(["count", "mean", "median", "std"]))

    plot_examples(result_df, out_dir / "prediction_examples.png")

    print(f"\nSaved predictions: {result_csv}")
    print(f"Saved plot: {out_dir / 'prediction_examples.png'}")


if __name__ == "__main__":
    main()
