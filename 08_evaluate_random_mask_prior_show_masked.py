import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.full_context_model import FullSkeletonPriorMLP
from src.utils_full import load_json


def make_one_hot(index: int, size: int):
    arr = np.zeros(size, dtype=np.float32)
    arr[index] = 1.0
    return arr


class RandomMaskEvalDataset(Dataset):
    """
    Evaluation dataset.

    It returns both:
        1. model input after random masking
        2. the exact visible context points used by the model

    This makes the prediction images easier to understand.
    """

    def __init__(self, df, num_joints=16, num_targets=4, mask_prob=0.0, min_keep=5, seed=123):
        self.df = df.reset_index(drop=True)
        self.num_joints = num_joints
        self.num_targets = num_targets
        self.mask_prob = float(mask_prob)
        self.min_keep = int(min_keep)
        self.seed = seed

    def __len__(self):
        return len(self.df)

    def _build_feature(self, row, idx):
        # deterministic random mask per sample and mask probability
        rng = np.random.default_rng(self.seed + idx + int(self.mask_prob * 1000))

        target_task_id = int(row["target_task_id"])
        joint_values = []
        visible_ids = []

        for j in range(self.num_joints):
            x = float(row[f"j{j}_x"])
            y = float(row[f"j{j}_y"])
            m = float(row[f"j{j}_mask"])
            if m > 0.5:
                visible_ids.append(j)
            joint_values.append([x, y, m])

        if self.mask_prob > 0:
            current_visible = list(visible_ids)
            rng.shuffle(current_visible)
            for j in current_visible:
                current_count = sum(1 for item in joint_values if item[2] > 0.5)
                if current_count <= self.min_keep:
                    break
                if rng.random() < self.mask_prob:
                    joint_values[j] = [0.0, 0.0, 0.0]

        feat = []
        used_xs, used_ys = [], []
        for j in range(self.num_joints):
            x, y, m = joint_values[j]
            feat.extend([x, y, m])
            if m > 0.5:
                used_xs.append(x)
                used_ys.append(y)

        feat.extend(make_one_hot(target_task_id, self.num_targets))
        x = np.array(feat, dtype=np.float32)
        y = np.array([row["target_x"], row["target_y"]], dtype=np.float32)

        return x, y, np.array(used_xs, dtype=np.float32), np.array(used_ys, dtype=np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x, y, used_xs, used_ys = self._build_feature(row, idx)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), idx, used_xs, used_ys


def collate_fn(batch):
    xs, ys, idxs, used_xs, used_ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys), torch.tensor(idxs), used_xs, used_ys


def evaluate(model, df, num_joints, num_targets, mask_prob, min_keep, device, seed=123):
    dataset = RandomMaskEvalDataset(df, num_joints, num_targets, mask_prob, min_keep, seed)
    loader = DataLoader(dataset, batch_size=512, shuffle=False, collate_fn=collate_fn)

    preds = np.zeros((len(df), 2), dtype=np.float32)
    used_context = {}

    model.eval()
    with torch.no_grad():
        for x, _y, idx, used_xs, used_ys in loader:
            x = x.to(device)
            pred = model(x).cpu().numpy()
            idx_np = idx.numpy()
            preds[idx_np] = pred
            for local_i, row_idx in enumerate(idx_np):
                used_context[int(row_idx)] = (used_xs[local_i], used_ys[local_i])

    targets = df[["target_x", "target_y"]].values.astype(np.float32)
    errors = np.linalg.norm(preds - targets, axis=1)

    result_df = df.copy()
    result_df["pred_x"] = preds[:, 0]
    result_df["pred_y"] = preds[:, 1]
    result_df["error_l2"] = errors
    result_df["eval_mask_prob"] = mask_prob
    return result_df, used_context


def plot_examples(result_df, used_context, out_path, max_examples=24):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(result_df) == 0:
        return

    sample_df = result_df.sample(n=min(max_examples, len(result_df)), random_state=9).reset_index()
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
        original_index = int(row["index"])
        xs, ys = used_context.get(original_index, (np.array([]), np.array([])))

        ax.scatter(xs, ys, s=25, label="used context")
        ax.scatter([row["target_x"]], [row["target_y"]], marker="x", s=100, label="true")
        ax.scatter([row["pred_x"]], [row["pred_y"]], marker="^", s=100, label="pred")
        ax.set_title(f"{row['target_name']} | err={row['error_l2']:.3f}")
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/eval_plots_random_mask_finetuned")
    parser.add_argument("--mask_probs", type=str, default="0.0,0.2,0.4,0.6")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_json(model_dir / "model_config.json")
    num_joints = int(config["num_joints"])
    num_targets = int(config["num_targets"])
    hidden_dim = int(config["hidden_dim"])
    dropout = float(config["dropout"])
    min_keep = int(config.get("min_keep", 5))

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    model = FullSkeletonPriorMLP(num_joints, num_targets, hidden_dim, dropout).to(device)
    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))

    df = pd.read_csv(args.csv_path)
    mask_probs = [float(x.strip()) for x in args.mask_probs.split(",") if x.strip()]

    summary_rows = []
    all_results = []

    for mp in mask_probs:
        result_df, used_context = evaluate(model, df, num_joints, num_targets, mp, min_keep, device, seed=123)
        all_results.append(result_df)
        summary_rows.append({
            "mask_prob": mp,
            "count": len(result_df),
            "mean_l2": result_df["error_l2"].mean(),
            "median_l2": result_df["error_l2"].median(),
            "std_l2": result_df["error_l2"].std(),
        })
        result_df.to_csv(out_dir / f"predictions_mask_{mp:.2f}.csv", index=False)
        plot_examples(result_df, used_context, out_dir / f"prediction_examples_mask_{mp:.2f}.png")

        print(f"\nMask prob {mp:.2f}")
        print(result_df.groupby("target_name")["error_l2"].agg(["count", "mean", "median", "std"]))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "random_mask_summary.csv", index=False)
    pd.concat(all_results, ignore_index=True).to_csv(out_dir / "random_mask_all_predictions.csv", index=False)

    print("\nSummary:")
    print(summary_df)
    print(f"\nSaved summary: {out_dir / 'random_mask_summary.csv'}")


if __name__ == "__main__":
    main()
