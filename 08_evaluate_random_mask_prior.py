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
    def __init__(self, df, num_joints=16, num_targets=4, mask_prob=0.0, min_keep=5, seed=123):
        self.df = df.reset_index(drop=True)
        self.num_joints = num_joints
        self.num_targets = num_targets
        self.mask_prob = float(mask_prob)
        self.min_keep = int(min_keep)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.df)

    def _build_feature(self, row):
        target_task_id = int(row['target_task_id'])
        joint_values = []
        visible_ids = []

        for j in range(self.num_joints):
            x = float(row[f'j{j}_x'])
            y = float(row[f'j{j}_y'])
            m = float(row[f'j{j}_mask'])
            if m > 0.5:
                visible_ids.append(j)
            joint_values.append([x, y, m])

        if self.mask_prob > 0:
            current_visible = list(visible_ids)
            self.rng.shuffle(current_visible)
            for j in current_visible:
                current_count = sum(1 for item in joint_values if item[2] > 0.5)
                if current_count <= self.min_keep:
                    break
                if self.rng.random() < self.mask_prob:
                    joint_values[j] = [0.0, 0.0, 0.0]

        feat = []
        for j in range(self.num_joints):
            feat.extend(joint_values[j])
        feat.extend(make_one_hot(target_task_id, self.num_targets))

        x = np.array(feat, dtype=np.float32)
        y = np.array([row['target_x'], row['target_y']], dtype=np.float32)
        return x, y

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x, y = self._build_feature(row)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), idx


def evaluate(model, df, num_joints, num_targets, mask_prob, min_keep, device, seed=123):
    dataset = RandomMaskEvalDataset(df, num_joints, num_targets, mask_prob, min_keep, seed)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    preds = np.zeros((len(df), 2), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for x, _y, idx in loader:
            x = x.to(device)
            pred = model(x).cpu().numpy()
            preds[idx.numpy()] = pred

    targets = df[['target_x', 'target_y']].values.astype(np.float32)
    errors = np.linalg.norm(preds - targets, axis=1)
    result_df = df.copy()
    result_df['pred_x'] = preds[:, 0]
    result_df['pred_y'] = preds[:, 1]
    result_df['error_l2'] = errors
    result_df['eval_mask_prob'] = mask_prob
    return result_df


def plot_examples(result_df, out_path, max_examples=24):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(result_df) == 0:
        return

    sample_df = result_df.sample(n=min(max_examples, len(result_df)), random_state=9).reset_index(drop=True)
    cols = 4
    rows = int(np.ceil(len(sample_df) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if rows == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for ax_idx, ax in enumerate(axes):
        ax.axis('equal')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        if ax_idx >= len(sample_df):
            ax.axis('off')
            continue
        row = sample_df.iloc[ax_idx]
        xs, ys = [], []
        for j in range(16):
            if row[f'j{j}_mask'] > 0.5:
                xs.append(row[f'j{j}_x'])
                ys.append(row[f'j{j}_y'])
        ax.scatter(xs, ys, s=25, label='original context')
        ax.scatter([row['target_x']], [row['target_y']], marker='x', s=100, label='true')
        ax.scatter([row['pred_x']], [row['pred_y']], marker='^', s=100, label='pred')
        ax.set_title(f"{row['target_name']} | err={row['error_l2']:.3f}")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='outputs/eval_plots_random_mask')
    parser.add_argument('--mask_probs', type=str, default='0.0,0.2,0.4,0.6')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_json(model_dir / 'model_config.json')
    num_joints = int(config['num_joints'])
    num_targets = int(config['num_targets'])
    hidden_dim = int(config['hidden_dim'])
    dropout = float(config['dropout'])
    min_keep = int(config.get('min_keep', 5))

    device = 'cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    model = FullSkeletonPriorMLP(num_joints=num_joints, num_targets=num_targets, hidden_dim=hidden_dim, dropout=dropout).to(device)
    model.load_state_dict(torch.load(model_dir / 'best_model.pt', map_location=device))

    df = pd.read_csv(args.csv_path)
    mask_probs = [float(x.strip()) for x in args.mask_probs.split(',') if x.strip()]
    summary_rows = []
    all_results = []

    for mp in mask_probs:
        result_df = evaluate(model, df, num_joints, num_targets, mp, min_keep, device, seed=123)
        all_results.append(result_df)
        summary_rows.append({
            'mask_prob': mp,
            'count': len(result_df),
            'mean_l2': result_df['error_l2'].mean(),
            'median_l2': result_df['error_l2'].median(),
            'std_l2': result_df['error_l2'].std(),
        })
        plot_examples(result_df, out_dir / f'prediction_examples_mask_{mp:.2f}.png')
        print(f'\nMask prob {mp:.2f}')
        print(result_df.groupby('target_name')['error_l2'].agg(['count', 'mean', 'median', 'std']))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / 'random_mask_summary.csv', index=False)
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(out_dir / 'random_mask_all_predictions.csv', index=False)
    print('\nSummary:')
    print(summary_df)
    print(f"\nSaved summary: {out_dir / 'random_mask_summary.csv'}")
    print(f"Saved all predictions: {out_dir / 'random_mask_all_predictions.csv'}")


if __name__ == '__main__':
    main()
