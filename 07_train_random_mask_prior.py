import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.full_context_model import FullSkeletonPriorMLP
from src.utils_full import TARGET_JOINTS, save_json, set_seed


def make_one_hot(index: int, size: int):
    arr = np.zeros(size, dtype=np.float32)
    arr[index] = 1.0
    return arr


class RandomMaskFullContextDataset(Dataset):
    """
    Dataset for Stage 1C.

    Each CSV row already has the target joint hidden.
    During training, we randomly hide additional visible context joints.
    """

    def __init__(self, df, num_joints=16, num_targets=4, mask_prob=0.30, min_keep=5, training=True, seed=42):
        self.df = df.reset_index(drop=True)
        self.num_joints = num_joints
        self.num_targets = num_targets
        self.mask_prob = float(mask_prob)
        self.min_keep = int(min_keep)
        self.training = bool(training)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.df)

    def _build_feature(self, row):
        target_task_id = int(row['target_task_id'])
        joint_values = []
        visible_joint_ids = []

        for j in range(self.num_joints):
            x = float(row[f'j{j}_x'])
            y = float(row[f'j{j}_y'])
            m = float(row[f'j{j}_mask'])
            if m > 0.5:
                visible_joint_ids.append(j)
            joint_values.append([x, y, m])

        if self.training and self.mask_prob > 0:
            current_visible = list(visible_joint_ids)
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
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def split_by_image(df, val_ratio=0.15, seed=42):
    rng = np.random.default_rng(seed)
    image_names = df['image_name'].dropna().unique()
    rng.shuffle(image_names)
    n_val = max(1, int(len(image_names) * val_ratio))
    val_images = set(image_names[:n_val])
    val_df = df[df['image_name'].isin(val_images)].reset_index(drop=True)
    train_df = df[~df['image_name'].isin(val_images)].reset_index(drop=True)
    return train_df, val_df


def compute_metrics(pred, target):
    err = torch.linalg.norm(pred - target, dim=1)
    return {'mean_l2': float(err.mean().item()), 'median_l2': float(err.median().item())}


def run_epoch(model, loader, optimizer, device, train=True):
    model.train() if train else model.eval()
    loss_fn = torch.nn.SmoothL1Loss()
    total_loss = 0.0
    all_pred = []
    all_target = []

    for x, y in tqdm(loader, leave=False):
        x = x.to(device)
        y = y.to(device)
        with torch.set_grad_enabled(train):
            pred = model(x)
            loss = loss_fn(pred, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * x.size(0)
        all_pred.append(pred.detach().cpu())
        all_target.append(y.detach().cpu())

    all_pred = torch.cat(all_pred, dim=0)
    all_target = torch.cat(all_target, dim=0)
    metrics = compute_metrics(all_pred, all_target)
    metrics['loss'] = total_loss / len(loader.dataset)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='outputs/random_mask_prior_mpii')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--mask_prob', type=float, default=0.30)
    parser.add_argument('--min_keep', type=int, default=5)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    if len(df) == 0:
        raise ValueError('CSV has 0 rows. Check your Stage 1B preparation step.')

    train_df, val_df = split_by_image(df, val_ratio=args.val_ratio, seed=args.seed)
    print(f'Total samples: {len(df)}')
    print(f'Train samples: {len(train_df)}')
    print(f'Val samples: {len(val_df)}')
    print(f'Training mask probability: {args.mask_prob}')
    print(f'Minimum context joints kept: {args.min_keep}')
    print('\nTrain target distribution:')
    print(train_df['target_name'].value_counts())

    num_joints = 16
    num_targets = len(TARGET_JOINTS)

    train_dataset = RandomMaskFullContextDataset(train_df, num_joints, num_targets, args.mask_prob, args.min_keep, True, args.seed)
    val_dataset = RandomMaskFullContextDataset(val_df, num_joints, num_targets, 0.0, args.min_keep, False, args.seed)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = 'cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nUsing device: {device}')

    model = FullSkeletonPriorMLP(num_joints=num_joints, num_targets=num_targets, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = float('inf')
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, train=False)
        row = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_mean_l2': train_metrics['mean_l2'],
            'val_loss_clean': val_metrics['loss'],
            'val_mean_l2_clean': val_metrics['mean_l2'],
            'val_median_l2_clean': val_metrics['median_l2'],
        }
        history.append(row)
        print(f"Epoch {epoch:03d} | train_loss={row['train_loss']:.5f} | train_l2={row['train_mean_l2']:.5f} | val_clean_loss={row['val_loss_clean']:.5f} | val_clean_l2={row['val_mean_l2_clean']:.5f}")
        if val_metrics['mean_l2'] < best_val:
            best_val = val_metrics['mean_l2']
            torch.save(model.state_dict(), out_dir / 'best_model.pt')
            save_json({
                'num_joints': num_joints,
                'num_targets': num_targets,
                'target_joints': TARGET_JOINTS,
                'hidden_dim': args.hidden_dim,
                'dropout': args.dropout,
                'train_mask_prob': args.mask_prob,
                'min_keep': args.min_keep,
                'best_val_mean_l2_clean': best_val,
            }, out_dir / 'model_config.json')

    pd.DataFrame(history).to_csv(out_dir / 'training_history.csv', index=False)
    train_df.to_csv(out_dir / 'train_split.csv', index=False)
    val_df.to_csv(out_dir / 'val_split.csv', index=False)
    print('\nTraining complete.')
    print(f'Best clean val mean L2: {best_val:.5f}')
    print(f"Saved model: {out_dir / 'best_model.pt'}")


if __name__ == '__main__':
    main()
