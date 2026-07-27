import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.anchor_heatmap_dataset import AnchorHeatmapDataset
from src.heatmap_utils import (
    ANCHOR_NAMES,
    crop_and_resize,
    decode_heatmaps,
    expand_bbox,
    load_json,
    orig_to_crop_xy,
    crop_to_orig_xy,
    save_json,
)
from src.strong_anchor_net import StrongAnchorNet


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_by_image(df, val_ratio=0.20, seed=42):
    rng = np.random.default_rng(seed)
    names = df['image_name'].dropna().unique()
    rng.shuffle(names)
    n_val = max(1, int(len(names) * val_ratio))
    val_names = set(names[:n_val])
    val = df[df['image_name'].isin(val_names)].reset_index(drop=True)
    train = df[~df['image_name'].isin(val_names)].reset_index(drop=True)
    return train, val


def heatmap_loss(pred, target, mask):
    # pred/target: B,J,H,W ; mask: B,J,1,1
    loss = (pred - target) ** 2
    loss = loss * mask
    denom = mask.sum() * pred.shape[2] * pred.shape[3]
    return loss.sum() / torch.clamp(denom, min=1.0)


def run_epoch(model, loader, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0

    with torch.set_grad_enabled(train):
        for x, y, m, _idx in tqdm(loader, leave=False):
            x = x.to(device)
            y = y.to(device)
            m = m.to(device)

            pred = model(x)
            loss = heatmap_loss(pred, y, m)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)

    return total_loss / max(len(loader.dataset), 1)


def evaluate_pixels(model, df, image_dir, device, input_w, input_h, hm_w, hm_h, bbox_margin, batch_size=32):
    ds = AnchorHeatmapDataset(
        df,
        image_dir=image_dir,
        input_w=input_w,
        input_h=input_h,
        hm_w=hm_w,
        hm_h=hm_h,
        sigma=2.0,
        train=False,
        bbox_margin=bbox_margin,
        cutout_prob=0.0,
        flip_prob=0.0,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    pred_maps = {}
    model.eval()
    with torch.no_grad():
        for x, _y, _m, idx in tqdm(loader, leave=False):
            x = x.to(device)
            pred = model(x).cpu()
            for k, sample_idx in enumerate(idx.numpy()):
                pred_maps[int(sample_idx)] = pred[k]

    rows = []
    all_errs = []
    for i, row in df.reset_index(drop=True).iterrows():
        pred = pred_maps[i]
        coords_crop, confs = decode_heatmaps(pred, input_w=input_w, input_h=input_h, apply_sigmoid=True)

        # Reconstruct crop meta same as dataset
        import cv2
        img = cv2.imread(str(Path(image_dir) / row['image_name']))
        if img is None:
            continue
        box = expand_bbox(row['bbox_x1'], row['bbox_y1'], row['bbox_x2'], row['bbox_y2'], margin=bbox_margin)
        _crop, meta = crop_and_resize(img, box, input_w, input_h)

        out = row.to_dict()
        errs = []
        for j, name in enumerate(ANCHOR_NAMES):
            px, py = crop_to_orig_xy(coords_crop[j, 0], coords_crop[j, 1], meta)
            out[f'pred_{name}_x'] = px
            out[f'pred_{name}_y'] = py
            out[f'pred_{name}_conf'] = float(confs[j])

            if float(row[f'{name}_vis']) > 0:
                gx = float(row[f'{name}_x'])
                gy = float(row[f'{name}_y'])
                err = float(np.sqrt((px - gx) ** 2 + (py - gy) ** 2))
                out[f'err_{name}'] = err
                errs.append(err)
                all_errs.append(err)
            else:
                out[f'err_{name}'] = np.nan

        out['mean_anchor_err'] = float(np.mean(errs)) if errs else np.nan
        rows.append(out)

    pred_df = pd.DataFrame(rows)
    mean_px = float(np.nanmean(all_errs)) if all_errs else float('inf')
    median_px = float(np.nanmedian(all_errs)) if all_errs else float('inf')
    return pred_df, mean_px, median_px


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='outputs/strong_anchor_net_mpii')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--input_h', type=int, default=384)
    parser.add_argument('--input_w', type=int, default=288)
    parser.add_argument('--heatmap_stride', type=int, default=4)
    parser.add_argument('--sigma', type=float, default=2.0)
    parser.add_argument('--bbox_margin', type=float, default=0.20)
    parser.add_argument('--val_ratio', type=float, default=0.20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_pretrained', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    train_df, val_df = split_by_image(df, val_ratio=args.val_ratio, seed=args.seed)

    train_df.to_csv(out_dir / 'train_split.csv', index=False)
    val_df.to_csv(out_dir / 'val_split.csv', index=False)

    hm_w = args.input_w // args.heatmap_stride
    hm_h = args.input_h // args.heatmap_stride

    train_ds = AnchorHeatmapDataset(
        train_df,
        image_dir=args.image_dir,
        input_w=args.input_w,
        input_h=args.input_h,
        hm_w=hm_w,
        hm_h=hm_h,
        sigma=args.sigma,
        train=True,
        bbox_margin=args.bbox_margin,
    )
    val_ds = AnchorHeatmapDataset(
        val_df,
        image_dir=args.image_dir,
        input_w=args.input_w,
        input_h=args.input_h,
        hm_w=hm_w,
        hm_h=hm_h,
        sigma=args.sigma,
        train=False,
        bbox_margin=args.bbox_margin,
        cutout_prob=0.0,
        flip_prob=0.0,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = 'cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Train samples: {len(train_df)} | Val samples: {len(val_df)}')
    print(f'Input: {args.input_w}x{args.input_h} | Heatmap: {hm_w}x{hm_h}')

    model = StrongAnchorNet(num_joints=len(ANCHOR_NAMES), pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    best_val_px = float('inf')
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, train=False)
        scheduler.step()

        # Pixel eval every epoch. If this is slow, increase eval frequency later.
        _val_pred, val_mean_px, val_median_px = evaluate_pixels(
            model,
            val_df,
            image_dir=args.image_dir,
            device=device,
            input_w=args.input_w,
            input_h=args.input_h,
            hm_w=hm_w,
            hm_h=hm_h,
            bbox_margin=args.bbox_margin,
            batch_size=args.batch_size,
        )

        row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_mean_anchor_px': val_mean_px,
            'val_median_anchor_px': val_median_px,
            'lr': optimizer.param_groups[0]['lr'],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"val_anchor_mean={val_mean_px:.2f}px | val_anchor_median={val_median_px:.2f}px"
        )

        if val_mean_px < best_val_px:
            best_val_px = val_mean_px
            torch.save(model.state_dict(), out_dir / 'best_model.pt')
            save_json({
                'input_w': args.input_w,
                'input_h': args.input_h,
                'hm_w': hm_w,
                'hm_h': hm_h,
                'heatmap_stride': args.heatmap_stride,
                'sigma': args.sigma,
                'bbox_margin': args.bbox_margin,
                'num_joints': len(ANCHOR_NAMES),
                'anchor_names': ANCHOR_NAMES,
                'pretrained': not args.no_pretrained,
                'best_val_mean_anchor_px': best_val_px,
            }, out_dir / 'model_config.json')

        pd.DataFrame(history).to_csv(out_dir / 'training_history.csv', index=False)

    # final val predictions using best model
    config = load_json(out_dir / 'model_config.json')
    model.load_state_dict(torch.load(out_dir / 'best_model.pt', map_location=device))
    val_pred, val_mean_px, val_median_px = evaluate_pixels(
        model,
        val_df,
        image_dir=args.image_dir,
        device=device,
        input_w=config['input_w'],
        input_h=config['input_h'],
        hm_w=config['hm_w'],
        hm_h=config['hm_h'],
        bbox_margin=config['bbox_margin'],
        batch_size=args.batch_size,
    )
    val_pred.to_csv(out_dir / 'val_predictions.csv', index=False)

    print('\nBest val mean anchor px:', best_val_px)
    print('Saved to:', out_dir)


if __name__ == '__main__':
    main()
