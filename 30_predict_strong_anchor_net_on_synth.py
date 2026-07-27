import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.heatmap_utils import (
    ANCHOR_NAMES,
    ANCHOR_TO_MPII,
    crop_and_resize,
    crop_to_orig_xy,
    decode_heatmaps,
    expand_bbox,
    image_to_tensor,
    load_json,
)
from src.strong_anchor_net import StrongAnchorNet


def bbox_from_gt_row(row):
    xs, ys = [], []
    for j in range(16):
        x = float(row.get(f'j{j}_orig_x', 0.0))
        y = float(row.get(f'j{j}_orig_y', 0.0))
        if x > 0 and y > 0:
            xs.append(x)
            ys.append(y)
    if len(xs) < 3:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def full_box(image_bgr):
    h, w = image_bgr.shape[:2]
    return 0, 0, w, h


def predict_one(model, image_bgr, box, config, device):
    box = expand_bbox(*box, margin=float(config.get('bbox_margin', 0.20)))
    crop, meta = crop_and_resize(image_bgr, box, int(config['input_w']), int(config['input_h']))
    x = image_to_tensor(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x)[0].cpu()
    coords_crop, confs = decode_heatmaps(pred, int(config['input_w']), int(config['input_h']), apply_sigmoid=True)

    coords_orig = []
    for j in range(len(ANCHOR_NAMES)):
        ox, oy = crop_to_orig_xy(coords_crop[j, 0], coords_crop[j, 1], meta)
        coords_orig.append([ox, oy])
    return np.array(coords_orig, dtype=np.float32), confs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_csv', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='outputs/stage4b_strong_anchor_predictions')
    parser.add_argument('--crop_mode', type=str, default='gt_bbox', choices=['gt_bbox', 'full'])
    parser.add_argument('--max_samples', type=int, default=0)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.metadata_csv)
    if args.max_samples > 0:
        df = df.head(args.max_samples).reset_index(drop=True)

    config = load_json(Path(args.model_dir) / 'model_config.json')
    device = 'cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = StrongAnchorNet(num_joints=len(ANCHOR_NAMES), pretrained=False).to(device)
    model.load_state_dict(torch.load(Path(args.model_dir) / 'best_model.pt', map_location=device))
    model.eval()

    image_dir = Path(args.image_dir)
    rows = []
    anchor_errs = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc='Predicting strong AnchorNet anchors'):
        out = row.to_dict()
        image_path = image_dir / row['occluded_image_name']
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            out['anchor_status'] = 'bad_image'
            rows.append(out)
            continue

        if args.crop_mode == 'gt_bbox':
            box = bbox_from_gt_row(row)
            if box is None:
                box = full_box(image_bgr)
        else:
            box = full_box(image_bgr)

        coords, confs = predict_one(model, image_bgr, box, config, device)

        out['anchor_status'] = 'ok'
        errs = []
        for j, name in enumerate(ANCHOR_NAMES):
            out[f'anchor_pred_{name}_x'] = float(coords[j, 0])
            out[f'anchor_pred_{name}_y'] = float(coords[j, 1])
            out[f'anchor_pred_{name}_conf'] = float(confs[j])

            mpii_id = ANCHOR_TO_MPII[name]
            gx = float(row.get(f'j{mpii_id}_orig_x', 0.0))
            gy = float(row.get(f'j{mpii_id}_orig_y', 0.0))
            if gx > 0 and gy > 0:
                err = float(np.sqrt((coords[j, 0] - gx) ** 2 + (coords[j, 1] - gy) ** 2))
                out[f'anchor_err_{name}'] = err
                errs.append(err)
                anchor_errs.append(err)
            else:
                out[f'anchor_err_{name}'] = np.nan

        out['mean_anchor_err'] = float(np.mean(errs)) if errs else np.nan
        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_csv = out_dir / 'strong_anchor_predictions.csv'
    out_df.to_csv(out_csv, index=False)

    summary = pd.DataFrame([{
        'count': len(out_df),
        'mean_anchor_err': float(np.nanmean(anchor_errs)) if anchor_errs else np.nan,
        'median_anchor_err': float(np.nanmedian(anchor_errs)) if anchor_errs else np.nan,
        'crop_mode': args.crop_mode,
    }])
    summary.to_csv(out_dir / 'strong_anchor_prediction_summary.csv', index=False)

    print('\nSaved:', out_csv)
    print(summary)


if __name__ == '__main__':
    main()
