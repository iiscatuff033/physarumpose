import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.slime_path_utils import (
    get_anchor_points,
    is_valid_point,
    generate_candidates,
    score_candidates,
    choose_best_candidate,
    point_error,
    get_yolo_target,
    get_prior_target,
    get_geometry_target,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='outputs/stage3a_slime_path')
    parser.add_argument('--max_samples', type=int, default=2000)
    parser.add_argument('--anchor_conf_thresh', type=float, default=0.25)
    parser.add_argument('--edge_w', type=float, default=1.0)
    parser.add_argument('--anatomy_w', type=float, default=1.0)
    parser.add_argument('--occ_w', type=float, default=1.2)
    parser.add_argument('--prior_w', type=float, default=0.35)
    parser.add_argument('--slime_iters', type=int, default=8)
    parser.add_argument('--slime_diffusion', type=float, default=0.20)
    parser.add_argument('--slime_decay', type=float, default=0.08)
    parser.add_argument('--edge_blur', type=int, default=3)
    parser.add_argument('--edge_low', type=int, default=50)
    parser.add_argument('--edge_high', type=int, default=150)
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    cand_dir = out_dir / 'candidate_tables'
    out_dir.mkdir(parents=True, exist_ok=True)
    cand_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if 'yolo_found_person' in df.columns:
        df = df[df['yolo_found_person'] == 1].reset_index(drop=True)
    if args.max_samples > 0:
        df = df.head(args.max_samples).reset_index(drop=True)

    weights = vars(args).copy()
    rows = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Slime path recovery'):
        out = row.to_dict()
        image_path = image_dir / row['occluded_image_name']
        if not image_path.exists():
            out['slime_status'] = 'missing_image'
            rows.append(out)
            continue
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            out['slime_status'] = 'bad_image'
            rows.append(out)
            continue

        a, b, ac, bc = get_anchor_points(row)
        out['anchor1_x'] = float(a[0]) if np.isfinite(a[0]) else np.nan
        out['anchor1_y'] = float(a[1]) if np.isfinite(a[1]) else np.nan
        out['anchor2_x'] = float(b[0]) if np.isfinite(b[0]) else np.nan
        out['anchor2_y'] = float(b[1]) if np.isfinite(b[1]) else np.nan
        out['anchor1_conf'] = float(ac)
        out['anchor2_conf'] = float(bc)
        out['anchor_min_conf'] = float(min(ac, bc))

        if not is_valid_point(a) or not is_valid_point(b):
            out['slime_status'] = 'bad_anchors'
            rows.append(out)
            continue
        if ac < args.anchor_conf_thresh or bc < args.anchor_conf_thresh:
            out['slime_status'] = 'low_anchor_conf'
            rows.append(out)
            continue

        box = (row['occ_x1'], row['occ_y1'], row['occ_x2'], row['occ_y2'])
        candidates = generate_candidates(a, b, image_bgr.shape, box=box)
        if len(candidates) == 0:
            out['slime_status'] = 'no_candidates'
            rows.append(out)
            continue

        cand_df = score_candidates(row, image_bgr, candidates, a, b, weights)
        best = choose_best_candidate(cand_df)
        if best is None:
            out['slime_status'] = 'no_best'
            rows.append(out)
            continue

        sx = float(best['x'])
        sy = float(best['y'])
        out['slime_status'] = 'ok'
        out['slime_pred_x'] = sx
        out['slime_pred_y'] = sy
        out['slime_score'] = float(best['slime_score'])
        out['slime_raw_score'] = float(best['raw_score'])
        out['slime_edge_score'] = float(best['edge_score'])
        out['slime_anatomy_score'] = float(best['anatomy_score'])
        out['slime_occ_score'] = float(best['occ_score'])
        out['slime_prior_score'] = float(best['prior_score'])
        out['num_candidates'] = int(len(cand_df))

        yx, yy = get_yolo_target(row)
        px, py = get_prior_target(row)
        gx, gy = get_geometry_target(row)
        out['slime_error_px'] = point_error(sx, sy, row['target_orig_x'], row['target_orig_y'])
        out['yolo_error_px_check'] = point_error(yx, yy, row['target_orig_x'], row['target_orig_y'])
        out['prior_error_px_check'] = point_error(px, py, row['target_orig_x'], row['target_orig_y'])
        out['geometry_error_px_check'] = point_error(gx, gy, row['target_orig_x'], row['target_orig_y'])
        out['slime_better_than_yolo'] = out['slime_error_px'] < out.get('yolo_target_error_px', out['yolo_error_px_check'])
        out['slime_better_than_prior'] = out['slime_error_px'] < out.get('prior_error_px', out['prior_error_px_check'])
        out['slime_better_than_geometry'] = out['slime_error_px'] < out.get('geometry_error_px', out['geometry_error_px_check'])

        if idx < 80:
            cand_df.sort_values('slime_score', ascending=False).head(80).to_csv(
                cand_dir / f"candidates_{idx:05d}_{row['target_name']}.csv", index=False
            )
        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_csv = out_dir / 'slime_path_results.csv'
    out_df.to_csv(out_csv, index=False)
    valid = out_df[out_df['slime_status'] == 'ok'].copy()
    print(f'\nSaved results: {out_csv}')
    print(f'Valid slime predictions: {len(valid)} / {len(out_df)}')
    if len(valid) > 0:
        print('\nQuick summary:')
        print(valid.groupby('target_name')[['yolo_target_error_px', 'slime_error_px']].agg(['count', 'mean', 'median']))


if __name__ == '__main__':
    main()
