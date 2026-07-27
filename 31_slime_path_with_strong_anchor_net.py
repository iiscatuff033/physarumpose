import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.slime_path_utils_strong_anchor import (
    get_anchor_points,
    is_valid_point,
    generate_candidates,
    score_candidates,
    point_error,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='outputs/stage4b_anchor_slime')
    parser.add_argument('--max_samples', type=int, default=2000)
    parser.add_argument('--anchor_conf_thresh', type=float, default=0.10)
    parser.add_argument('--edge_w', type=float, default=1.0)
    parser.add_argument('--anatomy_w', type=float, default=1.0)
    parser.add_argument('--occ_w', type=float, default=1.2)
    parser.add_argument('--slime_iters', type=int, default=8)
    parser.add_argument('--slime_diffusion', type=float, default=0.20)
    parser.add_argument('--slime_decay', type=float, default=0.08)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    if args.max_samples > 0:
        df = df.head(args.max_samples).reset_index(drop=True)

    weights = {
        'edge_w': args.edge_w,
        'anatomy_w': args.anatomy_w,
        'occ_w': args.occ_w,
        'slime_iters': args.slime_iters,
        'slime_diffusion': args.slime_diffusion,
        'slime_decay': args.slime_decay,
    }

    rows = []
    image_dir = Path(args.image_dir)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Strong AnchorNet + slime'):
        out = row.to_dict()
        image_path = image_dir / row['occluded_image_name']
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            out['strong_anchor_slime_status'] = 'bad_image'
            rows.append(out)
            continue

        a, b, ac, bc, aname, bname = get_anchor_points(row)
        out['slime_anchor1_name'] = aname
        out['slime_anchor2_name'] = bname
        out['slime_anchor1_x'] = float(a[0]) if np.isfinite(a[0]) else np.nan
        out['slime_anchor1_y'] = float(a[1]) if np.isfinite(a[1]) else np.nan
        out['slime_anchor2_x'] = float(b[0]) if np.isfinite(b[0]) else np.nan
        out['slime_anchor2_y'] = float(b[1]) if np.isfinite(b[1]) else np.nan
        out['slime_anchor1_conf'] = float(ac)
        out['slime_anchor2_conf'] = float(bc)

        if not is_valid_point(a) or not is_valid_point(b):
            out['strong_anchor_slime_status'] = 'bad_anchors'
            rows.append(out)
            continue
        if min(ac, bc) < args.anchor_conf_thresh:
            out['strong_anchor_slime_status'] = 'low_anchor_conf'
            rows.append(out)
            continue

        box = (row['occ_x1'], row['occ_y1'], row['occ_x2'], row['occ_y2'])
        candidates = generate_candidates(a, b, image_bgr.shape, box=box)
        cand_df = score_candidates(row, image_bgr, candidates, a, b, weights)
        if len(cand_df) == 0:
            out['strong_anchor_slime_status'] = 'no_candidates'
            rows.append(out)
            continue

        best = cand_df.sort_values('slime_score', ascending=False).iloc[0]
        sx = float(best['x'])
        sy = float(best['y'])

        out['strong_anchor_slime_status'] = 'ok'
        out['strong_anchor_slime_x'] = sx
        out['strong_anchor_slime_y'] = sy
        out['strong_anchor_slime_score'] = float(best['slime_score'])
        out['strong_anchor_slime_edge_score'] = float(best['edge_score'])
        out['strong_anchor_slime_anatomy_score'] = float(best['anatomy_score'])
        out['strong_anchor_slime_occ_score'] = float(best['occ_score'])
        out['num_candidates'] = int(len(cand_df))
        out['strong_anchor_slime_error_px'] = point_error(sx, sy, row['target_orig_x'], row['target_orig_y'])
        out['strong_anchor_slime_better_than_yolo'] = out['strong_anchor_slime_error_px'] < row['yolo_target_error_px']
        if 'slime_error_px' in row:
            out['strong_anchor_slime_better_than_yolo_anchor_slime'] = out['strong_anchor_slime_error_px'] < row['slime_error_px']

        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_csv = out_dir / 'strong_anchor_slime_results.csv'
    out_df.to_csv(out_csv, index=False)

    valid = out_df[out_df['strong_anchor_slime_status'] == 'ok']
    print('\nSaved:', out_csv)
    print(f'Valid: {len(valid)} / {len(out_df)}')
    if len(valid) > 0:
        print(valid.groupby('target_name')[['yolo_target_error_px', 'strong_anchor_slime_error_px']].agg(['count', 'mean', 'median']))


if __name__ == '__main__':
    main()
