import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def draw_marker(draw, x, y, kind, color, r=7):
    if pd.isna(x) or pd.isna(y):
        return
    x = int(round(x)); y = int(round(y))
    if kind == 'x':
        draw.line([x-r, y-r, x+r, y+r], fill=color, width=4)
        draw.line([x-r, y+r, x+r, y-r], fill=color, width=4)
    elif kind == 'circle':
        draw.ellipse([x-r, y-r, x+r, y+r], outline=color, width=4)
    elif kind == 'diamond':
        draw.polygon([(x, y-r), (x-r, y), (x, y+r), (x+r, y)], outline=color, fill=color)
    elif kind == 'dot':
        draw.ellipse([x-4, y-4, x+4, y+4], fill=color)


def visualize_one(image_path, row, out_path):
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)

    # occlusion
    x1, y1, x2, y2 = int(row['occ_x1']), int(row['occ_y1']), int(row['occ_x2']), int(row['occ_y2'])
    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)

    # anchors
    ax, ay = row.get('slime_anchor1_x', np.nan), row.get('slime_anchor1_y', np.nan)
    bx, by = row.get('slime_anchor2_x', np.nan), row.get('slime_anchor2_y', np.nan)
    draw_marker(draw, ax, ay, 'dot', (0, 120, 255))
    draw_marker(draw, bx, by, 'dot', (0, 120, 255))

    # chosen path
    if row.get('strong_anchor_slime_status', '') == 'ok':
        sx, sy = row['strong_anchor_slime_x'], row['strong_anchor_slime_y']
        draw.line([ax, ay, sx, sy], fill=(255, 255, 0), width=3)
        draw.line([sx, sy, bx, by], fill=(255, 255, 0), width=3)
        draw_marker(draw, sx, sy, 'diamond', (255, 255, 0))

    # true target
    draw_marker(draw, row['target_orig_x'], row['target_orig_y'], 'x', (255, 165, 0))

    # YOLO baseline target
    if 'yolo_target_x' in row:
        draw_marker(draw, row['yolo_target_x'], row['yolo_target_y'], 'circle', (160, 0, 255))

    text = f"{row['target_name']} | YOLO={row.get('yolo_target_error_px', np.nan):.1f} STRONG-SLIME={row.get('strong_anchor_slime_error_px', np.nan):.1f}"
    draw.rectangle([8, 8, 720, 38], fill=(0, 0, 0))
    draw.text((12, 14), text, fill=(255, 255, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_csv', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='outputs/stage4b_anchor_slime/visualizations')
    parser.add_argument('--max_images', type=int, default=150)
    parser.add_argument('--only_valid', action='store_true')
    parser.add_argument('--only_improved', action='store_true')
    args = parser.parse_args()

    df = pd.read_csv(args.results_csv)
    if args.only_valid:
        df = df[df['strong_anchor_slime_status'] == 'ok'].reset_index(drop=True)
    if args.only_improved:
        df = df[(df['strong_anchor_slime_status'] == 'ok') & (df['strong_anchor_slime_error_px'] < df['yolo_target_error_px'])].reset_index(drop=True)
    if len(df) == 0:
        print('No rows to visualize.')
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.image_dir)
    sample = df.sample(n=min(args.max_images, len(df)), random_state=31)
    for _, row in sample.iterrows():
        visualize_one(image_dir / row['occluded_image_name'], row, out_dir / row['occluded_image_name'])
    print('Saved visualizations to:', out_dir)


if __name__ == '__main__':
    main()
