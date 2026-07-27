import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.anchor_slime_utils import get_anchor_points, generate_candidates, get_occlusion_box


def draw_marker(draw, x, y, kind, color, r=7):
    if pd.isna(x) or pd.isna(y):
        return
    x = int(round(x))
    y = int(round(y))
    if kind == "x":
        draw.line([x - r, y - r, x + r, y + r], fill=color, width=4)
        draw.line([x - r, y + r, x + r, y - r], fill=color, width=4)
    elif kind == "circle":
        draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=4)
    elif kind == "triangle":
        tri = [(x, y - r), (x - r, y + r), (x + r, y + r)]
        draw.polygon(tri, outline=color, fill=color)
    elif kind == "diamond":
        dia = [(x, y - r), (x - r, y), (x, y + r), (x + r, y)]
        draw.polygon(dia, outline=color, fill=color)
    elif kind == "dot":
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)


def visualize_one(image_path, row, out_path, show_candidates=True):
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)

    box = get_occlusion_box(row)
    if box is not None:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)

    a, b, ac, bc, a_name, b_name = get_anchor_points(row)
    draw_marker(draw, a[0], a[1], "dot", (0, 120, 255), r=6)
    draw_marker(draw, b[0], b[1], "dot", (0, 120, 255), r=6)

    if show_candidates and row.get("anchor_slime_status", "") == "ok":
        cands = generate_candidates(a, b, bgr.shape, box=box)
        step = max(1, len(cands) // 80)
        for c in cands[::step]:
            draw_marker(draw, c[0], c[1], "dot", (150, 150, 150), r=2)

    if row.get("anchor_slime_status", "") == "ok":
        sx = row["anchor_slime_pred_x"]
        sy = row["anchor_slime_pred_y"]
        draw.line([a[0], a[1], sx, sy], fill=(255, 255, 0), width=3)
        draw.line([sx, sy, b[0], b[1]], fill=(255, 255, 0), width=3)

    # true target
    draw_marker(draw, row["target_orig_x"], row["target_orig_y"], "x", (255, 165, 0))

    # YOLO target baseline if present
    if "yolo_target_x" in row:
        draw_marker(draw, row["yolo_target_x"], row["yolo_target_y"], "circle", (160, 0, 255))

    # previous yolo-anchor slime baseline if present
    if "slime_pred_x" in row:
        draw_marker(draw, row["slime_pred_x"], row["slime_pred_y"], "triangle", (0, 255, 0))

    # anchor slime prediction
    if row.get("anchor_slime_status", "") == "ok":
        draw_marker(draw, row["anchor_slime_pred_x"], row["anchor_slime_pred_y"], "diamond", (255, 255, 0))

    text = (
        f"{row['target_name']} | "
        f"YOLO={row.get('yolo_target_error_px', np.nan):.1f} "
        f"A-SLIME={row.get('anchor_slime_error_px', np.nan):.1f} "
        f"{row.get('anchor_slime_status', '')}"
    )
    draw.rectangle([8, 8, 700, 38], fill=(0, 0, 0))
    draw.text((12, 14), text, fill=(255, 255, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/stage4_anchor_slime/visualizations")
    parser.add_argument("--max_images", type=int, default=150)
    parser.add_argument("--only_valid", action="store_true")
    parser.add_argument("--only_improved", action="store_true")
    parser.add_argument("--hide_candidates", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.image_dir)

    df = pd.read_csv(args.results_csv)
    if args.only_valid:
        df = df[df["anchor_slime_status"] == "ok"].reset_index(drop=True)
    if args.only_improved and "yolo_target_error_px" in df.columns:
        df = df[(df["anchor_slime_status"] == "ok") & (df["anchor_slime_error_px"] < df["yolo_target_error_px"])].reset_index(drop=True)

    if len(df) == 0:
        print("No rows to visualize.")
        return

    sample = df.sample(n=min(args.max_images, len(df)), random_state=31)
    for _, row in sample.iterrows():
        image_path = image_dir / row["occluded_image_name"]
        out_path = out_dir / row["occluded_image_name"]
        visualize_one(image_path, row, out_path, show_candidates=not args.hide_candidates)

    print(f"Saved visualizations to: {out_dir}")


if __name__ == "__main__":
    main()
