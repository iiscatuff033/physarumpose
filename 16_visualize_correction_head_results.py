import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def draw_marker(draw, x, y, kind, color, r=7):
    if np.isnan(x) or np.isnan(y):
        return

    x = int(round(x))
    y = int(round(y))

    if kind == "x":
        draw.line([x - r, y - r, x + r, y + r], fill=color, width=4)
        draw.line([x - r, y + r, x + r, y - r], fill=color, width=4)

    elif kind == "circle":
        draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=4)

    elif kind == "square":
        draw.rectangle([x - r, y - r, x + r, y + r], outline=color, width=4)

    elif kind == "triangle":
        tri = [(x, y - r), (x - r, y + r), (x + r, y + r)]
        draw.polygon(tri, outline=color, fill=color)

    elif kind == "diamond":
        dia = [(x, y - r), (x - r, y), (x, y + r), (x + r, y)]
        draw.polygon(dia, outline=color, fill=color)


def visualize_one(image_path, row, out_path):
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)

    # occlusion box
    x1, y1, x2, y2 = int(row["occ_x1"]), int(row["occ_y1"]), int(row["occ_x2"]), int(row["occ_y2"])
    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)

    # true hidden joint
    draw_marker(draw, row["target_orig_x"], row["target_orig_y"], "x", (255, 165, 0))

    # YOLO
    draw_marker(draw, row.get("yolo_target_x", np.nan), row.get("yolo_target_y", np.nan), "circle", (160, 0, 255))

    # geometry
    draw_marker(draw, row.get("geometry_x", np.nan), row.get("geometry_y", np.nan), "square", (0, 255, 255))

    # prior
    draw_marker(draw, row.get("prior_pred_x", np.nan), row.get("prior_pred_y", np.nan), "triangle", (0, 255, 0))

    # correction head final
    draw_marker(draw, row.get("corr_pred_x", np.nan), row.get("corr_pred_y", np.nan), "diamond", (255, 255, 0))

    text = (
        f"{row['target_name']} | "
        f"YOLO={row.get('yolo_target_error_px', np.nan):.1f} "
        f"PRIOR={row.get('prior_error_px', np.nan):.1f} "
        f"CORR={row.get('corr_error_px', np.nan):.1f}"
    )

    draw.rectangle([8, 8, 560, 38], fill=(0, 0, 0))
    draw.text((12, 14), text, fill=(255, 255, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/stage2c_correction_head/visualizations")
    parser.add_argument("--max_images", type=int, default=150)
    parser.add_argument("--only_improved", action="store_true")
    args = parser.parse_args()

    results_csv = Path(args.results_csv)
    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_csv)

    if args.only_improved:
        df = df[df["corr_error_px"] < df["yolo_target_error_px"]].reset_index(drop=True)

    if len(df) == 0:
        print("No rows to visualize.")
        return

    sample = df.sample(n=min(args.max_images, len(df)), random_state=13)

    for _, row in sample.iterrows():
        image_path = image_dir / row["occluded_image_name"]
        out_path = out_dir / row["occluded_image_name"]
        visualize_one(image_path, row, out_path)

    print(f"Saved visualizations to: {out_dir}")


if __name__ == "__main__":
    main()
