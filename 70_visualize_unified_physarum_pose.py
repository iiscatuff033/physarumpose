import argparse
from pathlib import Path

import cv2
import pandas as pd
from PIL import Image, ImageDraw


PATH_COLORS = [
    (255, 215, 0),
    (0, 200, 255),
    (255, 100, 180),
    (120, 255, 120),
    (180, 130, 255),
]


def draw_marker(draw, x, y, kind, color, r=7, width=4):
    if pd.isna(x) or pd.isna(y):
        return

    x = int(round(float(x)))
    y = int(round(float(y)))

    if kind == "x":
        draw.line([x - r, y - r, x + r, y + r], fill=color, width=width)
        draw.line([x - r, y + r, x + r, y - r], fill=color, width=width)
    elif kind == "circle":
        draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=width)
    elif kind == "filled":
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
    elif kind == "diamond":
        pts = [(x, y - r), (x - r, y), (x, y + r), (x + r, y)]
        draw.polygon(pts, outline=color, fill=color)
    elif kind == "square":
        draw.rectangle([x - r, y - r, x + r, y + r], outline=color, width=width)


def visualize_one(image_path, row, topk_rows, out_path, show_title=False):
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil_full = Image.fromarray(rgb)

    x1, y1, x2, y2 = int(row["crop_x1"]), int(row["crop_y1"]), int(row["crop_x2"]), int(row["crop_y2"])
    pil = pil_full.crop((x1, y1, x2, y2))
    draw = ImageDraw.Draw(pil)

    def sx(x): return float(x) - x1
    def sy(y): return float(y) - y1

    topk_rows = topk_rows.copy()
    for col in ["candidate_x", "anchor1_x", "anchor2_x"]:
        if col in topk_rows.columns:
            topk_rows[col] = topk_rows[col] - x1
    for col in ["candidate_y", "anchor1_y", "anchor2_y"]:
        if col in topk_rows.columns:
            topk_rows[col] = topk_rows[col] - y1

    # Draw top-K paths.
    if len(topk_rows) > 0:
        topk_rows = topk_rows.sort_values("rank")
        first = topk_rows.iloc[0]
        ax, ay = first["anchor1_x"], first["anchor1_y"]
        bx, by = first["anchor2_x"], first["anchor2_y"]

        for _, r in topk_rows.iloc[::-1].iterrows():
            rank = int(r["rank"])
            color = PATH_COLORS[(rank - 1) % len(PATH_COLORS)]
            width = 5 if rank == 1 else 3
            draw.line([ax, ay, r["candidate_x"], r["candidate_y"], bx, by], fill=color, width=width)
            draw_marker(draw, r["candidate_x"], r["candidate_y"], "filled", color, r=5)

        draw_marker(draw, ax, ay, "circle", (0, 100, 255), r=9)
        draw_marker(draw, bx, by, "circle", (0, 100, 255), r=9)

    # Region head and unified soft output.
    draw_marker(draw, sx(row["unified_region_x"]), sy(row["unified_region_y"]), "circle", (0, 220, 120), r=8)
    draw_marker(draw, sx(row["unified_soft_x"]), sy(row["unified_soft_y"]), "diamond", (255, 215, 0), r=10, width=5)

    # GT target.
    draw_marker(draw, sx(row["target_orig_x"]), sy(row["target_orig_y"]), "x", (255, 90, 0), r=10, width=5)

    # Optional previous staged gated output for comparison.
    if "gated_x" in row.index and not pd.isna(row["gated_x"]):
        draw_marker(draw, sx(row["gated_x"]), sy(row["gated_y"]), "square", (255, 0, 0), r=8, width=4)

    if show_title:
        title = f"{row['target_name']} | unified soft={row['unified_soft_error_px']:.1f}px | top1={row['unified_top1_error_px']:.1f}px"
        draw.rectangle([8, 8, min(pil.size[0] - 8, 820), 42], fill=(255, 255, 255), outline=(170, 170, 170))
        draw.text((18, 17), title, fill=(0, 0, 0))

    # No legend. User requested legends removed completely.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", type=str, required=True)
    parser.add_argument("--topk_csv", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/stage6a_unified_physarum_pose/eval/visualizations")
    parser.add_argument("--max_images", type=int, default=150)
    parser.add_argument("--only_good", action="store_true")
    parser.add_argument("--only_failures", action="store_true")
    parser.add_argument("--only_improved_vs_yolo", action="store_true")
    parser.add_argument("--show_title", action="store_true")
    args = parser.parse_args()

    pred = pd.read_csv(args.pred_csv)
    topk = pd.read_csv(args.topk_csv)

    if args.only_good:
        # Clean presentation examples: good error and stable anchors.
        pred = pred[
            (pred["unified_soft_error_px"] <= 20)
            & (pred["unified_anchor_pair_mean_error_px"] <= 60)
        ].reset_index(drop=True)

    if args.only_failures:
        pred = pred[pred["unified_soft_error_px"] > 40].reset_index(drop=True)

    if args.only_improved_vs_yolo and "yolo_target_error_px" in pred.columns:
        pred = pred[pred["unified_soft_error_px"] < pred["yolo_target_error_px"]].reset_index(drop=True)

    if len(pred) == 0:
        print("No rows to visualize after filtering.")
        return

    if args.only_good:
        sample = pred.sort_values("unified_soft_error_px").head(args.max_images)
    else:
        sample = pred.sample(n=min(args.max_images, len(pred)), random_state=606)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.image_dir)

    for _, row in sample.iterrows():
        sid = row["sample_id"]
        one_topk = topk[topk["sample_id"] == sid].sort_values("rank")
        image_path = image_dir / row["occluded_image_name"]
        stem = Path(row["occluded_image_name"]).stem
        out_path = out_dir / f"{stem}_unified_physarum.png"
        visualize_one(
            image_path=image_path,
            row=row,
            topk_rows=one_topk,
            out_path=out_path,
            show_title=args.show_title,
        )

    print("Saved visualizations to:", out_dir)


if __name__ == "__main__":
    main()
