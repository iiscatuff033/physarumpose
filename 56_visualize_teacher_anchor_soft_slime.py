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


def visualize_one(image_path, pred_row, topk_rows, out_path, presentation_clean=False):
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)

    if len(topk_rows) > 0:
        first = topk_rows.iloc[0]
        ax, ay = first["anchor1_x"], first["anchor1_y"]
        bx, by = first["anchor2_x"], first["anchor2_y"]

        # Top-k paths
        for _, row in topk_rows.iloc[::-1].iterrows():
            rank = int(row["rank"])
            color = PATH_COLORS[(rank - 1) % len(PATH_COLORS)]
            width = 5 if rank == 1 else 3
            draw.line([ax, ay, row["candidate_x"], row["candidate_y"], bx, by], fill=color, width=width)
            draw_marker(draw, row["candidate_x"], row["candidate_y"], "filled", color, r=6)
            draw.text((row["candidate_x"] + 6, row["candidate_y"] - 6), str(rank), fill=color)

        draw_marker(draw, ax, ay, "circle", (0, 100, 255), r=9)
        draw_marker(draw, bx, by, "circle", (0, 100, 255), r=9)

    # True joint
    draw_marker(draw, pred_row["target_orig_x"], pred_row["target_orig_y"], "x", (255, 90, 0), r=10, width=5)

    # Teacher-anchor soft prediction
    draw_marker(draw, pred_row["teacher_anchor_soft_x"], pred_row["teacher_anchor_soft_y"], "diamond", (255, 215, 0), r=9)

    # Region
    draw_marker(draw, pred_row["teacher_region_x"], pred_row["teacher_region_y"], "circle", (0, 220, 120), r=7)

    if not presentation_clean and "yolo_target_x" in pred_row.index:
        draw_marker(draw, pred_row["yolo_target_x"], pred_row["yolo_target_y"], "circle", (160, 0, 255), r=8)

    if presentation_clean:
        text = "Teacher-anchor differentiable soft-slime path aggregation"
        draw.rectangle([8, 8, 860, 42], fill=(255, 255, 255), outline=(170, 170, 170))
        draw.text((18, 17), text, fill=(0, 0, 0))
    else:
        text = (
            f"{pred_row['target_name']} | "
            f"Soft={pred_row['teacher_anchor_soft_error_px']:.1f}px "
            f"Region={pred_row['teacher_region_error_px']:.1f}px "
            f"Top1={pred_row['top1_candidate_error_px']:.1f}px"
        )
        draw.rectangle([8, 8, 950, 42], fill=(0, 0, 0))
        draw.text((18, 17), text, fill=(255, 255, 255))

    lines = [
        "yellow diamond/path: soft-slime output / top path",
        "cyan/pink/green/purple: alternate high-probability paths",
        "blue circles: Strong AnchorNet teacher anchors",
        "green circle: target-region heatmap point",
        "orange X: true hidden joint",
    ]
    if not presentation_clean:
        lines.append("purple circle: YOLO baseline")

    x0, y0 = 15, 55
    draw.rectangle([x0 - 8, y0 - 6, x0 + 540, y0 + 25 * len(lines) + 8], fill=(255, 255, 255), outline=(160, 160, 160))
    yy = y0
    for line in lines:
        draw.text((x0, yy), line, fill=(0, 0, 0))
        yy += 25

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", type=str, required=True)
    parser.add_argument("--topk_csv", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/stage5fb_teacher_anchor_soft_slime/eval/visualizations")
    parser.add_argument("--max_images", type=int, default=150)
    parser.add_argument("--presentation_clean", action="store_true")
    parser.add_argument("--only_improved_vs_yolo", action="store_true")
    args = parser.parse_args()

    pred = pd.read_csv(args.pred_csv)
    topk = pd.read_csv(args.topk_csv)

    if args.only_improved_vs_yolo and "yolo_target_error_px" in pred.columns:
        pred = pred[pred["teacher_anchor_soft_error_px"] < pred["yolo_target_error_px"]].reset_index(drop=True)

    if len(pred) == 0:
        print("No rows to visualize.")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.image_dir)

    sample = pred.sample(n=min(args.max_images, len(pred)), random_state=87)

    for _, row in sample.iterrows():
        sid = row["sample_id"]
        one_topk = topk[topk["sample_id"] == sid].sort_values("rank")
        image_path = image_dir / row["occluded_image_name"]
        stem = Path(row["occluded_image_name"]).stem
        out_path = out_dir / f"{stem}_teacher_anchor_soft_slime.png"
        visualize_one(image_path, row, one_topk, out_path, presentation_clean=args.presentation_clean)

    print("Saved visualizations to:", out_dir)


if __name__ == "__main__":
    main()
