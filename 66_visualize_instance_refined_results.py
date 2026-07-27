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


def crop_box(row, img_w, img_h, pad=120):
    pts = [
        (row.get("instance_refined_anchor1_x"), row.get("instance_refined_anchor1_y")),
        (row.get("instance_refined_anchor2_x"), row.get("instance_refined_anchor2_y")),
        (row.get("instance_refined_top1_x"), row.get("instance_refined_top1_y")),
        (row.get("instance_refined_region_x"), row.get("instance_refined_region_y")),
        (row.get("instance_refined_gated_x"), row.get("instance_refined_gated_y")),
        (row.get("target_orig_x"), row.get("target_orig_y")),
    ]

    xs, ys = [], []
    for x, y in pts:
        try:
            x, y = float(x), float(y)
            xs.append(x)
            ys.append(y)
        except Exception:
            pass

    if not xs:
        return 0, 0, img_w, img_h

    x1 = max(0, int(min(xs) - pad))
    y1 = max(0, int(min(ys) - pad))
    x2 = min(img_w, int(max(xs) + pad))
    y2 = min(img_h, int(max(ys) + pad))

    if x2 - x1 < 260:
        cx = (x1 + x2) // 2
        x1 = max(0, cx - 130)
        x2 = min(img_w, x1 + 260)
        x1 = max(0, x2 - 260)

    if y2 - y1 < 260:
        cy = (y1 + y2) // 2
        y1 = max(0, cy - 130)
        y2 = min(img_h, y1 + 260)
        y1 = max(0, y2 - 260)

    return x1, y1, x2, y2


def visualize_one(image_path, row, topk_rows, out_path, presentation_clean=False):
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    img_w, img_h = pil.size

    x1, y1, x2, y2 = crop_box(row, img_w, img_h, pad=120)
    pil = pil.crop((x1, y1, x2, y2))
    draw = ImageDraw.Draw(pil)

    def sx(x): return float(x) - x1
    def sy(y): return float(y) - y1

    topk_rows = topk_rows.copy()
    for c in ["candidate_x", "anchor1_x", "anchor2_x"]:
        if c in topk_rows.columns:
            topk_rows[c] = topk_rows[c] - x1
    for c in ["candidate_y", "anchor1_y", "anchor2_y"]:
        if c in topk_rows.columns:
            topk_rows[c] = topk_rows[c] - y1

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
            if not presentation_clean:
                draw.text((r["candidate_x"] + 6, r["candidate_y"] - 6), str(rank), fill=color)

        draw_marker(draw, ax, ay, "circle", (0, 100, 255), r=9)
        draw_marker(draw, bx, by, "circle", (0, 100, 255), r=9)

    # Previous gated output for comparison
    if not presentation_clean and "gated_x" in row.index:
        draw_marker(draw, sx(row["gated_x"]), sy(row["gated_y"]), "square", (255, 0, 0), r=8, width=4)

    # Region and final
    draw_marker(draw, sx(row["instance_refined_region_x"]), sy(row["instance_refined_region_y"]), "circle", (0, 220, 120), r=8)
    draw_marker(draw, sx(row["instance_refined_gated_x"]), sy(row["instance_refined_gated_y"]), "diamond", (255, 215, 0), r=10, width=5)

    # GT target
    draw_marker(draw, sx(row["target_orig_x"]), sy(row["target_orig_y"]), "x", (255, 90, 0), r=10, width=5)

    if presentation_clean:
        title = "Instance-aware anchor refinement + slime recovery"
        draw.rectangle([8, 8, min(pil.size[0] - 8, 760), 42], fill=(255, 255, 255), outline=(170, 170, 170))
        draw.text((18, 17), title, fill=(0, 0, 0))
    else:
        prev = row.get("gated_error_px", float("nan"))
        title = (
            f"{row['target_name']} | "
            f"refined={row['instance_refined_gated_error_px']:.1f}px | "
            f"previous={prev:.1f}px | "
            f"source={row['instance_refined_gate_source']}"
        )
        draw.rectangle([8, 8, min(pil.size[0] - 8, 1040), 42], fill=(0, 0, 0))
        draw.text((18, 17), title, fill=(255, 255, 255))
    if not presentation_clean:
        lines = [
            "yellow path/diamond: instance-refined slime output",
            "blue circles: refined anchor pair",
            "green circle: region head",
            "orange X: true hidden joint",
            "red square: previous gated output",
        ]

        x0, y0 = 12, max(52, pil.size[1] - (24 * len(lines) + 22))
        draw.rectangle(
            [x0 - 6, y0 - 6, x0 + 455, y0 + 24 * len(lines) + 6],
            fill=(255, 255, 255),
            outline=(160, 160, 160)
        )
        yy = y0
        for line in lines:
            draw.text((x0, yy), line, fill=(0, 0, 0))
            yy += 24        
  


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", type=str, required=True)
    parser.add_argument("--topk_csv", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/stage5h_instance_refined_soft_slime/visualizations")
    parser.add_argument("--max_images", type=int, default=150)
    parser.add_argument("--presentation_clean", action="store_true")
    parser.add_argument("--only_good", action="store_true")
    parser.add_argument("--only_failures", action="store_true")
    parser.add_argument("--only_improved", action="store_true")
    args = parser.parse_args()

    pred = pd.read_csv(args.pred_csv)
    topk = pd.read_csv(args.topk_csv)

    if args.only_good:
        pred = pred[pred["instance_refined_gated_error_px"] <= 20].reset_index(drop=True)
    if args.only_failures:
        pred = pred[pred["instance_refined_gated_error_px"] > 40].reset_index(drop=True)
    if args.only_improved and "gated_error_px" in pred.columns:
        pred = pred[pred["instance_refined_gated_error_px"] < pred["gated_error_px"]].reset_index(drop=True)

    if len(pred) == 0:
        print("No rows to visualize.")
        return

    if args.only_good:
        sample = pred.sort_values("instance_refined_gated_error_px").head(args.max_images)
    else:
        sample = pred.sample(n=min(args.max_images, len(pred)), random_state=202)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.image_dir)

    for _, row in sample.iterrows():
        sid = row["sample_id"]
        one_topk = topk[topk["sample_id"] == sid].sort_values("rank")
        image_path = image_dir / row["occluded_image_name"]
        stem = Path(row["occluded_image_name"]).stem
        out_path = out_dir / f"{stem}_instance_refined.png"
        visualize_one(image_path, row, one_topk, out_path, presentation_clean=args.presentation_clean)

    print("Saved visualizations to:", out_dir)


if __name__ == "__main__":
    main()
