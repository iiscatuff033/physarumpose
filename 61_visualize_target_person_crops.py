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


def shift(v, offset):
    try:
        return float(v) - offset
    except Exception:
        return v


def visualize_one(image_path, row, topk_rows, out_path, presentation_clean=False, draw_gt=True):
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    x1, y1, x2, y2 = int(row["crop_x1"]), int(row["crop_y1"]), int(row["crop_x2"]), int(row["crop_y2"])
    crop = pil.crop((x1, y1, x2, y2))
    draw = ImageDraw.Draw(crop)

    # Shift current row coordinates into crop coordinates.
    def sx(v): return shift(v, x1)
    def sy(v): return shift(v, y1)

    # Shift topk rows.
    topk_rows = topk_rows.copy()
    for col in ["candidate_x", "anchor1_x", "anchor2_x"]:
        if col in topk_rows.columns:
            topk_rows[col] = topk_rows[col] - x1
    for col in ["candidate_y", "anchor1_y", "anchor2_y"]:
        if col in topk_rows.columns:
            topk_rows[col] = topk_rows[col] - y1

    # Draw top-K candidate paths.
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

    # Region head
    draw_marker(draw, sx(row["teacher_region_x"]), sy(row["teacher_region_y"]), "circle", (0, 220, 120), r=8)

    # Final gated prediction
    if row["gate_source"] == "top1_slime_path":
        final_color = (255, 215, 0)
    else:
        final_color = (0, 220, 120)
    draw_marker(draw, sx(row["gated_x"]), sy(row["gated_y"]), "diamond", final_color, r=10, width=5)

    # GT hidden joint for evaluation figure.
    if draw_gt:
        draw_marker(draw, sx(row["target_orig_x"]), sy(row["target_orig_y"]), "x", (255, 90, 0), r=10, width=5)

    # YOLO baseline if wanted
    if not presentation_clean and "yolo_target_x" in row.index:
        draw_marker(draw, sx(row["yolo_target_x"]), sy(row["yolo_target_y"]), "circle", (160, 0, 255), r=8, width=4)

    # Header
    if presentation_clean:
        title = "Cropped target-person view: multi-path slime recovery"
        draw.rectangle([8, 8, min(crop.size[0] - 8, 740), 42], fill=(255, 255, 255), outline=(170, 170, 170))
        draw.text((18, 17), title, fill=(0, 0, 0))
    else:
        title = (
            f"{row['target_name']} | {row['gate_source']} | "
            f"gated={row.get('gated_error_px', float('nan')):.1f}px | "
            f"suspicious={int(row['suspicious_case'])}"
        )
        draw.rectangle([8, 8, min(crop.size[0] - 8, 1000), 42], fill=(0, 0, 0))
        draw.text((18, 17), title, fill=(255, 255, 255))

    # Legend
    if presentation_clean:
        lines = [
            "yellow: selected slime path",
            "blue circles: anchors",
            "green circle: region fallback point",
            "diamond: final gated output",
            "orange X: true hidden joint",
        ]
    else:
        lines = [
            "yellow: top-1 slime path",
            "other colors: alternate top-K paths",
            "blue circles: anchors",
            "green circle: region head",
            "diamond: gated final",
            "orange X: true hidden joint",
            "purple: YOLO",
        ]

    x0, y0 = 12, max(52, crop.size[1] - (24 * len(lines) + 22))
    draw.rectangle([x0 - 6, y0 - 6, x0 + 420, y0 + 24 * len(lines) + 6], fill=(255, 255, 255), outline=(160, 160, 160))
    yy = y0
    for line in lines:
        draw.text((x0, yy), line, fill=(0, 0, 0))
        yy += 24

    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop_csv", type=str, required=True)
    parser.add_argument("--topk_csv", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/stage5g_target_person_crop_filter/visualizations")
    parser.add_argument("--max_images", type=int, default=200)
    parser.add_argument("--presentation_clean", action="store_true")
    parser.add_argument("--only_good", action="store_true")
    parser.add_argument("--only_suspicious", action="store_true")
    parser.add_argument("--only_region_fallback", action="store_true")
    parser.add_argument("--hide_gt", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.crop_csv)
    topk = pd.read_csv(args.topk_csv)

    if args.only_good:
        df = df[df["good_paper_example"] == True].reset_index(drop=True)

    if args.only_suspicious:
        df = df[df["suspicious_case"] == True].reset_index(drop=True)

    if args.only_region_fallback:
        df = df[df["gate_source"] == "region_fallback"].reset_index(drop=True)

    if len(df) == 0:
        print("No rows after filtering.")
        return

    # Good examples first for presentation.
    if args.only_good:
        sample = df.sort_values("gated_error_px").head(args.max_images)
    else:
        sample = df.sample(n=min(args.max_images, len(df)), random_state=151)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.image_dir)

    for _, row in sample.iterrows():
        sid = row["sample_id"]
        one_topk = topk[topk["sample_id"] == sid].sort_values("rank")
        image_path = image_dir / row["occluded_image_name"]
        stem = Path(row["occluded_image_name"]).stem
        suffix = "good" if row.get("good_paper_example", False) else "case"
        out_path = out_dir / f"{stem}_{suffix}_crop.png"
        visualize_one(
            image_path=image_path,
            row=row,
            topk_rows=one_topk,
            out_path=out_path,
            presentation_clean=args.presentation_clean,
            draw_gt=not args.hide_gt,
        )

    print("Saved visualizations to:", out_dir)


if __name__ == "__main__":
    main()
