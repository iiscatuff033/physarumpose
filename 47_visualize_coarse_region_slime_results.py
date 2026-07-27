import argparse
from pathlib import Path

import cv2
import pandas as pd
from PIL import Image, ImageDraw


def draw_marker(draw, x, y, kind, color, r=7):
    if pd.isna(x) or pd.isna(y):
        return

    x = int(round(float(x)))
    y = int(round(float(y)))

    if kind == "x":
        draw.line([x - r, y - r, x + r, y + r], fill=color, width=4)
        draw.line([x - r, y + r, x + r, y - r], fill=color, width=4)
    elif kind == "circle":
        draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=4)
    elif kind == "diamond":
        pts = [(x, y - r), (x - r, y), (x, y + r), (x + r, y)]
        draw.polygon(pts, outline=color, fill=color)
    elif kind == "triangle":
        pts = [(x, y - r), (x - r, y + r), (x + r, y + r)]
        draw.polygon(pts, outline=color, fill=color)
    elif kind == "square":
        draw.rectangle([x - r, y - r, x + r, y + r], outline=color, width=4)
    elif kind == "dot":
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)


def visualize_one(image_path, row, cand_df, out_path, show_candidates=True):
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)

    # candidate cloud
    if show_candidates and cand_df is not None and len(cand_df) > 0:
        # draw lower scored paths as thin gray-blue lines
        top = cand_df.sort_values("learned_score", ascending=False).head(80)
        for _, c in top.iterrows():
            draw.line(
                [row["anchor1_x"], row["anchor1_y"], c["candidate_x"], c["candidate_y"], row["anchor2_x"], row["anchor2_y"]],
                fill=(150, 170, 210),
                width=1,
            )
            draw_marker(draw, c["candidate_x"], c["candidate_y"], "dot", (120, 120, 120), r=2)

    # coarse region box/circle
    cx = row["coarse_region_x"]
    cy = row["coarse_region_y"]
    cell = row.get("coarse_cell_px", 64)
    if not pd.isna(cx) and not pd.isna(cy):
        r = float(cell) / 2.0
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], outline=(0, 180, 0), width=3)

    # selected learned path
    lx = row["learned_x"]
    ly = row["learned_y"]
    draw.line(
        [row["anchor1_x"], row["anchor1_y"], lx, ly, row["anchor2_x"], row["anchor2_y"]],
        fill=(255, 205, 0),
        width=5,
    )

    # nearest coarse path baseline
    nx = row["nearest_coarse_x"]
    ny = row["nearest_coarse_y"]
    draw.line(
        [row["anchor1_x"], row["anchor1_y"], nx, ny, row["anchor2_x"], row["anchor2_y"]],
        fill=(255, 150, 0),
        width=2,
    )

    # anchors
    draw_marker(draw, row["anchor1_x"], row["anchor1_y"], "circle", (0, 120, 255), r=8)
    draw_marker(draw, row["anchor2_x"], row["anchor2_y"], "circle", (0, 120, 255), r=8)

    # exact predicted region for inspection only
    if "region_pred_x" in row.index:
        draw_marker(draw, row["region_pred_x"], row["region_pred_y"], "circle", (0, 220, 120), r=7)

    # true joint and predictions
    draw_marker(draw, row["target_orig_x"], row["target_orig_y"], "x", (255, 90, 0), r=9)
    draw_marker(draw, row["coarse_region_x"], row["coarse_region_y"], "square", (0, 180, 0), r=7)
    draw_marker(draw, row["nearest_coarse_x"], row["nearest_coarse_y"], "triangle", (255, 150, 0), r=8)
    draw_marker(draw, row["learned_x"], row["learned_y"], "diamond", (255, 205, 0), r=8)

    if "yolo_target_x" in row.index:
        draw_marker(draw, row["yolo_target_x"], row["yolo_target_y"], "circle", (160, 0, 255), r=8)

    text = (
        f"{row['target_name']} | "
        f"YOLO={row.get('yolo_target_error_px', float('nan')):.1f} "
        f"ExactRegion={row.get('region_error_px', float('nan')):.1f} "
        f"Coarse={row['coarse_region_error_px']:.1f} "
        f"Nearest={row['nearest_coarse_error_px']:.1f} "
        f"Learned={row['learned_error_px']:.1f}"
    )

    draw.rectangle([8, 8, 1050, 42], fill=(0, 0, 0))
    draw.text((12, 16), text, fill=(255, 255, 255))

    # Legend
    lx0, ly0 = 15, 55
    legend = [
        ("blue circles", "anchors"),
        ("green box", "coarse predicted region"),
        ("thin paths", "top candidate paths"),
        ("yellow path", "selected learned slime path"),
        ("orange triangle", "nearest-to-coarse candidate"),
        ("orange X", "true hidden joint"),
        ("purple circle", "YOLO baseline"),
    ]
    draw.rectangle([lx0 - 8, ly0 - 6, lx0 + 390, ly0 + 170], fill=(255, 255, 255), outline=(160, 160, 160))
    y = ly0
    for label, desc in legend:
        draw.text((lx0, y), f"{label}: {desc}", fill=(0, 0, 0))
        y += 22

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", type=str, required=True)
    parser.add_argument("--candidate_csv", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/stage5d_coarse_region_slime_scorer/visualizations")
    parser.add_argument("--max_images", type=int, default=150)
    parser.add_argument("--only_improved_vs_coarse", action="store_true")
    parser.add_argument("--only_improved_vs_yolo", action="store_true")
    parser.add_argument("--hide_candidates", action="store_true")
    args = parser.parse_args()

    results_csv = Path(args.results_csv)
    candidate_csv = Path(args.candidate_csv)
    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best = pd.read_csv(results_csv)
    cands = pd.read_csv(candidate_csv)

    if args.only_improved_vs_coarse:
        best = best[best["learned_error_px"] < best["coarse_region_error_px"]].reset_index(drop=True)

    if args.only_improved_vs_yolo and "yolo_target_error_px" in best.columns:
        best = best[best["learned_error_px"] < best["yolo_target_error_px"]].reset_index(drop=True)

    if len(best) == 0:
        print("No rows to visualize.")
        return

    sample = best.sample(n=min(args.max_images, len(best)), random_state=31)

    for _, row in sample.iterrows():
        image_path = image_dir / row["occluded_image_name"]
        one_cands = cands[cands["sample_id"] == row["sample_id"]]
        out_path = out_dir / row["occluded_image_name"]
        visualize_one(
            image_path=image_path,
            row=row,
            cand_df=one_cands,
            out_path=out_path,
            show_candidates=not args.hide_candidates,
        )

    print("Saved visualizations to:", out_dir)


if __name__ == "__main__":
    main()
