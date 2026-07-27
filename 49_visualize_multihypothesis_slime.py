import argparse
from pathlib import Path

import cv2
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PATH_COLORS = [
    (255, 215, 0),    # rank 1 yellow
    (0, 200, 255),    # rank 2 cyan
    (255, 100, 180),  # rank 3 pink
    (120, 255, 120),  # rank 4 green
    (180, 130, 255),  # rank 5 purple
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
    elif kind == "filled_circle":
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
    elif kind == "diamond":
        pts = [(x, y - r), (x - r, y), (x, y + r), (x + r, y)]
        draw.polygon(pts, outline=color, fill=color)
    elif kind == "square":
        draw.rectangle([x - r, y - r, x + r, y + r], outline=color, width=width)
    elif kind == "dot":
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)


def draw_text_box(draw, xy, lines, fill=(255, 255, 255), outline=(180, 180, 180), text_fill=(0, 0, 0)):
    x, y = xy
    line_h = 22
    width = max(320, max(len(str(line)) for line in lines) * 7 + 20)
    height = line_h * len(lines) + 14
    draw.rectangle([x, y, x + width, y + height], fill=fill, outline=outline)
    yy = y + 8
    for line in lines:
        draw.text((x + 10, yy), str(line), fill=text_fill)
        yy += line_h


def visualize_one(image_path, sample_topk, all_cands_sample, out_path, presentation_clean=False, show_all_candidates=True):
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)

    first = sample_topk.sort_values("rank").iloc[0]

    ax, ay = first["anchor1_x"], first["anchor1_y"]
    bx, by = first["anchor2_x"], first["anchor2_y"]

    # Draw many candidate paths in faint gray for "slime exploration".
    if show_all_candidates and all_cands_sample is not None and len(all_cands_sample) > 0:
        show = all_cands_sample.sort_values("learned_score", ascending=False).head(90)
        for _, c in show.iterrows():
            draw.line(
                [ax, ay, c["candidate_x"], c["candidate_y"], bx, by],
                fill=(185, 185, 185),
                width=1,
            )

    # Coarse predicted region if available.
    if "coarse_region_x" in first.index and not pd.isna(first.get("coarse_region_x", None)):
        cx, cy = first["coarse_region_x"], first["coarse_region_y"]
        cell = first.get("coarse_cell_px", 64)
        r = float(cell) / 2.0
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], outline=(0, 180, 0), width=3)

    # Draw top-K paths.
    sample_topk = sample_topk.sort_values("rank")
    for _, row in sample_topk.iloc[::-1].iterrows():
        rank = int(row["rank"])
        color = PATH_COLORS[(rank - 1) % len(PATH_COLORS)]
        width = 5 if rank == 1 else 3
        draw.line(
            [ax, ay, row["candidate_x"], row["candidate_y"], bx, by],
            fill=color,
            width=width,
        )

    # Anchors
    draw_marker(draw, ax, ay, "circle", (0, 100, 255), r=9, width=4)
    draw_marker(draw, bx, by, "circle", (0, 100, 255), r=9, width=4)

    # GT hidden joint
    draw_marker(draw, first["target_orig_x"], first["target_orig_y"], "x", (255, 90, 0), r=10, width=5)

    # top-K points
    for _, row in sample_topk.iterrows():
        rank = int(row["rank"])
        color = PATH_COLORS[(rank - 1) % len(PATH_COLORS)]
        draw_marker(draw, row["candidate_x"], row["candidate_y"], "filled_circle", color, r=6 if rank > 1 else 8)
        draw.text((row["candidate_x"] + 8, row["candidate_y"] - 8), f"{rank}", fill=color)

    if not presentation_clean:
        # YOLO baseline if present
        if "yolo_target_x" in first.index and not pd.isna(first.get("yolo_target_x", None)):
            draw_marker(draw, first["yolo_target_x"], first["yolo_target_y"], "circle", (160, 0, 255), r=8, width=4)

        # exact region point for debugging only
        if "region_pred_x" in first.index and not pd.isna(first.get("region_pred_x", None)):
            draw_marker(draw, first["region_pred_x"], first["region_pred_y"], "diamond", (0, 220, 120), r=7)

    # Header
    if presentation_clean:
        title = "Multi-path slime hypotheses for occluded joint recovery"
        draw.rectangle([8, 8, 770, 42], fill=(255, 255, 255), outline=(170, 170, 170))
        draw.text((18, 17), title, fill=(0, 0, 0))
    else:
        title = (
            f"{first['target_name']} | "
            f"Top1={first['candidate_error_px']:.1f}px "
            f"P={first['hypothesis_probability']:.2f}"
        )
        draw.rectangle([8, 8, 900, 42], fill=(0, 0, 0))
        draw.text((18, 17), title, fill=(255, 255, 255))

    # Legend
    if presentation_clean:
        lines = [
            "gray: explored candidate paths",
            "yellow: selected path",
            "other colors: alternate hypotheses",
            "blue circles: visible anchors",
            "orange X: true hidden joint",
            "green box: coarse predicted region",
        ]
    else:
        lines = [
            "yellow = top-1 selected path",
            "cyan/pink/green/purple = alternate paths",
            "gray = explored candidates",
            "blue circles = visible anchors",
            "orange X = true hidden joint",
            "green box = coarse region",
            "purple circle = YOLO baseline",
        ]
    draw_text_box(draw, (15, 55), lines)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk_csv", type=str, required=True)
    parser.add_argument("--candidate_csv", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/stage5e_multihypothesis_slime/visualizations")
    parser.add_argument("--max_images", type=int, default=150)
    parser.add_argument("--only_improved_vs_yolo", action="store_true")
    parser.add_argument("--only_improved_vs_coarse", action="store_true")
    parser.add_argument("--presentation_clean", action="store_true")
    parser.add_argument("--hide_all_candidates", action="store_true")
    args = parser.parse_args()

    topk_csv = Path(args.topk_csv)
    candidate_csv = Path(args.candidate_csv)
    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    topk = pd.read_csv(topk_csv)
    cands = pd.read_csv(candidate_csv)

    # Filter based on rank 1 rows.
    top1 = topk[topk["rank"] == 1].copy()

    if args.only_improved_vs_yolo and "yolo_target_error_px" in top1.columns:
        top1 = top1[top1["candidate_error_px"] < top1["yolo_target_error_px"]].copy()

    if args.only_improved_vs_coarse and "coarse_region_error_px" in top1.columns:
        top1 = top1[top1["candidate_error_px"] < top1["coarse_region_error_px"]].copy()

    if len(top1) == 0:
        print("No rows to visualize after filtering.")
        return

    sample = top1.sample(n=min(args.max_images, len(top1)), random_state=52)

    for _, row in sample.iterrows():
        sid = row["sample_id"]
        one_topk = topk[topk["sample_id"] == sid].copy()
        one_cands = cands[cands["sample_id"] == sid].copy()

        image_path = image_dir / row["occluded_image_name"]
        stem = Path(row["occluded_image_name"]).stem
        out_path = out_dir / f"{stem}_topk.png"

        visualize_one(
            image_path=image_path,
            sample_topk=one_topk,
            all_cands_sample=one_cands,
            out_path=out_path,
            presentation_clean=args.presentation_clean,
            show_all_candidates=not args.hide_all_candidates,
        )

    print("Saved visualizations to:", out_dir)


if __name__ == "__main__":
    main()
