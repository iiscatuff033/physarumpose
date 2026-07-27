import argparse
from pathlib import Path

import cv2
import numpy as np
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

    elif kind == "dot":
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)


def visualize_one(image_path, row, cand_df, out_path, show_candidates=True, draw_occ_box=True):
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)

    # Optional: draw GT synthetic occlusion box for human inspection only.
    # It was NOT used by the Stage 5B model.
    if draw_occ_box:
        for c in ["occ_x1", "occ_y1", "occ_x2", "occ_y2"]:
            if c not in row.index:
                draw_occ_box = False
        if draw_occ_box:
            x1, y1, x2, y2 = row["occ_x1"], row["occ_y1"], row["occ_x2"], row["occ_y2"]
            if not any(pd.isna(v) for v in [x1, y1, x2, y2]):
                draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)

    # Candidate cloud
    if show_candidates and cand_df is not None and len(cand_df) > 0:
        step = max(1, len(cand_df) // 120)
        for _, c in cand_df.iloc[::step].iterrows():
            draw_marker(draw, c["candidate_x"], c["candidate_y"], "dot", (150, 150, 150), r=2)

    # Anchors and path
    ax, ay = row["anchor1_x"], row["anchor1_y"]
    bx, by = row["anchor2_x"], row["anchor2_y"]
    lx, ly = row["learned_x"], row["learned_y"]

    draw_marker(draw, ax, ay, "dot", (0, 120, 255), r=7)
    draw_marker(draw, bx, by, "dot", (0, 120, 255), r=7)

    draw.line([ax, ay, lx, ly], fill=(255, 255, 0), width=3)
    draw.line([lx, ly, bx, by], fill=(255, 255, 0), width=3)

    # True hidden joint
    draw_marker(draw, row["target_orig_x"], row["target_orig_y"], "x", (255, 165, 0), r=8)

    # YOLO baseline if available
    if "yolo_target_x" in row.index:
        draw_marker(draw, row["yolo_target_x"], row["yolo_target_y"], "circle", (160, 0, 255), r=8)

    # Heuristic candidate
    draw_marker(draw, row["heuristic_x"], row["heuristic_y"], "triangle", (0, 255, 0), r=8)

    # Learned candidate
    draw_marker(draw, row["learned_x"], row["learned_y"], "diamond", (255, 255, 0), r=8)

    text = (
        f"{row['target_name']} | "
        f"YOLO={row.get('yolo_target_error_px', np.nan):.1f} "
        f"HEUR={row['heuristic_error_px']:.1f} "
        f"LEARN={row['learned_error_px']:.1f}"
    )
    draw.rectangle([8, 8, 720, 38], fill=(0, 0, 0))
    draw.text((12, 14), text, fill=(255, 255, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", type=str, required=True)
    parser.add_argument("--candidate_csv", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/stage5b_no_occ_slime_scorer/visualizations")
    parser.add_argument("--max_images", type=int, default=150)
    parser.add_argument("--only_improved", action="store_true")
    parser.add_argument("--hide_candidates", action="store_true")
    parser.add_argument("--hide_occ_box", action="store_true")
    args = parser.parse_args()

    results_csv = Path(args.results_csv)
    candidate_csv = Path(args.candidate_csv)
    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best = pd.read_csv(results_csv)
    cands = pd.read_csv(candidate_csv)

    if args.only_improved and "yolo_target_error_px" in best.columns:
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
            image_path,
            row,
            one_cands,
            out_path,
            show_candidates=not args.hide_candidates,
            draw_occ_box=not args.hide_occ_box,
        )

    print(f"Saved visualizations to: {out_dir}")


if __name__ == "__main__":
    main()
