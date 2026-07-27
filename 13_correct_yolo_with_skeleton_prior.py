import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from src.full_context_model import FullSkeletonPriorMLP
from src.utils_full import (
    TARGET_NAME_TO_TASK,
    TARGET_JOINTS,
    MPII_TO_COCO,
    ENDPOINT_MPII,
    load_json,
    one_hot,
)


def compute_center_scale_from_yolo(row, conf_thresh=0.25):
    xy = []

    # Use overlapping joints only.
    for mpii_id, coco_id in MPII_TO_COCO.items():
        conf = float(row.get(f"coco{coco_id}_conf", 0.0))
        x = float(row.get(f"coco{coco_id}_x", 0.0))
        y = float(row.get(f"coco{coco_id}_y", 0.0))

        if conf >= conf_thresh and x > 0 and y > 0:
            xy.append([x, y])

    if len(xy) < 5:
        return None, None

    xy = np.array(xy, dtype=np.float32)
    center = xy.mean(axis=0)
    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)

    width, height = max_xy - min_xy
    scale = max(float(width), float(height), 1.0)

    return center, scale


def norm_xy(x, y, center, scale):
    arr = np.array([x, y], dtype=np.float32)
    arr = (arr - center) / scale
    return float(arr[0]), float(arr[1])


def denorm_xy(nx, ny, center, scale):
    arr = np.array([nx, ny], dtype=np.float32) * scale + center
    return float(arr[0]), float(arr[1])


def build_prior_input_from_yolo(row, center, scale, target_name, conf_thresh=0.25, num_joints=16, num_targets=4):
    target_task_id = TARGET_NAME_TO_TASK[target_name]
    target_mpii_id = TARGET_JOINTS[target_task_id]["mpii_id"]

    joint_arr = [[0.0, 0.0, 0.0] for _ in range(num_joints)]

    for mpii_id, coco_id in MPII_TO_COCO.items():
        conf = float(row.get(f"coco{coco_id}_conf", 0.0))
        x = float(row.get(f"coco{coco_id}_x", 0.0))
        y = float(row.get(f"coco{coco_id}_y", 0.0))

        if conf >= conf_thresh and x > 0 and y > 0:
            nx, ny = norm_xy(x, y, center, scale)
            joint_arr[mpii_id] = [nx, ny, 1.0]

    # Add pelvis as average of hips if available.
    # MPII pelvis id = 6. COCO left hip=11, right hip=12
    hip_pts = []
    for coco_id in [11, 12]:
        conf = float(row.get(f"coco{coco_id}_conf", 0.0))
        x = float(row.get(f"coco{coco_id}_x", 0.0))
        y = float(row.get(f"coco{coco_id}_y", 0.0))
        if conf >= conf_thresh and x > 0 and y > 0:
            hip_pts.append([x, y])

    if len(hip_pts) == 2:
        hx, hy = np.mean(np.array(hip_pts), axis=0)
        nx, ny = norm_xy(hx, hy, center, scale)
        joint_arr[6] = [nx, ny, 1.0]

    # Add thorax as average of shoulders if available.
    # MPII thorax id = 7. COCO left shoulder=5, right shoulder=6
    sh_pts = []
    for coco_id in [5, 6]:
        conf = float(row.get(f"coco{coco_id}_conf", 0.0))
        x = float(row.get(f"coco{coco_id}_x", 0.0))
        y = float(row.get(f"coco{coco_id}_y", 0.0))
        if conf >= conf_thresh and x > 0 and y > 0:
            sh_pts.append([x, y])

    if len(sh_pts) == 2:
        sx, sy = np.mean(np.array(sh_pts), axis=0)
        nx, ny = norm_xy(sx, sy, center, scale)
        joint_arr[7] = [nx, ny, 1.0]

    # Use nose as a weak head/neck context if confident.
    # MPII upper_neck/head_top are not same as COCO nose, so keep conservative.
    nose_conf = float(row.get("coco0_conf", 0.0))
    nose_x = float(row.get("coco0_x", 0.0))
    nose_y = float(row.get("coco0_y", 0.0))
    if nose_conf >= conf_thresh and nose_x > 0 and nose_y > 0:
        nx, ny = norm_xy(nose_x, nose_y, center, scale)
        joint_arr[8] = [nx, ny, 1.0]

    # Hide target joint from input.
    joint_arr[target_mpii_id] = [0.0, 0.0, 0.0]

    feat = []
    for j in range(num_joints):
        feat.extend(joint_arr[j])

    feat.extend(one_hot(target_task_id, num_targets))

    return np.array(feat, dtype=np.float32), joint_arr


def geometry_from_yolo(row, target_name, conf_thresh=0.25):
    a, b = ENDPOINT_MPII[target_name]

    # convert MPII endpoint to COCO if possible
    if a not in MPII_TO_COCO or b not in MPII_TO_COCO:
        return np.nan, np.nan

    ca = MPII_TO_COCO[a]
    cb = MPII_TO_COCO[b]

    conf_a = float(row.get(f"coco{ca}_conf", 0.0))
    conf_b = float(row.get(f"coco{cb}_conf", 0.0))

    if conf_a < conf_thresh or conf_b < conf_thresh:
        return np.nan, np.nan

    ax = float(row.get(f"coco{ca}_x", np.nan))
    ay = float(row.get(f"coco{ca}_y", np.nan))
    bx = float(row.get(f"coco{cb}_x", np.nan))
    by = float(row.get(f"coco{cb}_y", np.nan))

    return (ax + bx) / 2.0, (ay + by) / 2.0


def draw_correction(image_path, row, out_path):
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)

    # red occlusion box
    x1, y1, x2, y2 = int(row["occ_x1"]), int(row["occ_y1"]), int(row["occ_x2"]), int(row["occ_y2"])
    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)

    # true orange X
    tx = int(round(row["target_orig_x"]))
    ty = int(round(row["target_orig_y"]))
    r = 7
    draw.line([tx - r, ty - r, tx + r, ty + r], fill=(255, 165, 0), width=4)
    draw.line([tx - r, ty + r, tx + r, ty - r], fill=(255, 165, 0), width=4)

    # YOLO purple circle
    if not pd.isna(row.get("yolo_target_x", np.nan)):
        yx = int(round(row["yolo_target_x"]))
        yy = int(round(row["yolo_target_y"]))
        draw.ellipse([yx - r, yy - r, yx + r, yy + r], outline=(150, 0, 255), width=4)

    # geometry cyan square
    if not pd.isna(row.get("geometry_x", np.nan)):
        gx = int(round(row["geometry_x"]))
        gy = int(round(row["geometry_y"]))
        draw.rectangle([gx - r, gy - r, gx + r, gy + r], outline=(0, 255, 255), width=4)

    # prior green triangle
    if not pd.isna(row.get("prior_pred_x", np.nan)):
        px = int(round(row["prior_pred_x"]))
        py = int(round(row["prior_pred_y"]))
        tri = [(px, py - r), (px - r, py + r), (px + r, py + r)]
        draw.polygon(tri, outline=(0, 255, 0), fill=(0, 255, 0))

    text = (
        f"{row['target_name']} | "
        f"YOLO={row.get('yolo_target_error_px', np.nan):.1f} "
        f"GEO={row.get('geometry_error_px', np.nan):.1f} "
        f"PRIOR={row.get('prior_error_px', np.nan):.1f}"
    )
    draw.rectangle([8, 8, 520, 36], fill=(0, 0, 0))
    draw.text((12, 12), text, fill=(255, 255, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo_csv", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--prior_model_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/stage2b_prior_correction")
    parser.add_argument("--conf_thresh", type=float, default=0.25)
    parser.add_argument("--max_visualizations", type=int, default=120)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    yolo_csv = Path(args.yolo_csv)
    image_dir = Path(args.image_dir)
    prior_model_dir = Path(args.prior_model_dir)
    out_dir = Path(args.out_dir)
    vis_dir = out_dir / "visualizations"

    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    config = load_json(prior_model_dir / "model_config.json")
    num_joints = int(config["num_joints"])
    num_targets = int(config["num_targets"])
    hidden_dim = int(config["hidden_dim"])
    dropout = float(config["dropout"])

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = FullSkeletonPriorMLP(
        num_joints=num_joints,
        num_targets=num_targets,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(torch.load(prior_model_dir / "best_model.pt", map_location=device))
    model.eval()

    df = pd.read_csv(yolo_csv)
    result_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Correcting YOLO with skeleton prior"):
        out_row = row.to_dict()

        if int(row.get("yolo_found_person", 0)) != 1:
            out_row["prior_pred_x"] = np.nan
            out_row["prior_pred_y"] = np.nan
            out_row["prior_error_px"] = np.nan
            out_row["geometry_x"] = np.nan
            out_row["geometry_y"] = np.nan
            out_row["geometry_error_px"] = np.nan
            result_rows.append(out_row)
            continue

        center, scale = compute_center_scale_from_yolo(row, conf_thresh=args.conf_thresh)

        if center is None:
            out_row["prior_pred_x"] = np.nan
            out_row["prior_pred_y"] = np.nan
            out_row["prior_error_px"] = np.nan
        else:
            feat, joint_arr = build_prior_input_from_yolo(
                row=row,
                center=center,
                scale=scale,
                target_name=row["target_name"],
                conf_thresh=args.conf_thresh,
                num_joints=num_joints,
                num_targets=num_targets,
            )

            x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                pred_norm = model(x).cpu().numpy()[0]

            px, py = denorm_xy(pred_norm[0], pred_norm[1], center, scale)

            out_row["prior_pred_x"] = px
            out_row["prior_pred_y"] = py
            out_row["prior_error_px"] = float(
                np.sqrt((px - row["target_orig_x"]) ** 2 + (py - row["target_orig_y"]) ** 2)
            )

        gx, gy = geometry_from_yolo(row, row["target_name"], conf_thresh=args.conf_thresh)
        out_row["geometry_x"] = gx
        out_row["geometry_y"] = gy

        if pd.isna(gx) or pd.isna(gy):
            out_row["geometry_error_px"] = np.nan
        else:
            out_row["geometry_error_px"] = float(
                np.sqrt((gx - row["target_orig_x"]) ** 2 + (gy - row["target_orig_y"]) ** 2)
            )

        result_rows.append(out_row)

    out_df = pd.DataFrame(result_rows)

    out_df["prior_better_than_yolo"] = out_df["prior_error_px"] < out_df["yolo_target_error_px"]
    out_df["prior_better_than_geometry"] = out_df["prior_error_px"] < out_df["geometry_error_px"]

    out_csv = out_dir / "yolo_prior_correction_results.csv"
    out_df.to_csv(out_csv, index=False)

    # Summary
    valid = out_df[out_df["yolo_found_person"] == 1].copy()

    summary = valid.groupby("target_name").agg(
        count=("target_name", "count"),
        yolo_mean_px=("yolo_target_error_px", "mean"),
        geometry_mean_px=("geometry_error_px", "mean"),
        prior_mean_px=("prior_error_px", "mean"),
        yolo_median_px=("yolo_target_error_px", "median"),
        geometry_median_px=("geometry_error_px", "median"),
        prior_median_px=("prior_error_px", "median"),
        prior_win_vs_yolo=("prior_better_than_yolo", "mean"),
        prior_win_vs_geometry=("prior_better_than_geometry", "mean"),
    ).reset_index()

    # Overall row
    overall = pd.DataFrame([{
        "target_name": "OVERALL",
        "count": len(valid),
        "yolo_mean_px": valid["yolo_target_error_px"].mean(),
        "geometry_mean_px": valid["geometry_error_px"].mean(),
        "prior_mean_px": valid["prior_error_px"].mean(),
        "yolo_median_px": valid["yolo_target_error_px"].median(),
        "geometry_median_px": valid["geometry_error_px"].median(),
        "prior_median_px": valid["prior_error_px"].median(),
        "prior_win_vs_yolo": valid["prior_better_than_yolo"].mean(),
        "prior_win_vs_geometry": valid["prior_better_than_geometry"].mean(),
    }])

    summary = pd.concat([summary, overall], ignore_index=True)
    summary_csv = out_dir / "comparison_summary.csv"
    summary.to_csv(summary_csv, index=False)

    print("\nComparison summary:")
    print(summary)

    # Visualizations
    sample_df = valid.sample(n=min(args.max_visualizations, len(valid)), random_state=11)
    for _, row in sample_df.iterrows():
        image_path = image_dir / row["occluded_image_name"]
        out_path = vis_dir / row["occluded_image_name"]
        draw_correction(image_path, row, out_path)

    print(f"\nSaved correction results: {out_csv}")
    print(f"Saved summary: {summary_csv}")
    print(f"Saved visualizations: {vis_dir}")


if __name__ == "__main__":
    main()
