import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm
from ultralytics import YOLO

from src.utils_full import TARGET_TO_COCO


def compute_gt_center(row):
    xs = []
    ys = []

    for j in range(16):
        x = float(row.get(f"j{j}_orig_x", 0.0))
        y = float(row.get(f"j{j}_orig_y", 0.0))
        m = float(row.get(f"j{j}_mask", 0.0))

        # target is masked in skeleton input, but its orig coordinate exists.
        if x > 0 and y > 0:
            xs.append(x)
            ys.append(y)

    if not xs:
        return float(row["target_orig_x"]), float(row["target_orig_y"])

    return float(np.mean(xs)), float(np.mean(ys))


def select_best_person(result, gt_cx, gt_cy):
    """
    Select YOLO detection closest to the MPII annotated person's joint center.
    """
    if result.keypoints is None:
        return None

    xy = result.keypoints.xy
    conf = result.keypoints.conf

    if xy is None or len(xy) == 0:
        return None

    xy = xy.cpu().numpy()
    conf = conf.cpu().numpy() if conf is not None else np.ones((xy.shape[0], xy.shape[1]), dtype=np.float32)

    best_idx = None
    best_dist = float("inf")

    for i in range(xy.shape[0]):
        valid = conf[i] > 0.25
        if valid.sum() == 0:
            continue

        cx = xy[i, valid, 0].mean()
        cy = xy[i, valid, 1].mean()
        dist = np.sqrt((cx - gt_cx) ** 2 + (cy - gt_cy) ** 2)

        if dist < best_dist:
            best_dist = dist
            best_idx = i

    if best_idx is None:
        return None

    return xy[best_idx], conf[best_idx], best_dist


def draw_yolo_vis(image_path, row, kpts_xy, kpts_conf, out_path):
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)

    # occlusion box
    x1, y1, x2, y2 = int(row["occ_x1"]), int(row["occ_y1"]), int(row["occ_x2"]), int(row["occ_y2"])
    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)

    # true target
    tx, ty = int(row["target_orig_x"]), int(row["target_orig_y"])
    r = 7
    draw.line([tx - r, ty - r, tx + r, ty + r], fill=(255, 165, 0), width=4)
    draw.line([tx - r, ty + r, tx + r, ty - r], fill=(255, 165, 0), width=4)

    # all YOLO keypoints
    for i in range(len(kpts_xy)):
        if kpts_conf[i] > 0.25:
            x, y = int(kpts_xy[i, 0]), int(kpts_xy[i, 1])
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(180, 80, 255))

    # YOLO target joint
    coco_id = TARGET_TO_COCO[row["target_name"]]
    px, py = int(kpts_xy[coco_id, 0]), int(kpts_xy[coco_id, 1])
    draw.ellipse([px - 8, py - 8, px + 8, py + 8], outline=(128, 0, 255), width=4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/stage2b_yolo_pose")
    parser.add_argument("--yolo_model", type=str, default="yolov8n-pose.pt")
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default=None, help="Example: 0 or cpu. Leave empty for auto.")
    parser.add_argument("--max_visualizations", type=int, default=80)
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    vis_dir = out_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if args.max_samples > 0:
        df = df.head(args.max_samples).reset_index(drop=True)

    print(f"Loading YOLO model: {args.yolo_model}")
    model = YOLO(args.yolo_model)

    rows = []
    vis_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Running YOLO-Pose"):
        image_path = image_dir / row["occluded_image_name"]
        if not image_path.exists():
            continue

        gt_cx, gt_cy = compute_gt_center(row)

        pred_args = {
            "source": str(image_path),
            "conf": args.conf,
            "imgsz": args.imgsz,
            "verbose": False,
        }
        if args.device is not None:
            pred_args["device"] = args.device

        results = model.predict(**pred_args)
        if not results:
            continue

        selected = select_best_person(results[0], gt_cx, gt_cy)
        if selected is None:
            out_row = row.to_dict()
            out_row["yolo_found_person"] = 0
            rows.append(out_row)
            continue

        kpts_xy, kpts_conf, person_select_dist = selected

        out_row = row.to_dict()
        out_row["yolo_found_person"] = 1
        out_row["person_select_dist"] = float(person_select_dist)

        for k in range(17):
            out_row[f"coco{k}_x"] = float(kpts_xy[k, 0])
            out_row[f"coco{k}_y"] = float(kpts_xy[k, 1])
            out_row[f"coco{k}_conf"] = float(kpts_conf[k])

        target_coco_id = TARGET_TO_COCO[row["target_name"]]
        yx = float(kpts_xy[target_coco_id, 0])
        yy = float(kpts_xy[target_coco_id, 1])
        yc = float(kpts_conf[target_coco_id])

        out_row["yolo_target_x"] = yx
        out_row["yolo_target_y"] = yy
        out_row["yolo_target_conf"] = yc
        out_row["yolo_target_error_px"] = float(np.sqrt((yx - row["target_orig_x"]) ** 2 + (yy - row["target_orig_y"]) ** 2))

        rows.append(out_row)

        if vis_count < args.max_visualizations:
            draw_yolo_vis(image_path, row, kpts_xy, kpts_conf, vis_dir / row["occluded_image_name"])
            vis_count += 1

    out_df = pd.DataFrame(rows)
    out_csv = out_dir / "yolo_pose_predictions.csv"
    out_df.to_csv(out_csv, index=False)

    print(f"\nSaved YOLO predictions: {out_csv}")
    if "yolo_found_person" in out_df.columns:
        print("Found person rate:", out_df["yolo_found_person"].mean())

    if "yolo_target_error_px" in out_df.columns:
        print("\nYOLO target error by joint:")
        print(out_df[out_df["yolo_found_person"] == 1].groupby("target_name")["yolo_target_error_px"].agg(["count", "mean", "median", "std"]))

    print(f"Saved visualizations: {vis_dir}")


if __name__ == "__main__":
    main()
