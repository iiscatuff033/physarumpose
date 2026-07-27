import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm

from src.anchor_dataset_utils import ANCHORS, ANCHOR_NAMES


def _as_list(x):
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return [item for item in x.flatten() if item is not None]
    return [x]


def _get_field(obj, name, default=None):
    return getattr(obj, name, default) if hasattr(obj, name) else default


def _to_float(x, default=None):
    try:
        if isinstance(x, np.ndarray):
            if x.size == 0:
                return default
            return float(x.flatten()[0])
        return float(x)
    except Exception:
        return default


def _to_int(x, default=None):
    try:
        if isinstance(x, np.ndarray):
            if x.size == 0:
                return default
            return int(x.flatten()[0])
        return int(x)
    except Exception:
        return default


def parse_visible_flag(point):
    vis = _get_field(point, "is_visible", None)
    if vis is None:
        return 1
    if isinstance(vis, np.ndarray):
        if vis.size == 0:
            return 1
        vis = vis.flatten()[0]
    try:
        return int(vis)
    except Exception:
        return 1


def parse_person_points(rect):
    points = {}
    annopoints = _get_field(rect, "annopoints", None)
    if annopoints is None:
        return points

    for point_container in _as_list(annopoints):
        raw_points = _get_field(point_container, "point", None)
        for p in _as_list(raw_points):
            joint_id = _to_int(_get_field(p, "id", None), None)
            if joint_id is None:
                continue
            x = _to_float(_get_field(p, "x", None), None)
            y = _to_float(_get_field(p, "y", None), None)
            if x is None or y is None:
                continue
            points[joint_id] = {"x": float(x), "y": float(y), "visible": parse_visible_flag(p)}
    return points


def get_image_name(ann):
    image_obj = _get_field(ann, "image", None)
    if image_obj is None:
        return ""
    name = _get_field(image_obj, "name", "")
    if isinstance(name, np.ndarray):
        if name.size == 0:
            return ""
        name = name.flatten()[0]
    return str(name)


def make_crop_from_points(points, image_w, image_h, margin_ratio=0.25):
    xs = []
    ys = []
    for p in points.values():
        if p["x"] > 0 and p["y"] > 0:
            xs.append(p["x"])
            ys.append(p["y"])

    if len(xs) < 3:
        return None

    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)

    bw = max(x2 - x1, 20.0)
    bh = max(y2 - y1, 20.0)
    side = max(bw, bh)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    side = side * (1.0 + margin_ratio * 2.0)

    x1 = max(0.0, cx - side / 2.0)
    y1 = max(0.0, cy - side / 2.0)
    x2 = min(float(image_w - 1), cx + side / 2.0)
    y2 = min(float(image_h - 1), cy + side / 2.0)

    if x2 <= x1 + 5 or y2 <= y1 + 5:
        return None

    return x1, y1, x2, y2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default="data/processed/anchor_net_mpii/anchor_dataset.csv")
    parser.add_argument("--min_visible_anchors", type=int, default=5)
    parser.add_argument("--include_test_split", action="store_true")
    args = parser.parse_args()

    mat_path = Path(args.mat_path)
    image_dir = Path(args.image_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    mat = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    release = mat["RELEASE"]
    annolist = _as_list(_get_field(release, "annolist"))
    img_train = _get_field(release, "img_train", None)
    if img_train is not None:
        img_train = np.array(img_train).flatten()

    rows = []

    for ann_index, ann in enumerate(tqdm(annolist, desc="Preparing anchor dataset")):
        if not args.include_test_split and img_train is not None:
            try:
                if int(img_train[ann_index]) != 1:
                    continue
            except Exception:
                pass

        image_name = get_image_name(ann)
        if not image_name:
            continue

        image_path = image_dir / image_name
        if not image_path.exists():
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        rects = _as_list(_get_field(ann, "annorect", None))

        for person_index, rect in enumerate(rects):
            points = parse_person_points(rect)
            if not points:
                continue

            crop = make_crop_from_points(points, w, h)
            if crop is None:
                continue

            row = {
                "image_name": image_name,
                "ann_index": ann_index,
                "person_index": person_index,
                "image_w": w,
                "image_h": h,
                "crop_x1": crop[0],
                "crop_y1": crop[1],
                "crop_x2": crop[2],
                "crop_y2": crop[3],
            }

            visible_anchors = 0
            for a in ANCHORS:
                name = a["name"]
                jid = a["mpii_id"]
                if jid in points and points[jid]["visible"] == 1:
                    x = float(points[jid]["x"])
                    y = float(points[jid]["y"])
                    v = 1.0
                    visible_anchors += 1
                elif jid in points:
                    x = float(points[jid]["x"])
                    y = float(points[jid]["y"])
                    v = 0.0
                else:
                    x, y, v = 0.0, 0.0, 0.0
                row[f"{name}_x"] = x
                row[f"{name}_y"] = y
                row[f"{name}_vis"] = v

            row["visible_anchor_count"] = visible_anchors

            if visible_anchors >= args.min_visible_anchors:
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    print(f"Saved: {out_csv}")
    print(f"Rows: {len(df)}")
    if len(df) > 0:
        print(df["visible_anchor_count"].describe())


if __name__ == "__main__":
    main()
