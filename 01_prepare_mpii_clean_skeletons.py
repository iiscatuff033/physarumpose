import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm

from src.utils import MPII_ID_TO_NAME, TASKS


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
    """
    MPII visibility:
    1 = visible
    0 = annotated but not visible/occluded

    If missing/empty, we treat it as visible because the coordinate is annotated.
    For stricter filtering, change empty visibility to 0.
    """
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
            jid = _to_int(_get_field(p, "id", None), None)
            if jid is None or jid not in MPII_ID_TO_NAME:
                continue

            x = _to_float(_get_field(p, "x", None), None)
            y = _to_float(_get_field(p, "y", None), None)
            if x is None or y is None:
                continue

            points[MPII_ID_TO_NAME[jid]] = {
                "x": x,
                "y": y,
                "visible": parse_visible_flag(p),
            }

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


def compute_normalization(points):
    xy = np.array([[p["x"], p["y"]] for p in points.values()], dtype=np.float32)
    if len(xy) < 3:
        return None, None

    center = xy.mean(axis=0)
    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)
    width, height = max_xy - min_xy
    scale = max(float(width), float(height), 1.0)
    return center, scale


def norm_point(p, center, scale):
    return (np.array([p["x"], p["y"]], dtype=np.float32) - center) / scale


def make_task_rows(points, image_name, person_index, ann_index, strict_visible=True):
    center, scale = compute_normalization(points)
    if center is None:
        return []

    rows = []

    for task_id, task in TASKS.items():
        a = task["endpoint_a"]
        b = task["endpoint_b"]
        t = task["target"]

        if a not in points or b not in points or t not in points:
            continue

        if strict_visible:
            if points[a]["visible"] != 1 or points[b]["visible"] != 1 or points[t]["visible"] != 1:
                continue

        p1 = norm_point(points[a], center, scale)
        p2 = norm_point(points[b], center, scale)
        target = norm_point(points[t], center, scale)

        rows.append({
            "image_name": image_name,
            "ann_index": ann_index,
            "person_index": person_index,
            "task_id": task_id,
            "task_name": task["name"],
            "endpoint_a": a,
            "endpoint_b": b,
            "target_joint": t,
            "x1": float(p1[0]),
            "y1": float(p1[1]),
            "x2": float(p2[0]),
            "y2": float(p2[1]),
            "target_x": float(target[0]),
            "target_y": float(target[1]),
            "center_x": float(center[0]),
            "center_y": float(center[1]),
            "scale": float(scale),
            "target_orig_x": float(points[t]["x"]),
            "target_orig_y": float(points[t]["y"]),
        })

    return rows


def prepare_mpii(mat_path, out_csv, use_train_split_only=True, strict_visible=True):
    mat_path = Path(mat_path)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading MPII annotation file: {mat_path}")
    mat = loadmat(mat_path, struct_as_record=False, squeeze_me=True)

    release = mat["RELEASE"]
    annolist = _as_list(_get_field(release, "annolist"))

    img_train = _get_field(release, "img_train", None)
    if img_train is not None:
        img_train = np.array(img_train).flatten()

    all_rows = []

    for ann_index, ann in enumerate(tqdm(annolist, desc="Parsing MPII annotations")):
        if use_train_split_only and img_train is not None:
            try:
                if int(img_train[ann_index]) != 1:
                    continue
            except Exception:
                pass

        image_name = get_image_name(ann)
        rects = _as_list(_get_field(ann, "annorect", None))

        for person_index, rect in enumerate(rects):
            points = parse_person_points(rect)
            if not points:
                continue

            all_rows.extend(make_task_rows(
                points=points,
                image_name=image_name,
                person_index=person_index,
                ann_index=ann_index,
                strict_visible=strict_visible,
            ))

    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)

    print(f"\nSaved: {out_csv}")
    print(f"Total task samples: {len(df)}")
    if len(df) > 0:
        print("\nSamples per task:")
        print(df["task_name"].value_counts())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat_path", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default="data/processed/mpii_clean_limb_tasks.csv")
    parser.add_argument("--include_test_split", action="store_true")
    parser.add_argument("--allow_occluded_points", action="store_true")
    args = parser.parse_args()

    prepare_mpii(
        mat_path=args.mat_path,
        out_csv=args.out_csv,
        use_train_split_only=not args.include_test_split,
        strict_visible=not args.allow_occluded_points,
    )


if __name__ == "__main__":
    main()
