import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm
from src.utils_full import MPII_ID_TO_NAME, TARGET_JOINTS

def _as_list(x):
    if x is None: return []
    if isinstance(x, np.ndarray): return [i for i in x.flatten() if i is not None]
    return [x]

def _get(obj, name, default=None):
    return getattr(obj, name, default) if hasattr(obj, name) else default

def _to_float(x, default=None):
    try:
        if isinstance(x, np.ndarray):
            if x.size == 0: return default
            return float(x.flatten()[0])
        return float(x)
    except Exception:
        return default

def _to_int(x, default=None):
    try:
        if isinstance(x, np.ndarray):
            if x.size == 0: return default
            return int(x.flatten()[0])
        return int(x)
    except Exception:
        return default

def parse_visible(p):
    vis = _get(p, "is_visible", None)
    if vis is None: return 1
    if isinstance(vis, np.ndarray):
        if vis.size == 0: return 1
        vis = vis.flatten()[0]
    try: return int(vis)
    except Exception: return 1

def parse_points(rect):
    pts = {}
    annopoints = _get(rect, "annopoints", None)
    if annopoints is None: return pts
    for cont in _as_list(annopoints):
        for p in _as_list(_get(cont, "point", None)):
            jid = _to_int(_get(p, "id", None), None)
            if jid is None or jid not in MPII_ID_TO_NAME: continue
            x, y = _to_float(_get(p, "x", None), None), _to_float(_get(p, "y", None), None)
            if x is None or y is None: continue
            pts[jid] = {"x": x, "y": y, "visible": parse_visible(p)}
    return pts

def image_name(ann):
    img = _get(ann, "image", None)
    name = _get(img, "name", "") if img is not None else ""
    if isinstance(name, np.ndarray):
        if name.size == 0: return ""
        name = name.flatten()[0]
    return str(name)

def norm_info(points):
    xy = np.array([[p["x"], p["y"]] for p in points.values()], dtype=np.float32)
    if len(xy) < 3: return None, None
    center = xy.mean(axis=0)
    mn, mx = xy.min(axis=0), xy.max(axis=0)
    scale = max(float((mx-mn)[0]), float((mx-mn)[1]), 1.0)
    return center, scale

def norm(x, y, center, scale):
    a = (np.array([x, y], dtype=np.float32) - center) / scale
    return float(a[0]), float(a[1])

def make_rows(points, img_name, ann_i, person_i, min_visible=8, require_all_visible=False):
    center, scale = norm_info(points)
    if center is None: return []
    visible_count = sum(1 for p in points.values() if p["visible"] == 1)
    if visible_count < min_visible: return []

    rows = []
    for task_id, tinfo in TARGET_JOINTS.items():
        tid = tinfo["mpii_id"]
        if tid not in points: continue
        if points[tid]["visible"] != 1: continue

        if require_all_visible:
            if len(points) < 16: continue
            if any(points.get(j, {}).get("visible", 0) != 1 for j in range(16)): continue

        tx, ty = norm(points[tid]["x"], points[tid]["y"], center, scale)
        row = {
            "image_name": img_name, "ann_index": ann_i, "person_index": person_i,
            "target_task_id": task_id, "target_name": tinfo["name"], "target_mpii_id": tid,
            "target_x": tx, "target_y": ty,
            "target_orig_x": float(points[tid]["x"]), "target_orig_y": float(points[tid]["y"]),
            "center_x": float(center[0]), "center_y": float(center[1]), "scale": float(scale),
            "visible_count": int(visible_count),
        }

        for j in range(16):
            if j in points:
                x, y = norm(points[j]["x"], points[j]["y"], center, scale)
                m = 1.0 if points[j]["visible"] == 1 else 0.0
            else:
                x, y, m = 0.0, 0.0, 0.0

            # hide the target joint from model input
            if j == tid:
                x, y, m = 0.0, 0.0, 0.0

            row[f"j{j}_x"] = x
            row[f"j{j}_y"] = y
            row[f"j{j}_mask"] = m

        rows.append(row)
    return rows

def prepare(mat_path, out_csv, train_only=True, min_visible=8, require_all_visible=False):
    mat_path = Path(mat_path)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    mat = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    rel = mat["RELEASE"]
    annolist = _as_list(_get(rel, "annolist"))
    img_train = _get(rel, "img_train", None)
    if img_train is not None: img_train = np.array(img_train).flatten()

    all_rows = []
    for ann_i, ann in enumerate(tqdm(annolist, desc="Parsing MPII full context")):
        if train_only and img_train is not None:
            try:
                if int(img_train[ann_i]) != 1: continue
            except Exception: pass
        name = image_name(ann)
        for person_i, rect in enumerate(_as_list(_get(ann, "annorect", None))):
            pts = parse_points(rect)
            if not pts: continue
            all_rows.extend(make_rows(pts, name, ann_i, person_i, min_visible, require_all_visible))

    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    print(f"Total samples: {len(df)}")
    if len(df):
        print(df["target_name"].value_counts())
        print(df["visible_count"].describe())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat_path", required=True)
    ap.add_argument("--out_csv", default="data/processed/mpii_full_context_tasks.csv")
    ap.add_argument("--include_test_split", action="store_true")
    ap.add_argument("--min_visible", type=int, default=8)
    ap.add_argument("--require_all_visible", action="store_true")
    args = ap.parse_args()
    prepare(args.mat_path, args.out_csv, not args.include_test_split, args.min_visible, args.require_all_visible)

if __name__ == "__main__":
    main()
