import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm

from src.heatmap_utils import ANCHOR_NAMES, ANCHOR_TO_MPII


MPII_ID_TO_NAME = {
    0: 'right_ankle',
    1: 'right_knee',
    2: 'right_hip',
    3: 'left_hip',
    4: 'left_knee',
    5: 'left_ankle',
    6: 'pelvis',
    7: 'thorax',
    8: 'upper_neck',
    9: 'head_top',
    10: 'right_wrist',
    11: 'right_elbow',
    12: 'right_shoulder',
    13: 'left_shoulder',
    14: 'left_elbow',
    15: 'left_wrist',
}


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


def parse_visible(point):
    vis = _get_field(point, 'is_visible', None)
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
    annopoints = _get_field(rect, 'annopoints', None)
    if annopoints is None:
        return points

    for point_container in _as_list(annopoints):
        raw_points = _get_field(point_container, 'point', None)
        for p in _as_list(raw_points):
            jid = _to_int(_get_field(p, 'id', None), None)
            if jid is None or jid not in MPII_ID_TO_NAME:
                continue
            x = _to_float(_get_field(p, 'x', None), None)
            y = _to_float(_get_field(p, 'y', None), None)
            if x is None or y is None:
                continue
            points[jid] = {'x': x, 'y': y, 'visible': parse_visible(p)}
    return points


def get_image_name(ann):
    image_obj = _get_field(ann, 'image', None)
    if image_obj is None:
        return ''
    name = _get_field(image_obj, 'name', '')
    if isinstance(name, np.ndarray):
        if name.size == 0:
            return ''
        name = name.flatten()[0]
    return str(name)


def compute_bbox(points):
    xs, ys = [], []
    for p in points.values():
        xs.append(p['x'])
        ys.append(p['y'])
    if len(xs) < 3:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mat_path', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--out_csv', type=str, required=True)
    parser.add_argument('--min_visible_anchors', type=int, default=5)
    parser.add_argument('--include_test_split', action='store_true')
    args = parser.parse_args()

    mat = loadmat(args.mat_path, struct_as_record=False, squeeze_me=True)
    if 'RELEASE' not in mat:
        raise KeyError('Could not find RELEASE key in mat file.')

    release = mat['RELEASE']
    annolist = _as_list(_get_field(release, 'annolist'))
    img_train = _get_field(release, 'img_train', None)
    if img_train is not None:
        img_train = np.array(img_train).flatten()

    image_dir = Path(args.image_dir)
    rows = []

    for ann_index, ann in enumerate(tqdm(annolist, desc='Preparing AnchorNet dataset')):
        if not args.include_test_split and img_train is not None:
            try:
                if int(img_train[ann_index]) != 1:
                    continue
            except Exception:
                pass

        image_name = get_image_name(ann)
        if not image_name:
            continue
        if not (image_dir / image_name).exists():
            continue

        rects = _as_list(_get_field(ann, 'annorect', None))
        for person_index, rect in enumerate(rects):
            points = parse_person_points(rect)
            if not points:
                continue

            bbox = compute_bbox(points)
            if bbox is None:
                continue

            row = {
                'image_name': image_name,
                'ann_index': ann_index,
                'person_index': person_index,
                'bbox_x1': float(bbox[0]),
                'bbox_y1': float(bbox[1]),
                'bbox_x2': float(bbox[2]),
                'bbox_y2': float(bbox[3]),
            }

            visible_count = 0
            for name in ANCHOR_NAMES:
                jid = ANCHOR_TO_MPII[name]
                if jid in points:
                    row[f'{name}_x'] = float(points[jid]['x'])
                    row[f'{name}_y'] = float(points[jid]['y'])
                    vis = 1.0 if int(points[jid]['visible']) == 1 else 0.0
                    row[f'{name}_vis'] = vis
                    visible_count += int(vis == 1.0)
                else:
                    row[f'{name}_x'] = 0.0
                    row[f'{name}_y'] = 0.0
                    row[f'{name}_vis'] = 0.0

            if visible_count < args.min_visible_anchors:
                continue
            row['num_visible_anchors'] = visible_count
            rows.append(row)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    print(f'Saved: {out_csv}')
    print(f'Total samples: {len(df)}')
    if len(df) > 0:
        print(df['num_visible_anchors'].describe())


if __name__ == '__main__':
    main()
