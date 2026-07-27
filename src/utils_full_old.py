import json, random
from pathlib import Path
import numpy as np
import torch

MPII_ID_TO_NAME = {
    0: "right_ankle", 1: "right_knee", 2: "right_hip", 3: "left_hip",
    4: "left_knee", 5: "left_ankle", 6: "pelvis", 7: "thorax",
    8: "upper_neck", 9: "head_top", 10: "right_wrist", 11: "right_elbow",
    12: "right_shoulder", 13: "left_shoulder", 14: "left_elbow", 15: "left_wrist",
}

TARGET_JOINTS = {
    0: {"name": "right_elbow", "mpii_id": 11},
    1: {"name": "left_elbow", "mpii_id": 14},
    2: {"name": "right_knee", "mpii_id": 1},
    3: {"name": "left_knee", "mpii_id": 4},
}

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def save_json(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f: json.dump(obj, f, indent=2)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f: return json.load(f)

def one_hot(i, n):
    a = np.zeros(n, dtype=np.float32); a[int(i)] = 1.0; return a

def build_feature_from_row(row, num_joints=16, num_targets=4):
    feat = []
    for j in range(num_joints):
        feat += [float(row[f"j{j}_x"]), float(row[f"j{j}_y"]), float(row[f"j{j}_mask"])]
    feat += list(one_hot(int(row["target_task_id"]), num_targets))
    return np.array(feat, dtype=np.float32)

def build_target_from_row(row):
    return np.array([row["target_x"], row["target_y"]], dtype=np.float32)
