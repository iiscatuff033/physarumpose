import json
import random
from pathlib import Path

import numpy as np
import torch


TASKS = {
    0: {
        "name": "right_elbow",
        "endpoint_a": "right_shoulder",
        "endpoint_b": "right_wrist",
        "target": "right_elbow",
    },
    1: {
        "name": "left_elbow",
        "endpoint_a": "left_shoulder",
        "endpoint_b": "left_wrist",
        "target": "left_elbow",
    },
    2: {
        "name": "right_knee",
        "endpoint_a": "right_hip",
        "endpoint_b": "right_ankle",
        "target": "right_knee",
    },
    3: {
        "name": "left_knee",
        "endpoint_a": "left_hip",
        "endpoint_b": "left_ankle",
        "target": "left_knee",
    },
}

MPII_ID_TO_NAME = {
    0: "right_ankle",
    1: "right_knee",
    2: "right_hip",
    3: "left_hip",
    4: "left_knee",
    5: "left_ankle",
    6: "pelvis",
    7: "thorax",
    8: "upper_neck",
    9: "head_top",
    10: "right_wrist",
    11: "right_elbow",
    12: "right_shoulder",
    13: "left_shoulder",
    14: "left_elbow",
    15: "left_wrist",
}

MPII_NAME_TO_ID = {v: k for k, v in MPII_ID_TO_NAME.items()}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def one_hot(task_id: int, num_tasks: int):
    vec = np.zeros(num_tasks, dtype=np.float32)
    vec[task_id] = 1.0
    return vec


def build_feature(row, num_tasks: int):
    endpoints = np.array([row["x1"], row["y1"], row["x2"], row["y2"]], dtype=np.float32)
    return np.concatenate([endpoints, one_hot(int(row["task_id"]), num_tasks)], axis=0)


def build_target(row):
    return np.array([row["target_x"], row["target_y"]], dtype=np.float32)
