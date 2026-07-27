import json
from pathlib import Path

import numpy as np


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

TARGET_JOINTS = {
    0: {"name": "right_elbow", "mpii_id": 11},
    1: {"name": "left_elbow", "mpii_id": 14},
    2: {"name": "right_knee", "mpii_id": 1},
    3: {"name": "left_knee", "mpii_id": 4},
}

TARGET_NAME_TO_TASK = {v["name"]: k for k, v in TARGET_JOINTS.items()}


COCO_ID_TO_NAME = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}

# MPII -> COCO mapping for overlapping joints
MPII_TO_COCO = {
    0: 16,  # right_ankle
    1: 14,  # right_knee
    2: 12,  # right_hip
    3: 11,  # left_hip
    4: 13,  # left_knee
    5: 15,  # left_ankle
    10: 10, # right_wrist
    11: 8,  # right_elbow
    12: 6,  # right_shoulder
    13: 5,  # left_shoulder
    14: 7,  # left_elbow
    15: 9,  # left_wrist
}

TARGET_TO_COCO = {
    "right_elbow": 8,
    "left_elbow": 7,
    "right_knee": 14,
    "left_knee": 13,
}

ENDPOINT_MPII = {
    "right_elbow": (12, 10),  # right shoulder, right wrist
    "left_elbow": (13, 15),   # left shoulder, left wrist
    "right_knee": (2, 0),     # right hip, right ankle
    "left_knee": (3, 5),      # left hip, left ankle
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def one_hot(index: int, size: int):
    arr = np.zeros(size, dtype=np.float32)
    arr[index] = 1.0
    return arr
