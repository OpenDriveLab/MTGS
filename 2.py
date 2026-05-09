import pickle
from collections import Counter, defaultdict
from pathlib import Path
import math

import numpy as np
from pyquaternion import Quaternion

PKL_PATH = Path("data/MTGS/road_block-331220_4690660_331190_4690710/video_scene_dict.pkl")
FRAME_START = 0
FRAME_END = None  # None=到最后
DIRECTION_COS_THRESHOLD = 0.7


def classify_direction(ego_disp, veh_disp, threshold):
    ego_norm = np.linalg.norm(ego_disp)
    veh_norm = np.linalg.norm(veh_disp)
    if ego_norm < 1e-6 or veh_norm < 1e-6:
        return 0.0, 0.0, "unknown"
    cos_sim = float(np.dot(ego_disp, veh_disp) / (ego_norm * veh_norm))
    cos_sim = max(min(cos_sim, 1.0), -1.0)
    angle = float(np.degrees(np.arccos(cos_sim)))
    if cos_sim >= threshold:
        label = "same"
    elif cos_sim <= -threshold:
        label = "opposite"
    else:
        label = "cross"
    return cos_sim, angle, label


def analyze_video(video_token, frames):
    if not frames:
        return
    if FRAME_END is None:
        end_idx = len(frames) - 1
    else:
        end_idx = min(FRAME_END, len(frames) - 1)

    stats = defaultdict(lambda: {"min": 1e9, "sum": 0.0, "count": 0, "min_frame": None})
    nearest_counts = Counter()
    track_positions = defaultdict(list)
    ego_positions = []

    for idx, f in enumerate(frames[FRAME_START:end_idx + 1], start=FRAME_START):
        best = None
        e2g_trans = np.array(f["ego2global_translation"], dtype=np.float32)
        e2g_rot = Quaternion(f["ego2global_rotation"]).rotation_matrix
        ego_positions.append(e2g_trans)
        for name, token, box in zip(f["gt_names"], f["track_tokens"], f["gt_boxes"]):
            if name != "vehicle":
                continue
            center_ego = np.array(box[:3], dtype=np.float32)
            center_global = center_ego @ e2g_rot.T + e2g_trans
            track_positions[token].append(center_global)
            x, y, z = box[:3]
            if x <= 0:  # 只看前方车辆；不需要可删
                continue
            d = math.sqrt(x * x + y * y + z * z)
            s = stats[token]
            s["sum"] += d
            s["count"] += 1
            if d < s["min"]:
                s["min"] = d
                s["min_frame"] = idx
            if best is None or d < best[0]:
                best = (d, token)
        if best:
            nearest_counts[best[1]] += 1

    ego_disp = np.zeros(3, dtype=np.float32)
    if len(ego_positions) >= 2:
        ego_disp = ego_positions[-1] - ego_positions[0]

    moving = []
    static = []
    moving_direction_counts = Counter()
    for token, centers in track_positions.items():
        if len(centers) < 2:
            continue
        veh_disp = centers[-1] - centers[0]
        disp = float(np.linalg.norm(veh_disp))
        cos_sim, angle, label = classify_direction(ego_disp, veh_disp, DIRECTION_COS_THRESHOLD)
        if disp < 3.0:
            static.append((token, disp, len(centers), cos_sim, angle, label))
        else:
            moving.append((token, disp, len(centers), cos_sim, angle, label))
            moving_direction_counts[label] += 1

    travel_id = int(video_token.split("-")[-1])
    print(f"\ntravel_id={travel_id} video_token={video_token} frames={end_idx - FRAME_START + 1}")

    print("nearest_counts top:")
    for token, cnt in nearest_counts.most_common(10):
        s = stats[token]
        avg = s["sum"] / s["count"]
        print(
            token,
            "nearest_frames",
            cnt,
            "avg_dist",
            round(avg, 2),
            "min_dist",
            round(s["min"], 2),
            "min_frame",
            s["min_frame"],
        )

    print("moving top:")
    for token, disp, count, cos_sim, angle, label in sorted(moving, key=lambda x: -x[1])[:10]:
        print(
            token,
            "disp",
            round(disp, 2),
            "frames",
            count,
            "cos",
            round(cos_sim, 2),
            "angle",
            round(angle, 1),
            label,
        )

    print("moving direction counts:", dict(moving_direction_counts))

    print("static top:")
    for token, disp, count, cos_sim, angle, label in sorted(static, key=lambda x: x[1])[:10]:
        print(
            token,
            "disp",
            round(disp, 2),
            "frames",
            count,
            "cos",
            round(cos_sim, 2),
            "angle",
            round(angle, 1),
            label,
        )


data = pickle.load(open(PKL_PATH, "rb"))
video_tokens = sorted(data.keys(), key=lambda vt: int(vt.split("-")[-1]))
for video_token in video_tokens:
    analyze_video(video_token, data[video_token]["frame_infos"])
