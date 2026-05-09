#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import os
import argparse
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.distance import cdist

from nuplan_scripts.utils.config import load_config, FrameCentralConfig
from nuplan_scripts.utils.video_scene_dict_tools import VideoScene
from nuplan_scripts.utils.nuplan_utils_custom import fix_pts_interpolate
from nuplan_scripts.utils.constants import CONSOLE, NUPLAN_TIMEZONE


def _frame_xy(frame):
    return np.asarray(frame['ego2global_translation'][:2], dtype=np.float64)


def _expand_frames(all_frames, sub_frames, expand_buffer):
    if expand_buffer <= 0:
        return sub_frames

    first_idx = all_frames.index(sub_frames[0])
    last_idx = all_frames.index(sub_frames[-1])

    expanded_start_idx = first_idx
    cumulative_dist = 0.0
    for i in range(first_idx - 1, -1, -1):
        dist = np.linalg.norm(_frame_xy(all_frames[i]) - _frame_xy(all_frames[i + 1]))
        cumulative_dist += dist
        if cumulative_dist >= expand_buffer:
            expanded_start_idx = i
            break

    expanded_end_idx = last_idx
    cumulative_dist = 0.0
    for i in range(last_idx, len(all_frames) - 1):
        dist = np.linalg.norm(_frame_xy(all_frames[i]) - _frame_xy(all_frames[i + 1]))
        cumulative_dist += dist
        if cumulative_dist >= expand_buffer:
            expanded_end_idx = i
            break

    return all_frames[expanded_start_idx:expanded_end_idx + 1]


def split_video_from_navsim_meta(config, video_scene: VideoScene):
    meta_root = Path(config.navsim_meta_root)
    assert meta_root.exists(), f'navsim_meta_root not found: {meta_root}'

    road_block = np.asarray(config.road_block, dtype=np.float64)
    video_infos = []

    for meta_path in sorted(meta_root.glob('*.pkl')):
        with open(meta_path, 'rb') as f:
            frames = pickle.load(f)

        if not frames:
            continue
        if frames[0]['map_location'] != config.city:
            continue

        sampled_frames = frames[::max(1, config.interval)]
        trajectory = np.asarray([_frame_xy(frame) for frame in sampled_frames], dtype=np.float64)
        in_region = np.all(trajectory > road_block[:2], axis=1) & np.all(trajectory < road_block[2:], axis=1)
        if not in_region.any():
            continue

        current_frames = []
        for idx, frame in enumerate(sampled_frames):
            if in_region[idx]:
                if idx > 0 and not in_region[idx - 1] and len(current_frames) > 1:
                    expanded_frames = _expand_frames(sampled_frames, current_frames, config.expand_buffer)
                    video_infos.append({
                        'video_token': '',
                        'log_token': expanded_frames[0]['log_token'],
                        'log_name': expanded_frames[0]['log_name'],
                        'map_location': expanded_frames[0]['map_location'],
                        'vehicle_name': expanded_frames[0]['vehicle_name'],
                        'start_ts': expanded_frames[0]['timestamp'],
                        'frames': expanded_frames,
                        'trajectory': np.asarray([_frame_xy(item) for item in expanded_frames], dtype=np.float64),
                    })
                    current_frames = []
                current_frames.append(frame)

        if len(current_frames) > 1:
            expanded_frames = _expand_frames(sampled_frames, current_frames, config.expand_buffer)
            video_infos.append({
                'video_token': '',
                'log_token': expanded_frames[0]['log_token'],
                'log_name': expanded_frames[0]['log_name'],
                'map_location': expanded_frames[0]['map_location'],
                'vehicle_name': expanded_frames[0]['vehicle_name'],
                'start_ts': expanded_frames[0]['timestamp'],
                'frames': expanded_frames,
                'trajectory': np.asarray([_frame_xy(item) for item in expanded_frames], dtype=np.float64),
            })

    return video_infos


def sort_navsim_video_infos(config, video_infos):
    if config.__class__.__name__ == 'RoadBlockConfig':
        video_infos = sorted(video_infos, key=lambda x: x['start_ts'])
        for idx, video in enumerate(video_infos):
            video['video_idx'] = idx
            video['video_token'] = f'{config.road_block_name}-{idx}'
        return video_infos

    if config.__class__.__name__ != 'FrameCentralConfig':
        raise NotImplementedError(f'Unsupported config type: {config.__class__.__name__}')

    central_log = config.central_log
    central_token = config.central_tokens[0]
    central_video_info = None
    for video in video_infos:
        if video['log_name'] != central_log:
            continue
        if any(frame['token'] == central_token for frame in video['frames']):
            central_video_info = video
            break
    assert central_video_info is not None, 'Central video not found in navsim video infos!'

    central_video_info['video_idx'] = 0
    central_video_info['video_token'] = f'{config.road_block_name}-0'

    central_timestamp = central_video_info['start_ts']
    video_infos = sorted(video_infos, key=lambda x: abs(x['start_ts'] - central_timestamp))
    video_infos = [video for video in video_infos if video is not central_video_info]

    road_block = np.asarray(config.road_block, dtype=np.float64)
    filtered_video_infos = [central_video_info]
    last_idx = 0
    for video in video_infos:
        video_traj = video['trajectory']
        within_mask = np.all(video_traj > road_block[:2], axis=1) & np.all(video_traj < road_block[2:], axis=1)
        video_traj_within = fix_pts_interpolate(video_traj[within_mask], 300)

        if np.linalg.norm(video_traj_within[-1] - video_traj_within[0]) < np.max(road_block[2:] - road_block[:2]) / 2:
            continue

        skipped = False
        for filtered_video in filtered_video_infos:
            filtered_video_traj = filtered_video['trajectory']
            filtered_within_mask = np.all(filtered_video_traj > road_block[:2], axis=1) & np.all(filtered_video_traj < road_block[2:], axis=1)
            filtered_video_traj_within = fix_pts_interpolate(filtered_video_traj[filtered_within_mask], 300)
            single_way_dist = cdist(video_traj_within, filtered_video_traj_within).min(-1).mean()
            if single_way_dist < 4:
                skipped = True
                break
        if skipped:
            continue

        video['video_idx'] = last_idx + 1
        video['video_token'] = f'{config.road_block_name}-{last_idx + 1}'
        filtered_video_infos.append(video)
        last_idx += 1

    return filtered_video_infos


def _normalize_cam_info(cam_info, lidar2ego, timestamp):
    sensor2lidar = np.eye(4, dtype=np.float64)
    sensor2lidar[:3, :3] = np.asarray(cam_info['sensor2lidar_rotation'], dtype=np.float64)
    sensor2lidar[:3, 3] = np.asarray(cam_info['sensor2lidar_translation'], dtype=np.float64)

    sensor2ego = lidar2ego @ sensor2lidar

    return {
        'data_path': cam_info['data_path'],
        'timestamp': timestamp,
        'token': os.path.splitext(os.path.basename(cam_info['data_path']))[0],
        'sensor2ego_rotation': Quaternion(matrix=sensor2ego[:3, :3]),
        'sensor2ego_translation': sensor2ego[:3, 3],
        'cam_intrinsic': np.asarray(cam_info['cam_intrinsic']),
        'distortion': np.asarray(cam_info['distortion']),
    }


def _extract_box_info(frame):
    anns = frame.get('anns', {})
    if len(anns) == 0:
        return {
            'gt_boxes': np.zeros((0, 7)),
            'gt_names': np.zeros((0,)),
            'gt_velocity': np.zeros((0, 2)),
            'gt_velocity_3d': np.zeros((0, 3)),
            'gt_confidence': np.zeros((0,)),
            'instance_tokens': np.zeros((0,)),
            'track_tokens': np.zeros((0,)),
        }

    gt_velocity_3d = np.asarray(anns['gt_velocity_3d'])
    return {
        'gt_boxes': np.asarray(anns['gt_boxes']),
        'gt_names': np.asarray(anns['gt_names']),
        'gt_velocity': gt_velocity_3d[:, :2] if gt_velocity_3d.size > 0 else np.zeros((0, 2)),
        'gt_velocity_3d': gt_velocity_3d,
        'gt_confidence': np.ones((len(anns['gt_names']),), dtype=np.float32),
        'instance_tokens': np.asarray(anns['instance_tokens']),
        'track_tokens': np.asarray(anns['track_tokens']),
    }


def produce_video_scene_dict(config, video_scene: VideoScene, video_infos):
    timezone = NUPLAN_TIMEZONE[config.city]
    road_block = np.asarray(config.road_block, dtype=np.float64)
    buffer = config.reconstruct_buffer
    buffered_road_block = road_block + np.asarray([-buffer, -buffer, buffer, buffer], dtype=np.float64)

    baseline_z = video_infos[0]['frames'][0]['ego2global_translation'][2]
    road_block_center = np.asarray([
        road_block[0] + road_block[2],
        road_block[1] + road_block[3],
        0,
    ], dtype=np.float64) / 2
    road_block_center[2] = baseline_z

    video_scene_dict = {}
    for video in video_infos:
        video_token = video['video_token']
        start_ts = video['frames'][0]['timestamp']
        video_scene_dict[video_token] = {
            'video_token': video_token,
            'log_token': video['log_token'],
            'log_name': video['log_name'],
            'map_location': video['map_location'],
            'vehicle_name': video['vehicle_name'],
            'start_ts': start_ts,
            'end_ts': video['frames'][-1]['timestamp'],
            'date': datetime.fromtimestamp(start_ts / 1e6, timezone).date(),
            'hour': datetime.fromtimestamp(start_ts / 1e6, timezone).hour,
            'global2world_translation': road_block_center,
            'frame_infos': [],
        }

        for frame_idx, frame in enumerate(video['frames']):
            ego_pose_xy = np.asarray(frame['ego2global_translation'][:2], dtype=np.float64)
            in_region = np.logical_and(
                (ego_pose_xy > buffered_road_block[:2]).all(),
                (ego_pose_xy < buffered_road_block[2:]).all(),
            )

            can_bus = np.asarray(frame['can_bus']).copy()
            can_bus[:3] -= road_block_center

            lidar_source_path = video_scene.source_lidar_path(frame['lidar_path'])
            if not os.path.exists(lidar_source_path):
                CONSOLE.log(f'LiDAR file missing for {frame["token"]}: {lidar_source_path}')
                continue

            info = {
                'skipped': "out_of_region" if not in_region else False,
                'token': frame['token'],
                'video_token': video_token,
                'frame_idx': frame_idx,
                'timestamp': frame['timestamp'],
                'log_name': video['log_name'],
                'log_token': video['log_token'],
                'can_bus': can_bus,
                'ego2global_translation': can_bus[:3],
                'ego2global_rotation': can_bus[3:7],
                'ego2global': None,
                'lidar_path': frame['lidar_path'],
                'lidar2ego_translation': np.asarray(frame['lidar2ego_translation']),
                'lidar2ego_rotation': np.asarray(frame['lidar2ego_rotation']),
                'lidar2ego': None,
                'lidar2global': None,
                'cams': {},
            }

            ego2global = Quaternion(info['ego2global_rotation']).transformation_matrix
            ego2global[:3, 3] = info['ego2global_translation']
            info['ego2global'] = ego2global

            lidar2ego = Quaternion(info['lidar2ego_rotation']).transformation_matrix
            lidar2ego[:3, 3] = info['lidar2ego_translation']
            info['lidar2ego'] = lidar2ego
            info['lidar2global'] = ego2global @ lidar2ego

            cams = {}
            camera_missing = False
            for cam_name, cam_info in frame['cams'].items():
                image_source_path = video_scene.source_image_path(cam_info['data_path'])
                if not os.path.exists(image_source_path):
                    camera_missing = True
                    CONSOLE.log(f'Camera image missing for {frame["token"]}: {image_source_path}')
                    break
                cams[cam_name] = _normalize_cam_info(cam_info, lidar2ego, frame['timestamp'])
            if camera_missing or len(cams) != 8:
                continue
            info['cams'] = cams
            info.update(_extract_box_info(frame))
            video_scene_dict[video_token]['frame_infos'].append(info)

    return video_scene_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--prefilter', action='store_true')
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    if getattr(config, 'data_source', 'nuplan') != 'navsim':
        raise ValueError('navsim_video_processing.py requires config.data_source == "navsim"')

    video_scene = VideoScene(config)
    video_infos = split_video_from_navsim_meta(config, video_scene)
    video_infos = sort_navsim_video_infos(config, video_infos)

    if args.prefilter and len(config.selected_videos) != 0:
        CONSOLE.print(f'Prefilter with selected video idx {config.selected_videos}')
        new_video_infos = []
        for idx in config.selected_videos:
            if type(idx) is int:
                new_video_infos.append(video_infos[idx])
            else:
                video_info = video_infos[idx['idx']]
                start_frame = idx.get('start_frame', 0)
                end_frame = idx.get('end_frame', -1)
                video_info['frames'] = video_info['frames'][start_frame:end_frame]
                video_info['trajectory'] = np.asarray([_frame_xy(item) for item in video_info['frames']], dtype=np.float64)
                new_video_infos.append(video_info)
        video_infos = new_video_infos

    video_scene_dict = produce_video_scene_dict(config, video_scene, video_infos)

    if isinstance(config, FrameCentralConfig):
        if config.multi_traversal_mode == 'reconstruction':
            pass
        elif config.multi_traversal_mode == 'off':
            video_scene_dict = {k: v for k, v in video_scene_dict.items() if k.endswith('-0')}
        else:
            raise ValueError(f'Unknown multi_traversal_mode: {config.multi_traversal_mode}')

    os.makedirs(os.path.dirname(video_scene.pickle_path_raw), exist_ok=True)
    with open(video_scene.pickle_path_raw, 'wb') as f:
        pickle.dump(video_scene_dict, f)
    video_scene.update_pickle_link(video_scene.pickle_path_raw)
