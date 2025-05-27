#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import os
import argparse
import jsonlines
import pickle
from datetime import datetime
from tqdm import tqdm

import numpy as np
from scipy.spatial.distance import cdist
from pyquaternion import Quaternion
from torch.utils.data import Dataset, DataLoader

from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB
from nuplan.database.nuplan_db_orm.lidar_pc import LidarPc

from nuplan_scripts.utils.config import load_config, FrmaeCentralConfig
from nuplan_scripts.utils.video_scene_dict_tools import VideoScene
from nuplan_scripts.utils.nuplan_utils_custom import (
    CanBus, get_closest_start_idx, get_cam_info_from_lidar_pc, get_box_info_from_lidar_pc, fix_pts_interpolate
)

from nuplan_scripts.utils.constants import (
    CONSOLE, NUPLAN_DATA_ROOT, NUPLAN_DB_FILES, NUPLAN_SENSOR_ROOT, NUPLAN_TIMEZONE
)


class SplitVideoFromNuPlanDB:

    class PseudoDataset(Dataset):

        def __init__(self, config, db_filenames):
            self.config = config
            self.city = config.city
            self.road_block = np.array(config.road_block)
            self.expand_buffer = config.expand_buffer
            self.interval = config.interval
            self.db_filenames = db_filenames

        def __getitem__(self, idx):
            db_filename = self.db_filenames[idx]
            assert os.path.exists(db_filename), f'{db_filename} not found!'
            log_db = NuPlanDB(NUPLAN_DATA_ROOT, db_filename, None, verbose=True)
            log = log_db.log
            video_infos = []

            if log.map_version != self.city:
                return None, []

            lidar_pcs = log.lidar_pcs
            lidar_pcs.sort(key=lambda x: x.timestamp)
            start_idx = get_closest_start_idx(log, lidar_pcs)

            lidar_pcs_new = lidar_pcs[start_idx::2 * self.interval]
            video_info = {
                'video_token': '',
                'log_token': log.token,
                'log_name': log.logfile,
                'start_ts': None,
                'lidar_pcs': [],
            }

            trajectory = np.asarray([
                [lidar_pc.ego_pose.x, lidar_pc.ego_pose.y]
                for lidar_pc in lidar_pcs_new
            ])

            # Check if any part of trajectory is within the road block
            in_region = np.all(trajectory > self.road_block[:2], axis=1) & np.all(trajectory < self.road_block[2:], axis=1)
            if not in_region.any():
                return None, []

            def create_new_video_info():
                return {
                    'video_token': '',
                    'log_token': log.token,
                    'log_name': log.logfile,
                    'start_ts': None,
                    'lidar_pcs': []
                }

            video_info = create_new_video_info()
            # Group continual in-region frames into separate video infos
            for idx, (is_in_region, lidar_pc) in enumerate(zip(in_region, lidar_pcs_new)):
                if is_in_region:
                    # Start new video info if previous frame was out of region
                    if idx > 0 and not in_region[idx - 1] and len(video_info['lidar_pcs']) > 1:
                        video_info['start_ts'] = video_info['lidar_pcs'][0].timestamp
                        video_info['trajectory'] = np.asarray(
                            [[lidar_pc.ego_pose.x, lidar_pc.ego_pose.y] for lidar_pc in video_info['lidar_pcs']])
                        video_infos.append(video_info)
                        video_info = create_new_video_info()

                    video_info['lidar_pcs'].append(lidar_pc)

            # Add final video info if it contains frames
            if len(video_info['lidar_pcs']) > 1:
                video_info['start_ts'] = video_info['lidar_pcs'][0].timestamp
                video_info['trajectory'] = np.asarray(
                    [[lidar_pc.ego_pose.x, lidar_pc.ego_pose.y] for lidar_pc in video_info['lidar_pcs']])
                video_infos.append(video_info)

            for video_info in video_infos:
                video_info['lidar_pcs'] = self.expand_trajectory(lidar_pcs_new, video_info['lidar_pcs'])

            log_db.session.close()
            return log_db, video_infos

        def expand_trajectory(self, lidar_pcs, sub_lidar_pcs):

            first_idx = lidar_pcs.index(sub_lidar_pcs[0])
            last_idx = lidar_pcs.index(sub_lidar_pcs[-1])
            # Calculate distance along trajectory
            def calc_distance(pose1, pose2):
                return np.linalg.norm(np.array([pose1.x, pose1.y]) - np.array([pose2.x, pose2.y]))

            # Expand backwards
            expanded_start_idx = first_idx
            cumulative_dist = 0
            for i in range(first_idx-1, -1, -1):
                dist = calc_distance(lidar_pcs[i].ego_pose, lidar_pcs[i+1].ego_pose)
                cumulative_dist += dist
                if cumulative_dist >= self.expand_buffer:
                    expanded_start_idx = i
                    break

            # Expand forwards
            expanded_end_idx = last_idx
            cumulative_dist = 0
            for i in range(last_idx, len(lidar_pcs)-1):
                dist = calc_distance(lidar_pcs[i].ego_pose, lidar_pcs[i+1].ego_pose)
                cumulative_dist += dist
                if cumulative_dist >= self.expand_buffer:
                    expanded_end_idx = i
                    break

            expanded_lidar_pcs = lidar_pcs[expanded_start_idx:expanded_end_idx+1]

            return expanded_lidar_pcs

        def __len__(self):
            return len(self.db_filenames)

    @staticmethod
    def run(
        config,
        db_filenames, 
        num_workers=8):
        dataset = SplitVideoFromNuPlanDB.PseudoDataset(config, db_filenames)
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: x, pin_memory=False, drop_last=False)

        video_infos = []
        log_dbs = []
        for batch in tqdm(dataloader, desc='Splitting video from log_dbs', ncols=120):
            log_db, batch_video_infos = batch[0]
            video_infos.extend(batch_video_infos)
            log_dbs.append(log_db)
        return log_dbs, video_infos

def sort_video_infos(config, video_infos):
    if config.__class__.__name__ == 'RoadBlockConfig':
        # sort video_infos by its start_ts
        video_infos = sorted(video_infos, key=lambda x: x['start_ts'])
        for idx, video in enumerate(video_infos):
            video['video_idx'] = idx
            video['video_token'] = f'{config.road_block_name}-{idx}'
        return video_infos

    elif config.__class__.__name__ == 'FrmaeCentralConfig':
        # find central video
        central_log = config.central_log
        central_token = config.central_tokens[0]
        central_video_info = None
        for video in video_infos:
            if central_video_info is not None:
                break
            if video['log_name'] != central_log:
                continue
            for lidar_pc in video['lidar_pcs']:
                if lidar_pc.token == central_token:
                    central_video_info = video
                    break
        assert central_video_info is not None, 'Central video not found in video_infos!'

        central_video_info['video_idx'] = 0
        central_video_info['video_token'] = f'{config.road_block_name}-0'

        central_timestamp = central_video_info['start_ts']
        # sort video_infos by the timestamp distance to the central video
        video_infos = sorted(video_infos, key=lambda x: abs(x['start_ts'] - central_timestamp))
        video_infos = video_infos[1:] # remove the central video

        road_block = np.array(config.road_block)
        filtered_video_infos = [central_video_info]
        last_idx = 0
        for video in video_infos:
            video_traj = video['trajectory']
            within_mask = np.all(video_traj > road_block[:2], axis=1) & np.all(video_traj < road_block[2:], axis=1)
            video_traj_within = video_traj[within_mask]
            video_traj_within = fix_pts_interpolate(video_traj_within, 300)

            # filter out short trajectory
            if np.linalg.norm(video_traj_within[-1] - video_traj_within[0]) < np.max(road_block[2:] - road_block[:2]) / 2:
                continue

            skipped = False
            for filtered_video in filtered_video_infos:
                filtered_video_traj = filtered_video['trajectory']
                within_mask = np.all(filtered_video_traj > road_block[:2], axis=1) & np.all(filtered_video_traj < road_block[2:], axis=1)
                filtered_video_traj_within = filtered_video_traj[within_mask]
                filtered_video_traj_within = fix_pts_interpolate(filtered_video_traj_within, 300)
                
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

class ProduceVideoSceneDict:

    class PseudoDataset(Dataset):

        def __init__(self, config, log_token2log_db, video_infos):
            self.config = config
            self.video_infos = video_infos
            self.log_token2log_db = log_token2log_db
            self.timezone = NUPLAN_TIMEZONE[config.city]
            self.road_block = np.array(config.road_block)
            buffer = config.reconstruct_buffer
            self.buffered_road_block = self.road_block + np.array([-buffer, -buffer, buffer, buffer])

            baseline_z = video_infos[0]['lidar_pcs'][0].ego_pose.z
            self.road_block_center = np.array(
                [self.road_block[0] + self.road_block[2], self.road_block[1] + self.road_block[3], 0]
            ) / 2
            self.road_block_center[2] = baseline_z

        def __getitem__(self, idx):
            video = self.video_infos[idx]
            video_token = video['video_token']
            log_db = self.log_token2log_db[video['log_token']]
            start_ts = video['lidar_pcs'][0].timestamp

            video_scene_dict = {}
            video_scene_dict[video_token] = {
                'video_token': video_token,
                'log_token': video['log_token'],
                'log_name': video['log_name'],
                'map_location': log_db.log.map_version,
                'vehicle_name': log_db.log.vehicle_name,
                'start_ts': start_ts,
                'end_ts': video['lidar_pcs'][-1].timestamp,
                'date': datetime.fromtimestamp(start_ts / 1e6, self.timezone).date(),
                'hour': datetime.fromtimestamp(start_ts / 1e6, self.timezone).hour,
                # NOTE: 'global' is the local coordinate of the road block, 'world' is refer to the city coordinate
                'global2world_translation': self.road_block_center,
                'frame_infos': []
            }

            for frame_idx, lidar_pc in enumerate(video['lidar_pcs']):
                lidar_pc: LidarPc

                if lidar_pc._session is None:
                    log_db.session.add(lidar_pc)

                lidar_pc_token = lidar_pc.token
                pc_file_name = lidar_pc.filename
                time_stamp = lidar_pc.timestamp

                ego_pose_xy = np.array([lidar_pc.ego_pose.x, lidar_pc.ego_pose.y])
                in_region = np.logical_and(
                    (ego_pose_xy > self.buffered_road_block[:2]).all(),
                    (ego_pose_xy < self.buffered_road_block[2:]).all()
                )

                can_bus = CanBus(lidar_pc).tensor
                # NOTE: change to relative coordinate
                can_bus[:3] -= self.road_block_center

                lidar = lidar_pc.lidar
                pc_file_path = os.path.join(NUPLAN_SENSOR_ROOT, pc_file_name)
                if not os.path.exists(pc_file_path):
                    tqdm.write(f'LiDAR file missing for {lidar_pc_token}')
                    continue

                info = {
                    'skipped': "out_of_region" if not in_region else False,
                    'token': lidar_pc_token,
                    'video_token': video_token,
                    'frame_idx': frame_idx,
                    'timestamp': time_stamp,
                    'log_name': video['log_name'],
                    'log_token': video['log_token'],
                    'can_bus': can_bus,
                    'ego2global_translation': can_bus[:3],
                    'ego2global_rotation': can_bus[3:7],
                    'ego2global': None,
                    'lidar_path': lidar_pc.filename,
                    'lidar2ego_translation': np.array(lidar.translation_np),
                    'lidar2ego_rotation': np.array([lidar.rotation.w, lidar.rotation.x, lidar.rotation.y, lidar.rotation.z]),
                    'lidar2ego': None,
                    'lidar2global': None,
                    'cams': dict(),
                }
                ego2global = Quaternion(info['ego2global_rotation']).transformation_matrix
                ego2global[:3, 3] = info['ego2global_translation']
                info['ego2global'] = ego2global

                lidar2ego = Quaternion(info['lidar2ego_rotation']).transformation_matrix
                lidar2ego[:3, 3] = info['lidar2ego_translation']
                info['lidar2ego'] = lidar2ego

                lidar2global = np.dot(ego2global, lidar2ego)
                info['lidar2global'] = lidar2global

                cams = get_cam_info_from_lidar_pc(
                    log_db.log, lidar_pc,
                    rolling_shutter_s=1/60
                )
                info['cams'] = cams
                if cams is None:
                    tqdm.write(f'Camera info broken for {lidar_pc_token}')
                    continue

                box_info = get_box_info_from_lidar_pc(lidar_pc, with_parking_cars=True)
                info.update(box_info)
                video_scene_dict[video_token]['frame_infos'].append(info)

            return video_scene_dict

        def __len__(self):
            return len(self.video_infos)

    @staticmethod
    def run(
        config,
        log_token2log_db, 
        video_infos,
        num_workers=8):
        dataset = ProduceVideoSceneDict.PseudoDataset(
            config, log_token2log_db, video_infos
        )
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: x, pin_memory=False, drop_last=False)

        video_scene_dict = {}
        for batch in tqdm(dataloader, desc='Processing Videos', ncols=120):
            video_scene_dict.update(batch[0])

        return video_scene_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument(
        '--prefilter',
        action='store_true',
        help='prefilter the video_infos by the selected_videos in the config.')
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)

    road_block_name = config.road_block_name
    data_root = config.data_root
    road_block = np.asarray(config.road_block)
    split = config.split

    with jsonlines.open(f'{data_root}/nuplan_log_infos.jsonl') as reader:
        log_name2lidar_pc_token = {item['log_name']: item for item in reader}

    if isinstance(config, FrmaeCentralConfig) and config.multi_traversal_mode == 'off':
        # only keep the central traversal.
        log_name2lidar_pc_token = {config.central_log: log_name2lidar_pc_token[config.central_log]}

    # filter db_filenames by pre-recorded log traj.
    in_range_db_filenames = []
    for log_name in log_name2lidar_pc_token:
        split = log_name2lidar_pc_token[log_name]['split']
        if config.split != 'all' and split != config.split:
            continue
        log_traj = log_name2lidar_pc_token[log_name]['trajectory']
        in_region = np.all(log_traj > road_block[:2], axis=1) & np.all(log_traj < road_block[2:], axis=1)
        if in_region.any():
            log_filepath = os.path.join(NUPLAN_DB_FILES, f'{log_name}.db')
            in_range_db_filenames.append(log_filepath)

    log_dbs, video_infos = SplitVideoFromNuPlanDB.run(
        config = config,
        db_filenames = in_range_db_filenames,
        num_workers = args.num_workers
    )

    log_token2log_db = {}
    for log_db in log_dbs:
        if log_db is not None:
            log_token2log_db[log_db.log.token] = log_db

    video_infos = sort_video_infos(config, video_infos)

    if args.prefilter and len(config.selected_videos) != 0:
        CONSOLE.print(f'Prefilter with selected video idx {config.selected_videos}')
        # NOTE: this may not always consistent with filter_trajectory.py, when specify start_frame and end_frame.
        # because sometime the lidar_pc will be skipped, when images or lidar are not available.
        new_video_infos = []
        for idx in config.selected_videos:
            if type(idx) is int:
                new_video_infos.append(video_infos[idx])
            else:
                video_info = video_infos[idx['idx']]
                start_frame = idx.get('start_frame', 0)
                end_frame = idx.get('end_frame', -1)
                video_info['lidar_pcs'] = video_info['lidar_pcs'][start_frame:end_frame]
                new_video_infos.append(video_info)
        video_infos = new_video_infos

    video_scene_dict = ProduceVideoSceneDict.run(
        config=config,
        log_token2log_db = log_token2log_db,
        video_infos = video_infos,
        num_workers = args.num_workers
    )

    if isinstance(config, FrmaeCentralConfig):
        if config.multi_traversal_mode == 'reconstruction':
            # keep the multi_traversal infos.
            # use them to reconstruct the whole road block.
            pass
        elif config.multi_traversal_mode == 'off':
            # only keep the central traversal.
            video_scene_dict = {k: v for k, v in video_scene_dict.items() if k.endswith('-0')}
        else:
            raise ValueError(f'Unknown multi_traversal_mode: {config.multi_traversal_mode}')

    video_scene = VideoScene(config)
    os.makedirs(os.path.dirname(video_scene.pickle_path_raw), exist_ok=True)
    with open(video_scene.pickle_path_raw, 'wb') as f:
        pickle.dump(video_scene_dict, f)
    video_scene.update_pickle_link(video_scene.pickle_path_raw)
