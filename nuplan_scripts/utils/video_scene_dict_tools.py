#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import os
import copy
import pickle
import numpy as np
from nuplan_scripts.utils.constants import CONSOLE

from .config import RoadBlockConfig
from .constants import NUPLAN_SENSOR_ROOT

class VideoScene:

    def __init__(self, config: RoadBlockConfig):
        self.config = config

    def load_pickle(self, path, verbose=True):
        if verbose:
            CONSOLE.log(f'Loading pickle from {path}')
        with open(path, 'rb') as f:
            self.video_scene_dict = pickle.load(f)
        return self.video_scene_dict

    def dump_pickle(self, path, verbose=True):
        if verbose:
            CONSOLE.log(f'Saving pickle to {path}')
        with open(path, 'wb') as f:
            pickle.dump(self.video_scene_dict, f)

    def update_pickle_link(self, path, verbose=True):
        if verbose:
            CONSOLE.log(f'Linking {os.path.basename(path)} to {os.path.basename(self.pickle_path)}')
        if os.path.exists(self.pickle_path) or os.path.islink(self.pickle_path):
            os.unlink(self.pickle_path)
        os.symlink(os.path.basename(path), self.pickle_path)

    def video_scene_dict_process(self, scene_filters, inline=False):

        if not isinstance(scene_filters, list):
            scene_filters = [scene_filters]

        video_scene_dict = copy.deepcopy(self.video_scene_dict)
        for scene_filter in scene_filters:
            if isinstance(scene_filter, str):
                assert scene_filter in SCENE_DICT_FACTORY
                filter_func = SCENE_DICT_FACTORY[scene_filter]
                video_scene_dict = filter_func(video_scene_dict)
            elif isinstance(scene_filter, dict):
                assert scene_filter['type'] in SCENE_DICT_FACTORY
                filter_func = SCENE_DICT_FACTORY[scene_filter['type']]
                video_scene_dict = filter_func(video_scene_dict, **scene_filter['kwargs'])

        if inline:
            self.video_scene_dict = video_scene_dict

        return video_scene_dict

    def total_frames(self):
        return sum([len(self.video_scene_dict[video_token]['frame_infos']) for video_token in self.video_scene_dict])

    @staticmethod
    def count_total_frames(video_scene_dict):
        return sum([len(video_scene_dict[video_token]['frame_infos']) for video_token in video_scene_dict])

    @property
    def block_size(self):
        return (
            self.config.road_block[2] - self.config.road_block[0],
            self.config.road_block[3] - self.config.road_block[1]
        )

    @property
    def road_block_center(self):
        return np.array([
            self.config.road_block[0] + self.config.road_block[2], 
            self.config.road_block[1] + self.config.road_block[3], 
            0
        ], dtype=np.float32) / 2

    @property
    def name(self):
        return self.config.road_block_name

    @property
    def data_root(self):
        return self.config.data_root

    @property
    def sub_data_root(self):
        return f'{self.data_root}/{self.name}'

    @property
    def pickle_path(self):
        return f'{self.sub_data_root}/video_scene_dict.pkl'

    @property
    def pickle_path_raw(self):
        return f'{self.sub_data_root}/video_scene_dict_raw.pkl'

    @property
    def pickle_path_filtered(self):
        return f'{self.sub_data_root}/video_scene_dict_filtered.pkl'

    @property
    def pickle_path_registered(self):
        return f'{self.sub_data_root}/video_scene_dict_registered.pkl'

    @property
    def pickle_path_colmap(self):
        return f'{self.sub_data_root}/video_scene_dict_colmap.pkl'

    @property
    def pickle_path_final(self):
        return f'{self.sub_data_root}/video_scene_dict_final.pkl'

    @property
    def raw_video_path(self):
        return f'{self.sub_data_root}/videos/raw'

    @property
    def registration_path(self):
        return f'{self.sub_data_root}/registration_results'

    @property
    def raw_lidar_path(self):
        if self.config.collect_raw:
            return f'{self.sub_data_root}/lidar'
        else:
            return NUPLAN_SENSOR_ROOT

    @property
    def raw_image_path(self):
        if self.config.collect_raw:
            return f'{self.sub_data_root}/images/raw'
        else:
            return NUPLAN_SENSOR_ROOT

    @property
    def undistorted_image_path(self):
        return f'{self.sub_data_root}/images/undistorted'

    @property
    def raw_mask_path(self):
        return f'{self.sub_data_root}/masks/raw'

    @property
    def undistorted_mask_path(self):
        return f'{self.sub_data_root}/masks/undistorted'

    mask_suffix_cityscape = 'cityscape'
    mask_suffix_cityscape_pano = 'cityscape_pano'
    mask_suffix_ego = 'ego'
    mask_suffix_box = 'box'
    mask_suffix_road = 'road'
    mask_suffix_sky = 'sky'
    mask_suffix_foreground = 'foreground'
    mask_suffix_colmap = 'colmap'

    @property
    def undistorted_depth_path(self):
        return f'{self.sub_data_root}/depth/undistorted'

    @property
    def optimal_undistorted_depth_path(self):
        return f'{self.sub_data_root}/depth/undistorted_optimal'

    @property
    def rgb_point_cloud_path(self):
        return f'{self.sub_data_root}/rgb_point_cloud'

    @property
    def sfm_point_cloud_path(self):
        return f'{self.sub_data_root}/sfm_point_cloud'

    @property
    def colmap_path(self):
        return f'{self.sub_data_root}/colmap'

    @property
    def instance_point_cloud_path(self):
        return f'{self.sub_data_root}/instance_point_cloud'

class Factory:
    def __init__(self):
        self._registry = dict()

    def register(self, name):
        def inner_wrapper(wrapped_class):
            self._registry[name] = wrapped_class
            return wrapped_class
        return inner_wrapper

    def create(self, name, *args, **kwargs):
        if name not in self._registry:
            raise ValueError(f"Invalid name {name}")
        return self._registry[name](*args, **kwargs)

    def __getitem__(self, name):
        return self._registry[name]

    def __contains__(self, name):
        return name in self._registry

    def __iter__(self):
        return iter(self._registry)

    def __len__(self):
        return len(self._registry)

    def __repr__(self):
        return f"{self.__class__.__name__}({self._registry!r})"

SCENE_DICT_FACTORY = Factory()

@SCENE_DICT_FACTORY.register("filter_by_video_idx")
def filter_by_video_idx(video_scene_dict, video_idxs):
    CONSOLE.log(f'Select videos {video_idxs}')
    if len(video_idxs) == 0:
        CONSOLE.log('No video selected, skipping...')
        return video_scene_dict

    idx_map = {}
    for video_token in video_scene_dict:
        video_idx = int(video_token.split('-')[-1])
        idx_map[video_idx] = video_scene_dict[video_token]

    video_scene_dict_filtered = {}
    for idx in video_idxs:
        if isinstance(idx, int):
            video_info = idx_map[idx]
            token = video_info['video_token']
            video_scene_dict_filtered[token] = video_info
        else:
            video_info = idx_map[idx['idx']]
            start_frame = idx.get('start_frame', 0)
            end_frame = idx.get('end_frame', -1)
            video_info['frame_infos'] = video_info['frame_infos'][start_frame:end_frame]
            token = video_info['video_token']
            video_scene_dict_filtered[token] = video_info

    return video_scene_dict_filtered

@SCENE_DICT_FACTORY.register("filter_low_velocity")
def filter_low_velocity(video_scene_dict):
    video_scene_dict_filtered = {}
    for video_token in video_scene_dict:
        frame_infos = video_scene_dict[video_token]['frame_infos']
        last_pose = frame_infos[0]['can_bus'][0:3]
        skip_count = 10
        for info in frame_infos:
            if np.linalg.norm(info['can_bus'][0:3] - last_pose) < 0.3 and skip_count < 10:
                info['skipped'] = "low_velocity"
                skip_count += 1
                continue
            skip_count = 0
            last_pose = info['can_bus'][0:3]

        count_after_skipped = sum([1 for info in frame_infos if not info.get('skipped', False)])
        CONSOLE.log(f'filter_low_velocity: {video_token}, {len(frame_infos)} -> {count_after_skipped}')

        video_scene_dict_filtered[video_token] = video_scene_dict[video_token]
    return video_scene_dict_filtered

@SCENE_DICT_FACTORY.register("inject_trajectory")
def inject_trajectory(video_scene_dict):
    for video_token in video_scene_dict:
        traj = [info['ego2global_translation'] for info in video_scene_dict[video_token]['frame_infos']]
        video_scene_dict[video_token]['trajectory'] = np.asarray(traj) # (n, 3)
    return video_scene_dict

@SCENE_DICT_FACTORY.register("filter_skipped_frames")
def filter_skipped_frames(video_scene_dict):
    for video_token in video_scene_dict:
        frame_infos = video_scene_dict[video_token]['frame_infos']
        frame_infos = [info for info in frame_infos if not info.get('skipped', False)]
        video_scene_dict[video_token]['frame_infos'] = frame_infos
    return video_scene_dict
