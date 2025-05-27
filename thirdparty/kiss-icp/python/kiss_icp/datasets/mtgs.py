#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import os
from pathlib import Path

import numpy as np
import cv2
from pyquaternion import Quaternion

from nuplan_scripts.utils.video_scene_dict_tools import VideoScene
from nuplan_scripts.utils.nuplan_utils_custom import load_lidar, get_semantic_point_cloud
from nuplan_scripts.utils.camera_utils import undistort_image_with_cam_info
from nuplan_scripts.utils.constants import NUPLAN_SENSOR_ROOT

class MTGSDataset:

    def __init__(
            self, 
            data_dir: Path, 
            video_scene: VideoScene, 
            video_order: tuple = tuple(), 
            filter_semantic = False,
            *_, 
            **__):
        self.scans = []
        self.traj_id = []
        self.gt_poses = []
        self.tokens = []
        self.sequence_id = os.path.basename(data_dir)
        # filter lidar points with semanti label, may leads to worse performance
        self.filter_semantic = filter_semantic

        self.video_scene = video_scene
        self.video_scene_dict = video_scene.video_scene_dict
        self.video_order = video_order
        if len(self.video_order) == 0:
            self.video_order = [int(video_token.split('-')[-1]) for video_token in self.video_scene_dict]

        self._load_data()

    def _load_data(self):

        idx2video_scene_dict = {}
        for video_token in self.video_scene_dict:
            video_idx = int(video_token.split('-')[-1])
            idx2video_scene_dict[video_idx] = self.video_scene_dict[video_token]

        for idx in self.video_order:
            frame_infos = idx2video_scene_dict[idx]["frame_infos"]
            for info in frame_infos:
                lidar_path = os.path.join(NUPLAN_SENSOR_ROOT, info['lidar_path'])
                if self.filter_semantic:
                    lidar_points = self.get_filtered_lidar(info)
                else:
                    lidar_points = load_lidar(lidar_path, remove_close=False, only_top=True)
                self.scans.append(lidar_points)
                self.traj_id.append(idx)
                self.gt_poses.append(info['lidar2global'])
                self.tokens.append(info['token'])

        self.first_pose = self.gt_poses[0]
        self.gt_poses = np.array(self.gt_poses)
        self.gt_poses[:, :3, 3] = self.gt_poses[:, :3, 3] - self.first_pose[:3, 3].reshape(1, 3)

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, idx):
        return self.scans[idx]

    def get_filtered_lidar(self, info):
        undistorted_sem_masks = []
        lidar2imgs = []
        lidar2ego = info['lidar2ego']
        for cam in info['cams']:
            cam_info = info['cams'][cam]
            if 'colmap_param' in cam_info:
                intrinsic = cam_info['colmap_param']['cam_intrinsic']
            else:
                intrinsic = cam_info['cam_intrinsic']

            sem_mask_path = os.path.join(
                self.video_scene.raw_mask_path,
                self.video_scene.mask_suffix_cityscape,
                cam_info['data_path'].replace('.jpg', '.png')
            )

            assert os.path.exists(sem_mask_path), f'{sem_mask_path} does not exist.'
            sem_mask = cv2.imread(sem_mask_path, cv2.IMREAD_GRAYSCALE)
            sem_mask = undistort_image_with_cam_info(sem_mask, cam_info, interpolation='nearest', mode='keep_focal_length')[:, :, None]  # [H, W, 1]
            undistorted_sem_masks.append(sem_mask)

            ego2cam = np.eye(4)
            ego2cam[:3, :3] = Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix.T
            ego2cam[:3, 3] = -ego2cam[:3, :3] @ cam_info['sensor2ego_translation']
            lidar2cam = ego2cam @ lidar2ego

            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img = viewpad @ lidar2cam
            lidar2imgs.append(lidar2img)
        
        undistorted_sem_masks = np.array(undistorted_sem_masks)  # (n_cam, H, W, 1)
        lidar2imgs = np.array(lidar2imgs)  # (n_cam, 4, 4)
        lidar_points = load_lidar(
            os.path.join(NUPLAN_SENSOR_ROOT, info['lidar_path']), remove_close=False, only_top=True)
        sem_labels, fov_mask_sem = get_semantic_point_cloud(lidar_points, lidar2imgs, undistorted_sem_masks)
        # filter out points in sky, person, rider, car, truck, bus, motorcycle, bicycle
        mask = np.logical_and(fov_mask_sem, sem_labels < 10)
        return lidar_points[mask]

    def apply_calibration(self, poses: np.ndarray) -> np.ndarray:
        poses = poses.copy()
        poses[:, :3, 3] = poses[:, :3, 3] + self.first_pose[:3, 3].reshape(1, 3)
        return poses
