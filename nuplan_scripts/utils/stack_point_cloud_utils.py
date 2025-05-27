#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import os
import numpy as np
import torch

from .nuplan_utils_custom import load_lidar
from .constants import NUPLAN_SENSOR_ROOT

TORCH_DEVICE = 'cpu'

class InstanceObjectTrack():
    def __init__(self, instance_id, frame_id=0, instance_object=None):
        self.instance_id = instance_id
        self.track_dict = {} 
        self.track_dict[frame_id] = instance_object
        self.class_name = instance_object.class_name
        self.accu_points = instance_object.points
        self.accu_rgbs = instance_object.rgbs
        self.accu_sem_labels = instance_object.sem_labels
        self.start_position = instance_object.global_center
        self.end_position = instance_object.global_center
    
    def add(self, frame_id, instance_object):
        self.track_dict[frame_id] = instance_object
        self.end_position = instance_object.global_center

    def __repr__(self):
        return 'track instance_id: {}'.format(self.instance_id)

class InstanceObject():
    """define the object instance"""
    def __init__(self, instance_id, class_name, gt_box, lidar2global, velocity=None):
        self.instance_id = instance_id
        self.class_name = class_name
        xc, yc, zc, length, width, height, theta = gt_box  # defined in the lidar coordinate
        self.length = length
        self.width = width
        self.height = height
        self.xc, self.yc, self.zc = xc, yc, zc 
        # pose: box2lidar
        self.pose = np.eye(4)
        self.pose[:3, :3] = np.array([[np.cos(theta), -np.sin(theta), 0],
                                      [np.sin(theta), np.cos(theta),  0],
                                      [0,             0,              1]])
        self.pose[:3, 3] = np.array([xc, yc, zc])

        assert len(velocity) == 2
        if velocity is None:
            self.vel_x, self.vel_y = 0, 0
        elif np.isnan(velocity).any():
            self.vel_x, self.vel_y = 0, 0
        else:
            self.vel_x = velocity[0]
            self.vel_y = velocity[1]

        # find the global coordinate of the box center
        self.global_center = self.get_global_center(xc, yc, zc, lidar2global)
        self.points = None
        self.rgbs = None
        self.sem_labels = None

    @property
    def points_ego(self):
        return self.points @ self.pose[:3, :3].T + self.pose[:3, 3]
    
    def get_global_center(self, xc, yc, zc, lidar2global):
        point = np.array([xc, yc, zc, 1]).reshape(4, 1)
        global_center = np.dot(lidar2global, point).squeeze()
        return global_center[:3]

    def __repr__(self):
        return 'track instance_id: {}'.format(self.instance_id)

def padding_column(points):
    # padding 1 in the last column: change the dimension of points from (n, 3) to (n, 4)
    ones = np.ones_like(points[:, 0]).reshape(-1, 1)
    homo_points = np.concatenate((points, ones), axis=-1)
    return homo_points

def split_points(background_points, instance_object: InstanceObject, fill_scale=0.25):
    """
    Args:
        background_points: (n, 4)
    Returns:
        background_points: (n, 4) background point defined in the lidar system
        points_in_box: (n, 3) in_box point defined in the object system
    """
    points = background_points

    box2lidar = instance_object.pose  # point in box-system to point in lidar-sytem
    lidar2box = np.linalg.inv(box2lidar)  # point in lidar-system to point in box-system
    lidar2box = points.new_tensor(lidar2box)

    point_in_box_sys = background_points @ lidar2box.T

    length, width, height = instance_object.length, instance_object.width, instance_object.height

    length_mask = point_in_box_sys[:, 0].abs() <= (length + fill_scale) / 2
    width_mask = point_in_box_sys[:, 1].abs() <= (width + fill_scale) / 2
    height_mask = point_in_box_sys[:, 2].abs() <= (height + fill_scale) / 2
    mask = length_mask & width_mask & height_mask
    background_points = points[torch.logical_not(mask)]

    length_mask = point_in_box_sys[:, 0].abs() <= (length) / 2
    width_mask = point_in_box_sys[:, 1].abs() <= (width) / 2
    height_mask = point_in_box_sys[:, 2].abs() <= (height) / 2

    mask = length_mask & width_mask & height_mask
    points_in_box = point_in_box_sys[mask][:, :3].cpu().numpy()
    return background_points, points_in_box

def extract_frame_background_instance_lidar(info, l2g=True, points=None):
    if points is None:
        background_points = load_lidar(
            os.path.join(NUPLAN_SENSOR_ROOT, info['lidar_path']), remove_close=True)
    else:
        background_points = points

    instance_tokens = info['track_tokens']
    gt_boxes = info['gt_boxes'] 
    gt_names = info['gt_names']
    lidar2global = info['lidar2global']
    gt_velocitys = info['gt_velocity']
    back_instance_info = {}
    back_instance_info['instance'] = {}
    back_instance_info['raw_points'] = background_points.copy()
    background_points = torch.tensor(background_points, dtype=torch.float64, device=TORCH_DEVICE)
    background_points = torch.cat([background_points, torch.ones_like(background_points[:, 0:1])], dim=-1)

    for i in range(len(instance_tokens)):
        instance_token = instance_tokens[i] 
        class_name = gt_names[i] 
        if class_name in ['traffic_cone', 'barrier', 'czone_sign']:
            continue

        gt_box = gt_boxes[i] 
        gt_velocity = gt_velocitys[i]
        instance_object = InstanceObject(instance_token, class_name, gt_box, lidar2global, velocity=gt_velocity)
        background_points, instance_points = split_points(background_points, instance_object)
        instance_object.points = instance_points
        back_instance_info['instance'][instance_token] = instance_object

    # transfer background points from lidar to global
    background_points = background_points[:, :3].cpu().numpy()

    if l2g:
        background_points = transform_points_lidar2global(background_points, lidar2global)

    back_instance_info['background_points'] = background_points

    return back_instance_info

def accumulate_box_point(instance_object_track: InstanceObjectTrack):
    accu_points = []
    track_dict = instance_object_track.track_dict
    for frame_idx in track_dict:
        accu_points.append(track_dict[frame_idx].points)
    accu_points = np.concatenate(accu_points, axis=0)
    instance_object_track.accu_points = accu_points

    if instance_object_track.accu_rgbs is not None:
        accu_rgbs = []
        for frame_idx in track_dict:
            accu_rgbs.append(track_dict[frame_idx].rgbs)
        accu_rgbs = np.concatenate(accu_rgbs, axis=0)
        instance_object_track.accu_rgbs = accu_rgbs
        assert accu_rgbs.shape[0] == accu_points.shape[0]
    
    if instance_object_track.accu_sem_labels is not None:
        accu_sem_labels = []
        for frame_idx in track_dict:
            accu_sem_labels.append(track_dict[frame_idx].sem_labels)
        accu_sem_labels = np.concatenate(accu_sem_labels, axis=0)
        instance_object_track.accu_sem_labels = accu_sem_labels
        assert accu_sem_labels.shape[0] == accu_points.shape[0]
    return 

def accumulate_background_point(background_track, scene_info):
    accu_global_points = []
    accu_global_rgbs = []
    accu_global_sem_labels = []
    for frame_idx in background_track.keys():
        points = background_track[frame_idx]
        if isinstance(points, tuple):
            points, rgbs, sem_labels = points
            accu_global_rgbs.append(rgbs)
            accu_global_sem_labels.append(sem_labels)
        accu_global_points.append(points)
    accu_global_points = np.concatenate(accu_global_points, axis=0)
    accu_global_rgbs = np.concatenate(accu_global_rgbs, axis=0)
    accu_global_sem_labels = np.concatenate(accu_global_sem_labels, axis=0)
    if len(accu_global_rgbs) > 0:
        accu_global_points = (accu_global_points, accu_global_rgbs, accu_global_sem_labels)
    return accu_global_points

def accumulate_background_box_point(scene_info, l2g=False):
    """accumulate the point in a sequence with about 40 frames"""
    instance_track = {} # {instance_id: InstanceObjectTrack}
    background_track = {}  # {frame_id: points, 'acumulate': points}

    # 1. generate the object track and background track in the whole scene
    for info in scene_info:
        # basic_info = info
        frame_idx = info['frame_idx']
        back_instance_info = info.pop('back_instance_info')
        background_points = back_instance_info['background_points']

        if l2g:
            background_points = transform_points_lidar2global(background_points, info['lidar2global'])

        background_points = [background_points]
        if 'background_RGBs' in back_instance_info:
            background_RGBs = back_instance_info['background_RGBs']
            background_points.append(background_RGBs)

        if 'background_sem_labels' in back_instance_info:
            background_sem_labels = back_instance_info['background_sem_labels']
            background_points.append(background_sem_labels)

        background_track[frame_idx] = tuple(background_points)

        for instance_id in back_instance_info['instance']:
            instance_object = back_instance_info['instance'][instance_id]
            if instance_id not in instance_track:
                instance_track[instance_id] = InstanceObjectTrack(instance_id, frame_idx, instance_object)
            instance_track[instance_id].add(frame_idx, instance_object)

    # 2. accumulate object point cloud in the object system
    for instance_id in instance_track:
        instance_object_track = instance_track[instance_id]
        accumulate_box_point(instance_object_track)

    # 3. accumulate background points in the global system
    background_track['accu_global'] = accumulate_background_point(background_track, scene_info)

    return background_track, instance_track

def transform_points_lidar2global(back_points, lidar2global):
    back_points_homo = padding_column(back_points).T  # (4, n)
    points_in_global = (lidar2global @ back_points_homo).T[:, :3]
    return points_in_global
