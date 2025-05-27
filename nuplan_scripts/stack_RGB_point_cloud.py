#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import os
import argparse
import copy

from tqdm import tqdm

import cv2
import numpy as np
from pyquaternion import Quaternion

from torch.utils.data import Dataset, DataLoader

from nuplan_scripts.utils.config import load_config, RoadBlockConfig
from nuplan_scripts.utils.video_scene_dict_tools import VideoScene
from nuplan_scripts.utils.nuplan_utils_custom import load_lidar, get_rgb_point_cloud, get_semantic_point_cloud, adjust_brightness_single_frame, adjust_brightness
from nuplan_scripts.utils.stack_point_cloud_utils import extract_frame_background_instance_lidar, accumulate_background_box_point
from nuplan_scripts.utils.camera_utils import undistort_image_with_cam_info
from nuplan_scripts.utils.constants import NUPLAN_SENSOR_ROOT

class StackRGBPointCloud:

    class PseudoDataset(Dataset):

        def __init__(self, data_info, video_scene: VideoScene):
            self.data_info = data_info
            self.video_scene = video_scene

        def __getitem__(self, idx):
            if self.data_info[idx].get('skipped', False) == 'low_velocity':
                return self.data_info[idx]

            info = copy.deepcopy(self.data_info[idx])

            undistorted_images = []
            undistorted_sem_masks = []
            lidar2imgs = []
            lidar2ego = info['lidar2ego']
            for cam in info['cams']:
                cam_info = info['cams'][cam]
                if 'colmap_param' in cam_info:
                    intrinsic = cam_info['colmap_param']['cam_intrinsic']
                else:
                    intrinsic = cam_info['cam_intrinsic']

                cam_path = os.path.join(NUPLAN_SENSOR_ROOT, cam_info['data_path'])
                assert os.path.exists(cam_path), f'{cam_path} does not exist.'
                image = cv2.imread(cam_path)
                image = undistort_image_with_cam_info(image, cam_info, interpolation='linear', mode='keep_focal_length')
                undistorted_images.append(image)

                sem_mask_path = os.path.join(
                    self.video_scene.raw_mask_path,
                    self.video_scene.mask_suffix_cityscape,
                    cam_info['data_path']
                ).replace('.jpg', '.png')

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

            undistorted_images = np.array(undistorted_images)  # (n_cam, H, W, 3)
            undistorted_sem_masks = np.array(undistorted_sem_masks)  # (n_cam, H, W, 1)
            lidar2imgs = np.array(lidar2imgs)  # (n_cam, 4, 4)

            # use combiled lidar_point_cloud to adjust brightness
            lidar_points = load_lidar(
                os.path.join(NUPLAN_SENSOR_ROOT, info['lidar_path']), remove_close=False, only_top=False)
            adjust_brightness_single_frame(info, lidar2imgs, undistorted_images, lidar_points)
            for idx, cam_info in enumerate(info['cams'].values()):
                adjust_factor = cam_info['v_adjust']
                undistorted_images[idx] = adjust_brightness(undistorted_images[idx], adjust_factor)

            top_lidar_points = load_lidar(
                os.path.join(NUPLAN_SENSOR_ROOT, info['lidar_path']), remove_close=False, only_top=True)
            info['back_instance_info'] = extract_frame_background_instance_lidar(info, l2g=False, points=top_lidar_points)

            # for background points
            bg_points = info['back_instance_info']['background_points']

            # aggregate all points to speed up the process
            # set background mask to 0, and instance mask to 1, 2, 3, ...
            all_points = [bg_points]
            all_masks = [np.zeros_like(bg_points[:, 0])]
            token2idx = {'background': 0}
            for idx, (instance_token, instance_object) in enumerate(info['back_instance_info']['instance'].items()):
                token2idx[instance_token] = idx + 1
                all_points.append(instance_object.points_ego)
                all_masks.append(np.ones_like(instance_object.points_ego[:, 0]) * (idx + 1))
            all_points = np.concatenate(all_points, axis=0)
            all_masks = np.concatenate(all_masks, axis=0)
            all_RGBs, fov_mask = get_rgb_point_cloud(all_points, lidar2imgs, undistorted_images)
            all_sem_labels, fov_mask_sem = get_semantic_point_cloud(all_points, lidar2imgs, undistorted_sem_masks)

            points_mask = (all_masks == 0) & fov_mask & fov_mask_sem
            bg_points = all_points[points_mask]
            bg_rgb = all_RGBs[points_mask]
            bg_sem_labels = all_sem_labels[points_mask]
            info['back_instance_info']['background_points'] = bg_points
            info['back_instance_info']['background_RGBs'] = bg_rgb
            info['back_instance_info']['background_sem_labels'] = bg_sem_labels

            # for instance points
            for instance_token, instance_object in info['back_instance_info']['instance'].items():
                points_mask = (all_masks == token2idx[instance_token]) & fov_mask & fov_mask_sem
                instance_points = instance_object.points[fov_mask[all_masks == token2idx[instance_token]]]
                instance_rgb = all_RGBs[points_mask]
                instance_sem_labels = all_sem_labels[points_mask]
                instance_object.points = instance_points
                instance_object.rgbs = instance_rgb
                instance_object.sem_labels = instance_sem_labels

            return info

        def __len__(self):
            return len(self.data_info)
    
    @staticmethod
    def run(video_scene: VideoScene, video_scene_dict, num_workers):

        for video_token in tqdm(video_scene_dict, desc='processing videos', ncols=120):
            frame_infos = video_scene_dict[video_token]['frame_infos']
            dataset = StackRGBPointCloud.PseudoDataset(
                frame_infos, video_scene)
            dataloader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=lambda x: x, pin_memory=False, drop_last=False)

            new_frame_infos = []
            for batch in tqdm(dataloader, desc='processing frames', ncols=120):
                info = batch[0]
                new_frame_infos.append(info)
            video_scene_dict[video_token]['frame_infos'] = new_frame_infos

            filtered_frame_infos = [info for info in new_frame_infos if 'back_instance_info' in info]
            background_track, instance_track = accumulate_background_box_point(filtered_frame_infos, l2g=True)
            stacked_points, stacked_rgb, stacked_sem_labels = background_track['accu_global']
            # filter out points in sky, person, rider, car, truck, bus, motorcycle, bicycle
            mask = stacked_sem_labels < 10
            stacked_sem_labels = stacked_sem_labels[mask]
            stacked_points = stacked_points[mask]
            stacked_rgb = stacked_rgb[mask]

            import open3d as o3d
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(stacked_points))
            pcd.colors = o3d.utility.Vector3dVector(stacked_rgb)
            filename = f'{video_scene.rgb_point_cloud_path}/{video_token}.pcd'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            o3d.io.write_point_cloud(filename, pcd)

            stacked_sem_labels = stacked_sem_labels.astype(np.uint8)
            np.save(f'{video_scene.rgb_point_cloud_path}/{video_token}_sem_labels.npy', stacked_sem_labels)

            for track_token in instance_track:
                instance_object_track = instance_track[track_token]
                stacked_points = instance_object_track.accu_points
                stacked_rgb = instance_object_track.accu_rgbs
                if stacked_points.shape[0] == 0:
                    continue

                pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(stacked_points))
                pcd.colors = o3d.utility.Vector3dVector(stacked_rgb)
                filename = f'{video_scene.instance_point_cloud_path}/{video_token}/{track_token}.pcd'
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                o3d.io.write_point_cloud(filename, pcd)

        return video_scene_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    config: RoadBlockConfig = load_config(args.config)
    video_scene = VideoScene(config)
    video_scene_dict = video_scene.load_pickle(video_scene.pickle_path)
    
    video_scene_dict = StackRGBPointCloud.run(video_scene, video_scene_dict, args.num_workers)
    video_scene.video_scene_dict = video_scene_dict
    video_scene.dump_pickle(video_scene.pickle_path_final)
    video_scene.update_pickle_link(video_scene.pickle_path_final)
