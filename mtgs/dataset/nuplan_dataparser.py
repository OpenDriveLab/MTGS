#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Tuple, Type, List, Union
from jaxtyping import Float

import numpy as np
import torch
from torch import Tensor
from pyquaternion import Quaternion

from nerfstudio.cameras.cameras import CameraType
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.rich_utils import CONSOLE

from nuplan_scripts.utils.video_scene_dict_tools import VideoScene
from nuplan_scripts.utils.config import FrmaeCentralConfig, RoadBlockConfig, load_config
from mtgs.utils.camera_utils import matrix_from_translation_and_quaternion, \
                                           inverse_matrix_from_translation_and_quaternion, calculate_camera_velocity_in_world
from mtgs.utils import chamfer_distance


@dataclass
class NuPlanDataparserOutputs(DataparserOutputs):
    lidar_filenames: Optional[List[Path]] = None
    depth_filenames: Optional[List[Path]] = None
    semantic_mask_filenames: Optional[List[Path]] = None
    panoptic_mask_filenames: Optional[List[Path]] = None
    ego_mask_filenames: Optional[List[Path]] = None
    undistort_params: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    camera_linear_velocities: Optional[List[np.ndarray]] = None
    camera_angular_velocities: Optional[List[np.ndarray]] = None
    lidar_to_cams: Optional[Float[Tensor, "num_cameras 4 4"]] = None
    v_adjust_factors: Optional[List[float]] = None
    travel_ids: Optional[List[int]] = None
    frame_tokens: Optional[List[str]] = None
    cam_tokens: Optional[List[str]] = None
    frame_ids: Optional[List[int]] = None
    scene_scale_factor: float = 1.0

@dataclass
class NuplanDataParserConfig(DataParserConfig):

    _target: Type = field(default_factory=lambda: NuplanDataParser)
    """target class to instantiate"""

    road_block_config: Path = Path()
    """ Relative path to the road block config file. """
    block_size: tuple = (-1.2, -1.2, -0.2, 1.2, 1.2, 0.4)

    train_scene_travels: tuple[int, ...] = ()
    eval_scene_travels: tuple[int, ...] = ()
    manual_split: bool = False
    """If True, the train and eval frames are interleaved one by one."""

    cameras: Tuple[Literal['CAM_F0', 'CAM_L0', 'CAM_R0', 'CAM_L1', 'CAM_R1', 'CAM_L2', 'CAM_R2', 'CAM_B0'], ...] = \
        ('CAM_F0', 'CAM_L0', 'CAM_L1', 'CAM_L2', 'CAM_R0', 'CAM_R1', 'CAM_R2', 'CAM_B0')
    """Which cameras to use."""

    use_colmap_pose: bool = False
    """If True, use the colmap pose for the images."""
    use_original_pose: bool = False
    """If True, use the original pose for the images. Not available when use_colmap_pose is True."""
    use_colmap_intrinsics: bool = True
    """If True, use the colmap intrinsics for the images."""

    load_3D_points: bool = True
    scale_factor: float = 1.

    only_moving: bool = False
    """If True, the parked car will not be loaded."""

    load_cam_optim_from: Optional[Path] = None
    """load camera optimizer results from given model path. Used in interpolate rendering."""
    cam_optim_key: Optional[str] = '_model.camera_optimizer.pose_adjustment'

    undistort_images: Literal["optimal", "keep_focal_length", False] = "optimal"
    """Whether to undistort the images. If False, the images are not undistorted."""

    eval_2hz: bool = False
    """If True, evaluate the model at 2Hz."""
    eval_openscene: bool = False
    """If True, evaluate the model on OpenScene tokens."""
    openscene_data_root: Optional[Path] = None

    use_exposure_alignment: bool = True
    """If True, use exposure alignment for the images."""

@dataclass
class NuplanDataParser(DataParser):

    config: NuplanDataParserConfig
    includes_time: bool = True

    def _generate_dataparser_outputs(self, split="train"):

        road_block_config: Union[FrmaeCentralConfig, RoadBlockConfig] = load_config(Path(self.config.road_block_config).as_posix())
        video_scene = VideoScene(road_block_config)
        self.video_scene = video_scene
        video_scene_dict = video_scene.load_pickle(video_scene.pickle_path, verbose=False)
        cameras = self.config.cameras

        # get image filenames and camera data
        image_filenames = []
        intrinsics = []
        undistort_params = []
        poses = []
        lidar_paths = []
        lidar2cams = []
        v_adjust_factors = []
        travel_ids = []
        depth_image_paths = []
        camera_timestamps = []
        camera_linear_velocities = []
        camera_angular_velocities = []
        frame_tokens = []
        cam_tokens = []
        frame_indices = []
        frame_timestamps = []
        semantic_mask_paths = []
        panoptic_mask_paths = []
        ego_mask_paths = []
        multi_travel_frame_timestamps = {}

        metadata = {}
        train_travel_ids = self.config.train_scene_travels
        if self.config.train_scene_travels is None or len(self.config.train_scene_travels) == 0:
            train_travel_ids = set([int(video_token.split("-")[-1]) for video_token in video_scene_dict])
        metadata['train_travel_ids'] = [idx['idx'] if isinstance(idx, dict) else idx for idx in train_travel_ids]

        eval_travel_ids = self.config.eval_scene_travels
        if self.config.eval_scene_travels is None or len(self.config.eval_scene_travels) == 0:
            eval_travel_ids = set([int(video_token.split("-")[-1]) for video_token in video_scene_dict])
        metadata['eval_travel_ids'] = [idx['idx'] if isinstance(idx, dict) else idx for idx in eval_travel_ids]

        id_to_trajectory = {}
        for video_token in video_scene_dict:
            travel_id = int(video_token.split("-")[-1])
            id_to_trajectory[travel_id] = video_scene_dict[video_token]['trajectory']

        nearest_traversal = {}
        for travel_id in metadata['eval_travel_ids']:
            if travel_id in metadata['train_travel_ids']:
                nearest_traversal[travel_id] = travel_id
            eval_trajectory = id_to_trajectory[travel_id]
            min_distance = float('inf')
            for train_travel_id in metadata['train_travel_ids']:
                train_trajectory = id_to_trajectory[train_travel_id]
                distance = chamfer_distance(eval_trajectory, train_trajectory)
                if distance < min_distance:
                    min_distance = distance
                    nearest_traversal[travel_id] = train_travel_id
        metadata['nearest_train_travel_of_eval'] = nearest_traversal

        travels = self.config.train_scene_travels if split == "train" else self.config.eval_scene_travels
        if travels is None or len(travels) == 0:
            CONSOLE.log('No specific scene travels, use all scenes')
            travels = None
        if travels is not None:
            video_scene_dict = video_scene.video_scene_dict_process(
                {'type': 'filter_by_video_idx', 'kwargs': {'video_idxs': travels}}, inline=True
            )

        for video_token in video_scene_dict:
            travel_id = int(video_token.split("-")[-1])
            frame_infos = video_scene_dict[video_token]['frame_infos']
            raw_all_timestamps = []
            for info in frame_infos:
                raw_all_timestamps.append(info['timestamp'])
                for cam_info in info['cams'].values():
                    raw_all_timestamps.append(cam_info['timestamp'])

            max_ts = max(raw_all_timestamps)
            min_ts = min(raw_all_timestamps)

            multi_travel_frame_timestamps[travel_id] = {
                'raw_timestamps': None,
                'min_ts': min_ts,
                'max_ts': max_ts
            }

        if not self.config.manual_split:
            if split == 'train':
                video_scene_dict = video_scene.video_scene_dict_process('filter_skipped_frames', inline=True)

            # filter test frames
            if split != 'train':
                if not self.config.eval_openscene:
                    video_scene_dict = video_scene.video_scene_dict_process('filter_skipped_frames', inline=True)

                for video_token in video_scene_dict:
                    frame_infos = video_scene_dict[video_token]['frame_infos']
                    if self.config.eval_2hz:
                        # keep the last frame
                        video_scene_dict[video_token]['frame_infos'] = frame_infos[:-1][::5] + frame_infos[-1:]

                    elif self.config.eval_openscene:
                        log_name = video_scene_dict[video_token]['log_name']
                        openscene_data_root = Path(self.config.openscene_data_root) / 'meta_datas' / 'trainval' / f'{log_name}.pkl'
                        if not openscene_data_root.exists():
                            openscene_data_root = Path(self.config.openscene_data_root) / 'meta_datas' / 'test' / f'{log_name}.pkl'
                        with open(openscene_data_root, 'rb') as f:
                            openscene_data_info = pickle.load(f)
                        openscene_tokens = set([info['token'] for info in openscene_data_info])
                        video_scene_dict[video_token]['frame_infos'] = [info for info in frame_infos if info['token'] in openscene_tokens]
        else:
            video_scene_dict = video_scene.video_scene_dict_process('filter_skipped_frames', inline=True)
            for video_token in video_scene_dict:
                frame_infos = video_scene_dict[video_token]['frame_infos']
                if split == "train":
                    # keep the last frame
                    video_scene_dict[video_token]['frame_infos'] = frame_infos[:-1][::2] + frame_infos[-1:]
                else:
                    video_scene_dict[video_token]['frame_infos'] = frame_infos[:-1][1::2]

        for video_token in video_scene_dict:
            travel_id = int(video_token.split("-")[-1])
            frame_infos = video_scene_dict[video_token]['frame_infos']

            multi_travel_frame_timestamps[travel_id]['raw_timestamps'] = torch.from_numpy(
                np.array([info['timestamp'] for info in frame_infos], dtype=np.int64))

            not_warned = True
            for frame_idx, info in enumerate(frame_infos):
                frame_token = info['token']
                lidar2ego = info['lidar2ego']
                all_tokens = info['track_tokens']
                token2bxid = dict()
                for token_id, token in enumerate(all_tokens):
                    token2bxid[token] = token_id

                for device_idx, camera in enumerate(cameras):
                    cam_info = info['cams'][camera]
                    cam_token = os.path.basename(cam_info['data_path']).split('.')[0]

                    if (not split == 'test') and self.config.use_colmap_pose: 
                        # for train and val, use valid flag.
                        # test is for rendering, keep the camera order.
                        if not cam_info.get('valid', True):
                            continue

                    # always use colmap intrinsic if available
                    if 'colmap_param' in cam_info and self.config.use_colmap_intrinsics:
                        intrinsic = cam_info['colmap_param']['cam_intrinsic']
                        distortion = cam_info['colmap_param']['distortion']
                    else:
                        intrinsic = cam_info['cam_intrinsic']
                        distortion = cam_info['distortion']

                    if split != 'test' and (self.config.use_colmap_pose) and ('colmap_param' in cam_info):
                        # assert 'colmap_param' in cam_info
                        pose = matrix_from_translation_and_quaternion(
                            cam_info['colmap_param']['sensor2global_translation'],
                            cam_info['colmap_param']['sensor2global_rotation'],
                            opencv2nf=True
                        )
                    else:
                        if split != 'test' and not_warned and self.config.use_colmap_pose:
                            CONSOLE.log(f"[WARNING] COLMAP pose not found for key \"{video_token}\". Use original pose instead.")
                            not_warned = False

                        ego2global = info['ego2global_original'] if self.config.use_original_pose else info['ego2global']
                        cam2ego = matrix_from_translation_and_quaternion(
                            cam_info['sensor2ego_translation'],
                            cam_info['sensor2ego_rotation'],
                            opencv2nf=True
                        )
                        pose = ego2global @ cam2ego

                    image_filenames.append(Path(
                        os.path.join(
                            video_scene.raw_image_path,
                            cam_info['data_path']
                        )
                    ))

                    if self.config.undistort_images == "optimal":
                        depth_image_paths.append(
                            os.path.join(
                                video_scene.optimal_undistorted_depth_path,
                                cam_info['data_path'].replace('jpg', 'png')
                            )
                        )
                    elif self.config.undistort_images == "keep_focal_length":
                        depth_image_paths.append(
                            os.path.join(
                                video_scene.undistorted_depth_path,
                                cam_info['data_path'].replace('jpg', 'png')
                            )
                        )
                    else:
                        depth_image_paths.append(None)

                    semantic_mask_paths.append(
                        os.path.join(
                            video_scene.raw_mask_path,
                            video_scene.mask_suffix_cityscape,
                            cam_info['data_path']
                        ).replace('.jpg', '.png')
                    )
                    panoptic_mask_paths.append(
                        os.path.join(
                            video_scene.raw_mask_path,
                            video_scene.mask_suffix_cityscape_pano,
                            cam_info['data_path'].replace('jpg', 'png')
                        )
                    )
                    ego_mask_paths.append(
                        os.path.join(
                            video_scene.raw_mask_path,
                            video_scene.mask_suffix_ego,
                            f'{camera}.png'
                        )
                    )

                    intrinsics.append(intrinsic)
                    undistort_params.append((intrinsic, distortion))
                    poses.append(pose)

                    travel_ids.append(travel_id)

                    ego2cam = inverse_matrix_from_translation_and_quaternion(
                        cam_info['sensor2ego_translation'],
                        cam_info['sensor2ego_rotation'],
                    )
                    lidar2cam = ego2cam @ lidar2ego

                    cam2ego = matrix_from_translation_and_quaternion(
                        cam_info['sensor2ego_translation'],
                        cam_info['sensor2ego_rotation'],
                    )
                    cam_linear_velocity, cam_angular_velocity = calculate_camera_velocity_in_world(
                        info['can_bus'][10:13],  # linear velocity
                        info['can_bus'][13:16],  # angular velocity
                        cam2ego,
                        info['ego2global']
                    )
                    camera_linear_velocities.append(cam_linear_velocity)
                    camera_angular_velocities.append(cam_angular_velocity)

                    lidar2cams.append(lidar2cam)    # opencv camera
                    lidar_paths.append(
                        os.path.join(video_scene.raw_lidar_path, info['lidar_path'])
                    )

                    v_adjust_factors.append(
                        cam_info.get('v_adjust', 1.0)
                    )

                    frame_tokens.append(frame_token)
                    frame_indices.append(frame_idx)
                    frame_timestamps.append(info['timestamp'])
                    cam_tokens.append(cam_token)
                    camera_timestamps.append(cam_info['timestamp'])

        assert len(poses) == len(intrinsics) == len(image_filenames) == len(lidar2cams) == len(lidar_paths) == len(v_adjust_factors) == \
               len(travel_ids) == len(camera_timestamps) == len(depth_image_paths) == \
               len(panoptic_mask_paths) == len(semantic_mask_paths) == len(ego_mask_paths) == len(undistort_params), "Lengths of lists do not match."

        poses = torch.from_numpy(np.array(poses, dtype=np.float32))
        lidar2cams = torch.from_numpy(np.array(lidar2cams, dtype=np.float32))
        intrinsics = torch.from_numpy(np.array(intrinsics, dtype=np.float32))
        frame_timestamps = torch.from_numpy(np.array(frame_timestamps, dtype=np.int64))
        camera_timestamps = torch.from_numpy(np.array(camera_timestamps, dtype=np.int64))
        normalized_camera_timestamps = torch.zeros_like(camera_timestamps, dtype=torch.float32)

        for travel_id in multi_travel_frame_timestamps:
            travel_mask = torch.from_numpy(np.array(travel_ids)) == travel_id
            min_ts = multi_travel_frame_timestamps[travel_id]['min_ts']
            max_ts = multi_travel_frame_timestamps[travel_id]['max_ts']
            multi_travel_frame_timestamps[travel_id]['frame_timestamps'] = torch.clamp((multi_travel_frame_timestamps[travel_id]['raw_timestamps'] - min_ts) / (max_ts - min_ts), 0., 1.)
            normalized_camera_timestamps[travel_mask] = torch.clamp((camera_timestamps[travel_mask] - min_ts) / (max_ts - min_ts), 0., 1.)

        # in x,y,z order
        # assumes that the scene is centered at the origin
        road_block = road_block_config.road_block
        road_block_size = np.array([road_block[2] - road_block[0], road_block[3] - road_block[1]]).max()
        block_size = np.array(self.config.block_size).reshape(2, 3) * road_block_size * self.config.scale_factor

        scene_box = SceneBox(
            aabb=torch.tensor(
                block_size, dtype=torch.float32
            )
        )
        poses[:, :3, 3] *= self.config.scale_factor

        if self.config.load_cam_optim_from is not None:
            model = torch.load(self.config.load_cam_optim_from, map_location='cpu')
            pose_adj = model['pipeline'][self.config.cam_optim_key]
            if pose_adj.shape[0] != poses.shape[0]:
                CONSOLE.log(f"[WARNING] pose_adj shape {pose_adj.shape[0]} does not match poses shape {poses.shape[0]}")
            else:
                from nerfstudio.cameras.lie_groups import exp_map_SO3xR3
                pose_adj = exp_map_SO3xR3(pose_adj) # n 3 4
                pose_adj = torch.cat([
                    pose_adj, 
                    pose_adj.new_tensor([0, 0, 0, 1])[None, None].repeat(pose_adj.shape[0], 1, 1)
                    ], dim=1)
                poses = torch.bmm(poses, pose_adj)

        cameras = Cameras(
            fx=intrinsics[:, 0, 0].detach().clone(),
            fy=intrinsics[:, 1, 1].detach().clone(),
            cx=intrinsics[:, 0, 2].detach().clone(),
            cy=intrinsics[:, 1, 2].detach().clone(),
            height=1080,
            width=1920,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
            times=normalized_camera_timestamps
        )

        # Load 3D points
        if split == 'train':
            if self.config.load_3D_points:
                metadata.update(self._load_3D_points(torch.eye(4), self.config.scale_factor))
            metadata['instances_info'] = self._generate_instance_infos(video_scene)
            metadata['frame_token2frame_idx'] = {frame_token: frame_idx for frame_idx, frame_token in zip(frame_indices, frame_tokens)}
            metadata['cam_token2cam_idx'] = {cam_token: cam_idx for cam_idx, cam_token in enumerate(cam_tokens)}
            metadata['multi_travel_frame_timestamps'] = multi_travel_frame_timestamps

        if not self.config.undistort_images:
            depth_image_paths = None

        dataparser_outputs = NuPlanDataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            metadata=metadata,
            lidar_to_cams=lidar2cams,
            lidar_filenames=lidar_paths,
            depth_filenames=depth_image_paths,
            semantic_mask_filenames=semantic_mask_paths,
            panoptic_mask_filenames=panoptic_mask_paths,
            ego_mask_filenames=ego_mask_paths,
            undistort_params=undistort_params,
            camera_linear_velocities=camera_linear_velocities,
            camera_angular_velocities=camera_angular_velocities,
            v_adjust_factors=v_adjust_factors if self.config.use_exposure_alignment else None,
            travel_ids=travel_ids,
            frame_ids=frame_indices,
            frame_tokens=frame_tokens,
            cam_tokens=cam_tokens,
            scene_scale_factor=self.config.scale_factor
        )
        return dataparser_outputs

    def _load_3D_points(self, transform_matrix: torch.Tensor, scale_factor: float):
        lidar_pcd_file_paths = []
        sfm_pcd_file_paths = []
        for video_token in self.video_scene.video_scene_dict:
            lidar_pcd_file_paths.append(
                Path(self.video_scene.rgb_point_cloud_path) / f"{video_token}.pcd"
            )
            sfm_pcd_file_paths.append(
                Path(self.video_scene.sfm_point_cloud_path) / f"{video_token}.pcd"
            )

        import open3d as o3d  # Importing open3d is slow, so we only do it if we need it.
        pcd = o3d.geometry.PointCloud()
        for path in lidar_pcd_file_paths:
            if os.path.exists(path):
                pcd += o3d.io.read_point_cloud(str(path))
            else:
                CONSOLE.print(f"[WARNING] no LiDAR stacked pcd found in {path.name}")

        pcd = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.5)[0]
        pcd = pcd.voxel_down_sample(voxel_size=0.15)

        for path in sfm_pcd_file_paths:
            if os.path.exists(path):
                pcd += o3d.io.read_point_cloud(str(path))
            else:
                CONSOLE.print(f"[WARNING] no SFM pcd found in {path.name}")

        points3D = torch.from_numpy(np.asarray(pcd.points, dtype=np.float32))
        points3D = points3D @ transform_matrix[:3, :3].T + transform_matrix[:3, 3]
        points3D *= scale_factor
        points3D_rgb = torch.from_numpy((np.asarray(pcd.colors) * 255).astype(np.uint8))
        if points3D.shape[0] == 0:
            points3D = torch.randn((200, 3))
            points3D_rgb = torch.zeros((200, 3))

        out = {
            "points3D_xyz": points3D,
            "points3D_rgb": points3D_rgb,
        }
        return out

    def _generate_instance_infos(self, video_scene: VideoScene):
        video_scene_dict = video_scene.video_scene_dict
        import open3d as o3d
        instances_info = {}
        
        for video_token in video_scene_dict:
            frame_infos = video_scene_dict[video_token]['frame_infos']
            travel_id = int(video_token.split("-")[-1])
            num_frames_cur_travel = len(frame_infos)

            for frame_idx, frame_info in enumerate(frame_infos):
                e2g_trans = frame_info['ego2global_translation']
                e2g_r_mat = Quaternion(frame_info['ego2global_rotation']).rotation_matrix
                e2g_yaw = Quaternion(frame_info['ego2global_rotation']).yaw_pitch_roll[0]

                gt_boxes = frame_info['gt_boxes']
                gt_names = frame_info['gt_names']
                track_tokens = frame_info['track_tokens']
                for box, name, track_token in zip(gt_boxes, gt_names, track_tokens):
                    if track_token not in instances_info:
                        pcd_path = os.path.join(video_scene.instance_point_cloud_path, video_token, f"{track_token}.pcd")
                        if os.path.exists(pcd_path):
                            pcd = o3d.io.read_point_cloud(pcd_path)
                            points3D = torch.from_numpy(np.asarray(pcd.points, dtype=np.float32))
                            points3D_rgb = torch.from_numpy((np.asarray(pcd.colors) * 255).astype(np.uint8))
                        else:
                            CONSOLE.print("[WARNING] no pcd found for {} {}".format(video_token, track_token))
                            points3D = torch.zeros((0, 3))
                            points3D_rgb = torch.zeros((0, 3))

                        instances_info[track_token] = {
                            "class_name": name,
                            "token": track_token,
                            "pts": points3D,
                            "colors": points3D_rgb,
                            "quats": [],
                            "trans": [],
                            "size": box[3:6],      # l w h
                            "in_frame_indices": [],
                            "travel_id": travel_id,
                            "num_frames_cur_travel": num_frames_cur_travel,
                        }

                    yaw = box[6] + e2g_yaw
                    quat = Quaternion(axis=[0, 0, 1], angle=yaw).q
                    trans = box[:3] @ e2g_r_mat.T + e2g_trans
                    # in global coordinates
                    instances_info[track_token]["quats"].append(quat)
                    instances_info[track_token]["trans"].append(trans)
                    instances_info[track_token]["in_frame_indices"].append(frame_idx)

        new_instances_info = {}
        for k, v in instances_info.items():
            if v['pts'].shape[0] < 100:
                continue
            static_vehicle_flag = v['class_name'] == 'vehicle' and np.linalg.norm(v['trans'][-1] - v['trans'][0]) < 3.0

            num_frames_cur_travel = v['num_frames_cur_travel']
            frame_mask = torch.zeros(num_frames_cur_travel, dtype=torch.bool)
            frame_mask[v['in_frame_indices']] = True
            v['in_frame_mask'] = frame_mask
            quats = torch.zeros((num_frames_cur_travel, 4))
            trans = torch.zeros((num_frames_cur_travel, 3))
            quats[frame_mask] = torch.tensor(np.array(v['quats'], dtype=np.float32))
            trans[frame_mask] = torch.tensor(np.array(v['trans'], dtype=np.float32))
            v['quats'] = quats
            v['trans'] = trans
            v['is_static'] = static_vehicle_flag

            if static_vehicle_flag:
                if self.config.only_moving:
                    # TODO: add static vehicle point cloud to background
                    continue
            new_instances_info[k] = v

        return new_instances_info
