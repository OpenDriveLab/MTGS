#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import os
import argparse
from tqdm import tqdm

import numpy as np
from pyquaternion import Quaternion
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator
from accelerate.utils import tqdm as accelerate_tqdm

from nuplan_scripts.utils.config import load_config, RoadBlockConfig
from nuplan_scripts.utils.video_scene_dict_tools import VideoScene
from nuplan_scripts.utils.camera_utils import undistort_image_with_cam_info
from nuplan_scripts.utils.nuplan_utils_custom import load_lidar
from unidepth.models import UniDepthV2


model_path = 'lpiccinelli/unidepth-v2-vitl14'

class MetricDepthError:

    class PseudoDataset(Dataset):
        H, W = 1080, 1920

        def __init__(
                self,
                video_scene: VideoScene,
                data_infos,
                undistort_mode,
                max_depth=80,
                min_depth=0.1):
            self.video_scene = video_scene
            self.data_infos = data_infos
            self.undistort_mode = undistort_mode
            self.max_depth = max_depth
            self.min_depth = min_depth

        def __len__(self):
            return len(self.data_infos)

        def __getitem__(self, idx):
            info = self.data_infos[idx]
            ref_errors = {}
            lidar_pts = load_lidar(
                os.path.join(video_scene.raw_lidar_path, info['lidar_path'])
            )
            lidar_pts_xyz1 = np.concatenate([lidar_pts, np.ones((lidar_pts.shape[0], 1))], axis=1)
            lidar2ego = info['lidar2ego']

            for cam_name, cam_info in info['cams'].items():

                ego2cam = np.eye(4)
                ego2cam[:3, :3] = Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix.T
                ego2cam[:3, 3] = -ego2cam[:3, :3] @ cam_info['sensor2ego_translation']
                lidar2cam = ego2cam @ lidar2ego

                if 'colmap_param' in cam_info:
                    intrinsic = cam_info['colmap_param']['cam_intrinsic']
                    distortion = cam_info['colmap_param']['distortion']
                    intrinsic = intrinsic.copy()
                    intrinsic[0, 2] = intrinsic[0, 2] - 0.5
                    intrinsic[1, 2] = intrinsic[1, 2] - 0.5
                else:
                    intrinsic = cam_info['cam_intrinsic']
                    distortion = cam_info['distortion']

                if self.undistort_mode == 'optimal':
                    new_intrinsic, roi = cv2.getOptimalNewCameraMatrix(
                        intrinsic, distortion, (self.W, self.H), 1
                    )
                else:
                    new_intrinsic = intrinsic
                    roi = (0, 0, self.W, self.H)

                # load depth
                if self.undistort_mode == 'optimal':
                    depth_path = os.path.join(
                        self.video_scene.optimal_undistorted_depth_path,
                        cam_info['data_path'].replace('.jpg', '.png'))
                else:
                    depth_path = os.path.join(
                        self.video_scene.undistorted_depth_path,
                        cam_info['data_path'].replace('.jpg', '.png'))

                depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
                depth = depth[:,:,0] + (depth[:,:,1] * 256)
                depth = depth * 0.01

                # project lidar points to image
                point_cam = (lidar_pts_xyz1 @ lidar2cam.T)[:, :3]
                uvz = point_cam[point_cam[:, 2] > 0]
                uvz = uvz @ new_intrinsic.T
                uvz[:, :2] /= uvz[:, 2:]
                uvz = uvz[(uvz[:, 0] >= roi[0]) & (uvz[:, 0] < roi[2]) &
                          (uvz[:, 1] >= roi[1]) & (uvz[:, 1] < roi[3]) &
                          (uvz[:, 2] > self.min_depth) & (uvz[:, 2] < self.max_depth)]
                # filter out out of range depth
                uvz = uvz[(depth[uvz[:, 1].astype(int), uvz[:, 0].astype(int)] > self.min_depth) & 
                          (depth[uvz[:, 1].astype(int), uvz[:, 0].astype(int)] < self.max_depth)]
                uv = uvz[:, :2]
                uv = uv.astype(int)

                pts_depth = np.zeros((self.H, self.W, 1), dtype=np.float32)
                pts_depth[uv[:, 1], uv[:, 0], 0] = uvz[:, 2]

                depth_pts = depth[uv[:, 1], uv[:, 0]]
                valid_mask = depth_pts != 0

                # calculate error
                ref_error = np.abs(depth_pts - uvz[:, 2])[valid_mask].mean()
                ref_errors[cam_name] = ref_error

            return ref_errors

    @staticmethod
    def run(
        video_scene: VideoScene,
        data_infos,
        undistort_mode,
        max_range,
        num_workers):

        dataset = MetricDepthError.PseudoDataset(
            video_scene, 
            data_infos,
            undistort_mode,
            max_depth=max_range)
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: x, pin_memory=False, drop_last=False)

        ref_errors = {}
        for batch in tqdm(dataloader, desc='processing frames', ncols=120):
            ref_error = batch[0]
            for cam_name, error in ref_error.items():
                if cam_name not in ref_errors:
                    ref_errors[cam_name] = []
                ref_errors[cam_name].append(error)

        return ref_errors


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--undistort_mode', type=str, default='optimal', choices=['optimal', 'keep_focal_length'])
    parser.add_argument('--max_range', type=int, default=80)
    parser.add_argument('--report_errors', action='store_true')
    args = parser.parse_args()

    config: RoadBlockConfig = load_config(args.config)

    video_scene = VideoScene(config)
    video_scene_dict = video_scene.load_pickle(video_scene.pickle_path)

    total_frame_infos = []
    total_cams = []
    for video_token in video_scene_dict:
        frame_infos = video_scene_dict[video_token]['frame_infos']
        frame_infos = [info for info in frame_infos if not info.get('skipped', False)]
        total_frame_infos.extend(frame_infos)
        for info in frame_infos:
            total_cams.extend(list(info['cams'].values()))

    model = UniDepthV2.from_pretrained(model_path)

    distributed_state = Accelerator()
    distributed_state.prepare_model(model, evaluation_mode=True)
    model.eval()

    pbar = accelerate_tqdm(total=len(total_cams), ncols=120, desc="Generating dense depths")
    with distributed_state.split_between_processes(total_cams) as partial_frames:
        for cam_info in partial_frames:
            image_path = os.path.join(
                video_scene.raw_image_path,
                cam_info['data_path']
            )
            image = Image.open(image_path)
            rgb = np.array(image)
            raw_height, raw_width = rgb.shape[:2]

            if args.undistort_mode == 'optimal':

                rgb, intrinsic, roi = undistort_image_with_cam_info(
                    rgb, cam_info, 
                    return_mask=False, interpolation='cubic', mode='optimal'
                )
                x, y, w, h = roi
                rgb = rgb[y:y+h, x:x+w]
                intrinsic[0, 2] = intrinsic[0, 2] - x
                intrinsic[1, 2] = intrinsic[1, 2] - y

                inputs = torch.from_numpy(rgb).permute(2, 0, 1)
                inputs.to(distributed_state.device)
                intrinsic = torch.from_numpy(intrinsic).to(device=distributed_state.device, dtype=torch.float32)

                output_path = os.path.join(
                    video_scene.optimal_undistorted_depth_path,
                    cam_info['data_path'].replace('.jpg', '.png')
                )

            elif args.undistort_mode == 'keep_focal_length':
                rgb = undistort_image_with_cam_info(
                    rgb, cam_info, 
                    return_mask=False, interpolation='cubic', mode='keep_focal_length'
                )
                intrinsic = cam_info['cam_intrinsic']
                x, y, w, h = 0, 0, raw_width, raw_height

                inputs = torch.from_numpy(rgb).permute(2, 0, 1)
                inputs.to(distributed_state.device)
                intrinsic = torch.from_numpy(intrinsic).to(device=distributed_state.device, dtype=torch.float32)

                output_path = os.path.join(
                    video_scene.undistorted_depth_path,
                    cam_info['data_path'].replace('.jpg', '.png')
                )

            with torch.no_grad():
                outputs = model.infer(inputs, intrinsic)

            depth = outputs['depth'].squeeze()
            depth = depth.cpu().numpy()
            # filter out out of range depth
            depth[depth > args.max_range] = 0
            depth[depth < 0.1] = 0

            depth_image = np.zeros((raw_height, raw_width, 3), dtype=np.uint8)
            depth_image[y:y+h, x:x+w, 0] = ((depth * 100) % 256).astype(np.uint8)
            depth_image[y:y+h, x:x+w, 1] = ((depth * 100) // 256).astype(np.uint8)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, depth_image)

            pbar.update(distributed_state.num_processes)

    distributed_state.wait_for_everyone()
    pbar.close()

    if not distributed_state.is_main_process:
        print(f'exiting subprocess {distributed_state.process_index}')
        exit()

    if not args.report_errors:
        exit()

    # Clear memory
    import gc
    # del image_processor, model, distributed_state
    del model, distributed_state
    gc.collect()
    torch.cuda.empty_cache()

    ref_errors = MetricDepthError.run(
        video_scene,
        total_frame_infos,
        args.undistort_mode,
        args.max_range,
        num_workers=args.num_workers)

    from prettytable import PrettyTable
    table = PrettyTable()
    table.field_names = ["Camera", "Mean Error"]
    for key, value in ref_errors.items():
        table.add_row([key, f"{np.mean(value):.4f}"])

    print(table)
