#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import os
import copy
import shutil
import argparse

import multiprocessing
import subprocess
from functools import partial

from tqdm import tqdm

import numpy as np
import cv2
from pyquaternion import Quaternion

from nuplan_scripts.utils.constants import CONSOLE
from nuplan_scripts.utils.config import load_config, RoadBlockConfig
from nuplan_scripts.utils.video_scene_dict_tools import VideoScene
from nuplan_scripts.utils.camera_utils import field_of_view_intrinsic
from nuplan_scripts.utils.colmap_utils.gen_colmap_db import create_colmap_database
from nuplan_scripts.utils.colmap_utils.read_write_model import read_model
from nuplan_scripts.utils.colmap_utils.align_model import compute_transformation_matrix_with_scaling
from nuplan_scripts.utils.constants import NUPLAN_SENSOR_ROOT


def copy_ego_masks(video_scene: VideoScene):
    data_root = video_scene.data_root
    raw_mask_path = os.path.join(data_root, 'ego_masks/raw')
    assert os.path.exists(raw_mask_path), f'Ego masks not found at {raw_mask_path}'
    output_raw_mask_path = os.path.join(video_scene.raw_mask_path, video_scene.mask_suffix_ego)
    if not os.path.exists(output_raw_mask_path):
        os.makedirs(os.path.dirname(output_raw_mask_path), exist_ok=True)
        shutil.copytree(raw_mask_path, output_raw_mask_path)

def create_colmap_folder(colmap_path):

    if os.path.exists(colmap_path):
        shutil.rmtree(colmap_path)

    os.makedirs(f"{colmap_path}/images", exist_ok=True)
    os.makedirs(f"{colmap_path}/masks", exist_ok=True)
    os.makedirs(f"{colmap_path}/sfm_model", exist_ok=True)
    os.makedirs(f"{colmap_path}/sparse_model", exist_ok=True)
    open(
        f"{colmap_path}/sparse_model/points3D.txt", "w"
    ).close()  # create an empty points3D.txt

def clean_colmap_folder(colmap_path):
    if os.path.exists(colmap_path):
        shutil.rmtree(colmap_path)

def create_cameras_per_vehicle(video_scene_dict, colmap_path):
    exsiting_veh_name = {}
    f_cameras = open(f"{colmap_path}/sparse_model/cameras.txt", "w")
    global_cam_id = 1
    for video_token in video_scene_dict:
        frame_infos = video_scene_dict[video_token]["frame_infos"]

        # same vehicle has same camera calibration
        veh_name = video_scene_dict[video_token]['vehicle_name']
        if veh_name not in exsiting_veh_name:
            exsiting_veh_name[veh_name] = {}
            cam_infos = frame_infos[0]["cams"]
            for cam in cam_infos:
                cam_info = cam_infos[cam]
                intrinsic = cam_info['cam_intrinsic']
                distortion = cam_info['distortion']
                line = (
                    f"{global_cam_id} OPENCV 1920 1080 "
                    + f"{intrinsic[0, 0]} {intrinsic[1, 1]} {intrinsic[0, 2]} {intrinsic[1, 2]} "
                    + f"{distortion[0]} {distortion[1]} {distortion[2]} {distortion[3]}\n"
                )
                f_cameras.write(line)
                exsiting_veh_name[veh_name][cam] = global_cam_id
                global_cam_id += 1
    f_cameras.close()

    cam_name_mapper = {}
    for veh_name in exsiting_veh_name:
        for cam in exsiting_veh_name[veh_name]:
            cam_name_colmap = f'{veh_name}_{cam}'
            cam_name_mapper[cam_name_colmap] = exsiting_veh_name[veh_name][cam]
            os.makedirs(f'{colmap_path}/images/{cam_name_colmap}', exist_ok=True)
            os.makedirs(f'{colmap_path}/masks/{cam_name_colmap}', exist_ok=True)

    return cam_name_mapper


def create_colmap_model(video_scene: VideoScene, video_scene_dict, colmap_path):

    create_colmap_folder(colmap_path)
    cam_name_mapper = create_cameras_per_vehicle(video_scene_dict, colmap_path)

    progress_bar = tqdm(
        total=VideoScene.count_total_frames(video_scene_dict),
        desc='Loading image infos', ncols=120)

    scale_factor = 10 / max(video_scene.block_size)

    f_images = open(f"{colmap_path}/sparse_model/images.txt", "w")
    global_image_id = 1
    invalid_images = 0
    cam_views = []
    image_names = set()
    for video_token in video_scene_dict:
        frame_infos = video_scene_dict[video_token]["frame_infos"]
        frame_infos = [info for info in frame_infos if not info.get('skipped', False)]

        # same vehicle has same camera calibration
        veh_name = video_scene_dict[video_token]['vehicle_name']

        for idx, info in enumerate(frame_infos):

            ego2global = info['ego2global']

            for cam_name in info['cams']:
                cam_info = info['cams'][cam_name]

                data_path = os.path.join(NUPLAN_SENSOR_ROOT, cam_info['data_path'])

                sem_mask_path = os.path.join(
                    video_scene.raw_mask_path,
                    video_scene.mask_suffix_cityscape,
                    cam_info['data_path'].replace('.jpg', '.png')
                )
                ego_mask_path = os.path.join(
                    video_scene.raw_mask_path,
                    video_scene.mask_suffix_ego,
                    f'{cam_name}.png'
                )
                sem_mask = cv2.imread(sem_mask_path, cv2.IMREAD_GRAYSCALE)
                ego_mask = cv2.imread(ego_mask_path, cv2.IMREAD_GRAYSCALE) != 0
                foreground_mask_sem = (sem_mask >= 10)  # sky, person, ...
                invalid_mask = ego_mask | foreground_mask_sem
                valid_mask = np.logical_not(invalid_mask)

                # TODO: this threshold should be further tuned.
                if invalid_mask.sum() > 0.8 * invalid_mask.shape[0] * invalid_mask.shape[1]:
                    invalid_images += 1
                    cam_info['valid'] = False
                    continue
                cam_info['valid'] = True

                intrinsic = cam_info['cam_intrinsic']
                cam2ego = Quaternion(cam_info['sensor2ego_rotation']).transformation_matrix
                cam2ego[:3, 3] = cam_info['sensor2ego_translation']
                cam2global = np.dot(ego2global, cam2ego)
                cam2global[:3, 3] *= scale_factor  # scale to about 10 meters for colmap

                # global2cam
                Q = Quaternion(matrix=cam2global[:3, :3].T).q
                T = -cam2global[:3, :3].T @ cam2global[:3, 3]

                cam_name_colmap = f'{veh_name}_{cam_name}'
                cam_id = cam_name_mapper[cam_name_colmap]
                image_path = f'{cam_name_colmap}/{os.path.basename(data_path)}'

                # deduplicate images
                if image_path in image_names:
                    invalid_images += 1
                    cam_info['valid'] = False
                    continue
                image_names.add(image_path)

                view_field = field_of_view_intrinsic(intrinsic, 30 * scale_factor, cam2global)
                cam_views.append(dict(
                    token=image_path,
                    ego_token=info['token'],
                    view_field=view_field
                ))

                # write to images.txt
                line = f'{global_image_id} {Q[0]} {Q[1]} {Q[2]} {Q[3]} {T[0]} {T[1]} {T[2]} {cam_id} {image_path}' + '\n\n'
                f_images.write(line)

                # save image and mask
                image_output_path = f'{colmap_path}/images/{image_path}'
                shutil.copy2(data_path, image_output_path)
                mask_output_path = f'{colmap_path}/masks/{image_path}.png'
                cv2.imwrite(mask_output_path, valid_mask.astype(np.uint8) * 255)

                global_image_id += 1

            progress_bar.update(1)
    progress_bar.close()
    f_images.close()

    print(f'Filter out invalid images: {invalid_images}/{len(frame_infos) * 8}')

    # compute image pairs
    pairs = set()
    for i, view1 in enumerate(tqdm(cam_views, desc='Computing camera pairs...', ncols=120)):
        for j, view2 in enumerate(cam_views):
            if j <= i:
                continue
            v1 = view1['view_field']
            v2 = view2['view_field']
            iou = v1.intersection(v2).area / v1.union(v2).area
            if iou > 0.0:
                pairs.add((view1['token'], view2['token']))

    f_pairs = open(f'{colmap_path}/image_pairs.txt', 'w')
    for pair in pairs:
        f_pairs.write(f'{pair[0]} {pair[1]}\n')
    f_pairs.close()

    # create colmap database after everything is done
    create_colmap_database(colmap_path)
    return video_scene_dict

def create_sparse_colmap_model(video_scene: VideoScene, video_scene_dict, colmap_path):

    create_colmap_folder(colmap_path)
    cam_name_mapper = create_cameras_per_vehicle(video_scene_dict, colmap_path)

    progress_bar = tqdm(
        total=VideoScene.count_total_frames(video_scene_dict),
        desc='Loading image infos', ncols=120)

    f_images = open(f"{colmap_path}/sparse_model/images.txt", "w")
    global_image_id = 1
    for video_token in video_scene_dict:
        frame_infos = video_scene_dict[video_token]['frame_infos']
        frame_infos = [info for info in frame_infos if not info.get('skipped', False)]

        # same vehicle has same camera calibration
        veh_name = video_scene_dict[video_token]['vehicle_name']

        for idx, info in enumerate(frame_infos):

            ego2global = info['ego2global']

            for cam in info['cams']:
                cam_info = info['cams'][cam]

                data_path = os.path.join(NUPLAN_SENSOR_ROOT, cam_info['data_path'])

                cam2ego = Quaternion(cam_info['sensor2ego_rotation']).transformation_matrix
                cam2ego[:3, 3] = cam_info['sensor2ego_translation']
                cam2global = np.dot(ego2global, cam2ego)

                # global2cam
                Q = Quaternion(matrix=cam2global[:3, :3].T).q
                T = -cam2global[:3, :3].T @ cam2global[:3, 3]

                cam_name_colmap = f'{veh_name}_{cam}'
                cam_id = cam_name_mapper[cam_name_colmap]
                image_path = f'{cam_name_colmap}/{os.path.basename(data_path)}'

                # write to images.txt
                line = f'{global_image_id} {Q[0]} {Q[1]} {Q[2]} {Q[3]} {T[0]} {T[1]} {T[2]} {cam_id} {image_path}' + '\n\n'
                f_images.write(line)
                global_image_id += 1

            progress_bar.update(1)
    progress_bar.close()
    f_images.close()
    # create colmap database after everything is done
    create_colmap_database(colmap_path)
    return video_scene_dict


def align_model(video_scene: VideoScene, video_scene_dict, colmap_path, thresh=1.0):
    cameras, images, points3D = read_model(f'{colmap_path}/sfm_model/0', ext='.bin')

    # read points3D
    points3D_new = {}
    for key, val in points3D.items():
        if val.image_ids.shape[0] < 3:
            continue
        points3D_new[key] = val
    print('[Read Colmap] filter {}/{} points3D, min view = {}'.format(len(points3D_new), len(points3D), 3))

    # read camera
    camera_params = {}
    for key in cameras.keys():
        # print(cameras[key])
        p = cameras[key].params
        if cameras[key].model == 'PINHOLE':
            fx, fy, cx, cy = p
            K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3)
            dist = None
        elif cameras[key].model == 'OPENCV':
            K = np.array([[p[0], 0, p[2], 0, p[1], p[3], 0, 0, 1]]).reshape(3, 3)
            dist = np.array([p[4], p[5], p[6], p[7], 0.])
        else:
            raise NotImplementedError
        camera_params[cameras[key].id] = (K, dist)

    # read image
    images_out = {}
    for img in images.values():
        intrinsic, distortion = camera_params[img.camera_id]
        R = Quaternion(img.qvec)
        t = img.tvec

        c2w = np.eye(4)
        c2w[:3, :3] = R.rotation_matrix.T
        c2w[:3, 3] = -R.rotation_matrix.T @ t

        token = img.name.split('.')[0].split('/')[1]
        cam_info = {}
        cam_info['sensor2global_rotation'] = Quaternion(matrix=c2w[:3, :3])
        cam_info['sensor2global_translation'] = c2w[:3, 3]
        cam_info['cam_intrinsic'] = intrinsic
        cam_info['distortion'] = distortion
        cam_info['data_path'] = img.name
        images_out[token] = cam_info

    raw_cam_T = {}
    for video_token in video_scene_dict:
        frame_infos = video_scene_dict[video_token]['frame_infos']
        frame_infos = [info for info in frame_infos if not info.get('skipped', False)]
        for info in frame_infos:
            ego2global = info['ego2global']

            for cam in info['cams']:
                cam_info = info['cams'][cam]

                cam2ego = np.eye(4)
                cam2ego[:3, :3] = Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix
                cam2ego[:3, 3] = cam_info['sensor2ego_translation']
                cam2global = np.dot(ego2global, cam2ego)
                token = cam_info['token']

                if token not in images_out:
                    cam_info['valid'] = False
                    continue

                raw_cam_T[token] = cam2global[:3, 3]

    colmap_cam_array = []
    raw_cam_array = []
    for cam_token in images_out.keys():
        colmap_cam_array.append(images_out[cam_token]['sensor2global_translation'])
        raw_cam_array.append(raw_cam_T[cam_token])
    colmap_cam_array = np.asarray(colmap_cam_array)
    raw_cam_array = np.asarray(raw_cam_array)

    scale, R, t = compute_transformation_matrix_with_scaling(colmap_cam_array, raw_cam_array)

    images_aligned = {}
    for token, cam_info in images_out.items():
        cam_info = copy.deepcopy(cam_info)
        sensor2global_rotation = R @ Quaternion(cam_info['sensor2global_rotation']).rotation_matrix
        cam_info['sensor2global_rotation'] = Quaternion(matrix=sensor2global_rotation)
        cam_info['sensor2global_translation'] = scale * (R @ cam_info['sensor2global_translation']).T + t
        images_aligned[token] = cam_info

    pts3D_xyz = np.array([p.xyz for p in points3D_new.values()], dtype=np.float32)
    pts3D_xyz = scale * (pts3D_xyz @ R.T) + t
    pts3D_rgb = np.array([p.rgb for p in points3D_new.values()], dtype=np.float32)
    pts3D_rgb /= 255.0

    import open3d as o3d
    o3d_points3D = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts3D_xyz))
    o3d_points3D.colors = o3d.utility.Vector3dVector(pts3D_rgb)
    os.makedirs(video_scene.sfm_point_cloud_path, exist_ok=True)
    o3d.io.write_point_cloud(f'{video_scene.sfm_point_cloud_path}/{os.path.basename(colmap_path)}.pcd', o3d_points3D)

    invalid_count = 0
    errors = []
    for video_token in video_scene_dict:
        frame_infos = video_scene_dict[video_token]['frame_infos']
        frame_infos = [info for info in frame_infos if not info.get('skipped', False)]
        for info in frame_infos:
            ego2global = info['ego2global']
            for cam in info['cams']:
                cam_info = info['cams'][cam]
                token = cam_info['token']
                if cam_info.get('valid', True):
                    colmap_cam_info = images_aligned[token]
                    colmap_cam2global_T = colmap_cam_info['sensor2global_translation']

                    cam2ego = np.eye(4)
                    cam2ego[:3, :3] = Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix
                    cam2ego[:3, 3] = cam_info['sensor2ego_translation']
                    cam2global = np.dot(ego2global, cam2ego)
                    cam2global_T = cam2global[:3, 3]

                    error = np.linalg.norm(colmap_cam2global_T - cam2global_T)
                    errors.append(error)
                    if error > thresh:
                        cam_info['valid'] = False
                        invalid_count += 1
                        continue
                    else:
                        cam_info['valid'] = True
                        cam_info['colmap_param'] = colmap_cam_info

    CONSOLE.log(f'[Align Colmap] for {os.path.basename(colmap_path)}')
    CONSOLE.log(f'[Align Colmap] invalid count = {invalid_count}, total = {len(images_out)}')
    CONSOLE.log(f'[Align Colmap] error = {np.mean(errors):.3f}')
    return video_scene_dict

def run_colmap_ba(colmap_path, num_workers=8):
    CONSOLE.log(f'Colmap BA for {os.path.basename(colmap_path)} started')
    log_file = f'{colmap_path}/colmap_ba.log'

    if os.path.exists(log_file):
        os.remove(log_file)

    command = f'python -m nuplan_scripts.utils.colmap_utils.bundle_adjustment --colmap_path {colmap_path} --num_workers {num_workers}'
    with open(log_file, 'w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, shell=True)
        process.communicate()
        if process.returncode != 0:
            CONSOLE.log(f'[ERROR] Colmap Triangulation for {os.path.basename(colmap_path)} failed')
            return False

    CONSOLE.log(f'Colmap BA for {os.path.basename(colmap_path)} done')
    return True

def run_colmap_triangulation(colmap_path, num_workers=8):
    CONSOLE.log(f'Colmap Triangulation for {os.path.basename(colmap_path)} started')
    log_file = f'{colmap_path}/colmap_triangulation.log'

    if os.path.exists(log_file):
        os.remove(log_file)

    command = f'python -m nuplan_scripts.utils.colmap_utils.point_triangulator --colmap_path {colmap_path} --num_workers {num_workers}'
    with open(log_file, 'w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, shell=True)
        process.communicate()
        if process.returncode != 0:
            CONSOLE.log(f'[ERROR] Colmap Triangulation for {os.path.basename(colmap_path)} failed')
            return False

    CONSOLE.log(f'Colmap Triangulation for {os.path.basename(colmap_path)} done')
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--skip_create_model', action='store_true')
    parser.add_argument('--skip_colmap_ba', action='store_true')
    args = parser.parse_args()

    skip_create_model = args.skip_create_model
    skip_colmap_ba = args.skip_colmap_ba

    # 1. load config and video scene dict
    config: RoadBlockConfig = load_config(args.config)
    video_scene = VideoScene(config)
    all_video_scene_dict = video_scene.load_pickle(video_scene.pickle_path)

    # copy ego mask to sub data root
    copy_ego_masks(video_scene)

    colmap_paths = []
    single_video_scene_dicts = []
    # 2. process each video scene
    for video_token in all_video_scene_dict:
        video_idx = int(video_token.split('-')[-1])
        video_scene_dict = video_scene.video_scene_dict_process([
            {'type': 'filter_by_video_idx', 'kwargs': {'video_idxs': (video_idx,)}}
        ])

        colmap_path = os.path.join(video_scene.colmap_path, video_token)
        if not skip_create_model:
            create_colmap_model(video_scene, video_scene_dict, colmap_path)
        else:
            CONSOLE.log(f'Skipping craete colmap model for {video_token}')
        colmap_paths.append(colmap_path)
        single_video_scene_dicts.append(video_scene_dict)

    # 3. run colmap for each video scene
    if not skip_colmap_ba:
        num_tasks = len(colmap_paths)
        total_threads = multiprocessing.cpu_count()
        num_workers = max(total_threads // num_tasks, 8)
        pool_size = num_tasks

        CONSOLE.log(f'Using {num_workers} workers with pool size {pool_size} for colmap')

        if config.use_colmap_ba:
            run_colmap_partial = partial(run_colmap_ba, num_workers=num_workers)
        else:
            run_colmap_partial = partial(run_colmap_triangulation, num_workers=num_workers)

        with multiprocessing.Pool(processes=pool_size) as pool:
            return_codes = pool.map(run_colmap_partial, colmap_paths)

        if not all(return_codes):
            for colmap_path, return_code in zip(colmap_paths, return_codes):
                if not return_code:
                    CONSOLE.log(f'[ERROR] Colmap failed for {os.path.basename(colmap_path)}')
            exit(1)

        CONSOLE.log('All colmap done')
    else:
        CONSOLE.log('Skipping colmap')

    # 4. merge all the models
    video_scene_dict_new = {}
    for video_scene_dict, colmap_path in zip(single_video_scene_dicts, colmap_paths):
        video_scene_dict = align_model(video_scene, video_scene_dict, colmap_path)
        video_scene_dict_new.update(video_scene_dict)

    # exporting sparse model for all videos, only for checking
    create_sparse_colmap_model(video_scene, video_scene_dict_new, os.path.join(video_scene.colmap_path, 'all'))

    for colmap_path in colmap_paths:
        clean_colmap_folder(colmap_path)

    video_scene.video_scene_dict = video_scene_dict_new
    video_scene.dump_pickle(video_scene.pickle_path_colmap)
    video_scene.update_pickle_link(video_scene.pickle_path_colmap)
