#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import os
import copy
import argparse

from pathlib import Path

import numpy as np
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from prettytable import PrettyTable

from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.actor_state.state_representation import Point2D

from nuplan_scripts.utils.config import load_config, RoadBlockConfig, FrmaeCentralConfig
from nuplan_scripts.utils.video_scene_dict_tools import VideoScene
from nuplan_scripts.utils.constants import NUPLAN_MAP_VERSION, NUPLAN_MAPS_ROOT, CONSOLE


def calculate_errors(video_scene_dict: dict, result_poses_dict: dict):
    error_dict = {}
    total_errors = []
    total_heading_errors = []
    for video_token in video_scene_dict:
        frame_infos = video_scene_dict[video_token]['frame_infos']
        errors = []
        heading_errors = []
        for idx, info in enumerate(frame_infos):
            token = info['token']
            pose_icp = result_poses_dict[token]

            T = pose_icp[:2, 3]
            T_orginal = info['ego2global'][:2, 3]
            error = np.linalg.norm(T - T_orginal)
            errors.append(error)

            heading_icp = Quaternion(matrix=pose_icp[:3, :3]).yaw_pitch_roll[0]
            heading_orginal = Quaternion(matrix=info['ego2global'][:3, :3]).yaw_pitch_roll[0]
            heading_error = np.rad2deg(np.abs(heading_icp - heading_orginal))
            heading_errors.append(heading_error)

        total_errors.extend(errors)
        total_heading_errors.extend(heading_errors)
        max_end_point_error = max(errors[0], errors[-1])
        error_dict[video_token] = {
            'max_end_point_error': max_end_point_error,
            'mean_error': np.mean(errors),
            'mean_heading_error': np.mean(heading_errors)
        }

    table = PrettyTable(
        field_names = ["Video idx", "EPE", "ATE-BEV", "ARE-BEV"]
    )
    table.float_format = ".4"
    table.align = "r"
    for video_token, errors in error_dict.items():
        table.add_row([int(video_token.split('-')[-1]), errors['max_end_point_error'], errors['mean_error'], errors['mean_heading_error']])
    table.add_row(["Total", max(total_errors), np.mean(total_errors), np.mean(total_heading_errors)])
    print(table)
    return error_dict

def align_poses(video_scene_dict:dict, result_poses_dict: dict):
    original_poses = []
    icp_poses = []
    for video_token in video_scene_dict:
        frame_infos = video_scene_dict[video_token]['frame_infos']
        for idx, info in enumerate(frame_infos):
            token = info['token']
            pose_icp = result_poses_dict[token]

            T = pose_icp[:2, 3]
            T_orginal = info['ego2global'][:2, 3]
            original_poses.append(T_orginal)
            icp_poses.append(T)
    original_poses = np.array(original_poses)
    icp_poses = np.array(icp_poses)

    # Center the points
    original_centered = original_poses - np.mean(original_poses, axis=0)
    icp_centered = icp_poses - np.mean(icp_poses, axis=0)

    H = icp_centered.T @ original_centered
    U, _, Vh = np.linalg.svd(H)
    R = Vh.T @ U.T
    if np.linalg.det(R) < 0:
        Vh[-1, :] *= -1
        R = Vh.T @ U.T
    t = np.mean(original_poses, axis=0) - np.mean(icp_poses, axis=0)

    aligned_transform = np.eye(4)
    aligned_transform[:2, :2] = R[:2, :2]
    aligned_transform[:2, 3] = t[:2]

    # Update the result_poses_dict with aligned poses
    result_poses_dict = copy.deepcopy(result_poses_dict)
    for video_token in video_scene_dict:
        frame_infos = video_scene_dict[video_token]['frame_infos']
        for info in frame_infos:
            token = info['token']
            pose_icp = result_poses_dict[token]
            # Update the translation part of the pose
            pose_icp = aligned_transform @ pose_icp
            result_poses_dict[token] = pose_icp

    return result_poses_dict

def visualize_lidar_registration(video_scene: VideoScene, result_poses_dict: dict, filename="registration.pdf"):
    config = video_scene.config
    road_block = config.road_block
    buffer = config.expand_buffer

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.axis('equal')
    ax.set_xlim(road_block[0] - buffer, road_block[2] + buffer)
    ax.set_ylim(road_block[1] - buffer, road_block[3] + buffer)

    # visualize the map
    map_api = get_maps_api(NUPLAN_MAPS_ROOT, NUPLAN_MAP_VERSION, config.city)
    map_objects = map_api.get_proximal_map_objects(
        point=Point2D(x=video_scene.road_block_center[0], y=video_scene.road_block_center[1]), 
        radius=(max(video_scene.block_size) + video_scene.config.expand_buffer) / 2,
        layers=[
            SemanticMapLayer.ROADBLOCK, 
            SemanticMapLayer.INTERSECTION, 
            SemanticMapLayer.CARPARK_AREA
        ]
    )
    for geos in map_objects.values():
        for geo in geos:
            polygon_points = np.asarray(geo.polygon.exterior.coords)
            ax.plot(polygon_points[:, 0], polygon_points[:, 1], 'magenta')

    # visualize ego trajectory
    for video_token in video_scene_dict.keys():
        frame_infos = video_scene_dict[video_token]["frame_infos"]
        raw_trajectory = np.array([info['ego2global'][:2, 3] for info in frame_infos]) + video_scene.road_block_center[:2][None]
        icp_trajectory = np.array([result_poses_dict[info['token']][:2, 3] for info in frame_infos]) + video_scene.road_block_center[:2][None]

        ax.plot(raw_trajectory[:, 0], raw_trajectory[:, 1], 'lime', linewidth=2, alpha=0.8)
        ax.plot(icp_trajectory[:, 0], icp_trajectory[:, 1], 'cyan', linewidth=2, alpha=0.8)

    # visualize the road block
    ax.add_patch(
        plt.Rectangle(
            (config.road_block[0] - video_scene.road_block_center[0],
             config.road_block[1] - video_scene.road_block_center[1]),
            config.road_block[2] - config.road_block[0],
            config.road_block[3] - config.road_block[1],
            edgecolor='red', 
            facecolor='none'
        )
    )
    plt.savefig(f"{video_scene.sub_data_root}/map_vis/{filename}", bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    config: RoadBlockConfig = load_config(args.config)
    video_scene = VideoScene(config)
    video_scene_dict = video_scene.load_pickle(video_scene.pickle_path_raw)

    from kiss_icp.datasets.mtgs import MTGSDataset
    from kiss_icp.pipeline import OdometryPipeline

    kiss_icp_dataset = MTGSDataset(
        data_dir = Path(os.path.dirname(video_scene.pickle_path)),
        video_scene = video_scene
    )

    kiss_icp_pipeline = OdometryPipeline(
        dataset=kiss_icp_dataset,
        config=None,
        max_range=max(video_scene.block_size) / 2 + config.expand_buffer + 50,
        deskew=False,
        visualize=False,
    )
    kiss_icp_pipeline.config.out_dir = video_scene.registration_path
    kiss_icp_pipeline.run()

    # The registritaion works on the first frames coordinate.
    # We translate the poses back to the global coordinate here.
    result_poses = kiss_icp_dataset.apply_calibration(kiss_icp_pipeline.poses)
    tokens = kiss_icp_dataset.tokens
    result_poses_dict = {token: pose for token, pose in zip(tokens, result_poses)}

    CONSOLE.log("ICP errors", style="bold green")
    error_dict = calculate_errors(video_scene_dict, result_poses_dict)
    visualize_lidar_registration(video_scene, result_poses_dict)

    CONSOLE.log("ICP errors after alignment", style="bold green")
    result_poses_dict_aligned = align_poses(video_scene_dict, result_poses_dict)
    error_dict_aligned = calculate_errors(video_scene_dict, result_poses_dict_aligned)
    visualize_lidar_registration(video_scene, result_poses_dict_aligned, filename="registration_aligned.pdf")

    # filter out bad video_info
    bad_videos = []
    for video_token in list(video_scene_dict.keys()):
        max_end_point_error = error_dict_aligned[video_token]['max_end_point_error']
        mean_error = error_dict_aligned[video_token]['mean_error']
        if max_end_point_error > 1.0 or mean_error > 0.5:
            bad_videos.append(video_token)

    if isinstance(config, FrmaeCentralConfig) and list(video_scene_dict.keys())[0] in bad_videos:
        CONSOLE.log("WARNING: The central log has bad registration result. Use log pose and disable multi-traversal.", style="bold red")
        first_key = list(video_scene_dict.keys())[0]
        video_scene_dict = {first_key: video_scene_dict[first_key]}
    else:
        if config.exclude_bad_registration:
            for video_token in bad_videos:
                CONSOLE.log(f"Remove video {video_token.split('-')[-1]} due to bad registration result.", style="yellow")
                video_scene_dict.pop(video_token)
            if len(video_scene_dict) == 0:
                raise ValueError("All videos have bad registration result.")

        for video_token in video_scene_dict:
            frame_infos = video_scene_dict[video_token]['frame_infos']
            for info in frame_infos:
                token = info['token']
                pose_icp = result_poses_dict_aligned[token]
                info['ego2global_original'] = info['ego2global'].copy()
                info['ego2global'] = pose_icp
                info['ego2global_translation'] = pose_icp[:3, 3]
                info['ego2global_rotation'] = Quaternion(matrix=pose_icp[:3, :3])
                info['lidar2global'] = pose_icp @ info['lidar2ego']

    video_scene.video_scene_dict = video_scene_dict
    # after registration, we filter out low velocity frames
    video_scene_dict = video_scene.video_scene_dict_process([
        # skip frames with low velocity
        'filter_low_velocity',
        'inject_trajectory'
    ], inline=True)

    video_scene.dump_pickle(video_scene.pickle_path_registered)
    video_scene.update_pickle_link(video_scene.pickle_path_registered)
