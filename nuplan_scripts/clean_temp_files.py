#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import os
import argparse
import shutil
from nuplan_scripts.utils.config import load_config, RoadBlockConfig
from nuplan_scripts.utils.video_scene_dict_tools import VideoScene

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    config: RoadBlockConfig = load_config(args.config)
    video_scene = VideoScene(config)

    data_root = video_scene.data_root

    folders_to_remove = [
        "masks",
        "depth",
        "registration_results",
        "colmap",
        "instance_point_cloud",
        "rgb_point_cloud",
        "sfm_point_cloud"
    ]

    for folder in folders_to_remove:
        dir_path = os.path.join(data_root, config.road_block_name, folder)
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
