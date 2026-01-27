#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import os
import argparse
from nuplan_scripts.utils.config import load_config, RoadBlockConfig
from nuplan_scripts.utils.video_scene_dict_tools import VideoScene

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    config: RoadBlockConfig = load_config(args.config)
    video_scene = VideoScene(config)

    data_root = video_scene.data_root
    os.system(f"rm -rf {data_root}/{config.road_block_name}/masks")
    os.system(f"rm -rf {data_root}/{config.road_block_name}/depth")
    os.system(f"rm -rf {data_root}/{config.road_block_name}/registration_results")
    os.system(f"rm -rf {data_root}/{config.road_block_name}/colmap")
    os.system(f"rm -rf {data_root}/{config.road_block_name}/instance_point_cloud")
    os.system(f"rm -rf {data_root}/{config.road_block_name}/rgb_point_cloud")
    os.system(f"rm -rf {data_root}/{config.road_block_name}/sfm_point_cloud")
