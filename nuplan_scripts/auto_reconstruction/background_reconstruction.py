#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import os
import argparse

os.environ["NERFSTUDIO_DATAPARSER_CONFIGS"] = "nuplan=mtgs.config.nuplan_dataparser:nuplan_dataparser"
os.environ["NERFSTUDIO_METHOD_CONFIGS"] = "mtgs_st=mtgs.config.MTGS_ST:method"

from nerfstudio.scripts.train import main
from mtgs.config.MTGS_ST import config as ns_config
from nuplan_scripts.utils.config import load_config, FrameCentralConfig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = os.path.join(config.data_root, 'background_reconstruction', config.road_block_name)
    os.makedirs(output_dir, exist_ok=True)

    assert isinstance(config, FrameCentralConfig)

    ns_config.experiment_name = config.road_block_name
    ns_config.pipeline.datamanager.cache_strategy = 'prefetch'
    ns_config.pipeline.datamanager.dataparser.road_block_config = args.config
    ns_config.pipeline.datamanager.dataparser.train_scene_travels = (0,)
    ns_config.pipeline.datamanager.dataparser.eval_scene_travels = (0,)
    ns_config.pipeline.datamanager.dataparser.only_moving = True
    ns_config.pipeline.model.color_corrected_metrics = False
    ns_config.pipeline.model.lpips_metric = False
    ns_config.pipeline.model.model_config['background'].verbose = False
    ns_config.output_dir = output_dir
    ns_config.set_base_dir(output_dir)

    main(ns_config)
