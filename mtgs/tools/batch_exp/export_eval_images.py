#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import os
import argparse
import yaml
import torch
from pathlib import Path
from mtgs.tools.batch_exp.mtgs_tasks import tasks_registry

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ns-config', type=str, default='mtgs/config/MTGS.py')
    parser.add_argument('--output-dir', type=str, default='experiments/main_mt/MTGS')
    parser.add_argument('--task-name', type=str, default='main_mt')
    args = parser.parse_args()

    # We hack the enviroment variable to register the method and dataparser to nerfstudio.
    # NOTE: the method name is `mtgs` here, not guaranteed to be the same as in the config.
    module_name = args.ns_config.replace('.py', '').replace('/', '.')
    os.environ["NERFSTUDIO_DATAPARSER_CONFIGS"] = "nuplan=mtgs.config.nuplan_dataparser:nuplan_dataparser"
    os.environ["NERFSTUDIO_METHOD_CONFIGS"] = 'mtgs=' + module_name + ':method'
    from nerfstudio.utils.eval_utils import eval_load_checkpoint

    def update_config(config):
        config.pipeline.image_saving_mode = "sequential"
        config.pipeline.datamanager.dataparser.eval_2hz = True
        # disable loading to speed up.
        config.pipeline.datamanager.load_mask = False
        config.pipeline.datamanager.load_custom_masks = ()
        config.pipeline.datamanager.load_instance_masks = False
        config.pipeline.datamanager.load_semantic_masks_from = False
        config.pipeline.datamanager.load_lidar_depth = False
        config.pipeline.datamanager.load_pseudo_depth = False

        config.pipeline.model.output_depth_during_training = False
        config.pipeline.model.predict_normals = False
        config.pipeline.model.color_corrected_metrics = False
        config.pipeline.model.lpips_metric = False
        config.pipeline.model.dinov2_metric = False
        return config

    def eval_setup(config_path, test_mode, update_config_callback):
        # load save config
        config = yaml.load(config_path.read_text(), Loader=yaml.Loader)

        if update_config_callback is not None:
            config = update_config_callback(config)

        # load checkpoints from wherever they were saved
        config.load_dir = config.get_checkpoint_dir()

        # setup pipeline (which includes the DataManager)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline = config.pipeline.setup(device=device, test_mode=test_mode)
        pipeline.eval()

        # load checkpointed information
        checkpoint_path, step = eval_load_checkpoint(config, pipeline)

        return config, pipeline, checkpoint_path, step


    tasks = tasks_registry[args.task_name]
    for task in tasks:
        load_config = Path(args.output_dir) / task["config"].split('/')[-1].split('.')[0] / 'config.yml'
        image_output_path = Path(args.output_dir) / 'rendered_images' / task["config"].split('/')[-1].split('.')[0]
        config, pipeline, checkpoint_path, _ = eval_setup(load_config, update_config_callback=update_config)
        metrics_dict = pipeline.get_average_eval_image_metrics(
            output_path=image_output_path,
            get_std=False
        )
