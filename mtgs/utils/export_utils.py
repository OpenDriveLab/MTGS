#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
from pathlib import Path
from collections import OrderedDict

import torch

from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.pipelines.base_pipeline import VanillaPipeline
from mtgs.scene_model.mtgs_scene_graph import MTGSSceneModel

def extract_portable_ckpt(pipeline: VanillaPipeline, output_path: Path) -> torch.nn.Module:
    model: MTGSSceneModel = pipeline.model
    model.eval()
    gaussians = getattr(model, "gaussian_models", None)
    if gaussians is None:
        CONSOLE.log("[WARNING] No Gaussian models found in the model.", "red")
        return None

    video_scene = pipeline.datamanager.dataparser.video_scene
    train_scene_travels = pipeline.datamanager.dataparser.config.train_scene_travels
    video_scene.video_scene_dict_process(
        {'type': 'filter_by_video_idx', 'kwargs': {'video_idxs': train_scene_travels}}, inline=True
    )

    recon2world_translation = pipeline.datamanager.dataparser.recon2world_translation

    portable_model = OrderedDict()
    for submodel_name, submodel in gaussians.items():
        portable_config = submodel.portable_config

        if submodel_name == "background":
            portable_config["recon2world_translation"] = recon2world_translation

        portable_model[submodel_name] = OrderedDict({
            "config": portable_config,
            "state_dict": submodel.state_dict()
        })

    torch.save(portable_model, output_path)
    return portable_model
