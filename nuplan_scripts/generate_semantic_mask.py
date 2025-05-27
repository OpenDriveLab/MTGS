#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import os
import argparse


import cv2
import numpy as np
from PIL import Image

import torch

from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
from accelerate import Accelerator
from accelerate.utils import tqdm

from nuplan_scripts.utils.config import load_config, RoadBlockConfig
from nuplan_scripts.utils.video_scene_dict_tools import VideoScene
from nuplan_scripts.utils.constants import NUPLAN_SENSOR_ROOT

model_path = "facebook/mask2former-swin-large-cityscapes-semantic"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    config: RoadBlockConfig = load_config(args.config)

    video_scene = VideoScene(config)
    video_scene_dict = video_scene.load_pickle(video_scene.pickle_path)

    total_cams = []
    for video_token in video_scene_dict:
        frame_infos = video_scene_dict[video_token]['frame_infos']
        # generate semantic mask for stacking RGB point cloud
        frame_infos = [info for info in frame_infos if info.get('skipped', False) != 'low_velocity']
        for info in frame_infos:
            total_cams.extend(list(info['cams'].values()))

    processor = Mask2FormerImageProcessor(do_resize=False)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path)
    distributed_state = Accelerator()

    distributed_state.prepare_model(model, evaluation_mode=True)
    model.eval()
    # model.to(distributed_state.device).eval()

    pbar = tqdm(total=len(total_cams), ncols=120, desc="Generating semantic masks")

    with distributed_state.split_between_processes(total_cams) as partial_frames:
        for cam_info in partial_frames:
            image_path = os.path.join(NUPLAN_SENSOR_ROOT, cam_info['data_path'])
            mask_path = os.path.join(
                video_scene.raw_mask_path,
                video_scene.mask_suffix_cityscape,
                cam_info['data_path']).replace('.jpg', '.png')

            image = Image.open(image_path)
            input = processor(images=image, return_tensors="pt").pixel_values.to(distributed_state.device)
            with torch.no_grad():
                outputs = model(input)
            predicted_segmentation_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
            predicted_segmentation_map = predicted_segmentation_map.cpu().numpy().astype(np.uint8)

            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            cv2.imwrite(mask_path, predicted_segmentation_map)

            pbar.update(distributed_state.num_processes)
