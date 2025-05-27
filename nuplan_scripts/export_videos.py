#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import os
import argparse
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import cv2
import mediapy as media
from torch.utils.data import Dataset, DataLoader

from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.actor_state.state_representation import Point2D

from nuplan_scripts.utils.config import load_config, RoadBlockConfig
from nuplan_scripts.utils.video_scene_dict_tools import VideoScene
from nuplan_scripts.utils.constants import NUPLAN_SENSOR_ROOT, NUPLAN_MAP_VERSION, NUPLAN_MAPS_ROOT


class ExportVideos:
    class PseudoDataset(Dataset):
        CAMS = [
            'CAM_L0', 'CAM_F0', 'CAM_R0',
            'CAM_L1', 'WHITE', 'CAM_R1',
            'CAM_R2', 'CAM_B0', 'CAM_L2',
        ]

        def __init__(self, data_info):
            self.data_info = data_info

        def __getitem__(self, idx):
            info = self.data_info[idx]
            images = []
            for cam in self.CAMS:
                if cam == 'WHITE':
                    images.append(np.ones((360, 640, 3), dtype=np.uint8))
                    continue

                cam_info = info['cams'][cam]
                cam_path = os.path.join(NUPLAN_SENSOR_ROOT, cam_info['data_path'])
                image = cv2.imread(cam_path)[..., ::-1]
                image = cv2.resize(image, (640, 360))
                images.append(image)
            first_row = np.concatenate(images[:3], axis=1)
            mid_row = np.concatenate(images[3:6], axis=1)
            last_row = np.concatenate(images[6:], axis=1)
            whole_image = np.concatenate([first_row, mid_row, last_row], axis=0)

            return whole_image

        def __len__(self):
            return len(self.data_info)

    @staticmethod
    def run(data_infos, num_workers, output_path):
        dataset = ExportVideos.PseudoDataset(data_infos)
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: x, pin_memory=False, drop_last=False)

        video_frames = []
        for batch in tqdm(dataloader, desc='processing frames', ncols=120, leave=False):
            whole_image = batch[0]
            video_frames.append(whole_image)

        media.write_video(
            output_path, 
            video_frames, 
            fps=10
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    config: RoadBlockConfig = load_config(args.config)

    video_scene = VideoScene(config)
    video_scene_dict = video_scene.load_pickle(video_scene.pickle_path_raw)
    video_scene_dict = video_scene.video_scene_dict_process(['inject_trajectory'])
    map_api = get_maps_api(NUPLAN_MAPS_ROOT, NUPLAN_MAP_VERSION, config.city)

    road_block = config.road_block
    road_block_center = np.array([road_block[0] + road_block[2], road_block[1] + road_block[3]]) / 2
    center_point = Point2D(road_block_center[0], road_block_center[1])
    road_block_size = np.array([road_block[2] - road_block[0], road_block[3] - road_block[1]]).max()

    map_objects = map_api.get_proximal_map_objects(
        center_point, 
        road_block_size * 0.6, 
        [SemanticMapLayer.LANE, SemanticMapLayer.INTERSECTION]
    )

    all_map_objects = []
    for geos in map_objects.values():
        for geo in geos:
            geo = geo.polygon
            all_map_objects.append(geo)
    exteriors = []
    interiors = []

    for poly in all_map_objects:
        exteriors.append(poly.exterior)
        for inter in poly.interiors:
            interiors.append(inter)

    os.makedirs(f'{video_scene.sub_data_root}/map_vis', exist_ok=True)
    for video_token in video_scene_dict:
        video_idx = int(video_token.split('-')[-1])
        trajectory = np.array(video_scene_dict[video_token]['trajectory'])
        trajectory = trajectory[:, :2] + road_block_center[None]
        trajectory = trajectory[::5]
        if len(trajectory) < 2:
            continue

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.axis('equal')
        ax.axis('off')
        buffer = config.expand_buffer
        ax.set_xlim(road_block[0]-buffer, road_block[2]+buffer)
        ax.set_ylim(road_block[1]-buffer, road_block[3]+buffer)

        for ex in exteriors:
            ax.plot(*ex.xy, linewidth=0.8, alpha=0.5, c='r')
        for inter in interiors:
            ax.plot(*inter.xy, linewidth=0.8, alpha=0.5, c='r')

        ax.add_patch(
            plt.Rectangle(
                (config.road_block[0], config.road_block[1]), 
                config.road_block[2] - config.road_block[0], 
                config.road_block[3] - config.road_block[1], 
                edgecolor='red', 
                facecolor='none'
            )
        )

        ax.plot(
            trajectory[:, 0], 
            trajectory[:, 1], 
            linewidth=1.0, 
            marker='.', 
            label=f'{video_idx}', 
            alpha=0.8,
            markersize=1.2
        )
        ax.annotate('', xy=(trajectory[-1, 0], trajectory[-1, 1]),
                    xytext=(trajectory[-2, 0], trajectory[-2, 1]),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.0),
                    annotation_clip=False)
        ax.scatter(
            trajectory[-1, 0], 
            trajectory[-1, 1], 
            alpha=1.0,
            s=2
        )

        plt.legend()
        plt.savefig(f'{video_scene.sub_data_root}/map_vis/{video_token}.png', bbox_inches='tight', dpi=300)
        plt.close()

    os.makedirs(video_scene.raw_video_path, exist_ok=True)
    for video_token in tqdm(video_scene_dict, desc='exporting videos', ncols=120):
        ExportVideos.run(
            data_infos=video_scene_dict[video_token]['frame_infos'],
            num_workers=args.num_workers,
            output_path=f'{video_scene.raw_video_path}/{video_token}.mp4'
        )
