#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import argparse
import os
import shutil
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from nuplan_scripts.utils.config import load_config, RoadBlockConfig
from nuplan_scripts.utils.video_scene_dict_tools import VideoScene
from nuplan_scripts.utils.constants import NUPLAN_SENSOR_ROOT


class CollectRawData:

    class PseudoDataset(Dataset):
        def __init__(self, frame_infos, video_scene: VideoScene):
            self.frame_infos = frame_infos
            self.video_scene = video_scene

        def __getitem__(self, idx):
            frame_info = self.frame_infos[idx]

            raw_lidar_path = os.path.join(self.video_scene.raw_lidar_path, frame_info['lidar_path'])
            if not os.path.exists(raw_lidar_path):
                os.makedirs(os.path.dirname(raw_lidar_path), exist_ok=True)
                shutil.copy2(
                    os.path.join(NUPLAN_SENSOR_ROOT, frame_info['lidar_path']),
                    raw_lidar_path
                )

            for cam_info in frame_info['cams'].values():
                raw_path = os.path.join(self.video_scene.raw_image_path, cam_info['data_path'])
                if os.path.exists(raw_path):
                    continue
                os.makedirs(os.path.dirname(raw_path), exist_ok=True)
                shutil.copy2(
                    os.path.join(NUPLAN_SENSOR_ROOT, cam_info['data_path']),
                    raw_path
                )

            return frame_info

        def __len__(self):
            return len(self.frame_infos)
    
    @staticmethod
    def run(
        video_scene: VideoScene, 
        video_scene_dict,
        num_workers):

        total_frame_infos = []
        for video_token in video_scene_dict:
            frame_infos = video_scene_dict[video_token]['frame_infos']
            total_frame_infos.extend(frame_infos)

        dataset = CollectRawData.PseudoDataset(
            frame_infos=total_frame_infos,
            video_scene=video_scene
        )

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: x, pin_memory=False, drop_last=False)

        for _ in tqdm(dataloader, desc='processing images', ncols=120, leave=False):
            pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    config: RoadBlockConfig = load_config(args.config)

    if config.collect_raw:
        video_scene = VideoScene(config)
        video_scene_dict = video_scene.load_pickle(video_scene.pickle_path)

        CollectRawData.run(
            video_scene=video_scene,
            video_scene_dict=video_scene_dict,
            num_workers=args.num_workers
        )
