#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.actor_state.state_representation import Point2D

from nuplan_scripts.utils.config import load_config, RoadBlockConfig
from nuplan_scripts.utils.video_scene_dict_tools import VideoScene
from nuplan_scripts.utils.constants import NUPLAN_MAP_VERSION, NUPLAN_MAPS_ROOT

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    config: RoadBlockConfig = load_config(args.config)

    video_scene = VideoScene(config)
    video_scene_dict = video_scene.load_pickle(video_scene.pickle_path_raw)

    video_scene_dict = video_scene.video_scene_dict_process([
        {'type': 'filter_by_video_idx', 'kwargs': {'video_idxs': config.selected_videos}},
        'inject_trajectory'
    ], inline=True)

    video_scene.dump_pickle(video_scene.pickle_path_filtered)
    video_scene.update_pickle_link(video_scene.pickle_path_filtered)


    # visualize trajectories
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

    for video_token in video_scene_dict:
        video_idx = int(video_token.split('-')[-1])
        trajectory = np.array(video_scene_dict[video_token]['trajectory'])
        trajectory = trajectory[:, :2] + road_block_center[None]
        trajectory = trajectory[::5]
        if len(trajectory) < 2:
            continue

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
    plt.savefig(f'{video_scene.sub_data_root}/map_vis/{video_scene.name}_filtered.png', bbox_inches='tight', dpi=300)
    plt.close()
