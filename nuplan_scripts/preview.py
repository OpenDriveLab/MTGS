#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import copy
import glob
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import os
import cv2
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.actor_state.state_representation import Point2D
from nuplan_scripts.utils.config import load_config, RoadBlockConfig
from nuplan_scripts.utils.video_scene_dict_tools import VideoScene
from nuplan_scripts.utils.constants import NUPLAN_MAP_VERSION, NUPLAN_MAPS_ROOT, NUPLAN_SENSOR_ROOT

def load_mtgs_config(config_path: str):
    config: RoadBlockConfig = load_config(config_path)
    video_scene = VideoScene(config)
    video_scene_dict, map_api = load_mtgs_data(config_path, video_scene)
    return video_scene, video_scene_dict, map_api

@st.cache_resource
def load_mtgs_data(config, _video_scene: VideoScene):
    video_scene_dict = _video_scene.load_pickle(_video_scene.pickle_path_raw)
    video_scene_dict = _video_scene.video_scene_dict_process([
        'inject_trajectory'
    ], inline=True)
    map_api = get_maps_api(NUPLAN_MAPS_ROOT, NUPLAN_MAP_VERSION, _video_scene.config.city)
    return video_scene_dict, map_api

def load_frame(idx, frame_infos):
    frame_info = frame_infos[idx]
    return load_frame_image(frame_info)

@st.cache_resource
def preparing_map_infos(config, _map_api):
    road_block = config.road_block
    road_block_center = np.array([road_block[0] + road_block[2], road_block[1] + road_block[3]]) / 2
    center_point = Point2D(road_block_center[0], road_block_center[1])
    road_block_size = np.array([road_block[2] - road_block[0], road_block[3] - road_block[1]]).max()

    map_objects = _map_api.get_proximal_map_objects(
        center_point, 
        road_block_size * 0.6, 
        [SemanticMapLayer.LANE, SemanticMapLayer.INTERSECTION]
    )

    return map_objects

def plot_trajectories(video_scene: VideoScene, video_scene_dict, map_api, selected_videos):
    config = video_scene.config
    road_block = config.road_block
    road_block_center = np.array([road_block[0] + road_block[2], road_block[1] + road_block[3]]) / 2
    map_objects = preparing_map_infos(config, map_api)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis('equal')
    ax.axis('off')
    buffer = config.expand_buffer
    ax.set_xlim(road_block[0]-buffer, road_block[2]+buffer)
    ax.set_ylim(road_block[1]-buffer, road_block[3]+buffer)

    # Plot map
    for geos in map_objects.values():
        for geo in geos:
            poly = geo.polygon
            ax.plot(*poly.exterior.xy, linewidth=0.8, alpha=0.5, c='r')
            for inter in poly.interiors:
                ax.plot(*inter.xy, linewidth=0.8, alpha=0.5, c='r')

    # Plot road block
    ax.add_patch(plt.Rectangle(
        (road_block[0], road_block[1]), 
        road_block[2] - road_block[0], 
        road_block[3] - road_block[1], 
        edgecolor='red', 
        facecolor='none'
    ))

    # Plot selected trajectory
    for selected_video in selected_videos:
        video_token = f"{video_scene.name}-{selected_video}"
        trajectory = np.array(video_scene_dict[video_token]['trajectory'])
        trajectory = trajectory[:, :2] + road_block_center[None]
        trajectory = trajectory[::5]

        ax.plot(
            trajectory[:, 0], 
            trajectory[:, 1], 
            linewidth=1.0, 
            marker='.', 
            label=f'{selected_video}', 
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
    fig.legend()

    return fig

def load_frame_image(frame_info):
    CAMS = [
        'CAM_L0', 'CAM_F0', 'CAM_R0',
        'CAM_L1', 'WHITE', 'CAM_R1',
        'CAM_R2', 'CAM_B0', 'CAM_L2',
    ]
    
    images = []
    for cam in CAMS:
        if cam == 'WHITE':
            images.append(np.ones((360, 640, 3), dtype=np.uint8))
            continue

        cam_info = frame_info['cams'][cam]
        cam_path = os.path.join(NUPLAN_SENSOR_ROOT, cam_info['data_path'])
        image = cv2.imread(cam_path)[..., ::-1]
        image = cv2.resize(image, (640, 360))
        images.append(image)
    
    first_row = np.concatenate(images[:3], axis=1)
    mid_row = np.concatenate(images[3:6], axis=1)
    last_row = np.concatenate(images[6:], axis=1)
    whole_image = np.concatenate([first_row, mid_row, last_row], axis=0)
    
    return whole_image

def create_app(
    config_list
):
    col1, col2, col3 = st.columns([1.5, 2, 4])

    with col1:
        # Display and select configs
        selected_config = st.selectbox(
            "Select Config",
            options=config_list,
            format_func=lambda x: os.path.basename(x).replace('.yml', '').replace('road_block-', '')
        )

        # Load the selected config
        if selected_config is not None:
            video_scene, video_scene_dict, map_api = load_mtgs_config(selected_config)
            st.session_state.video_choices = list(map(lambda x: int(x.split('-')[-1]), video_scene_dict.keys()))

            if selected_config != st.session_state.get('last_selected_config'):
                st.session_state.selected_videos = pd.DataFrame(columns=['idx', 'start_frame', 'end_frame'], dtype='Int64')
                for video_idx in video_scene.config.selected_videos:
                    if isinstance(video_idx, int):
                        video_token = f"{video_scene.name}-{video_idx}"
                        trajectory_length = len(video_scene_dict[video_token]['trajectory'])
                        st.session_state.selected_videos = pd.concat([st.session_state.selected_videos, pd.DataFrame({
                            'idx': [video_idx],
                            'start_frame': [None],
                            'end_frame': [None]
                        }, dtype='Int64')], ignore_index=True)
                    elif isinstance(video_idx, dict):
                        st.session_state.selected_videos = pd.concat([st.session_state.selected_videos, pd.DataFrame({
                            'idx': [video_idx['idx']],
                            'start_frame': [video_idx.get('start_frame', None)],
                            'end_frame': [video_idx.get('end_frame', None)]
                        }, dtype='Int64')], ignore_index=True)

            st.session_state.last_selected_config = selected_config

        # Separator for visual clarity
        st.markdown("---")
        selected_video = st.selectbox("Select Video", options=st.session_state.video_choices)

        if selected_video is not None:
            video_token = f"{video_scene.name}-{selected_video}"
            trajectory_length = len(video_scene_dict[video_token]['trajectory'])
            start_idx, end_idx = st.slider(
                "Trajectory Range",
                min_value=0,
                max_value=trajectory_length-1,
                value=(0, trajectory_length-1),
                step=1
            )

        col_add, col_export, col_confirm = st.columns(3)
        st.session_state.show_confirm_button = False

        with col_add:
            add_button = st.button("Add", use_container_width=True)
        with col_export:
            export_button = st.button("Export", use_container_width=True)

        if add_button:
            video_token = f"{video_scene.name}-{selected_video}"
            trajectory_length = len(video_scene_dict[video_token]['trajectory'])
            new_row = pd.DataFrame({
                'idx': [selected_video],
                'start_frame': [start_idx if start_idx != 0 else None],
                'end_frame': [end_idx if end_idx != trajectory_length-1 else None]
            }, dtype='Int64')
            st.session_state.selected_videos = pd.concat([st.session_state.selected_videos, new_row], ignore_index=True)

        if export_button:
            # Organize selected videos into the config
            _selected_videos = st.session_state.selected_videos.to_dict('records')
            formatted_selected_videos = []
            for video_idx in _selected_videos:
                if pd.isna(video_idx['start_frame']):
                    video_idx.pop('start_frame')
                elif isinstance(video_idx['start_frame'], str):
                    video_idx['start_frame'] = int(video_idx['start_frame'])

                if pd.isna(video_idx['end_frame']):
                    video_idx.pop('end_frame')
                elif isinstance(video_idx['end_frame'], str):
                    video_idx['end_frame'] = int(video_idx['end_frame'])

                if len(video_idx) == 1:
                    video_idx = video_idx['idx']
                formatted_selected_videos.append(video_idx)
            video_scene.config.selected_videos = tuple(formatted_selected_videos)

            # Print the config for user confirmation
            st.write(selected_config)
            st.json(video_scene.config.__dict__)
            st.session_state.show_confirm_button = True

        with col_confirm:
            save_button = st.button(
                "Confirm", 
                disabled=not st.session_state.get('show_confirm_button', True), 
                use_container_width=True)

        # Confirm save action
        if save_button:
            # Save the config to local disk
            video_scene.config.save_config(selected_config)
            st.success(f"Config saved to {selected_config}")
            print(f"Config saved to {selected_config}")

            st.session_state.last_selected_config = None
            st.session_state.show_confirm_button = False

            # Refresh the page after saving
            st.rerun()

        # Make the dataframe editable
        edited_df = st.data_editor(
            st.session_state.selected_videos,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
        )

        # Update the session state with the edited dataframe
        st.session_state.selected_videos = edited_df

    with col2:
        if selected_video is not None:
            video_token = f"{video_scene.name}-{selected_video}"
            trajectory = np.array(video_scene_dict[video_token]['trajectory'][start_idx:end_idx+1])
            fig = plot_trajectories(video_scene, {video_token: {'trajectory': trajectory}}, map_api, [selected_video])
            fig.set_size_inches(4, 4)
            st.pyplot(fig, use_container_width=True)

        if not st.session_state.selected_videos.empty:
            all_trajectories = {}
            video_indices = []
            for _, row in st.session_state.selected_videos.iterrows():
                _video_idx = row['idx']
                video_token = f"{video_scene.name}-{_video_idx}"

                _start_idx = row['start_frame'] if pd.notna(row['start_frame']) else 0
                _end_idx = row['end_frame'] if pd.notna(row['end_frame']) else len(video_scene_dict[video_token]['trajectory']) - 1

                trajectory = np.array(video_scene_dict[video_token]['trajectory'][_start_idx:_end_idx+1])
                all_trajectories[video_token] = {'trajectory': trajectory}
                video_indices.append(_video_idx)

            fig = plot_trajectories(video_scene, all_trajectories, map_api, video_indices)
            fig.set_size_inches(4, 4)
            st.pyplot(fig, use_container_width=True)
        else:
            st.write("No trajectories selected. Please add trajectories using the panel on the left.")
    
    with col3:
        # Add a button to load and play all video frames
        if st.button("Play Video"):
            if selected_video is not None:
                video_token = f"{video_scene.name}-{selected_video}"
                st.video(f'{video_scene.raw_video_path}/{video_token}.mp4', autoplay=True)
            else:
                st.warning("Please select a video first.")

        if selected_video is not None:
            video_token = f"{video_scene.name}-{selected_video}"
            frame_infos = video_scene_dict[video_token]['frame_infos']
            frame_idx = st.slider("Select Frame", start_idx, end_idx, start_idx)

            # Load and display the selected frame
            frame_info = frame_infos[frame_idx]
            image = load_frame_image(frame_info)
            st.image(image, caption=f"Video {selected_video}, Frame {frame_idx}", use_column_width=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', type=str, required=True)
    args = parser.parse_args()

    config_list = glob.glob(args.config_dir + '/*.yml')
    config_list.sort()

    st.set_page_config(
        page_title="Trajectory Visualization Tool",
        layout="wide"
    )

    create_app(config_list)
