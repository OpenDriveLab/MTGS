# Instructions for preparing the dataset

## Overview
The MTGS is built with multi-traversal data from the [nuplan](https://www.nuscenes.org/nuplan) dataset. 
You can download the prepared dataset or prepare the dataset by yourself.

## Contents
- [Download the prepared dataset](#download-the-prepared-dataset)
- [Understanding the dataset structure](#understanding-the-dataset-structure)
    - [Data config](#data-config)
    - [Folder structure](#folder-structure)
    - [video_scene_dict schema](#video_scene_dict-schema)
- [Prepare the dataset by yourself](#prepare-the-dataset-by-yourself)

## Download the prepared dataset

We upload the prepared dataset used in the MTGS paper to the [Hugging Face](https://huggingface.co/datasets/OpenDriveLab/MTGS/tree/main/MTGS_paper_data). Please download the dataset and extract it in following structure:

```
MTGS/data/MTGS/
├── {road_block_name_1}
├── {road_block_name_2}
├── ...
├── ego_masks
└── nuplan_log_infos.jsonl
```

## Understanding the dataset structure

We build a custom data format form MTGS. Each reconstruction data can be represented with a data config file. After data preprocessing, all the relavent data, including the image, lidar, depth, semantic mask, etc, are stored in the folder named with `{road_block_name}`. A `video_scene_dict.pkl` file is generated and includes the metadata of the reconstruction data.

### Data config

The main configuration class for MTGS dataset preparation is `RoadBlockConfig`. Here are the available parameters:

#### RoadBlockConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `road_block_name` | `str` | *required* | Unique identifier for the road block reconstruction |
| `road_block` | `Tuple[int, int, int, int]` | *required* | Global coordinates (x_min, y_min, x_max, y_max) defining the road block region |
| `city` | `Literal` | *required* | City location from nuPlan dataset. Options: `'sg-one-north'`, `'us-ma-boston'`, `'us-na-las-vegas-strip'`, `'us-pa-pittsburgh-hazelwood'` |
| `data_root` | `str` | `"./data/MTGS"` | Root directory where processed data will be stored |
| `interval` | `int` | `1` | Frame sampling interval (interval = 1 means 10Hz in nuplan) |
| `expand_buffer` | `int` | `0` | Expand trajectory by this many meters. LiDAR point clouds within this buffer are used for LiDAR registration and stacking LiDAR points|
| `reconstruct_buffer` | `int` | `0` | Buffer in meters for better in-range reconstruction. Frames within this buffer are used for reconstruction, including caching images, LiDAR, semantic map, etc. |
| `selected_videos` | `Tuple` | `()` | Tuple of video indices or dict of video_idx with start_frame and end_frame for manual video selection |
| `split` | `Literal` | `'trainval'` | Data source from nuPlan split. Options: `'trainval'`, `'test'`, `'all'` |
| `collect_raw` | `bool` | `False` | Whether to collect raw data from nuPlan sensor root to data_root. If False, the codebase will read the data from nuplan sensor root. |
| `exclude_bad_registration` | `bool` | `True` | Whether to exclude videos with bad LiDAR registration results |
| `use_colmap_ba` | `bool` | `False` | Whether to use serveral rounds of COLMAP bundle adjustment to refine the registration result |

We store the config files used in the MTGS paper in `nuplan_scripts/configs/mtgs_exp` folder as yaml format.

#### Example config File
```yaml
!!python/object:nuplan_scripts.utils.config.RoadBlockConfig
city: us-ma-boston
data_root: ./data/MTGS
interval: 1
reconstruct_buffer: 0
expand_buffer: 0
exclude_bad_registration: false
use_colmap_ba: false
collect_raw: false
split: trainval
road_block: !!python/tuple
- 331120
- 4690660
- 331190
- 4690710
road_block_name: road_block-331220_4690660_331190_4690710
selected_videos: !!python/tuple
- 0
- 1
- 2
- 6
- 7
- 8

```

### Folder structure

```
MTGS/data/MTGS/{road_block_name}
├── videos
├── map_vis
├── masks
│   └── raw
│       ├── cityscape
│       └── ego
├── images
│   └── raw
├── lidar
├── depth
│   └── undistorted_optimal
├── sfm_point_cloud
├── rgb_point_cloud
├── instance_point_cloud
└── video_scene_dict.pkl
```

### video_scene_dict schema

`video_scene_dict.pkl` is a stored dict with python pickle format. The key is the token of a video from one traversal, and the value is a dict with the following fields:
```
{
    'video_token':      <str> -- The token of a video from one traversal. Usually named with '{road_block_name}-{video_idx}'.
    'log_token':        <str> -- The 16-digits uuid token of a log from nuplan.
    'log_name':         <str> -- The name of a log from nuplan, e.g. '2021.05.12.19.36.12_veh-35_00005_00204'
    'map_location':     <str> -- The location of a log from nuplan.
    'vehicle_name':     <str> -- The name of a vehicle from nuplan, e.g. 'veh-35'
    'start_ts':         <int> -- The start timestamp of the video.
    'end_ts':           <int> -- The end timestamp of the video.
    'date':             <datetime.date> -- The date of the video in relavant timezone.
    'hour':             <int> -- The hour of the video in relavant timezone.
    'trajectory':       <array> -- [n, 3] The trajectory of the ego vehicle.
    'frame_infos':      <list> -- [n] List of dicts, each dict is a frame_info.
}
```

#### frame_info schema

```
{
    'token':                                <str> -- Unique record identifier. Pointing to the nuPlan lidar_pc.
    'frame_idx':                            <int> -- Indicates the idx of the current frame.
    'skipped':                              Literal<'out_of_region', 'low_velocity', False> -- label for skipping frames.
    'timestamp':                            <int> -- Unix time stamp.
    'log_name':                             <str> -- Short string identifier.
    'log_token':                            <str> -- Foreign key pointing to the log.
    'map_location':                         <str> -- Area where log was captured.
    'vehicle_name':                         <str> -- String identifier for the current car.
    'can_bus':                              <array> -- [18] definition see below
    'ego2global_translation':               <array> -- [3] Translation matrix from ego coordinate system to global coordinate system
    'ego2global_rotation':                  <array> -- [4] Quaternion from ego coordinate system to global coordinate system
    'ego2global':                           <array> -- [4, 4] Transformation matrix
    'lidar_path':                           <str> -- The relative path to store the lidar data.
    'lidar2ego_translation':                <array> -- [3] Translation vector from the lidar coordinate system to the ego coordinate system.
    'lidar2ego_rotation':                   <array> -- [4] Quaternion from the lidar coordinate system to the ego coordinate system.
    'lidar2ego':                            <array> -- [4, 4] Transformation matrix
    'lidar2global':                         <array> -- [4, 4] Transformation matrix
    'cams': {
        {cam_name}: {
            'data_path':                    <str> -- The relative path to store the image data.
            'sensor2ego_rotation':          <array> -- [4] Quaternion from sensor to ego coordinate system.
            'sensor2ego_translation':       <array> -- [3] Translation matrix from sensor to ego coordinate system.
            'cam_intrinsic':                <array> -- [3, 3] Intrinsic matrix of the camera 
            'distortion':                   <array> -- [5] The distortion coefficiants of camera. [k1, k2, p1, p2, k3]
            'token':                        <str> -- Unique image token.
            'timestamp':                    <int> -- Unix time stamp.
        }
        ...
    }
    'gt_boxes':                         <array> -- [n, 7], Ground truth boxes. (x,y,z,l,w,h,heading) in ego coordinate system.
    'gt_names':                         <list> -- [n] List of class names.
    'gt_velocity_3d':                   <array> -- [n, 3] 3D velocity. Note: the velocity from nuplan is not accurate.
    'instance_tokens':                  <list> -- Unique record identifier of single frame instance.
    'track_tokens':                     <list> -- Unique record identifier of tracking instance.
}
```

#### can_bus schema

```
can_bus [                          <array> -- [18] loc: [0:3], quat: [3:7], accel: [7:10], velocity: [10:13], rotation_rate: [13:16]
   "x":                            <float> -- Ego vehicle location center_x in global coordinates(in meters).
   "y":                            <float> -- Ego vehicle location center_y in global coordinates(in meters).
   "z":                            <float> -- Ego vehicle location center_z in global coordinates(in meters).
   "qw":                           <float> -- Ego vehicle orientation in quaternions in global coordinates.
   "qx":                           <float> -- Ego vehicle orientation in quaternions in global coordinates.
   "qy":                           <float> -- Ego vehicle orientation in quaternions in global coordinates.
   "qz":                           <float> -- Ego vehicle orientation in quaternions in global coordinates.
   "acceleration_x":               <float> -- Ego vehicle acceleration x in local coordinates(in m/s2).
   "acceleration_y":               <float> -- Ego vehicle acceleration y in local coordinates(in m/s2).
   "acceleration_z":               <float> -- Ego vehicle acceleration z in local coordinates(in m/s2).
   "vx":                           <float> -- Ego vehicle velocity x in local coordinates(in m/s).
   "vy":                           <float> -- Ego vehicle velocity y in local coordinates(in m/s).
   "vz":                           <float> -- Ego vehicle velocity z in local coordinate(in m/s)s.
   "angular_rate_x":               <float> -- Ego vehicle angular rate x in local coordinates.
   "angular_rate_y":               <float> -- Ego vehicle angular rate y in local coordinates.
   "angular_rate_z":               <float> -- Ego vehicle angular rate z in local coordinates.
   0,
   0
]
```

## Prepare the dataset by yourself

### Download the nuplan data

Please first download the nuplan data from [nuplan](https://www.nuscenes.org/nuplan), including image, LiDAR, map and database files.
The whole nuplan data is about 20 TB.

You can modify the predefined path in `nuplan_scripts/configs/common/nuplan_path.yml` or put the data to the following structure:

```
MTGS/data/
├── nuplan/dataset/
│   ├── maps
│   └── nuplan-v1.1
└── MTGS
```

### Generate nuplan log info cache

Used to generate the mapping from log name to lidar pc token, to speed up the selection of multi-traversal logs within a given range. 
This script is designed to run with [OpenScene](https://github.com/OpenDriveLab/OpenScene) data. You do not need to run this script if you download the file `nuplan_log_infos.jsonl`.

```bash
python nuplan_scripts/misc/generate_nuplan_log_info.py \
    --openscene_dataroot ./data/OpenScene/openscene-v1.1 \
    --output_dir ./data/MTGS
```

### Run the preprocessing.

For a new road block config, you may need to first export all the videos and do a manual filter. We provide an online preview tool to help you do the filter.

``` bash
# some parameters
CONFIG=nuplan_scripts/configs/mtgs_exp/road_block-331220_4690660_331190_4690710.yml
NUM_WORKERS=16
NUM_GPUS=8

bash nuplan_scripts/preprocess_stage_1.sh $CONFIG $NUM_WORKERS

# filter trajectory by manual using online preview
python -m streamlit run preview.py -- --config_dir configs

bash nuplan_scripts/preprocess_stage_2.sh $CONFIG $NUM_WORKERS $NUM_GPUS
```

If you already specify the `selected_videos` in the config file, you can directly run the following command to generate the data.
```bash
bash nuplan_scripts/preprocess.sh $CONFIG $NUM_WORKERS $NUM_GPUS
```

### Generate frame central configs from given token list

You can also generate a list of `FrameCentralConfig` from a given token list. We read the token list from `navsim` filter config file.

``` bash
python -m nuplan_scripts.misc.generate_config_from_token \
    --navsim_config path/to/navsim/filter.yaml \
    --data_root data/{output_dir} \
    --output_dir  data/{output_dir}/configs \
    --num_workers $NUM_WORKERS
```

These configs can used to reconstruct data centered with a given token. You may need to look into the script to understand the parameters. By default, the reconstruction is in single-traversal mode. You can also enabling multi-traversal reconstruction for the FrameCentralConfig. In that case, the nearby traversal logs are automatically selected.
