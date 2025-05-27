#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import os
import yaml
import pytz
from rich.console import Console

CONSOLE = Console(width=120)
os.environ['TQDM_NCOLS'] = '120'

# load nuplan config
nuplan_config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'common', 'nuplan_path.yml')
nuplan_config = yaml.load(open(nuplan_config_path, 'r'), Loader=yaml.FullLoader)
NUPLAN_DATA_ROOT = nuplan_config['NUPLAN_DATA_ROOT']
NUPLAN_DB_FILES = nuplan_config['NUPLAN_DB_FILES']
NUPLAN_SENSOR_ROOT = nuplan_config['NUPLAN_SENSOR_ROOT']
NUPLAN_MAP_VERSION = nuplan_config['NUPLAN_MAP_VERSION']
NUPLAN_MAPS_ROOT = nuplan_config['NUPLAN_MAPS_ROOT']

cityscape_label = {
    'road': 0,
    'sidewalk': 1,
    'building': 2,
    'wall': 3,
    'fence': 4,
    'pole': 5,
    'traffic light': 6,
    'traffic sign': 7,
    'vegetation': 8,
    'terrain': 9,
    'sky': 10,
    'person': 11,
    'rider': 12,
    'car': 13,
    'truck': 14,
    'bus': 15,
    'train': 16,
    'motorcycle': 17,
    'bicycle': 18
}

cityscape_colormap = {
    'person': (220, 20, 60),       # Red
    'car': (0, 0, 142),            # Dark blue
    'truck': (0, 0, 70),           # Darker blue
    'bus': (0, 60, 100),           # Blue
    'rider': (255, 0, 0),          # Bright red
    'motorcycle': (0, 0, 230),     # Bright blue
    'bicycle': (119, 11, 32),      # Dark red
}

NUPLAN_ACCEPTABLE_CITYSCAPE_LABELS = {
    "vehicle": [13, 14, 15], 
    "bicycle": [12, 17, 18], 
    "pedestrian": [11], 
    "traffic cone": [5], 
    "barrier": [], 
    "construction zone sign": [], 
    "generic object": [], 
    "background": []
}

NUPLAN_TIMEZONE = {
    'us-ma-boston': pytz.timezone('America/New_York'),
    'sg-one-north': pytz.timezone('Asia/Singapore'),
    'us-nv-las-vegas-strip': pytz.timezone('America/Los_Angeles'),
    'us-pa-pittsburgh-hazelwood': pytz.timezone('US/Eastern'),
}
