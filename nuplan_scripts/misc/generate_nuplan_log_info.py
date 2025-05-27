"""
Used to generate the mapping from log name to lidar pc token.
This file is designed to run with OpenScene data, which means log with sensor data only.
If you don't have one, you can rewrite it with nuplan log_db, but will be slow.
"""
import argparse
import os
import jsonlines
import pickle
from pathlib import Path
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--openscene_dataroot', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    splits = ['trainval', 'test']
    files = []
    for split in splits:
        metadata_folder_path = Path(args.openscene_dataroot) / 'meta_datas' / split
        for file in metadata_folder_path.iterdir():
            if file.is_file():
                files.append(file)

    log2lidar_pc_token = {}
    for filename in tqdm(files, ncols=120):
        with filename.open('rb') as f:
            data_infos = pickle.load(f)
        log_name = filename.name[:-4]

        trajectory = np.asarray([info['ego2global_translation'] for info in data_infos])[:, :2]
        trajectory = np.round(trajectory, 2).tolist()

        log2lidar_pc_token[log_name] = {
            "log_name": log_name,
            "log_token": data_infos[0]['log_token'],
            "city": data_infos[0]['map_location'],
            "split": filename.parent.name,
            "lidar_pc_tokens": [info['token'] for info in data_infos],
            "trajectory": trajectory
        }

    with jsonlines.open(os.path.join(args.output_dir, 'nuplan_log_infos.jsonl'), 'w') as writer:
        writer.write_all(log2lidar_pc_token.values())
