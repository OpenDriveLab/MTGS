import os
import argparse
import jsonlines
import yaml
import numpy as np
import shapely
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB
from nuplan_scripts.utils.nuplan_utils_custom import get_closest_start_idx
from nuplan_scripts.utils.constants import NUPLAN_DATA_ROOT, NUPLAN_DB_FILES
from nuplan_scripts.utils.config import FrmaeCentralConfig

second_before_token = 4
second_after_token = 8
sample_interval = 1

TRAJECTORY_LENGTH_THRES = 50    # when lower then thres, generate a fixed size road block
TRAJECTORY_OFFSET_RANGE = 20
MIN_ROADBLOCK_SIZE = 50


class GenerateConfigFromToken:

    class PseudoDataset(Dataset):
        def __init__(self, log_name2tokens: dict, data_root: str, split: str):
            self.log_names = list(log_name2tokens.keys())
            self.tokens = log_name2tokens
            self.data_root = data_root
            self.split = split

        def __getitem__(self, idx):
            central_log = self.log_names[idx]
            central_tokens = self.tokens[central_log]

            db_filename = os.path.join(NUPLAN_DB_FILES, central_log + '.db')
            log_db = NuPlanDB(NUPLAN_DATA_ROOT, db_filename, None, verbose=True)
            log = log_db.log
            lidar_pcs = log.lidar_pcs
            start_idx = get_closest_start_idx(log, lidar_pcs)
            lidar_pcs = lidar_pcs[start_idx::2 * sample_interval]
            lidar_pc_tokens = [lidar_pc.token for lidar_pc in lidar_pcs]

            token_configs = []
            token_config_meta_datas = []

            for token in central_tokens:
                # check if the token are in the range of the neighborhood
                duplicate = False
                for idx, meta_data in enumerate(token_config_meta_datas):
                    if token in meta_data['in_range_tokens']:
                        duplicate = True
                        token_configs[idx].central_tokens.append(token)
                        break

                if duplicate:
                    continue

                # get the in range lidar pcs
                selected_idx = lidar_pc_tokens.index(token)
                start_idx = selected_idx - second_before_token * 20 // (2*sample_interval)
                end_idx = selected_idx + second_after_token * 20 // (2*sample_interval)
                start_idx = max(0, start_idx)
                end_idx = min(len(lidar_pcs), end_idx)
                in_range_lidar_pcs = lidar_pcs[start_idx:end_idx]

                trajectory = [lidar_pc.ego_pose.translation_np for lidar_pc in in_range_lidar_pcs]
                trajectory = np.array(trajectory)[..., :2]
                trajectory_polyline = shapely.LineString(trajectory)

                road_block = shapely.GeometryCollection([
                    trajectory_polyline, 
                    trajectory_polyline.offset_curve(TRAJECTORY_OFFSET_RANGE), 
                    trajectory_polyline.offset_curve(-TRAJECTORY_OFFSET_RANGE)
                ]).bounds
                road_block = np.array(road_block, dtype=int)

                # Get the road_block.
                # If the trajectory is too short, use a fixed size road block.
                if trajectory_polyline.length < TRAJECTORY_LENGTH_THRES:
                    extended_lidar_pcs = lidar_pcs[start_idx:]
                    extended_trajectory = [lidar_pc.ego_pose.translation_np for lidar_pc in extended_lidar_pcs]
                    extended_trajectory = np.array(extended_trajectory)[..., :2]
                    extended_traj_polyline = shapely.LineString(extended_trajectory)
                    extended_traj_polyline = [extended_traj_polyline.interpolate(dist) for dist in np.linspace(0, TRAJECTORY_LENGTH_THRES, 10, endpoint=True)]
                    extended_traj_polyline = shapely.LineString(extended_traj_polyline)
                    new_road_block = shapely.GeometryCollection([
                        extended_traj_polyline, 
                        extended_traj_polyline.offset_curve(TRAJECTORY_OFFSET_RANGE), 
                        extended_traj_polyline.offset_curve(-TRAJECTORY_OFFSET_RANGE)
                    ]).bounds
                    new_road_block = np.array(new_road_block, dtype=int)
                    road_block = np.concatenate([
                        np.min([road_block[:2], new_road_block[:2]], axis=0), 
                        np.max([road_block[2:], new_road_block[2:]], axis=0)], axis=0)

                token_config = FrmaeCentralConfig(
                    road_block_name=f"{central_log}-{token}",
                    road_block=tuple(road_block.tolist()),
                    data_root=self.data_root,
                    city=log.map_version,
                    interval=sample_interval,
                    expand_buffer=0,
                    reconstruct_buffer=0,
                    selected_videos=(),
                    split=self.split,
                    collect_raw=False,
                    central_log=central_log,
                    central_tokens=[token],
                    multi_traversal_mode='off'
                )
                token_configs.append(token_config)
                token_config_meta_datas.append({
                    'in_range_tokens': [lidar_pc.token for lidar_pc in in_range_lidar_pcs],
                    'road_block': road_block
                })

            log_db.session.close()
            return token_configs

        def __len__(self):
            return len(self.log_names)

    @staticmethod
    def run(
        log_name2tokens, 
        data_root,
        split,
        num_workers):

        dataset = GenerateConfigFromToken.PseudoDataset(log_name2tokens, data_root, split)
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: x, pin_memory=False, drop_last=False)

        token_configs = []
        for batch in tqdm(dataloader, desc='Processing central logs', ncols=120):
            token_configs.extend(batch[0])

        return token_configs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--navsim_filter', type=str, required=True, help='Path to the navsim filter yaml')
    parser.add_argument('--data_root', type=str, required=True, help='Path to the MTGS data root')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory to save the generated configs')
    parser.add_argument('--split', type=str, default='trainval', choices=['trainval', 'test', 'private_test', 'all'])
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    navsim_filter = yaml.load(open(args.navsim_filter, 'r'), Loader=yaml.FullLoader)

    with jsonlines.open(os.path.join(args.data_root, 'nuplan_log_infos.jsonl'), 'r') as reader:
        log_name2lidar_pc_token = {item['log_name']: item for item in reader}

    selected_tokens = set(navsim_filter['tokens'])
    token_mapping = {}
    for log_name in log_name2lidar_pc_token:
        for tokens in log_name2lidar_pc_token[log_name]['lidar_pc_tokens']:
            if tokens in selected_tokens:
                token_mapping[tokens] = log_name

    # tokens in token_mapping is in time order.
    selected_log_name2tokens = {}
    for token in token_mapping:
        log_name = token_mapping[token]
        if log_name not in selected_log_name2tokens:
            selected_log_name2tokens[log_name] = []
        selected_log_name2tokens[log_name].append(token)

    token_configs = GenerateConfigFromToken.run(selected_log_name2tokens, args.data_root, args.split, args.num_workers)

    os.makedirs(args.output_dir, exist_ok=True)
    for token_config in token_configs:
        token_config.save_config(os.path.join(args.output_dir, f"{token_config.road_block_name}.yaml"))
