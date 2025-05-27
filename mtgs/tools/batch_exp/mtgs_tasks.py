#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
main_mt = [
    {'config': 'nuplan_scripts/configs/mtgs_exp/road_block-365530_143960_365630_144060.yml', 'train_traversal': (1, 2), 'eval_traversal': (1, 2, 6)},
    {'config': 'nuplan_scripts/configs/mtgs_exp/road_block-365000_144000_365100_144080.yml', 'train_traversal': (4, 5), 'eval_traversal': (3, 4, 5)},
    {'config': 'nuplan_scripts/configs/mtgs_exp/road_block-587400_4475700_587480_4475800.yml', 'train_traversal': (0, 23, 31), 'eval_traversal': (0, 2, 23, 31)},
    {'config': 'nuplan_scripts/configs/mtgs_exp/road_block-331220_4690660_331190_4690710.yml', 'train_traversal': (0, 1, 7), 'eval_traversal': (0, 1, 6, 7)},
    {'config': 'nuplan_scripts/configs/mtgs_exp/road_block-587640_4475600_587710_4475660.yml', 'train_traversal': (23, 30, 41), 'eval_traversal': (2, 23, 30, 41)},
    {'config': 'nuplan_scripts/configs/mtgs_exp/road_block-587860_4475510_587910_4475570.yml', 'train_traversal': (4, 12, 35), 'eval_traversal': (4, 12, 35, 33)},
]

main_st = [
    {'config': 'nuplan_scripts/configs/mtgs_exp/road_block-365530_143960_365630_144060.yml', 'train_traversal': (1,), 'eval_traversal': (1, 6)},
    {'config': 'nuplan_scripts/configs/mtgs_exp/road_block-365000_144000_365100_144080.yml', 'train_traversal': (4,), 'eval_traversal': (4, 3)},
    {'config': 'nuplan_scripts/configs/mtgs_exp/road_block-587400_4475700_587480_4475800.yml', 'train_traversal': (0,), 'eval_traversal': (0, 31)},
    {'config': 'nuplan_scripts/configs/mtgs_exp/road_block-331220_4690660_331190_4690710.yml', 'train_traversal': (0,), 'eval_traversal': (0, 6)},
    {'config': 'nuplan_scripts/configs/mtgs_exp/road_block-587640_4475600_587710_4475660.yml', 'train_traversal': (41,), 'eval_traversal': (41, 2)},
    {'config': 'nuplan_scripts/configs/mtgs_exp/road_block-587860_4475510_587910_4475570.yml', 'train_traversal': (4,), 'eval_traversal': (4, 33)},
]

ablation = [
    {'config': 'nuplan_scripts/configs/mtgs_exp/road_block-587400_4475700_587480_4475800.yml', 'train_traversal': (0, 23, 31), 'eval_traversal': (0, 2, 23, 31)},
    {'config': 'nuplan_scripts/configs/mtgs_exp/road_block-331220_4690660_331190_4690710.yml', 'train_traversal': (0, 1, 7), 'eval_traversal': (0, 1, 6, 7)},
    {'config': 'nuplan_scripts/configs/mtgs_exp/road_block-587640_4475600_587710_4475660.yml', 'train_traversal': (23, 30, 41), 'eval_traversal': (2, 23, 30, 41)},
    {'config': 'nuplan_scripts/configs/mtgs_exp/road_block-587860_4475510_587910_4475570.yml', 'train_traversal': (4, 12, 35), 'eval_traversal': (4, 12, 35, 33)},
]

mt_ablation_1_trv = [
    {'config': 'nuplan_scripts/configs/mtgs_exp/road_block-331220_4690660_331190_4690710.yml', 'train_traversal': (0,), 'eval_traversal': (0, 6)},
    {'config': 'nuplan_scripts/configs/mtgs_exp/road_block-587640_4475600_587710_4475660.yml', 'train_traversal': (41,), 'eval_traversal': (2, 41)},
    {'config': 'nuplan_scripts/configs/mtgs_exp/road_block-587860_4475510_587910_4475570.yml', 'train_traversal': (4,), 'eval_traversal': (4, 33)},
]

mt_ablation_2_trv = [
    {'config': 'nuplan_scripts/configs/mtgs_exp/road_block-331220_4690660_331190_4690710.yml', 'train_traversal': (0, 1), 'eval_traversal': (0, 6)},
    {'config': 'nuplan_scripts/configs/mtgs_exp/road_block-587640_4475600_587710_4475660.yml', 'train_traversal': (41, 30), 'eval_traversal': (2, 41)},
    {'config': 'nuplan_scripts/configs/mtgs_exp/road_block-587860_4475510_587910_4475570.yml', 'train_traversal': (4, 35), 'eval_traversal': (4, 33)},
]

mt_ablation_3_trv = [
    {'config': 'nuplan_scripts/configs/mtgs_exp/road_block-331220_4690660_331190_4690710.yml', 'train_traversal': (0, 1, 7), 'eval_traversal': (0, 6)},
    {'config': 'nuplan_scripts/configs/mtgs_exp/road_block-587640_4475600_587710_4475660.yml', 'train_traversal': (41, 30, 23), 'eval_traversal': (2, 41)},
    {'config': 'nuplan_scripts/configs/mtgs_exp/road_block-587860_4475510_587910_4475570.yml', 'train_traversal': (4, 35, 12), 'eval_traversal': (4, 33)},
]

mt_ablation_4_trv = [
    {'config': 'nuplan_scripts/configs/mtgs_exp/road_block-331220_4690660_331190_4690710.yml', 'train_traversal': (0, 1, 7, 2), 'eval_traversal': (0, 6)},
    {'config': 'nuplan_scripts/configs/mtgs_exp/road_block-587640_4475600_587710_4475660.yml', 'train_traversal': (41, 30, 23, 3), 'eval_traversal': (2, 41)},
    {'config': 'nuplan_scripts/configs/mtgs_exp/road_block-587860_4475510_587910_4475570.yml', 'train_traversal': (4, 35, 12, 16), 'eval_traversal': (4, 33)},
]

mt_ablation_5_trv = [
    {'config': 'nuplan_scripts/configs/mtgs_exp/road_block-331220_4690660_331190_4690710.yml', 'train_traversal': (0, 1, 7, 2, 8), 'eval_traversal': (0, 6)},
    {'config': 'nuplan_scripts/configs/mtgs_exp/road_block-587640_4475600_587710_4475660.yml', 'train_traversal': (41, 30, 23, 3, 0), 'eval_traversal': (2, 41)},
    {'config': 'nuplan_scripts/configs/mtgs_exp/road_block-587860_4475510_587910_4475570.yml', 'train_traversal': (4, 35, 12, 16, 1), 'eval_traversal': (4, 33)},
]

tasks_registry = {
    'main_mt': main_mt,
    'main_st': main_st,
    'ablation': ablation,
    'mt_ablation_1_trv': mt_ablation_1_trv,
    'mt_ablation_2_trv': mt_ablation_2_trv,
    'mt_ablation_3_trv': mt_ablation_3_trv,
    'mt_ablation_4_trv': mt_ablation_4_trv,
    'mt_ablation_5_trv': mt_ablation_5_trv,
}
