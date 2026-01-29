#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import os
import sys
import argparse
import shutil
from pathlib import Path
os.environ["NERFSTUDIO_DATAPARSER_CONFIGS"] = "nuplan=mtgs.config.nuplan_dataparser:nuplan_dataparser"
os.environ["NERFSTUDIO_METHOD_CONFIGS"] = "mtgs_st=mtgs.config.MTGS_ST:method"

from nuplan_scripts.utils.config import load_config
from nuplan_scripts.utils.video_scene_dict_tools import VideoScene
from nuplan_scripts.utils.constants import CONSOLE

def find_portable_ckpt_path(nerfstudio_config: Path) -> Path:
    ckpt_dir = nerfstudio_config.parent / "nerfstudio_models"
    if not os.path.exists(ckpt_dir):
        CONSOLE.rule("Error", style="red")
        CONSOLE.print(f"No checkpoint directory found at {ckpt_dir}, ", justify="center")
        CONSOLE.print(
            "Please make sure the checkpoint exists, they should be generated periodically during training",
            justify="center",
        )
        sys.exit(1)
    load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(ckpt_dir))[-1]
    portable_ckpt_path = ckpt_dir / f"portable_step-{load_step:09d}.ckpt"
    return portable_ckpt_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    video_scene = VideoScene(config)

    CONSOLE.log("Exporting background reconstruction model.", style='bold magenta')
    reconstruction_dir = Path(config.data_root) / 'background_reconstruction' / config.road_block_name
    nerfstudio_config = reconstruction_dir / 'config.yml'

    dst_ckpt_path = Path(args.output_dir) / f"{config.road_block_name}/background/{config.road_block_name}.ckpt"
    dst_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(dst_ckpt_path):
        os.remove(dst_ckpt_path)

    portable_ckpt_path = find_portable_ckpt_path(nerfstudio_config)
    if portable_ckpt_path.exists():
        # copy or link the portable checkpoint to the output directory
        if os.stat(portable_ckpt_path).st_dev == os.stat(dst_ckpt_path.parent).st_dev:
            os.link(portable_ckpt_path, dst_ckpt_path)
        else:
            shutil.copy2(portable_ckpt_path, dst_ckpt_path)
    else:
        # lazy import to speed up the script
        from nerfstudio.utils.eval_utils import eval_setup
        from mtgs.utils.export_utils import extract_portable_ckpt
        nerfstudio_config, pipeline, _, _ = eval_setup(nerfstudio_config, test_mode='inference')
        extract_portable_ckpt(pipeline, dst_ckpt_path)

    CONSOLE.log("Exporting config.", style='bold magenta')
    save_config_path = os.path.join(video_scene.data_root, 'configs', os.path.basename(args.config))
    os.makedirs(os.path.dirname(save_config_path), exist_ok=True)
    if os.path.exists(save_config_path):
        os.remove(save_config_path)
    shutil.copy2(args.config, save_config_path)

    CONSOLE.log("Exporting video scene dict.", style='bold magenta')
    save_pickle_path = Path(args.output_dir) / f"{config.road_block_name}/video_scene_dict.pkl"
    if os.path.exists(save_pickle_path):
        os.remove(save_pickle_path)
    if os.stat(video_scene.pickle_path_final).st_dev == os.stat(save_pickle_path.parent).st_dev:
        os.link(video_scene.pickle_path_final, save_pickle_path)
    else:
        shutil.copy2(video_scene.pickle_path_final, save_pickle_path)
