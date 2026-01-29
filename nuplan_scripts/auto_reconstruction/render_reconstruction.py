#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import os
import argparse
from pathlib import Path
os.environ["NERFSTUDIO_DATAPARSER_CONFIGS"] = "nuplan=mtgs.config.nuplan_dataparser:nuplan_dataparser"
os.environ["NERFSTUDIO_METHOD_CONFIGS"] = "mtgs_st=mtgs.config.MTGS_ST:method"

from nuplan_scripts.utils.config import load_config
from nuplan_scripts.utils.constants import CONSOLE

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    reconstruction_dir = os.path.join(config.data_root, 'background_reconstruction', config.road_block_name)
    nerfstudio_config = Path(reconstruction_dir) / 'config.yml'

    CONSOLE.log("Rendering background reconstruction model.", style='bold magenta')
    from mtgs.tools.render import RenderInterpolated
    render = RenderInterpolated(
        load_config=nerfstudio_config,
        output_path=Path(reconstruction_dir) / 'render'
    )
    render.main()

    CONSOLE.log("Evaluating background reconstruction model.", style='bold magenta')
    from nerfstudio.scripts.eval import ComputePSNR
    ComputePSNR(
        load_config=nerfstudio_config,
        output_path=Path(reconstruction_dir) / 'eval_result.json'
    ).main()
