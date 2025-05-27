#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from mtgs.dataset.nuplan_dataparser import NuplanDataParserConfig

nuplan_dataparser = DataParserSpecification(config=NuplanDataParserConfig())
