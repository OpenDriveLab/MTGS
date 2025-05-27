#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
from dataclasses import dataclass, field
from typing import Optional, Type

import torch
import torch.nn as nn

from jaxtyping import Float, Int
from torch import Tensor, nn

from nerfstudio.configs.base_config import InstantiateConfig

@dataclass
class GSAppearanceModelConfig(InstantiateConfig):
    """Configuration for Gaussian Splatting AppearanceModel"""
    _target: Type = field(default_factory=lambda: GSAppearanceModel)
    appearance_embedding_dim: int = 32

class GSAppearanceModel(nn.Module):
    """Appearance model for Gaussian Splatting. This is a dummy class that does nothing."""
    config: GSAppearanceModelConfig

    def __init__(
        self,
        config: GSAppearanceModelConfig,
        num_cameras: int
    ) -> None:
        super().__init__()
        self.config = config

    def get_param_groups(self, param_groups: dict) -> None:
        """Get camera optimizer parameters"""
        pass
 
    def forward(
        self,
        image: torch.Tensor,
        indices: Int[Tensor, "camera_indices"]
    ):
        return image

@dataclass
class LearnableExposureRGBModelConfig(GSAppearanceModelConfig):
    """Configuration for VastAppearanceModel"""
    _target: Type = field(default_factory=lambda: LearnableExposureRGBModel)

class LearnableExposureRGBModel(GSAppearanceModel):
    """Decoupled Appearance Modeling in VastGaussian: Vast 3D Gaussians for Large Scene Reconstruction"""
    config: LearnableExposureRGBModelConfig

    def __init__(
        self,
        config: LearnableExposureRGBModelConfig,
        num_cameras: int,
        # v_adjust: torch.Tensor
    ) -> None:

        super().__init__(config, num_cameras)

        self.exposure_factor = nn.Parameter(
            torch.eye(3, 4, dtype=torch.float32, requires_grad=True)[None].repeat(num_cameras, 1, 1)
        )

    def get_param_groups(self, param_groups: dict) -> None:
        """Get camera optimizer parameters"""
        opt_params = list(self.parameters())
        param_groups["appearance"] = opt_params

    def forward(self, 
        image: Float[Tensor, "image"],
        indices: Optional[Int[Tensor, "camera_indices"]] = None,
    ):
        # image shape: W x H x 3
        if indices is None:
            return image

        exposure_factor = self.exposure_factor[indices[0]]

        # exposure_factor = exposure_factor * self.v_adjust.data[indices]
        optimized_image = image.matmul(exposure_factor[:3, :3]) + exposure_factor[None, None, :3, 3]
        optimized_image = torch.clamp(optimized_image, 0, 1)

        return optimized_image
