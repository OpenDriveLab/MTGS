#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import math
from dataclasses import dataclass, field
from typing import Optional, Type
from typing_extensions import Literal

import torch

from nerfstudio.utils.rich_utils import CONSOLE
from .utils import num_sh_bases
from .multi_color_gaussian_splatting import MultiColorGaussianSplattingModel, MultiColorGaussianSplattingModelConfig

@dataclass
class SkyboxGaussianSplattingModelConfig(MultiColorGaussianSplattingModelConfig):
    """Gaussian Splatting Model Config"""

    _target: Type = field(default_factory=lambda: SkyboxGaussianSplattingModel)

    skybox_radius: float = 2000.0
    num_sky_gaussians: int = 100000
    """
    whether to use a large skybox, i.e. a large sphere at about 10x of the diameter of the scene extent with
    `num_sky_gaussians` gaussians sampled on or inside the hemisphere, during training.
    """
    skybox_type: Literal["spheric", "volumetric", "homocentric"] = "spheric"
    skybox_scale_factor: float = 40.0

    mono_sky: bool = True

class SkyboxGaussianSplattingModel(MultiColorGaussianSplattingModel):

    config: SkyboxGaussianSplattingModelConfig
    def __init__(self, config: MultiColorGaussianSplattingModelConfig, **kwargs):
        if config.mono_sky:
            config.multi_feature_rest = False
        super().__init__(config, **kwargs)

    def populate_modules(self, max_distance: float = None, **kwargs):
        self.scene_extent = max_distance

        num_points_sky = self.config.num_sky_gaussians
        dim_sh = num_sh_bases(self.config.control.sh_degree)
        if num_points_sky == 0:
            self._skip_current_model(dim_sh)
            return

        skybox_radius = self.config.skybox_radius
        # uniformly sample points on a sphere
        if skybox_radius < max_distance * 10:
            skybox_radius = max(skybox_radius, max_distance * 2)
            CONSOLE.log(
                f"[WARNING] Skybox radius is smaller than 10x the scene extent.\n"
                f"\tSkybox radius: {skybox_radius}\n"
                f"\tScene extent: {max_distance}"
            )

        # uniformly sample points on a sphere
        if self.config.skybox_type == "spheric":
            radii = torch.ones(num_points_sky) * skybox_radius
            # sample points on a sphere
        elif self.config.skybox_type == "volumetric":
            radii = torch.rand(num_points_sky) * skybox_radius
            # sample points in a hemisphere
        else:
            radii = max_distance + torch.rand(num_points_sky) * (skybox_radius - max_distance)

        theta = torch.rand(num_points_sky) * 2 * math.pi
        phi = torch.rand(num_points_sky) * math.pi / 4 + math.pi / 4  # Sample phi from [π/4, π/2]
        skybox_means = torch.stack(
            [
                radii * torch.sin(phi) * torch.cos(theta),
                radii * torch.sin(phi) * torch.sin(theta),
                radii * torch.cos(phi),
            ],
            dim=-1,
        )

        if self.config.mono_sky:
            kwargs["num_traversals"] = None 
        else:
            assert kwargs.get('num_traversals', None) is not None, 'num_traversals must be provided for multi-color skybox'
        
        # assume skybox is white
        skybox_colors = torch.ones((num_points_sky, 3)) * 255
        points_3d={
                "xyz": skybox_means,
                "rgb": skybox_colors
            }
        if self.config.mono_sky:       # use VanillaGaussianSplattingModel init.
            super(MultiColorGaussianSplattingModel, self).populate_modules(points_3d=points_3d, **kwargs)
        else:
            super().populate_modules(points_3d, **kwargs)

    @property
    def features_adapters(self):
        if self.config.mono_sky:
            return None
        return self.gauss_params['features_adapters']

    def get_pertravel_features(self, traversal_id):
        if self.config.mono_sky:
            return self.features_dc, self.features_rest
        else:
            return super().get_pertravel_features(traversal_id)

    @property
    def portable_config(self):
        portable_config = super().portable_config
        portable_config.update({
            "skybox_radius": self.config.skybox_radius,
            "skybox_type": self.config.skybox_type,
            "skybox_scale_factor": self.config.skybox_scale_factor,
        })
        return portable_config

    def get_rgbs(self, camera_to_worlds, traversal_id=None, **kwargs):
        if self.config.mono_sky:
            return super(MultiColorGaussianSplattingModel, self).get_rgbs(camera_to_worlds)
        return super().get_rgbs(camera_to_worlds, traversal_id)

    def split_properties(self, split_mask, samps):
        if self.config.mono_sky:
            return super(MultiColorGaussianSplattingModel, self).split_properties(split_mask, samps)
        return super().split_properties(split_mask, samps)

    def cull_gaussians(self, extra_cull_mask: Optional[torch.Tensor] = None):
        """
        This function deletes gaussians with under a certain opacity threshold
        extra_cull_mask: a mask indicates extra gaussians to cull besides existing culling criterion
        """
        n_bef = self.num_points

        # cull transparent ones
        culls = (torch.sigmoid(self.opacities) < self.ctrl_config.cull_alpha_thresh).squeeze()


        if extra_cull_mask is not None:
            culls = culls | extra_cull_mask
            n_bef -= torch.sum(extra_cull_mask).item()

        if self.step > self.ctrl_config.refine_every * self.ctrl_config.reset_alpha_every:
            # cull huge ones
            skybox_mask = self.means.norm(dim=-1) > (self.config.skybox_radius / 10.)
            cull_scale_thresh = torch.where(skybox_mask, self.config.skybox_scale_factor, 1.) * self.ctrl_config.cull_scale_thresh
            toobigs = (torch.exp(self.scales).max(dim=-1).values > cull_scale_thresh).squeeze()

            if self.step < self.ctrl_config.stop_screen_size_at:
                # cull big screen space
                assert self.max_2Dsize is not None
                cull_screen_size = self.ctrl_config.cull_screen_size
                toobigs_vs = (self.max_2Dsize > cull_screen_size).squeeze()
                toobigs = toobigs | toobigs_vs

            culls = culls | toobigs

        for name, param in self.gauss_params.items():
            self.gauss_params[name] = torch.nn.Parameter(param[~culls])

        return culls
