#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
from dataclasses import dataclass, field
from typing import Optional, Type, Literal

import torch

try:
    from gsplat.cuda._wrapper import spherical_harmonics
except ImportError:
    print("Please install gsplat>=1.0.0")

from .vanilla_gaussian_splatting import VanillaGaussianSplattingModel, VanillaGaussianSplattingModelConfig

@dataclass
class MultiColorGaussianSplattingModelConfig(VanillaGaussianSplattingModelConfig):
    """Gaussian Splatting Model Config"""

    _target: Type = field(default_factory=lambda: MultiColorGaussianSplattingModel)
    multi_feature_rest: bool = False
    eval_mode: Optional[Literal["null", "mean", "first_travel", "nearest_travel"]] = "nearest_travel"

class MultiColorGaussianSplattingModel(VanillaGaussianSplattingModel):
    """Gaussian Splatting model

    Args:
        config: Gaussian Splatting configuration to instantiate model
    """

    config: MultiColorGaussianSplattingModelConfig

    def __init__(
        self,
        config: MultiColorGaussianSplattingModelConfig,
        travel_ids: dict = None,
        nearest_train_travel_of_eval: dict = None,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self.travel_mapping = {
            travel_id: i for i, travel_id in enumerate(travel_ids)
        }
        self.nearest_train_travel_of_eval = nearest_train_travel_of_eval

    def populate_modules(self, points_3d=None, num_traversals=None, **kwargs):
        super().populate_modules(points_3d, **kwargs)
        if num_traversals is None:
            return

        if self.config.multi_feature_rest:
            features_rest = self.gauss_params.pop("features_rest")
            features_rest = torch.zeros((points_3d['rgb'].shape[0], num_traversals, features_rest.shape[-2], 3), dtype=torch.float32)
            features_rest = torch.nn.Parameter(features_rest)
            self.gauss_params = torch.nn.ParameterDict(
                {
                    **self.gauss_params,
                    "features_rest": features_rest
                }
            )
        
        features_adapters = torch.zeros((points_3d['rgb'].shape[0], num_traversals, 3), dtype=torch.float32)
        features_adapters = torch.nn.Parameter(features_adapters)
        self.gauss_params = torch.nn.ParameterDict(
                {
                    **self.gauss_params,
                    "features_adapters": features_adapters
                }
            )

    @property
    def features_adapters(self):
        return self.gauss_params['features_adapters']

    def get_pertravel_features(self, traversal_id):
        if traversal_id is not None:
            features_dc = self.features_dc + self.features_adapters[:, traversal_id, :]
            features_rest = self.features_rest[:, traversal_id, :, :] if self.config.multi_feature_rest else self.features_rest
        elif self.config.eval_mode == "mean":
            features_dc = self.features_dc + self.features_adapters.mean(dim=1)
            features_rest = self.features_rest.mean(dim=1) if self.config.multi_feature_rest else self.features_rest
        else:
            features_dc = self.features_dc
            features_rest = torch.zeros_like(self.features_rest[:, 0, :, :]) if self.config.multi_feature_rest else self.features_rest
        return features_dc, features_rest

    def get_rgbs(self, camera_to_worlds, traversal_id=None):
        features_dc, features_rest = self.get_pertravel_features(traversal_id)
        colors = torch.cat((features_dc[:, None, :], features_rest), dim=1)
        if self.sh_degree > 0:
            viewdirs = self.means.detach() - camera_to_worlds[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.ctrl_config.sh_degree_interval, self.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors)
            rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
        else:
            rgbs = torch.sigmoid(colors[:, 0, :])

        return rgbs

    def get_gaussians(self, camera_to_worlds, travel_id, return_features=False, return_v=False, **kwargs):
        if travel_id in ["null", "mean"]:
            self.config.eval_mode = travel_id

        traversal_embedding_id = self.travel_mapping.get(travel_id, None)
        if traversal_embedding_id is None and travel_id is not None:
            if self.config.eval_mode == "first_travel":
                traversal_embedding_id = 0
            elif self.config.eval_mode == "nearest_travel":
                nearest_travel_id = self.nearest_train_travel_of_eval.get(travel_id, None)
                traversal_embedding_id = self.travel_mapping.get(nearest_travel_id, None)
            elif self.config.eval_mode in ["null", "mean"]:
                traversal_embedding_id = None
            else:
                raise ValueError(f"Invalid eval_mode: {self.config.eval_mode}")

        return_dict = {
            "means": self.get_means(),
            "scales": self.get_scales(),
            "quats": self.get_quats(),
            "opacities": self.get_opacity(),
        }
        if return_features:
            features_dc, features_rest = self.get_pertravel_features(traversal_embedding_id)
            return_dict["features_dc"] = features_dc
            return_dict["features_rest"] = features_rest
        else:
            return_dict["rgbs"] = self.get_rgbs(camera_to_worlds, traversal_embedding_id)

        if return_v:
            return_dict["velocities"] = torch.zeros_like(return_dict["means"])
        return return_dict

    def split_properties(self, split_mask, samps):
        new_features_rest = self.features_rest[split_mask].repeat(samps, 1, 1, 1) if self.config.multi_feature_rest \
            else self.features_rest[split_mask].repeat(samps, 1, 1)
        return {
            "features_dc": self.split_features_dc(split_mask, samps),
            "features_rest": new_features_rest,
            "opacities": self.opacities[split_mask].repeat(samps, 1),
            "features_adapters": self.features_adapters[split_mask].repeat(samps, 1, 1)
        }

    def load_state_dict(self, dict, **kwargs):
        if "gauss_params.features_adapters" in dict and dict['gauss_params.features_adapters'].shape[1] != len(self.travel_mapping):
            if self.config.eval_mode == "mean":
                dict['gauss_params.features_adapters'] = dict['gauss_params.features_adapters'].mean(dim=1, keepdim=True).expand(-1, len(self.travel_mapping), -1)
                if self.config.multi_feature_rest:
                    dict['gauss_params.features_rest'] = dict['gauss_params.features_rest'].mean(dim=1, keepdim=True).expand(-1, len(self.travel_mapping), -1, -1)
            else:
                dict.pop('gauss_params.features_adapters')
                if self.config.multi_feature_rest:
                    dict.pop('gauss_params.features_rest')

        if 'strict' in kwargs:
            kwargs.pop('strict')

        super().load_state_dict(dict, strict=False, **kwargs)
