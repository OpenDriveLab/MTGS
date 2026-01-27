#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union, Any

import torch
from torch.nn import Parameter, Module

try:
    from gsplat.cuda._wrapper import spherical_harmonics
except ImportError:
    print("Please install gsplat>=1.0.0")

from .utils import quat_mult, quat_to_rotmat

from .rigid_node import RigidSubModelConfig, RigidSubModel


def flip_spherical_harmonics(coeff, sh_degree=3):
    """
    Flip the spherical harmonics coefficients along the y-axis.

    Args:
        coeff (torch.Tensor): A tensor of shape [N, 16, 3], where N is the number of Gaussians,
                              16 is the number of spherical harmonics coefficients (up to degree l=3),
                              and 3 is the feature dimension.

    Returns:
        torch.Tensor: The flipped spherical harmonics coefficients.
    """
    # Indices corresponding to m < 0 for l up to 3
    if sh_degree == 0:
        return coeff
    elif sh_degree == 1:
        indices_m_negative = [1]
    elif sh_degree == 2:
        indices_m_negative = [1, 4, 5]
    elif sh_degree == 3:
        indices_m_negative = [1, 4, 5, 9, 10, 11]
    else:
        raise ValueError(f"Unsupported SH degree: {sh_degree}")

    # Create a flip factor tensor of ones and minus ones
    flip_factors = torch.ones(coeff.shape[1], device=coeff.device)
    flip_factors[indices_m_negative] = -1

    # Reshape flip_factors to [1, 16, 1] for broadcasting
    flip_factors = flip_factors.view(1, -1, 1)

    # Apply the flip factors to the coefficients
    flipped_coeff = coeff * flip_factors

    return flipped_coeff

@dataclass
class MirroredRigidSubModelConfig(RigidSubModelConfig):
    """Gaussian Splatting Model Config"""

    _target: Type = field(default_factory=lambda: MirroredRigidSubModel)
    mirror_static: bool = True


class MirroredRigidSubModel(RigidSubModel):

    config: MirroredRigidSubModelConfig

    def get_means(self, quat_cur_frame, trans_cur_frame):
        if self.is_static and not self.config.mirror_static:
            return super().get_means(quat_cur_frame, trans_cur_frame)

        local_means: torch.Tensor = self.gauss_params['means']
        local_means_flipped = local_means * local_means.new_tensor([1, -1, 1]).view(1, 3)
        local_means = torch.cat([local_means, local_means_flipped], dim=0)

        rot_cur_frame = quat_to_rotmat(quat_cur_frame)
        global_means = local_means @ rot_cur_frame.T + trans_cur_frame
        return global_means

    def get_quats(self, quat_cur_frame, trans_cur_frame):
        if self.is_static and not self.config.mirror_static:
            return super().get_quats(quat_cur_frame, trans_cur_frame)

        local_quats = self.quats / self.quats.norm(dim=-1, keepdim=True)
        flip_tensor = local_quats.new_tensor([1, -1, 1, -1]).view(1, 4)
        local_quats_flipped = local_quats * flip_tensor
        local_quats = torch.cat([local_quats, local_quats_flipped], dim=0)
        global_quats = quat_mult(quat_cur_frame, local_quats)

        return global_quats

    def get_scales(self):
        if self.is_static and not self.config.mirror_static:
            return super().get_scales()
        scales = torch.exp(self.scales)
        return torch.cat([scales, scales], dim=0)

    def get_rgbs(self, camera_to_worlds, quat_cur_frame=None, trans_cur_frame=None, timestamp=None, global_current_means=None):
        if self.is_static and not self.config.mirror_static:
            return super().get_rgbs(camera_to_worlds, quat_cur_frame, trans_cur_frame, timestamp, global_current_means)
        cam_obj_yaw = self.get_cam_obj_yaw(camera_to_worlds, quat_cur_frame)
        true_features_dc = self.get_true_features_dc(timestamp, cam_obj_yaw)
        colors = torch.cat((true_features_dc[:, None, :], self.features_rest), dim=1)
        colors = colors.unsqueeze(0).repeat(2, 1, 1, 1)
        colors[1, ...] = flip_spherical_harmonics(colors[1, ...], self.sh_degree)
        colors = colors.view(-1, colors.shape[-2], 3)

        if self.sh_degree > 0:
            viewdirs = self.get_means(quat_cur_frame, trans_cur_frame) if global_current_means is None else global_current_means
            viewdirs = viewdirs.detach() - camera_to_worlds[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.ctrl_config.sh_degree_interval, self.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors)
            rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
        else:
            rgbs = torch.sigmoid(colors[:, 0, :])

        return rgbs

    def get_opacity(self):
        if self.is_static and not self.config.mirror_static:
            return super().get_opacity()
        return torch.sigmoid(self.gauss_params['opacities']).squeeze(-1).repeat(2)

    def get_gaussian_params(self, travel_id=None, frame_idx=None, timestamp=None, **kwargs):
        if self.is_static and not self.config.mirror_static:
            return super().get_gaussian_params(travel_id, frame_idx, timestamp, **kwargs)
        if travel_id != self.travel_id or (frame_idx is None and timestamp is None):
            return None

        if frame_idx is not None:
            assert frame_idx < self.num_frames

        quat_cur_frame, trans_cur_frame = self.get_object_pose(frame_idx, timestamp)
        if quat_cur_frame is None or trans_cur_frame is None:
            return None

        if timestamp is None:
            timestamp = self.dataframe_dict["frame_timestamps"][frame_idx]

        true_features_dc = self.get_true_features_dc(timestamp)
        colors = torch.cat((true_features_dc[:, None, :], self.features_rest), dim=1)
        colors = colors.unsqueeze(0).repeat(2, 1, 1, 1)
        colors[1, ...] = flip_spherical_harmonics(colors[1, ...], self.sh_degree)
        colors = colors.view(-1, 16, 3)

        return {
            "means": self.get_means(quat_cur_frame, trans_cur_frame),
            "scales": self.scales.repeat(2, 1),
            "quats": self.get_quats(quat_cur_frame, trans_cur_frame),
            "features_dc": colors[:, 0, :],
            "features_rest": colors[:, 1:, :],
            "opacities": self.opacities.repeat(2, 1),
        }

    def update_statistics(self, xys_grad: torch.Tensor, radii: torch.Tensor):
        if self.is_static and not self.config.mirror_static:
            return super().update_statistics(xys_grad, radii)

        if xys_grad is None or radii is None:
            self.xys_grad = None
            self.radii = None
            return

        N = xys_grad.shape[0] // 2
        assert N == self.num_points
        xys_grad = xys_grad.view(2, N).max(dim=0).values
        radii = radii.view(2, N).max(dim=0).values

        self.xys_grad = xys_grad
        self.radii = radii
