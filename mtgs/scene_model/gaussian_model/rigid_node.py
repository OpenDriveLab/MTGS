#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union, Any, Literal
import warnings

import numpy as np
import torch

from torch import Tensor
from torch.nn import Parameter

try:
    from gsplat.cuda._wrapper import spherical_harmonics
except ImportError:
    print("Please install gsplat>=1.0.0")

from nerfstudio.utils.rich_utils import CONSOLE

from .utils import quat_mult, matrix_to_quaternion, quat_to_rotmat, interpolate_quats, IDFT
from .vanilla_gaussian_splatting import VanillaGaussianSplattingModel, VanillaGaussianSplattingModelConfig

@dataclass
class RigidSubModelConfig(VanillaGaussianSplattingModelConfig):
    """Gaussian Splatting Model Config"""

    _target: Type = field(default_factory=lambda: RigidSubModel)
    fourier_features_dim: Optional[int] = None
    """dimension of the Fourier features used for the diffuse component of the SH. set the value <=1 or set it to `None` to disable it."""
    fourier_features_scale: Optional[int] = 1
    """scale of the Fourier features used for the diffuse component of the SH."""
    is_static: Optional[bool] = False
    """whether to set the object to static, i.e., not optimizing the object pose."""
    fourier_in_space: Optional[Literal['spatial', 'temporal']] = 'temporal'
    """whether to use the Fourier features in spatial or temporal domain."""

class RigidSubModel(VanillaGaussianSplattingModel):

    config: RigidSubModelConfig

    def __init__(self, config: RigidSubModelConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.instance_info = kwargs.get("instance_info", None)

        # Caching
        self.last_step = None
        self.cached_gaussians = None
        self.last_frame_idx = None

    def populate_modules(self, instance_dict, data_frame_dict, **kwargs):
        """
        instance_dict: {
            "class_name": str,
            "token": str,
            "pts": torch.Tensor[float], (N, 3)
            "colors": torch.Tensor[float], (N, 3)
            "quats": torch.Tensor[float], (num_frames_cur_travel, 4)
            "trans": torch.Tensor[float], (num_frames_cur_travel, 3)
            "size": torch.Tensor[float], (3, )
            "in_frame_indices" : torch.Tensor[int]
            "in_frame_mask" : torch.Tensor[bool], (num_frames_cur_travel, )
            "num_frames_cur_travel": int
            "travel_id": int
        }
        data_frame_dict: {
            travel_id: {
                raw_timestamps: torch.Tensor[float], (num_frames_cur_travel, )
                frame_timestamps: torch.Tensor[float], (num_frames_cur_travel, )
                min_ts: float
                max_ts: float
            } 
        }
        """
        points_dict = dict(
            xyz=instance_dict["pts"],
            rgb=instance_dict["colors"],
        )
        if (self.config.fourier_features_dim is not None) and (self.config.fourier_features_dim <= 1):
            self.config.fourier_features_dim = None
        super().populate_modules(points_3d=points_dict, features_dc_dim=self.config.fourier_features_dim)

        self.instance_size = instance_dict["size"]
        self.in_frame_mask = instance_dict["in_frame_mask"]
        self.travel_id = instance_dict["travel_id"]
        self.dataframe_dict = data_frame_dict[self.travel_id]
        self.is_static = instance_dict.get("is_static", False)

        instance_quats = instance_dict["quats"]
        instance_trans = instance_dict["trans"]

        # set the pose of the object on the top of the sky, to make it invisible
        instance_quats[~self.in_frame_mask, 0] = 1
        instance_trans[~self.in_frame_mask, -1] = 100000
        self.num_frames = instance_dict["num_frames_cur_travel"]

        # pose refinement
        if self.is_static:
            # the global pose of the object is fixed, just need to optimize the one and only global pose
            self.instance_quats = Parameter(instance_quats[self.in_frame_mask].mean(dim=0))  # (4)
            self.instance_trans = Parameter(instance_trans[self.in_frame_mask].mean(dim=0))  # (3)
            self.in_frame_mask = torch.ones_like(self.in_frame_mask, dtype=torch.bool)
        else:
            self.instance_quats = Parameter(instance_quats)  # (num_frame, 4)
            self.instance_trans = Parameter(instance_trans)  # (num_frame, 3)

        if (self.config.fourier_in_space == 'spatial') and (self.config.fourier_features_scale != 1):
            print("[WARNING] For spatial Fourier features, set the scale to 1. Automatically correcting...")
            self.config.fourier_features_scale = 1

    @property
    def portable_config(self):
        portable_config = super().portable_config
        portable_config.update({
            "fourier_features_dim": self.config.fourier_features_dim,
            "fourier_features_scale": self.config.fourier_features_scale,
            "fourier_in_space": self.config.fourier_in_space,
        })
        if not self.is_static:
            portable_config.update({
                "log_timestamps": self.dataframe_dict["raw_timestamps"],
            })
        return portable_config

    def get_object_pose(self, frame_idx, timestamp):
        """
            (quat, trans) for the current frame
            quat: (4, ), trans: (3, )
            The timestamp is used for interpolation between frames.
        """
        # if given frame_idx, return the pose at the frame_idx rather than interpolating
        # if timestamp is None:
        if self.is_static:
            return self.instance_quats, self.instance_trans
        
        if frame_idx is not None:
            if frame_idx >= self.num_frames or not self.in_frame_mask[frame_idx]:
                return None, None

            quat_cur_frame = self.instance_quats[frame_idx] / self.instance_quats[frame_idx].norm(dim=-1, keepdim=True)
            trans_cur_frame = self.instance_trans[frame_idx]
            return quat_cur_frame, trans_cur_frame
        else:
            frame_timestamps = self.dataframe_dict["frame_timestamps"].to(self.device)  # (num_frames, )
            # Find the two adjacent frames for interpolation
            diffs = timestamp - frame_timestamps
            prev_frame = torch.argmin(torch.where(diffs >= 0, diffs, float('inf')))
            next_frame = torch.argmin(torch.where(diffs <= 0, -diffs, float('inf')))

            if not self.in_frame_mask[next_frame] or not self.in_frame_mask[prev_frame]:
                return None, None

            if next_frame == prev_frame:
                # Timestamp exactly matches a frame, no interpolation needed
                return self.instance_quats[next_frame], self.instance_trans[next_frame]

            # Calculate interpolation factor
            t = (timestamp - frame_timestamps[prev_frame]) / (frame_timestamps[next_frame] - frame_timestamps[prev_frame])

            # Interpolate quaternions (using slerp) and translations
            quat_interp = interpolate_quats(self.instance_quats[prev_frame], self.instance_quats[next_frame], t).squeeze()
            trans_interp = torch.lerp(self.instance_trans[prev_frame], self.instance_trans[next_frame], t)

            return quat_interp, trans_interp

    def get_velocity(self, frame_idx, timestamp, global_current_means=None):
        if self.is_static:
            return torch.zeros_like(global_current_means)
        if frame_idx is not None:
            if (frame_idx == self.num_frames - 1) or ((self.in_frame_mask[frame_idx]) and not (self.in_frame_mask[frame_idx + 1])):
                frame_idx -= 1
                next_position = global_current_means
                global_current_means = None
            else:
                quat_next_frame, trans_next_frame = self.get_object_pose(frame_idx+1, None)
                next_position = self.get_means(quat_next_frame, trans_next_frame)

            if global_current_means is None:
                quat_cur_frame, trans_cur_frame = self.get_object_pose(frame_idx, None)
                current_position = self.get_means(quat_cur_frame, trans_cur_frame)
            else:
                current_position = global_current_means

            time_interval = (self.dataframe_dict["raw_timestamps"][frame_idx+1] - self.dataframe_dict["raw_timestamps"][frame_idx]) * 1e-6
            return (next_position - current_position) / time_interval
        else:
            frame_timestamps = self.dataframe_dict["frame_timestamps"].to(self.device)  # (num_frames, )
            # Find the two adjacent frames for interpolation
            diffs = timestamp - frame_timestamps
            prev_frame = torch.argmin(torch.where(diffs >= 0, diffs, float('inf')))
            next_frame = torch.argmin(torch.where(diffs <= 0, -diffs, float('inf')))
            assert self.in_frame_mask[prev_frame] and self.in_frame_mask[next_frame]
            if next_frame == prev_frame:
                # Timestamp exactly matches a frame, no interpolation needed
                return self.get_velocity(next_frame, None, global_current_means)

            quat_prev_frame, trans_prev_frame = self.get_object_pose(prev_frame, None)
            quat_next_frame, trans_next_frame = self.get_object_pose(next_frame, None)
            prev_position = self.get_means(quat_prev_frame, trans_prev_frame)
            next_position = self.get_means(quat_next_frame, trans_next_frame)
            time_interval = (self.dataframe_dict["raw_timestamps"][next_frame] - self.dataframe_dict["raw_timestamps"][prev_frame]) * 1e-6
            return (next_position - prev_position) / time_interval

    def get_means(self, quat_cur_frame, trans_cur_frame):
        local_means = self.gauss_params['means']
        rot_cur_frame = quat_to_rotmat(quat_cur_frame)
        global_means = local_means @ rot_cur_frame.T + trans_cur_frame
        return global_means

    def get_quats(self, quat_cur_frame, trans_cur_frame):
        local_quats = self.quats / self.quats.norm(dim=-1, keepdim=True)
        global_quats = quat_mult(quat_cur_frame, local_quats)
        return global_quats

    def get_fourier_features(self, x):
        scaled_x = x * self.config.fourier_features_scale
        input_is_normalized = (self.config.fourier_in_space == 'temporal')
        idft_base = IDFT(scaled_x, self.config.fourier_features_dim, input_is_normalized).to(self.device)
        return torch.sum(self.features_dc * idft_base[..., None], dim=1, keepdim=False)
    
    def get_true_features_dc(self, timestamp=None, cam_obj_yaw=None):
        normalized_x = timestamp if self.config.fourier_in_space == 'temporal' else cam_obj_yaw
        assert normalized_x is not None
        if self.config.fourier_features_dim is None:
            return self.features_dc
        return self.get_fourier_features(normalized_x)

    def get_cam_obj_yaw(self, camera_to_world, quat_cur_frame):
        if quat_cur_frame is None:
            return None
        obj_rot_mat = quat_to_rotmat(quat_cur_frame[None, ...])[0, ...]
        cam_yaw = torch.atan2(camera_to_world[..., 0, 0], camera_to_world[..., 0, 2])
        obj_yaw = torch.atan2(obj_rot_mat[0, 0], obj_rot_mat[0, 2])
        return cam_yaw - obj_yaw

    def get_rgbs(self, camera_to_worlds, quat_cur_frame=None, trans_cur_frame=None, timestamp=None, global_current_means=None):
        cam_obj_yaw = self.get_cam_obj_yaw(camera_to_worlds, quat_cur_frame)
        true_features_dc = self.get_true_features_dc(timestamp, cam_obj_yaw)
        colors = torch.cat((true_features_dc[:, None, :], self.features_rest), dim=1)
        
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
        return torch.sigmoid(self.gauss_params['opacities']).squeeze(-1)

    def get_gaussians(self, camera_to_worlds, travel_id=None, frame_idx=None, timestamp=None, return_features=False, return_v=False, **kwargs):
        if travel_id != self.travel_id or (frame_idx is None and timestamp is None):
            self.frame_idx = None
            return None

        self.frame_idx = frame_idx
        if frame_idx is not None:
            assert frame_idx < self.num_frames

            if self.use_cache(frame_idx):
                return_dict = self.cached_gaussians
                if not return_features:
                    return_dict['rgbs'] = self.get_rgbs(
                        camera_to_worlds, timestamp=timestamp, global_current_means=return_dict['means'])
                return return_dict

        quat_cur_frame, trans_cur_frame = self.get_object_pose(frame_idx, timestamp)
        if quat_cur_frame is None or trans_cur_frame is None:
            self.frame_idx = None
            return None

        if timestamp is None:
            timestamp = self.dataframe_dict["frame_timestamps"][frame_idx]

        global_means = self.get_means(quat_cur_frame, trans_cur_frame)
        return_dict = {
            "means": global_means,
            "scales": self.get_scales(),
            "quats": self.get_quats(quat_cur_frame, trans_cur_frame),
            "opacities": self.get_opacity(),
        }
        if return_features:
            return_dict.update({
                "features_dc": self.features_dc,
                "features_rest": self.features_rest,
            })
        else:
            return_dict['rgbs'] = self.get_rgbs(camera_to_worlds, timestamp=timestamp, global_current_means=global_means)

        if return_v:
            return_dict["velocities"] = self.get_velocity(frame_idx, timestamp, global_means)

        # Caching the gaussians for the next step
        if not self.training:
            self.last_step = self.step
            self.cached_gaussians = return_dict
            self.last_frame_idx = frame_idx

        return return_dict

    def use_cache(self, frame_idx):
        if not self.training:
            if self.last_step == self.step and (self.is_static or frame_idx == self.last_frame_idx):
                return True
        return False

    def get_gaussian_params(self, travel_id=None, frame_idx=None, timestamp=None, **kwargs):
        if travel_id != self.travel_id or (frame_idx is None and timestamp is None):
            return None

        if frame_idx is not None:
            assert frame_idx < self.num_frames

        quat_cur_frame, trans_cur_frame = self.get_object_pose(frame_idx, timestamp)
        if quat_cur_frame is None or trans_cur_frame is None:
            return None

        if timestamp is None:
            timestamp = self.dataframe_dict["frame_timestamps"][frame_idx]

        return {
            "means": self.get_means(quat_cur_frame, trans_cur_frame),
            "scales": self.scales,
            "quats": self.get_quats(quat_cur_frame, trans_cur_frame),
            "features_dc": self.get_true_features_dc(timestamp),
            "features_rest": self.features_rest,
            "opacities": self.opacities,
        }

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        return {
            f"{self.model_name}.{self.model_type}.{name}": [self.gauss_params[name]]
            for name in self.gauss_params.keys()
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = self.get_gaussian_param_groups()
        param_groups[f"{self.model_name}.{self.model_type}.ins_rotation"] = [self.instance_quats]
        param_groups[f"{self.model_name}.{self.model_type}.ins_translation"] = [self.instance_trans]
        return param_groups

    def after_train(self, step: int):
        # if the object is not in the current frame, do not update the gaussians grads.
        if self.frame_idx is None or not self.in_frame_mask[self.frame_idx]:
            return

        super().after_train(step)
    
    def refinement_after(self, optimizers, step):
        assert step == self.step

        if self.step <= self.ctrl_config.densify_from_iter:
            return
        if self.step < self.ctrl_config.stop_split_at:
            if self.xys_grad_norm is None or self.vis_counts is None or self.max_2Dsize is None:
                CONSOLE.log(f"skip refinement after for rigid object {self.model_name}")
                return

        super().refinement_after(optimizers, step)

    def load_state_dict(self, dict: dict, **kwargs):  # type: ignore
        # the object is not in the state_dict.
        if len(dict) == 0:
            return

        if dict['instance_quats'].ndim == 2 and dict['instance_quats'].shape[0] != self.num_frames:
            warnings.warn(
                f"{self.model_name} has different number of frames in the state_dict. "
                "Will not load the instance_quats and instance_trans."
            )
            dict.pop('instance_quats')
            dict.pop('instance_trans')
            kwargs['strict'] = False
        else:
            if self.is_static and dict['instance_quats'].ndim == 2:
                warnings.warn(
                    f"In pretrain weight, the {self.model_name} is not treat as a static object"
                    "we keep the original instance_quats and instance_trans"
                )
                self.is_static = False
                self.instance_quats = Parameter(dict['instance_quats'].to(self.device))
                self.instance_trans = Parameter(dict['instance_trans'].to(self.device))

        super().load_state_dict(dict, **kwargs)

    def split_features_dc(self, split_mask, samps):
        if self.config.fourier_features_dim is None:
            return self.features_dc[split_mask].repeat(samps, 1)
        return self.features_dc[split_mask].repeat(samps, 1, 1)
    
    def translate(self, translate_vector: Tensor):
        assert translate_vector.shape == (3,)
        new_means = self.means + translate_vector
        new_instance_trans = self.instance_trans - translate_vector
        self.means.data = new_means
        self.instance_trans.data = new_instance_trans

    def rotate(self, rotation: Tensor):
        # rotate by the center of the object
        if rotation.shape == (4,):
            rotation = rotation if isinstance(rotation, torch.Tensor) else torch.tensor(rotation)
            rotation_matrix = quat_to_rotmat(rotation).to(self.device).float()
        elif rotation.shape == (3, 3):
            rotation_matrix = rotation.copy() if isinstance(rotation, np.ndarray) else rotation.clone()
            rotation = matrix_to_quaternion(rotation_matrix)
        else:
            raise ValueError("Invalid rotation shape. Should be (4,), i.e., a quaternion, or (3, 3), i.e. a matrix.")
        
        new_means = self.means @ rotation_matrix.T
        new_quats = quat_mult(rotation, self.quats)
        new_instance_quats = quat_mult(-rotation, self.instance_quats)
        self.means.data = new_means
        self.quats.data = new_quats
        self.instance_quats.data = new_instance_quats
