#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type
import warnings

import torch

from torch import Tensor
from torch.nn import Parameter

try:
    from gsplat.cuda._wrapper import spherical_harmonics
except ImportError:
    print("Please install gsplat>=1.0.0")

from nerfstudio.utils.rich_utils import CONSOLE

from .utils import quat_mult, quat_to_rotmat, interpolate_quats, ConditionalDeformNetwork

from .vanilla_gaussian_splatting import VanillaGaussianSplattingModel, VanillaGaussianSplattingModelConfig

@dataclass
class DeformableSubModelConfig(VanillaGaussianSplattingModelConfig):
    _target: Type = field(default_factory=lambda: DeformableSubModel)
    embed_dim: int = 16
    use_deformgs_for_nonrigid: Optional[bool] = True
    """Whether to use deformation network for nonrigid objects"""
    use_deformgs_after: Optional[int] = 3000
    """Start using deformation network after this step"""
    stop_optimizing_canonical_xyz: Optional[bool] = True
    """Whether to stop optimizing canonical xyz after using deformation network"""


class DeformableSubModel(VanillaGaussianSplattingModel):

    config: DeformableSubModelConfig

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
        super().populate_modules(points_3d=points_dict)

        self.instance_size = instance_dict["size"]
        self.in_frame_mask = instance_dict["in_frame_mask"]
        self.travel_id = instance_dict["travel_id"]
        self.dataframe_dict = data_frame_dict[self.travel_id]

        instance_quats = instance_dict["quats"]
        instance_trans = instance_dict["trans"]

        # set the pose of the object on the top of the sky, to make it invisible
        instance_quats[~self.in_frame_mask, 0] = 1
        instance_trans[~self.in_frame_mask, -1] = 100000
        self.num_frames = instance_dict["num_frames_cur_travel"]

        # pose refinement
        self.instance_quats = Parameter(instance_quats)  # (num_frame, 4)
        self.instance_trans = Parameter(instance_trans)  # (num_frame, 3)

        init_embedding = torch.rand(1, self.config.embed_dim, device=self.device)
        self.instances_embedding = Parameter(init_embedding)
        self.deform_network = ConditionalDeformNetwork(
            input_ch=3, embed_dim=self.config.embed_dim
        ).to(self.device)

    def get_means(self, quat_cur_frame, trans_cur_frame, delta_xyz=None):
        local_means = self.gauss_params['means']
        if delta_xyz is not None:
            if self.config.stop_optimizing_canonical_xyz:
                local_means = local_means.data + delta_xyz
            else:
                local_means = local_means + delta_xyz
        rot_cur_frame = quat_to_rotmat(quat_cur_frame[None, ...])[0, ...]
        global_means = local_means @ rot_cur_frame.T + trans_cur_frame
        return global_means

    def get_quats(self, quat_cur_frame, trans_cur_frame, quats=None):
        local_quats = self.gauss_params['quats'] / self.gauss_params['quats'].norm(dim=-1, keepdim=True)
        if quats is not None:
            local_quats = quats / quats.norm(dim=-1, keepdim=True)
        global_quats = quat_mult(quat_cur_frame[None, ...], local_quats)
        return global_quats

    def get_scales(self, delta_scale=None):
        scales = torch.exp(self.scales)
        if delta_scale is not None:
            scales = scales + delta_scale
        return scales

    def get_rgbs(self, camera_to_worlds, quat_cur_frame, trans_cur_frame, delta_xyz=None):
        colors = torch.cat((self.features_dc[:, None, :], self.features_rest), dim=1)
        if self.sh_degree > 0:
            viewdirs = self.get_means(quat_cur_frame, trans_cur_frame, delta_xyz).detach() - camera_to_worlds[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.ctrl_config.sh_degree_interval, self.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors)
            rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
        else:
            rgbs = torch.sigmoid(colors[:, 0, :])

        return rgbs     

    def get_opacity(self):
        return torch.sigmoid(self.gauss_params['opacities']).squeeze(-1)
    
    def get_object_pose(self, frame_idx, timestamp):
        """
            (quat, trans) for the current frame
            quat: (4, ), trans: (3, )
            The timestamp is used for interpolation between frames.
        """
        # if given frame_idx, return the pose at the frame_idx rather than interpolating
        # if timestamp is None:
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
    
    def get_deformation(self, frame_idx=None, timestamp=None):
        """
        get the deformation of the nonrigid instances
        """
        local_means = self.gauss_params['means']
        ins_height = self.instance_size[..., 2]
        ins_height = torch.tensor(ins_height, device=local_means.device)
        ins_height = ins_height.repeat(local_means.shape[0])
        x = local_means.data / ins_height[:, None] * 2
        if frame_idx is not None:
            t = self.dataframe_dict["frame_timestamps"][frame_idx]
        else:
            frame_timestamps = self.dataframe_dict["frame_timestamps"].to(self.device)  # (num_frames, )
            # Find the two adjacent frames for interpolation
            diffs = timestamp - frame_timestamps
            prev_frame = torch.argmin(torch.where(diffs >= 0, diffs, float('inf')))
            next_frame = torch.argmin(torch.where(diffs <= 0, -diffs, float('inf')))

            if not self.in_frame_mask[next_frame] or not self.in_frame_mask[prev_frame]:
                return None, None, None
            
            # Calculate interpolation factor
            t = (timestamp - frame_timestamps[prev_frame]) / (frame_timestamps[next_frame] - frame_timestamps[prev_frame])
        t = t.to(device=x.device, dtype=torch.float32)
        t = t.repeat(x.shape[0], 1)

        instance_embedding = self.instances_embedding.repeat(x.shape[0], 1)
        delta_xyz, delta_quat, delta_scale = self.deform_network(x, t, instance_embedding)
        return delta_xyz, delta_quat, delta_scale


    def get_gaussians(self, camera_to_worlds, travel_id=None, frame_idx=None, timestamp=None, return_features=False, return_v=False, **kwargs):
        if travel_id != self.travel_id or (frame_idx is None and timestamp is None):
            self.frame_idx = None
            return None

        self.frame_idx = frame_idx
        if frame_idx is not None:
            assert frame_idx < self.num_frames

        quat_cur_frame, trans_cur_frame = self.get_object_pose(frame_idx, timestamp)
        if quat_cur_frame is None or trans_cur_frame is None:
            self.frame_idx = None
            return None

        if timestamp is None:
            timestamp = self.dataframe_dict["frame_timestamps"][frame_idx]

        delta_xyz, delta_quat, delta_scale = None, None, None
        if self.config.use_deformgs_for_nonrigid and self.step > self.config.use_deformgs_after:
            delta_xyz, delta_quat, delta_scale = self.get_deformation(frame_idx, timestamp)

        quats = self.gauss_params['quats'] / self.gauss_params['quats'].norm(dim=-1, keepdim=True)
        if delta_quat is not None:
            quats = quats + delta_quat

        return_dict = {
            "means": self.get_means(quat_cur_frame, trans_cur_frame, delta_xyz),
            "scales": self.get_scales(delta_scale),
            "quats": self.get_quats(quat_cur_frame, trans_cur_frame, quats),
            "opacities": self.get_opacity(),
        }

        if return_features:
            return_dict.update({
                "features_dc": self.features_dc,
                "features_rest": self.features_rest,
            })
        else:
            return_dict['rgbs'] = self.get_rgbs(camera_to_worlds, quat_cur_frame, trans_cur_frame, delta_xyz)

        if return_v:
            raise NotImplementedError("Velocity is not implemented for deformable objects")

        return return_dict

    def get_gaussian_params(self, travel_id=None, frame_idx=None, timestamp=None, **kwargs):
        if travel_id != self.travel_id or (frame_idx is None and timestamp is None):
            return None

        if frame_idx is not None:
            assert frame_idx < self.num_frames

        quat_cur_frame, trans_cur_frame = self.get_object_pose(frame_idx, timestamp)
        if quat_cur_frame is None or trans_cur_frame is None:
            return None

        return {
            "means": self.get_means(quat_cur_frame, trans_cur_frame),
            "scales": self.scales,
            "quats": self.get_quats(quat_cur_frame, trans_cur_frame),
            "features_dc": self.features_dc,
            "features_rest": self.features_rest,
            "opacities": self.opacities,
        }
        
    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        return {
            f"{self.model_name}.{self.model_type}.{name}": [self.gauss_params[name]]
            for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
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
                CONSOLE.log(f"skip refinement after for non rigid object {self.model_name}")
                return

        super().refinement_after(optimizers, step)

    def state_dict(self, **kwargs):
        state_dict = super().state_dict(**kwargs)
        state_dict.update({
            "instance_size": self.instance_size,
            "in_frame_mask": self.in_frame_mask,
            "dataframe_dict": self.dataframe_dict,
            "travel_id": self.travel_id,
        })
        return state_dict

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
        super().load_state_dict(dict, **kwargs)

    # def get_loss_dict(self):
    #     loss_dict = super().get_loss_dict()
    #     if self.config.use_deformgs_for_nonrigid and self.step > self.config.use_deformgs_after:
    #         loss_dict["deform_loss"] = self.deform_network.loss()
    #     return loss_dict

    def translate(self, translate_vector: Tensor):
        assert translate_vector.shape == (3,)
        new_instance_trans = self.instance_trans + translate_vector
        self.instance_trans.data = new_instance_trans
