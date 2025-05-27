#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

import numpy as np
import torch

from torch import Tensor
from torch.nn import Parameter

try:
    from gsplat.cuda._wrapper import spherical_harmonics
except ImportError:
    print("Please install gsplat>=1.0.0")

from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers

# need following import for background color override
from nerfstudio.configs.base_config import InstantiateConfig, PrintableConfig
from nerfstudio.utils.rich_utils import CONSOLE

from .utils import num_sh_bases, random_quat_tensor, matrix_to_quaternion, rotate_vector_to_vector, RGB2SH, quat_to_rotmat

@dataclass
class GaussianSplattingControlConfig(PrintableConfig):
    densify_from_iter: Optional[int] = None
    """period of steps where refinement is turned off"""
    refine_every: Optional[int] = None
    """period of steps where gaussians are culled and densified"""
    stop_split_at: Optional[int] = None
    """stop splitting at this step"""
    reset_alpha_every: Optional[int] = None
    """Every this many refinement steps, reset the alpha"""
    continue_cull_post_densification: Optional[bool] = None
    """If True, continue to cull gaussians post refinement"""

    cull_alpha_thresh: Optional[float] = None
    """threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""
    cull_scale_thresh: Optional[float] = None
    """threshold of scale for culling huge gaussians"""
    densify_grad_thresh: Optional[float] = None
    """threshold of positional gradient norm for densifying gaussians"""
    densify_size_thresh: Optional[float] = None
    """below this size, gaussians are *duplicated*, otherwise split, unit: meters"""
    n_split_samples: Optional[int] = None
    """number of samples to split gaussians into"""
    clone_sample_means: Optional[bool] = None
    """When duplicating gaussians, whether to sample new means or clone the existing ones."""

    cull_scale_ratio: Optional[float] = None
    """minimum scale threshold for culling or splitting gaussians."""
    densify_size_ratio: Optional[float] = None
    """minimum scale threshold for splitting gaussians."""

    stop_screen_size_at: Optional[int] = None
    """stop culling/splitting at this step WRT screen size of gaussians"""
    cull_screen_size: Optional[float] = None
    """if a gaussian is more than this pixel of screen space, cull it"""
    split_screen_size: Optional[float] = None
    """if a gaussian is more than this pixel of screen space, split it"""

    sh_degree: Optional[int] = None
    """maximum degree of spherical harmonics to use"""
    sh_degree_interval: Optional[int] = None
    """every n intervals turn on another sh degree"""
    use_abs_grad: Optional[bool] = None
    """Whether to use absolute gradient or not"""

    scale_dim: int = 3
    """dimension of scale for gaussians
    for scale_dim == 3, original 3D Gaussian splatting.
    for scale_dim == 1, Gaussian marbles.
    """


@dataclass
class VanillaGaussianSplattingModelConfig(InstantiateConfig):
    """Gaussian Splatting Model Config"""

    _target: Type = field(default_factory=lambda: VanillaGaussianSplattingModel)
    model_type: str = 'vanilla'
    control: Optional[GaussianSplattingControlConfig] = None
    verbose: bool = False


class VanillaGaussianSplattingModel(torch.nn.Module):
    """Gaussian Splatting model

    Args:
        config: Gaussian Splatting configuration to instantiate model
    """

    config: VanillaGaussianSplattingModelConfig

    def __init__(
        self,
        config: VanillaGaussianSplattingModelConfig,
        model_name: Optional[str] = None,
        model_id: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.model_name = model_name
        self.model_type = self.config.model_type
        self.model_id = model_id
        self.frozen = False

    def populate_modules(self, points_3d=None, features_dc_dim=None, **kwargs):
        assert points_3d is not None
        assert (features_dc_dim is None) or (features_dc_dim is not None and features_dc_dim > 1), \
            "invalid features_dc_dim [{}], must be set to `None` or larger than 1".format(features_dc_dim)
        means = torch.nn.Parameter(points_3d['xyz'])  # (Location, Color)

        self.xys_grad_norm = None
        self.max_2Dsize = None
        self.vis_counts = None
        num_points = means.shape[0]
        dim_sh = num_sh_bases(self.sh_degree)
        if num_points == 0:
            self._skip_current_model(dim_sh)
            return

        shs = torch.zeros((points_3d['rgb'].shape[0], dim_sh, 3)).float().cuda()
        if self.sh_degree > 0:
            shs[:, 0, :3] = RGB2SH(points_3d['rgb'] / 255)
            shs[:, 1:, 3:] = 0.0
        else:
            CONSOLE.log("use color only optimization with sigmoid activation")
            shs[:, 0, :3] = torch.logit(points_3d['rgb'] / 255, eps=1e-10)
        
        if features_dc_dim is None:
            features_dc = torch.nn.Parameter(shs[:, 0, :])
        else:
            features_dc = torch.zeros(num_points, features_dc_dim, 3)
            features_dc[:, 0, :] = shs[:, 0, :]
            features_dc = torch.nn.Parameter(features_dc)
        features_rest = torch.nn.Parameter(shs[:, 1:, :])

        distances, _ = self.k_nearest_sklearn(means.data, 3)
        distances = torch.from_numpy(distances)
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True)

        opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(num_points, 1)))

        if self.scale_dim == 3:
            if points_3d.get('normals', None) is not None:
                normals_seed = points_3d['normals'].float()
                normals_seed = normals_seed / torch.norm(
                    normals_seed, dim=-1, keepdim=True
                )
                scales = torch.log(avg_dist.repeat(1, 3))
                scales[:, 2] = torch.log((avg_dist / 10)[:, 0])
                scales = torch.nn.Parameter(scales.detach())
                quats = torch.zeros(len(normals_seed), 4)
                mat = rotate_vector_to_vector(
                    torch.tensor(
                        [0, 0, 1], dtype=torch.float, device=normals_seed.device
                    ).repeat(normals_seed.shape[0], 1),
                    normals_seed,
                )
                quats = matrix_to_quaternion(mat)
                quats = torch.nn.Parameter(quats.detach())
            else:
                scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
                quats = torch.nn.Parameter(random_quat_tensor(num_points))
            
            self.gauss_params = torch.nn.ParameterDict(
                {
                    "means": means,
                    "scales": scales,
                    "quats": quats,
                    "features_dc": features_dc,
                    "features_rest": features_rest,
                    "opacities": opacities,
                }
            )

        elif self.scale_dim == 1:
            scales = torch.nn.Parameter(torch.log(avg_dist))

            self.gauss_params = torch.nn.ParameterDict(
                {
                    "means": means,
                    "scales": scales,
                    "features_dc": features_dc,
                    "features_rest": features_rest,
                    "opacities": opacities,
                }
            )

    def __empty_gaussians(self, num_points=0, dim_sh=None, features_dc_dim=None):
        dim_sh = num_sh_bases(self.sh_degree) if dim_sh is None else dim_sh
        features_dc = torch.zeros(num_points, 3) if features_dc_dim is None else torch.zeros(num_points, features_dc_dim, 3)
        self.xys_grad_norm = None
        self.max_2Dsize = None
        self.vis_counts = None
        self.gauss_params = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(torch.zeros(num_points, 3)),
                "scales": torch.nn.Parameter(torch.zeros(num_points, self.scale_dim)),
                "quats": torch.nn.Parameter(torch.zeros(num_points, 4)),
                "features_dc": torch.nn.Parameter(features_dc),
                "features_rest": torch.nn.Parameter(torch.zeros(num_points, dim_sh - 1, 3)),
                "opacities": torch.nn.Parameter(torch.logit(0.1 * torch.ones(num_points, 1))),
            }
        )
        return self.gauss_params

    def _skip_current_model(self, dim_sh):
        self.__empty_gaussians(0, dim_sh)
        # if we skip a model, there is no need to use time-aware fourier embedding

    def freeze(self):
        for param in self.gauss_params.values():
            param.requires_grad = False
        self.frozen = True

    def update_statistics(self, xys_grad: torch.Tensor, radii: torch.Tensor):
        self.xys_grad = xys_grad
        self.radii = radii

    def get_isotropic_quats(self, num_points):
        quats = torch.zeros(num_points, 4)
        quats[:, 0] = 1.0
        return quats.to(self.device)

    @property
    def device(self):
        return self.gauss_params["means"].device

    @property
    def sh_degree(self):
        return self.config.control.sh_degree
    
    @property
    def ctrl_config(self):
        return self.config.control
    
    @property
    def scale_dim(self):
        return self.config.control.scale_dim

    @property
    def portable_config(self):
        return {
            "type": type(self).__name__,
            "sh_degree": self.sh_degree,
            "scale_dim": self.scale_dim,
        }
    
    @property
    def num_points(self):
        return self.means.shape[0]

    @property
    def means(self) -> torch.nn.Parameter:
        return self.gauss_params["means"]

    @property
    def scales(self) -> torch.nn.Parameter:
        if self.scale_dim == 3:
            return self.gauss_params["scales"]
        elif self.scale_dim == 1:
            return self.gauss_params["scales"].repeat(1, 3)

    @property
    def quats(self) -> torch.nn.Parameter:
        if self.scale_dim == 3:
            return self.gauss_params["quats"]
        elif self.scale_dim == 1:
            return self.get_isotropic_quats(self.num_points)

    @property
    def features_dc(self) -> torch.nn.Parameter:
        return self.gauss_params["features_dc"]

    @property
    def features_rest(self) -> torch.nn.Parameter:
        return self.gauss_params["features_rest"]

    @property
    def opacities(self) -> torch.nn.Parameter:
        return self.gauss_params["opacities"]

    @property
    def model_name_abbr(self):
        return self.model_name.split("_")[-1]

    def get_means(self):
        return self.gauss_params['means']

    def get_scales(self):
        return torch.exp(self.scales)

    def get_quats(self): 
        quats = self.quats
        return quats / quats.norm(dim=-1, keepdim=True)

    def get_opacity(self):
        return torch.sigmoid(self.gauss_params['opacities']).squeeze(-1)
    
    def get_rgbs(self, camera_to_worlds):
        assert self.features_dc.dim() == 2, \
            "if Fourier embedding is used in models derived from `VanillaGaussianSplattingModel`, a new `get_rgbs` function must be implemented"
        colors = torch.cat((self.features_dc[:, None, :], self.features_rest), dim=1)
        if self.sh_degree > 0:
            viewdirs = self.means.detach() - camera_to_worlds[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.ctrl_config.sh_degree_interval, self.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors)
            rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
        else:
            rgbs = torch.sigmoid(colors[:, 0, :])
        
        return rgbs

    def get_gaussians(self, camera_to_worlds, return_features=False, return_v=False, **kwargs):

        return_dict = {
            "means": self.get_means(),
            "scales": self.get_scales(),
            "quats": self.get_quats(),
            "opacities": self.get_opacity(),
        }

        if return_features:
            return_dict["features_dc"] = self.features_dc
            return_dict["features_rest"] = self.features_rest
        else:
            return_dict["rgbs"] = self.get_rgbs(camera_to_worlds)

        if return_v:
            return_dict["velocities"] = torch.zeros_like(return_dict["means"])
        return return_dict

    def get_gaussian_params(self, **kwargs):
        return self.gauss_params

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        return {
            f"{self.model_name}.{self.model_type}.{name}": [self.gauss_params[name]]
            for name in self.gauss_params.keys()
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        return self.get_gaussian_param_groups()

    def load_state_dict(self, dict: dict, **kwargs):  # type: ignore

        # resize the parameters to match the new number of points
        newp = dict["gauss_params.means"].shape[0]
        if getattr(self, "num_points", None) is None:
            self.__empty_gaussians(newp)
        else:
            for name, param in self.gauss_params.items():
                old_shape = param.shape
                new_shape = (newp,) + old_shape[1:]
                self.gauss_params[name] = torch.nn.Parameter(torch.zeros(new_shape, device=self.device))

        if 'strict' in kwargs:
            kwargs.pop('strict')

        super().load_state_dict(dict, strict=False, **kwargs)

    def k_nearest_sklearn(self, x: torch.Tensor, k: int):
        """
            Find k-nearest neighbors using sklearn's NearestNeighbors.
        x: The data tensor of shape [num_samples, num_features]
        k: The number of neighbors to retrieve
        """
        # Convert tensor to numpy array
        x_np = x.cpu().numpy()

        # Build the nearest neighbors model
        from sklearn.neighbors import NearestNeighbors

        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)

        # Find the k-nearest neighbors
        distances, indices = nn_model.kneighbors(x_np)

        # Exclude the point itself from the result and return
        return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)

    def remove_from_optim(self, optimizer, deleted_mask, new_params):
        """removes the deleted_mask from the optimizer provided"""
        assert len(new_params) == 1
        # assert isinstance(optimizer, torch.optim.Adam), "Only works with Adam"

        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        del optimizer.state[param]

        # Modify the state directly without deleting and reassigning.
        if "exp_avg" in param_state:
            param_state["exp_avg"] = param_state["exp_avg"][~deleted_mask]
            param_state["exp_avg_sq"] = param_state["exp_avg_sq"][~deleted_mask]

        # Update the parameter in the optimizer's param group.
        del optimizer.param_groups[0]["params"][0]
        del optimizer.param_groups[0]["params"]
        optimizer.param_groups[0]["params"] = new_params
        optimizer.state[new_params[0]] = param_state

    def remove_from_all_optim(self, optimizers, deleted_mask):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.remove_from_optim(optimizers.optimizers[group], deleted_mask, param)
        torch.cuda.empty_cache()

    def dup_in_optim(self, optimizer, dup_mask, new_params, n=2):
        """adds the parameters to the optimizer"""
        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        if "exp_avg" in param_state:
            repeat_dims = (n,) + tuple(1 for _ in range(param_state["exp_avg"].dim() - 1))
            param_state["exp_avg"] = torch.cat(
                [
                    param_state["exp_avg"],
                    torch.zeros_like(param_state["exp_avg"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
            param_state["exp_avg_sq"] = torch.cat(
                [
                    param_state["exp_avg_sq"],
                    torch.zeros_like(param_state["exp_avg_sq"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
        del optimizer.state[param]
        optimizer.state[new_params[0]] = param_state
        optimizer.param_groups[0]["params"] = new_params
        del param

    def dup_in_all_optim(self, optimizers, dup_mask, n):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.dup_in_optim(optimizers.optimizers[group], dup_mask, param, n)

    def after_train(self, step: int):
        assert step == self.step
        if self.frozen:
            return
        # to save some training time, we no longer need to update those stats post refinement
        if self.step >= self.ctrl_config.stop_split_at:
            return
        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = (self.radii > 0).flatten()
            grads = self.xys_grad[visible_mask]

            if self.xys_grad_norm is None:
                self.xys_grad_norm = torch.zeros(self.num_points, device=self.device, dtype=torch.float32)
                self.vis_counts = torch.ones(self.num_points, device=self.device, dtype=torch.float32)
            assert self.vis_counts is not None
            self.vis_counts[visible_mask] += + 1
            self.xys_grad_norm[visible_mask] += grads

            # update the max screen size, as number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros_like(self.radii, dtype=torch.float32)
            newradii = self.radii.detach()[visible_mask]
            self.max_2Dsize[visible_mask] = torch.maximum(
                self.max_2Dsize[visible_mask],
                newradii,
            )

    def refinement_after(self, optimizers: Optimizers, step):
        assert step == self.step
        if self.frozen:
            return
        if self.step <= self.ctrl_config.densify_from_iter:
            return
        with torch.no_grad():
            if self.num_points <= 0:
                return
            reset_interval = self.ctrl_config.reset_alpha_every * self.ctrl_config.refine_every
            if self.step < self.ctrl_config.stop_split_at:
                # then we densify
                if self.config.verbose:
                    CONSOLE.log(f"[=================== INFO for {self.model_name_abbr} <{self.model_type}> ===================]")
                    CONSOLE.log(f"Do densification at step {self.step}")
                assert self.xys_grad_norm is not None and self.vis_counts is not None and self.max_2Dsize is not None
                # avg_grad_norm = (self.xys_grad_norm / self.vis_counts) * 0.5 * max(self.last_size[0], self.last_size[1])
                avg_grad_norm = self.xys_grad_norm / self.vis_counts

                if self.config.verbose:
                    CONSOLE.log(f"avg_grad_norm: {avg_grad_norm.mean().item()}")

                high_grads = (avg_grad_norm > self.ctrl_config.densify_grad_thresh).squeeze()

                splits = (self.scales.exp().max(dim=-1).values > self.ctrl_config.densify_size_thresh).squeeze()
                splits &= high_grads

                if self.step < self.ctrl_config.stop_screen_size_at:
                    splits |= (self.max_2Dsize > self.ctrl_config.split_screen_size).squeeze()

                nsamps = self.ctrl_config.n_split_samples
                split_params = self.split_gaussians(splits, nsamps)

                dups = (self.scales.exp().max(dim=-1).values <= self.ctrl_config.densify_size_thresh).squeeze()
                dups &= high_grads
                dup_params = self.dup_gaussians(dups)
                for name, param in self.gauss_params.items():
                    self.gauss_params[name] = torch.nn.Parameter(
                        torch.cat([param.detach(), split_params[name], dup_params[name]], dim=0)
                    )

                # append zeros to the max_2Dsize tensor
                self.max_2Dsize = torch.cat(
                    [
                        self.max_2Dsize,
                        torch.zeros_like(split_params["scales"][:, 0]),
                        torch.zeros_like(dup_params["scales"][:, 0]),
                    ],
                    dim=0,
                )

                split_idcs = torch.where(splits)[0]
                self.dup_in_all_optim(optimizers, split_idcs, nsamps)

                dup_idcs = torch.where(dups)[0]
                self.dup_in_all_optim(optimizers, dup_idcs, 1)

                # After a guassian is split into two new gaussians, the original one should also be pruned.
                splits_mask = torch.cat(
                    (
                        splits,
                        torch.zeros(
                            nsamps * splits.sum() + dups.sum(),
                            device=self.device,
                            dtype=torch.bool,
                        ),
                    )
                )

                deleted_mask = self.cull_gaussians(splits_mask)
            elif self.step >= self.ctrl_config.stop_split_at and self.ctrl_config.continue_cull_post_densification:
                deleted_mask = self.cull_gaussians()
            else:
                # if we donot allow culling post refinement, no more gaussians will be pruned.
                deleted_mask = None

            if deleted_mask is not None:
                self.remove_from_all_optim(optimizers, deleted_mask)

            if self.step < self.ctrl_config.stop_split_at and self.step % reset_interval == self.ctrl_config.refine_every:
                if self.config.verbose:
                    CONSOLE.log(f"Reset opacity at step {self.step}")
                # Reset value is set to be twice of the cull_alpha_thresh
                reset_value = self.ctrl_config.cull_alpha_thresh * 2.0
                self.opacities.data = torch.clamp(
                    self.opacities.data,
                    max=torch.logit(torch.tensor(reset_value, device=self.device)).item(),
                )
                # reset the exp of optimizer
                gaussian_param_groups = self.get_gaussian_param_groups()
                for name, param in gaussian_param_groups.items():
                    if "opacities" in name:
                        optim = optimizers.optimizers[name]
                        param = optim.param_groups[0]["params"][0]
                        param_state = optim.state[param]
                        param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                        param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])
                        break

            self.xys_grad_norm = None
            self.vis_counts = None
            self.max_2Dsize = None

    def cull_gaussians(self, extra_cull_mask: Optional[torch.Tensor] = None):
        """
        This function deletes gaussians with under a certain opacity threshold
        extra_cull_mask: a mask indicates extra gaussians to cull besides existing culling criterion
        """
        n_bef = self.num_points
        # cull transparent ones
        culls = (torch.sigmoid(self.opacities) < self.ctrl_config.cull_alpha_thresh).squeeze()
        below_alpha_count = torch.sum(culls).item()
        toobigs_count = 0
        toobigs_ws_count = 0
        toobigs_vs_count = 0
        if extra_cull_mask is not None:
            culls = culls | extra_cull_mask
            n_bef -= torch.sum(extra_cull_mask).item()

        if self.step > self.ctrl_config.refine_every * self.ctrl_config.reset_alpha_every:

            # Temporal fix for far background gaussians.
            # Do not cull far background gaussians.
            far_mask = self.means.norm(dim=-1) > 100
            cull_scale_thresh = torch.where(far_mask, 40, 1.) * self.ctrl_config.cull_scale_thresh

            # cull huge ones
            toobigs = (torch.exp(self.scales).max(dim=-1).values > cull_scale_thresh).squeeze()
            toobigs_ws_count = torch.sum(toobigs).item()
            if self.step < self.ctrl_config.stop_screen_size_at:
                # cull big screen space
                assert self.max_2Dsize is not None
                toobigs_vs = (self.max_2Dsize > self.ctrl_config.cull_screen_size).squeeze()
                toobigs_vs_count = torch.sum(toobigs_vs).item()
                toobigs = toobigs | toobigs_vs

            culls = culls | toobigs
            toobigs_count = torch.sum(toobigs).item()
        for name, param in self.gauss_params.items():
            self.gauss_params[name] = torch.nn.Parameter(param[~culls])

        if self.config.verbose:
            CONSOLE.log(
                f"Culled {n_bef - self.num_points} gaussians "
                f"({below_alpha_count} below alpha thresh, {toobigs_count}({toobigs_ws_count}, {toobigs_vs_count}) too bigs, {self.num_points} remaining)"
            )

        return culls

    def split_features_dc(self, split_mask, samps):
        return self.features_dc[split_mask].repeat(samps, 1)

    def split_properties(self, split_mask, samps):
        return {
            "features_dc": self.split_features_dc(split_mask, samps),
            "features_rest": self.features_rest[split_mask].repeat(samps, 1, 1),
            "opacities": self.opacities[split_mask].repeat(samps, 1),
        }

    def split_gaussians(self, split_mask, samps):
        """
        This function splits gaussians that are too large
        """
        n_splits = split_mask.sum().item()
        if self.config.verbose:
            CONSOLE.log(f"Splitting {split_mask.sum().item()/self.num_points} gaussians: {n_splits}/{self.num_points}")
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
            torch.exp(self.scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        quats = self.quats[split_mask] / self.quats[split_mask].norm(dim=-1, keepdim=True)  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self.means[split_mask].repeat(samps, 1)
        # step 2 & 3, sample new colors, new opacities, and other properties
        new_properties = self.split_properties(split_mask, samps)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self.gauss_params["scales"][split_mask]) / size_fac).repeat(samps, 1)
        self.scales[split_mask] = torch.log(torch.exp(self.scales[split_mask]) / size_fac)
        if self.scale_dim == 3:    
            # step 5, sample new quats
            new_quats = self.quats[split_mask].repeat(samps, 1)
            out = {
                "means": new_means,
                **new_properties,
                "scales": new_scales,
                "quats": new_quats,
            }
        else:
            out = {
                "means": new_means,
                **new_properties,
                "scales": new_scales,
            }
        for name, param in self.gauss_params.items():
            if name not in out:
                out[name] = param[split_mask].repeat(samps, 1)
        return out

    def dup_gaussians(self, dup_mask):
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        if self.config.verbose:
            CONSOLE.log(f"Duplicating {dup_mask.sum().item()/self.num_points} gaussians: {n_dups}/{self.num_points}")
        new_dups = {}
        for name, param in self.gauss_params.items():

            if self.ctrl_config.clone_sample_means and name == 'means':
                centered_samples = torch.randn((n_dups, 3), device=self.device)  # Nx3 of axis-aligned scales
                scaled_samples = (
                    torch.exp(self.scales[dup_mask]) * centered_samples
                )  # how these scales are rotated
                quats = self.quats[dup_mask] / self.quats[dup_mask].norm(dim=-1, keepdim=True)  # normalize them first
                rots = quat_to_rotmat(quats)  # how these scales are rotated
                rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
                new_means = rotated_samples + self.means[dup_mask]
                new_dups[name] = new_means
                continue

            new_dups[name] = param[dup_mask]
        return new_dups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.step_cb))
        # The order of these matters
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.after_train,
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.refinement_after,
                update_every_num_iters=self.ctrl_config.refine_every,
                args=[training_callback_attributes.optimizers],
            )
        )
        return cbs

    def step_cb(self, step):
        self.step = step
    
    def translate(self, translate_vector: Tensor):
        assert translate_vector.shape == (3,)
        new_means = self.gauss_params['means'] + translate_vector
        self.gauss_params['means'].data = new_means
