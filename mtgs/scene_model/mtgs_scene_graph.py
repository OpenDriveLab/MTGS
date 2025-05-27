#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import copy
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, List, Optional, Tuple, Type, Union
from typing_extensions import Literal

import numpy as np
import torch
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Parameter
from torch import nn

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.3.0")

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation

from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils import colormaps
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig

from mtgs.utils.geometric_loss import DepthLossType, DepthLoss, TVLoss, calculate_depth_ncc_loss, normal_from_depth_image
from mtgs.utils.ssim import MaskedSSIM
from mtgs.utils.pnsr import MaskedPSNR, color_correct

from .gaussian_model.vanilla_gaussian_splatting import VanillaGaussianSplattingModelConfig, GaussianSplattingControlConfig
from .gaussian_model.rigid_node import RigidSubModel
from .gaussian_model.utils import quat_to_rotmat, SH2RGB
from .module.appearance import GSAppearanceModelConfig, GSAppearanceModel

@dataclass
class MTGSSceneModelConfig(ModelConfig):

    _target: Type = field(default_factory=lambda: MTGSSceneModel)
    enable_collider: bool = False

    control: GaussianSplattingControlConfig = field(default_factory=lambda: GaussianSplattingControlConfig())
    """ Default control config for all gaussian models """

    model_config: Dict[str, VanillaGaussianSplattingModelConfig] = field(default_factory=lambda: {})

    background_color: Literal["random", "black", "white"] = "black"
    """Whether to randomize the background color."""

    output_depth_during_training: bool = False
    """If True, output depth during training. Otherwise, only output depth during evaluation."""
    predict_normals: bool = False
    """If True, predict normals for each gaussian."""

    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="off"))
    """Config of the camera optimizer to use"""

    appearance_model: GSAppearanceModelConfig = field(default_factory=lambda: GSAppearanceModelConfig())
    """Config of the full image appearance model for 3DGS to use."""

    color_corrected_metrics: bool = True
    """If True, apply color correction to the rendered images before computing the metrics."""

    lpips_metric: bool = True
    """If True, use lpips metric."""

    dinov2_metric: bool = False
    """If True, use dinov2 metric."""

    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    """
    Classic mode of rendering will use the EWA volume splatting with a [0.3, 0.3] screen space blurring kernel. This
    approach is however not suitable to render tiny gaussians at higher or lower resolution than the captured, which
    results "aliasing-like" artifacts. The antialiased mode overcomes this limitation by calculating compensation factors
    and apply them to the opacities of gaussians to preserve the total integrated density of splats.

    However, PLY exported with antialiased rasterize mode is not compatible with classic mode. Thus many web viewers that
    were implemented for classic mode can not render antialiased mode PLY properly without modifications.
    """

    ssim_lambda: float = 0.2
    """weight of ssim loss"""

    # Depth and normal related losses. Adapted from DN-splatter
    use_depth_loss: bool = False
    """Enable depth loss while training"""
    depth_source: Literal["lidar", "pseudo"] = "lidar"
    """Enable lidar depth loss while training"""
    depth_loss_type: DepthLossType = DepthLossType.InverseL1
    """Choose which depth loss to train with Literal["MSE", "LogL1", "HuberL1", "L1", "EdgeAwareLogL1")"""
    depth_lambda: float = 0.0
    """loss weight of depth loss"""

    ncc_loss_lambda: float = 0.0
    """Weight of NCC loss"""
    ncc_patch_size: int = 32
    """NCC patch size"""
    ncc_stride: int = 16
    """NCC stride"""

    use_normal_loss: bool = False
    """Enables normal loss('s)"""
    normal_supervision: Literal["mono", "depth"] = "depth"
    """Type of supervision for normals. Mono for monocular normals and depth for pseudo normals from depth maps."""
    use_normal_tv_loss: bool = False
    """Use TV loss on predicted normals."""
    normal_lambda: float = 0.1
    """Regularizer for normal loss"""
    two_d_gaussians: bool = False
    """Encourage 2D Gaussians"""

    adapter_lambda: float = 0.0
    """Regularizer for feature adapters"""

    oob_lambda: float = 0.0
    """Weight of the out-of-bound loss"""
    oob_tolerance: float = 1.5

    sharp_shape_reg_lambda: float = 0.0
    """Weight of the sharp shape regularization"""
    sharp_shape_reg_max_gauss_ratio: float = 10.0
    """Max gauss ratio for the sharp shape regularization"""
    sharp_shape_reg_step_interval: int = 10
    """Step interval for the sharp shape regularization"""

    use_wild_gaussians: bool = False
    """Reproduced version of WildGaussians (https://arxiv.org/abs/2407.08447)."""

    use_ssim_on_raw_rgb: bool = True
    """If True, use SSIM on raw RGB images and use L1 on color corrected images."""


class MTGSSceneModel(Model):

    config: MTGSSceneModelConfig

    def __init__(
        self,
        config: ModelConfig,
        scene_box,
        num_train_data: int,
        datamanager,
        **kwargs,
    ):
        super(Model, self).__init__()
        self.config = config
        self.scene_box = scene_box
        self.render_aabb = None
        self.num_train_data = num_train_data
        self.datamanager = datamanager
        self.device_indicator_param = Parameter(torch.empty(0))
        self.callbacks = None

        self.masked_out_classes = self.datamanager.config.load_custom_masks
        self.populate_modules()  # populate the modules

    def populate_modules(self):
        self._init_scene(self.scene_box.aabb)
        self._prepare_metas()
        self._init_gaussian_models()
        self._prepare_modules_and_metrics()
        self.update_to_step(0)

    def _init_scene(self, scene_aabb: Tensor):
        self.aabb = scene_aabb.to(self.device)
        scene_origin = (self.aabb[0] + self.aabb[1]) / 2
        scene_radius = torch.max(self.aabb[1] - self.aabb[0]) / 2 * 1.1
        self.scene_radius = scene_radius.item()
        self.scene_origin = scene_origin

    def _prepare_metas(self):
        self.data_frame_dict = self.datamanager.train_dataparser_outputs.metadata['multi_travel_frame_timestamps']
        self.train_travel_infos = self.datamanager.train_dataparser_outputs.metadata.get("train_travel_ids", ())
        self.train_travel_ids = [travel['idx'] if isinstance(travel, dict) else travel for travel in self.train_travel_infos]
        self.frame_token2frame_idx = self.datamanager.train_dataparser_outputs.metadata['frame_token2frame_idx']
        self.cam_token2cam_idx = self.datamanager.train_dataparser_outputs.metadata['cam_token2cam_idx']
        self.instance_info = self.datamanager.train_dataparser_outputs.metadata.get("instances_info", None)
        self.model_ids = {}
        self.model_types = {}
        self.node_type = {}
        self.cur_cam_metas = {
            'frame_idx': 0,
            'travel_id': 0,
        }

    def _update_gaussian_cfg(self, model_cfg):
        default_ctrl_cfg = copy.deepcopy(self.config.control)

        class_ctrl_cfg: GaussianSplattingControlConfig = model_cfg.control
        if class_ctrl_cfg is not None:
            for attr, value in class_ctrl_cfg.__dict__.items():
                if value is not None:
                    setattr(default_ctrl_cfg, attr, value)

        model_cfg.control = default_ctrl_cfg

        return model_cfg

    def _init_gaussian_models(self):
        self.gaussian_models = torch.nn.ModuleDict()
        cur_model_id = 0
        for config_name, model_config in self.config.model_config.items():
            if config_name == 'background':
                model_name = 'background'
                self.node_type[model_name] = 'background'
                # init background model
                model_config = self._update_gaussian_cfg(model_config)
                background_model = model_config.setup(
                    model_name=model_name, 
                    model_id=cur_model_id,
                    travel_ids=self.train_travel_ids,
                    nearest_train_travel_of_eval=self.datamanager.train_dataparser_outputs.metadata['nearest_train_travel_of_eval'],
                )
                background_model.populate_modules(
                    points_3d = dict(
                        xyz = self.datamanager.train_dataparser_outputs.metadata["points3D_xyz"],
                        rgb = self.datamanager.train_dataparser_outputs.metadata["points3D_rgb"],
                        normal = None
                    ),
                    num_traversals = self.num_traversals
                )
                self.gaussian_models[model_name] = background_model
                self.model_ids[model_name] = cur_model_id
                self.model_types[model_name] = model_config.model_type
                cur_model_id += 1
            elif config_name == 'skybox':
                max_distance = torch.max(
                    self.datamanager.train_dataparser_outputs.metadata["points3D_xyz"].norm(dim=-1))
                model_config = self._update_gaussian_cfg(model_config)
                model_name = 'skybox'
                self.node_type[model_name] = 'skybox'
                sky_model = model_config.setup(
                    model_name=model_name,
                    model_id=cur_model_id,
                    travel_ids=self.train_travel_ids,
                    nearest_train_travel_of_eval=self.datamanager.train_dataparser_outputs.metadata['nearest_train_travel_of_eval'],
                )
                sky_model.populate_modules(
                    max_distance=max_distance,
                    num_traversals=self.num_traversals
                )
                self.gaussian_models[model_name] = sky_model
                self.model_ids[model_name] = cur_model_id
                self.model_types[model_name] = model_config.model_type
                cur_model_id += 1
            elif config_name == 'rigid_object':
                if 'vehicle' in self.masked_out_classes:
                    continue
                instances_info = self.datamanager.train_dataparser_outputs.metadata["instances_info"]
                model_config = self._update_gaussian_cfg(model_config)
                for instance_token, instance_dict in instances_info.items():
                    class_name = instance_dict["class_name"]
                    if class_name not in ["vehicle"]:
                        continue
                    self.model_types[instance_token] = model_config.model_type
                    self.node_type[instance_token] = config_name
                    model_name = self.get_submodel_name(instance_token)
                    rigid_model = model_config.setup(
                        model_name=model_name,
                        model_id=cur_model_id,
                        instance_info=self.instance_info[instance_token]
                    )
                    rigid_model.populate_modules(instance_dict, self.data_frame_dict)
                    self.gaussian_models[model_name] = rigid_model
                    self.model_ids[model_name] = cur_model_id
                    cur_model_id += 1
            elif config_name == 'deformable_node':
                instances_info = self.datamanager.train_dataparser_outputs.metadata["instances_info"]
                for instance_token, instance_dict in instances_info.items():
                    class_name = instance_dict["class_name"]
                    if class_name not in ["bicycle", "pedestrian"]:
                        continue
                    if class_name in self.masked_out_classes:
                        continue

                    self.model_types[instance_token] = model_config.model_type
                    self.node_type[instance_token] = config_name
                    model_name = self.get_submodel_name(instance_token)
                    model_config = self._update_gaussian_cfg(model_config)
                    deformable_node = model_config.setup(
                        model_name=model_name,
                        model_id=cur_model_id
                    )
                    deformable_node.populate_modules(instance_dict, self.data_frame_dict)
                    self.gaussian_models[model_name] = deformable_node
                    self.model_ids[model_name] = cur_model_id
                    cur_model_id += 1
            else:
                raise NotImplementedError

    def _prepare_modules_and_metrics(self):
        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

        self.appearance_model: GSAppearanceModel = self.config.appearance_model.setup(
            num_cameras=self.num_train_data
        )

        if self.config.use_wild_gaussians:
            self.camera_embedding = Embedding(
                self.num_train_data, 32
            )
            self.wild_MLP = nn.Sequential(
                nn.Linear(32 + 3 + 6 * 4, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 6),
            )

        # metrics
        self.psnr = MaskedPSNR(data_range=1.0)
        self.ssim = MaskedSSIM(data_range=1.0, size_average=True, channel=3)
        if self.config.lpips_metric:
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
            self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        if self.config.dinov2_metric:
            from mtgs.utils.dinov2 import DINOv2Similarity
            self.dinov2 = DINOv2Similarity()

        self.step = 0

        self.crop_box: Optional[OrientedBox] = None
        self.bg_color = self._init_background_color()

        # depth loss
        if self.config.use_depth_loss and self.config.depth_source == "pseudo":
            self.depth_loss = DepthLoss(self.config.depth_loss_type)

        if self.config.use_normal_tv_loss:
            self.tv_loss = TVLoss()

    def _init_background_color(self):
        if self.config.background_color == "random":
            if self.training:
                background = torch.rand(3, device=self.device)
            else:
                background = self.background_color.to(self.device)
        elif self.config.background_color == "white":
            background = torch.ones(3, device=self.device)
        elif self.config.background_color == "black":
            background = torch.zeros(3, device=self.device)
        else:
            raise ValueError(f"Unknown background color {self.config.background_color}")
        return background

    def get_submodel_name(self, token):
        node_type = self.node_type[token]
        return f"{node_type}_{token}"

    @cached_property
    def num_points(self):
        return self.means.shape[0]

    @cached_property
    def num_traversals(self):
        return len(self.train_travel_ids)

    @cached_property
    def means(self) -> torch.nn.Parameter:
        return self.get_gaussian_params(self.cur_travel_id, self.cur_frame_idx)["means"]

    @cached_property
    def scales(self) -> torch.nn.Parameter:
        return self.get_gaussian_params(self.cur_travel_id, self.cur_frame_idx)["scales"]

    @cached_property
    def quats(self) -> torch.nn.Parameter:
        return self.get_gaussian_params(self.cur_travel_id, self.cur_frame_idx)["quats"]

    @cached_property
    def features_dc(self) -> torch.nn.Parameter:
        return self.get_gaussian_params(self.cur_travel_id, self.cur_frame_idx)["features_dc"]

    @cached_property
    def features_rest(self) -> torch.nn.Parameter:
        return self.get_gaussian_params(self.cur_travel_id, self.cur_frame_idx)["features_rest"]

    @cached_property
    def opacities(self) -> torch.nn.Parameter:
        return self.get_gaussian_params(self.cur_travel_id, self.cur_frame_idx)["opacities"]

    @cached_property
    def background_color(self) -> torch.Tensor:
        return self.bg_color.to(self.device)

    @property
    def cur_frame_idx(self):
        return self.cur_cam_metas["frame_idx"]

    @property
    def cur_travel_id(self):
        return self.cur_cam_metas["travel_id"]

    def _record_cur_cam_meta(self, cam_metadata: dict):
        self.cur_cam_metas["frame_idx"] = cam_metadata.get("frame_idx")
        self.cur_cam_metas["travel_id"] = cam_metadata.get("travel_id")

    def get_gaussians(self, visible_tokens=None, **kwargs):
        collected_gaussians = {}
        gs_dict = {
            "means": [],
            "scales": [],
            "quats": [],
            "opacities": [],
            "model_id": []
        }
        if kwargs.get("return_features", False):
            gs_dict["features_dc"] = []
            gs_dict["features_rest"] = []
        else:
            gs_dict["rgbs"] = []

        if kwargs.get("return_v", False):
            gs_dict["velocities"] = []

        if visible_tokens is None:
            visible_tokens = self.gaussian_models.keys()
        for model_name in visible_tokens:
            if model_name not in self.gaussian_models:
                gaussian_model = self.gaussian_models[self.get_submodel_name(model_name)]
                model_id = self.model_ids[self.get_submodel_name(model_name)]
            else:
                gaussian_model = self.gaussian_models[model_name]
                model_id = self.model_ids[model_name]
            if gaussian_model.model_type == "multicolor" and "multicolor_travel_id" in kwargs:
                multicolor_kwargs = kwargs.copy()
                multicolor_kwargs["travel_id"] = kwargs["multicolor_travel_id"]
                gs = gaussian_model.get_gaussians(**multicolor_kwargs)
            else:
                gs = gaussian_model.get_gaussians(**kwargs)
            if gs is None:
                continue
            for k, _ in gs.items():
                gs_dict[k].append(gs[k])

            if "rbgs" in gs:
                if torch.isnan(gs['rgbs']).any() or torch.isinf(gs['rgbs']).any():
                    print(f"NaN or inf in rgbs for model {model_name}")

            gs_dict["model_id"].append(
                torch.full((gs["means"].shape[0],), model_id, device=self.device)
            )

        for key, value in gs_dict.items():
            collected_gaussians[key] = torch.cat(value, dim=0)

        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(collected_gaussians['means']).squeeze()
            collected_gaussians = {k: v[crop_ids] for k, v in collected_gaussians.items()}

        return collected_gaussians

    def get_gaussian_params(self, travel_id, frame_idx):
        collected_gaussians = {}
        gs_dict = {
            "means": [],
            "scales": [],
            "quats": [],
            "features_dc": [],
            "features_rest": [],
            "opacities": [],
            "model_id": []
        }

        for model_name, gaussian_model in self.gaussian_models.items():
            gs = gaussian_model.get_gaussian_params(
                travel_id=travel_id,
                frame_idx=frame_idx 
            )
            if gs is None:
                continue
            for k, _ in gs.items():
                gs_dict[k].append(gs[k])

            model_id = self.model_ids[model_name]
            gs_dict["model_id"].append(
                torch.full((gs["means"].shape[0],), model_id, device=self.device)
            )

        for key, value in gs_dict.items():
            collected_gaussians[key] = torch.cat(value, dim=0)

        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(collected_gaussians['means']).squeeze()
            collected_gaussians = {k: v[crop_ids] for k, v in collected_gaussians.items()}

        return collected_gaussians

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        for model_name, gaussian_model in self.gaussian_models.items():
            param_groups.update(gaussian_model.get_param_groups())

        self.camera_optimizer.get_param_groups(param_groups=param_groups)
        self.appearance_model.get_param_groups(param_groups=param_groups)

        if self.config.use_wild_gaussians:
            param_groups["appearance_embedding"] = list(self.camera_embedding.parameters())
            param_groups["appearance_mlp"] = list(self.wild_MLP.parameters())

        return param_groups

    @staticmethod
    def get_empty_outputs(width: int, height: int, background: torch.Tensor, travel_id: int = None) -> Dict[str, Union[torch.Tensor, List]]:
        rgb = background.repeat(height, width, 1)
        depth = background.new_ones(*rgb.shape[:2], 1) * 10
        accumulation = background.new_zeros(*rgb.shape[:2], 1)
        return {
            "rgb": rgb, 
            "depth": depth, 
            "accumulation": accumulation, 
            "background": background,
            "travel_id": travel_id
        }

    def _get_gaussian_camera_space_normals(self, gaussians, camera: Cameras):
        normals = F.one_hot(
            torch.argmin(gaussians['scales'], dim=-1), num_classes=3
        ).float()
        rots = quat_to_rotmat(gaussians['quats'])
        normals = torch.bmm(rots, normals[:, :, None]).squeeze(-1)
        normals = F.normalize(normals, dim=1)

        viewdirs = (
            -gaussians['means'].detach() + camera.camera_to_worlds.detach()[..., :3, 3]
        )
        viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
        dots = (normals * viewdirs).sum(-1)
        negative_dot_indices = dots < 0
        normals[negative_dot_indices] = -normals[negative_dot_indices]

        # convert normals from world space to camera space
        normals = normals @ camera.camera_to_worlds.squeeze(0)[:3, :3]

        return normals

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        assert camera.shape[0] == 1, "Only one camera at a time"
        camera = copy.deepcopy(camera)

        if camera.metadata is None:
            camera.metadata = {}
        if camera.times is not None:
            camera.metadata["timestamp"] = camera.times.item()

        travel_id = camera.metadata.get("travel_id")
        camera.metadata["travel_id"] = travel_id
        frame_idx = camera.metadata.get("frame_idx")
        if frame_idx is None:
            frame_token = camera.metadata.get("frame_token")
            frame_idx = self.frame_token2frame_idx.get(frame_token)
        camera.metadata["frame_idx"] = frame_idx
        cam_idx = camera.metadata.get("cam_idx")
        if cam_idx is None:
            cam_token = camera.metadata.get("cam_token")
            cam_idx = self.cam_token2cam_idx.get(cam_token)
        camera.metadata["cam_idx"] = cam_idx

        linear_velocity = camera.metadata.get("linear_velocity", np.array([0, 0, 0], dtype=np.float32))
        angular_velocity = camera.metadata.get("angular_velocity", np.array([0, 0, 0], dtype=np.float32))
        linear_velocity = torch.from_numpy(linear_velocity).float().to(self.device)[None]
        angular_velocity = torch.from_numpy(angular_velocity).float().to(self.device)[None]
        camera.metadata["linear_velocity"] = linear_velocity
        camera.metadata["angular_velocity"] = angular_velocity

        self._record_cur_cam_meta(camera.metadata)

        if camera.metadata.get("cam_idx") is not None:
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)[0, ...]
        else:
            optimized_camera_to_world = camera.camera_to_worlds[0, ...]

        if camera.metadata.get("cam_idx") is not None:
            camera_idx = torch.tensor([camera.metadata["cam_idx"]], dtype=torch.long, device=camera.device)
        else:
            camera_idx = None

        get_gaussians_kwargs = camera.metadata.copy()
        get_gaussians_kwargs["return_features"] = self.config.use_wild_gaussians
        get_gaussians_kwargs["return_v"] = False

        collected_gaussians = self.get_gaussians(
            camera_to_worlds=optimized_camera_to_world, **get_gaussians_kwargs
        )
        if collected_gaussians['means'].shape[0] == 0:
            return self.get_empty_outputs(
                int(camera.width.item()), int(camera.height.item()), self.background_color, travel_id
            )
        self.collected_gaussians = collected_gaussians

        # shift the camera to center of scene looking at center
        R = optimized_camera_to_world[:3, :3]  # 3 x 3
        T = optimized_camera_to_world[:3, 3:4]  # 3 x 1
        # flip the z and y axes to align with gsplat conventions
        R_edit = torch.diag(torch.tensor([1, -1, -1], device=self.device, dtype=R.dtype))
        R = R @ R_edit
        # analytic matrix inverse to get world2camera matrix
        R_inv = R.T
        T_inv = -R_inv @ T
        viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
        viewmat[:3, :3] = R_inv
        viewmat[:3, 3:4] = T_inv
        viewmat = viewmat.unsqueeze(0)

        W, H = int(camera.width.item()), int(camera.height.item())
        background = self.background_color

        if self.config.output_depth_during_training or not self.training:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"

        if self.config.use_wild_gaussians:
            rgb = torch.clamp(SH2RGB(collected_gaussians["features_dc"]), 0.0, 1.0)
            features_rest = collected_gaussians["features_rest"].view(collected_gaussians["features_rest"].shape[0], -1)[:, :24]
            if camera_idx is not None:
                camera_embedding = self.camera_embedding(camera_idx).expand(collected_gaussians["features_rest"].shape[0], -1)   # [1, camera_embedding_dim]
            else:
                camera_embedding = rgb.new_zeros(rgb.shape[0], 32)
            mlp_input = torch.cat([rgb, features_rest, camera_embedding], dim=-1)
            offset, mul = torch.split(self.wild_MLP(mlp_input) * 0.01, [3, 3], dim=-1)
            render_colors = rgb * (1.0 + mul) + offset
        else:
            render_colors = collected_gaussians["rgbs"]

        if self.config.predict_normals:
            normals = self._get_gaussian_camera_space_normals(collected_gaussians, camera)
            render_colors = torch.cat([render_colors, normals], dim=-1)

        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default
        gsplat_kwargs = dict(
            means=collected_gaussians['means'],
            quats=collected_gaussians['quats'],
            scales=collected_gaussians['scales'],
            opacities=collected_gaussians['opacities'],
            colors=render_colors,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=camera.get_intrinsics_matrices().cuda(),  # [1, 3, 3]
            width=W,
            height=H,
            tile_size=BLOCK_WIDTH,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sparse_grad=False,
            absgrad=self.config.control.use_abs_grad,
            rasterize_mode=self.config.rasterize_mode,
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )
        render, alpha, info = rasterization(**gsplat_kwargs)
        if info["radii"].ndim == 3:  # [1, N, 2] in glsplat
            info["radii"] = (info["radii"][..., 0] * info["radii"][..., 1]).sqrt().int()

        if self.training and info["means2d"].requires_grad:
            info["means2d"].retain_grad()
        self.xys = info["means2d"]  # [1, N, 2]
        self.radii = info["radii"]  # [1, N]
        self.last_size = (H, W)

        rgb = render[..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)
        rgb = rgb.squeeze(0)

        rgb_appearance = self.appearance_model(rgb, camera_idx)

        if render_mode == "RGB+ED":
            depth_im = render[..., -1:]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
        else:
            depth_im = None

        if self.config.predict_normals:
            start_ch_idx = 3
            normals_im = render[..., start_ch_idx:start_ch_idx+3].squeeze(0)
            normals_im = normals_im / normals_im.norm(dim=-1, keepdim=True)
            normals_im = (normals_im + 1) / 2
        else:
            normals_im = None

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        render_results = {
            "rgb": rgb,  # [H, W, 3]
            "rgb_appearance": rgb_appearance,  # [H, W, 3]
            "depth": depth_im,  # [H, W, 1]
            "normal": normals_im,  # [H, W, 3]
            "accumulation": alpha.squeeze(0),  # [H, W, 1]
            "background": background,  # [3]
            "travel_id": travel_id,
            "means2D": info["means2d"], # [1, N, 2]
            "radii": info["radii"], # [1, N]
            "camera": camera,
        }  # type: ignore

        return render_results

    def forward(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        return self.get_outputs(camera)

    def get_gt_img(self, image: torch.Tensor):
        """Compute groundtruth image with iteration dependent downscale factor for evaluation purpose

        Args:
            image: tensor.Tensor in type uint8 or float32
        """
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        return image.to(self.device)

    def composite_with_background(self, image, background) -> torch.Tensor:
        """Composite the ground truth image with a background color when it has an alpha channel.

        Args:
            image: the image to composite
            background: the background color
        """
        if image.shape[2] == 4:
            alpha = image[..., -1].unsqueeze(-1).repeat((1, 1, 3))
            return alpha * image[..., :3] + (1 - alpha) * background
        else:
            return image

    def combine_visible_mask(self, pred, gt, batch) -> torch.Tensor:
        combined_mask = torch.ones(gt.shape[:2], device=self.device, dtype=torch.bool).unsqueeze(-1)

        if "mask" in batch:
            # batch["mask"] : [H, W, 1]
            mask = batch["mask"].to(self.device)
            assert mask.shape[:2] == gt.shape[:2] == pred.shape[:2]
            combined_mask = mask & combined_mask

        return combined_mask

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics for TRAIN set.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        metrics_dict = {}
        pred_img = outputs["rgb"]
        pred_img_appearance = outputs["rgb_appearance"]
        mask = self.combine_visible_mask(pred_img, gt_rgb, batch=batch)
        if mask.sum() == 0:
            assert "no visible mask for the camera."

        if self.config.color_corrected_metrics:
            if mask is not None:
                cc_rgb = color_correct(pred_img_appearance*mask, gt_rgb*mask)
            else:
                cc_rgb = color_correct(pred_img_appearance, gt_rgb)
            cc_rgb = torch.moveaxis(cc_rgb, -1, 0)[None, ...]

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        pred_img = torch.moveaxis(pred_img, -1, 0)[None, ...]
        pred_img_appearance = torch.moveaxis(pred_img_appearance, -1, 0)[None, ...]

        metrics_dict["psnr"] = self.psnr(gt_rgb, pred_img_appearance, mask=mask)
        if self.config.color_corrected_metrics:
            metrics_dict["cc_psnr"] = self.psnr(cc_rgb, gt_rgb, mask=mask)
        metrics_dict["ssim"] = self.ssim(gt_rgb, pred_img, mask=mask)

        if self.config.lpips_metric:
            mask_nchw = torch.moveaxis(mask, -1, 0)[None, ...]
            lpips = self.lpips(gt_rgb * mask_nchw, pred_img_appearance * mask_nchw)
            metrics_dict["lpips"] = float(lpips)

        if self.config.dinov2_metric:
            dinov2_metric = self.dinov2.similarity(gt_rgb, pred_img_appearance, mask)
            metrics_dict["dinov2_sim"] = float(dinov2_metric)

        if 'lidar_depth' in batch and 'depth' in outputs:
            gt_depth = batch['lidar_depth'].to(self.device)
            pred_depth = outputs['depth']
            lidar_mask = (gt_depth > 0.1) & (gt_depth < 80)
            lidar_mask = lidar_mask & mask
            pred_depth = pred_depth[lidar_mask]
            gt_depth = gt_depth[lidar_mask]
            error = gt_depth - pred_depth
            metrics_dict["depth_RMSE"] = torch.sqrt((error ** 2).mean())
            metrics_dict["depth_absRel"] = (error.abs() / gt_depth).mean()
            metrics_dict["depth_delta1"] = (torch.max(pred_depth / gt_depth, gt_depth / pred_depth) < 1.25).float().mean()

        metrics_dict["gaussian_count"] = self.collected_gaussians["means"].shape[0]

        self.camera_optimizer.get_metrics_dict(metrics_dict)

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["rgb"]
        pred_img_appearance = outputs["rgb_appearance"]
        camera = outputs["camera"]

        # RGB Loss: PSNR & SSIM.
        # Set masked part of both ground-truth and rendered image to black.
        # Note that here the mask is a combined mask showing all visible objects in the camera.
        # This is a little bit sketchy for the SSIM loss.
        combined_mask = self.combine_visible_mask(pred_img, gt_img, batch=batch)

        l1_Loss = torch.abs(gt_img - pred_img_appearance)[combined_mask.squeeze(-1)].mean()
        loss_dict = {
            "L1 Loss": (1 - self.config.ssim_lambda) * l1_Loss
        }
        if self.config.ssim_lambda > 0:
            if self.config.use_ssim_on_raw_rgb:
                ssim_loss = 1 - self.ssim(
                    gt_img.permute(2, 0, 1)[None, ...], 
                    pred_img.permute(2, 0, 1)[None, ...],
                    mask=combined_mask
                )
            else:
                ssim_loss = 1 - self.ssim(
                    gt_img.permute(2, 0, 1)[None, ...], 
                    pred_img_appearance.permute(2, 0, 1)[None, ...],
                    mask=combined_mask
                )
            loss_dict["SSIM Loss"] = self.config.ssim_lambda * ssim_loss

        if self.config.output_depth_during_training:
            # batch["depth"] : [H, W, 1]
            if self.config.depth_source == "pseudo":
                gt_depth = batch["depth"].to(self.device)
                depth_loss_mask = (gt_depth > 0.1) & (gt_depth < 50)
            elif self.config.depth_source == "lidar":
                gt_depth = batch["lidar_depth"].to(self.device)
                depth_loss_mask = (gt_depth > 0.1) & (gt_depth < 80)
            else:
                raise ValueError(f"Unknown depth source: {self.config.depth_source}")

            depth_loss_mask = depth_loss_mask & combined_mask
 
            if depth_loss_mask.sum() == 0:
                loss_dict["Depth Loss"] = torch.tensor(0.0, device=self.device)
            else:
                if self.config.use_depth_loss:
                    pred_depth = outputs["depth"].to(self.device)
                    if self.config.depth_source == "pseudo":
                        # use mono dpeth to compute depth loss
                        if self.config.depth_loss_type == DepthLossType.EdgeAwareLogL1:
                            depth_loss = self.depth_loss(
                                pred_depth, gt_depth.float(), gt_img, depth_loss_mask
                            )
                            loss_dict['Depth Loss'] = self.config.depth_lambda * depth_loss
                        else:
                            depth_loss = self.depth_loss(
                               pred_depth[depth_loss_mask], gt_depth[depth_loss_mask].float()
                            )
                            loss_dict['Depth Loss'] = self.config.depth_lambda * depth_loss

                    elif self.config.depth_source == "lidar":
                        if self.config.depth_loss_type == DepthLossType.InverseL1:
                            inverse_gt_depth = 1 / (gt_depth + 1e-5)
                            inverse_pred_depth = 1 / (pred_depth + 1e-5)
                            loss_depth = torch.abs(inverse_gt_depth - inverse_pred_depth)[depth_loss_mask].mean()
                        elif self.config.depth_loss_type == DepthLossType.L1:
                            loss_depth = torch.abs(gt_depth - pred_depth)[depth_loss_mask].mean()
                        else:
                            raise NotImplementedError(f"Lidar depth loss type {self.config.depth_loss_type} not implemented")
                        loss_dict["Depth Loss"] = self.config.depth_lambda * loss_depth

            if self.config.ncc_loss_lambda > 0:
                patch_size = self.config.ncc_patch_size
                stride = self.config.ncc_stride
                pred_depth = outputs["depth"].to(self.device)
                gt_depth = batch["depth"].to(self.device)
                depth_loss_mask = (gt_depth > 0.1) & (gt_depth < 80) & combined_mask
                if depth_loss_mask.sum() != 0:
                    ncc_loss = calculate_depth_ncc_loss(pred_depth, gt_depth, patch_size, stride, mask=depth_loss_mask)
                    loss_dict['ncc_loss'] = ncc_loss * self.config.ncc_loss_lambda

        # Normal Loss
        if self.config.use_normal_loss:
            normal_loss = 0.0
            pred_normal = outputs["normal"].to(self.device)

            if "normal" in batch and self.config.normal_supervision == "mono":
                gt_normal = batch["normal"].to(self.device)
                normal_loss_mask = combined_mask.squeeze(-1)

            elif self.config.normal_supervision == "depth":
                gt_depth = batch["depth"].to(self.device)
                normal_loss_mask = (gt_depth > 0.1) & (gt_depth < 50)
                normal_loss_mask = normal_loss_mask & combined_mask

                normal_loss_mask = normal_loss_mask.squeeze(-1)
                camera = outputs["camera"].to(self.device)
                gt_normal = normal_from_depth_image(
                    depths=gt_depth.detach(),
                    fx=camera.fx.item(),
                    fy=camera.fy.item(),
                    cx=camera.cx.item(),
                    cy=camera.cy.item(),
                    img_size=(camera.width.item(), camera.height.item()),
                    c2w=torch.eye(4, dtype=torch.float, device=self.device),
                    device=self.device,
                    smooth=False,
                )
                gt_normal = gt_normal @ torch.diag(
                    gt_normal.new_tensor([1, -1, -1])
                )
                gt_normal = (1 + gt_normal) / 2

            if gt_normal is not None:
                normal_loss = torch.abs(gt_normal - pred_normal)[normal_loss_mask].mean()

            if self.config.use_normal_tv_loss:
                normal_loss += self.tv_loss(pred_normal)

            if torch.isfinite(normal_loss):
                loss_dict["Normal Loss"] = self.config.normal_lambda * normal_loss

        if self.config.two_d_gaussians:
            # loss to minimise gaussian scale corresponding to normal direction
            two_d_reg_loss = torch.min(self.collected_gaussians['scales'], dim=1, keepdim=True)[0].mean()
            loss_dict["2D reg Loss"] = two_d_reg_loss

        if self.config.adapter_lambda != 0.0:
            adapter_loss = 0.
            for model in self.gaussian_models.values():
                if model.config.model_type == "multicolor":
                    adapter_loss += torch.abs(model.features_adapters).sum()
            loss_dict["Adapter Loss"] = self.config.adapter_lambda * adapter_loss

        if self.config.oob_lambda != 0.0:
            oob_loss = 0.0
            overall_gaussians = 0
            visible_mask = (outputs['radii'] > 0).flatten()
            for model_name, model in self.gaussian_models.items():
                if isinstance(model, RigidSubModel):
                    model_mask = self.collected_gaussians['model_id'] == self.model_ids[model_name]
                    if visible_mask[model_mask].sum() == 0:
                        continue
                    means = model.means
                    instance_size = means.new_tensor(model.instance_size)
                    oob_mask = means.abs() > (instance_size / 2 + self.config.oob_tolerance)[None]
                    oob_mask = oob_mask.any(-1).detach()
                    if oob_mask.sum() != 0:
                        oob_loss += -torch.log(1 - model.opacities[oob_mask].sigmoid() + 1e-6).sum()
                        overall_gaussians += oob_mask.sum()
            if overall_gaussians != 0:
                oob_loss = oob_loss / overall_gaussians
                loss_dict["OOB Loss"] = self.config.oob_lambda * oob_loss

        if self.config.sharp_shape_reg_lambda != 0.0:
            max_gauss_ratio = self.config.sharp_shape_reg_max_gauss_ratio
            step_interval = self.config.sharp_shape_reg_step_interval
            if self.step % step_interval == 0:
                # scale regularization
                scale_exp = self.collected_gaussians['scales']
                if self.config.two_d_gaussians:
                    sorted_scales, _ = torch.sort(scale_exp, dim=-1, descending=True)
                    scale_reg = torch.maximum(sorted_scales[..., 0] / sorted_scales[..., 1], torch.tensor(max_gauss_ratio)) - max_gauss_ratio
                else:
                    scale_reg = torch.maximum(scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1), torch.tensor(max_gauss_ratio)) - max_gauss_ratio
                scale_reg = scale_reg.mean() * self.config.sharp_shape_reg_lambda
                loss_dict["Sharp Shape Reg Loss"] = scale_reg

        if self.training:
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)

        return loss_dict

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)

        outs = self.get_outputs(camera.to(self.device))
        return outs  # type: ignore

    def travel_metric(self, metric, travel_id, filter_travel_ids):
        if isinstance(filter_travel_ids, int):
            filter_travel_ids = [filter_travel_ids]
        if travel_id in filter_travel_ids:
            return metric
        return torch.nan

    def get_image_metrics_and_images(
        self, 
        outputs: Dict[str, torch.Tensor], 
        batch: Dict[str, torch.Tensor], 
        travel_id_set: Optional[List[int]] = None,
        return_image_dict = True
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.
        for EVAL processs during training.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        predicted_rgb = outputs["rgb_appearance"].to(self.device)

        mask = self.combine_visible_mask(predicted_rgb, gt_rgb, batch=batch)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        if self.config.color_corrected_metrics:
            if mask is not None:
                cc_rgb = color_correct(predicted_rgb * mask, gt_rgb * mask)
            else:
                cc_rgb = color_correct(predicted_rgb, gt_rgb)

            cc_rgb = torch.moveaxis(cc_rgb, -1, 0)[None, ...]
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb, mask=mask)
        ssim = self.ssim(gt_rgb, predicted_rgb, mask=mask)
        psnr, ssim = float(psnr.item()), float(ssim.item())
        metrics_dict = {"psnr": psnr, "ssim": ssim}

        if self.config.color_corrected_metrics:
            cc_psnr = self.psnr(gt_rgb, cc_rgb, mask=mask)
            metrics_dict["cc_psnr"] = float(cc_psnr.item())
        
        if 'lidar_depth' in batch and 'depth' in outputs:
            gt_depth = batch['lidar_depth'].to(self.device)
            pred_depth = outputs['depth']
            lidar_mask = (gt_depth > 0.1) & (gt_depth < 80)
            lidar_mask = lidar_mask & mask
            pred_depth = pred_depth[lidar_mask]
            gt_depth = gt_depth[lidar_mask]
            error = gt_depth - pred_depth
            metrics_dict["depth_RMSE"] = torch.sqrt((error ** 2).mean())
            metrics_dict["depth_absRel"] = (error.abs() / gt_depth).mean()
            metrics_dict["depth_delta1"] = (torch.max(pred_depth / gt_depth, gt_depth / pred_depth) < 1.25).float().mean()

        if self.config.lpips_metric:
            mask_nchw = mask.permute(2, 0, 1).unsqueeze(0)
            lpips = self.lpips(gt_rgb * mask_nchw, predicted_rgb * mask_nchw)
            metrics_dict["lpips"] = float(lpips)

        if self.config.dinov2_metric:
            dinov2_metric = self.dinov2.similarity(gt_rgb, predicted_rgb, mask)
            metrics_dict["dinov2_sim"] = float(dinov2_metric)

        # all of these metrics will be logged as scalars
        travel_id = outputs["travel_id"]
        if travel_id is not None and travel_id_set is not None:
            if not isinstance(travel_id, int) and travel_id.ndim > 0:
                travel_id = travel_id.squeeze()       # in fact one camera at a time
            for idx in travel_id_set:
                metrics_dict[f"trv{idx}_psnr"] = self.travel_metric(psnr, travel_id, idx)
                metrics_dict[f"trv{idx}_ssim"] = self.travel_metric(ssim, travel_id, idx)
                if self.config.color_corrected_metrics:
                    metrics_dict[f"trv{idx}_cc_psnr"] = self.travel_metric(cc_psnr, travel_id, idx)
                if self.config.lpips_metric:
                    metrics_dict[f"trv{idx}_lpips"] = self.travel_metric(metrics_dict["lpips"], travel_id, idx)
                if self.config.dinov2_metric:
                    metrics_dict[f"trv{idx}_dinov2_sim"] = self.travel_metric(metrics_dict["dinov2_sim"], travel_id, idx)
                if 'depth_RMSE' in metrics_dict:
                    metrics_dict[f"trv{idx}_depth_RMSE"] = self.travel_metric(metrics_dict["depth_RMSE"], travel_id, idx)
                    metrics_dict[f"trv{idx}_depth_absRel"] = self.travel_metric(metrics_dict["depth_absRel"], travel_id, idx)
                    metrics_dict[f"trv{idx}_depth_delta1"] = self.travel_metric(metrics_dict["depth_delta1"], travel_id, idx)

        images_dict = {}
        if return_image_dict:
            images_dict['image'] = outputs["rgb_appearance"]
            images_dict['gt_image'] = self.get_gt_img(batch["image"])

            if self.config.output_depth_during_training:
                depth_color = colormaps.apply_depth_colormap(outputs['depth'], near_plane=0.1, far_plane=100)
                images_dict["depth"] = depth_color

            if self.config.predict_normals:
                camera = outputs["camera"].to(outputs["normal"].device)
                combined_normal = outputs["normal"]
                if "depth" in batch:
                    gt_normal = normal_from_depth_image(
                        depths=batch["depth"],
                        fx=camera.fx.item(),
                        fy=camera.fy.item(),
                        cx=camera.cx.item(),
                        cy=camera.cy.item(),
                        img_size=(camera.width.item(), camera.height.item()),
                        c2w=torch.eye(4).to(outputs["normal"]),
                        device=outputs["normal"].device,
                        smooth=False,
                    )
                    gt_normal = gt_normal @ torch.diag(outputs["normal"].new_tensor([1, -1, -1]))
                    gt_normal = (1 + gt_normal) / 2
                    combined_normal = torch.cat([gt_normal, combined_normal], dim=1)
                images_dict["normal"] = combined_normal

        return metrics_dict, images_dict

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []

        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                self.step_cb
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.update_submodel_statistics,
            )
        )

        for model_name, model in self.gaussian_models.items():
            cbs.extend(model.get_training_callbacks(training_callback_attributes))
        return cbs

    def step_cb(self, step):
        self.step = step

    def set_crop(self, crop_box: Optional[OrientedBox]):
        self.crop_box = crop_box

    def set_background(self, background_color: torch.Tensor):
        assert background_color.shape == (3,)
        self.bg_color = background_color.to(self.device)

    def update_submodel_statistics(self, step):
        for model_name, gaussian_model in self.gaussian_models.items():
            model_id = self.model_ids[model_name]
            submodel_mask = self.collected_gaussians["model_id"] == model_id

            if submodel_mask.sum() == 0:
                gaussian_model.update_statistics(
                    xys_grad=None,
                    radii=None
                )
                continue

            with torch.no_grad():
                assert self.xys.grad is not None
                if gaussian_model.config.control.use_abs_grad:
                    grads = self.xys.absgrad[0, submodel_mask].detach()
                else:
                    grads = self.xys.grad[0, submodel_mask].detach()

                image_size = grads.new_tensor([self.last_size[1], self.last_size[0]]).unsqueeze(0)
                grads = (grads * image_size * 0.5).norm(dim=-1)
                radii = self.radii[0, submodel_mask]

            gaussian_model.update_statistics(
                xys_grad=grads,
                radii=radii
            )

    def load_state_dict(self, dict: dict, **kwargs):  # type: ignore
        sub_model_state_dict = {model_name: {} for model_name in self.gaussian_models.keys()}

        for key, value in dict.items():
            if key.startswith("gaussian_models."):
                model_name = key.replace("gaussian_models.", "")
                model_name, subkey = model_name.split(".", maxsplit=1)
                if model_name not in self.gaussian_models.keys():
                    print("Skipping loading of submodel", model_name)
                    continue
                sub_model_state_dict[model_name][subkey] = value

        for model_name, model in self.gaussian_models.items():
            instance_trans = sub_model_state_dict[model_name].get("instance_trans", None)
            if (instance_trans is not None) and (instance_trans.dim() > 1) and (model.num_frames != instance_trans.shape[0]):
                CONSOLE.print("[WARNING] Mismatch in number of frames for", model_name, ". Loading as static object, placed at the position in the first frame.")
                in_frame_mask = instance_trans[:, 2] < 10.
                sub_model_state_dict[model_name]['instance_quats'] = sub_model_state_dict[model_name]['instance_quats'][in_frame_mask][0]
                sub_model_state_dict[model_name]['instance_trans'] = instance_trans[in_frame_mask][0]
            model.load_state_dict(sub_model_state_dict[model_name], **kwargs)

        other_state_dict = {key: value for key, value in dict.items() if not key.startswith("gaussian_models.")}
        if 'camera_optimizer.pose_adjustment' in other_state_dict:
            if len(other_state_dict['camera_optimizer.pose_adjustment']) != self.num_train_data:
                CONSOLE.print("[WARNING] Mismatch in number of frames for camera optimizer and appearance model. Ignoring.")
                other_state_dict.pop('camera_optimizer.pose_adjustment')
                other_state_dict.pop('appearance_model.exposure_factor')
        if 'strict' in kwargs:
            kwargs.pop('strict')

        return super().load_state_dict(other_state_dict, strict=False, **kwargs)

    def update_to_step(self, step: int) -> None:
        self.step = step
        for model_name, model in self.gaussian_models.items():
            model.step_cb(step)
