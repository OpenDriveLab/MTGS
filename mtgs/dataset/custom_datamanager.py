#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Dict, ForwardRef, Generic, List, Literal, Optional, Tuple, Type, Union, cast, get_args, get_origin

import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datamanagers.base_datamanager import TDataset
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig, FullImageDatamanager
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.utils.misc import get_orig_class
from nerfstudio.utils.rich_utils import CONSOLE

from .custom_dataset import CustomInputDataset
from .utils.dataloader import AsyncDataLoader, FixedIndicesPseudoDataloader, OnDemandDataLoader, PrefetchDataLoader
from .utils.sampler import CameraSampler, MultiTraversalBalancedSampler

@dataclass
class CustomFullImageDatamanagerConfig(FullImageDatamanagerConfig):
    _target: Type = field(default_factory=lambda: CustomFullImageDatamanager[CustomInputDataset])
    camera_res_scale_factor: float = 1.0
    """The scale factor for scaling spatial data such as images, mask, semantics
    along with relevant information about camera intrinsics
    """
    dynamic_scaling_factor_stages: int = 0
    """The number of stages of dynamic scaling factor"""
    dynamic_scaling_factor_interval: int = 3000

    load_mask: bool = False
    load_custom_masks: Optional[Tuple[str]] = field(default_factory=lambda: ())     # vehicle, pedestrian, bicycle
    load_instance_masks: bool = False
    load_semantic_masks_from: Literal["panoptic", "semantic", False] = False
    load_lidar_depth: bool = False
    load_pseudo_depth: bool = False
    cache_strategy: Literal["on_demand", "prefetch", "async"] = "async"
    eval_cache_strategy: Optional[Literal["on_demand", "prefetch", "async"]] = None
    num_workers: int = 4
    cache_images: Literal["cpu", "gpu"] = "cpu"
    """CPU by default, too much images"""
    cache_images_type: Literal["uint8", "float32"] = "uint8"
    """The image type returned from manager, caching images in uint8 saves memory"""
    crop_image: bool = False
    """whether to crop the image to 1920x1024, to match some CNN input size"""
    undistort_images: Literal["optimal", "keep_focal_length", False] = "keep_focal_length"
    multi_traversal_balanced_sampling: bool = False


class CustomFullImageDatamanager(FullImageDatamanager, Generic[TDataset]):
    """
    A datamanager that outputs full images and cameras instead of raybundles. This makes the
    datamanager more lightweight since we don't have to do generate rays. Useful for full-image
    training e.g. rasterization pipelines
    """

    config: CustomFullImageDatamanagerConfig

    DATALOADER_MAP = {
        "on_demand": OnDemandDataLoader,
        "prefetch": PrefetchDataLoader,
        "async": AsyncDataLoader,
    }

    def __init__(
        self,
        config: CustomFullImageDatamanagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"
        self.dataparser_config = self.config.dataparser

        assert self.dataparser_config.undistort_images == self.config.undistort_images

        if self.config.data is not None:
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data
        self.dataparser = self.dataparser_config.setup()
        if test_mode == "inference":
            self.dataparser.downscale_factor = 1  # Avoid opening images
        self.includes_time = self.dataparser.includes_time

        self.train_dataparser_outputs: DataparserOutputs = self.dataparser.get_dataparser_outputs(split="train")
        self.eval_dataparser_outputs: DataparserOutputs = self.dataparser.get_dataparser_outputs(split=self.test_split)

        if set(self.train_dataparser_outputs.image_filenames) == set(self.eval_dataparser_outputs.image_filenames):
            CONSOLE.log("Train and eval images are the same.")
            self.same_train_eval = True
        else:
            self.same_train_eval = False

        train_cache_strategy = eval_cache_strategy = self.config.cache_strategy
        if self.config.eval_cache_strategy is not None:
            CONSOLE.log(f"Using {self.config.eval_cache_strategy} cache strategy for evaluation")
            eval_cache_strategy = self.config.eval_cache_strategy

        if test_mode in ["inference", "test"]:
            if not self.same_train_eval:
                train_cache_strategy = "on_demand"
                eval_cache_strategy = "prefetch"
            else:
                train_cache_strategy = eval_cache_strategy = "prefetch"
            self.config.load_pseudo_depth = False

        if test_mode == "inference":
            self.config.load_mask = False
            self.config.load_custom_masks = ()
            self.config.load_instance_masks = False
            self.config.load_semantic_masks_from = False
            self.config.load_lidar_depth = False

        self.scaling_factors = []
        for i in range(self.config.dynamic_scaling_factor_stages + 1):
            scaling_factor = self.config.camera_res_scale_factor / (2 ** (self.config.dynamic_scaling_factor_stages - i))
            self.scaling_factors.append(scaling_factor)

        CONSOLE.log(f"Scaling factors: {self.scaling_factors}")
        self.train_dataset = self.create_train_dataset()
        self.train_dataloaders = []
        for scaling_factor in self.scaling_factors:
            self.train_dataloaders.append(self.DATALOADER_MAP[train_cache_strategy](self, self.train_dataset, scaling_factor))

        if not self.same_train_eval:
            self.eval_dataset = self.create_eval_dataset()
            self.eval_dataloader = self.DATALOADER_MAP[eval_cache_strategy](self, self.eval_dataset, self.scaling_factors[-1])
        else:
            self.eval_dataset = self.train_dataset
            self.eval_dataloader = self.train_dataloaders[-1]

        # Some logic to make sure we sample every camera in equal amounts
        assert len(self.train_dataset) > 0, "No data found in dataset"
        if self.config.multi_traversal_balanced_sampling:
            self.train_sampler = MultiTraversalBalancedSampler(self.train_dataset)
        else:
            self.train_sampler = CameraSampler(self.train_dataset)

        self.eval_sampler = CameraSampler(self.eval_dataset, in_order=True)
        self._fixed_indices_eval_dataloader = FixedIndicesPseudoDataloader(self)

        super(FullImageDatamanager, self).__init__()

    def create_train_dataset(self) -> TDataset:
        """Sets up the data loaders for training"""
        return self.dataset_type(
            dataparser_outputs=self.train_dataparser_outputs,
            fake_data=self.test_mode == "inference",
            crop_size=(0, 0, 1920, 1024) if self.config.crop_image else None,
            load_mask=self.config.load_mask,
            load_custom_masks=self.config.load_custom_masks,
            load_instance_masks=self.config.load_instance_masks,
            load_semantic_masks_from=self.config.load_semantic_masks_from,
            load_lidar_depth=self.config.load_lidar_depth,
            load_pseudo_depth=self.config.load_pseudo_depth,
            undistort_images=self.config.undistort_images
        )

    def create_eval_dataset(self) -> TDataset:
        """Sets up the data loaders for evaluation"""
        return self.dataset_type(
            dataparser_outputs=self.eval_dataparser_outputs,
            fake_data=self.test_mode == "inference",
            crop_size=(0, 0, 1920, 1024) if self.config.crop_image else None,
            load_mask=self.config.load_mask,
            load_custom_masks=self.config.load_custom_masks,
            load_instance_masks=self.config.load_instance_masks,
            load_semantic_masks_from=self.config.load_semantic_masks_from,
            load_lidar_depth=self.config.load_lidar_depth,
            load_pseudo_depth=False,   # Never load pseudo depth for evaluation
            undistort_images=self.config.undistort_images,
            eval_dataset=True
        )

    def cached_train(self, idx: int, scale_factor_idx: int):
        return self.train_dataloaders[scale_factor_idx].get_cache(idx)

    def cached_eval(self, idx: int):
        return self.eval_dataloader.get_cache(idx)

    def _load_data(self, dataset, idx, scale_factor: float):
        data, camera = dataset.get_data_and_camera(
            idx, 
            image_type=self.config.cache_images_type,
            scale_factor=scale_factor
        )
        return data, camera

    @cached_property
    def dataset_type(self) -> Type[TDataset]:
        """Returns the dataset type passed as the generic argument"""
        default: Type[TDataset] = cast(TDataset, TDataset.__default__)  # type: ignore
        orig_class: Type[CustomFullImageDatamanager] = get_orig_class(self, default=None)  # type: ignore
        if orig_class is not None and get_origin(orig_class) is CustomFullImageDatamanager:
            return get_args(orig_class)[0]

        # For inherited classes, we need to find the correct type to instantiate
        for base in getattr(self, "__orig_bases__", []):
            if get_origin(base) is CustomFullImageDatamanager:
                for value in get_args(base):
                    if isinstance(value, ForwardRef):
                        if value.__forward_evaluated__:
                            value = value.__forward_value__
                        elif value.__forward_module__ is None:
                            value.__forward_module__ = type(self).__module__
                            value = getattr(value, "_evaluate")(None, None, set())
                    assert isinstance(value, type)
                    if issubclass(value, InputDataset):
                        return cast(Type[TDataset], value)
        return default

    @property
    def fixed_indices_eval_dataloader(self) -> FixedIndicesPseudoDataloader:
        return self._fixed_indices_eval_dataloader

    def get_train_rays_per_batch(self):
        return 1

    def get_scale_factor_idx(self, step: int) -> int:
        """Get the scale factor index based on the step"""
        return min(step // self.config.dynamic_scaling_factor_interval, self.config.dynamic_scaling_factor_stages)
    
    def release_memory(self, step: int):
        if step > 0 and step % self.config.dynamic_scaling_factor_interval == 0:
            last_scale_factor_idx = self.get_scale_factor_idx(step - 1)
            if last_scale_factor_idx < self.config.dynamic_scaling_factor_stages:
                last_dataloader = self.train_dataloaders[last_scale_factor_idx]
                self.train_dataloaders[last_scale_factor_idx] = None
                del last_dataloader

    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next training batch

        Returns a Camera instead of raybundle"""
        image_idx = self.train_sampler.get_next_image_idx()

        data, camera = self.cached_train(
            image_idx, scale_factor_idx=self.get_scale_factor_idx(step)
        )

        self.release_memory(step)
        data["image"] = data["image"].to(self.device)
        camera = camera.to(self.device)
        return camera, data

    def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch

        Returns a Camera instead of raybundle"""
        image_idx = self.eval_sampler.get_next_image_idx()
        data, camera = self.cached_eval(image_idx)
        data["image"] = data["image"].to(self.device)
        camera = camera.to(self.device)
        return camera, data

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch

        Returns a Camera instead of raybundle"""
        return self.next_eval(step)
