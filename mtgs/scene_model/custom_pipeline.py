#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import os
from dataclasses import dataclass, field
from pathlib import Path
import typing
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Type, Union, cast
from time import time
from math import isnan
import numpy as np
from PIL import Image

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
from torch.cuda.amp.grad_scaler import GradScaler
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig, VanillaPipeline, Pipeline
from nerfstudio.data.datamanagers.base_datamanager import DataManagerConfig, VanillaDataManager
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import profiler

from mtgs.utils.camera_utils import invert_distortion

@dataclass
class MultiTravelEvalPipielineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: MultiTravelEvalPipieline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = field(default_factory=DataManagerConfig)
    """specifies the datamanager config"""
    model: ModelConfig = field(default_factory=ModelConfig)
    """specifies the model config"""

    image_saving_mode: Literal["sequential", "sequential_with_gt", "nuplan"] = "sequential"
    """The folder structure of the image output"""

class MultiTravelEvalPipieline(VanillaPipeline):
    def __init__(self,
        config: MultiTravelEvalPipielineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode
        self.datamanager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        if self.config.image_saving_mode == "nuplan":
            assert config.datamanager.camera_res_scale_factor == 1.0
        # TODO make cleaner
        seed_pts = None
        if (
            hasattr(self.datamanager, "train_dataparser_outputs")
            and "points3D_xyz" in self.datamanager.train_dataparser_outputs.metadata
        ):
            pts = self.datamanager.train_dataparser_outputs.metadata["points3D_xyz"]
            pts_rgb = self.datamanager.train_dataparser_outputs.metadata["points3D_rgb"]
            seed_pts = (pts, pts_rgb)
        self.datamanager.to(device)
        # TODO(ethan): get rid of scene_bounds from the model
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
            seed_points=seed_pts,
            datamanager=self.datamanager,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])

    def _save_images(self, camera, images_dict, ouput_path: Path):
        dataparser_outputs = self.datamanager.eval_dataparser_outputs
        if self.config.image_saving_mode == "sequential":
            cam_idx = dataparser_outputs.cam_tokens.index(camera.metadata["cam_token"])
            travel_id = dataparser_outputs.travel_ids[cam_idx]
            cam_name = dataparser_outputs.ego_mask_filenames[cam_idx].split("/")[-1].split(".")[0]

            rendered_image = images_dict["image"].cpu().numpy() * 255.0
            rendered_image = rendered_image.astype(np.uint8)
            rendered_output_path = ouput_path / f"traversal_{travel_id}" / cam_name / f"{cam_idx}_rendered.jpg"
            rendered_output_path.parent.mkdir(parents=True, exist_ok=True)
            rendered_image = Image.fromarray(rendered_image)
            rendered_image.save(rendered_output_path)

        elif self.config.image_saving_mode == "sequential_with_gt":
            cam_idx = dataparser_outputs.cam_tokens.index(camera.metadata["cam_token"])
            travel_id = dataparser_outputs.travel_ids[cam_idx]
            cam_name = dataparser_outputs.ego_mask_filenames[cam_idx].split("/")[-1].split(".")[0]

            rendered_image = images_dict["image"].cpu().numpy() * 255.0
            rendered_image = rendered_image.astype(np.uint8)
            rendered_output_path = ouput_path / f"traversal_{travel_id}" / cam_name / f"{cam_idx}_rendered.jpg"
            rendered_output_path.parent.mkdir(parents=True, exist_ok=True)
            rendered_image = Image.fromarray(rendered_image)
            rendered_image.save(rendered_output_path)

            gt_image = images_dict["gt_image"].cpu().numpy() * 255.0
            gt_image = gt_image.astype(np.uint8)
            gt_output_path = ouput_path / f"traversal_{travel_id}" / cam_name / f"{cam_idx}_gt_processed.jpg"
            gt_output_path.parent.mkdir(parents=True, exist_ok=True)
            gt_image = Image.fromarray(gt_image)
            gt_image.save(gt_output_path)

            gt_image_raw_path = dataparser_outputs.image_filenames[cam_idx].absolute()
            gt_raw_output_path = ouput_path / f"traversal_{travel_id}" / cam_name / f"{cam_idx}_gt.jpg"
            gt_raw_output_path.parent.mkdir(parents=True, exist_ok=True)
            if gt_raw_output_path.exists():
                os.remove(gt_raw_output_path)
            os.symlink(gt_image_raw_path, gt_raw_output_path)

        elif self.config.image_saving_mode == "nuplan":
            cam_idx = dataparser_outputs.cam_tokens.index(camera.metadata["cam_token"])
            intrinsic, distortion = dataparser_outputs.undistort_params[cam_idx]
            gt_image_raw_path = dataparser_outputs.image_filenames[cam_idx]
            nuplan_relative_path = '/'.join(gt_image_raw_path.parts[-3:])        # {log_name}/{cam_name}/{cam_token}.jpg

            rendered_image = images_dict["image"].cpu().numpy() * 255.0
            rendered_image = rendered_image.astype(np.uint8)
            rendered_image = invert_distortion(rendered_image, intrinsic, distortion)
            rendered_image = Image.fromarray(rendered_image)
            rendered_output_path = ouput_path / nuplan_relative_path
            rendered_output_path.parent.mkdir(parents=True, exist_ok=True)
            rendered_image.save(rendered_output_path)

    @profiler.time_function
    def get_average_eval_image_metrics(
        self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False
    ):
        self.eval()
        metrics_dict_list = []
        assert isinstance(self.datamanager, (VanillaDataManager, ParallelDataManager, FullImageDatamanager))
        num_images = len(self.datamanager.fixed_indices_eval_dataloader)
        travel_id_set = None
        if hasattr(self.datamanager.eval_dataparser_outputs, "travel_ids") and self.datamanager.eval_dataparser_outputs.travel_ids is not None:
            travel_ids = self.datamanager.eval_dataparser_outputs.travel_ids
            travel_id_set = list(set(travel_ids))

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            for camera, batch in self.datamanager.fixed_indices_eval_dataloader:
                # time this the following line
                inner_start = time()
                outputs = self.model.get_outputs_for_camera(camera=camera)
                height, width = camera.height, camera.width
                num_rays = height * width
                num_rays_per_sec = (num_rays / (time() - inner_start)).item()

                metrics_dict, images_dict = self.model.get_image_metrics_and_images(
                    outputs, batch, travel_id_set=travel_id_set, return_image_dict=output_path is not None)
                if output_path is not None:
                    self._save_images(camera, images_dict, output_path)

                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = (num_rays_per_sec / (height * width)).item()
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
        # average the metrics list
        eval_metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            true_metrics = [metrics_dict[key] for metrics_dict in metrics_dict_list]
            true_metrics = [m for m in true_metrics if not isnan(m)]
            if get_std:
                key_std, key_mean = torch.std_mean(torch.tensor(true_metrics))
                eval_metrics_dict[key] = float(key_mean)
                eval_metrics_dict[f"{key}_std"] = float(key_std)
            else:
                eval_metrics_dict[key] = float(
                    torch.mean(torch.tensor(true_metrics))
                )
        self.train()
        return eval_metrics_dict

    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
            step: training step of the loaded checkpoint
        """
        state = {
            (key[len("module.") :] if key.startswith("module.") else key): value for key, value in loaded_state.items()
        }
        self.model.update_to_step(step)
        self.load_state_dict(state)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: Optional[bool] = None):
        is_ddp_model_state = True
        model_state = {}
        for key, value in state_dict.items():
            if key.startswith("_model."):
                # remove the "_model." prefix from key
                model_state[key[len("_model.") :]] = value
                # make sure that the "module." prefix comes from DDP,
                # rather than an attribute of the model named "module"
                if not key.startswith("_model.module."):
                    is_ddp_model_state = False
        # remove "module." prefix added by DDP
        if is_ddp_model_state:
            model_state = {key[len("module.") :]: value for key, value in model_state.items()}

        pipeline_state = {key: value for key, value in state_dict.items() if not key.startswith("_model.")}

        try:
            self.model.load_state_dict(model_state, strict=True)
        except RuntimeError:
            if not strict:
                self.model.load_state_dict(model_state, strict=False)
            else:
                raise

        return super(Pipeline, self).load_state_dict(pipeline_state, strict=False)

    def insert_model(self, checkpoint_path: str, config_path: str):
        """
        Insert a foreground object into the scene model
        """
        if hasattr(self.model, "insert_model"):
            self.model.insert_model(
                checkpoint_path=Path(checkpoint_path),
                config_path=Path(config_path)
            )
        else:
            raise NotImplementedError("Model type" + type(self.model) +  "does not support inserting models")
