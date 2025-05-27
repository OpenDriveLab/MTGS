#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import time
import dataclasses
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal, Type

import torch
from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.utils.decorators import check_main_thread
from nerfstudio.engine.callbacks import TrainingCallbackAttributes
from nerfstudio.utils import profiler, writer
from nerfstudio.utils.rich_utils import CONSOLE
from mtgs.custom_viewer.viewer import Viewer

class CustomOptimizers(Optimizers):
    def scheduler_step_all(self, step: int) -> None:
        """Run step for all schedulers.
        We disable the lr logging for MTGS model.

        Args:
            step: the current step
        """
        for param_group_name, scheduler in self.schedulers.items():
            scheduler.step()

@dataclass
class CustomTrainerConfig(TrainerConfig):
    _target: Type = field(default_factory=lambda: CustomTrainer)

    def get_base_dir(self) -> Path:
        # once the base_dir is set, use it
        if hasattr(self, 'base_dir'):
            return Path(self.base_dir)
        else:
            return super().get_base_dir()

    def set_base_dir(self, base_dir: Path) -> None:
        self.base_dir = base_dir

class CustomTrainer(Trainer):
    config: CustomTrainerConfig
    optimizers: CustomOptimizers

    def __init__(self, config: CustomTrainerConfig, **kwargs):
        super().__init__(config, **kwargs)

    def setup(self, test_mode: Literal["test", "val", "inference"] = "val") -> None:
        """Setup the Trainer by calling other setup functions.

        Args:
            test_mode:
                'val': loads train/val datasets into memory
                'test': loads train/test datasets into memory
                'inference': does not load any dataset into memory
        """
        self.pipeline = self.config.pipeline.setup(
            device=self.device,
            test_mode=test_mode,
            world_size=self.world_size,
            local_rank=self.local_rank,
            grad_scaler=self.grad_scaler,
        )
        self.optimizers = self.setup_optimizers()

        # set up viewer if enabled
        viewer_log_path = self.base_dir / self.config.viewer.relative_log_filename
        self.viewer_state, banner_messages = None, None
        if self.config.is_viewer_enabled() and self.local_rank == 0:
            datapath = self.config.data
            if datapath is None:
                datapath = self.base_dir
            self.viewer_state = Viewer(
                self.config.viewer,
                log_filename=viewer_log_path,
                datapath=datapath,
                pipeline=self.pipeline,
                trainer=self,
                train_lock=self.train_lock,
                share=self.config.viewer.make_share_url,
            )
            banner_messages = self.viewer_state.viewer_info
        self._check_viewer_warnings()

        self._load_checkpoint()

        self.callbacks = self.pipeline.get_training_callbacks(
            TrainingCallbackAttributes(
                optimizers=self.optimizers, grad_scaler=self.grad_scaler, pipeline=self.pipeline, trainer=self
            )
        )

        # set up writers/profilers if enabled
        writer_log_path = self.base_dir / self.config.logging.relative_log_dir
        writer.setup_event_writer(
            self.config.is_wandb_enabled(),
            self.config.is_tensorboard_enabled(),
            self.config.is_comet_enabled(),
            log_dir=writer_log_path,
            experiment_name=self.config.experiment_name,
            project_name=self.config.project_name,
        )
        writer.setup_local_writer(
            self.config.logging, max_iter=self.config.max_num_iterations, banner_messages=banner_messages
        )
        writer.put_config(name="config", config_dict=dataclasses.asdict(self.config), step=0)
        profiler.setup_profiler(self.config.logging, writer_log_path)

    def setup_optimizers(self) -> CustomOptimizers:
        """Helper to set up the optimizers

        Returns:
            The optimizers object given the trainer config.
        """
        param_groups = self.pipeline.get_param_groups()
        optimizer_config = {}
        for key, value in param_groups.items():
            if key not in self.config.optimizers:    # in the format of {model_name}.{model_type}.{param_name}
                sub_key = key
                while '.' in sub_key:
                    sub_key = sub_key.split('.', maxsplit=1)[-1]
                    if sub_key in self.config.optimizers:
                        break

                assert sub_key in self.config.optimizers, f"Optimizer for {key} not found in config"
                optimizer_config[key] = self.config.optimizers[sub_key].copy()
            else:
                optimizer_config[key] = self.config.optimizers[key].copy()

        return CustomOptimizers(optimizer_config, param_groups)

    @check_main_thread
    def save_checkpoint(self, step: int) -> None:
        """Save the model and optimizers

        Args:
            step: number of steps in training for given checkpoint
        """
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # save the checkpoint
        ckpt_path: Path = self.checkpoint_dir / f"step-{step:09d}.ckpt"
        ckpt_dict = {
            "step": step,
            "pipeline": self.pipeline.module.state_dict()  # type: ignore
            if hasattr(self.pipeline, "module")
            else self.pipeline.state_dict(),
            "optimizers": {k: v.state_dict() for (k, v) in self.optimizers.optimizers.items()},
            "schedulers": {k: v.state_dict() for (k, v) in self.optimizers.schedulers.items()},
            "scalers": self.grad_scaler.state_dict(),
        }
        if self.training_state == "completed":
            ckpt_dict.pop("optimizers")
            ckpt_dict.pop("schedulers")
            ckpt_dict.pop("scalers")

        max_retries = 5
        for attempt in range(max_retries):
            try:
                torch.save(ckpt_dict, ckpt_path)
                break
            except Exception as e:
                CONSOLE.log(f"[ERROR] saving checkpoint on attempt {attempt + 1}", style="bold red")
                CONSOLE.print(e)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise e

        # possibly delete old checkpoints
        if self.config.save_only_latest_checkpoint:
            # delete every other checkpoint in the checkpoint folder
            for f in self.checkpoint_dir.glob("*.ckpt"):
                if f != ckpt_path:
                    f.unlink()

    def _load_checkpoint(self) -> None:
        """Helper function to load pipeline and optimizer from prespecified checkpoint"""
        load_dir = self.config.load_dir
        load_checkpoint = self.config.load_checkpoint
        if load_dir is not None:
            load_step = self.config.load_step
            if load_step is None:
                print("Loading latest Nerfstudio checkpoint from load_dir...")
                # NOTE: this is specific to the checkpoint name format
                load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(load_dir))[-1]
            load_path: Path = load_dir / f"step-{load_step:09d}.ckpt"
            assert load_path.exists(), f"Checkpoint {load_path} does not exist"
            loaded_state = torch.load(load_path, map_location="cpu")
        elif load_checkpoint is not None:
            assert load_checkpoint.exists(), f"Checkpoint {load_checkpoint} does not exist"
            load_path = load_checkpoint
            loaded_state = torch.load(load_path, map_location="cpu")
        else:
            CONSOLE.print("No Nerfstudio checkpoint to load, so training from scratch.")
            return
        # load the checkpoints for pipeline, optimizers, and gradient scalar
        self._start_step = loaded_state["step"] + 1
        self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
        if "optimizers" in loaded_state:
            self.optimizers.load_optimizers(loaded_state["optimizers"])
        if "schedulers" in loaded_state and self.config.load_scheduler:
            self.optimizers.load_schedulers(loaded_state["schedulers"])
        if "scalers" in loaded_state:
            self.grad_scaler.load_state_dict(loaded_state["scalers"])
        CONSOLE.print(f"Done loading Nerfstudio checkpoint from {load_path}")
