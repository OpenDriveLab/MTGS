#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
from nerfstudio.configs.base_config import ViewerConfig, MachineConfig, LoggingConfig, LocalWriterConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig

from mtgs.dataset.nuplan_dataparser import NuplanDataParserConfig
from mtgs.dataset.custom_datamanager import CustomFullImageDatamanagerConfig
from mtgs.scene_model.custom_trainer import CustomTrainerConfig
from mtgs.scene_model.custom_pipeline import MultiTravelEvalPipielineConfig
from mtgs.scene_model.mtgs_scene_graph import MTGSSceneModelConfig
from mtgs.scene_model.module.appearance import GSAppearanceModelConfig
from mtgs.scene_model.gaussian_model.vanilla_gaussian_splatting import VanillaGaussianSplattingModelConfig, GaussianSplattingControlConfig
from mtgs.utils.geometric_loss import DepthLossType

iteration_factor = 1
config = CustomTrainerConfig(
    method_name="3dgs",
    project_name="MTGS",
    experiment_name="3DGS",
    steps_per_eval_image=30001 * iteration_factor,
    steps_per_eval_batch=0 * iteration_factor,
    steps_per_save=3000 * iteration_factor,
    steps_per_eval_all_images=2000 * iteration_factor,
    max_num_iterations=30001 * iteration_factor,
    mixed_precision=False,
    pipeline=MultiTravelEvalPipielineConfig(
        datamanager=CustomFullImageDatamanagerConfig(
            dataparser=NuplanDataParserConfig(
                load_3D_points=True,
                only_moving=False,          # if multi-traversal, disable this.
                undistort_images="optimal",
                eval_2hz=True,
                use_exposure_alignment=False,
                scale_factor=0.005
            ),
            camera_res_scale_factor=0.5,
            cache_strategy="async",
            num_workers=4,
            cache_images="cpu",
            cache_images_type="uint8",
            load_mask=True,
            load_semantic_masks_from="semantic",
            load_instance_masks=False,
            load_custom_masks=('pedestrian', 'bicycle'),
            load_lidar_depth=True,
            load_pseudo_depth=False,
            undistort_images="optimal",
        ),
        model=MTGSSceneModelConfig(
            control=GaussianSplattingControlConfig(
                densify_from_iter=500*iteration_factor,
                refine_every=100*iteration_factor,
                stop_split_at=15000*iteration_factor,
                reset_alpha_every=30,  # opacity_reset_interval reset_alpha_every * refine_every
                continue_cull_post_densification=False,
                cull_alpha_thresh=0.005,
                cull_scale_thresh=0.1,
                densify_size_thresh=0.01,
                densify_grad_thresh=0.002,
                n_split_samples=2,
                clone_sample_means=False,
                stop_screen_size_at=0*iteration_factor,
                cull_screen_size=150,
                split_screen_size=100,
                sh_degree=3,
                sh_degree_interval=1000*iteration_factor,
                use_abs_grad=False,
            ),
            model_config=dict(
                background=VanillaGaussianSplattingModelConfig(
                    model_type='vanilla',  # param group name
                    verbose=True,
                )
            ),
            background_color='black',
            camera_optimizer=CameraOptimizerConfig(
                mode='off'
            ),
            appearance_model=GSAppearanceModelConfig(),
            rasterize_mode='classic',
            color_corrected_metrics=True,
            lpips_metric=True,
            ssim_lambda=0.2,
            output_depth_during_training=True,
            use_depth_loss=False,
            depth_source='lidar',
            depth_loss_type=DepthLossType.InverseL1,
            depth_lambda=0.5,
            predict_normals=False,
            use_normal_loss=False,
            normal_supervision='depth',
            use_normal_tv_loss=False,
            normal_lambda=0.1,
            two_d_gaussians=False,
        ),
    ),
    optimizers={
        "means": {
            "optimizer": AdamOptimizerConfig(lr=0.00016, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=0.0000016,
                max_steps=30001 * iteration_factor,
            ),
        },
        "features_dc": {
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        },
        "features_rest": {
            "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
            "scheduler": None,
        },
        "opacities": {
            "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
            "scheduler": None,
        },
        "scales": {
            "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
            "scheduler": None,
        },
        "quats": {
            "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), 
            "scheduler": None
        },
    },
    vis="none",
    viewer=ViewerConfig(
        camera_frustum_scale=0.3,
        default_composite_depth=False,
        max_num_display_images=500,
        num_rays_per_chunk=1 << 15),
    machine=MachineConfig(
        seed=0,
        num_devices=1,
        device_type='cuda',
    ),
    logging=LoggingConfig(
        profiler='none',
        steps_per_log=50,
        max_buffer_size=100,
        local_writer=LocalWriterConfig(
            enable=True,
            max_log_size=0,
        )
    )
)

method = MethodSpecification(
  config=config,
  description="3dgs"
)
