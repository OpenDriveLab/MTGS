#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
from __future__ import annotations

import os
from copy import deepcopy
from typing import Dict, List, Literal, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from jaxtyping import Bool, Float, Int32, UInt8
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.utils.rich_utils import CONSOLE
from PIL import Image

from mtgs.dataset.nuplan_dataparser import NuPlanDataparserOutputs
from mtgs.utils.nuplan_pointcloud import load_lidar
from nuplan_scripts.utils.constants import cityscape_label


def adjust_brightness(image, factor):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float64)
    image_hsv[..., 2] = (image_hsv[..., 2] * factor)
    image_hsv[..., 2] = np.clip(image_hsv[..., 2], 0, 255)
    image_hsv = image_hsv.astype(np.uint8)
    image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
    return image

class CustomInputDataset(InputDataset):

    _dataparser_outputs: NuPlanDataparserOutputs

    RESAMPLE_MAP = {
        'image': Image.Resampling.BILINEAR,
        'mask': Image.Resampling.NEAREST,
        'semantic_map': Image.Resampling.NEAREST,
        'instance_map': Image.Resampling.NEAREST,
        'depth': Image.Resampling.BILINEAR
    }

    CV2_INTER_MAP = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC
    }

    def __init__(self, 
                 dataparser_outputs: NuPlanDataparserOutputs, 
                 crop_size: Optional[Tuple[int, int, int, int]] = None,
                 fake_data: bool = False,
                 load_mask: bool = False,
                 load_custom_masks: Tuple = (),   # vehicle, pedestrian, bicycle
                 load_instance_masks: bool = False,
                 load_semantic_masks_from: Literal["panoptic", "semantic", False] = "panoptic",
                 load_lidar_depth: bool = False,
                 load_pseudo_depth: bool = False,
                 undistort_images: Literal["optimal", "keep_focal_length", False] = "optimal",
                 eval_dataset: bool = False,
                 ):
        CONSOLE.log(f"Using CustomInputDataset, eval: {eval_dataset}")
        self.crop_size = crop_size
        self.fake_data = fake_data
        self.load_mask = load_mask
        self.load_custom_masks = load_custom_masks
        self.load_instance_masks = load_instance_masks
        self.load_semantic_masks_from = load_semantic_masks_from
        self.load_lidar_depth = load_lidar_depth
        self.load_pseudo_depth = load_pseudo_depth
        self.undistort_images = undistort_images
        self.eval_dataset = eval_dataset
        super().__init__(dataparser_outputs, 1.0)

    def get_numpy_image(self, image_idx: int) -> UInt8[np.ndarray, "h w c"]:
        """Returns the image of shape (H, W, 3 or 4).

        Args:
            image_idx: The image index in the dataset.
        """
        image_filename = self._dataparser_outputs.image_filenames[image_idx]
        pil_image = Image.open(image_filename)
        image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(3, axis=2)

        if self._dataparser_outputs.v_adjust_factors is not None:
            v_factor = self._dataparser_outputs.v_adjust_factors[image_idx]
            image = adjust_brightness(image, v_factor)

        assert len(image.shape) == 3
        assert image.dtype == np.uint8
        assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."
        return image

    def _undistort_image(
        self, image: UInt8[np.ndarray, "h w *c"], 
        image_idx: int, 
        return_mask: bool = False,
        interpolation: Literal["nearest", "linear", "cubic"] = "linear"
    ):
        width, height = image.shape[1], image.shape[0]
        intrinsic, distortion = self._dataparser_outputs.undistort_params[image_idx]
        intrinsic = intrinsic.copy()
        intrinsic[0, 2] -= 0.5
        intrinsic[1, 2] -= 0.5

        recover_last_dim = False
        if len(image.shape) == 3 and image.shape[2] == 1:
            # cv2.remap will return [H, W] if the last dim is 1
            recover_last_dim = True
            image = image[:, :, 0]

        if self.undistort_images == "optimal":
            new_intrinsic, roi = cv2.getOptimalNewCameraMatrix(
                intrinsic, distortion, (width, height), 1
            )
            map_undistort = cv2.initUndistortRectifyMap(
                intrinsic, distortion, None, new_intrinsic, (width, height), cv2.CV_32FC2
            )
        elif self.undistort_images == "keep_focal_length":
            new_intrinsic = intrinsic.copy()
            roi = (0, 0, width, height)
            map_undistort = cv2.initUndistortRectifyMap(
                intrinsic, distortion, None, new_intrinsic, (width, height), cv2.CV_32FC2
            )

        new_intrinsic[0, 2] += 0.5
        new_intrinsic[1, 2] += 0.5

        image = cv2.remap(
            image, map_undistort[0], map_undistort[1], 
            interpolation=self.CV2_INTER_MAP[interpolation]
        )

        if recover_last_dim and len(image.shape) == 2:
            image = image[:, :, None]

        return_dict = {"image": image, "new_intrinsic": new_intrinsic, "roi": roi}

        if return_mask:
            valid_mask = np.ones_like(image[..., 0], dtype=np.uint8) * 255
            valid_mask = cv2.remap(
                valid_mask, map_undistort[0], map_undistort[1], 
                interpolation=cv2.INTER_NEAREST
            )
            return_dict["valid_mask"] = valid_mask.astype(bool)[:, :, None]

        return return_dict

    def _get_ego_mask(self, image_idx: int):
        ego_mask = cv2.imread(self._dataparser_outputs.ego_mask_filenames[image_idx], cv2.IMREAD_GRAYSCALE) != 0
        valid_mask = np.logical_not(ego_mask).astype(np.uint8) * 255
        if self.undistort_images:
            valid_mask = self._undistort_image(
                valid_mask, image_idx, return_mask=False, interpolation="nearest")["image"]
        return valid_mask[:, :, None]

    def _get_depth_from_image(self, image_idx: int):
        def read_depth_image(path):
            if not os.path.exists(path):
                return None
            depth_img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            depth = depth_img[:,:,0] + (depth_img[:,:,1] * 256.0)
            depth = depth * 0.01
            return depth

        depth_filename = self._dataparser_outputs.depth_filenames[image_idx]
        depth = read_depth_image(depth_filename)
        return depth[:, :, None]

    def _get_depth_from_lidar(self, image_idx: int, camera: Cameras, points: Optional[Float[np.ndarray, "n_pts 3"]] = None):
        lidar2cam = self._dataparser_outputs.lidar_to_cams[image_idx].numpy()
        lidar_filename = self._dataparser_outputs.lidar_filenames[image_idx]

        camera = camera.reshape(())
        h = camera.height.item()
        w = camera.width.item()
        K = camera.get_intrinsics_matrices().numpy()

        if points is None:
            points = load_lidar(lidar_filename)

        points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        point_cam = (points @ lidar2cam.T)[:, :3]
        uvz = point_cam[point_cam[:, 2] > 0]
        uvz = uvz @ K.T
        uvz[:, :2] /= uvz[:, 2:]
        uvz = uvz[uvz[:, 1] >= 0]
        uvz = uvz[uvz[:, 1] < h]
        uvz = uvz[uvz[:, 0] >= 0]
        uvz = uvz[uvz[:, 0] < w]
        uv = uvz[:, :2]
        uv = uv.astype(int)
        pts_depth = np.zeros((h, w), dtype=np.float32)
        pts_depth[uv[:, 1], uv[:, 0]] = uvz[:, 2]

        return pts_depth[:, :, None]

    def _get_panoptic_map(self, image_idx: int, valid_mask: Bool[np.ndarray, "h w 1"] = None):

        def read_panoptic_mask_image(path):
            if not os.path.exists(path):
                assert "The instance mask does not exist in the path {}.".format(path)
            if path.endswith(".npy"):
                return np.load(path)
            panoptic_map = cv2.imread(path).astype(np.uint8)
            return panoptic_map

        def get_true_instance_map(panoptic_map):
            return panoptic_map[:, :, 0] + panoptic_map[:, :, 1] * 256
        
        def get_true_semantic_map(panoptic_map):
            return panoptic_map[:, :, 2]

        panoptic_mask_filenames = self._dataparser_outputs.panoptic_mask_filenames[image_idx]
        panoptic_map = read_panoptic_mask_image(panoptic_mask_filenames)
        if self.undistort_images:
            return_dict = self._undistort_image(panoptic_map, image_idx, return_mask=False, interpolation="nearest")
            panoptic_map = return_dict["image"]

        instance_map = get_true_instance_map(panoptic_map.astype(np.int32))[:, :, None]
        semantic_map = get_true_semantic_map(panoptic_map)[:, :, None]

        if valid_mask is not None:
            instance_map[~valid_mask] = 0     # 0 is invalid instance id
            semantic_map[~valid_mask] = 255   # 255 is invalid semantic class

        return instance_map, semantic_map

    def _get_semantic_map(self, image_idx: int, valid_mask: Bool[np.ndarray, "h w 1"] = None):
        semantic_map_filename = self._dataparser_outputs.semantic_mask_filenames[image_idx]
        semantic_map = cv2.imread(semantic_map_filename, cv2.IMREAD_GRAYSCALE)
        if self.undistort_images:
            return_dict = self._undistort_image(semantic_map, image_idx, return_mask=False, interpolation="nearest")
            semantic_map = return_dict["image"]
        semantic_map = semantic_map[:, :, None]
        if valid_mask is not None:
            semantic_map[~valid_mask] = 255
        return semantic_map
    
    def _is_training_traversal(self, image_idx: int):
        travel_id = self._dataparser_outputs.travel_ids[image_idx]
        train_travel_ids = self._dataparser_outputs.metadata['train_travel_ids']
        if train_travel_ids is not None and len(train_travel_ids) > 0:
            for train_travel_id in train_travel_ids:
                train_travel_id = train_travel_id['idx'] if isinstance(train_travel_id, dict) else train_travel_id
                if train_travel_id == travel_id:   # find traversal in training.
                    return True
        else:
            return True
        return False

    def _get_custom_mask(self, semantic_map: UInt8[np.ndarray, "h w 1"], image_idx: int):
        # mask all foreground objects if traversal is evaluation only.
        mask_all_foreground = not self._is_training_traversal(image_idx)
        masked_labels = [255] # 255 is invalid semantic class
        if 'pedestrian' in self.load_custom_masks or mask_all_foreground:
            masked_labels.append(cityscape_label['person'])
        if 'bicycle' in self.load_custom_masks or mask_all_foreground:
            masked_labels.append(cityscape_label['bicycle'])
            masked_labels.append(cityscape_label['motorcycle'])
            masked_labels.append(cityscape_label['rider'])
        if 'vehicle' in self.load_custom_masks or mask_all_foreground:
            masked_labels.append(cityscape_label['car'])
            masked_labels.append(cityscape_label['truck'])
            masked_labels.append(cityscape_label['bus'])

        mask = np.isin(semantic_map, masked_labels)
        mask = np.logical_not(mask)
        return mask

    def _crop_data(self, data: Dict, camera: Cameras, crop_size: Tuple[int, int, int, int]):
        pass

    def _resize_numpy_image(self, image: np.ndarray, newsize: Tuple[int, int], resample: Image.Resampling):
        assert image.dtype in (np.uint8, np.float32, np.int32, bool)
        raw_type = image.dtype
        if raw_type is bool:
            image = image.astype(np.uint8)

        recover_last_dim = False
        if len(image.shape) == 3 and image.shape[2] == 1:
            # PIL Image can only handle [H, W]
            recover_last_dim = True
            image = image[:, :, 0]

        image_pil = Image.fromarray(image)
        image_pil = image_pil.resize(newsize, resample=resample)
        image = np.array(image_pil, dtype=raw_type)
        if recover_last_dim and len(image.shape) == 2:
            image = image[:, :, None]

        return image

    def _resize_data(self, data: Dict, camera: Cameras, scale_factor: float):
        camera.rescale_output_resolution(scaling_factor=scale_factor)
        newsize = (camera.width.item(), camera.height.item())
        for key in data:
            if isinstance(data[key], np.ndarray):
                data[key] = self._resize_numpy_image(data[key], newsize, resample=self.RESAMPLE_MAP[key])

    def get_fake_data(self, image_idx: int, image_type: Literal["uint8", "float32"] = "float32", scale_factor: float = 1.0):
        camera = deepcopy(self.cameras[image_idx: image_idx + 1])
        height = int(camera.height.item() * scale_factor)
        width = int(camera.width.item() * scale_factor)
        image = torch.from_numpy(np.zeros((height, width, 3), dtype=image_type))
        data = {
            "image_idx": image_idx,
            "image": image,
        }
        camera = self.push_cam_details_to_metadata(camera, self._dataparser_outputs, image_idx)
        return data, camera

    def get_data_and_camera(self, 
                            image_idx: int, 
                            image_type: Literal["uint8", "float32"] = "float32", 
                            scale_factor: float = 1.0):
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
            image_type: the type of images returned
            scale_factor: the scale factor to resize the image
        
        Returns:
            data: {
                "image_idx": image_idx,
                "image": torch.Tensor, [H, W, 3]
                "mask": torch.Tensor, [H, W, 1]
                "semantic_map": torch.Tensor, [H, W, 1]
                "instance_map": torch.Tensor, [H, W, 1]
                "depth": torch.Tensor, [H, W, 1]
                "lidar_depth": torch.Tensor, [H, W, 1]
            }
        """
        if self.fake_data:
            return self.get_fake_data(image_idx, image_type, scale_factor)

        camera = deepcopy(self.cameras[image_idx: image_idx + 1])
        image = self.get_numpy_image(image_idx)
        valid_mask = np.ones_like(image[..., 0:1], dtype=bool)
        if self.undistort_images:
            return_dict = self._undistort_image(image, image_idx, return_mask=True, interpolation="linear")
            image = return_dict["image"]
            valid_mask = np.logical_and(valid_mask, return_dict["valid_mask"])
            new_intrinsic = return_dict["new_intrinsic"]
            camera.fx[:] = new_intrinsic[0, 0]
            camera.fy[:] = new_intrinsic[1, 1]
            camera.cx[:] = new_intrinsic[0, 2]
            camera.cy[:] = new_intrinsic[1, 2]
            camera.height[:] = int(image.shape[0])
            camera.width[:] = int(image.shape[1])

        data = {
            "image_idx": image_idx, 
            "image": image
        }

        if self._dataparser_outputs.ego_mask_filenames is not None:
            ego_valid_mask = self._get_ego_mask(image_idx)
            valid_mask = np.logical_and(valid_mask, ego_valid_mask)

        if self._dataparser_outputs.panoptic_mask_filenames is not None and (self.load_instance_masks or self.load_semantic_masks_from == "panoptic"):
            instance_map, semantic_map = self._get_panoptic_map(image_idx, valid_mask=valid_mask)
            if self.load_instance_masks:
                data["instance_map"] = instance_map
            if self.load_semantic_masks_from == "panoptic":
                data["semantic_map"] = semantic_map

        if self._dataparser_outputs.semantic_mask_filenames is not None and self.load_semantic_masks_from == "semantic":
            data["semantic_map"] = self._get_semantic_map(image_idx, valid_mask=valid_mask)

        if self.load_mask:
            if "semantic_map" in data and len(self.load_custom_masks) > 0:
                custom_mask = self._get_custom_mask(semantic_map=data["semantic_map"], image_idx=image_idx)
                data["mask"] = np.logical_and(valid_mask, custom_mask)
            else:
                data["mask"] = valid_mask

        if self._dataparser_outputs.depth_filenames is not None and self.load_pseudo_depth:
            # depth image is already undistorted
            data["depth"] = self._get_depth_from_image(image_idx) * self._dataparser_outputs.scene_scale_factor

        if scale_factor != 1.0:
            self._resize_data(data, camera, scale_factor)

        if self._dataparser_outputs.lidar_filenames is not None and self.load_lidar_depth:
            # ensure that the cameras are already undistorted / resized / cropped.
            data["lidar_depth"] = self._get_depth_from_lidar(image_idx, camera=camera) * self._dataparser_outputs.scene_scale_factor

        # transform all the data to torch tensors
        if image_type == "float32":
            data["image"] = image = torch.from_numpy(data["image"].astype(np.float32) / 255.0)
        elif image_type == "uint8":
            data["image"] = image = torch.from_numpy(data["image"])
        else:
            raise NotImplementedError(f"image_type (={image_type}) getter was not implemented, use uint8 or float32")
        for key in data:
            if isinstance(data[key], np.ndarray):
                data[key] = torch.from_numpy(data[key])

        # push metadata to data and camera
        metadata = self.get_metadata(data)
        data.update(metadata)
        camera = self.push_cam_details_to_metadata(camera, self._dataparser_outputs, image_idx)
        return data, camera

    def push_cam_details_to_metadata(
            self, camera: Cameras, dataparser_outputs: NuPlanDataparserOutputs, idx: int) -> Cameras:
        """
        fetch camera details from dataparser_outputs.
        """ 
        if camera.metadata is None:
            camera.metadata = {}

        camera.metadata["cam_token"] = dataparser_outputs.cam_tokens[idx]
        camera.metadata["frame_token"] = dataparser_outputs.frame_tokens[idx]
        camera.metadata["travel_id"] = dataparser_outputs.travel_ids[idx]
        if not self.eval_dataset:
            camera.metadata["frame_idx"] = dataparser_outputs.frame_ids[idx]
            camera.metadata["cam_idx"] = idx
        else:
            camera.metadata["frame_idx"] = None
            camera.metadata["cam_idx"] = None
        camera.metadata["data_path"] = dataparser_outputs.image_filenames[idx].parts[-3:]
        camera.metadata["linear_velocity"] = dataparser_outputs.camera_linear_velocities[idx]
        camera.metadata["angular_velocity"] = dataparser_outputs.camera_angular_velocities[idx]
        return camera
