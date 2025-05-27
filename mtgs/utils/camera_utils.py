#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import math
from typing import List, Optional, Tuple

import numpy as np
import cv2
import torch
from torch import Tensor
from pyquaternion import Quaternion

# opengl to opencv transformation matrix
OPENGL_TO_OPENCV = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

# ndc space is x to the right y up. uv space is x to the right, y down.
def pix2ndc_x(x, W):
    x = x.float()
    return (2 * x) / W - 1


def pix2ndc_y(y, H):
    y = y.float()
    return 1 - (2 * y) / H


# ndc is y up and x right. uv is y down and x right
def ndc2pix_x(x, W):
    return (x + 1) * 0.5 * W


def ndc2pix_y(y, H):
    return (1 - y) * 0.5 * H


def euclidean_to_z_depth(
    depths: Tensor,
    fx: float,
    fy: float,
    cx: int,
    cy: int,
    img_size: tuple,
    device: torch.device,
) -> Tensor:
    """Convert euclidean depths to z_depths given camera intrinsics"""
    if depths.dim() == 3:
        depths = depths.view(-1, 1)
    elif depths.shape[-1] != 1:
        depths = depths.unsqueeze(-1).contiguous()
        depths = depths.view(-1, 1)
    if depths.dtype != torch.float:
        depths = depths.float()
    image_coords = get_camera_coords(img_size=img_size)
    image_coords = image_coords.to(device)

    z_depth = torch.empty(
        size=(img_size[0], img_size[1], 3), dtype=torch.float32, device=device
    ).view(-1, 3)
    z_depth[:, 0] = (image_coords[:, 0] - cx) / fx  # x
    z_depth[:, 1] = (image_coords[:, 1] - cy) / fy  # y
    z_depth[:, 2] = 1  # z

    z_depth = z_depth / torch.norm(z_depth, dim=-1, keepdim=True)
    z_depth = (z_depth * depths)[:, 2]  # pick only z component

    z_depth = z_depth[..., None]
    z_depth = z_depth.view(img_size[1], img_size[0], 1)

    return z_depth


def get_camera_coords(img_size: tuple, pixel_offset: float = 0.5) -> Tensor:
    """Generates camera pixel coordinates [W,H]

    Returns:
        stacked coords [H*W,2] where [:,0] corresponds to W and [:,1] corresponds to H
    """

    # img size is (w,h)
    image_coords = torch.meshgrid(
        torch.arange(img_size[0]),
        torch.arange(img_size[1]),
        indexing="xy",  # W = u by H = v
    )
    image_coords = (
        torch.stack(image_coords, dim=-1) + pixel_offset
    )  # stored as (x, y) coordinates
    image_coords = image_coords.view(-1, 2)
    image_coords = image_coords.float()

    return image_coords


def get_means3d_backproj(
    depths: Tensor,
    fx: float,
    fy: float,
    cx: int,
    cy: int,
    img_size: tuple,
    c2w: Tensor,
    device: torch.device,
    mask: Optional[Tensor] = None,
) -> Tuple[Tensor, List]:
    """Backprojection using camera intrinsics and extrinsics

    image_coords -> (x,y,depth) -> (X, Y, depth)

    Returns:
        Tuple of (means: Tensor, image_coords: Tensor)
    """

    if depths.dim() == 3:
        depths = depths.view(-1, 1)
    elif depths.shape[-1] != 1:
        depths = depths.unsqueeze(-1).contiguous()
        depths = depths.view(-1, 1)
    if depths.dtype != torch.float:
        depths = depths.float()
        c2w = c2w.float()
    if c2w.device != device:
        c2w = c2w.to(device)

    image_coords = get_camera_coords(img_size)
    image_coords = image_coords.to(device)  # note image_coords is (H,W)

    # TODO: account for skew / radial distortion
    means3d = torch.empty(
        size=(img_size[0], img_size[1], 3), dtype=torch.float32, device=device
    ).view(-1, 3)
    means3d[:, 0] = (image_coords[:, 0] - cx) * depths[:, 0] / fx  # x
    means3d[:, 1] = (image_coords[:, 1] - cy) * depths[:, 0] / fy  # y
    means3d[:, 2] = depths[:, 0]  # z

    if mask is not None:
        if not torch.is_tensor(mask):
            mask = torch.tensor(mask, device=depths.device)
        means3d = means3d[mask]
        image_coords = image_coords[mask]

    if c2w is None:
        c2w = torch.eye((means3d.shape[0], 4, 4), device=device)

    # to world coords
    means3d = means3d @ torch.linalg.inv(c2w[..., :3, :3]) + c2w[..., :3, 3]
    return means3d, image_coords


def project_pix(
    p: Tensor,
    fx: float,
    fy: float,
    cx: int,
    cy: int,
    c2w: Tensor,
    device: torch.device,
    return_z_depths: bool = False,
) -> Tensor:
    """Projects a world 3D point to uv coordinates using intrinsics/extrinsics

    Returns:
        uv coords
    """
    if c2w is None:
        c2w = torch.eye((p.shape[0], 4, 4), device=device)  # type: ignore
    if c2w.device != device:
        c2w = c2w.to(device)

    points_cam = (p.to(device) - c2w[..., :3, 3]) @ c2w[..., :3, :3]
    u = points_cam[:, 0] * fx / points_cam[:, 2] + cx  # x
    v = points_cam[:, 1] * fy / points_cam[:, 2] + cy  # y
    if return_z_depths:
        return torch.stack([u, v, points_cam[:, 2]], dim=-1)
    return torch.stack([u, v], dim=-1)


def get_colored_points_from_depth(
    depths: Tensor,
    rgbs: Tensor,
    c2w: Tensor,
    fx: float,
    fy: float,
    cx: int,
    cy: int,
    img_size: tuple,
    mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Return colored pointclouds from depth and rgb frame and c2w. Optional masking.

    Returns:
        Tuple of (points, colors)
    """
    points, _ = get_means3d_backproj(
        depths=depths.float(),
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        img_size=img_size,
        c2w=c2w.float(),
        device=depths.device,
    )
    points = points.squeeze(0)
    if mask is not None:
        if not torch.is_tensor(mask):
            mask = torch.tensor(mask, device=depths.device)
        colors = rgbs.view(-1, 3)[mask]
        points = points[mask]
    else:
        colors = rgbs.view(-1, 3)
        points = points
    return (points, colors)


def get_rays_x_y_1(H, W, focal, c2w):
    """Get ray origins and directions in world coordinates.

    Convention here is (x,y,-1) such that depth*rays_d give real z depth values in world coordinates.
    """
    assert c2w.shape == torch.Size([3, 4])
    image_coords = torch.meshgrid(
        torch.arange(W, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32),
        indexing="ij",
    )
    i, j = image_coords
    # dirs = torch.stack([(i-W*0.5)/focal, -(j-H*0.5)/focal, -torch.ones_like(i)], dim = -1)
    dirs = torch.stack(
        [(pix2ndc_x(i, W)) / focal, pix2ndc_y(j, H) / focal, -torch.ones_like(i)],
        dim=-1,
    )
    dirs = dirs.view(-1, 3)
    rays_d = dirs[..., :] @ c2w[:3, :3]
    rays_o = c2w[:3, -1].expand_as(rays_d)

    # return world coordinate rays_o and rays_d
    return rays_o, rays_d


def get_projection_matrix(znear=0.001, zfar=1000, fovx=None, fovy=None, **kwargs):
    """Opengl projection matrix

    Returns:
        projmat: Tensor
    """

    t = znear * math.tan(0.5 * fovy)
    b = -t
    r = znear * math.tan(0.5 * fovx)
    l = -r
    n = znear
    f = zfar
    return torch.tensor(
        [
            [2 * n / (r - l), 0.0, (r + l) / (r - l), 0.0],
            [0.0, 2 * n / (t - b), (t + b) / (t - b), 0.0],
            [0.0, 0.0, (f + n) / (f - n), -1.0 * f * n / (f - n)],
            [0.0, 0.0, 1.0, 0.0],
        ],
        **kwargs,
    )


def get_intrinsics_from_camera(fx, fy, cx, cy):
    """Get intrinsics matrix from camera parameters"""
    intrinsic = np.eye(3)
    intrinsic[0, 0] = fx
    intrinsic[1, 1] = fy
    intrinsic[0, 2] = cx
    intrinsic[1, 2] = cy
    return intrinsic


def matrix_from_translation_and_quaternion(translation, quaternion, opencv2nf=False):
    matrix = np.eye(4)
    rotation = Quaternion(quaternion).rotation_matrix
    if opencv2nf:
        rotation = rotation @ np.diag([1, -1, -1])
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def inverse_matrix_from_translation_and_quaternion(translation, quaternion, opencv2nf=False):
    matrix = np.eye(4)
    rotation = Quaternion(quaternion).rotation_matrix
    if opencv2nf:
        rotation = rotation @ np.diag([1, -1, -1])
    matrix[:3, :3] = rotation.T
    matrix[:3, 3] = -np.dot(rotation.T, translation)
    return matrix

def calculate_camera_velocity_in_world(
    ego_linear_velocity, 
    ego_angular_velocity, 
    cam2ego_transform,  # or ego2cam_transform
    ego2global_transform,
):
    """
    Calculate camera linear and angular velocity in world coordinates.
    
    Args:
        ego_linear_velocity: numpy array [vx, vy, vz] - ego vehicle's linear velocity in ego frame
        ego_angular_velocity: numpy array [wx, wy, wz] - ego vehicle's angular velocity in ego frame
        cam2ego_transform: 4x4 transformation matrix from camera to ego frame (or ego to camera if is_cam2ego=False)
        ego2global_transform: 4x4 transformation matrix from ego to world frame
    
    Returns:
        cam_velocity_world: numpy array [vx, vy, vz] - camera's linear velocity in world frame
        cam_angular_velocity_world: numpy array [wx, wy, wz] - camera's angular velocity in world frame
    """
    # Get the translation from ego to camera in ego coordinates
    cam_position_ego = cam2ego_transform[:3, 3]

    # Extract rotation matrix from ego to world
    R_ego2world = ego2global_transform[:3, :3]
    
    # Compute camera's linear velocity in ego frame
    # v_cam = v_ego + w_ego × r_cam/ego
    # Create skew-symmetric matrix for cross product
    w_cross = np.array([
        [0, -ego_angular_velocity[2], ego_angular_velocity[1]],
        [ego_angular_velocity[2], 0, -ego_angular_velocity[0]],
        [-ego_angular_velocity[1], ego_angular_velocity[0], 0]
    ])
    
    # Calculate camera velocity in ego frame
    cam_velocity_ego = ego_linear_velocity + w_cross @ cam_position_ego
    
    # Transform to world frame
    cam_velocity_world = R_ego2world @ cam_velocity_ego
    
    # Angular velocity also needs to be transformed to world frame
    cam_angular_velocity_world = R_ego2world @ ego_angular_velocity
    
    return cam_velocity_world, cam_angular_velocity_world

def invert_distortion(image, intrinsic, distortion):
    intrinsic = intrinsic.copy()
    intrinsic[0, 2] -= 0.5
    intrinsic[1, 2] -= 0.5

    width, height = image.shape[1], image.shape[0]
    new_intrinsic, roi = cv2.getOptimalNewCameraMatrix(
        intrinsic, distortion, (width, height), 1
    )
    map_inverse_distort = cv2.initInverseRectificationMap(
        intrinsic, distortion, None, new_intrinsic, (width, height), cv2.CV_32FC1
    )
    image = cv2.remap(
        image, map_inverse_distort[0], map_inverse_distort[1], 
        interpolation=cv2.INTER_LINEAR
    )
    return image
