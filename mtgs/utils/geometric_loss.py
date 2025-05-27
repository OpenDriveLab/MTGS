import abc
from enum import Enum
from typing import Literal, Optional
import cv2

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchmetrics.image import (
    MultiScaleStructuralSimilarityIndexMeasure,
    StructuralSimilarityIndexMeasure,
)

from mtgs.utils.camera_utils import get_means3d_backproj

class DepthLossType(Enum):
    """Enum for specifying depth loss"""

    MSE = "mse"
    L1 = "L1"
    LogL1 = "LogL1"
    HuberL1 = "HuberL1"
    TV = "TV"
    EdgeAwareLogL1 = "EdgeAwareLogL1"
    EdgeAwareTV = "EdgeAwareTV"
    InverseL1 = "InverseL1"

class DepthLoss(nn.Module):
    """Factory method class for various depth losses"""

    def __init__(self, depth_loss_type: DepthLossType, **kwargs):
        super().__init__()
        self.depth_loss_type = depth_loss_type
        self.kwargs = kwargs
        self.loss = self._get_loss_instance()

    @abc.abstractmethod
    def forward(self, *args) -> Tensor:
        return self.loss(*args)

    def _get_loss_instance(self) -> nn.Module:
        if self.depth_loss_type == DepthLossType.MSE:
            return torch.nn.MSELoss()
        if self.depth_loss_type == DepthLossType.L1:
            return L1(**self.kwargs)
        elif self.depth_loss_type == DepthLossType.InverseL1:
            return InverseL1(**self.kwargs)
        elif self.depth_loss_type == DepthLossType.LogL1:
            return LogL1(**self.kwargs)
        elif self.depth_loss_type == DepthLossType.HuberL1:
            return HuberL1(**self.kwargs)
        elif self.depth_loss_type == DepthLossType.EdgeAwareLogL1:
            return EdgeAwareLogL1(**self.kwargs)
        elif self.depth_loss_type == DepthLossType.EdgeAwareTV:
            return EdgeAwareTV(**self.kwargs)
        elif self.depth_loss_type == DepthLossType.TV:
            return TVLoss(**self.kwargs)
        else:
            raise ValueError(f"Unsupported loss type: {self.depth_loss_type}")
        

class DSSIML1(nn.Module):
    """Implementation of DSSIM+L1 loss

    Args:
        implementation: use 'scalar' to return scalar value, use 'per-pixel' to return per-pixel loss

    reference:
        https://arxiv.org/abs/1909.09051 and
        https://arxiv.org/abs/1609.03677

    original implementation uses 3x3 kernel size and single resolution SSIM
    """

    def __init__(
        self,
        kernel_size: int = 3,
        alpha: float = 0.85,
        single_resolution: bool = True,
        implementation: Literal["scalar", "per-pixel"] = "per-pixel",
        **kwargs,
    ):
        super().__init__()
        self.implementation = implementation

        # torchvision SSIM returns a scalar value for SSIM, not per pixel tensor
        if single_resolution and implementation == "scalar":
            self.ssim = StructuralSimilarityIndexMeasure(
                gaussian_kernel=True,
                kernel_size=kernel_size,
                reduction="elementwise_mean",
            )
        elif implementation == "scalar":
            self.ssim = MultiScaleStructuralSimilarityIndexMeasure(
                gaussian_kernel=True,
                kernel_size=kernel_size,
                reduction="elementwise_mean",
            )
        else:
            self.mu_x_pool = nn.AvgPool2d(kernel_size, 1)
            self.mu_y_pool = nn.AvgPool2d(kernel_size, 1)
            self.sig_x_pool = nn.AvgPool2d(kernel_size, 1)
            self.sig_y_pool = nn.AvgPool2d(kernel_size, 1)
            self.sig_xy_pool = nn.AvgPool2d(kernel_size, 1)
            self.refl = nn.ReflectionPad2d(int((kernel_size - 1) / 2))
            self.C1 = 0.01**2
            self.C2 = 0.03**2

        self.alpha = alpha

    def ssim_per_pixel(self, pred, gt):
        x = self.refl(pred)
        y = self.refl(gt)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x = self.sig_x_pool(x**2) - mu_x**2
        sigma_y = self.sig_y_pool(y**2) - mu_y**2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x**2 + mu_y**2 + self.C1) * (sigma_x + sigma_y + self.C2)
        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

    def forward(self, pred: Tensor, gt: Tensor):
        """Compute DSSIM+L1 loss"""
        if (pred.shape[-1] == 1 or pred.shape[-1] == 3) and pred.dim() == 3:
            pred = pred.permute(2, 0, 1).unsqueeze(0)

        if (gt.shape[-1] == 1 or pred.shape[-1] == 3) and gt.dim() == 3:
            gt = gt.permute(2, 0, 1).unsqueeze(0)

        if self.implementation == "scalar":
            abs_diff = torch.abs(pred - gt)
            l1_loss = abs_diff.mean()
            ssim_loss = self.ssim(pred, gt)
            return self.alpha * (1 - ssim_loss) / 2 + (1 - self.alpha) * l1_loss
        else:
            abs_diff = torch.abs(pred - gt)
            l1_loss = abs_diff.mean(1, True)
            ssim_loss = self.ssim_per_pixel(pred, gt).mean(1, True)
            return self.alpha * ssim_loss + (1 - self.alpha) * l1_loss


class L1(nn.Module):
    """L1 loss"""

    def __init__(
        self, implementation: Literal["scalar", "per-pixel"] = "scalar", **kwargs
    ):
        super().__init__()
        self.implementation = implementation

    def forward(self, pred, gt):
        if self.implementation == "scalar":
            return torch.abs(pred - gt).mean()
        else:
            return torch.abs(pred - gt)

class InverseL1(nn.Module):
    """Inverse L1 loss"""

    def __init__(
        self, implementation: Literal["scalar", "per-pixel"] = "scalar", **kwargs
    ):
        super().__init__()
        self.implementation = implementation

    def forward(self, pred, gt):
        if self.implementation == "scalar":
            pred = 1 / (pred + 1e-6)
            gt = 1 / (gt + 1e-6)
            return torch.abs(pred - gt).mean()
        else:
            pred = 1 / (pred + 1e-6)
            gt = 1 / (gt + 1e-6)
            return torch.abs(pred - gt)


class LogL1(nn.Module):
    """Log-L1 loss"""

    def __init__(
        self, implementation: Literal["scalar", "per-pixel"] = "scalar", **kwargs
    ):
        super().__init__()
        self.implementation = implementation

    def forward(self, pred, gt):
        if self.implementation == "scalar":
            return torch.log(1 + torch.abs(pred - gt)).mean()
        else:
            return torch.log(1 + torch.abs(pred - gt))


class EdgeAwareLogL1(nn.Module):
    """Gradient aware Log-L1 loss"""

    def __init__(
        self, implementation: Literal["scalar", "per-pixel"] = "scalar", **kwargs
    ):
        super().__init__()
        self.implementation = implementation
        self.logl1 = LogL1(implementation="per-pixel")

    def forward(self, pred: Tensor, gt: Tensor, rgb: Tensor, mask: Optional[Tensor]):
        logl1 = self.logl1(pred, gt)

        grad_img_x = torch.mean(
            torch.abs(rgb[..., :, :-1, :] - rgb[..., :, 1:, :]), -1, keepdim=True
        )
        grad_img_y = torch.mean(
            torch.abs(rgb[..., :-1, :, :] - rgb[..., 1:, :, :]), -1, keepdim=True
        )
        lambda_x = torch.exp(-grad_img_x)
        lambda_y = torch.exp(-grad_img_y)

        loss_x = lambda_x * logl1[..., :, :-1, :]
        loss_y = lambda_y * logl1[..., :-1, :, :]

        if self.implementation == "per-pixel":
            if mask is not None:
                loss_x[~mask[..., :, :-1, :]] = 0
                loss_y[~mask[..., :-1, :, :]] = 0
            return loss_x[..., :-1, :, :] + loss_y[..., :, :-1, :]

        if mask is not None:
            assert mask.shape[:2] == pred.shape[:2]
            loss_x = loss_x[mask[..., :, :-1, :]]
            loss_y = loss_y[mask[..., :-1, :, :]]

        if self.implementation == "scalar":
            return loss_x.mean() + loss_y.mean()


class HuberL1(nn.Module):
    """L1+huber loss"""

    def __init__(
        self,
        tresh=0.2,
        implementation: Literal["scalar", "per-pixel"] = "scalar",
        **kwargs,
    ):
        super().__init__()
        self.tresh = tresh
        self.implementation = implementation

    def forward(self, pred, gt):
        mask = gt != 0
        l1 = torch.abs(pred[mask] - gt[mask])
        d = self.tresh * torch.max(l1)
        loss = torch.where(l1 < d, ((pred - gt) ** 2 + d**2) / (2 * d), l1)
        if self.implementation == "scalar":
            return loss.mean()
        else:
            return loss


class EdgeAwareTV(nn.Module):
    """Edge Aware Smooth Loss"""

    def __init__(self):
        super().__init__()

    def forward(self, depth: Tensor, rgb: Tensor):
        """
        Args:
            depth: [batch, H, W, 1]
            rgb: [batch, H, W, 3]
        """
        grad_depth_x = torch.abs(depth[..., :, :-1, :] - depth[..., :, 1:, :])
        grad_depth_y = torch.abs(depth[..., :-1, :, :] - depth[..., 1:, :, :])

        grad_img_x = torch.mean(
            torch.abs(rgb[..., :, :-1, :] - rgb[..., :, 1:, :]), -1, keepdim=True
        )
        grad_img_y = torch.mean(
            torch.abs(rgb[..., :-1, :, :] - rgb[..., 1:, :, :]), -1, keepdim=True
        )

        grad_depth_x *= torch.exp(-grad_img_x)
        grad_depth_y *= torch.exp(-grad_img_y)

        return grad_depth_x.mean() + grad_depth_y.mean()


class TVLoss(nn.Module):
    """TV loss"""

    def __init__(self):
        super().__init__()

    def forward(self, pred):
        """
        Args:
            pred: [batch, H, W, 3]

        Returns:
            tv_loss: [batch]
        """
        h_diff = pred[..., :, :-1, :] - pred[..., :, 1:, :]
        w_diff = pred[..., :-1, :, :] - pred[..., 1:, :, :]
        return torch.mean(torch.abs(h_diff)) + torch.mean(torch.abs(w_diff))

def mean_angular_error(pred: Tensor, gt: Tensor) -> Tensor:
    """Compute the mean angular error between predicted and reference normals

    Args:
        predicted_normals: [B, C, H, W] tensor of predicted normals
        reference_normals : [B, C, H, W] tensor of gt normals

    Returns:
        mae: [B, H, W] mean angular error
    """
    dot_products = torch.sum(gt * pred, dim=1)  # over the C dimension
    # Clamp the dot product to ensure valid cosine values (to avoid nans)
    dot_products = torch.clamp(dot_products, -1.0, 1.0)
    # Calculate the angle between the vectors (in radians)
    mae = torch.acos(dot_products)
    return mae

def calculate_depth_ncc_loss(pred_depth, gt_depth, patch_size=7, stride=7, mask=None):
    pred_depth = pred_depth.squeeze(-1)
    gt_depth = gt_depth.squeeze(-1)
    H, W = pred_depth.shape
    pad = patch_size // 2
    mask = mask.squeeze(-1).float()
    pred_depth_patches = F.unfold(pred_depth.unsqueeze(0).unsqueeze(0), kernel_size=patch_size, padding=pad, stride=stride)
    gt_depth_patches = F.unfold(gt_depth.unsqueeze(0).unsqueeze(0), kernel_size=patch_size, padding=pad, stride=stride)
    mask_depth_patches = F.unfold(mask.unsqueeze(0).unsqueeze(0), kernel_size=patch_size, padding=pad, stride=stride)
    valid_mask = mask_depth_patches.all(dim=1)
    valid_mask = valid_mask.squeeze(0)
    pred_depth_patches = pred_depth_patches[:, :, valid_mask]
    gt_depth_patches = gt_depth_patches[:, :, valid_mask]
    pred_mean = pred_depth_patches.mean(dim=1, keepdim=True)
    gt_mean = gt_depth_patches.mean(dim=1, keepdim=True)
    pred_centered = pred_depth_patches - pred_mean
    gt_centered = gt_depth_patches - gt_mean

    pred_std = torch.sqrt((pred_centered ** 2).mean(dim=1, keepdim=True) + 1e-8)
    gt_std = torch.sqrt((gt_centered ** 2).mean(dim=1, keepdim=True) + 1e-8)
    
    pred_normalized = pred_centered / pred_std
    gt_normalized = gt_centered / gt_std
    ncc = (pred_normalized * gt_normalized).mean(dim=1)
    ncc_loss = 1 - ncc.mean()

    return ncc_loss

def pcd_to_normal(xyz: Tensor):
    hd, wd, _ = xyz.shape
    bottom_point = xyz[..., 2:hd, 1 : wd - 1, :]
    top_point = xyz[..., 0 : hd - 2, 1 : wd - 1, :]
    right_point = xyz[..., 1 : hd - 1, 2:wd, :]
    left_point = xyz[..., 1 : hd - 1, 0 : wd - 2, :]
    left_to_right = right_point - left_point
    bottom_to_top = top_point - bottom_point
    xyz_normal = torch.cross(left_to_right, bottom_to_top, dim=-1)
    xyz_normal = torch.nn.functional.normalize(xyz_normal, p=2, dim=-1)
    xyz_normal = torch.nn.functional.pad(
        xyz_normal.permute(2, 0, 1), (1, 1, 1, 1), mode="constant"
    ).permute(1, 2, 0)
    return xyz_normal

def normal_from_depth_image(
    depths: Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_size: tuple,
    c2w: Tensor,
    device: torch.device,
    smooth: bool = False,
):
    """estimate normals from depth map"""
    if smooth:
        if torch.count_nonzero(depths) > 0:
            print("Input depth map contains 0 elements, skipping smoothing filter")
        else:
            kernel_size = (9, 9)
            depths = torch.from_numpy(
                cv2.GaussianBlur(depths.cpu().numpy(), kernel_size, 0)
            ).to(device)
    means3d, _ = get_means3d_backproj(depths, fx, fy, cx, cy, img_size, c2w, device)
    means3d = means3d.view(img_size[1], img_size[0], 3)
    normals = pcd_to_normal(means3d)
    return normals
