#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchmetrics import Metric

class DINOv2Similarity:
    def __init__(self):
        """Initialize DINOv2 similarity metric.
        
        Args:
            device: Device to run the model on
        """
        self._load_model()

    def _load_model(self):
        """Load DINOv2 model from torch hub."""
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.model.eval()
        
        # Define the image transformation
        self.transform = transforms.Compose([
            transforms.Resize(518),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5)
            ),
        ])

    def prepare_mask(self, mask: torch.Tensor):
        """Prepare mask for DINOv2 similarity calculation.
        
        Args:
            mask: Mask tensor of shape [H, W]
            
        Returns:
            Patch weights of shape [1369]
        """
        if mask.dim() == 3:
            mask = mask.squeeze()
        mask = mask.to(torch.float32)
        mask_pil = TF.to_pil_image(mask)

        mask_resized = TF.resize(mask_pil, [518], interpolation=Image.NEAREST)
        mask_cropped = TF.center_crop(mask_resized, [518, 518])

        # Convert to [518,518]
        mask_tensor = TF.to_tensor(mask_cropped).squeeze(0) > 0.5  # [518,518]
        mask_tensor = mask_tensor.float().to(mask.device)

        # Visible portion
        patches = mask_tensor.unfold(0, 14, 14).unfold(1, 14, 14)  # [37,37,14,14]
        patch_counts = patches.sum(dim=(-1,-2))                    # [37,37]
        patch_weights = patch_counts / (14*14)                     # [37,37]

        return patch_weights.flatten()  # [1369]

    def _calculate_weighted_cossim(self, feat_1, feat_2, patch_weights):
        """Calculate weighted cosine similarity between features.
        
        Args:
            feat_1: Features from rendered image [B, patch_num, C]
            feat_2: Features from ground truth image [B, patch_num, C]
            patch_weights: Weights for each patch based on mask [patch_num]
            
        Returns:
            Weighted cosine similarity
        """
        cos_per_patch = torch.nn.CosineSimilarity(dim=2)(feat_1, feat_2)

        # Apply patch weights
        total_weight = patch_weights.sum()
        if total_weight > 1e-6:
            # Weight each patch similarity and then sum
            weighted_sim = (cos_per_patch * patch_weights).sum() / total_weight
            return weighted_sim
        return torch.tensor(0.0, device=feat_1.device)  # all mask case

    def _preprocess_image(self, img):
        """Preprocess image for DINOv2 model.
        
        Args:
            img: Image tensor of shape [H, W, 3], [3, H, W], [1, H, W, 3] or PIL Image
            
        Returns:
            Preprocessed image tensor
        """
        if img.dim() == 3 and img.shape[-1] == 3:  # [H, W, 3]
            img_pil = TF.to_pil_image(img.permute(2, 0, 1))
        elif img.dim() == 3 and img.shape[0] == 3:  # [3, H, W]
            img_pil = TF.to_pil_image(img)
        elif img.dim() == 4 and img.shape[0] == 1:
            img_pil = TF.to_pil_image(img.squeeze(0))

        return self.transform(img_pil).unsqueeze(0).to(img.device)
    
    def extract_features(self, img):
        """Extract features from an image using DINOv2.
        
        Args:
            img: Image tensor of shape [H, W, 3] or PIL Image
            
        Returns:
            Features from DINOv2 model
        """
        input_img = self._preprocess_image(img)
        with torch.no_grad():
            features = self.model.forward_features(input_img)['x_norm_patchtokens']
        return features

    def similarity(self, img1, img2, mask):
        """Calculate masked DINOv2 similarity between two images.
        
        Args:
            img1: First image, shape [H, W, 3], [3, H, W] or[1, 3, H, W], RGB, [0, 1]
            img2: Second image, shape [H, W, 3], [3, H, W] or [1, 3, H, W], RGB, [0, 1]
            mask: Binary mask, shape [H, W], [H, W, 1] or [1, H, W]
        Returns:
            Similarity score (higher is better)
        """
        self.model.to(img1.device)
        # Extract features
        feat1 = self.extract_features(img1)
        feat2 = self.extract_features(img2)

        # Prepare mask
        patch_weights = self.prepare_mask(mask)

        # Calculate similarity
        similarity = self._calculate_weighted_cossim(feat1, feat2, patch_weights)

        return similarity.item()
