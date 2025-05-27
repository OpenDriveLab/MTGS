#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import numpy as np

def compute_transformation_matrix_with_scaling(source_points, target_points):
    assert source_points.shape == target_points.shape

    centroid_source = np.mean(source_points, axis=0)
    centroid_target = np.mean(target_points, axis=0)

    source_centered = source_points - centroid_source
    target_centered = target_points - centroid_target

    H = source_centered.T @ target_centered

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    scale = np.sum(S) / np.sum(source_centered ** 2)

    t = centroid_target.T - (R * scale) @ centroid_source.T

    return scale, R, t

def compute_transformation_matrix_without_scaling(source_points, target_points):
    assert source_points.shape == target_points.shape

    centroid_source = np.mean(source_points, axis=0)
    centroid_target = np.mean(target_points, axis=0)

    source_centered = source_points - centroid_source
    target_centered = target_points - centroid_target

    H = source_centered.T @ target_centered

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_target.T - R @ centroid_source.T

    return R, t
