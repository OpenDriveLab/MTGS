#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import numpy as np
import cv2

import shapely
from shapely.geometry import Polygon

def field_of_view_intrinsic(intrinsic, scale=30, transform_matrix=None):
    """Return camera fov in 2d space, from its intrinsic.
    camera coordinate: x right, y up, z front.

    Args: 
    intrinsic: 3x3 matrix
    scale: in meter
    transform_matrix: 4x4 matrix

    Returns:
        shapely.geometry.Polygon
    """

    half_wide = intrinsic[0, 2] / intrinsic[0, 0]
    fov = np.array([
        [0, 0, 0],
        [half_wide, 0, 1],
        [-half_wide, 0, 1],
    ]) * scale

    if transform_matrix is not None:
        fov = np.dot(transform_matrix[:3, :3], fov.T).T + transform_matrix[:3, 3]
    
    fov_geom = Polygon(fov[:, :2])

    return fov_geom

def get_safe_projs(obj_pts_cam, distCoeffs):
    # https://github.com/opencv/opencv/issues/17768
    # Define a list of booleans to denote if a variable is safe
    obj_pts_safe = np.ones(len(obj_pts_cam), dtype=bool)

    # Define the homogenous coordiantes
    x_homo_vals = (obj_pts_cam[:, 0] / obj_pts_cam[:, 2]).astype(complex)
    y_homo_vals = (obj_pts_cam[:, 1] / obj_pts_cam[:, 2]).astype(complex)

    # Define the distortion terms, and vectorize calculating of powers of x_homo_vals
    #   and y_homo_vals
    k1, k2, p1, p2, k3 = distCoeffs.tolist()
    y_homo_vals_2 = np.power(y_homo_vals, 2)
    y_homo_vals_4 = np.power(y_homo_vals, 4)
    y_homo_vals_6 = np.power(y_homo_vals, 6)

    # Find the bounds on the x_homo coordinate to ensure it is closer than the
    #   inflection point of x_proj as a function of x_homo
    x_homo_min = np.full(x_homo_vals.shape, np.inf)
    x_homo_max = np.full(x_homo_vals.shape, -np.inf)
    for i in range(len(y_homo_vals)):
        # Expanded projection function polynomial coefficients
        x_proj_coeffs = np.array([k3,
                                  0,
                                  k2 + 3*k3*y_homo_vals_2[i],
                                  0,
                                  k1 + 2*k2*y_homo_vals_2[i] + 3*k3*y_homo_vals_4[i],
                                  3*p2,
                                  1 + k1 * y_homo_vals_2[i] + k2 * y_homo_vals_4[i] + k3*y_homo_vals_6[i] + 2*p1*y_homo_vals[i],
                                  p2*y_homo_vals_2[i]])

        # Projection function derivative polynomial coefficients
        x_proj_der_coeffs = np.polyder(x_proj_coeffs)

        # Find the root of the derivative
        roots = np.roots(x_proj_der_coeffs)

        # Get the real roots
        # Approximation of real[np.where(np.isreal(roots))]
        real_roots = np.real(roots[np.where(np.abs(np.imag(roots)) < 1e-10)])

        for real_root in real_roots:
            x_homo_min[i] = np.minimum(x_homo_min[i], real_root)
            x_homo_max[i] = np.maximum(x_homo_max[i], real_root)

    # Check that the x_homo values are within the bounds
    obj_pts_safe *= np.where(x_homo_vals > x_homo_min, True, False)
    obj_pts_safe *= np.where(x_homo_vals < x_homo_max, True, False)

    return obj_pts_safe

def undistort_image_optimal(image, intrinsic, distortion, return_mask=False, interpolation='linear'):
    """
    interpolation: 'nearest', 'linear' or 'cubic'
    """
    interpolation_map = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC
    }

    height, width = image.shape[:2]
    new_intrinsic, roi = cv2.getOptimalNewCameraMatrix(
        intrinsic, distortion, (width, height), 1
    )
    map_undistort_optimal = cv2.initUndistortRectifyMap(
        intrinsic, distortion, None, new_intrinsic, (width, height), cv2.CV_32FC2
    )
    image_optimal = cv2.remap(
        image, map_undistort_optimal[0], map_undistort_optimal[1], 
        interpolation=interpolation_map[interpolation]
    )

    if return_mask:
        valid_mask = np.ones_like(image_optimal, dtype=np.uint8) * 255
        valid_mask = cv2.remap(
            valid_mask, map_undistort_optimal[0], map_undistort_optimal[1], 
            interpolation=cv2.INTER_NEAREST
        )
        return image_optimal, new_intrinsic, roi, valid_mask
    else:
        return image_optimal, new_intrinsic, roi

def undistort_image_keep_focal_length(image, intrinsic, distortion, interpolation='linear'):
    interpolation_map = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC
    }

    height, width = image.shape[:2]
    map_undistort = cv2.initUndistortRectifyMap(
        intrinsic, distortion, None, intrinsic, (width, height), cv2.CV_32FC2
    )
    image_undistorted = cv2.remap(
        image, map_undistort[0], map_undistort[1], 
        interpolation=interpolation_map[interpolation]
    )

    return image_undistorted

def undistort_image_with_cam_info(image, cam_info, return_mask=False, interpolation='linear', mode='optimal'):
    if 'colmap_param' in cam_info:
        intrinsic = cam_info['colmap_param']['cam_intrinsic']
        distortion = cam_info['colmap_param']['distortion']
        intrinsic = intrinsic.copy()
        intrinsic[0, 2] = intrinsic[0, 2] - 0.5
        intrinsic[1, 2] = intrinsic[1, 2] - 0.5
    else:
        intrinsic = cam_info['cam_intrinsic']
        distortion = cam_info['distortion']
    
    if mode == 'optimal':
        return undistort_image_optimal(image, intrinsic, distortion, return_mask, interpolation)
    elif mode == 'keep_focal_length':
        return undistort_image_keep_focal_length(image, intrinsic, distortion, interpolation)
    else:
        raise ValueError(f'Invalid mode: {mode}')
