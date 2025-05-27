#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import os
import numpy as np
import cv2
from pyquaternion import Quaternion
from shapely.geometry import LineString
import torch
from sqlalchemy import func

from nuplan.database.nuplan_db_orm.ego_pose import EgoPose
from nuplan.database.nuplan_db.nuplan_scenario_queries import get_images_from_lidar_tokens
from nuplan.database.utils.pointclouds.pointcloud import PointCloud
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import CameraChannel

from .constants import NUPLAN_SENSOR_ROOT, NUPLAN_DB_FILES

class CanBus:
    """Wrapper class to convert lidar_can_bus to numpy array"""

    def __init__(self, lidar_pc):
        self.x = lidar_pc.ego_pose.x
        self.y = lidar_pc.ego_pose.y
        self.z = lidar_pc.ego_pose.z

        self.qw = lidar_pc.ego_pose.qw
        self.qx = lidar_pc.ego_pose.qx
        self.qy = lidar_pc.ego_pose.qy
        self.qz = lidar_pc.ego_pose.qz

        self.acceleration_x = lidar_pc.ego_pose.acceleration_x
        self.acceleration_y = lidar_pc.ego_pose.acceleration_y
        self.acceleration_z = lidar_pc.ego_pose.acceleration_z

        self.vx = lidar_pc.ego_pose.vx
        self.vy = lidar_pc.ego_pose.vy
        self.vz = lidar_pc.ego_pose.vz

        self.angular_rate_x = lidar_pc.ego_pose.angular_rate_x
        self.angular_rate_y = lidar_pc.ego_pose.angular_rate_y
        self.angular_rate_z = lidar_pc.ego_pose.angular_rate_z
    
        # loc: [0:3], quat: [3:7], accel: [7:10], velocity: [10:13], rotation_rate: [13:16]
        self.tensor = np.array(
            [
                self.x,
                self.y,
                self.z,
                self.qw,
                self.qx,
                self.qy,
                self.qz,
                self.acceleration_x,
                self.acceleration_y,
                self.acceleration_z,
                self.vx,
                self.vy,
                self.vz,
                self.angular_rate_x,
                self.angular_rate_y,
                self.angular_rate_z,
                0.0,
                0.0,
            ]
        )

def load_lidar(lidar_path, remove_close=False, only_top=False):

    points = PointCloud.parse_from_file(lidar_path).to_pcd_bin2().T

    if only_top:
        points = points[points[:, -1] == 0]

    points = points[:, :3]  # (n, 3)

    if remove_close:
        x_radius = 3.0
        y_radius = 3.0
        z_radius = 2.0
        x_filt = np.abs(points[:, 0]) < x_radius
        y_filt = np.abs(points[:, 1]) < y_radius
        z_filt = np.abs(points[:, 2]) < z_radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt, z_filt))
        return points[not_close]

    return points

def get_closest_start_idx(log, lidar_pcs):
    log_name = log.logfile
    log_file = os.path.join(NUPLAN_DB_FILES, log_name + '.db')

    # find first lidar_pc with all 8 cams ready
    for start_idx in range(0, len(lidar_pcs)):
        retrieved_images = get_images_from_lidar_tokens(
            log_file, [lidar_pcs[start_idx].token], [str(channel.value) for channel in CameraChannel]
        )
        if len(list(retrieved_images)) == 8:
            break

    # find the lidar_pc closest with the camera timestamp
    retrieved_images = get_images_from_lidar_tokens(
        log_file, [lidar_pcs[start_idx].token], ['CAM_F0']
    )
    diff_0 = abs(next(retrieved_images).timestamp - lidar_pcs[start_idx].timestamp)

    retrieved_images = get_images_from_lidar_tokens(
        log_file, [lidar_pcs[start_idx + 1].token], ['CAM_F0']
    )
    diff_1 = abs(next(retrieved_images).timestamp - lidar_pcs[start_idx + 1].timestamp)

    start_idx = start_idx if diff_0 < diff_1 else start_idx + 1
    return start_idx

def get_cam_info_from_lidar_pc(log, lidar_pc, rolling_shutter_s=1/60):
    log_name = log.logfile
    log_file = os.path.join(NUPLAN_DB_FILES, log_name + '.db')
    retrieved_images = get_images_from_lidar_tokens(
        log_file, [lidar_pc.token], [str(channel.value) for channel in CameraChannel]
    )

    # if interp_trans:
    #     neighbours = []
    #     ego_poses_dict = {}
    #     for ego_pose in log.ego_poses:
    #         ego_poses_dict[ego_pose.token] = ego_pose
    #         if abs(ego_pose.timestamp - lidar_pc.ego_pose.timestamp) / 1e6 < 0.5:
    #             neighbours.append(ego_pose)
    #     timestamps = [pose.timestamp for pose in neighbours]
    #     translations = [pose.translation_np for pose in neighbours]
    #     splines = [CubicSpline(timestamps, [translation[i] for translation in translations]) for i in range(2)]

    log_cam_infos = {camera.token : camera for camera in log.cameras}
    cams = {}
    for img in retrieved_images:
        channel = img.channel
        filename = img.filename_jpg
        filepath = os.path.join(NUPLAN_SENSOR_ROOT, filename)
        if not os.path.exists(filepath):
            return None

        # if interp_trans:
            # img_ego_pose = ego_poses_dict[img.ego_pose_token]
            # interpolated_translation = np.array([splines[0](timestamp), splines[1](timestamp), img_ego_pose.z])
            # delta = interpolated_translation - lidar_pc.ego_pose.translation_np
            # delta = np.dot(lidar_pc.ego_pose.quaternion.rotation_matrix.T, delta)

        timestamp = img.timestamp + (rolling_shutter_s * 1e6)
        img_ego_pose = log._session.query(EgoPose).order_by(func.abs(EgoPose.timestamp - timestamp)).first()
        img_e2g = img_ego_pose.trans_matrix
        g2e = lidar_pc.ego_pose.trans_matrix_inv
        img_e2e = g2e @ img_e2g
        cam_info = log_cam_infos[img.camera_token]
        c2img_e = cam_info.trans_matrix
        c2e = img_e2e @ c2img_e

        cams[channel] = dict(
            data_path = filename,
            timestamp = img.timestamp,
            token=img.token,
            sensor2ego_rotation = Quaternion(matrix=c2e[:3, :3]),
            sensor2ego_translation = c2e[:3, 3],
            cam_intrinsic = cam_info.intrinsic_np,
            distortion = cam_info.distortion_np,
        )

    if len(cams) != 8:
        return None

    return cams

def get_rgb_point_cloud(points, lidar2imgs, images):
    """ get rdb point cloud from multi-view images 
    """
    image_shape = np.array([[image.shape[1], image.shape[0]] for image in images])[:, None, :] # [N, 1, 2]

    xyz1 = np.concatenate([points, np.ones_like(points[:, 0:1])], axis=-1)
    pts_2d = lidar2imgs @ xyz1.T
    pts_2d = pts_2d.transpose(0, 2, 1)
    fov_inds = pts_2d[..., 2] > 0.1
    pts_2d[..., 2] = np.clip(pts_2d[..., 2], a_min=1e-5, a_max=99999)
    pts_2d[..., :2] /= pts_2d[..., 2:3]
    pts_2d = pts_2d[..., :2]

    pts_2d /= image_shape # normalize to [0, 1]
    fov_inds = (fov_inds & (pts_2d[..., 0] < 1)
                & (pts_2d[..., 0] >= 0)
                & (pts_2d[..., 1] < 1)
                & (pts_2d[..., 1] >= 0))

    pts_2d = torch.from_numpy(pts_2d)
    pts_2d = pts_2d * 2 - 1  # [-1, +1]

    images = np.array(images, dtype=np.float64)[..., ::-1] / 255.0  # BGR to RGB [0, 1]
    images = torch.from_numpy(images).permute(0, 3, 1, 2)  # to NCHW

    RGB = torch.nn.functional.grid_sample(
        images, pts_2d.unsqueeze(2), align_corners=True, padding_mode='zeros').squeeze(3).permute(0, 2, 1)
    RGB = RGB.cpu().numpy()
    RGB = np.where(fov_inds[..., None], RGB, 0).sum(0)
    fov_num = fov_inds.sum(0)
    fov_mask = fov_num>0
    RGB[fov_mask] /= fov_num[fov_mask][:, None]
    return RGB, fov_mask

def get_semantic_point_cloud(points, lidar2imgs, sem_masks):
    """ get semantic point cloud from multi-view masks
    Args:
        points: [N, 3] point cloud coordinates
        lidar2imgs: [N_views, 4, 4] projection matrices
        sem_masks: [N_views, H, W, 1] semantic masks with integer labels
    Returns:
        semantic_labels: [N] semantic labels for each point
        fov_mask: [N] boolean mask indicating if point is visible in any view
    """
    image_shape = np.array([[mask.shape[1], mask.shape[0]] for mask in sem_masks])[:, None, :] # [N_views, 1, 2]

    xyz1 = np.concatenate([points, np.ones_like(points[:, 0:1])], axis=-1)
    pts_2d = lidar2imgs @ xyz1.T
    pts_2d = pts_2d.transpose(0, 2, 1)
    fov_inds = pts_2d[..., 2] > 0.1
    pts_2d[..., 2] = np.clip(pts_2d[..., 2], a_min=1e-5, a_max=99999)
    pts_2d[..., :2] /= pts_2d[..., 2:3]
    pts_2d = pts_2d[..., :2]

    pts_2d /= image_shape # normalize to [0, 1]
    fov_inds = (fov_inds & (pts_2d[..., 0] < 1)
                & (pts_2d[..., 0] >= 0)
                & (pts_2d[..., 1] < 1)
                & (pts_2d[..., 1] >= 0))

    pts_2d = torch.from_numpy(pts_2d)
    pts_2d = pts_2d * 2 - 1  # [-1, +1]

    # Convert masks to [N_views, 1, H, W] format for grid_sample
    masks = torch.from_numpy(sem_masks).permute(0, 3, 1, 2).to(torch.float64)

    # Use nearest neighbor interpolation for discrete labels
    labels = torch.nn.functional.grid_sample(
        masks, pts_2d.unsqueeze(2), 
        mode='nearest', 
        align_corners=True,
        padding_mode='border').squeeze(3).permute(0, 2, 1)
    
    labels = labels.cpu().numpy()
    labels = labels.astype(np.int32)
    
    # Mask invalid projections with -1
    labels = np.where(fov_inds[..., None], labels, -1)
    
    # Get first valid label for each point
    first_valid_idx = np.argmax(labels >= 0, axis=0)
    rows = first_valid_idx.squeeze()
    cols = np.arange(len(points))
    semantic_labels = labels[rows, cols, 0]
    
    # Create fov mask
    fov_mask = (labels >= 0).any(axis=0).squeeze()
    
    # Set labels to 0 for points not visible in any view
    semantic_labels = np.where(fov_mask, semantic_labels, 0)

    return semantic_labels, fov_mask

def get_box_info_from_lidar_pc(lidar_pc, with_parking_cars=True):

    boxes = lidar_pc.lidar_boxes
    if len(boxes) == 0:
        info = {}
        info['gt_boxes'] = np.zeros((0, 7))
        info['gt_names'] = np.zeros((0,))
        info['gt_velocity'] = np.zeros((0, 2))
        info['gt_velocity_3d'] = np.zeros((0, 3))
        info['gt_confidence'] = np.zeros((0,))
        info['instance_tokens'] = np.zeros((0,))
        info['track_tokens'] = np.zeros((0,))
        return info

    # get the box id for tracking the box in the scene
    instance_tokens = np.array([item.token for item in boxes])
    track_tokens = np.array([item.track_token for item in boxes])
    confidence = np.array([item.confidence for item in boxes])

    e2g_r_mat = lidar_pc.ego_pose.quaternion.rotation_matrix
    inv_ego_r = e2g_r_mat.T
    ego_yaw = lidar_pc.ego_pose.quaternion.yaw_pitch_roll[0]

    locs_global = np.array([b.translation_np for b in boxes])
    locs = (locs_global - lidar_pc.ego_pose.translation_np) @ inv_ego_r.T

    dims = np.array([[b.length, b.width, b.height] for b in boxes]).reshape(-1, 3)
    rots = np.array([b.yaw for b in boxes]).reshape(-1, 1) - ego_yaw

    velocity_3d = np.array([[b.vx, b.vy, b.vz] for b in boxes]).reshape(-1, 3)
    velocity_3d = velocity_3d @ inv_ego_r.T

    names = [box.category.name for box in boxes]
    names = np.array(names)
 
    # dynamic objects mask
    if not with_parking_cars:
        only_foreground_mask = np.array([name in ['bicycle', 'pedestrian'] for name in names])
        scene_dynamic_mask = []
        for box in boxes:
            if box.category.name != 'vehicle':
                scene_dynamic_mask.append(False)
                continue
            velo = np.array([[b.vx, b.vy] for b in box._get_box_items()[1][::5]])
            scene_dynamic = (np.linalg.norm(velo, axis=1) > 0.5).any()
            scene_dynamic_mask.append(scene_dynamic)
        scene_dynamic_mask = np.array(scene_dynamic_mask, dtype=bool)
        dynamic_mask = only_foreground_mask | scene_dynamic_mask
    else:
        dynamic_mask = np.array([name in ['vehicle', 'bicycle', 'pedestrian'] for name in names])

    gt_boxes = np.concatenate([locs, dims, rots], axis=1)

    assert len(gt_boxes) == len(
        boxes), f'{len(gt_boxes)}, {len(boxes)}'

    info = {}
    info['gt_boxes'] = gt_boxes[dynamic_mask]
    info['gt_names'] = names[dynamic_mask]
    info['gt_velocity'] = velocity_3d[dynamic_mask, :2]
    info['gt_velocity_3d'] = velocity_3d[dynamic_mask]
    info['gt_confidence'] = confidence[dynamic_mask]
    info['instance_tokens'] = instance_tokens[dynamic_mask]
    info['track_tokens'] = track_tokens[dynamic_mask]

    return info

def adjust_brightness_single_frame(info, lidar2imgs, images, points=None):
    """ adjust the brightness of the multiview image to front view
        the matching points is come from the lidar point cloud
    """
    image_shape = np.array([[image.shape[1], image.shape[0]] for image in images])[:, None, :]
    if points is None:
        points = load_lidar(
            os.path.join(NUPLAN_SENSOR_ROOT, info['lidar_path']), remove_close=False)

    xyz1 = np.concatenate([points, np.ones_like(points[:, 0:1])], axis=-1)
    pts_2d = lidar2imgs @ xyz1.T
    pts_2d = pts_2d.transpose(0, 2, 1)
    fov_inds = pts_2d[..., 2] > 0.1
    pts_2d[..., 2] = np.clip(pts_2d[..., 2], a_min=1e-5, a_max=99999)
    pts_2d[..., :2] /= pts_2d[..., 2:3]
    pts_2d = pts_2d[..., :2]

    pts_2d /= image_shape # normalize to [0, 1]
    fov_inds = (fov_inds & (pts_2d[..., 0] < 1)
                & (pts_2d[..., 0] >= 0)
                & (pts_2d[..., 1] < 1)
                & (pts_2d[..., 1] >= 0))

    pts_2d = torch.from_numpy(pts_2d)
    pts_2d = pts_2d * 2 - 1  # [-1, +1]

    imgs = np.array(images, dtype=np.float64)[..., ::-1] / 255.0  # BGR to RGB [0, 1]
    imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2)  # to NCHW

    RGB = torch.nn.functional.grid_sample(
        imgs, pts_2d.unsqueeze(2), align_corners=True, padding_mode='zeros').squeeze(3).permute(0, 2, 1)
    RGB = RGB.cpu().numpy()

    matches = [
        ('CAM_F0', 'CAM_L0'), ('CAM_F0', 'CAM_R0'),
        ('CAM_L0', 'CAM_L1'), ('CAM_R0', 'CAM_R1'),
        ('CAM_L1', 'CAM_L2'), ('CAM_R1', 'CAM_R2'),
        ('CAM_L2', 'CAM_B0'), ('CAM_R2', 'CAM_B0')
    ]
    overlapped_regions = {
        ('CAM_F0', 'CAM_L0'): (slice(None, 100), slice(-100, None)),
        ('CAM_L0', 'CAM_L1'): (slice(None, 100), slice(-100, None)),
        ('CAM_L1', 'CAM_L2'): (slice(None, 100), slice(-100, None)),
        ('CAM_L2', 'CAM_B0'): (slice(None, 100), slice(-100, None)),
        ('CAM_F0', 'CAM_R0'): (slice(-100, None), slice(None, 100)),
        ('CAM_R0', 'CAM_R1'): (slice(-100, None), slice(None, 100)),
        ('CAM_R1', 'CAM_R2'): (slice(-100, None), slice(None, 100)),
        ('CAM_R2', 'CAM_B0'): (slice(-100, None), slice(None, 100)),
    }

    CAMS = list(info['cams'].keys())
    for cam in info['cams']:
        info['cams'][cam].pop('v_adjust', None)

    info['cams']['CAM_F0']['v_adjust'] = 1.
    for pair in matches:
        cam1, cam2 = pair
        cam1_info = info['cams'][cam1]
        cam2_info = info['cams'][cam2]
        cam1_idx = list(info['cams'].keys()).index(cam1)
        cam2_idx = list(info['cams'].keys()).index(cam2)
        cam1_rgb = RGB[cam1_idx]
        cam2_rgb = RGB[cam2_idx]
        cam1_fov = fov_inds[cam1_idx]
        cam2_fov = fov_inds[cam2_idx]
        overlap_mask = cam1_fov & cam2_fov
        if overlap_mask.sum() < 10:
            # if the overlap region is too small, use the predefined region
            # the number 10 is set arbitrarily
            overlap = overlapped_regions[pair]
            cam1_rgb = images[CAMS.index(cam1)][:, overlap[0], ::-1].reshape(1, -1, 3)
            cam2_rgb = images[CAMS.index(cam2)][:, overlap[1], ::-1].reshape(1, -1, 3)
        else:
            cam1_rgb = (cam1_rgb[overlap_mask] * 255).astype(np.uint8)[None]
            cam2_rgb = (cam2_rgb[overlap_mask] * 255).astype(np.uint8)[None]
        cam1_hsv = cv2.cvtColor(cam1_rgb, cv2.COLOR_RGB2HSV)[0].astype(np.float64)
        cam2_hsv = cv2.cvtColor(cam2_rgb, cv2.COLOR_RGB2HSV)[0].astype(np.float64)
        v_ratio = np.mean(cam1_hsv[..., 2]) / np.mean(cam2_hsv[..., 2])

        base_ratio = cam1_info.get('v_adjust', 1.)
        ratio = base_ratio * v_ratio
        if 'v_adjust' not in cam2_info:
            cam2_info['v_adjust'] = ratio
        else:
            cam2_info['v_adjust'] = (cam2_info['v_adjust'] + ratio) / 2

    adjust_factors = [cam['v_adjust'] for cam in info['cams'].values()]
    mean_factor = np.mean(adjust_factors)
    for cam in info['cams']:
        info['cams'][cam]['v_adjust'] /= mean_factor

def adjust_brightness(image, factor):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float64)
    image_hsv[..., 2] = (image_hsv[..., 2] * factor)
    image_hsv[..., 2] = np.clip(image_hsv[..., 2], 0, 255)
    image_hsv = image_hsv.astype(np.uint8)
    image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
    return image

def fix_pts_interpolate(line, n_points):
    ls = LineString(line)
    distances = np.linspace(0, ls.length, n_points)
    line = np.array([ls.interpolate(distance).coords[0] for distance in distances])
    return line
