#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import argparse
import numpy as np
from pyquaternion import Quaternion

from nuplan_scripts.utils.colmap_utils.database import COLMAPDatabase

def create_colmap_database(colmap_path, input_model='sparse_model', camera_type='OPENCV'):
    camera_type = 4 if camera_type == 'OPENCV' else 1

    colmap_db = COLMAPDatabase.connect(f"{colmap_path}/database.db")
    colmap_db.create_tables()
    for cameras in open(f"{colmap_path}/{input_model}/cameras.txt", "r").readlines():
        cameras = cameras.strip().split()
        colmap_db.add_camera(
            camera_id=int(cameras[0]),
            model=camera_type, # 1 for PINHOLE, 4 for OPENCV
            width=int(cameras[2]),
            height=int(cameras[3]),
            params=cameras[4:],
            prior_focal_length=True
        )
    for image in open(f"{colmap_path}/{input_model}/images.txt", "r").readlines()[::2]:
        image = image.strip().split()
        colmap_db.add_image(
            name=image[-1],
            camera_id=int(image[-2]),
            image_id=int(image[0])
        )

        prior_position = -Quaternion(*image[1:5]).rotation_matrix.T @ np.array(image[5:8], dtype=np.float64)
        colmap_db.add_pose_prior(
            image_id=int(image[0]),
            position=prior_position,
            coordinate_system=1  # 1 for CARTESIAN
        )

    colmap_db.commit()
    colmap_db.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--colmap_path', type=str, help='Input colmap dir', required=True)
    parser.add_argument('--input_model', type=str, default='sparse_model')
    args = parser.parse_args()

    create_colmap_database(args.colmap_path, args.input_model)
