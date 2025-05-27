#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import os
import sys
import shutil
import platform
import subprocess
import argparse

def run_bundle_adjustment(
    colmap_path, 
    input_model='sparse_model', 
    output_model='sfm_model',
    num_workers=8
):

    colmap_exe = "colmap.bat" if platform.system() == "Windows" else "colmap"

    print("extracting features...")
    colmap_feature_extractor_args = [
        colmap_exe, "feature_extractor",
        "--database_path", f"{colmap_path}/database.db",
        "--image_path", f"{colmap_path}/images",
    ]
    if os.path.exists(f"{colmap_path}/masks"):
        colmap_feature_extractor_args += ["--ImageReader.mask_path", f"{colmap_path}/masks"]
    colmap_feature_extractor_args += ["--SiftExtraction.num_threads", f"{num_workers}"]

    try:
        subprocess.run(colmap_feature_extractor_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing colmap feature_extractor: {e}")
        sys.exit(1)


    print("feature matching...")
    colmap_matches_importer_args = [
        colmap_exe, "matches_importer",
        "--database_path", f"{colmap_path}/database.db",
        "--match_list_path", f"{colmap_path}/image_pairs.txt",
        "--SiftMatching.num_threads", f"{num_workers}"
    ]

    try:
        subprocess.run(colmap_matches_importer_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing colmap matches_importer: {e}")
        sys.exit(1)

    if os.path.exists(os.path.join(colmap_path, output_model)):
        shutil.rmtree(os.path.join(colmap_path, output_model))

    os.makedirs(os.path.join(colmap_path, output_model, "t"), exist_ok=True)
    os.makedirs(os.path.join(colmap_path, output_model, "b"), exist_ok=True)
    os.makedirs(os.path.join(colmap_path, output_model, "t2"), exist_ok=True)
    os.makedirs(os.path.join(colmap_path, output_model, "0"), exist_ok=True)

    colmap_point_triangulator_args = [
        colmap_exe, "point_triangulator",
        "--refine_intrinsics", "1",
        "--Mapper.num_threads", f"{num_workers}",
        "--Mapper.ba_global_function_tolerance", "0.000001",
        "--Mapper.ba_global_max_num_iterations", "30",
        "--Mapper.ba_global_max_refinements", "3",
    ]

    colmap_bundle_adjuster_args = [
        colmap_exe, "bundle_adjuster",
        "--BundleAdjustment.function_tolerance", "0.000001",
        "--BundleAdjustment.max_linear_solver_iterations", "200",
        "--BundleAdjustment.max_num_iterations", "100",
        "--BundleAdjustment.refine_focal_length", "1",
        "--BundleAdjustment.refine_principal_point", "0",
        "--BundleAdjustment.refine_extra_params", "1"
    ]
    # 2 rounds of triangulation + bundle adjustment
    try:
        subprocess.run(colmap_point_triangulator_args + [
            "--database_path", f"{colmap_path}/database.db",
            "--image_path", f"{colmap_path}/images",
            "--input_path", f"{colmap_path}/{input_model}",
            "--output_path", f"{colmap_path}/{output_model}/t",
            ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing colmap_point_triangulator_args: {e}")
        sys.exit(1)

    try:
        subprocess.run(colmap_bundle_adjuster_args + [
            "--input_path", f"{colmap_path}/{output_model}/t",
            "--output_path", f"{colmap_path}/{output_model}/b",
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing colmap_bundle_adjuster_args: {e}")
        sys.exit(1)

    try:
        subprocess.run(colmap_point_triangulator_args + [
            "--database_path", f"{colmap_path}/database.db",
            "--image_path", f"{colmap_path}/images",
            "--input_path", f"{colmap_path}/{output_model}/b",
            "--output_path", f"{colmap_path}/{output_model}/t2",
            ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing colmap_point_triangulator_args: {e}")
        sys.exit(1)

    try:
        subprocess.run(colmap_bundle_adjuster_args + [
            "--input_path", f"{colmap_path}/{output_model}/t2",
            "--output_path", f"{colmap_path}/{output_model}/0",
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing colmap_bundle_adjuster_args: {e}")
        sys.exit(1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--colmap_path', type=str, help='Input colmap dir', required=True)
    parser.add_argument('--input_model', type=str, default='sparse_model')
    parser.add_argument('--output_model', type=str, default='sfm_model')
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    colmap_path = args.colmap_path

    run_bundle_adjustment(colmap_path, args.input_model, args.output_model, args.num_workers)
