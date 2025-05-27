#!/bin/bash

CONFIG=$1
NUM_WORKERS=$2
NUM_GPUS=$3
PORT=${4:-29500}

print_banner() {
    echo "=================================="
    echo "Executing: $1"
    echo "=================================="
}

run_command() {
    print_banner "$1"
    eval "$1"
    if [ $? -ne 0 ]; then
        exit 1
    fi
}

run_command "python -m nuplan_scripts.filter_trajectory --config $CONFIG"

run_command "python -m nuplan_scripts.collect_raw_data --config $CONFIG --num_workers $NUM_WORKERS"

run_command "accelerate launch --num_processes $NUM_GPUS --main_process_port $PORT -m nuplan_scripts.generate_semantic_mask --config $CONFIG"

run_command "python -m nuplan_scripts.lidar_registration_multi_traversal --config $CONFIG"

run_command "python -m nuplan_scripts.ba_multi_traversal --config $CONFIG"

run_command "accelerate launch --num_processes $NUM_GPUS --main_process_port $PORT -m nuplan_scripts.generate_dense_depth --config $CONFIG --num_workers $NUM_WORKERS"

run_command "python -m nuplan_scripts.stack_RGB_point_cloud --config $CONFIG --num_workers $NUM_WORKERS"
