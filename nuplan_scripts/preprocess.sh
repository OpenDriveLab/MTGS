#!/bin/bash

CONFIG=$1
NUM_WORKERS=$2
NUM_GPUS=$3
STEP=${4:-1}
PORT=${5:-29500}

print_banner() {
    echo "=================================="
    echo "Executing: $1"
    echo "=================================="
}

run_command() {
    if [ $STEP -le $1 ]; then
        print_banner "$2"
        eval "$2"
        if [ $? -ne 0 ]; then
            exit 1
        fi
    fi
}

command_list=(
    "python -m nuplan_scripts.nuplan_video_processing --config $CONFIG --prefilter --num_workers $NUM_WORKERS"

    "python -m nuplan_scripts.export_videos --config $CONFIG --num_workers $NUM_WORKERS"

    "python -m nuplan_scripts.collect_raw_data --config $CONFIG --num_workers $NUM_WORKERS"

    "accelerate launch --num_processes $NUM_GPUS --main_process_port $PORT -m nuplan_scripts.generate_semantic_mask --config $CONFIG"

    "python -m nuplan_scripts.lidar_registration_multi_traversal --config $CONFIG"

    "python -m nuplan_scripts.ba_multi_traversal --config $CONFIG"

    "accelerate launch --num_processes $NUM_GPUS --main_process_port $PORT -m nuplan_scripts.generate_dense_depth --config $CONFIG --num_workers $NUM_WORKERS"

    "python -m nuplan_scripts.stack_RGB_point_cloud --config $CONFIG --num_workers $NUM_WORKERS"
)

for i in $(seq 1 ${#command_list[@]}); do
    run_command $i "${command_list[$i - 1]}"
done
