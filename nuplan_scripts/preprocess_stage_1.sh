#!/bin/bash

CONFIG=$1
NUM_WORKERS=$2

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

run_command "python -m nuplan_scripts.nuplan_video_processing --config $CONFIG --num_workers $NUM_WORKERS"

run_command "python -m nuplan_scripts.export_videos --config $CONFIG --num_workers $NUM_WORKERS"
