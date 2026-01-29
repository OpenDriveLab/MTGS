#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import os
import shutil
import argparse
import subprocess
import multiprocessing
import time
import torch
import fcntl
import socket
from pathlib import Path

from nuplan_scripts.utils.constants import CONSOLE
from nuplan_scripts.utils.config import load_config

GLOBAL_CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES', None)


class TaskManager:
    """Task manager for multi-machine coordination"""
    def __init__(self, output_dir, config_dir):
        
        self.config_dir = config_dir
        self.configs = sorted(os.listdir(config_dir))
        self.configs = [os.path.join(config_dir, config) for config in self.configs]

        self.task_lock_dir = os.path.join(output_dir, '.task_locks')
        os.makedirs(self.task_lock_dir, exist_ok=True)

        # Get machine identifier
        self.machine_id = socket.gethostname()
        
    def is_task_completed(self, config_path):
        """Check if task is already completed"""
        config = load_config(config_path)
        config_id = config.road_block_name
        
        # Check if config file exists in output (completed)
        config_output_file = Path(config.data_root) / "configs" / f"{config_id}.yaml"
        return config_output_file.exists()

    def is_task_running(self, config_path):
        """Check if task is currently running by checking output directory"""
        config = load_config(config_path)
        config_id = config.road_block_name
        output_dir = os.path.join(config.data_root, config_id)
        return os.path.exists(output_dir) and not self.is_task_completed(config_path)
    
    def acquire_task(self, config_path):
        """Try to acquire a task using file locking"""
        config = load_config(config_path)
        config_id = config.road_block_name
        lock_file = os.path.join(self.task_lock_dir, f"{config_id}.lock")
        if os.path.exists(lock_file):
            return False, None, None
        
        try:
            # Try to create and lock the file
            fd = os.open(lock_file, os.O_CREAT | os.O_RDWR)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            
            # Double-check: verify task is still available after acquiring lock
            if self.is_task_completed(config_path) or self.is_task_running(config_path):
                # Task was taken by another process, release lock
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
                return False, None, None
            
            # Write machine info to lock file
            with open(lock_file, 'w') as f:
                f.write(f"Machine: {self.machine_id}\nTime: {time.strftime('%Y-%m-%d %H:%M:%S')}\nConfig: {config_path}\n")
            
            return True, fd, lock_file
        except (OSError, IOError):
            # File is locked by another process
            return False, None, None

    def release_task(self, fd, lock_file):
        """Release task lock"""
        if fd is not None:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

        try:
            os.remove(lock_file)
        except FileNotFoundError:
            pass

    def get_next_available_task(self):
        """Get next available task with atomic acquisition"""
        # Shuffle configs to avoid all machines starting with the same task
        import random
        shuffled_configs = self.configs.copy()
        random.shuffle(shuffled_configs)
        
        for config_path in shuffled_configs:
            if not self.is_task_completed(config_path) and not self.is_task_running(config_path):
                # Try to acquire this task
                acquired, fd, lock_file = self.acquire_task(config_path)
                if acquired:
                    return config_path, fd, lock_file
        return None, None, None
    
    def get_available_tasks(self):
        """Get list of tasks that are not completed or running (for display only)"""
        available_tasks = []
        for config_path in self.configs:
            if not self.is_task_completed(config_path) and not self.is_task_running(config_path):
                available_tasks.append(config_path)
        return available_tasks


class GPUManager:
    def __init__(self, lock_dir="/tmp/gpu_locks"):
        self.lock_dir = lock_dir
        os.makedirs(lock_dir, exist_ok=True)
        self.available_gpus = GLOBAL_CUDA_VISIBLE_DEVICES.split(',') if GLOBAL_CUDA_VISIBLE_DEVICES else list(range(torch.cuda.device_count()))

    def acquire_gpu(self):
        while True:
            for gpu_id in self.available_gpus:
                lock_file = os.path.join(self.lock_dir, f"gpu_{gpu_id}.lock")
                try:
                    # Try to create lock file - will fail if it already exists
                    fd = os.open(lock_file, os.O_CREAT | os.O_EXCL)
                    os.close(fd)
                    return gpu_id, lock_file
                except FileExistsError:
                    continue
            time.sleep(1)  # Wait before trying again

    def release_gpu(self, lock_file):
        try:
            os.remove(lock_file)
        except FileNotFoundError:
            pass

def run_reconstruction(task_manager, gpu_manager, args):
    """Run reconstruction with atomic task acquisition"""
    # Try to acquire the next available task
    config_path, task_fd, task_lock_file = task_manager.get_next_available_task()
    if config_path is None:
        return False  # No more tasks available
    
    gpu_id, gpu_lock_file = gpu_manager.acquire_gpu()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Load config to get task info
    config = load_config(config_path)
    config_id = config.road_block_name
    output_dir = os.path.join(config.data_root, config_id)
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, f'log_{time.strftime("%Y%m%d_%H%M%S")}.log')

    success = True
    try:
        CONSOLE.log(f'Reconstruction for {config_id} started (GPU {gpu_id})')
        start_time = time.time()
        command = [
            "bash",
            "nuplan_scripts/auto_reconstruction/recon_single_config.sh",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--export-dir",
            str(args.export_dir),
            "--workers",
            str(args.workers),
        ]

        with open(log_file, 'w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=log)
            process.communicate()
            if process.returncode != 0:
                raise Exception(f'Reconstruction for {config_id} failed')
        end_time = time.time()
        CONSOLE.log(f'{config_id} done (GPU {gpu_id}) in {end_time - start_time:.2f} seconds')

    except Exception as e:
        CONSOLE.log(f'[ERROR] Reconstruction for {config_id} failed: {e}')
        success = False
    finally:
        gpu_manager.release_gpu(gpu_lock_file)
        task_manager.release_task(task_fd, task_lock_file)

    return success

def worker_process(task_manager, gpu_manager, args, worker_id):
    """Worker process that continuously processes tasks from queue"""
    CONSOLE.log(f'Worker {worker_id} started')
    
    while True:
        try:
            # Try to get next available task
            success = run_reconstruction(task_manager, gpu_manager, args)
            if not success:
                # No more tasks available, wait a bit and try again
                time.sleep(5)
                # Check if we should keep trying or exit
                available_count = len(task_manager.get_available_tasks())
                if available_count == 0:
                    CONSOLE.log(f'Worker {worker_id}: No more tasks available, exiting')
                    break
                continue
                
            CONSOLE.log(f'Worker {worker_id}: Task completed successfully')
                
        except Exception as e:
            CONSOLE.log(f'Worker {worker_id} error: {e}')
            time.sleep(1)
    
    CONSOLE.log(f'Worker {worker_id} finished')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', type=str, default=None,
                       help='Directory containing MTGS config files')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--export_dir', type=str, default=None,
                       help='Export directory for assets')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of workers per task')
    args = parser.parse_args()

    # Initialize task manager
    task_manager = TaskManager(args.output_dir, config_dir=args.config_dir)
    
    # Get initial count of available tasks (for display only)
    initial_available_tasks = task_manager.get_available_tasks()
    CONSOLE.log(f"Found {len(initial_available_tasks)} available tasks out of {len(task_manager.configs)} total tasks")
    
    if not initial_available_tasks:
        CONSOLE.log("No available tasks to process. All tasks are either completed or running.")
        exit(0)

    available_devices = torch.cuda.device_count()
    CONSOLE.log(f"Found {available_devices} GPU devices")

    # Clean up old locks
    if os.path.exists(f'{os.path.expanduser("~")}/.cache/gpu_locks'):
        shutil.rmtree(f'{os.path.expanduser("~")}/.cache/gpu_locks')

    # Determine number of workers
    num_workers = available_devices
    CONSOLE.log(f"Starting {num_workers} worker processes")
    
    # Create shared managers
    gpu_manager = GPUManager(lock_dir=f'{os.path.expanduser("~")}/.cache/gpu_locks')
    
    # Start worker processes
    processes = []
    for i in range(num_workers):
        p = multiprocessing.Process(target=worker_process, args=(task_manager, gpu_manager, args, i))
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    CONSOLE.log("All worker processes completed")

    shutil.rmtree(f'{os.path.expanduser("~")}/.cache/gpu_locks')
