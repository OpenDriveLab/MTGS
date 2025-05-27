#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import os
import shutil
import argparse
import json
import subprocess
import multiprocessing
import time
import pandas as pd
import torch

from nuplan_scripts.utils.constants import CONSOLE
from mtgs.tools.batch_exp.mtgs_tasks import tasks_registry

GLOBAL_CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES', None)


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

def run_reconstruction(task, args):
    gpu_manager = GPUManager(lock_dir=args.output_dir + '/gpu_locks')
    gpu_id, lock_file = gpu_manager.acquire_gpu()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    output_dir = os.path.join(args.output_dir, task['config'].split('/')[-1].split('.')[0])
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f'log_{time.strftime("%Y%m%d_%H%M%S")}.log')
    eval_file = os.path.join(output_dir, 'eval_result.json')

    success = True
    try:
        if args.resume and os.path.exists(eval_file):
            CONSOLE.log(f'[INFO] Reconstruction for {os.path.basename(output_dir)} already completed. Skipping...')
        else:
            if os.path.exists(log_file):
                os.remove(log_file)

            train_traversal_str = ','.join(map(str, task["train_traversal"])) if task["train_traversal"] else "None"
            eval_traversal_str = ','.join(map(str, task["eval_traversal"])) if task["eval_traversal"] else "None"

            CONSOLE.log(f'Reconstruction for {os.path.basename(output_dir)} started (GPU {gpu_id})')
            start_time = time.time()
            command = f'python batch_exp_scripts/run_single_road_block.py --ns-config {args.ns_config} --config {task["config"]} --train-traversal {train_traversal_str} --eval-traversal {eval_traversal_str} --output-dir {args.output_dir}'
            if args.eval_only:
                command += ' --eval-only'
            with open(log_file, 'w') as log:
                process = subprocess.Popen(command, stdout=log, stderr=log, shell=True)
                process.communicate()
                if process.returncode != 0:
                    raise Exception(f'Reconstruction for {os.path.basename(output_dir)} failed')
            end_time = time.time()
            CONSOLE.log(f'{os.path.basename(output_dir)} done (GPU {gpu_id}) in {end_time - start_time:.2f} seconds')

    except Exception as e:
        CONSOLE.log(f'[ERROR] Reconstruction for {os.path.basename(output_dir)} failed: {e}')
        success = False
    finally:
        gpu_manager.release_gpu(lock_file)

    return success

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ns-config', 
        type=str, 
        default='mtgs/config/MTGS.py')
    parser.add_argument('--output-dir', type=str, default='experiments/main_mt')
    parser.add_argument('--task-name', type=str, default='main_mt')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--eval-only', action='store_true')
    args = parser.parse_args()

    tasks = tasks_registry[args.task_name]

    available_devices = torch.cuda.device_count()
    CONSOLE.log(f"Found {available_devices} GPU devices")
    if os.path.exists(args.output_dir + '/gpu_locks'):
        shutil.rmtree(args.output_dir + '/gpu_locks')

    with multiprocessing.Pool(processes=available_devices) as pool:
        pool.starmap(run_reconstruction, [(task, args) for task in tasks])

    # Summarize results
    all_metrics = ['psnr', 'ssim', 'cc_psnr', 'lpips', 'dinov2_sim', 'depth_RMSE', 'depth_absRel', 'depth_delta1']
    available_metrics = None
    all_results = []
    for task in tasks:
        output_dir = os.path.join(args.output_dir, task['config'].split('/')[-1].split('.')[0])
        eval_json_file = os.path.join(output_dir, 'eval_result.json')
        assert os.path.exists(eval_json_file), f"Evaluation results file not found for {task['config']}"

        with open(eval_json_file, 'r') as f:
            eval_results = json.load(f)['results']

        if available_metrics is None:
            available_metrics = list(eval_results.keys())
            available_metrics = [metric for metric in all_metrics if metric in available_metrics]

        # Detect available traversal indices
        available_trv_idx = []
        for key in eval_results.keys():
            if key.startswith('trv') and '_psnr' in key:
                trv_idx = int(key.split('_')[0][3:])  # Extract number from 'trvX_psnr'
                available_trv_idx.append(trv_idx)
        available_trv_idx.sort()
        
        train_traversals = set(task['train_traversal'])
        
        # Initialize metric accumulators
        seen_metrics = {metric: [] for metric in available_metrics}
        unseen_metrics = {metric: [] for metric in available_metrics}

        # Collect metrics for each traversal
        for trv_idx in available_trv_idx:
            prefix = f'trv{trv_idx}_'
            metrics = seen_metrics if trv_idx in train_traversals else unseen_metrics

            if f'{prefix}psnr' in eval_results:
                for metric in metrics.keys():
                    metrics[metric].append(eval_results[f'{prefix}{metric}'])

        num_traversals = len(available_trv_idx)
        # Calculate averages
        task_name = task['config'].split('/')[-1].split('.')[0]
        task_result = {
            'task': task_name
        }

        for metric in available_metrics:
            task_result[f'seen_{metric}'] = sum(seen_metrics[metric]) / len(seen_metrics[metric]) if seen_metrics[metric] else 0
            task_result[f'unseen_{metric}'] = sum(unseen_metrics[metric]) / len(unseen_metrics[metric]) if unseen_metrics[metric] else 0
            task_result[f'overall_{metric}'] = sum(eval_results[f'trv{i}_{metric}'] for i in available_trv_idx) / num_traversals

        all_results.append(task_result)

    # Create DataFrame and calculate averages
    df = pd.DataFrame(all_results)
    avg_row = df.mean(numeric_only=True)
    df.loc['Average'] = ['Average', *avg_row]

    # Format the table
    pd.set_option('display.float_format', lambda x: '%.3f' % x if isinstance(x, float) else str(x))
    print("\nResults Summary:")
    print(df.to_string(index=False))
    df.to_csv(f'{args.output_dir}/results_summary.csv', index=False, float_format='%.3f')

    # Save summary metrics to TSV
    summary_data = {
        'Metric': ['Overall', 'Seen', 'Unseen'],
    }
    for metric in available_metrics:
        summary_data[metric] = [avg_row[f'overall_{metric}'], avg_row[f'seen_{metric}'], avg_row[f'unseen_{metric}']]

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f'{args.output_dir}/paste_table.tsv', sep='\t', index=False, float_format='%.3f')
