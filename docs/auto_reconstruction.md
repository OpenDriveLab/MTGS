# Auto Reconstruction Pipeline

MTGS provides a robust, scalable auto reconstruction pipeline for the nuPlan dataset. Given a list of tokens (start frames of scenarios), the pipeline automatically generates optimal reconstruction configurations and runs the full reconstruction workflow.

## Overview

The auto reconstruction pipeline consists of three main stages:

1. **Config Generation**: Generate `FrameCentralConfig` files from a given token list with automatic temporal deduplication
2. **Multi-GPU Reconstruction**: Distributed reconstruction across multiple GPUs/nodes with automatic task management
3. **Export**: Export reconstructed scenes to portable checkpoints for downstream simulation tasks

## Pipeline Architecture

```
Token List (NAVSIM filter)
        │
        ▼
┌───────────────────────────────┐
│  generate_configs_from_navsim │  ← Temporal deduplication
│         _filter.py            │  ← Road block generation
└───────────────────────────────┘
        │
        ▼
    Config Files (.yaml)
        │
        ▼
┌───────────────────────────────┐
│   multi_gpu_auto_recon.py     │  ← Task Manager (multi-node coordination)
│                               │  ← GPU Manager (resource allocation)
└───────────────────────────────┘
        │
        ▼ (per config, runs recon_single_config.sh)
┌───────────────────────────────┐
│  1. stage_all.sh              │  ← Data preprocessing
│  2. background_reconstruction │  ← Gaussian splatting training
│  3. render_reconstruction     │  ← Render & evaluate
│  4. export_reconstruction     │  ← Export portable checkpoint
│  5. clean_temp_files          │  ← Cleanup intermediate files
└───────────────────────────────┘
        │
        ▼
   Portable Checkpoints
   (Ready for simulation)
```

## Quick Start

This tutorial assumes you are familiar with the [NAVSIM benchmark](https://github.com/autonomousvision/navsim), a popular benchmark for end-to-end autonomous driving on nuPlan/OpenScene.

### Step 1: Generate Configs from NAVSIM Token List

Generate `FrameCentralConfig` files from NAVSIM filter configs. The script automatically performs temporal deduplication, merging nearby tokens into single reconstruction configurations.

```bash
python -m nuplan_scripts.auto_reconstruction.generate_configs_from_navsim_filter \
    --navsim_config path/to/navtrain.yaml \
    --data_root data/auto_reconstruction/navtrain \
    --output_dir data/auto_reconstruction/navtrain/configs \
    --num_workers 32
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `--navsim_config` | Path to NAVSIM filter config file(s). Supports multiple paths separated by space. |
| `--data_root` | Root directory for reconstruction data |
| `--output_dir` | Output directory for generated config files |
| `--split` | Data split: `trainval`, `test`, `private_test`, or `all` (default: `trainval`) |
| `--num_workers` | Number of parallel workers (default: 0) |

### Step 2: Run Multi-GPU Auto Reconstruction

Launch the distributed reconstruction pipeline:

```bash
python -m nuplan_scripts.auto_reconstruction.multi_gpu_auto_recon \
    --config-dir data/auto_reconstruction/navtrain/configs \
    --output-dir data/auto_reconstruction/navtrain \
    --export-dir data/export_dir/navtrain \
    --workers 12
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `--config-dir` | Directory containing generated config files |
| `--output-dir` | Output directory for reconstruction results |
| `--export-dir` | Export directory for portable checkpoints |
| `--workers` | Number of CPU workers per task (default: 12) |

**GPU Control:**
By default, the pipeline uses all available GPUs. Use `CUDA_VISIBLE_DEVICES` to specify which GPUs to use:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m nuplan_scripts.auto_reconstruction.multi_gpu_auto_recon ...
```

## Key Features

### Task Manager (Multi-Node Support)

The task manager provides automatic coordination across multiple machines:

- **Atomic task acquisition**: Uses file locking to prevent duplicate processing
- **Shared filesystem support**: Automatically detects completed/running tasks across nodes
- **Fault tolerance**: Gracefully handles task failures
- **Auto-resume**: Re-run the same command to retry failed tasks

### GPU Manager

Efficient GPU resource allocation:

- **Automatic GPU detection**: Discovers all available GPUs
- **Lock-based allocation**: Ensures exclusive GPU access per task
- **Dynamic scheduling**: Workers continuously process tasks until completion

### Failure Recovery

If some tasks fail during reconstruction:

1. Failed tasks are moved to directories with `_failed` suffix
2. Simply re-run the same command to retry failed tasks
3. The task manager will skip completed tasks and only process remaining ones

Logs are saved in the output directory for debugging.

## Output Structure

After reconstruction, the export directory contains:

```
export_dir/
├── assets/
│   ├── {config_id}/
│   │   ├── background/
│   │   │   └── {config_id}.ckpt     # Portable Gaussian splatting checkpoint
│   │   └── video_scene_dict.pkl     # Scene metadata for simulation
│   └── ...
├── configs/
│   ├── {config_id}.yaml             # FrameCentralConfig
└── ...
```

## Advanced Usage

### Running Single Config Manually

To run reconstruction for a single config:

```bash
bash nuplan_scripts/auto_reconstruction/recon_single_config.sh \
    --config path/to/config.yaml \
    --export-dir path/to/export \
    --output-dir path/to/output \
    --workers 12
```

### Render Reconstruction Results

To render and evaluate an existing reconstruction:

```bash
python -m nuplan_scripts.auto_reconstruction.render_reconstruction \
    --config path/to/config.yaml
```

### Export Reconstruction

To export a reconstruction to portable checkpoint:

```bash
python -m nuplan_scripts.auto_reconstruction.export_reconstruction \
    --config path/to/config.yaml \
    --output_dir path/to/export
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Tasks stuck in "running" state | Check for stale lock files in `.task_locks` directory |
| Tasks fail silently | Check logs in `{output_dir}/{config_id}_failed/log_*.log` |
