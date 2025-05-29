# Running the codebase

We offer three ways to run the codebase, 

1. Run the codebase using nerfstudio-style command line interface.
2. (Recommended) For development, you can run the codebase with the command in `dev.sh` for customized configs.
3. (Recommended) Batch processing the reconstruction and aggregate the results.

**More features supported by MTGS**

- Our customized datamanager supports multi-process async data loading. See `mtgs/dataset/custom_datamanager.py:CustomFullImageDatamanagerConfig.cache_strategy` for more details.
- Rendering with pose interpolation to **60Hz** and multi-traversal support.
- **Web viewer** with multi-traversal support and time slicer.
- Multi-GPU training for multiple scenes. See [below](#batch-processing-the-reconstruction-and-aggregate-the-results) for more details.

## (Optional) Download the checkpoints

We upload the checkpoints used in the MTGS paper to the [Hugging Face](https://huggingface.co/datasets/OpenDriveLab/MTGS/tree/main/MTGS_paper_ckpts). 
You can use the checkpoints to visualize the reconstruction and evaluate the performance.
Please download the checkpoints and extract it in following structure:

```
MTGS/experiments/
└── main_mt
    ├── 3DGS
    └── MTGS
        ├── {road_block_name_1}
        |   ├── nerfstudio_models      # The checkpoint of the model.
        |   ├── config.yml             # The config file of the model.
        |   └── eval_result.json       # The evaluation results of the model.
        ├── {road_block_name_2}
        ├── ...
        ├── paste_table.tsv            # The aggregated results of all the road blocks.
        └── results_summary.csv        # The results for each road block.
```

## Run the codebase using nerfstudio-style command line interface
First, you need to install the MTGS as a nerfstudio plugin.
```bash
cd MTGS
pip install .
```

Then, you can run MTGS with the command:
```bash
ns-train mtgs \
    --experiment-name road_block-331220_4690660_331190_4690710 \
    --vis=viewer \
    nuplan \
    --road-block-config nuplan_scripts/configs/mtgs_exp/road_block-331220_4690660_331190_4690710.yml \
    --train-scene-travels 0 1 7 \
    --eval-scene-travels 0 1 6 7
```
`road_block-331220_4690660_331190_4690710` is the road block in the teaser of MTGS paper.

## (Recommended) For development
You can run the codebase with the command in `dev.sh` for customized configs.
```bash
source dev.sh
```

Then, you can use the following commands to train and visualize the reconstruction.
```bash
mtgs_setup mtgs/config/MTGS.py   # set the config file
mtgs_train {args} # train the model, you can set the args in the config file or in the command line
mtgs_viewer {/path/to/output/config.yml} # visualize the reconstruction with web viewer
mtgs_render {/path/to/output/config.yml} # render the multi-traversal reconstruction under interpolated poses
```

## Batch processing the reconstruction and aggregate the results

We provide a script to batch process the reconstruction and aggregate the results.
The defined tasks including data configurations in `mtgs/tools/batch_exp/mtgs_tasks.py` are the tasks we used in the paper.

```bash
python mtgs/tools/batch_exp/run_base_benchmarking.py \
    --ns-config mtgs/config/MTGS.py \
    --task-name main_mt \
    --output-dir experiments/main_mt/MTGS
```

The script supports multi-GPU training. You can set the `CUDA_VISIBLE_DEVICES` environment variable to specify the GPUs to use.

You can also run the script to export all the evaluation images.
```bash
python mtgs/tools/batch_exp/export_eval_images.py \
    --ns-config mtgs/config/MTGS.py \
    --task-name main_mt \
    --output-dir experiments/main_mt/MTGS
```

The exported images will be saved in `experiments/main_mt/MTGS/rendered_images`.
