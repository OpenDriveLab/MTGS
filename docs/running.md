# Running the codebase

We offer three ways to run the codebase, 

1. Run the codebase using nerfstudio-style command line interface.
2. (Recommended) For development, you can run the codebase with the command in `dev.sh` for customized configs.
3. (Recommended) Batch processing the reconstruction and aggregate the results.

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
mtgs_viewer {/path/to/output/config.yaml} # visualize the reconstruction
mtgs_render {/path/to/output/config.yaml} # render the reconstruction
```

## Batch processing the reconstruction and aggregate the results

We provide a script to batch process the reconstruction and aggregate the results.
The defined tasks including data configurations in `mtgs/tools/batch_exp/mtgs_tasks.py` are the tasks we used in the paper.

```bash
python mtgs/tools/batch_exp/run_base_benchmarking.py \
    --ns-config mtgs/config/MTGS.py \
    --task-name main_mt \
    --output-dir experiments/main_mt
```
