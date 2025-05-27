import os
import argparse
import importlib
import copy
import json
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ns-config', type=str, default='mtgs/config/MTGS.py')
    parser.add_argument('--config', type=str, required=True, help='Path to the road block config')
    parser.add_argument('--train-traversal', type=str, required=True,
                       help='Comma-separated list of traversal indices. If None, all traversals will be used for evaluation')
    parser.add_argument('--eval-traversal', type=str, required=True,
                       help='Comma-separated list of traversal indices. If None, all traversals will be used for evaluation')
    parser.add_argument('--output-dir', type=str, default='experiments/main_mt')
    parser.add_argument('--eval-only', action='store_true')
    args = parser.parse_args()

    module_name = args.ns_config.replace('.py', '').replace('/', '.')
    os.environ["NERFSTUDIO_DATAPARSER_CONFIGS"] = "nuplan=mtgs.config.nuplan_dataparser:nuplan_dataparser"
    os.environ["NERFSTUDIO_METHOD_CONFIGS"] = 'mtgs=' + module_name + ':method'
    # Convert string arguments back to tuples
    train_traversal = tuple(int(x) for x in args.train_traversal.split(',')) if args.train_traversal != "None" else ()
    eval_traversal = tuple(int(x) for x in args.eval_traversal.split(',')) if args.eval_traversal != "None" else ()

    from nerfstudio.scripts.train import main
    ns_config = importlib.import_module(module_name).config

    output_dir = os.path.join(args.output_dir, args.config.split('/')[-1].split('.')[0])
    os.makedirs(output_dir, exist_ok=True)
    if not args.eval_only:
        ns_config = copy.deepcopy(ns_config)
        ns_config.pipeline.datamanager.dataparser.road_block_config = args.config
        ns_config.pipeline.datamanager.eval_cache_strategy = "on_demand"
        ns_config.experiment_name = args.config.split('/')[-1].split('.')[0] + '_' + '_'.join(map(str, train_traversal))
        ns_config.pipeline.datamanager.dataparser.train_scene_travels = train_traversal
        ns_config.pipeline.datamanager.dataparser.eval_scene_travels = eval_traversal
        ns_config.pipeline.datamanager.dataparser.only_moving = False
        ns_config.pipeline.model.model_config['background'].verbose = False
        ns_config.output_dir = output_dir
        ns_config.set_base_dir(output_dir)
        main(ns_config)

    # evaluation
    from nerfstudio.utils.eval_utils import eval_setup
    load_config = Path(output_dir) / 'config.yml'
    output_path = Path(output_dir) / 'eval_result.json'

    def update_config(config):
        config.pipeline.datamanager.dataparser.eval_2hz = True
        config.pipeline.model.predict_normals = False
        config.pipeline.model.color_corrected_metrics = True
        config.pipeline.model.lpips_metric = True
        config.pipeline.model.dinov2_metric = True
        return config

    config, pipeline, checkpoint_path, _ = eval_setup(load_config, update_config_callback=update_config)
    metrics_dict = pipeline.get_average_eval_image_metrics(get_std=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Get the output and define the names to save to
    benchmark_info = {
        "experiment_name": config.experiment_name,
        "method_name": config.method_name,
        "checkpoint": str(checkpoint_path),
        "results": metrics_dict,
    }
    # Save output to output file
    output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
