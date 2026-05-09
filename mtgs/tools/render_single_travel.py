#!/usr/bin/env python3
#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import torch


CAMERA_NAMES = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2", "CAM_B0")


def _replace_or_append(text: str, key: str, value: str) -> str:
    pattern = rf"^{re.escape(key)}:.*$"
    if re.search(pattern, text, flags=re.MULTILINE):
        return re.sub(pattern, f"{key}: {value}", text, flags=re.MULTILINE)
    return text.rstrip() + f"\n{key}: {value}\n"


def _set_eval_scene_travels(text: str, travel_id: int) -> str:
    pattern = r"^(\s*)eval_scene_travels:.*$"
    match = re.search(pattern, text, flags=re.MULTILINE)
    replacement = f"eval_scene_travels: !!python/tuple [{travel_id}]"
    if not match:
        return text.rstrip() + f"\n{replacement}\n"
    indent = match.group(1)
    rest = text[match.end():]
    # Remove any remaining list items ("- value") that belonged to the old multiline tuple
    rest = re.sub(r"\n(?:\s*-.*\n)*", "\n", rest, count=1)
    return text[: match.start()] + indent + replacement + rest


def _parse_step_from_ckpt(ckpt_path: Path) -> int:
    match = re.search(r"step-(\d+)\.ckpt$", ckpt_path.name)
    if match is None:
        raise ValueError(f"Cannot parse training step from checkpoint name: {ckpt_path}")
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a single traversal from an MTGS config.")
    parser.add_argument("--config", required=True, help="Path to base config.yml.")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint .ckpt.")
    parser.add_argument("--travel-id", type=int, default=0, help="Traversal id to render.")
    parser.add_argument("--output-dir", default="renders/single_travel", help="Output directory for renders.")
    parser.add_argument("--out-config", default=None, help="Path to write the rendered config.yml.")
    parser.add_argument("--interpolation-steps", type=int, default=6)
    parser.add_argument("--frame-rate", type=int, default=60)
    parser.add_argument("--downscale-factor", type=float, default=2.0)
    parser.add_argument("--pose-source", choices=["eval", "train"], default="eval")
    parser.add_argument("--camera", choices=CAMERA_NAMES, default=None, help="Render only one camera view.")
    args = parser.parse_args()

    base_config = Path(args.config).resolve()
    ckpt_path = Path(args.ckpt).resolve()
    out_config = Path(args.out_config).resolve() if args.out_config else base_config.with_name(
        f"{base_config.stem}_travel{args.travel_id}{base_config.suffix}"
    )

    cfg_text = base_config.read_text()
    cfg_text = _replace_or_append(cfg_text, "load_checkpoint", "null")
    cfg_text = _replace_or_append(cfg_text, "load_dir", "null")
    cfg_text = _replace_or_append(cfg_text, "load_step", str(_parse_step_from_ckpt(ckpt_path)))
    cfg_text = _set_eval_scene_travels(cfg_text, args.travel_id)
    out_config.write_text(cfg_text)

    env = os.environ.copy()
    env.setdefault("NERFSTUDIO_METHOD_CONFIGS", "mtgs=mtgs.config.MTGS:method")
    env.setdefault("NERFSTUDIO_DATAPARSER_CONFIGS", "nuplan=mtgs.config.nuplan_dataparser:nuplan_dataparser")

    # Inject gsplat-mps compatibility shim via sitecustomize if on MPS device
    import tempfile
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and not torch.cuda.is_available():
        site_dir = tempfile.mkdtemp(prefix="gsplat_mps_compat_")
        sitecustomize = os.path.join(site_dir, "sitecustomize.py")
        with open(sitecustomize, "w") as f:
            f.write("import mtgs.tools.gsplat_mps_compat\n")
        env["PYTHONPATH"] = site_dir + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable,
        "mtgs/tools/render.py",
        "interpolate",
        "--load-config",
        str(out_config),
        "--output-path",
        str(Path(args.output_dir).resolve()),
        "--pose-source",
        args.pose_source,
        "--interpolation-steps",
        str(args.interpolation_steps),
        "--frame-rate",
        str(args.frame_rate),
        "--downscale-factor",
        str(args.downscale_factor),
        "--output-format",
        "video",
    ]
    if args.camera is not None:
        cmd.extend(["--multi-view-camera", args.camera])
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
