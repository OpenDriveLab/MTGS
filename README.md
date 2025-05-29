<div align="center">

# **MTGS: Multi-Traversal Gaussian Splatting**

[![Arxiv](https://img.shields.io/badge/arXiv-2503.12552-b31b1b.svg?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/2503.12552)
[![dataset](https://img.shields.io/badge/HF-Dataset-yellow?style=for-the-badge&logo=huggingface)](https://huggingface.co/datasets/OpenDriveLab/MTGS)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![Python](https://img.shields.io/badge/python-3.9-blue?style=for-the-badge)](https://www.python.org)

</div>

<div id="top" align="center">
<p align="center">
<img src="https://raw.githubusercontent.com/OpenDriveLab/opendrivelab.github.io/refs/heads/master/MTGS/figure_teaser.png" width="1000px" >
</p>
</div>

## 🔥 Highlights

- **MTGS** leverages **multi-traversal** data for scene reconstruction with better geometry.
- We conduct a robust pipeline to calibrate and reconstruct the [nuPlan](https://www.nuscenes.org/nuplan) dataset with multi-traversal data, which is widely used in the autonomous driving community. See downstream applications in [NAVSIM v2](https://github.com/autonomousvision/navsim).
- We integrate a **web viewer** from nerfstudio to visualize the reconstructed scene and switch nodes between different traversals.
- [Getting started](docs/install.md) with our codebase now! 🚀

<div align="center">
<p align="center">
<img src="https://raw.githubusercontent.com/OpenDriveLab/opendrivelab.github.io/refs/heads/master/MTGS/figure_pipeline.png" width="1000px" >
</p>
</div>

## 🎬 Video Demos

All the videos below are reconstructed and rendered with our method, MTGS, from `road_block-331220_4690660_331190_4690710`.

<div align="center">

**Rendered results on training traversals 1, 2, and 3, from top to bottom.**

<img src="https://raw.githubusercontent.com/OpenDriveLab/opendrivelab.github.io/refs/heads/master/MTGS/road_block-331220_4690660_331190_4690710/traversal_1_trimmed.gif" width="1000px" ><br>
<img src="https://raw.githubusercontent.com/OpenDriveLab/opendrivelab.github.io/refs/heads/master/MTGS/road_block-331220_4690660_331190_4690710/traversal_2_trimmed.gif" width="1000px" ><br>
<img src="https://raw.githubusercontent.com/OpenDriveLab/opendrivelab.github.io/refs/heads/master/MTGS/road_block-331220_4690660_331190_4690710/traversal_3_trimmed.gif" width="1000px" >

***Novel-view*** results on the testing traversal.

<img src="https://raw.githubusercontent.com/OpenDriveLab/opendrivelab.github.io/refs/heads/master/MTGS/road_block-331220_4690660_331190_4690710/traversal_test_trimmed.gif" width="1000px" >

</div>

## 📢 News

- **[2025/05/29]** We release the checkpoints. [Check it out](docs/running.md#optional-download-the-checkpoints)!
- **[2025/05/27]** Official code release.
- **[2025/05/14]** Video demo release.
- **[2025/03/16]** We released our [paper](https://arxiv.org/abs/2503.12552) on arXiv. 

## 📋 TODO List

- [x] Official code release.
- [x] Release the checkpoints.
- [ ] Demo page.

## 🕹️ Getting Started

- 📦 [Installation](docs/install.md)
- 📊 [Prepare Data](docs/prepare_dataset.md)
- 🚀 [Running](docs/running.md)

## ⭐ Citation

If any parts of our paper and code help your research, please consider citing us and giving a star to our repository.

```bibtex
@article{li2025mtgs,
  title={MTGS: Multi-Traversal Gaussian Splatting},
  author={Li, Tianyu and Qiu, Yihang and Wu, Zhenhua and Lindstr{\"o}m, Carl and Su, Peng and Nie{\ss}ner, Matthias and Li, Hongyang},
  journal={arXiv preprint arXiv:2503.12552},
  year={2025}
}
```

## ⚖️ License

All content in this repository is under the [Apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0).
The released data is based on [nuPlan](https://www.nuscenes.org/nuplan) and are under the [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

## ❤️ Related resources

We acknowledge all the open-source contributors for the following projects to make this work possible:

- [nerfstudio](https://github.com/nerfstudio-project/nerfstudio)
- [gsplat](https://github.com/nerfstudio-project/gsplat)
- [drivestudio](https://github.com/ziyc/drivestudio)
- [kiss-icp](https://github.com/PRBonn/kiss-icp)
- [UniDepth](https://github.com/lpiccinelli-eth/UniDepth)
