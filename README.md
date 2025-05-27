<div align="center">

# **MTGS: Multi-Traversal Gaussian Splatting**

</div>

> 📜 [[arxiv](https://arxiv.org/abs/2503.12552)] 🤗 [[dataset](https://huggingface.co/datasets/OpenDriveLab/MTGS/tree/main/MTGS_paper_data)]

<div id="top" align="center">
<p align="center">
<img src="assets/figure_teaser.png" width="1000px" >
</p>
</div>

## 🔥 Highlights

**MTGS** manages to leverage multi-traversal data for scene reconstruction with better geometry. We utilize the [nuPlan](https://www.nuscenes.org/nuplan) dataset with extensive multi-traversal data. 

<div id="top" align="center">
<p align="center">
<img src="assets/figure_pipeline.png" width="1000px" >
</p>
</div>

## 🕹️ Video Demos

Roadblock 331220_4690660_331190_4690710 in `nuplan`.

Rendered results on training traversals.

https://github.com/user-attachments/assets/8548e307-1669-4968-aeba-1cb63851478b

https://github.com/user-attachments/assets/b7cd65b5-3470-4a0d-9f13-c3f7a1dd05c4

https://github.com/user-attachments/assets/235ea9b3-6419-424d-b04f-97f25faebf6c

Rendered results on the testing traversal.

https://github.com/user-attachments/assets/073458fe-0806-4b3c-9645-129cc066b6d1

## 📢 News

- **[2025/05/27]** Official code release.

## 📋 TODO List

- [x] Official code release.
- [ ] More visualizations in different roadblocks.

## 🔥 Getting Started

- 📦 [Installation](docs/installation.md)
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

All content in this repository are under the [Apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0).
The released data is based on [nuPlan](https://www.nuscenes.org/nuplan) and are under the [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

## Related resources

We acknowledge all the open-source contributors for the following projects to make this work possible:

- [nerfstudio](https://github.com/nerfstudio-project/nerfstudio)
- [gsplat](https://github.com/nerfstudio-project/gsplat)
- [drivestudio](https://github.com/ziyc/drivestudio)
- [kiss-icp](https://github.com/PRBonn/kiss-icp)
- [UniDepth](https://github.com/lpiccinelli-eth/UniDepth)
