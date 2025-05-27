# Installation Instructions of MTGS

## Prerequisites

You must have a CUDA-enabled NVIDIA video card.
The code is tested on Ubuntu 20.04 and CUDA 11.8.

## Dependencies

The repository is highly dependent on [nerfstudio](https://github.com/nerfstudio-project/nerfstudio) and [gsplat](https://github.com/nerfstudio-project/gsplat).

```bash
# clone the repository
git clone https://github.com/OpenDriveLab/MTGS.git
cd MTGS/

# create a conda environment
conda create --name mtgs -y python=3.9
conda activate mtgs

# tiny-cuda-nn requires >gcc 9.0
conda install -c conda-forge gxx=9.5.0

# if you do not have cuda 11.8 locally, install it with the following command
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

# install torch and torchvision
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# install the requirements
pip install -r requirements.txt
```

## Install the dependencies of nuplan data processing

If you want to process the nuplan data by yourself, you need to first install the [nuplan-devkit](https://github.com/motional/nuplan-devkit) with version 1.2.0 and [colmap](https://github.com/colmap/colmap) with version 3.11.1.

Then, you need to install the customized version of [kiss-icp](https://github.com/PRBonn/kiss-icp) and [UniDepth](https://github.com/lpiccinelli-eth/UniDepth) in this repository.

```bash
# Install the requirements for nuplan data processing
pip install -r requirements_data.txt

# Install kiss-icp.
# this is a customized version with some changes, based on kiss-icp 1.0.0
cd thirdparty/kiss-icp
make editable

# Install UniDepth.
# The requirements are aligned with MTGS.
pip install -e thirdparty/UniDepth
```
