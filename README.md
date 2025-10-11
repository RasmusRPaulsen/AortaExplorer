# AortaExplorer

An open source tool for accurate segmentation of the aorta in 3D computed tomography scans.

**UNDER CONSTRUCTION**

**Note**: This is not a medical device and is not intended for clinical usage. It is meant for research purposes and explorative analysis of the human aorta.

## Source code
The source code, instructions and examples will be release upon paper acceptance.

## Installation

The requires a few steps. You can combine these into less steps, but easier to debug this way. 

1. Create and load an environment. e.g a conda environment
```
conda create -n AortaExplorerEnv python=3.11
conda activate AortaExplorerEnv
```
Unfortunately, VMTK does not support Python >3.11 yet.

2. Install [PyTorch](https://pytorch.org/get-started/locally/). Choose the cuda version that matches with what you have available for your GPU

3. Install [VMTK](http://www.vmtk.org/) using conda (use the conda forge version):
```
conda install conda-forge::vmtk
```

4. Install AortaExplorer
```
pip install AortaExplorer
```
