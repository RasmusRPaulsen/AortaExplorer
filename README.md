# AortaExplorer
**UNDER CONSTRUCTION**

An open source tool for accurate segmentation of the aorta in 3D computed tomography scans. 

![AortaExplorer](https://github.com/RasmusRPaulsen/AortaExplorer/blob/main/figs/aortaexplorer_visualization.png)

**Highlights:**

- Based on the excellent work of the [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) team.
- Works for contrast enhanced CT scans with a variety of scan field-of-views.
- Computes cross sectional lumen areas in aorta segments defined by the [European society of cardiology (ESC)](https://pubmed.ncbi.nlm.nih.gov/25173340/).
- Computes tortuosity.
- Lumen areas have been validated against a large (10.000+) population with manual annotations.
- Tortuosity measures on a large population (10.000+) are consistent with previously reported results.
- Automatically determines [scan field-of-view (FOW)](SCANFOV.md)
- Provides an experimental and non-validated calcification visualization.
- Generates visualizations for easy validation of outputs.
- Designed as a research tool for population studies.

**Note**: This is not a medical device and is not intended for clinical usage. It is meant for research purposes and explorative analysis of the human aorta.

It is based on work done at [DTU Compute](https://www.compute.dtu.dk/) and the [Cardiovascular Research Unit, Rigshospitalet, Copenhagen, Denmark](https://www.rigshospitalet.dk/english/research-and-innovation/units-and-groups/Pages/cardiovascular-ct-research-unit.aspx) 

Please cite this paper if you use AortaExplorer (coming, in revision)
```bibtex
@article{tbd25,
  author  = "Rasmus R. Paulsen and Linnea Hjordt Juul and Michael Huy Cuong Pham and J{\o}rgen Tobias K{\"u}hl and Klaus Fuglsang Kofoed and Kristine Aavild S{\o}rensen and Josefine Vilsb{\o}ll Sundgaard",
  title   = "AortaExplorer: AI-driven analysis of the aorta in CT images",
  journal = "TBD",
  year    = 2025,
  volume  = "TBD",
  number  = "TBD",
  pages   = "1--10"
}
```
Please also cite the [TotalSegmentator paper](https://pubs.rsna.org/doi/10.1148/ryai.230024) since AortaExplorer is heavily dependent on it.


## Installation

AortaExplorer has been tested on Ubuntu Linux and Windows but should work on most systems. The GPU usage is limited to the TotalSegmentator segmentation tasks and you can see the requirements [here](https://github.com/wasserth/TotalSegmentator?tab=readme-ov-file).

The installation requires a few steps:

1. Create and activate an environment. e.g a conda environment
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

Sometimes, it is needed to install PyTorch after the last step with the `-U` to force an install with GPU support.

5. Install the [TotalSegmentator license](https://github.com/wasserth/TotalSegmentator/blob/master/README.md#subtasks). AortaExplorer is dependent on the `heartchambers_highres`subtask and you need to obtain the [license](https://backend.totalsegmentator.com/license-academic/) and install the license key.


## Usage

AortaExplorer can process single NIFTI files, a folder with NIFTI files or a text file with NIFTI file names.

```bash
AortaExplorer -i /data/aorta/aorta_scan.nii.gz -o /data/aorta/AortaExplorerOutput/
```

Will process the NIFTI file `aorta_scan.nii.gz` and create a sub-folder with results in the specified output folder `/data/aorta/AortaExplorerOutput/`.

If the input is a folder, it will process all `.nii.gz` and `.nii` files in that folder. If the input is the name of a text file, it will process all files (specified with full path names) listed in the text file. Note when processing several files, they all need to have unique base names, since the output will named according to this.

**Expected processing time**: AortaExplore is not optimized for speed. On a standard PC (October 2025) the processing time is between 5 and 15 minutes per scan. However, it can process several scans simultanously using multiprocessing and it scales well with the number of cores. The main speed bottleneck is our heavy use of Euclidean Distance Functions (EDT) that might be optimized quite a lot.

## Outputs

AortaExplorer main outputs are:
- CSV file with the aortic measurements. It is called `AortaExplorer_measurements.csv` and will be placed in the specified output folder. One row per input file.
- Visualizations of the results placed in a sub-folder called `all_visualizations`.
- A log file with potential errors and warnings placed in the specified output folder.

If you can not find your expected measurements in the CSV file, you should check the log file to see if there are any reported errors.

**NOTE**: AortaExplorer is not a diagnostic tool and does not check for aortic pathologies like aneurysms. 

## Advanced settings

The AortaExplorer commandline accept a set of arguments:

- `--device`: Choose `cpu` or `gpu` or `gpu:X (e.g., gpu:1 -> cuda:1)` see [TotalSegmentor](https://github.com/wasserth/TotalSegmentator/blob/master/README.md#advanced-settings) for details 
- `--image-x-size` : Visualization image x-side length (default 1920).
- `--image-y-size` : Visualization image y-side length (default 1080).

See [here](aortaexplorer/bin/AortaExplorer.py) for the total set of command line parameters.

## Python API

AortaExplorer can be used as a Python API. For Example:

```Python
from aortaexplorer.python_api import aortaexplorer, get_default_parameters

def test_aortaexplorer():
    params = get_default_parameters()

    input_file = "/data/aorta/aorta_scan.nii.gz"
    output_folder = "/data/aorta/AortaExplorerOutput/"
    device = "gpu"
    verbose = True
    quiet = False

    success = aortaexplorer(input_file, output_folder, params, device=device, verbose=verbose, quiet=quiet)
    assert success, "AortaExplorer failed to run successfully"

if __name__ == "__main__":
    test_aortaexplorer()
```

The `params` is a dictionary, where the individual values can be changed before the call to aortaexplorer. The parameters can be seen [here](aortaexplorer/python_api.py).


## Aortic segments and measurements

AortaExplorer computes aortic segments based on the ESC standard. This is illustrated here:

![AortaExplorer](https://github.com/RasmusRPaulsen/AortaExplorer/blob/main/figs/aortic_sections.png)

From these, the following cross sectional areas are computed:

![AortaExplorer](https://github.com/RasmusRPaulsen/AortaExplorer/blob/main/figs/aortic_measurements.png)


## TotalSegmentator aorta segmentations and lumen segmentations

[TotalSegmentor](https://github.com/wasserth/TotalSegmentator/) has been trained on a CT dataset with 1228 cases. It seems that the aorta groundtruth annotations cover both lumen (the part of the aorta where blood flows), the aortic wall and also potentioal aortic aneurysmic sacs. For population studies, it is valuable to have the diameters of the pure lumen. AortaExplorer computes exactly that. Having both the TotalSegmentator segmentation and the AortaExplorer lumen segmentations potentially opens up for identification of pathological cases.

## Scan FOV cases and handling

CT scanning protocols are typically tailored to clinical examination or research protocols. It is common that the aorta might be cropped due to limited field-of-view (FOV) of the scan protocol. 
We provide [a description of FOV cases](SCANFOV.md) that we have encountered. 

AortaExplorer automatically detects a majority of these FOV cases. Currently, we only do full analysis of these cases:

- **CASE 1: Full aorta** : The entire aorta can be seen in the scan.
- **CASE 2: Abdominal** : The lower part of the aorta and the top of the iliac arteries are visible.
- **CASE 5: Cardiac**: A typical cardiac scan, where the aorta is split into an ascending and a descending part.

If need arises, we might extend the analysis to other scan FOV cases.

## Error handling and pathological cases

AortaExplorer includes a large range of checks for the validity and type of scan. It tries to determine the scan FOV case and only continues on the cases described above. There is also a range of checks on Hounsfield value distributions. If the processing fails, the log file should be inspected for the cause.

## Example data and outputs

**Visualization of a full (type 1) aorta:**

![AortaExplorer](https://github.com/RasmusRPaulsen/AortaExplorer/blob/main/figs/aortaexplorer_visualization2.png)

**Visualization of an abdominal (type 2) aorta:**

![AortaExplorer](https://github.com/RasmusRPaulsen/AortaExplorer/blob/main/figs/aortaexplorer_visualization3.png)

**Visualization of cardiac (type 5) aorta:**

![AortaExplorer](https://github.com/RasmusRPaulsen/AortaExplorer/blob/main/figs/aortaexplorer_visualization4.png)


## Relevant references

-  [*2014 ESC Guidelines on the diagnosis and treatment of aortic diseases*](https://pubmed.ncbi.nlm.nih.gov/25173340/). European Heart Journal (2014) 35, 2873–2926
- MHC Pham et al. [*Normal values of aortic dimensions assessed by multidetector computed tomography in the Copenhagen General Population Study*](https://academic.oup.com/ehjcimaging/article-abstract/20/8/939/5365490). European Heart Journal-Cardiovascular Imaging, 2019.
- Goldstein, Steven A., et al. [*Multimodality imaging of diseases of the thoracic aorta in adults: from the American Society of Echocardiography and the European Association of Cardiovascular Imaging: endorsed by the Society of Cardiovascular Computed Tomography and Society for Cardiovascular Magnetic Resonance.*](https://www.sciencedirect.com/science/article/pii/S0894731714008591) Journal of the American Society of Echocardiography 28.2 (2015): 119-182.
