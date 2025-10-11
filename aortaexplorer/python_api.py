# import sys
# import os
from pathlib import Path
# import time
# import textwrap
from typing import Union
# import tempfile
# import numpy as np
# import nibabel as nib
# from nibabel.nifti1 import Nifti1Image
import torch
# from totalsegmentator.statistics import get_basic_statistics, get_radiomics_features_for_entire_dir
# from totalsegmentator.libs import download_pretrained_weights
# from totalsegmentator.config import setup_nnunet, setup_totalseg, increase_prediction_counter
# from totalsegmentator.config import send_usage_stats, set_license_number, has_valid_license_offline
# from totalsegmentator.config import get_config_key, set_config_key
# from totalsegmentator.map_to_binary import class_map
# from totalsegmentator.map_to_total import map_to_total
import re
from aortaexplorer.general_utils import write_message_to_log_file, gather_input_files_from_input
from aortaexplorer.totalsegmentator_utils import compute_totalsegmentator_segmentations
from aortaexplorer.aorta_utils import aorta_analysis


def validate_device_type_api(value):
    valid_strings = ["gpu", "cpu", "mps"]
    if value in valid_strings:
        return value

    # Check if the value matches the pattern "gpu:X" where X is an integer
    pattern = r"^gpu:(\d+)$"
    match = re.match(pattern, value)
    if match:
        device_id = int(match.group(1))
        return value

    raise ValueError(
        f"Invalid device type: '{value}'. Must be 'gpu', 'cpu', 'mps', or 'gpu:X' where X is an integer representing the GPU device ID.")


def convert_device_to_cuda(device):
    if device in ["cpu", "mps", "gpu"]:
        return device
    else:  # gpu:X
        return f"cuda:{device.split(':')[1]}"


def convert_device_to_string(device):
    if hasattr(device, 'type'):  # torch.device object
        if device.type == "cuda":
            return "gpu"
        else:
            return device.type


def select_device(device):
    device = convert_device_to_cuda(device)

    # available devices: gpu | cpu | mps | gpu:1, gpu:2, etc.
    if device == "gpu": 
        device = "cuda"
    if device.startswith("cuda"): 
        if device == "cuda": device = "cuda:0"
        if not torch.cuda.is_available():
            print("No GPU detected. Running on CPU. This can be very slow. The '--fast' or the `--roi_subset` option can help to reduce runtime.")
            device = "cpu"
        else:
            device_id = int(device[5:])
            if device_id < torch.cuda.device_count():
                device = torch.device(device)
            else:
                print("Invalid GPU config, running on the CPU")
                device = "cpu"
    return device


def get_default_parameters():
    default_parms = {
    "num_proc_total_segmentator": 1,
    "num_proc_general": 8,
    "forced_aorta_min_hu_value": None,
    "forced_aorta_max_hu_value": None,
    "aorta_min_hu_value": 80,
    "aorta_min_max_hu_value": 400,
    "aorta_calcification_min_hu_value": 400,
    "aorta_calcification_max_hu_value": 1500,
    "compute_centerline_from_ts_segmentation": True,
    "rendering_window_size": [1920, 1080]}
    return default_parms


def aortaexplorer(in_name: Union[str, Path], output: Union[str, Path], aorta_parameters,
                  device="gpu", verbose=False, quiet=False, write_log_file=True) -> bool:
    """
    Run AortaExplorer from within Python.

    For explanation of the arguments see description of command line
    arguments in bin/AortaExplorer.

    Return: success or not
    """
    # TODO: Need real article link
    if not quiet:
        print("\nIf you use this tool please cite AortaExplorer article\n")

    ts_nr_proc = aorta_parameters.get("num_proc_total_segmentator", 1)
    tg_nr_proc = aorta_parameters.get("num_proc_general", 1)

    output = str(output)
    Path(output).mkdir(parents=True, exist_ok=True)

    in_files, msg = gather_input_files_from_input(in_name=in_name)
    if len(in_files) < 1:
        if write_log_file:
            write_message_to_log_file(base_dir=output, message=msg, level="error")
        if not quiet:
            print(msg)
        return False
    if verbose:
        print(f"Found {len(in_files)} input files")

    compute_totalsegmentator_segmentations(in_files=in_files, output_folder=output, nr_ts=ts_nr_proc, device=device,
                                           verbose=verbose, quiet=quiet, write_log_file=write_log_file)

    aorta_analysis(in_files=in_files, output_folder=output, params=aorta_parameters, nr_tg=tg_nr_proc, device=device,
                                           verbose=verbose, quiet=quiet, write_log_file=write_log_file)

    # # Store initial torch settings
    # initial_cudnn_benchmark = torch.backends.cudnn.benchmark
    # initial_num_threads = torch.get_num_threads()
    #
    # validate_device_type_api(device)
    # device = select_device(device)
    # if verbose: print(f"Using Device: {device}")
    #
    #
    # # Restore initial torch settings
    # torch.backends.cudnn.benchmark = initial_cudnn_benchmark
    # torch.set_num_threads(initial_num_threads)

    return True
