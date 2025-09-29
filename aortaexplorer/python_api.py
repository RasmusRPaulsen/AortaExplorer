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



def aortaexplorer(input: Union[str, Path], output: Union[str, Path], device="gpu", verbose=False, quiet=False,):
    """
    Run AortaExplorer from within Python.

    For explanation of the arguments see description of command line
    arguments in bin/AortaExplorer.

    Return: success or not
    """
    # Store initial torch settings
    initial_cudnn_benchmark = torch.backends.cudnn.benchmark
    initial_num_threads = torch.get_num_threads()

    validate_device_type_api(device)
    device = select_device(device)
    if verbose: print(f"Using Device: {device}")

    # TODO: Need real article link
    if not quiet:
        print("\nIf you use this tool please cite AortaExplorer article\n")

    # Restore initial torch settings
    torch.backends.cudnn.benchmark = initial_cudnn_benchmark
    torch.set_num_threads(initial_num_threads)

    return True
