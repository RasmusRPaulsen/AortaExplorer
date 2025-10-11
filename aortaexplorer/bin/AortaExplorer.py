#!/usr/bin/env python
import argparse
import importlib.metadata
from pathlib import Path
# import re
from aortaexplorer.python_api import aortaexplorer, validate_device_type_api, get_default_parameters


def validate_device_type(value):
    try:
        return validate_device_type_api(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid device type: '{value}'. Must be 'gpu', 'cpu', 'mps', or 'gpu:X' where X is an integer representing the GPU device ID.")


# TODO: Update AortaExplorer article
def main():
    parser = argparse.ArgumentParser(description="Segment and analyse the aorta in CT images.",
                                     epilog="Written by Rasmus R. Paulsen If you use this tool please cite AortaExplorer article.")

    parser.add_argument("-i", metavar="filepath", dest="input",
                        help="CT nifti image or folder of dicom slices or zip file of dicom slices.",
                        type=lambda p: Path(p).absolute(), required=True)

    parser.add_argument("-o", metavar="directory", dest="output",
                        help="Output directory for segmentation masks",
                        type=lambda p: Path(p).absolute(), required=True)


    parser.add_argument("-nt", "--nr_ts", type=int, help="Number of processes for TotalSegmentator", default=1)

    parser.add_argument("-np", "--nr_proc", type=int, help="Number of processes for general processing",
                        default=6)


    # "mps" is for apple silicon; the latest pytorch nightly version supports 3D Conv but not ConvTranspose3D which is
    # also needed by nnU-Net. So "mps" not working for now.
    # https://github.com/pytorch/pytorch/issues/77818
    parser.add_argument("-d",'--device', type=validate_device_type, default="gpu",
                        help="Device type: 'gpu', 'cpu', 'mps', or 'gpu:X' where X is an integer representing the GPU device ID.")

    parser.add_argument("-q", "--quiet", action="store_true", help="Print no intermediate outputs",
                        default=False)

    parser.add_argument("-v", "--verbose", action="store_true", help="Show more intermediate output",
                        default=False)

    parser.add_argument('--version', action='version', version=importlib.metadata.version("AortaExplorer"))

    args = parser.parse_args()

    # Get default aorta parameters and update with any user provided parameters
    aorta_parms = get_default_parameters()
    aortaexplorer(args.input, args.output, aorta_parms,
                  device=args.device, verbose=args.verbose, quiet=args.quiet)


if __name__ == "__main__":
    main()