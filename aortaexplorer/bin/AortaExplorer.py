#!/usr/bin/env python
import argparse
import importlib.metadata
from pathlib import Path
import re
from aortaexplorer.python_api import aortaexplorer, get_default_parameters


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
        f"Invalid device type: '{value}'. Must be 'gpu', 'cpu', 'mps', or 'gpu:X' where X is an integer representing the GPU device ID."
    )


def validate_device_type(value):
    try:
        return validate_device_type_api(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid device type: '{value}'. Must be 'gpu', 'cpu', 'mps', or 'gpu:X' where X is an integer representing the GPU device ID."
        )


# TODO: Update AortaExplorer article
def main():
    parser = argparse.ArgumentParser(
        description="Segment and analyse the aorta in CT images.",
        epilog="Written by Rasmus R. Paulsen If you use this tool please cite AortaExplorer article.",
    )

    parser.add_argument(
        "-i",
        metavar="filepath",
        dest="input",
        help="CT NIFTI, NRRD, DICOM image file name, or name of folder with NIFTI, NRRD, DICOM images, or a txt file with filenames or DICOM folders.",
        type=lambda p: Path(p).absolute(),
        required=True,
    )

    parser.add_argument(
        "-o",
        metavar="directory",
        dest="output",
        help="Output directory for aortic segmentations and analysis results.",
        type=lambda p: Path(p).absolute(),
        required=True,
    )

    parser.add_argument(
        "-nt",
        "--nr_ts",
        type=int,
        help="Number of processes for TotalSegmentator",
        default=1,
    )

    parser.add_argument(
        "-np",
        "--nr_proc",
        type=int,
        help="Number of processes for general processing",
        default=6,
    )

    # "mps" is for apple silicon; the latest pytorch nightly version supports 3D Conv but not ConvTranspose3D which is
    # also needed by nnU-Net. So "mps" not working for now.
    # https://github.com/pytorch/pytorch/issues/77818
    parser.add_argument(
        "-d",
        "--device",
        type=validate_device_type,
        default="gpu",
        help="Device type: 'gpu', 'cpu', 'mps', or 'gpu:X' where X is an integer representing the GPU device ID.",
    )

    parser.add_argument(
        "-r",
        "--recurse",
        action="store_true",
        help="Do recursive search for NIFTI, NRRD and DICOM files starting with the input folder",
        default=False,
    )


    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Print no intermediate outputs",
        default=False,
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show more intermediate output",
        default=False,
    )

    # By default we write log file, so default False here
    parser.add_argument(
        "--no-logfile",
        action="store_true",
        dest="logfile",
        help="Do not write log file to output folder",
        default=False,
    )

    parser.add_argument(
        "-oh",
        "--out_hu",
        type=float,
        help="Out-of-scan reconstruction HU value - what did the scanner software use outside the scan area?",
        default=-2048,
    )

    parser.add_argument(
        "-fmi",
        "--forced_min_hu",
        type=float,
        help="Force a minimum HU value for lumen segmentation",
        default=None,
    )
    parser.add_argument(
        "-fma",
        "--forced_max_hu",
        type=float,
        help="Force a maximum HU value for lumen segmentation",
        default=None,
    )
    parser.add_argument(
        "-lhu",
        "--low_hu",
        type=float,
        help="The lowest possible minimum HU value for lumen segmentation",
        default=80,
    )
    parser.add_argument(
        "-mhu",
        "--max_hu",
        type=float,
        help="The lowest possible maximum HU value for lumen segmentation",
        default=400,
    )
    parser.add_argument(
        "-clhu",
        "--calc_low_hu",
        type=float,
        help="The minimum HU value for calcification segmentation",
        default=400,
    )
    parser.add_argument(
        "-cmhu",
        "--calc_max_hu",
        type=float,
        help="The maximum HU value for calcification segmentation",
        default=1500,
    )
    parser.add_argument(
        "-huo",
        "--hu_offset",
        type=float,
        help="Offset to apply to Hounsfield units in the CT scan before processing",
        default=0,
    )
    # Default False means that by default we do compute centerline from TS segmentation
    parser.add_argument(
        "-nts",
        "--no_ts_centerline",
        action="store_true",
        help="Do not compute centerline from TotalSegmentator segmentation",
        default=False,
    )
    parser.add_argument(
        "-cts",
        "--compare_with_ts",
        action="store_true",
        help="Compare aorta diameters with those from TotalSegmentator",
        default=False,
    )
    parser.add_argument(
        "-ix",
        "--image-x-size",
        type=int,
        help="Visualization image x-side length",
        default=1920,
    )
    parser.add_argument(
        "-iy",
        "--image-y-size",
        type=int,
        help="Visualization image y-side length",
        default=1080,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=importlib.metadata.version("AortaExplorer"),
    )

    args = parser.parse_args()

    # Get default aorta parameters and update with any user provided parameters
    aorta_parms = get_default_parameters()
    aorta_parms["num_proc_total_segmentator"] = args.nr_ts
    aorta_parms["num_proc_general"] = args.nr_proc
    aorta_parms["out_of_reconstruction_value"] = args.out_hu
    aorta_parms["forced_aorta_min_hu_value"] = args.forced_min_hu
    aorta_parms["forced_aorta_max_hu_value"] = args.forced_max_hu
    aorta_parms["aorta_min_hu_value"] = args.low_hu
    aorta_parms["aorta_min_max_hu_value"] = args.max_hu
    aorta_parms["aorta_calcification_min_hu_value"] = args.calc_low_hu
    aorta_parms["aorta_calcification_max_hu_value"] = args.calc_max_hu
    aorta_parms["compute_centerline_from_ts_segmentation"] = not args.no_ts_centerline
    aorta_parms["compare_with_totalsegmentator"] = args.compare_with_ts
    aorta_parms["rendering_window_size"] = [args.image_x_size, args.image_y_size]
    aorta_parms["hounsfield_unit_offset"] = args.hu_offset
    aorta_parms["recurse_subfolders"] = args.recurse

    aortaexplorer(
        str(args.input),
        str(args.output) + "/",
        aorta_parms,
        device=args.device,
        verbose=args.verbose,
        quiet=args.quiet,
        write_log_file=not args.logfile,
    )


if __name__ == "__main__":
    main()
