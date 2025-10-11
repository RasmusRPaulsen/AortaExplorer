#!/usr/bin/env python
import argparse
import importlib.metadata
from pathlib import Path
from aortaexplorer.python_api import (
    aortaexplorer,
    validate_device_type_api,
    get_default_parameters,
)


def validate_device_type(value):
    try:
        return validate_device_type_api(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid device type: '{value}'. Must be 'gpu', 'cpu', 'mps', or 'gpu:X' where X is an integer representing the GPU device ID.")


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
        help="CT nifti image file name, or name of folder with nifti images, or a txt file with filenames.",
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

    parser.add_argument(
        "-l",
        "--logfile",
        action="store_true",
        help="Write log file to output folder",
        default=True,
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
        default=80,
    )
    parser.add_argument(
        "-clhu",
        "--calc_low_hu",
        type=float,
        help="The minimum HU value for calcification segmentation",
        default=80,
    )
    parser.add_argument(
        "-cmhu",
        "--calc_max_hu",
        type=float,
        help="The maximum HU value for calcification segmentation",
        default=80,
    )
    parser.add_argument(
        "-ts",
        "--ts_centerline",
        action="store_true",
        help="Compute centerline from TotalSegmentator segmentation",
        default=True,
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
    aorta_parms["num_proc_total_segmentator"] = args.nt
    aorta_parms["num_proc_general"] = args.np
    aorta_parms["forced_aorta_min_hu_value"] = args.forced_min_hu
    aorta_parms["forced_aorta_max_hu_value"] = args.forced_max_hu
    aorta_parms["aorta_min_hu_value"] = args.low_hu
    aorta_parms["aorta_min_max_hu_value"] = args.max_hu
    aorta_parms["aorta_calcification_min_hu_value"] = args.calc_low_hu
    aorta_parms["aorta_calcification_max_hu_value"] = args.calc_max_hu
    aorta_parms["compute_centerline_from_ts_segmentation"] = args.ts_centerline
    aorta_parms["rendering_window_size"] = [args.image_x_size, args.image_y_size]

    aortaexplorer(
        args.input,
        args.output,
        aorta_parms,
        device=args.device,
        verbose=args.verbose,
        quiet=args.quiet,
        write_log_file=args.logfile,
    )


if __name__ == "__main__":
    main()
