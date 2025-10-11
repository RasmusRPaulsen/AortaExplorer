from pathlib import Path
from typing import Union
from aortaexplorer.general_utils import write_message_to_log_file, gather_input_files_from_input
from aortaexplorer.totalsegmentator_utils import compute_totalsegmentator_segmentations
from aortaexplorer.aorta_utils import aorta_analysis


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

    aorta_analysis(in_files=in_files, output_folder=output, params=aorta_parameters, nr_tg=tg_nr_proc, verbose=verbose,
                   quiet=quiet, write_log_file=write_log_file)
    return True
