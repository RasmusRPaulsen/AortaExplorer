import os
from totalsegmentator.python_api import totalsegmentator
from pathlib import Path
import time
import multiprocessing as mp
from aortaexplorer.general_utils import write_message_to_log_file
import SimpleITK as sitk
import numpy as np


def do_totalsegmentator(device, verbose, quiet, write_log_file, output_folder, input_file):
    """
    Use TotalSegmentator to compute segmentations
    input_file: full path to input file
    """
    if not os.path.exists(input_file):
        msg = f"Could not find {input_file} for TotalSegmentator"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
        return False

    # Get pure name of input file without path and extension
    scan_id = os.path.basename(input_file)
    scan_id = os.path.splitext(scan_id)[0]
    if scan_id.endswith(".nii"):
        scan_id = os.path.splitext(scan_id)[0]

    ts_output_folder = f"{output_folder}{scan_id}/segmentations/"
    total_out_name = f"{ts_output_folder}total.nii.gz"
    hc_out_name = f"{ts_output_folder}heartchambers_highres.nii.gz"

    output_base_dir = ts_output_folder
    Path(output_base_dir).mkdir(parents=True, exist_ok=True)

    # Nr of threads for resampling
    nr_thr_resamp = 1
    # Nr of threads for saving segmentations
    nr_thr_saving = 1
    # Run faster lower resolution model
    fast_model = False

    # Calc volume (in mm3) and mean intensity. Results will be in statistics.json
    calc_statistics = False
    # Calc radiomics features. Requires pyradiomics. Results will be in statistics_radiomics.json
    calc_radiomics = False
    # Do initial rough body segmentation and crop image to body region
    body_seg = False
    # Process image in 3 chunks for less memory consumption
    force_split = False
    run_quit = quiet
    verbose = verbose
    multi_label = True

    if not os.path.exists(total_out_name):
        # First the total task to get the main segmentation
        task = "total"
        totalsegmentator(
            input_file,
            total_out_name,
            multi_label,
            nr_thr_resamp,
            nr_thr_saving,
            fast_model,
            device=device,
            nora_tag="None",
            preview=False,
            task=task,
            roi_subset=None,
            statistics=calc_statistics,
            radiomics=calc_radiomics,
            crop_path=None,
            body_seg=body_seg,
            force_split=force_split,
            output_type="nifti",
            quiet=run_quit,
            verbose=verbose,
            test=False,
        )
    elif verbose:
        print(f"{total_out_name} already exists - skipping!")

    # Check if the output segmentation is present and of reasonable size
    if not os.path.exists(total_out_name):
        msg = f"Could not find {total_out_name} after TotalSegmentator run"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
        return False

    if not os.path.exists(hc_out_name):
        # Check if the heart is present in the segmentation
        # Square millimeters - one square centimeter
        volume_threshold = 1000
        heart_label = 51

        try:
            label_img = sitk.ReadImage(total_out_name)
        except RuntimeError as e:
            print(f"Got an exception {str(e)}")
            print(f"Error reading {total_out_name}")
            return False

        spacing = label_img.GetSpacing()
        vox_size = spacing[0] * spacing[1] * spacing[2]
        label_img_np = sitk.GetArrayFromImage(label_img)
        mask_np = label_img_np == heart_label
        sum_pix = np.sum(mask_np)
        if sum_pix * vox_size < volume_threshold:
            msg = f"Heart segmentation volume {sum_pix * vox_size:.1f} mm3 is below threshold {volume_threshold} mm3 for {input_file} - skipping high-res heart segmentation!"
            if not quiet:
                print(msg)
            if write_log_file:
                write_message_to_log_file(base_dir=output_folder, message=msg, level="info")
            return True

        task = "heartchambers_highres"
        totalsegmentator(
            input_file,
            hc_out_name,
            multi_label,
            nr_thr_resamp,
            nr_thr_saving,
            fast_model,
            device=device,
            nora_tag="None",
            preview=False,
            task=task,
            roi_subset=None,
            statistics=calc_statistics,
            radiomics=calc_radiomics,
            crop_path=None,
            body_seg=body_seg,
            force_split=force_split,
            output_type="nifti",
            quiet=run_quit,
            verbose=verbose,
            test=False,
        )

        if not os.path.exists(hc_out_name):
            msg = f"Could not find {hc_out_name} after TotalSegmentator run"
            if not quiet:
                print(msg)
            if write_log_file:
                write_message_to_log_file(base_dir=output_folder, message=msg, level="warning")
    elif verbose:
        print(f"{hc_out_name} already exists - skipping!")

    return True


def computer_process(device, verbose, quiet, write_log_file, output_folder, process_queue, process_id):
    while not process_queue.empty():
        q_size = process_queue.qsize()
        input_file = process_queue.get()
        print(f"Process {process_id} running TotalSegmentator on: {input_file} - {q_size} left")
        local_start_time = time.time()
        do_totalsegmentator(device, verbose, quiet, write_log_file, output_folder, input_file)
        elapsed_time = time.time() - local_start_time
        q_size = process_queue.qsize()
        est_time_left = q_size * elapsed_time
        print(f"Process {process_id} done with {input_file} - took {elapsed_time:.1f} s. Time left {est_time_left:.1f} s")


def compute_totalsegmentator_segmentations(
    in_files,
    output_folder,
    nr_ts=1,
    device="gpu",
    verbose=False,
    quiet=False,
    write_log_file=True,
):
    if verbose:
        print(f"Computing TotalSegmentator segmentations with {nr_ts} processes on device {device} on {len(in_files)} files. Output to {output_folder}")

    num_processes = nr_ts
    process_queue = mp.Queue()
    for idx in in_files:
        input_file = idx.strip()
        process_queue.put(input_file)

    print(f"Starting {num_processes} processes")

    processes = []
    for i in range(num_processes):
        p = mp.Process(
            target=computer_process,
            args=(
                device,
                verbose,
                quiet,
                write_log_file,
                output_folder,
                process_queue,
                i + 1,
            ),
        )
        p.start()
        processes.append(p)

    print("Waiting for processes to finish")
    for p in processes:
        p.join()
