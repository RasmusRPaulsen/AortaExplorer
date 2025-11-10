import os
from pathlib import Path
import time
import multiprocessing as mp
from aortaexplorer.general_utils import write_message_to_log_file, clear_last_error_message, get_pure_scan_file_name
import SimpleITK as sitk


def do_convert(verbose, quiet, write_log_file, output_folder, input_file):
    """
    """
    # Do not inherit any previous error message
    clear_last_error_message()
    pure_id = get_pure_scan_file_name(input_file)
    conv_output_folder = f"{output_folder}{pure_id}/NIFTI/"
    conv_out_name = f"{conv_output_folder}{pure_id}.nii.gz"
    Path(conv_output_folder).mkdir(parents=True, exist_ok=True)

    # Check if input is nrrd file
    if input_file.lower().endswith(".nrrd"):
        try:
            # Read nrrd file with SimpleITK
            sitk_image = sitk.ReadImage(input_file)
            # Write as NIfTI
            sitk.WriteImage(sitk_image, conv_out_name)
            if verbose:
                print(f"Converted NRRD file {input_file} to NIfTI file {conv_out_name}")
        except Exception as e:
            msg = f"Failed to convert NRRD file {input_file} to NIfTI: {e}"
            if not quiet:
                print(msg)
            if write_log_file:
                write_message_to_log_file(
                    base_dir=output_folder, message=msg, level="error"
                )
            return False
    else:
        # Assume input is DICOM folder
        try:
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(input_file)
            reader.SetFileNames(dicom_names)
            sitk_image = reader.Execute()
            sitk.WriteImage(sitk_image, conv_out_name)
            if verbose:
                print(f"Converted DICOM folder {input_file} to NIfTI file {conv_out_name}")
        except Exception as e:
            msg = f"Failed to convert DICOM folder {input_file} to NIfTI: {e}"
            if not quiet:
                print(msg)
            if write_log_file:
                write_message_to_log_file(
                    base_dir=output_folder, message=msg, level="error"
                )
            return False

    return True


def computer_process(verbose, quiet, write_log_file, output_folder, process_queue, process_id):
    while not process_queue.empty():
        q_size = process_queue.qsize()
        input_file = process_queue.get()
        if verbose:
            print(
                f"Process {process_id} running TotalSegmentator on: {input_file} - {q_size} left"
            )
        local_start_time = time.time()
        do_convert(verbose, quiet, write_log_file, output_folder, input_file)
        elapsed_time = time.time() - local_start_time
        pure_id = get_pure_scan_file_name(input_file)
        stats_folder = f"{output_folder}{pure_id}/statistics/"
        Path(stats_folder).mkdir(parents=True, exist_ok=True)
        time_stats_out = f"{stats_folder}conversion_time.txt"
        with open(time_stats_out, "w") as f:
            f.write(f"{elapsed_time}\n")

        q_size = process_queue.qsize()
        est_time_left = q_size * elapsed_time
        if verbose:
            print(f"Process {process_id} done with {input_file} - took {elapsed_time:.1f} s. Time left {est_time_left:.1f} s")
    return True

def convert_input_files(in_files, output_folder, nr_tg=1, verbose=False, quiet=False, write_log_file=True):
    if verbose:
        print(f"Converting{len(in_files)} files. Output to {output_folder}")

    num_processes = nr_tg
    # no need to spawn more processes than files
    num_processes = min(num_processes, len(in_files))

    files_to_process = []
    output_files = []
    for fname in in_files:
        # Get extension and check if it an nrrd file or if it is a DICOM folder
        if fname.lower().endswith(".nrrd"):
            is_nrrd = True
        else:
            is_nrrd = False
        if is_nrrd and not os.path.isfile(fname):
            # not a file
            continue
        if not is_nrrd and not os.path.isdir(fname):
            # not a folder and not nrrd file
            continue

        pure_id = get_pure_scan_file_name(fname)
        conv_output_folder = f"{output_folder}{pure_id}/NIFTI/"
        conv_out_name = f"{conv_output_folder}{pure_id}.nii.gz"
        if not os.path.exists(conv_out_name):
            files_to_process.append(fname)
        output_files.append(conv_out_name)

    if verbose:
        print(f"Found {len(files_to_process)} files/directories to process out of {len(in_files)} files/directories")

    in_files = files_to_process
    if len(in_files) == 0:
        if verbose:
            print("No files to convert!")
        return output_files

    # no need to do multiprocessing for one file
    if len(in_files) == 1:
        input_file = in_files[0].strip()
        if verbose:
            print(f"Converting: {input_file}")
        local_start_time = time.time()
        do_convert(verbose, quiet, write_log_file, output_folder, input_file)
        elapsed_time = time.time() - local_start_time
        if verbose:
            print(f"Done with {input_file} - took {elapsed_time:.1f} s.")
        pure_id = get_pure_scan_file_name(input_file)
        stats_folder = f"{output_folder}{pure_id}/statistics/"
        time_stats_out = f"{stats_folder}totalsegmentator_proc_time.txt"
        Path(stats_folder).mkdir(parents=True, exist_ok=True)
        with open(time_stats_out, "w") as f:
            f.write(f"{elapsed_time}\n")
    else:
        process_queue = mp.Queue()
        for idx in in_files:
            input_file = idx.strip()
            process_queue.put(input_file)

        if verbose:
            print(f"Starting {num_processes} processes")

        processes = []
        for i in range(num_processes):
            p = mp.Process(
                target=computer_process,
                args=(
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

        if verbose:
            print("Waiting for processes to finish")
        for p in processes:
            p.join()
    return output_files
