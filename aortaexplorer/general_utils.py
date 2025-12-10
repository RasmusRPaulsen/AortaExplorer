import os
import glob
from typing import List, Union, Tuple
from pathlib import Path
from datetime import datetime
import json

last_error_message = ""

def clear_last_error_message():
    global last_error_message
    last_error_message = ""


def write_message_to_log_file(base_dir, message, level="warning"):
    global last_error_message
    if level == "error":
        last_error_message = message
    if os.path.isdir(base_dir):
        pdir = base_dir
    else:
        pdir = os.path.dirname(base_dir)

    log_file = f"{pdir}/AortaExporer_log.txt"
    if not os.path.isdir(os.path.dirname(log_file)):
        Path(os.path.dirname(log_file)).mkdir(parents=True, exist_ok=True)

    now_date = datetime.strftime(datetime.now(), "%d-%m-%Y-%H-%M-%S")
    with open(log_file, "a") as file:
        file.write(f"{now_date}: {level} : {message}\n")


def get_last_error_message():
    global last_error_message
    return last_error_message


def get_pure_scan_file_name(scan_file_name: str) -> str:
    # check if scan name is actually a directory
    if os.path.isdir(scan_file_name):
        scan_id = os.path.basename(os.path.normpath(scan_file_name))
        return scan_id

    # Get pure name of input file without path and extension
    scan_id = os.path.basename(scan_file_name)
    scan_id = os.path.splitext(scan_id)[0]
    if scan_id.endswith(".nii"):
        scan_id = os.path.splitext(scan_id)[0]
    return scan_id


def gather_input_files_from_input(in_name: Union[str, Path], recurse_subfolders=False) -> Tuple[List[str], str]:
    """
    Gathers a list of input files from the given input, which can be a single file, a text file with entries or a directory

    Args:
        in_name (Union[str, Path]): The input path which can be a file, directory, or a text file containing a list of files.

    Returns:
        List[str]: A list of strings representing the gathered input files.
    """
    in_name = in_name.strip()
    if recurse_subfolders and os.path.isdir(in_name):
        # recursively search for nii, nii.gz, nrrd files
        in_files = glob.glob(f"{in_name}/**/*.nrrd", recursive=True)
        in_files += glob.glob(f"{in_name}/**/*.nii", recursive=True)
        in_files += glob.glob(f"{in_name}/**/*.nii.gz", recursive=True)

        # now recursively search for non-empty subdirectories. These can contain DICOM files
        for root, dirs, files in os.walk(in_name):
            for d in dirs:
                full_d = os.path.join(root, d)
                if os.path.isdir(full_d):
                    # check if directory is non-empty
                    if len(os.listdir(full_d)) > 10:
                        # Check if the directory does not containg nii, nii.gz, nrrd files to avoid duplicates
                        non_dicom_files = glob.glob(f"{full_d}/*.nrrd")
                        non_dicom_files += glob.glob(f"{full_d}/*.nii")
                        non_dicom_files += glob.glob(f"{full_d}/*.nii.gz")
                        if len(non_dicom_files) < 1:
                            in_files.append(full_d)
        if len(in_files) < 1:
            msg = f"No nii, nii.gz, nrrd files or DICOM folders found in {in_name} or its subdirectories"
            print(msg)
            return [], msg
    elif os.path.isdir(in_name):
        # glob for both .nii and .nii.gz files and .nrrd files
        in_files = glob.glob(f"{in_name}/*.nrrd")
        in_files += glob.glob(f"{in_name}/*.nii")
        in_files += glob.glob(f"{in_name}/*.nii.gz")
        if len(in_files) < 1:
            # try finding non-empty subdirectories. These can contain DICOM files
            for d in os.listdir(in_name):
                full_d = os.path.join(in_name, d)
                if os.path.isdir(full_d):
                    # check if directory is non-empty
                    if len(os.listdir(full_d)) > 0:
                        in_files.append(full_d)
        if len(in_files) < 1:
            # Check if there files in the current directory. These can be DICOM files
            if len(os.listdir(in_name)) > 0:
                in_files.append(in_name)
        if len(in_files) < 1:
            msg = f"No nii, nii.gz, nrrd files or DICOM folders found in {in_name}"
            print(msg)
            return [], msg
    elif os.path.isfile(in_name):
        if in_name.endswith(".txt"):
            try:
                with open(in_name, "r") as f:
                    in_files = f.readlines()
                in_files = [x.strip() for x in in_files]
                # Remove empty lines
                in_files = [x for x in in_files if x]
                if len(in_files) < 1:
                    msg = f"No files found in {in_name}"
                    print(msg)
                    return [], msg
            except Exception as e:
                msg = f"Could not read {in_name}: {str(e)}"
                print("msg")
                return [], msg
        else:
            in_files = [in_name]
    else:
        msg = f"Input {in_name} is not a file or directory"
        print(msg)
        return [], msg

    return in_files, ""


def read_json_file(json_name):
    if os.path.exists(json_name):
        try:
            with open(json_name, "r") as openfile:
                json_stuff = json.load(openfile)
                return json_stuff
        except IOError as e:
            print(f"I/O error({e.errno}): {e.strerror}: {json_name}")
            return None
    return None


def display_time(seconds):
    intervals = (
        ('w', 604800),  # 60 * 60 * 24 * 7
        ('d', 86400),  # 60 * 60 * 24
        ('h', 3600),  # 60 * 60
        ('m', 60),
        ('s', 1),
    )
    result = []
    if seconds < 60:
        return f"{seconds}s"
    for name, count in intervals:
        value = seconds // count
        if value > 0:
            seconds -= value * count
            # if value == 1:
            #     name = name.rstrip('s')
            result.append(f"{value}{name}")
    return ' '.join(result)
