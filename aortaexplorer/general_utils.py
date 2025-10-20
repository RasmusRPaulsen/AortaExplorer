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


def gather_input_files_from_input(in_name: Union[str, Path]) -> Tuple[List[str], str]:
    """
    Gathers a list of input files from the given input, which can be a single file, a text file with entries or a directory

    Args:
        in_name (Union[str, Path]): The input path which can be a file, directory, or a text file containing a list of files.

    Returns:
        List[str]: A list of strings representing the gathered input files.
    """
    in_name = in_name.strip()
    if os.path.isdir(in_name):
        in_files = glob.glob(f"{in_name}/*.nii*")
        if len(in_files) < 1:
            msg = f"No nii or nii.gz files found in {in_name}"
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
