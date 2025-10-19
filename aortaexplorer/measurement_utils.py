import os.path
from aortaexplorer.general_utils import write_message_to_log_file, read_json_file
import shutil
from pathlib import Path


def flatten_json(y):
    out = {}

    def flatten(x, name=""):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + "_")
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + "_")
                i += 1
        else:
            out[str(name[:-1])] = str(x)

    flatten(y)
    return out


def gather_all_stat_columen_names(
    in_files, output_folder, verbose=False, quiet=False, write_log_file=True
):
    all_column_names_set = set()
    all_column_names = []
    for in_file in in_files:
        # Get pure name of input file without path and extension
        scan_id = os.path.basename(in_file)
        scan_id = os.path.splitext(scan_id)[0]
        if scan_id.endswith(".nii"):
            scan_id = os.path.splitext(scan_id)[0]
        stats_folder = f"{output_folder}{scan_id}/statistics/"
        stats_file = f"{stats_folder}aorta_statistics.json"

        if not os.path.isfile(stats_file):
            if verbose:
                print(f"Measurement file {stats_file} not found, skipping")
            continue
        json_stuff = read_json_file(stats_file)
        if json_stuff is None:
            if not quiet:
                print(f"Error reading {stats_file}")
            if write_log_file:
                write_message_to_log_file(
                    base_dir=output_folder,
                    message=f"Error reading {stats_file}",
                    level="error",
                )
            continue
        flat_json = flatten_json(json_stuff)
        columns = flat_json.keys()

        for c in columns:
            if c not in all_column_names_set:
                all_column_names_set.add(c)
                all_column_names.append(c)

    return list(all_column_names)


def gather_measurements_from_file(
    in_file,
    all_column_names,
    output_folder,
    verbose=False,
    quiet=False,
    write_log_file=True,
):
    # Get pure name of input file without path and extension
    scan_id = os.path.basename(in_file)
    scan_id = os.path.splitext(scan_id)[0]
    if scan_id.endswith(".nii"):
        scan_id = os.path.splitext(scan_id)[0]
    stats_folder = f"{output_folder}{scan_id}/statistics/"
    stats_file = f"{stats_folder}aorta_statistics.json"

    if not os.path.isfile(stats_file):
        if verbose:
            print(f"Measurement file {stats_file} not found, skipping")
        return None
    json_stuff = read_json_file(stats_file)
    if json_stuff is None:
        if not quiet:
            print(f"Error reading {stats_file}")
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder,
                message=f"Error reading {stats_file}",
                level="error",
            )
        return None
    flat_json = flatten_json(json_stuff)
    columns = flat_json.keys()

    values = {}
    for c in all_column_names:
        values[c] = ""
    for c in columns:
        values[c] = flat_json[c]

    return values


def get_all_measurement(
    in_files,
    all_column_names,
    output_folder,
    verbose=False,
    quiet=False,
    write_log_file=True,
):
    measures_out = f"{output_folder}AortaExplorer_measurements.csv"

    f = open(measures_out, "a")
    for in_file in in_files:
        values = gather_measurements_from_file(
            in_file, all_column_names, output_folder, verbose, quiet, write_log_file
        )
        if values is not None:
            for c in all_column_names:
                f.write(f"{values[c]},")
            f.write("\n")
    f.close()


def copy_all_visualization(
    in_files, output_folder, verbose=False, quiet=False, write_log_file=True
):
    out_vis_folder = f"{output_folder}all_visualizations/"
    Path(out_vis_folder).mkdir(parents=True, exist_ok=True)

    for in_file in in_files:
        # Get pure name of input file without path and extension
        scan_id = os.path.basename(in_file)
        scan_id = os.path.splitext(scan_id)[0]
        if scan_id.endswith(".nii"):
            scan_id = os.path.splitext(scan_id)[0]
        vis_in = f"{output_folder}{scan_id}/visualization/aorta_visualization.png"
        vis_out = f"{out_vis_folder}{scan_id}_aorta_visualization.png"

        if not os.path.isfile(vis_in):
            if verbose:
                print(f"Visualization {vis_in} not found, skipping")
            continue

        shutil.copy(vis_in, vis_out)
    return True


def process_measurements(
    in_files, output_folder, verbose=False, quiet=False, write_log_file=True
):
    measures_out = f"{output_folder}AortaExplorer_measurements.csv"

    if verbose:
        print(
            f"Gathering measurements from {len(in_files)} files. Output to {output_folder}"
        )

    all_column_names = gather_all_stat_columen_names(
        in_files, output_folder, verbose, quiet, write_log_file
    )
    if len(all_column_names) < 1:
        msg = f"No measurement files found in {output_folder}"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    if verbose:
        print(f"Found {len(all_column_names)} different measurement columns")

    try:
        f = open(measures_out, "w")
        for c in all_column_names:
            f.write(f"{c},")
        f.write("\n")
        f.close()

        get_all_measurement(
            in_files, all_column_names, output_folder, verbose, quiet, write_log_file
        )
    except Exception as e:
        msg = f"Error writing to {measures_out}: {str(e)}"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )

    copy_all_visualization(in_files, output_folder, verbose, quiet, write_log_file)

    return True
