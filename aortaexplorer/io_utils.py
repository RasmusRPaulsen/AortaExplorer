import tempfile
from pathlib import Path
import nibabel as nib
import SimpleITK as sitk
from aortaexplorer.general_utils import write_message_to_log_file
from functools import cache
import os


def read_nifti_file_robustly(image_file: str):
    """
    Reads a NIfTI file robustly using SimpleITK. If SimpleITK fails to read the file,
    it falls back to using nibabel to read and convert the file before reading it again with
    SimpleITK.
    It is needed when there are issues with the orthonal qform/sform matrices in the NIfTI header.
    From:
    https://github.com/MIC-DKFZ/nnDetection/issues/24#issuecomment-2627684467
    """
    message = ""

    if not os.path.isfile(image_file):
        message = f"File {image_file} does not exist."
        return None, message

    try:
        image = sitk.ReadImage(image_file)
    except RuntimeError as e:
        message = f"SimpleITK could not to read {image_file} with error: {e}. Falling back to nibabel."
        image = None
    if image is None:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpfile = Path(tmpdir) / "tmp.nii.gz"
                img = nib.load(image_file)
                nib.save(nib.Nifti1Image(img.get_fdata(), img.get_qform()), tmpfile)
                if tmpfile is not None:
                    try:
                        image = sitk.ReadImage(tmpfile)
                    except RuntimeError as e:
                        message += f"SimpleITK could not to read NIBABEL converted {tmpfile} with error: {e}. Giving up."
                        image = None
        except RuntimeError as e:
            message += f" nibabel also failed to read {image_file} with error: {e}."
            image = None
    return image, message


def read_nifti_with_logging(image_file, verbose, quiet, write_log_file, output_folder):
    """
    Reads a NIfTI file with logging options.
    """
    image, message = read_nifti_file_robustly(image_file)
    if image is None:
        if not quiet:
            print(message)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=message, level="error"
            )
        return None

    if message != "":
        if verbose and not quiet:
            print(message)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=message, level="warning"
            )
    return image


@cache
def read_nifti_with_logging_cached(
    image_file, verbose, quiet, write_log_file, output_folder
):
    """
    Reads a NIfTI file with logging options.
    Caches the result to avoid redundant reads.
    NOTE: Should only be used with files that are not modified during runtime.
    for example input image files
    """
    image, message = read_nifti_file_robustly(image_file)
    if image is None:
        if not quiet:
            print(message)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=message, level="error"
            )
        return None

    if message != "":
        if verbose and not quiet:
            print(message)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=message, level="warning"
            )
    return image
