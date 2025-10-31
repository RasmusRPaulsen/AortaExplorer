import os
from scipy.ndimage import measurements
from pathlib import Path
import time
import multiprocessing as mp
import aortaexplorer.surface_utils as surfutils
from aortaexplorer.general_utils import (
    write_message_to_log_file,
    read_json_file,
    get_last_error_message,
    clear_last_error_message,
)
from aortaexplorer.io_utils import read_nifti_with_logging_cached
from aortaexplorer.surface_utils import (
    convert_label_map_to_surface,
    compute_min_and_max_z_landmark,
    find_closests_points_on_two_surfaces_with_start_point,
    preprocess_surface_for_centerline_extraction,
)
from aortaexplorer.segmentation_utils import (
    get_components_over_certain_size,
    get_components_over_certain_size_as_individual_volumes,
    edt_based_opening,
    edt_based_closing,
    close_cavities_in_segmentations,
    edt_based_dilation,
    compute_segmentation_volume,
    edt_based_overlap,
    edt_based_compute_landmark_from_segmentation_overlap,
    read_nifti_itk_to_numpy,
    check_if_segmentation_hit_sides_of_scan,
)
from aortaexplorer.aorta_centerline_utils import AortaCenterliner
import aortaexplorer.centerline_utils as clutils
from aortaexplorer.visualization_utils import RenderAortaData
import SimpleITK as sitk
import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label
from skimage.segmentation import find_boundaries
import edt
import json
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import imageio
import skimage.io
from skimage import color
from skimage.util import img_as_ubyte
import importlib.metadata


def setup_vtk_error_handling(err_dir):
    """
    Create a text file where potential VTK errors are dumped instead of a window popping up
    """
    error_out_file = os.path.join(err_dir, "vtk_errors.txt")

    error_out = vtk.vtkFileOutputWindow()
    error_out.SetFileName(error_out_file)
    vtk_std_error_out = vtk.vtkOutputWindow()
    vtk_std_error_out.SetInstance(error_out)


def compute_body_segmentation(
    input_file, segm_folder, verbose, quiet, write_log_file, output_folder
):
    """
    Use simple HU thresholding to compute the body segmentation
    """
    low_thresh = -200
    high_thresh = 1500
    segm_out_name = f"{segm_folder}body.nii.gz"

    if os.path.exists(segm_out_name):
        if not quiet:
            print(f"{segm_out_name} already exists - skipping")
        return True

    if verbose:
        print(f"Computing {segm_out_name}")

    ct_img = read_nifti_with_logging_cached(
        input_file, verbose, quiet, write_log_file, output_folder
    )
    if ct_img is None:
        return False

    #
    # if not os.path.exists(input_file):
    #     msg = f"Could not find {input_file} for extracting body segmentation"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     return False
    #
    # try:
    #     ct_img = sitk.ReadImage(input_file)
    # except RuntimeError as e:
    #     msg = f"Could not read {input_file} for body segmentation: {str(e)} got an exception"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     return False

    ct_np = sitk.GetArrayFromImage(ct_img)

    img_mask_1 = low_thresh < ct_np
    img_mask_2 = ct_np < high_thresh
    combined_mask = np.bitwise_and(img_mask_1, img_mask_2)

    combined_mask, _ = get_components_over_certain_size(combined_mask, 5000, 1)
    if combined_mask is None:
        msg = f"Could not find body segmentation in {input_file}. No connected components found."
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    img_o = sitk.GetImageFromArray(combined_mask.astype(int))
    img_o.CopyInformation(ct_img)

    # print(f"saving")
    sitk.WriteImage(img_o, segm_out_name)

    return True


def compute_out_scan_field_segmentation_and_sdf(
    input_file, segm_folder, params, verbose, quiet, write_log_file, output_folder
):
    """
    Use simple HU thresholding to compute the area of the scan that is marked as
    invalid values (very low HU). Also compute an SDF so it can be quickly determined
    if a segmentation is close to the scan side.
    """
    segm_out_name = f"{segm_folder}out_of_scan.nii.gz"
    sdf_out_name = f"{segm_folder}out_of_scan_sdf.nii.gz"
    low_thresh = params["out_of_reconstruction_value"]

    # ct_name = input_file
    # low_thresh = -2000
    high_thresh = 16000

    if os.path.exists(segm_out_name) and os.path.exists(sdf_out_name):
        if verbose:
            print(f"{segm_out_name} already exists - skipping")
        return True

    if verbose:
        print(f"Computing out-of-scan-field:{segm_out_name} and {sdf_out_name}")

    ct_img = read_nifti_with_logging_cached(
        input_file, verbose, quiet, write_log_file, output_folder
    )
    if ct_img is None:
        return False
    #
    # try:
    #     ct_img = sitk.ReadImage(ct_name)
    # except RuntimeError as e:
    #     msg = f"Could not read {input_file} for body segmentation: {str(e)} got an exception"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     return False

    ct_np = sitk.GetArrayFromImage(ct_img)

    combined_mask = (low_thresh >= ct_np) | (ct_np > high_thresh)

    # mark all the sides as well
    combined_mask[:, :, 0] = True
    combined_mask[:, :, -1] = True
    combined_mask[:, 0, :] = True
    combined_mask[:, -1, :] = True
    combined_mask[0, :, :] = True
    combined_mask[-1, :, :] = True

    img_o = sitk.GetImageFromArray(combined_mask.astype(int))
    img_o.CopyInformation(ct_img)
    # print(f"saving")
    sitk.WriteImage(img_o, segm_out_name)

    if verbose:
        print("Computing SDF for out-of-scan-field")
    spacing = ct_img.GetSpacing()
    sdf_mask = -edt.sdf(
        combined_mask,
        anisotropy=[spacing[2], spacing[1], spacing[0]],
        parallel=8,  # number of threads, <= 0 sets to num CPU
    )

    # For certain scan with very isotropic voxels (CFA with spacing up to 3 mm)
    # we need a larger SDF for later out-of-scan-queries
    max_space = np.max(np.asarray(spacing))
    max_dist_2 = 6 * max_space

    # Clamp SDF to make nifti file smaller
    max_dist_to_keep = max(6.0, max_dist_2)
    sdf_mask = np.clip(sdf_mask, -max_dist_to_keep, max_dist_to_keep)

    img_o = sitk.GetImageFromArray(sdf_mask)
    img_o.CopyInformation(ct_img)
    sitk.WriteImage(img_o, sdf_out_name)

    return True


def refine_single_aorta_part(
    input_file,
    params,
    segm_folder,
    stats_folder,
    verbose,
    quiet,
    write_log_file,
    output_folder,
    ct_np,
    part="_annulus",
):
    """
    This is one of the core functions of AortaExplorer - it takes the
    aorta segmentation from TotalSegmentator and refines it to only contain
    the lumen. It does this by computing HU statistics from a skeleton
    of the aorta and then thresholds the aorta based on these statistics.
    """
    segm_in_name = f"{segm_folder}aorta_lumen{part}_raw.nii.gz"
    segm_out_name = f"{segm_folder}aorta_lumen{part}.nii.gz"
    segm_skeleton_out_name = f"{segm_folder}aorta_lumen{part}_skeleton.nii.gz"
    segm_out_thres_name = f"{segm_folder}aorta_lumen{part}_thresholded.nii.gz"
    segm_open_out_name = f"{segm_folder}aorta_lumen{part}_open.nii.gz"
    segm_closed_out_name = f"{segm_folder}aorta_lumen{part}_closed.nii.gz"
    segm_no_holes_name = f"{segm_folder}aorta_lumen{part}_no_holes.nii.gz"
    forced_min_hu_value = params.get("forced_aorta_min_hu_value", None)
    forced_max_hu_value = params.get("forced_aorta_max_hu_value", None)
    hu_stats_file = f"{stats_folder}/aorta_skeleton{part}_hu_stats.json"

    debug = False

    if os.path.exists(segm_out_name):
        if verbose:
            print(f"{segm_out_name} already exists - skipping")
        return True

    if verbose:
        print(f"Computing pure aorta lumen segmentation to {segm_out_name}")

    if not os.path.exists(segm_in_name):
        msg = f"Could not find {segm_in_name} can not compute {part} aorta lumen"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    label_img_aorta = read_nifti_with_logging_cached(
        segm_in_name, verbose, quiet, write_log_file, output_folder
    )
    if label_img_aorta is None:
        return False
    #
    # try:
    #     label_img_aorta = sitk.ReadImage(segm_in_name)
    # except RuntimeError as e:
    #     msg = f"Could not read {segm_in_name}: {str(e)} got an exception"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     return False

    spacing = label_img_aorta.GetSpacing()
    # in_slice_spacing = spacing[0]
    slice_thickness = spacing[2]
    # ratio = slice_thickness / in_slice_spacing
    # ratio = max(ratio, 1.0)

    label_img_aorta_np = sitk.GetArrayFromImage(label_img_aorta)
    mask_np_aorta = label_img_aorta_np == 1

    if debug:
        print("SDF for erosion")
    sdf_mask = -edt.sdf(
        mask_np_aorta,
        anisotropy=[spacing[2], spacing[1], spacing[0]],
        parallel=8,  # number of threads, <= 0 sets to num CPU
    )

    erode_size = 2
    eroded_mask = sdf_mask < -erode_size

    if verbose:
        print(f"Skeletonizing aorta from {segm_skeleton_out_name}")
    skeleton = skeletonize(eroded_mask)

    if verbose:
        print("SDF for dilation of skeleton")
    sdf_mask = -edt.sdf(
        skeleton, anisotropy=[spacing[2], spacing[1], spacing[0]], parallel=8
    )  # number of threads, <= 0 sets to num CPU

    dilate_size = 2
    dilated_mask = sdf_mask < dilate_size

    if debug:
        img_o = sitk.GetImageFromArray(dilated_mask.astype(int))
        img_o.CopyInformation(label_img_aorta)

        print(f"Debug: saving {segm_skeleton_out_name}")
        sitk.WriteImage(img_o, segm_skeleton_out_name)

    hu_values = ct_np[dilated_mask > 0]

    # Check for out of scanfield values
    hu_in_mask = hu_values > -2000
    hu_values = hu_values[hu_in_mask]

    if len(hu_values) < 10:
        msg = f"Too few voxels in {segm_skeleton_out_name} - can not compute HU statistics"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    stats = {}
    avg_hu = np.average(hu_values)
    q01_hu = np.percentile(hu_values, 1)
    stats["skeleton_avg_hu"] = avg_hu
    stats["skeleton_std_hu"] = np.std(hu_values)
    stats["skeleton_med_hu"] = np.median(hu_values)
    stats["skeleton_q99_hu"] = np.percentile(hu_values, 99)
    stats["skeleton_q01_hu"] = q01_hu
    # print(stats)

    hu_stdev = stats["skeleton_std_hu"]
    low_thresh = avg_hu - 5 * hu_stdev
    high_thresh = avg_hu + 3 * hu_stdev

    # TODO: Update this to something smarter
    high_std = False
    if hu_stdev > 100:
        high_std = True
        msg = f"High HU stdev in {input_file} avg: {avg_hu:.1f} stdev: {hu_stdev:.1f}. Could be caused by metal implants."
        if verbose:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="warning"
            )

    low_hu_values = False
    if q01_hu < params["aorta_min_hu_value"]:
        low_hu_values = True
        msg = f"Low threshold HU values in {input_file} avg: {avg_hu:.1f} q01: {q01_hu:.1f} low thresh {low_thresh:.1f}"
        if high_std:
            msg += f". Could be caused by the high stdev {hu_stdev}."
        if verbose:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="warning"
            )

    low_thresh = max(low_thresh, params["aorta_min_hu_value"])
    high_thresh = max(high_thresh, params["aorta_min_max_hu_value"])
    if forced_min_hu_value is not None and forced_min_hu_value > 0:
        low_thresh = forced_min_hu_value
        stats["forced_min_hu_value"] = forced_min_hu_value
    if forced_max_hu_value is not None and forced_max_hu_value > 0:
        high_thresh = forced_max_hu_value
        stats["forced_max_hu_value"] = forced_max_hu_value

    stats["low_thresh"] = low_thresh
    stats["high_thresh"] = high_thresh

    json_object = json.dumps(stats, indent=4)
    with open(hu_stats_file, "w") as outfile:
        outfile.write(json_object)

    if verbose:
        print(
            f"HU Average: {avg_hu:.1f} stdev {hu_stdev:.1f} low_thresh: {low_thresh:.1f} high_thresh {high_thresh:.1f}"
        )

    img_mask_1 = low_thresh < ct_np
    img_mask_2 = ct_np < high_thresh
    combined_mask = np.bitwise_and(img_mask_1, img_mask_2)
    combined_mask = np.bitwise_and(combined_mask, mask_np_aorta)

    combined_mask, _ = close_cavities_in_segmentations(combined_mask)

    if debug:
        img_o = sitk.GetImageFromArray(combined_mask.astype(int))
        img_o.CopyInformation(label_img_aorta)

        print(f"Debug: saving {segm_out_thres_name}")
        sitk.WriteImage(img_o, segm_out_thres_name)

    # TODO: 2 mm is guesswork
    est_open_close_radius = 2.0

    open_close_radius = est_open_close_radius

    do_open_close = True
    if do_open_close:
        if verbose:
            print("EDT based opening")
        combined_mask = edt_based_opening(
            combined_mask, [spacing[2], spacing[1], spacing[0]], open_close_radius
        )
        if debug:
            img_o = sitk.GetImageFromArray(combined_mask.astype(int))
            img_o.CopyInformation(label_img_aorta)

            print(f"Debug: saving {segm_open_out_name}")
            sitk.WriteImage(img_o, segm_open_out_name)

        if verbose:
            print("EDT based closing")
        combined_mask = edt_based_closing(
            combined_mask, [spacing[2], spacing[1], spacing[0]], open_close_radius
        )
        if debug:
            img_o = sitk.GetImageFromArray(combined_mask.astype(int))
            img_o.CopyInformation(label_img_aorta)

            print(f"Debug: saving {segm_closed_out_name}")
            sitk.WriteImage(img_o, segm_closed_out_name)

    if verbose:
        print("Closing cavities")
    combined_mask, _ = close_cavities_in_segmentations(combined_mask)

    # Remove invalid out-of-scan voxels (typically values -2048)
    # here we set it to something that should not be in an aorta
    combined_mask = (-200 < ct_np) & combined_mask

    # Do a sanity check of the estimated aorta based on the surface to volume ratio
    aorta_volume = np.sum(combined_mask) * spacing[0] * spacing[1] * spacing[2]
    if aorta_volume is None:
        msg = f"Could not compute volume of aorta of {segm_in_name}. Something wrong with refinement."
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    img_o = sitk.GetImageFromArray(combined_mask.astype(int))
    img_o.CopyInformation(label_img_aorta)
    sitk.WriteImage(img_o, segm_no_holes_name)

    aorta_surface = convert_label_map_to_surface(
        segm_no_holes_name, only_largest_component=False
    )
    if aorta_surface is None:
        msg = f"Could not compute surface area of {segm_in_name}. Something wrong with refinement."
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    if aorta_surface.GetNumberOfPoints() < 10:
        msg = f"Could not compute surface area of {segm_in_name}. Something wrong with refinement."
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    mass = vtk.vtkMassProperties()
    mass.SetInputData(aorta_surface)
    mass.Update()

    aorta_surface = mass.GetSurfaceArea()

    surface_volume_ratio = aorta_surface / aorta_volume
    if low_hu_values and surface_volume_ratio > 0.22:
        msg = f"Surface to volume ratio {surface_volume_ratio:.3f} is higher than 0.22 and very low HU units avg: {avg_hu:.1f} - we do not dare to do more analysis {input_file}"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    # TODO: This is completely guesswork
    if slice_thickness < 1:
        min_comp_size = 25000
    elif slice_thickness < 2:
        min_comp_size = 15000
    else:
        min_comp_size = 10000

    if verbose:
        print(f"Finding componenents with min_comp_size: {min_comp_size}")
    components = get_components_over_certain_size_as_individual_volumes(
        combined_mask, min_comp_size, 1
    )
    if components is None:
        msg = (
            f"Could not find any components of size > {min_comp_size} in {segm_in_name}"
        )
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    n_components = len(components)

    if n_components < 1:
        msg = f"Could not find any components of size > {min_comp_size} in {segm_in_name} for {part}"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    img_o = sitk.GetImageFromArray(components[0].astype(int))
    img_o.CopyInformation(label_img_aorta)

    if debug:
        print(f"Debug: saving {segm_out_name}")
    sitk.WriteImage(img_o, segm_out_name)

    return True


def extract_pure_aorta_lumen_start_by_finding_parts(
    input_file,
    params,
    segm_folder,
    stats_folder,
    verbose,
    quiet,
    write_log_file,
    output_folder,
):
    """
    Take the original aorta segmentation and keep only the lumen
    """
    store_raw_hires_aorta = True
    store_raw_aorta = True
    segm_out_name = f"{segm_folder}aorta_lumen_raw.nii.gz"
    segm_out_name_hires = f"{segm_folder}aorta_lumen_hires_raw.nii.gz"
    segm_out_name_raw = f"{segm_folder}aorta_ts_original.nii.gz"
    segm_out_name_annulus = f"{segm_folder}aorta_lumen_annulus_raw.nii.gz"
    segm_out_name_descending = f"{segm_folder}aorta_lumen_descending_raw.nii.gz"
    segm_in_name_annulus = f"{segm_folder}aorta_lumen_annulus.nii.gz"
    segm_in_name_descending = f"{segm_folder}aorta_lumen_descending.nii.gz"
    segm_total_out = f"{segm_folder}aorta_lumen.nii.gz"
    stats_file = f"{stats_folder}/aorta_parts.json"
    stats_file_type = f"{stats_folder}/aorta_scan_type.json"
    segm_name_aorta = f"{segm_folder}total.nii.gz"
    segm_name_aorta_hires = f"{segm_folder}heartchambers_highres.nii.gz"
    aorta_segm_id = 52
    aorta_hires_segm_id = 6

    # ct_name = input_file
    # debug = False

    if os.path.exists(segm_out_name):
        if verbose:
            print(f"{segm_out_name} already exists - skipping")
        return True

    if verbose:
        print(f"Computing pure aorta lumen segmentation to {segm_out_name}")

    if not os.path.exists(segm_name_aorta):
        msg = f"TotalSegmentator aorta segmentation {segm_name_aorta} not found. Can not extract aorta lumen."
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    highres_present = os.path.exists(segm_name_aorta_hires)
    if verbose:
        if highres_present:
            print(f"Found high-res heart segmentation {segm_name_aorta_hires}")
        else:
            print(f"Did not find high-res heart segmentation {segm_name_aorta_hires}")

    if verbose:
        print(f"Computing {segm_out_name}")

    label_img_aorta = read_nifti_with_logging_cached(
        segm_name_aorta, verbose, quiet, write_log_file, output_folder
    )
    if label_img_aorta is None:
        return False

    # try:
    #     label_img_aorta = sitk.ReadImage(segm_name_aorta)
    # except RuntimeError as e:
    #     msg = f"Could not read {segm_name_aorta}: {str(e)} got an exception"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     return False

    label_img_aorta_hires = None
    if os.path.exists(segm_name_aorta_hires):
        label_img_aorta_hires = read_nifti_with_logging_cached(
            segm_name_aorta_hires, verbose, quiet, write_log_file, output_folder
        )
        if label_img_aorta_hires is None:
            return False
        # try:
        #     label_img_aorta_hires = sitk.ReadImage(segm_name_aorta_hires)
        # except RuntimeError as e:
        #     msg = f"Could not read {segm_name_aorta_hires}: {str(e)} got an exception"
        #     if not quiet:
        #         print(msg)
        #     if write_log_file:
        #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
        #     return False

    ct_img = read_nifti_with_logging_cached(
        input_file, verbose, quiet, write_log_file, output_folder
    )
    if ct_img is None:
        return False
    #
    # try:
    #     ct_img = sitk.ReadImage(ct_name)
    # except RuntimeError as e:
    #     msg = f"Could not read {input_file}: {str(e)} got an exception"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     return False

    ct_np = sitk.GetArrayFromImage(ct_img)

    spacing = ct_img.GetSpacing()
    # in_slice_spacing = spacing[0]
    slice_thickness = spacing[2]
    # ratio = slice_thickness / in_slice_spacing
    # ratio = max(ratio, 1.0)

    label_img_aorta_np = sitk.GetArrayFromImage(label_img_aorta)
    mask_np_aorta = label_img_aorta_np == aorta_segm_id

    if store_raw_aorta:
        img_o = sitk.GetImageFromArray(mask_np_aorta.astype(int))
        img_o.CopyInformation(label_img_aorta)
        sitk.WriteImage(img_o, segm_out_name_raw)

    if label_img_aorta_hires:
        # print("Adding hires aorta to baseline segmentation")
        label_img_aorta_hires_np = sitk.GetArrayFromImage(label_img_aorta_hires)
        mask_np_aorta_hires = label_img_aorta_hires_np == aorta_hires_segm_id
        mask_np_aorta = np.bitwise_or(mask_np_aorta, mask_np_aorta_hires)

    if store_raw_hires_aorta:
        img_o = sitk.GetImageFromArray(mask_np_aorta.astype(int))
        img_o.CopyInformation(label_img_aorta)
        sitk.WriteImage(img_o, segm_out_name_hires)

    # TODO: This is completely guesswork
    if slice_thickness < 1:
        min_comp_size = 25000
    elif slice_thickness < 2:
        min_comp_size = 15000
    else:
        min_comp_size = 10000

    if verbose:
        print(f"Finding aorta components with min_comp_size: {min_comp_size} voxels")
    components = get_components_over_certain_size_as_individual_volumes(
        mask_np_aorta, min_comp_size, 2
    )
    if components is None:
        msg = f"No aorta lumen found left after connected components {input_file}"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )

        type_stats = {
            "scan_type": "unknown",
            "scan_type_desc": "Unknown -No aorta found in scan",
        }
        try:
            with Path(stats_file_type).open("wt") as handle:
                json.dump(type_stats, handle, indent=4, sort_keys=False)
        except IOError as e:
            print(f"I/O error({e.errno}): {e.strerror}: {stats_file}")
        return False

    n_components = len(components)
    aorta_parts = {"aorta_parts": n_components}
    json_object = json.dumps(aorta_parts, indent=4)
    with open(stats_file, "w") as outfile:
        outfile.write(json_object)

    if verbose:
        print(f"Found {n_components} components and now refining them")
    if n_components == 2:
        # divide into two parts: the annulus and the descending part
        # The annulus part is more anterior
        com_1 = measurements.center_of_mass(components[0])
        com_2 = measurements.center_of_mass(components[1])
        if com_1[1] > com_2[1]:
            name_o_1 = segm_out_name_annulus
            name_o_2 = segm_out_name_descending
        else:
            name_o_2 = segm_out_name_annulus
            name_o_1 = segm_out_name_descending

        img_o = sitk.GetImageFromArray(components[0].astype(int))
        img_o.CopyInformation(label_img_aorta)
        # print(f"saving {name_o_1}")
        sitk.WriteImage(img_o, name_o_1)

        img_o = sitk.GetImageFromArray(components[1].astype(int))
        img_o.CopyInformation(label_img_aorta)
        # print(f"saving {name_o_2}")
        sitk.WriteImage(img_o, name_o_2)

        # Combine to make one lumen segmentation
        combined = np.bitwise_or(components[0], components[1])
        img_o = sitk.GetImageFromArray(combined.astype(int))
        img_o.CopyInformation(label_img_aorta)

        # print(f"saving {segm_out_name}")
        sitk.WriteImage(img_o, segm_out_name)

        if not refine_single_aorta_part(
            input_file,
            params,
            segm_folder,
            stats_folder,
            verbose,
            quiet,
            write_log_file,
            output_folder,
            ct_np,
            "_descending",
        ):
            return False
        if not refine_single_aorta_part(
            input_file,
            params,
            segm_folder,
            stats_folder,
            verbose,
            quiet,
            write_log_file,
            output_folder,
            ct_np,
            "_annulus",
        ):
            return False

        # make a combined segmentation
        label_img_annulus = read_nifti_with_logging_cached(
            segm_in_name_annulus, verbose, quiet, write_log_file, output_folder
        )
        if label_img_annulus is None:
            return False

        # try:
        #     label_img_annulus = sitk.ReadImage(segm_in_name_annulus)
        # except RuntimeError as e:
        #     msg = f"Could not read {segm_in_name_annulus}: {str(e)} got an exception"
        #     if not quiet:
        #         print(msg)
        #     if write_log_file:
        #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
        #     return False

        label_img_descending = read_nifti_with_logging_cached(
            segm_in_name_descending, verbose, quiet, write_log_file, output_folder
        )
        if label_img_descending is None:
            return False
        #
        # try:
        #     label_img_descending = sitk.ReadImage(segm_in_name_descending)
        # except RuntimeError as e:
        #     msg = f"Could not read {segm_in_name_descending}: {str(e)} got an exception"
        #     if not quiet:
        #         print(msg)
        #     if write_log_file:
        #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
        #     return False

        label_img_comb = sitk.GetArrayFromImage(
            label_img_annulus
        ) | sitk.GetArrayFromImage(label_img_descending)

        img_o = sitk.GetImageFromArray(label_img_comb)
        img_o.CopyInformation(label_img_aorta)
        sitk.WriteImage(img_o, segm_total_out)
    else:
        img_o = sitk.GetImageFromArray(components[0].astype(int))
        img_o.CopyInformation(label_img_aorta)

        # print(f"saving {segm_out_name}")
        sitk.WriteImage(img_o, segm_out_name)
        if not refine_single_aorta_part(
            input_file,
            params,
            segm_folder,
            stats_folder,
            verbose,
            quiet,
            write_log_file,
            output_folder,
            ct_np,
            "",
        ):
            return False

    return True


def extract_top_of_iliac_arteries(
    input_file, segm_folder, verbose, quiet, write_log_file, output_folder
):
    """
    Extract the top part of the iliac arteries if present
    """
    segm_out_l_name = f"{segm_folder}iliac_artery_left_top.nii.gz"
    segm_out_r_name = f"{segm_folder}iliac_artery_right_top.nii.gz"

    segm_in_name = f"{segm_folder}total.nii.gz"
    iliac_left_segm_id = 65
    iliac_right_segm_id = 66
    top_length = 10.0  # mm
    min_size = 10 * 5 * 5  # mm^3

    if os.path.exists(segm_out_l_name) and os.path.exists(segm_out_r_name):
        if verbose:
            print(f"{segm_out_l_name} and {segm_out_r_name} already exist - skipping")
        return True

    if verbose:
        print(
            f"Extracting top {top_length} mm of iliac arteries to {segm_out_l_name} and {segm_out_r_name}"
        )

    label_img = read_nifti_with_logging_cached(
        segm_in_name, verbose, quiet, write_log_file, output_folder
    )
    if label_img is None:
        return False
    #
    # try:
    #     label_img = sitk.ReadImage(segm_in_name)
    # except RuntimeError as e:
    #     msg = f"Could not read {segm_in_name}: {str(e)} got an exception"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     return False

    spacing = label_img.GetSpacing()
    # in_slice_spacing = spacing[0]
    slice_thickness = spacing[2]

    n_top_slices = int(top_length / slice_thickness)
    min_size_vox = min_size / (spacing[0] * spacing[1] * spacing[2])
    if verbose:
        print(f"n_top_slices: {n_top_slices} min_size_vox: {min_size_vox:.1f}")

    label_img_np = sitk.GetArrayFromImage(label_img)
    mask_np_left = label_img_np == iliac_left_segm_id
    if np.sum(mask_np_left) == 0:
        if verbose:
            print(f"No iliac artery left found in {input_file}")
    else:
        shp = mask_np_left.shape
        start_slice = None
        # Remember that NP are in the order of z,y,x
        for z in range(shp[0] - 1, 0, -1):
            if sum(sum(mask_np_left[z, :, :])) > 0:
                if start_slice is None:
                    start_slice = z
                elif start_slice - z > n_top_slices:
                    mask_np_left[z, :, :] = 0

        large_components, _ = get_components_over_certain_size(
            mask_np_left, min_size_vox, 1
        )
        if large_components is None or np.sum(large_components) == 0:
            if verbose:
                print(f"No iliac artery left found in {input_file}")
        else:
            img_o = sitk.GetImageFromArray(large_components.astype(int))
            img_o.CopyInformation(label_img)
            sitk.WriteImage(img_o, segm_out_l_name)

    mask_np_right = label_img_np == iliac_right_segm_id
    if np.sum(mask_np_right) == 0:
        if verbose:
            print(f"No iliac artery right found in {input_file}")
    else:
        shp = mask_np_right.shape
        start_slice = None
        # Remember that NP are in the order of z,y,x
        for z in range(shp[0] - 1, 0, -1):
            if sum(sum(mask_np_right[z, :, :])) > 0:
                if start_slice is None:
                    start_slice = z
                elif start_slice - z > n_top_slices:
                    mask_np_right[z, :, :] = 0

        large_components, _ = get_components_over_certain_size(
            mask_np_right, min_size_vox, 1
        )
        if large_components is None or np.sum(large_components) == 0:
            if verbose:
                print(f"No iliac artery right found in {input_file}")
        else:
            img_o = sitk.GetImageFromArray(large_components.astype(int))
            img_o.CopyInformation(label_img)
            sitk.WriteImage(img_o, segm_out_r_name)
    return True


def extract_aortic_calcifications(
    input_file,
    params,
    segm_folder,
    stats_folder,
    verbose,
    quiet,
    write_log_file,
    output_folder,
):
    """ """
    hu_stats_file = f"{stats_folder}/aorta_skeleton_hu_stats.json"
    hu_stats_file_2 = f"{stats_folder}/aorta_skeleton_descending_hu_stats.json"
    calcification_stats_file = f"{stats_folder}/aorta_calcification_stats.json"

    segm_in_name_hires = f"{segm_folder}aorta_lumen_hires_raw.nii.gz"

    segm_out_name = f"{segm_folder}aorta_calcification.nii.gz"
    segm_dilated_out_name = f"{segm_folder}aorta_lumen_dilated.nii.gz"
    segm_raw_out_name = f"{segm_folder}aorta_calcification_raw.nii.gz"
    # segm_out_name_annulus = f'{segm_folder}aorta_calcification_annulus.nii.gz'
    # segm_out_name_descending = f'{segm_folder}aorta_calcification_descending.nii.gz'
    segm_name_lumen = f"{segm_folder}aorta_lumen.nii.gz"
    calc_min_hu = params["aorta_calcification_min_hu_value"]
    calc_max_hu = params["aorta_calcification_max_hu_value"]

    aorta_segm_id = 1
    debug = False

    if os.path.exists(calcification_stats_file):
        if verbose:
            print(f"{calcification_stats_file} already exists - skipping")
        return True

    if verbose:
        print(f"Computing aortic calcifications to {segm_out_name}")

    stats = {}
    label_img_aorta = read_nifti_with_logging_cached(
        segm_in_name_hires, verbose, quiet, write_log_file, output_folder
    )
    if label_img_aorta is None:
        return False

    ct_img = read_nifti_with_logging_cached(
        input_file, verbose, quiet, write_log_file, output_folder
    )
    if ct_img is None:
        return False

    # try:
    #     label_img_aorta = sitk.ReadImage(segm_in_name_hires)
    # except RuntimeError as e:
    #     msg = f"Could not read {segm_in_name_hires}: {str(e)} got an exception"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     return False

    # try:
    #     ct_img = sitk.ReadImage(input_file)
    # except RuntimeError as e:
    #     msg = f"Could not read {input_file} for calcification segmentation: {str(e)} got an exception"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     return False

    spacing = ct_img.GetSpacing()
    label_img_aorta_np = sitk.GetArrayFromImage(label_img_aorta)
    mask_np_aorta = label_img_aorta_np == aorta_segm_id
    ct_np = sitk.GetArrayFromImage(ct_img)

    if debug:
        print("SDF for dilation")
    # 1 mm radius
    dilation_size = 1
    dilated_mask = edt_based_dilation(
        mask_np_aorta, [spacing[2], spacing[1], spacing[0]], dilation_size
    )

    dilated_mask, _ = get_components_over_certain_size(dilated_mask, 5000, 2)
    if debug:
        img_o = sitk.GetImageFromArray(dilated_mask.astype(int))
        img_o.CopyInformation(label_img_aorta)

        print(f"Debug: saving {segm_dilated_out_name}")
        sitk.WriteImage(img_o, segm_dilated_out_name)

    # low_thresh = 400
    # # TODO: Should this be different - what about metals?
    # high_thresh = 1500
    low_thresh = calc_min_hu
    high_thresh = calc_max_hu

    # We try both
    hu_stats = read_json_file(hu_stats_file)
    if not hu_stats:
        hu_stats = read_json_file(hu_stats_file_2)
    if hu_stats:
        low_thresh = hu_stats["high_thresh"]

    if verbose:
        print(f"Calcification: min HU: {low_thresh} max HU: {high_thresh} ")

    img_mask_1 = low_thresh < ct_np
    img_mask_2 = ct_np < high_thresh
    combined_mask = np.bitwise_and(img_mask_1, img_mask_2)
    # combined_mask = np.bitwise_and(combined_mask, mask_np_aorta)
    combined_mask = np.bitwise_and(combined_mask, dilated_mask)

    calcification_volume = np.sum(combined_mask) * spacing[0] * spacing[1] * spacing[2]
    stats["calcification_volume"] = calcification_volume

    # if debug:
    img_o = sitk.GetImageFromArray(combined_mask.astype(int))
    img_o.CopyInformation(label_img_aorta)

    if debug:
        print(f"Saving {segm_raw_out_name}")
    sitk.WriteImage(img_o, segm_raw_out_name)

    aorta_lumen_volume = compute_segmentation_volume(segm_name_lumen, segm_id=1)
    stats["aorta_lumen_volume"] = aorta_lumen_volume

    try:
        with Path(calcification_stats_file).open("wt") as handle:
            json.dump(stats, handle, indent=4, sort_keys=False)
    except IOError as e:
        print(f"I/O error({e.errno}): {e.strerror}: {calcification_stats_file}")
        return False

    return True


def check_for_aneurysm_sac(
    segm_folder, stats_folder, verbose, quiet, write_log_file, output_folder
):
    """
    Compare the raw TotalSegmentator aorta segmentation and the refined lumen segmentation
    to see if there is an aneurysm sac present
    Since the TotalSegmentator segmentation includes the wall and possible sacs
    """
    aneurysm_sac_stats_file = f"{stats_folder}aorta_aneurysm_sac_stats.json"
    calcification_stats_file = f"{stats_folder}aorta_calcification_stats.json"
    segm_in_name_hires = f"{segm_folder}aorta_lumen_hires_raw.nii.gz"
    segm_lumen_in = f"{segm_folder}aorta_lumen.nii.gz"

    if os.path.exists(aneurysm_sac_stats_file):
        if verbose:
            print(f"{aneurysm_sac_stats_file} already exists - skipping")
        return True
    if verbose:
        print(f"Computing aneurysm sac statistics to {aneurysm_sac_stats_file}")

    volume_stats = {}

    # n_aorta_parts = 1
    # parts_stats = read_json_file(stats_file)
    # if parts_stats:
    #     n_aorta_parts = parts_stats["aorta_parts"]
    #
    # if n_aorta_parts != 1:
    #     print(f"Found {n_aorta_parts} aorta parts - not computing aneurysm sac statistics. We can only handle one part.")
    #     return True

    label_img_aorta_raw = read_nifti_with_logging_cached(
        segm_in_name_hires, verbose, quiet, write_log_file, output_folder
    )
    if label_img_aorta_raw is None:
        return False

    label_img_aorta_lumen = read_nifti_with_logging_cached(
        segm_lumen_in, verbose, quiet, write_log_file, output_folder
    )
    if label_img_aorta_lumen is None:
        return False

    # try:
    #     label_img_aorta_raw = sitk.ReadImage(segm_in_name_hires)
    # except RuntimeError as e:
    #     msg = f"Could not read {segm_in_name_hires}: {str(e)} got an exception"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     return False

    # try:
    #     label_img_aorta_lumen = sitk.ReadImage(segm_lumen_in)
    # except RuntimeError as e:
    #     msg = f"Could not read {segm_lumen_in}: {str(e)} got an exception"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     return False

    calc_stats = read_json_file(calcification_stats_file)
    if calc_stats:
        calcification_volume = calc_stats["calcification_volume"]
    else:
        calcification_volume = 0

    volume_stats["calcification_volume"] = calcification_volume

    label_img_aorta_raw_np = sitk.GetArrayFromImage(label_img_aorta_raw)
    label_img_aorta_lumen_np = sitk.GetArrayFromImage(label_img_aorta_lumen)
    spacing = label_img_aorta_raw.GetSpacing()
    aorta_segm_id = 1

    mask_np_aorta_raw = label_img_aorta_raw_np == aorta_segm_id
    mask_np_aorta_lumen = label_img_aorta_lumen_np == aorta_segm_id
    aorta_volumen_raw = np.sum(mask_np_aorta_raw) * spacing[0] * spacing[1] * spacing[2]
    aorta_lumen_volume = (
        np.sum(mask_np_aorta_lumen) * spacing[0] * spacing[1] * spacing[2]
    )

    volume_stats["original_aorta_volume"] = aorta_volumen_raw
    volume_stats["aorta_lumen"] = aorta_lumen_volume

    ratio = 0
    if aorta_lumen_volume > 0:
        ratio = aorta_volumen_raw / (aorta_lumen_volume + calcification_volume)
        volume_stats["aorta_ratio"] = ratio

    if verbose:
        print(
            f"Found original aorta volume {aorta_volumen_raw:.0f} and lumen volume {aorta_lumen_volume:.0f} ratio {ratio:.2f}"
        )

    aorta_surface_raw = convert_label_map_to_surface(
        segm_in_name_hires, only_largest_component=False
    )
    aorta_surface_lumen = convert_label_map_to_surface(
        segm_lumen_in, only_largest_component=False
    )

    if aorta_surface_raw is None or aorta_surface_lumen is None:
        msg = f"Could not compute surfaces of {segm_in_name_hires} or {segm_lumen_in}.."
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    locator = vtk.vtkPointLocator()
    locator.SetDataSet(aorta_surface_lumen)
    locator.BuildLocator()

    n_points = aorta_surface_raw.GetNumberOfPoints()
    dists = []
    for i in range(n_points):
        point = aorta_surface_raw.GetPoint(i)
        closest_point = locator.FindClosestPoint(point)
        dist = np.linalg.norm(
            np.array(point) - np.array(aorta_surface_lumen.GetPoint(closest_point))
        )
        dists.append(dist)

    q95 = np.quantile(dists, 0.95)
    if verbose:
        print(f"95% quantile of distance to lumen {q95:.2f}")
    volume_stats["q95_distances"] = q95

    try:
        with Path(aneurysm_sac_stats_file).open("wt") as handle:
            json.dump(volume_stats, handle, indent=4, sort_keys=False)
    except IOError as e:
        print(f"I/O error({e.errno}): {e.strerror}: {aneurysm_sac_stats_file}")

    return True


def compute_ventricularoaortic_landmark(
    segm_folder, lm_folder, stats_folder, verbose, quiet, write_log_file, output_folder
):
    """
    The ventricularoaortic landmark is between the end of the aorta and the start of the left ventricle
    """
    overlap_name = f"{segm_folder}aorta_lv_overlap.nii.gz"
    ventricularoaortic_p_out_file = f"{lm_folder}ventricularoaortic_point.txt"
    ventricularoaortic_p_none_out_file = f"{lm_folder}ventricularoaortic_no_point.txt"
    stats_file = f"{stats_folder}/aorta_parts.json"

    segm_name_aorta = f"{segm_folder}aorta_lumen_hires_raw.nii.gz"
    segm_name_hc = f"{segm_folder}heartchambers_highres.nii.gz"
    aorta_segm_id = 1
    left_ventricle_id = 3

    debug = False
    if os.path.exists(ventricularoaortic_p_out_file) or os.path.exists(
        ventricularoaortic_p_none_out_file
    ):
        if verbose:
            print(
                f"Landmark file {ventricularoaortic_p_out_file} already exists - skipping"
            )
        return True

    if verbose:
        print(f"Computing {ventricularoaortic_p_out_file}")

    # Aorta should always be present
    if not os.path.exists(segm_name_aorta):
        msg = f"Aorta segmentation {segm_name_aorta} not found. Can not compute ventricularoaortic landmark."
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    # Heart might not be present and that is fine
    if not os.path.exists(segm_name_hc):
        if verbose:
            print(
                f"Could not find {segm_name_hc} can not compute ventricularoaortic landmark"
            )
        f_p_out = open(ventricularoaortic_p_none_out_file, "w")
        f_p_out.write("no point")
        f_p_out.close()
        return True

    n_aorta_parts = 1
    parts_stats = read_json_file(stats_file)
    if parts_stats:
        n_aorta_parts = parts_stats["aorta_parts"]

    label_img_aorta = read_nifti_with_logging_cached(
        segm_name_aorta, verbose, quiet, write_log_file, output_folder
    )
    if label_img_aorta is None:
        return False

    label_img_lv = read_nifti_with_logging_cached(
        segm_name_hc, verbose, quiet, write_log_file, output_folder
    )
    if label_img_lv is None:
        f_p_out = open(ventricularoaortic_p_none_out_file, "w")
        f_p_out.write("no point")
        f_p_out.close()
        return True
    #
    # try:
    #     label_img_aorta = sitk.ReadImage(segm_name_aorta)
    # except RuntimeError as e:
    #     msg = f"Could not read {segm_name_aorta}: {str(e)} got an exception"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     return False

    # try:
    #     label_img_lv = sitk.ReadImage(segm_name_hc)
    # except RuntimeError as e:
    #     msg = f"Could not read {segm_name_hc}: {str(e)} got an exception"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     f_p_out = open(ventricularoaortic_p_none_out_file, "w")
    #     f_p_out.write("no point")
    #     f_p_out.close()
    #     return True

    label_img_aorta_np = sitk.GetArrayFromImage(label_img_aorta)
    mask_np_aorta = label_img_aorta_np == aorta_segm_id

    # Force it to have only the found components (since we can use the original ts segmentations with spurious parts)
    min_comp_size = 5000
    if verbose:
        print(f"Finding aorta components with min_comp_size: {min_comp_size} voxels")
    components, _ = get_components_over_certain_size(
        mask_np_aorta, min_comp_size, n_aorta_parts
    )
    if components is None:
        msg = f"No aorta lumen found left after connected components {input_file}"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False
    mask_np_aorta = components

    label_img_lv_np = sitk.GetArrayFromImage(label_img_lv)
    spacing = label_img_aorta.GetSpacing()

    mask_np_lv = label_img_lv_np == left_ventricle_id

    if verbose:
        print("Finding overlap between aorta and LV using SDF")
    footprint_radius_mm = 5
    overlap_mask = edt_based_overlap(
        mask_np_aorta,
        mask_np_lv,
        spacing=[spacing[2], spacing[1], spacing[0]],
        radius=footprint_radius_mm,
    )

    min_comp_size = 100
    overlap_mask = get_components_over_certain_size_as_individual_volumes(
        overlap_mask, min_comp_size, 1
    )
    if overlap_mask is None:
        if verbose:
            print("No overlap between aorta and LV found: No ventriculaortic point")
        f_p_out = open(ventricularoaortic_p_none_out_file, "w")
        f_p_out.write("no point")
        f_p_out.close()
        return True

    overlap_mask = overlap_mask[0]

    if np.sum(overlap_mask) < 1:
        if verbose:
            print("No overlap between aorta and LV found: No ventriculaortic point")
        f_p_out = open(ventricularoaortic_p_none_out_file, "w")
        f_p_out.write("no point")
        f_p_out.close()
        return True

    com_np = measurements.center_of_mass(overlap_mask)
    # Do the transpose of the coordinates (SimpleITK vs. numpy)
    com_np = [com_np[2], com_np[1], com_np[0]]
    com_phys = label_img_aorta.TransformIndexToPhysicalPoint(
        [int(com_np[0]), int(com_np[1]), int(com_np[2])]
    )

    f_p_out = open(ventricularoaortic_p_out_file, "w")
    f_p_out.write(f"{com_phys[0]} {com_phys[1]} {com_phys[2]}")
    f_p_out.close()

    if debug:
        img_o = sitk.GetImageFromArray(overlap_mask.astype(int))
        img_o.CopyInformation(label_img_aorta)

        print(f"saving {overlap_name}")
        sitk.WriteImage(img_o, overlap_name)

    return True


def combine_aorta_and_left_ventricle_and_iliac_arteries(
    input_file,
    segm_folder,
    lm_folder,
    stats_folder,
    verbose,
    quiet,
    write_log_file,
    output_folder,
    use_ts_org_segmentations=False,
):
    """
    Combine aorta, left ventricle and iliac arteries since this is beneficial for computing the centerline.
    It is ignored if no LV is present and just the aorta is returned
    """
    ventricularoaortic_p_in_file = f"{lm_folder}ventricularoaortic_point.txt"
    stats_file = f"{stats_folder}/aorta_parts.json"
    aorta_segm_id = 1
    left_ventricle_id = 3
    iliac_left_id = 65
    iliac_right_id = 66

    debug = False
    segm_name_hc = f"{segm_folder}heartchambers_highres.nii.gz"
    total_in_name = f"{segm_folder}total.nii.gz"

    # Heart might not be present and that is fine
    # if not os.path.exists(segm_name_hc):
    #     if verbose:
    #         print(
    #             f"Could not find {segm_name_hc} can not compute combined lv and aorta"
    #         )
    #     return True

    # if not os.path.exists(ventricularoaortic_p_in_file):
    #     if verbose:
    #         print(
    #             f"Could not find {ventricularoaortic_p_in_file} can not compute combined lv and aorta"
    #         )
    #     return True

    n_aorta_parts = 1
    parts_stats = read_json_file(stats_file)
    if parts_stats:
        n_aorta_parts = parts_stats["aorta_parts"]

    if n_aorta_parts == 1:
        if use_ts_org_segmentations:
            segm_name_aorta = f"{segm_folder}aorta_lumen_hires_raw.nii.gz"
            segm_out_name = f"{segm_folder}aorta_lumen_extended_ts_org.nii.gz"
        else:
            segm_name_aorta = f"{segm_folder}aorta_lumen.nii.gz"
            segm_out_name = f"{segm_folder}aorta_lumen_extended.nii.gz"
    elif n_aorta_parts == 2:
        if use_ts_org_segmentations:
            segm_name_aorta = f"{segm_folder}aorta_lumen_annulus_raw.nii.gz"
            segm_out_name = f"{segm_folder}aorta_lumen_extended_ts_org.nii.gz"
        else:
            segm_name_aorta = f"{segm_folder}aorta_lumen_annulus.nii.gz"
            segm_out_name = f"{segm_folder}aorta_lumen_extended.nii.gz"
    else:
        msg = f"Can not handle more than 2 parts. Found {n_aorta_parts}"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    if os.path.exists(segm_out_name):
        if verbose:
            print(f"{segm_out_name} already exists - skipping")
        return True
    if verbose:
        print(f"Computing {segm_out_name}")

    if not os.path.exists(segm_name_aorta):
        msg = f"Aorta segmentation {segm_name_aorta} not found. Can not compute combined aorta and left ventricle segmentation."
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    label_img_aorta = read_nifti_with_logging_cached(
        segm_name_aorta, verbose, quiet, write_log_file, output_folder
    )
    if label_img_aorta is None:
        return False

    ct_img = read_nifti_with_logging_cached(
        input_file, verbose, quiet, write_log_file, output_folder
    )
    if ct_img is None:
        return False

    label_img_lv = read_nifti_with_logging_cached(
        segm_name_hc, verbose, quiet, write_log_file, output_folder
    )
    # if label_img_lv is None:
    #     return False

    label_img_total = read_nifti_with_logging_cached(
        total_in_name, verbose, quiet, write_log_file, output_folder
    )
    if label_img_total is None:
        return False

    label_img_aorta_np = sitk.GetArrayFromImage(label_img_aorta)
    label_img_total_np = sitk.GetArrayFromImage(label_img_total)
    mask_np_aorta = label_img_aorta_np == aorta_segm_id
    mask_np_iliac_left = label_img_total_np == iliac_left_id
    mask_np_iliac_right = label_img_total_np == iliac_right_id
    combined_mask = np.bitwise_or(mask_np_aorta, mask_np_iliac_left)
    combined_mask = np.bitwise_or(combined_mask, mask_np_iliac_right)

    if label_img_lv:
        label_img_lv_np = sitk.GetArrayFromImage(label_img_lv)
        mask_np_lv = label_img_lv_np == left_ventricle_id
        combined_mask = np.bitwise_or(combined_mask, mask_np_lv)

    spc = label_img_aorta.GetSpacing()
    # Make the spacing fit the numpy array
    spacing = [spc[2], spc[1], spc[0]]
    footprint_radius_mm = 5
    if verbose:
        print("EDF based closing and other EDF based operations")
    closed_mask = edt_based_closing(combined_mask, spacing, footprint_radius_mm)
    large_components, _ = get_components_over_certain_size(
        closed_mask, 5000, n_aorta_parts
    )
    large_components, _ = close_cavities_in_segmentations(large_components)

    ct_np = sitk.GetArrayFromImage(ct_img)
    # Remove invalid out-of-scan voxels (typically values -2048)
    large_components = (-200 < ct_np) & large_components

    img_o = sitk.GetImageFromArray(large_components.astype(int))
    img_o.CopyInformation(label_img_aorta)

    if debug:
        print(f"saving {segm_out_name}")
    sitk.WriteImage(img_o, segm_out_name)

    return True


def compute_aortic_arch_landmarks(
    segm_folder, lm_folder, verbose, quiet, write_log_file, output_folder
):
    """
    Find the aortic arch landmarks by looking at the overlap between the aorta and the
    brachiocephalic trunk, left common carotid artery and left subclavian artery
    """
    debug = False
    segm_name_aorta = f"{segm_folder}aorta_lumen_hires_raw.nii.gz"
    segm_name_total = f"{segm_folder}total.nii.gz"
    aorta_segm_id = 1

    lm_names = [
        "brachiocephalic_trunc",
        "common_carotid_artery_left",
        "subclavian_artery_left",
    ]
    segm_ids = [54, 58, 56]

    label_img_aorta = read_nifti_with_logging_cached(
        segm_name_aorta, verbose, quiet, write_log_file, output_folder
    )
    if label_img_aorta is None:
        return False
    label_img_total = read_nifti_with_logging_cached(
        segm_name_total, verbose, quiet, write_log_file, output_folder
    )
    if label_img_total is None:
        return False

    # try:
    #     label_img_aorta = sitk.ReadImage(segm_name_aorta)
    # except RuntimeError as e:
    #     msg = f"Could not read {segm_name_aorta}: {str(e)} got an exception"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     return False

    label_img_aorta_np = sitk.GetArrayFromImage(label_img_aorta)
    mask_np_aorta = label_img_aorta_np == aorta_segm_id

    # try:
    #     label_img = sitk.ReadImage(segm_name_total)
    # except RuntimeError as e:
    #     msg = f"Could not read {segm_name_total}: {str(e)} got an exception"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     return False
    label_img_np = sitk.GetArrayFromImage(label_img_total)

    for idx in range(len(lm_names)):
        lm_name = lm_names[idx]
        segm_id = segm_ids[idx]
        lm_out_name = f"{lm_folder}{lm_name}.txt"
        lm_out_name_none = f"{lm_folder}{lm_name}_done.txt"
        segm_oname = f"{segm_folder}{lm_name}_aorta_overlap.nii.gz"

        if os.path.exists(lm_out_name) or os.path.exists(lm_out_name_none):
            if verbose:
                print(f"Landmark file {lm_out_name} already exists - skipping")
            continue
        if verbose:
            print(f"Computing {lm_out_name}")

        radius = 5
        current_segm = label_img_np == segm_id
        if np.sum(current_segm) > 10:
            if verbose:
                print(f"EDT based overlap for {lm_out_name}")
            if not edt_based_compute_landmark_from_segmentation_overlap(
                mask_np_aorta,
                current_segm,
                radius,
                label_img_total,
                segm_oname,
                lm_out_name,
                only_larges_components=True,
                debug=debug,
            ):
                f_p_out = open(lm_out_name_none, "w")
                f_p_out.write("no point")
                f_p_out.close()
                if verbose:
                    print(f"Could not compute landmark {lm_out_name}")
        else:
            f_p_out = open(lm_out_name_none, "w")
            f_p_out.write("no point")
            f_p_out.close()
            if debug:
                print(f"Found no {lm_name}")

    return True


def compute_diaphragm_landmarks_from_surfaces(
    segm_folder, lm_folder, verbose, quiet, write_log_file, output_folder
):
    """
    Compute the diaphragm landmark by finding the highest point of the liver surface
    and the lowest point of the right ventricle surface
    The landmark is then between these two points
    If no liver or no right ventricle is found, no landmark is computed
    If the liver and right ventricle overlap too much, something is wrong with the segmentation
    and no landmark is computed.
    """
    segm_name_hc = f"{segm_folder}heartchambers_highres.nii.gz"
    segm_name_total = f"{segm_folder}total.nii.gz"
    segm_id_heart = 51
    segm_id_liver = 5
    segm_id_rv = 5

    # Square millimeters - one square centimeter
    volume_threshold = 1000
    # debug = False

    lm_check_name = f"{lm_folder}diaphragm.txt"
    lm_check_name_none = f"{lm_folder}diaphragm_done.txt"
    if os.path.exists(lm_check_name) or os.path.exists(lm_check_name_none):
        if verbose:
            print(f"Landmark file {lm_check_name} already exists - skipping")
        return True

    if verbose:
        print(f"Computing {lm_check_name}")

    # Heart might not be present and that is fine
    if not os.path.exists(segm_name_hc):
        if verbose:
            print(f"Could not find {segm_name_hc} can not compute diagphragm landmark")
        return True

    label_img_total = read_nifti_with_logging_cached(
        segm_name_total, verbose, quiet, write_log_file, output_folder
    )
    if label_img_total is None:
        return False

    label_img_rv = read_nifti_with_logging_cached(
        segm_name_hc, verbose, quiet, write_log_file, output_folder
    )
    if label_img_rv is None:
        return False
    #
    # try:
    #     label_img_total = sitk.ReadImage(segm_name_total)
    # except RuntimeError as e:
    #     msg = f"Could not read {segm_name_total}: {str(e)} got an exception"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     return False

    # try:
    #     label_img_rv = sitk.ReadImage(segm_name_hc)
    # except RuntimeError as e:
    #     msg = f"Could not read {segm_name_hc}: {str(e)} got an exception"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     return False

    label_img_rv_np = sitk.GetArrayFromImage(label_img_rv)
    label_img_total_np = sitk.GetArrayFromImage(label_img_total)
    mask_np_heart = label_img_total_np == segm_id_heart
    mask_np_liver = label_img_total_np == segm_id_liver
    mask_np_rv = label_img_rv_np == segm_id_rv

    n_heart = np.sum(mask_np_heart)
    n_liver = np.sum(mask_np_liver)
    n_rv = np.sum(mask_np_rv)

    if (
        n_heart < volume_threshold
        or n_liver < volume_threshold
        or n_rv < volume_threshold
    ):
        if verbose:
            print("Heart, liver or RV too small or not present - no diaphragm landmark")
        f_p_out = open(lm_check_name_none, "w")
        f_p_out.write("no point")
        f_p_out.close()
        return True

    # Check if there is an overlap between liver and rv...if there is something is wrong
    overlap = np.bitwise_and(mask_np_liver, mask_np_rv)
    overlap_size = np.sum(overlap)
    if n_rv > 0:
        overlap_percent = overlap_size / n_rv * 100
    else:
        overlap_percent = 0

    if overlap_percent > 20:
        msg = f"Overlap percent {overlap_percent} found between liver and RV. Something wrong in segmentation. No {lm_check_name}"
        if verbose:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="warning"
            )
        f_p_out = open(lm_check_name_none, "w")
        f_p_out.write("no point")
        f_p_out.close()
        return True

    rv_surface = convert_label_map_to_surface(
        segm_name_hc, segment_id=segm_id_rv, only_largest_component=True
    )
    liver_surface = convert_label_map_to_surface(
        segm_name_total, segment_id=segm_id_liver, only_largest_component=True
    )

    if rv_surface is None or liver_surface is None:
        if verbose:
            print(
                f"Could not compute surfaces of {segm_name_hc} or {segm_name_total}. No diaphragm landmark"
            )
        f_p_out = open(lm_check_name_none, "w")
        f_p_out.write("no point")
        f_p_out.close()
        return True

    min_rv_p, _ = compute_min_and_max_z_landmark(rv_surface)
    if min_rv_p is None:
        if verbose:
            print(f"Could not compute min z of {segm_name_hc}. No diaphragm landmark")
        f_p_out = open(lm_check_name_none, "w")
        f_p_out.write("no point")
        f_p_out.close()
        return True

    idx_1, idx_2, avg_p, dist = find_closests_points_on_two_surfaces_with_start_point(
        rv_surface, liver_surface, min_rv_p
    )

    f_p_out = open(lm_check_name, "w")
    f_p_out.write(f"{avg_p[0]} {avg_p[1]} {avg_p[2]}")
    f_p_out.close()

    return True


def compute_aorta_scan_type(
    input_file,
    segm_folder,
    lm_folder,
    stats_folder,
    verbose,
    quiet,
    write_log_file,
    output_folder,
):
    """
    Compute aorta scan type.
    Is it a full aorta, only the aorta at the cardiac section or in the abdomen?
    We use the case types defined here:
    https://github.com/RasmusRPaulsen/AortaExplorer
    """
    stats_file = f"{stats_folder}/aorta_scan_type.json"

    if os.path.exists(stats_file):
        if verbose:
            print(f"{stats_file} already exists - not recomputing")
        return True

    if verbose:
        print(f"Computing {stats_file}")

    n_aorta_parts = 1
    parts_stats = read_json_file(f"{stats_folder}aorta_parts.json")
    if parts_stats:
        n_aorta_parts = parts_stats["aorta_parts"]

    if n_aorta_parts == 1:
        segm_name = f"{segm_folder}aorta_lumen.nii.gz"
        aorta_segment_id = 1
    else:
        segm_name = f"{segm_folder}aorta_lumen_descending.nii.gz"
        aorta_segment_id = 1

    stats = {}
    stats["scan_name"] = input_file
    stats["parts"] = n_aorta_parts

    # Check for iliac arteries
    segm_il_left = f"{segm_folder}iliac_artery_left_top.nii.gz"
    segm_il_right = f"{segm_folder}iliac_artery_right_top.nii.gz"
    stats["iliac_left"] = il_left = os.path.exists(segm_il_left)
    stats["iliac_right"] = il_right = os.path.exists(segm_il_right)

    # Check for top of aortic arch
    # We need all three top landmarks
    lm_check_names = [
        "brachiocephalic_trunc.txt",
        "common_carotid_artery_left.txt",
        "subclavian_artery_left.txt",
    ]
    aortic_arch_present = True
    for cn in lm_check_names:
        if not os.path.exists(f"{lm_folder}{cn}"):
            aortic_arch_present = False
            break
    stats["aortic_arch_present"] = aortic_arch_present

    # Check for top of annulus
    lm_check_name = f"{lm_folder}ventricularoaortic_point.txt"
    stats["lvot_present"] = lvot_present = os.path.exists(lm_check_name)

    segm_data, _, _ = read_nifti_itk_to_numpy(segm_name)
    if segm_data is None:
        msg = f"Could not read {segm_name} can not compute scan type"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False
    # sides = set()

    # TODO: Use the out-of-scan SDF to determine if the aorta is touching the side of the scan
    sides = check_if_segmentation_hit_sides_of_scan(segm_data, aorta_segment_id)
    stats["sides"] = list(sides)

    scan_type = "unknown"
    scan_type_desc = "unknown"
    if (
        len(sides) == 0
        and aortic_arch_present
        and lvot_present
        and il_left
        and il_right
        and n_aorta_parts == 1
    ):
        scan_type = "1"
        scan_type_desc = "full aorta"
    elif (
        len(sides) == 0
        and not aortic_arch_present
        and lvot_present
        and il_left
        and il_right
        and n_aorta_parts == 1
    ):
        scan_type = "1b"
        scan_type_desc = "full aorta but missing top of aortic arch"
    elif (
        len(sides) == 1
        and "up" in sides
        and not aortic_arch_present
        and lvot_present
        and il_left
        and il_right
        and n_aorta_parts == 1
    ):
        scan_type = "1d"
        scan_type_desc = "full aorta but missing top of aortic arch and it is touching the upper side of the scan"
    elif (
        len(sides) == 1
        and "up" in sides
        and lvot_present
        and il_left
        and il_right
        and n_aorta_parts == 1
    ):
        scan_type = "1c"
        scan_type_desc = "full aorta but it is touching the upper side of the scan"
    elif (
        len(sides) == 1
        and "up" in sides
        and not aortic_arch_present
        and not lvot_present
        and il_left
        and il_right
        and n_aorta_parts == 1
    ):
        scan_type = "2"
        scan_type_desc = "Lower aorta with iliac bifurcation"
    elif (
        len(sides) == 1
        and "down" in sides
        and aortic_arch_present
        and lvot_present
        and not il_left
        and not il_right
        and n_aorta_parts == 1
    ):
        scan_type = "3"
        scan_type_desc = "Upper aorta with aortic arch and LVOT (cardiac scan)"
    elif (
        len(sides) == 2
        and "up" in sides
        and "down" in sides
        and not aortic_arch_present
        and lvot_present
        and not il_left
        and not il_right
        and n_aorta_parts == 1
    ):
        scan_type = "3b"
        scan_type_desc = "Upper aorta with partial aortic arch and LVOT (cardiac scan)"
    elif (
        len(sides) == 1
        and "up" in sides
        and not aortic_arch_present
        and lvot_present
        and il_left
        and il_right
        and n_aorta_parts == 2
    ):
        scan_type = "4"
        scan_type_desc = "Two aorta parts and LVOT (cardiac) and no aortic arch and iliac bifurcation"
    elif (
        len(sides) == 2
        and "up" in sides
        and "down" in sides
        and not aortic_arch_present
        and lvot_present
        and not il_left
        and not il_right
        and n_aorta_parts == 2
    ):
        scan_type = "5"
        scan_type_desc = "Two aorta parts and LVOT (cardiac) and no aortic arch"
    elif (
        len(sides) > 2
        and "up" in sides
        and "down" in sides
        and not aortic_arch_present
        and lvot_present
        and not il_left
        and not il_right
        and n_aorta_parts == 2
    ):
        scan_type = "5b"
        scan_type_desc = "Two aorta parts, LVOT (cardiac), no aortic arch and aorta hits multiple sides and might be cropped."
    elif (
        len(sides) == 1
        and "up" in sides
        and not aortic_arch_present
        and lvot_present
        and not il_left
        and not il_right
        and n_aorta_parts == 1
    ):
        scan_type = "6"
        scan_type_desc = "Only cardiac aorta and LVOT (cardiac) and no aortic arch and no iliac bifurcation"
    elif (
        len(sides) == 2
        and "up" in sides
        and "down" in sides
        and not aortic_arch_present
        and not lvot_present
        and not il_left
        and not il_right
        and n_aorta_parts == 1
    ):
        scan_type = "7 or 8b"
        scan_type_desc = "Only middle part of aorta or only cropped aortic arch"
    elif (
        len(sides) == 1
        and "down" in sides
        and aortic_arch_present
        and not lvot_present
        and not il_left
        and not il_right
        and n_aorta_parts == 1
    ):
        scan_type = "8"
        scan_type_desc = "Only aortic arch"
    # Sometimes the scan does not extend to the side of the volume and the side check is not reliable
    elif (
        not aortic_arch_present
        and lvot_present
        and not il_left
        and not il_right
        and n_aorta_parts == 2
    ):
        scan_type = "5"
        scan_type_desc = (
            "Two aorta parts and LVOT (cardiac scan) and no aortic arch (best guess)"
        )
    elif (
        not aortic_arch_present
        and lvot_present
        and il_left
        and il_right
        and n_aorta_parts == 2
    ):
        scan_type = "4"
        scan_type_desc = "Two aorta parts and LVOT (cardiac scan) and no aortic arch and iliac bifurcation (best guess)"
    elif (
        not aortic_arch_present
        and not lvot_present
        and il_left
        and il_right
        and n_aorta_parts == 1
    ):
        scan_type = "2"
        scan_type_desc = "Lower aorta with iliac bifurcation (best guess)"
    else:
        msg = (
            f"Could not determine scan type. sides: {list(sides)}, aortic_arch_present: {aortic_arch_present}, "
            f"lvot_present: {lvot_present}, il_left: {il_left}, il_right: {il_right}, n_aorta_parts: {n_aorta_parts}"
            f" for {input_file}"
        )
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="warning"
            )

    if verbose:
        print(f"Scan type: {scan_type}: {scan_type_desc} for {input_file}")
    stats["scan_type"] = scan_type
    stats["scan_type_desc"] = scan_type_desc
    try:
        with Path(stats_file).open("wt") as handle:
            json.dump(stats, handle, indent=4, sort_keys=False)
    except IOError as e:
        print(f"I/O error({e.errno}): {e.strerror}: {stats_file}")

    return True


def compute_aorta_iliac_artery_landmark(
    segm_folder, quiet, write_log_file, output_folder, use_ts_org_segmentations=True
):
    """
    Compute the aorta-iliac artery landmark by finding the center of the top of the iliac arteries
    and then finding the closest point on the aorta surface to the midpoint between the two iliac arteries
    """
    segm_l_name = f"{segm_folder}iliac_artery_left_top.nii.gz"
    segm_r_name = f"{segm_folder}iliac_artery_right_top.nii.gz"
    segm_aorta_name = f"{segm_folder}aorta_lumen.nii.gz"
    if use_ts_org_segmentations:
        segm_aorta_name = f"{segm_folder}aorta_lumen_hires_raw.nii.gz"

    label_img_l = read_nifti_with_logging_cached(
        segm_l_name, False, quiet, write_log_file, output_folder
    )
    if label_img_l is None:
        return None
    label_img_r = read_nifti_with_logging_cached(
        segm_r_name, False, quiet, write_log_file, output_folder
    )
    if label_img_r is None:
        return None

    # try:
    #     label_img_l = sitk.ReadImage(segm_l_name)
    # except RuntimeError as e:
    #     msg = f"Could not read {segm_l_name}: {str(e)} got an exception"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     return None

    label_img_l_np = sitk.GetArrayFromImage(label_img_l)

    # try:
    #     label_img_r = sitk.ReadImage(segm_r_name)
    # except RuntimeError as e:
    #     msg = f"Could not read {segm_r_name}: {str(e)} got an exception"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     return None
    label_img_r_np = sitk.GetArrayFromImage(label_img_r)

    # try:
    #     label_img_aorta = sitk.ReadImage(segm_aorta_name)
    # except RuntimeError as e:
    #     msg = f"Could not read {segm_aorta_name}: {str(e)} got an exception"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     return None

    aorta_surface = convert_label_map_to_surface(
        segm_aorta_name, segment_id=1, only_largest_component=True
    )
    if aorta_surface is None:
        msg = f"Could not compute aorta surface from {segm_aorta_name}"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return None

    com_np = measurements.center_of_mass(label_img_l_np)
    # Do the transpose of the coordinates (SimpleITK vs. numpy)
    com_np = [com_np[2], com_np[1], com_np[0]]
    com_phys_1 = label_img_l.TransformIndexToPhysicalPoint(
        [int(com_np[0]), int(com_np[1]), int(com_np[2])]
    )

    com_np = measurements.center_of_mass(label_img_r_np)
    # Do the transpose of the coordinates (SimpleITK vs. numpy)
    com_np = [com_np[2], com_np[1], com_np[0]]
    com_phys_2 = label_img_l.TransformIndexToPhysicalPoint(
        [int(com_np[0]), int(com_np[1]), int(com_np[2])]
    )

    mid_point = [
        (com_phys_1[0] + com_phys_2[0]) / 2,
        (com_phys_1[1] + com_phys_2[1]) / 2,
        (com_phys_1[2] + com_phys_2[2]) / 2,
    ]

    locator = vtk.vtkPointLocator()
    locator.SetDataSet(aorta_surface)
    locator.BuildLocator()
    idx_min = locator.FindClosestPoint(mid_point)

    start_p = aorta_surface.GetPoint(idx_min)

    return start_p


def compute_centerline_landmarks_for_aorta_type_2(
    segm_folder,
    lm_folder,
    verbose,
    quiet,
    write_log_file,
    output_folder,
    use_ts_org_segmentations=True,
):
    """
    Type 2 is a single aorta with iliac arteries and cut at the top
    The start landmark is at the aorta-iliac artery bifurcation
    The end landmark is at the highest point of the aorta (where it exits the scan)
    """
    start_p_out_file = f"{lm_folder}aorta_start_point.txt"
    end_p_out_file = f"{lm_folder}aorta_end_point.txt"
    sdf_name = f"{segm_folder}out_of_scan_sdf.nii.gz"
    aorta_name = f"{segm_folder}aorta_lumen.nii.gz"
    if use_ts_org_segmentations:
        aorta_name = f"{segm_folder}aorta_lumen_hires_raw.nii.gz"
    overlap_name_1 = f"{segm_folder}aorta_side_region.nii.gz"
    debug = False

    if os.path.exists(start_p_out_file) and os.path.exists(end_p_out_file):
        if verbose:
            print(f"{start_p_out_file} and {end_p_out_file} already exists - skipping")
        return True

    if verbose:
        print(f"Computing {start_p_out_file} and {end_p_out_file}")

    if not os.path.exists(sdf_name):
        msg = f"Out-of-scan SDF {sdf_name} not found. Can not compute centerline landmarks for type 2"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    start_p = compute_aorta_iliac_artery_landmark(
        segm_folder, quiet, write_log_file, output_folder, use_ts_org_segmentations
    )
    if start_p is None:
        msg = "Could not compute aorta-iliac artery landmark. Can not compute centerline landmarks for type 2"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    f_p_out = open(start_p_out_file, "w")
    f_p_out.write(f"{start_p[0]} {start_p[1]} {start_p[2]}")
    f_p_out.close()

    sdf_img = read_nifti_with_logging_cached(
        sdf_name, False, quiet, write_log_file, output_folder
    )
    if sdf_img is None:
        return False
    #
    # try:
    #     sdf_img = sitk.ReadImage(sdf_name)
    # except RuntimeError as e:
    #     msg = f"Could not read {sdf_name}: {str(e)} got an exception"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     return False

    sdf_np = sitk.GetArrayFromImage(sdf_img)
    label_img = read_nifti_with_logging_cached(
        aorta_name, False, quiet, write_log_file, output_folder
    )
    if label_img is None:
        return False
    #
    #
    # try:
    #     label_img = sitk.ReadImage(aorta_name)
    # except RuntimeError as e:
    #     msg = f"Could not read {aorta_name}: {str(e)} got an exception"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     return False

    label_img_np = sitk.GetArrayFromImage(label_img)

    # Force it to one component
    min_comp_size = 5000
    if verbose:
        print(f"Finding aorta components with min_comp_size: {min_comp_size} voxels")
    components, _ = get_components_over_certain_size(label_img_np, min_comp_size, 1)
    if components is None:
        msg = f"No aorta lumen found left after connected components {input_file}"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False
    label_img_np = components

    spacing = label_img.GetSpacing()
    max_space = np.max(np.asarray(spacing))
    # overlap_dist = 3.0
    overlap_dist = max(3 * max_space, 3)

    # overlap_dist = 3.0
    # Find the part of the aorta that hits the side of the scan
    overlap_region = label_img_np & (sdf_np < overlap_dist)
    if np.sum(overlap_region) == 0:
        msg = f"No part of the aorta touch the side of the scan - and it should for type 2. For {aorta_name}"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    if debug:
        img_o = sitk.GetImageFromArray(overlap_region.astype(int))
        img_o.CopyInformation(label_img)

        print(f"Debug: saving {overlap_name_1}")
        sitk.WriteImage(img_o, overlap_name_1)

    regions = get_components_over_certain_size_as_individual_volumes(
        overlap_region, 50, 1
    )
    if regions is None:
        msg = f"Could not separate the aorta touching the side of the scan into components - and it should for type 2. For {aorta_name}"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    if len(regions) < 1:
        msg = f"Did not find that aorta touch the side of the scan - and it should for type 2. For {aorta_name}"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    com_np = measurements.center_of_mass(regions[0])
    # Do the transpose of the coordinates (SimpleITK vs. numpy)
    com_np = [com_np[2], com_np[1], com_np[0]]
    com_phys_1 = label_img.TransformIndexToPhysicalPoint(
        [int(com_np[0]), int(com_np[1]), int(com_np[2])]
    )

    # Check if the landmark is actually at the top of the scan
    size = label_img.GetSize()
    if com_np[2] < size[2] / 4:
        msg = f"The found landmark is not at the top of the scan - and it should for type 2. For {aorta_name}"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    f_p_out = open(end_p_out_file, "w")
    f_p_out.write(f"{com_phys_1[0]} {com_phys_1[1]} {com_phys_1[2]}")
    f_p_out.close()

    return True


def compute_centerline_landmarks_for_aorta_type_1(
    segm_folder,
    lm_folder,
    verbose,
    quiet,
    write_log_file,
    output_folder,
    use_ts_org_segmentations=True,
):
    """
    Type 1 is an entire aorta - can sometime touch the top of the scan
    """
    start_p_out_file = f"{lm_folder}aorta_start_point.txt"
    end_p_out_file = f"{lm_folder}aorta_end_point.txt"
    aorta_name = f"{segm_folder}aorta_lumen_extended.nii.gz"
    if use_ts_org_segmentations:
        aorta_name = f"{segm_folder}aorta_lumen_extended_ts_org.nii.gz"

    # debug = False
    if os.path.exists(start_p_out_file) and os.path.exists(end_p_out_file):
        if verbose:
            print(f"{start_p_out_file} and {end_p_out_file} already exists - skipping")
        return True

    if verbose:
        print(f"Computing {start_p_out_file} and {end_p_out_file}")

    start_p = compute_aorta_iliac_artery_landmark(
        segm_folder, quiet, write_log_file, output_folder, use_ts_org_segmentations
    )
    if start_p is None:
        msg = "Could not compute aorta-iliac artery landmark. Can not compute centerline landmarks for type 1"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    f_p_out = open(start_p_out_file, "w")
    f_p_out.write(f"{start_p[0]} {start_p[1]} {start_p[2]}")
    f_p_out.close()

    aorta_surf = convert_label_map_to_surface(
        aorta_name, segment_id=1, only_largest_component=True
    )
    if aorta_surf is None:
        msg = f"Could not compute aorta surface from {aorta_name}"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    locator = vtk.vtkPointLocator()
    locator.SetDataSet(aorta_surf)
    locator.BuildLocator()
    idx_min = locator.FindClosestPoint(start_p)

    if verbose:
        print("Dijkstra on aorta+LV surface to find highest point")
    dijkstra = vtk.vtkDijkstraGraphGeodesicPath()
    dijkstra.SetInputData(aorta_surf)
    dijkstra.SetStartVertex(idx_min)
    dijkstra.Update()
    weights = vtk.vtkDoubleArray()
    dijkstra.GetCumulativeWeights(weights)
    aorta_surf.GetPointData().SetScalars(weights)

    w_temp = vtk_to_numpy(weights)
    idx_max = np.argmax(w_temp)
    max_p = aorta_surf.GetPoint(idx_max)
    end_p = max_p

    end_p_out = open(end_p_out_file, "w")
    end_p_out.write(f"{end_p[0]} {end_p[1]} {end_p[2]}")
    end_p_out.close()

    return True


def compute_centerline_landmarks_for_aorta_type_5_annulus(
    segm_folder,
    lm_folder,
    verbose,
    quiet,
    write_log_file,
    output_folder,
    use_ts_org_segmentations=True,
):
    """
    Type 5 is a two part aorta with an annulus and a descending part.
    For the annulus part:
    The start landmark is at the center of the part of the aorta that touches the top side of the scan
    The end landmark is at point that is geodistically the furthest away on the aorta+LV surface
    """
    start_p_out_file = f"{lm_folder}aorta_start_point_annulus.txt"
    end_p_out_file = f"{lm_folder}aorta_end_point_annulus.txt"
    sdf_name = f"{segm_folder}out_of_scan_sdf.nii.gz"
    aorta_lv_name = f"{segm_folder}aorta_lumen_extended.nii.gz"
    overlap_name_1 = f"{segm_folder}aorta_lumen_extended_side_region.nii.gz"
    if use_ts_org_segmentations:
        aorta_lv_name = f"{segm_folder}aorta_lumen_extended_ts_org.nii.gz"

    debug = False

    if os.path.exists(start_p_out_file) and os.path.exists(end_p_out_file):
        if verbose:
            print(f"{start_p_out_file} and {end_p_out_file} already exists - skipping")
        return True

    if verbose:
        print(f"Computing {start_p_out_file} and {end_p_out_file}")

    if not os.path.exists(sdf_name):
        msg = f"Out-of-scan SDF {sdf_name} not found. Can not compute centerline landmarks for type 2"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    sdf_img = read_nifti_with_logging_cached(
        sdf_name, False, quiet, write_log_file, output_folder
    )
    if sdf_img is None:
        return False
    #
    # try:
    #     sdf_img = sitk.ReadImage(sdf_name)
    # except RuntimeError as e:
    #     msg = f"Could not read {sdf_name}: {str(e)} got an exception"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     return False

    sdf_np = sitk.GetArrayFromImage(sdf_img)

    label_img = read_nifti_with_logging_cached(
        aorta_lv_name, False, quiet, write_log_file, output_folder
    )
    if label_img is None:
        return False
    #
    # # Start by the annulus part (aorta+LV)
    # try:
    #     label_img = sitk.ReadImage(aorta_lv_name)
    # except RuntimeError as e:
    #     msg = f"Could not read {aorta_lv_name}: {str(e)} got an exception"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     return False

    label_img_np = sitk.GetArrayFromImage(label_img)
    spacing = label_img.GetSpacing()
    max_space = np.max(np.asarray(spacing))
    # overlap_dist = 3.0
    overlap_dist = max(3, 3 * max_space)

    # Find the part of the aorta that hits the side of the scan
    overlap_region = label_img_np & (sdf_np < overlap_dist)
    if np.sum(overlap_region) == 0:
        msg = f"No part of the ascending aorta touch the side of the scan - and it should for type 5. Can not compute start {start_p_out_file}"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    if debug:
        img_o = sitk.GetImageFromArray(overlap_region.astype(int))
        img_o.CopyInformation(label_img)

        print(f"Debug: saving {overlap_name_1}")
        sitk.WriteImage(img_o, overlap_name_1)

    overlap_region, _ = get_components_over_certain_size(overlap_region, 20, 1)
    if overlap_region is None:
        msg = f"Aorta region that hits the side of scan is too small for type 5. For {aorta_lv_name}"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    com_np = measurements.center_of_mass(overlap_region)

    # Do the transpose of the coordinates (SimpleITK vs. numpy)
    com_np = [com_np[2], com_np[1], com_np[0]]
    com_phys = label_img.TransformIndexToPhysicalPoint(
        [int(com_np[0]), int(com_np[1]), int(com_np[2])]
    )

    f_p_out = open(start_p_out_file, "w")
    f_p_out.write(f"{com_phys[0]} {com_phys[1]} {com_phys[2]}")
    f_p_out.close()

    start_p = com_phys

    aorta_surf = convert_label_map_to_surface(
        aorta_lv_name, segment_id=1, only_largest_component=True
    )
    if aorta_surf is None:
        msg = f"Could not compute aorta surface from {aorta_lv_name}"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    locator = vtk.vtkPointLocator()
    locator.SetDataSet(aorta_surf)
    locator.BuildLocator()
    idx_min = locator.FindClosestPoint(start_p)

    if verbose:
        print("Dijkstra on aorta+LV surface to find farthest point")
    dijkstra = vtk.vtkDijkstraGraphGeodesicPath()
    dijkstra.SetInputData(aorta_surf)
    dijkstra.SetStartVertex(idx_min)
    dijkstra.Update()
    weights = vtk.vtkDoubleArray()
    dijkstra.GetCumulativeWeights(weights)
    aorta_surf.GetPointData().SetScalars(weights)

    w_temp = vtk_to_numpy(weights)
    idx_max = np.argmax(w_temp)
    max_p = aorta_surf.GetPoint(idx_max)
    end_p = max_p

    end_p_out = open(end_p_out_file, "w")
    end_p_out.write(f"{end_p[0]} {end_p[1]} {end_p[2]}")
    end_p_out.close()

    return True


def compute_centerline_landmarks_for_aorta_type_5_descending(
    segm_folder,
    lm_folder,
    verbose,
    quiet,
    write_log_file,
    output_folder,
    use_ts_org_segmentations=True,
):
    """
    Type 5 is a two part aorta with an annulus and a descending part
    For the descending part:
    The start landmark is at the center of the part of the descending aorta that touches the

    """
    start_p_out_file = f"{lm_folder}aorta_start_point_descending.txt"
    end_p_out_file = f"{lm_folder}aorta_end_point_descending.txt"
    sdf_name = f"{segm_folder}out_of_scan_sdf.nii.gz"
    aorta_name = f"{segm_folder}aorta_lumen_descending.nii.gz"
    if use_ts_org_segmentations:
        aorta_name = f"{segm_folder}aorta_lumen_descending_raw.nii.gz"

    overlap_name_1 = f"{segm_folder}aorta_descending_side_regions.nii.gz"
    debug = False

    if os.path.exists(start_p_out_file) and os.path.exists(end_p_out_file):
        if verbose:
            print(f"{start_p_out_file} and {end_p_out_file} already exists - skipping")
        return True

    if verbose:
        print(f"Computing {start_p_out_file} and {end_p_out_file}")

    if not os.path.exists(sdf_name):
        msg = f"Out-of-scan SDF {sdf_name} not found. Can not compute centerline landmarks for type 2"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    sdf_img = read_nifti_with_logging_cached(
        sdf_name, False, quiet, write_log_file, output_folder
    )
    if sdf_img is None:
        return False
    #
    # try:
    #     sdf_img = sitk.ReadImage(sdf_name)
    # except RuntimeError as e:
    #     msg = f"Could not read {sdf_name}: {str(e)} got an exception"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     return False

    sdf_np = sitk.GetArrayFromImage(sdf_img)
    label_img = read_nifti_with_logging_cached(
        aorta_name, verbose, quiet, write_log_file, output_folder
    )
    if label_img is None:
        return False
    #
    #
    # try:
    #     label_img = sitk.ReadImage(aorta_name)
    # except RuntimeError as e:
    #     msg = f"Could not read {aorta_name}: {str(e)} got an exception"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     return False

    label_img_np = sitk.GetArrayFromImage(label_img)
    spacing = label_img.GetSpacing()
    max_space = np.max(np.asarray(spacing))
    # overlap_dist = 3.0
    overlap_dist = max(3, 3 * max_space)
    # overlap_dist = 3.0

    # Find the part of the aorta that hits the side of the scan
    overlap_region = label_img_np & (sdf_np < overlap_dist)
    if np.sum(overlap_region) == 0:
        msg = f"No part of the descending aorta touch the side of the scan - and it should for type 5. Can not compute start {start_p_out_file}"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    if debug:
        img_o = sitk.GetImageFromArray(overlap_region.astype(int))
        img_o.CopyInformation(label_img)

        print(f"Debug: saving {overlap_name_1}")
        sitk.WriteImage(img_o, overlap_name_1)

    regions = get_components_over_certain_size_as_individual_volumes(
        overlap_region, 20, 2
    )
    if regions is None:
        msg = f"Aorta region that hits the side of scan is too small for type 5. For {aorta_name}"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    if len(regions) < 2:
        msg = f"Did not find two parts of the descending aorta touch the side of the scan - and it should for type 5. For {aorta_name}"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    com_np = measurements.center_of_mass(regions[0])
    # Do the transpose of the coordinates (SimpleITK vs. numpy)
    com_np = [com_np[2], com_np[1], com_np[0]]
    com_phys_1 = label_img.TransformIndexToPhysicalPoint(
        [int(com_np[0]), int(com_np[1]), int(com_np[2])]
    )

    com_np = measurements.center_of_mass(regions[1])
    # Do the transpose of the coordinates (SimpleITK vs. numpy)
    com_np = [com_np[2], com_np[1], com_np[0]]
    com_phys_2 = label_img.TransformIndexToPhysicalPoint(
        [int(com_np[0]), int(com_np[1]), int(com_np[2])]
    )

    # Compare which is the two points is the uppermost
    p_1_name = start_p_out_file
    p_2_name = end_p_out_file
    if com_phys_1[2] > com_phys_2[2]:
        p_1_name = end_p_out_file
        p_2_name = start_p_out_file

    # TODO: Check that the landmark is actually at the top of the scan and bottom of the scan
    # compute physical coordinates of min and max z in the label image
    slack = 10.0  # mm
    size = label_img.GetSize()
    min_z_phys = label_img.TransformIndexToPhysicalPoint([0, 0, 0])[2]
    max_z_phys = label_img.TransformIndexToPhysicalPoint([0, 0, size[2] - 1])[2]

    dist_1 = min(abs(max_z_phys - com_phys_1[2]), abs(min_z_phys - com_phys_1[2]))
    dist_2 = min(abs(max_z_phys - com_phys_2[2]), abs(min_z_phys - com_phys_2[2]))
    if verbose:
        print(
            f"Type 5 descending aorta landmark distances to top/bottom of scan: {dist_1:.1f} mm and {dist_2:.1f} mm"
        )

    if dist_1 > slack or dist_2 > slack:
        msg = f"A landmark is not at the top or bottom of the scan - and it should for type 5. Distances to borders: {dist_1:.1f} mm and {dist_2:.1f} mm. For {aorta_name}"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    f_p_out = open(p_1_name, "w")
    f_p_out.write(f"{com_phys_1[0]} {com_phys_1[1]} {com_phys_1[2]}")
    f_p_out.close()

    f_p_out = open(p_2_name, "w")
    f_p_out.write(f"{com_phys_2[0]} {com_phys_2[1]} {com_phys_2[2]}")
    f_p_out.close()

    return True


def compute_centerline_landmarks_based_on_scan_type(
    segm_folder,
    lm_folder,
    stats_folder,
    verbose,
    quiet,
    write_log_file,
    output_folder,
    use_ts_org_segmentations=True,
):
    """
    We need start and end points for the centerline computation
    Depending on the scan type we have different ways of computing these points
    """
    stats_file = f"{stats_folder}aorta_scan_type.json"

    scan_type_stats = read_json_file(stats_file)
    if not scan_type_stats:
        msg = f"Could not read {stats_file} can not compute centerline landmarks"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    scan_type = scan_type_stats["scan_type"]

    if scan_type == "2":
        return compute_centerline_landmarks_for_aorta_type_2(
            segm_folder,
            lm_folder,
            verbose,
            quiet,
            write_log_file,
            output_folder,
            use_ts_org_segmentations=use_ts_org_segmentations,
        )
    if scan_type == "5":
        if not compute_centerline_landmarks_for_aorta_type_5_annulus(
            segm_folder,
            lm_folder,
            verbose,
            quiet,
            write_log_file,
            output_folder,
            use_ts_org_segmentations=use_ts_org_segmentations,
        ):
            return False
        return compute_centerline_landmarks_for_aorta_type_5_descending(
            segm_folder,
            lm_folder,
            verbose,
            quiet,
            write_log_file,
            output_folder,
            use_ts_org_segmentations=use_ts_org_segmentations,
        )
    if scan_type in ["1", "1b", "1c", "1d"]:
        return compute_centerline_landmarks_for_aorta_type_1(
            segm_folder,
            lm_folder,
            verbose,
            quiet,
            write_log_file,
            output_folder,
            use_ts_org_segmentations=use_ts_org_segmentations,
        )

    msg = f"Can not compute centerline landmarks for scan type {scan_type} for {segm_folder}"
    if not quiet:
        print(msg)
    if write_log_file:
        write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    return False


def extract_surfaces_for_centerlines(
    segm_folder,
    stats_folder,
    surface_folder,
    verbose,
    quiet,
    write_log_file,
    output_folder,
    use_ts_org_segmentations=True,
):
    """
    Extract the surface(s) needed for centerline computation
    """
    stats_file = f"{stats_folder}aorta_scan_type.json"

    scan_type_stats = read_json_file(stats_file)
    if not scan_type_stats:
        msg = f"Could not read {stats_file} can not compute centerline landmarks"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    scan_type = scan_type_stats["scan_type"]

    if scan_type in ["1", "1b", "1c", "1d", "2"]:
        aorta_segm_in = f"{segm_folder}aorta_lumen.nii.gz"
        if use_ts_org_segmentations:
            aorta_segm_in = f"{segm_folder}aorta_lumen_hires_raw.nii.gz"

        if scan_type in ["1", "1b", "1c", "1d"]:
            aorta_segm_in = f"{segm_folder}aorta_lumen_extended.nii.gz"
            if use_ts_org_segmentations:
                aorta_segm_in = f"{segm_folder}aorta_lumen_extended_ts_org.nii.gz"

        aorta_surface_out = f"{surface_folder}aorta_surface_raw.vtp"
        aorta_surface_cl_out = f"{surface_folder}aorta_surface_for_centerline.vtp"
        if os.path.exists(aorta_surface_cl_out):
            if verbose:
                print(f"{aorta_surface_cl_out} already exists - skipping")
        else:
            if verbose:
                print(f"Extracting {aorta_surface_out}")
            aorta_surface = convert_label_map_to_surface(
                aorta_segm_in, segment_id=1, only_largest_component=True
            )
            if aorta_surface is None:
                msg = f"Could not compute aorta surface from {aorta_segm_in}"
                if not quiet:
                    print(msg)
                if write_log_file:
                    write_message_to_log_file(
                        base_dir=output_folder, message=msg, level="error"
                    )
                return False
            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetFileName(aorta_surface_out)
            writer.SetInputData(aorta_surface)
            writer.Write()

            surface_cl = preprocess_surface_for_centerline_extraction(aorta_surface)
            if surface_cl is None:
                msg = f"Could not preprocess aorta surface from {aorta_segm_in} for centerline computation"
                if not quiet:
                    print(msg)
                if write_log_file:
                    write_message_to_log_file(
                        base_dir=output_folder, message=msg, level="error"
                    )
                return False
            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetFileName(aorta_surface_cl_out)
            writer.SetInputData(surface_cl)
            writer.Write()

        return True

    if scan_type == "5":
        aorta_segm_in = f"{segm_folder}aorta_lumen_extended.nii.gz"
        if use_ts_org_segmentations:
            aorta_segm_in = f"{segm_folder}aorta_lumen_extended_ts_org.nii.gz"

        aorta_surface_out = f"{surface_folder}aorta_annulus_surface_raw.vtp"
        aorta_surface_cl_out = (
            f"{surface_folder}aorta_annulus_surface_for_centerline.vtp"
        )
        if os.path.exists(aorta_surface_cl_out):
            if verbose:
                print(f"{aorta_surface_cl_out} already exists - skipping")
        else:
            if verbose:
                print(f"Extracting {aorta_surface_out}")
            aorta_surface = convert_label_map_to_surface(
                aorta_segm_in, segment_id=1, only_largest_component=True
            )
            if aorta_surface is None:
                msg = f"Could not compute aorta surface from {aorta_segm_in}"
                if not quiet:
                    print(msg)
                if write_log_file:
                    write_message_to_log_file(
                        base_dir=output_folder, message=msg, level="error"
                    )
                return False
            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetFileName(aorta_surface_out)
            writer.SetInputData(aorta_surface)
            writer.Write()

            surface_cl = preprocess_surface_for_centerline_extraction(aorta_surface)
            if surface_cl is None:
                msg = f"Could not preprocess aorta surface from {aorta_segm_in} for centerline computation"
                if not quiet:
                    print(msg)
                if write_log_file:
                    write_message_to_log_file(
                        base_dir=output_folder, message=msg, level="error"
                    )
                return False
            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetFileName(aorta_surface_cl_out)
            writer.SetInputData(surface_cl)
            writer.Write()

        aorta_segm_in = f"{segm_folder}aorta_lumen_descending.nii.gz"
        if use_ts_org_segmentations:
            aorta_segm_in = f"{segm_folder}aorta_lumen_descending_raw.nii.gz"

        aorta_surface_out = f"{surface_folder}aorta_descending_surface_raw.vtp"
        aorta_surface_cl_out = (
            f"{surface_folder}aorta_descending_surface_for_centerline.vtp"
        )
        if os.path.exists(aorta_surface_cl_out):
            if verbose:
                print(f"{aorta_surface_cl_out} already exists - skipping")
        else:
            if verbose:
                print(f"Extracting {aorta_surface_out}")
            aorta_surface = convert_label_map_to_surface(
                aorta_segm_in, segment_id=1, only_largest_component=True
            )
            if aorta_surface is None:
                msg = f"Could not compute aorta surface from {aorta_segm_in}"
                if not quiet:
                    print(msg)
                if write_log_file:
                    write_message_to_log_file(
                        base_dir=output_folder, message=msg, level="error"
                    )
                return False
            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetFileName(aorta_surface_out)
            writer.SetInputData(aorta_surface)
            writer.Write()

            surface_cl = preprocess_surface_for_centerline_extraction(aorta_surface)
            if surface_cl is None:
                msg = f"Could not preprocess aorta surface from {aorta_segm_in} for centerline computation"
                if not quiet:
                    print(msg)
                if write_log_file:
                    write_message_to_log_file(
                        base_dir=output_folder, message=msg, level="error"
                    )
                return False
            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetFileName(aorta_surface_cl_out)
            writer.SetInputData(surface_cl)
            writer.Write()
        return True

    msg = f"Can not compute extract surface for centerline for scan type {scan_type} for {segm_folder}"
    if not quiet:
        print(msg)
    if write_log_file:
        write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    return False


def compute_center_line(
    stats_folder,
    lm_folder,
    surface_folder,
    cl_folder,
    verbose,
    quiet,
    write_log_file,
    output_folder,
):
    stats_file = f"{stats_folder}aorta_scan_type.json"

    scan_type_stats = read_json_file(stats_file)
    if not scan_type_stats:
        msg = f"Could not read {stats_file} can not compute centerline landmarks"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    scan_type = scan_type_stats["scan_type"]

    if scan_type in ["1", "1b", "1c", "1d", "2"]:
        aorta_surf_name = f"{surface_folder}aorta_surface_for_centerline.vtp"

        cl_name = f"{cl_folder}aorta_centerline.vtp"
        cl_name_fail = f"{cl_folder}aorta_centerline_failed.txt"
        start_p_file = f"{lm_folder}aorta_start_point.txt"
        end_p_file = f"{lm_folder}aorta_end_point.txt"

        if os.path.exists(cl_name):
            if verbose:
                print(f"{cl_name} already exists - skipping")
            return True

        if os.path.exists(cl_name_fail):
            msg = f"Centerline failed before on {aorta_surf_name}"
            if not quiet:
                print(msg)
            if write_log_file:
                write_message_to_log_file(
                    base_dir=output_folder, message=msg, level="error"
                )
            return False

        if verbose:
            print(f"Computing centerline from {aorta_surf_name}")

        if not clutils.compute_single_center_line(
            aorta_surf_name, cl_name, start_p_file, end_p_file
        ):
            msg = f"Failed to compute centerline from {aorta_surf_name}"
            if not quiet:
                print(msg)
            if write_log_file:
                write_message_to_log_file(
                    base_dir=output_folder, message=msg, level="error"
                )
            with open(cl_name_fail, "w") as f:
                f.write(f"Failed to compute centerline from {aorta_surf_name}")
            return False
        return True

    if scan_type == "5":
        aorta_surf_name = f"{surface_folder}aorta_annulus_surface_for_centerline.vtp"
        cl_name = f"{cl_folder}aorta_centerline_annulus.vtp"
        cl_name_fail = f"{cl_folder}aorta_centerline_annulus_failed.txt"
        start_p_file = f"{lm_folder}aorta_start_point_annulus.txt"
        end_p_file = f"{lm_folder}aorta_end_point_annulus.txt"

        if os.path.exists(cl_name):
            if verbose:
                print(f"{cl_name} already exists - skipping")
            return True

        if os.path.exists(cl_name_fail):
            msg = f"Centerline failed before on {aorta_surf_name}"
            if not quiet:
                print(msg)
            if write_log_file:
                write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
            return False

        if verbose:
            print(f"Computing centerline from {aorta_surf_name}")

        if not clutils.compute_single_center_line(aorta_surf_name, cl_name, start_p_file, end_p_file):
            msg = f"Failed to compute centerline from {aorta_surf_name}"
            if not quiet:
                print(msg)
            if write_log_file:
                write_message_to_log_file(
                    base_dir=output_folder, message=msg, level="error"
                )
            with open(cl_name_fail, "w") as f:
                f.write(f"Failed to compute centerline from {aorta_surf_name}")
            return False

        aorta_surf_name = f"{surface_folder}aorta_descending_surface_for_centerline.vtp"
        cl_name = f"{cl_folder}aorta_centerline_descending.vtp"
        cl_name_fail = f"{cl_folder}aorta_centerline_descending_failed.txt"
        start_p_file = f"{lm_folder}aorta_start_point_descending.txt"
        end_p_file = f"{lm_folder}aorta_end_point_descending.txt"

        if os.path.exists(cl_name):
            if verbose:
                print(f"{cl_name} already exists - skipping")
            return True

        if os.path.exists(cl_name_fail):
            msg = f"Centerline failed before on {aorta_surf_name}"
            if not quiet:
                print(msg)
            if write_log_file:
                write_message_to_log_file(
                    base_dir=output_folder, message=msg, level="error"
                )
            return False

        if verbose:
            print(f"Computing centerline from {aorta_surf_name}")

        if not clutils.compute_single_center_line(
            aorta_surf_name, cl_name, start_p_file, end_p_file
        ):
            msg = f"Failed to compute centerline from {aorta_surf_name}"
            if not quiet:
                print(msg)
            if write_log_file:
                write_message_to_log_file(
                    base_dir=output_folder, message=msg, level="error"
                )
            with open(cl_name_fail, "w") as f:
                f.write(f"Failed to compute centerline from {aorta_surf_name}")
            return False
        return True

    msg = f"Can not compute centerline for scan type {scan_type}"
    if not quiet:
        print(msg)
    if write_log_file:
        write_message_to_log_file(base_dir=output_folder, message=msg, level="warning")
    return False


def compute_center_line_using_skeleton(segm_folder, stats_folder, lm_folder, surface_folder, cl_folder, verbose, quiet,
                                       write_log_file, output_folder, use_ts_org_segmentations=True):
    stats_file = f"{stats_folder}aorta_scan_type.json"

    scan_type_stats = read_json_file(stats_file)
    if not scan_type_stats:
        msg = f"Could not read {stats_file} can not compute centerline landmarks"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    scan_type = scan_type_stats["scan_type"]

    if scan_type in ["2"]:
        aorta_segm_in = f"{segm_folder}aorta_lumen_extended.nii.gz"
        if use_ts_org_segmentations:
            aorta_segm_in = f"{segm_folder}aorta_lumen_extended_ts_org.nii.gz"

        skeleton_pd_name = f"{surface_folder}aorta_skeleton.vtp"
        pruned_skeleton_pd_name = f"{surface_folder}aorta_pruned_skeleton.vtp"
        cl_name = f"{cl_folder}aorta_centerline.vtp"
        cl_name_fail = f"{cl_folder}aorta_centerline_failed.txt"
        start_p_file = f"{lm_folder}aorta_start_point.txt"
        end_p_file = f"{lm_folder}aorta_end_point.txt"

        # if os.path.exists(cl_name):
        #     if verbose:
        #         print(f"{cl_name} already exists - skipping")
        #     return True
        #
        # if os.path.exists(cl_name_fail):
        #     msg = f"Centerline failed before on {aorta_segm_in}"
        #     if not quiet:
        #         print(msg)
        #     if write_log_file:
        #         write_message_to_log_file(
        #             base_dir=output_folder, message=msg, level="error"
        #         )
        #     return False

        if verbose:
            print(f"Computing centerline from {aorta_segm_in}")

        label_map = read_nifti_with_logging_cached(
            aorta_segm_in, verbose, quiet, write_log_file, output_folder)
        if label_map is None:
            return False

        start_point = clutils.read_landmarks(start_p_file)
        end_point = clutils.read_landmarks(end_p_file)
        if start_point is None or end_point is None:
            print(f"Could not read landmarks {start_p_name} or {end_p_name}")
            return False

        aorta_centerline = AortaCenterliner(label_map, scan_type, start_point, end_point,
                                      output_folder, verbose, quiet, write_log_file)
        if not aorta_centerline.compute_centerline():
            msg = f"Failed to compute centerline from {aorta_segm_in}"
            if not quiet:
                print(msg)
            if write_log_file:
                write_message_to_log_file(
                    base_dir=output_folder, message=msg, level="error"
                )
            with open(cl_name_fail, "w") as f:
                f.write(f"Failed to compute centerline from {aorta_segm_in}")
            return False

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(skeleton_pd_name)
        writer.SetInputData(aorta_centerline.skeleton_polydata)
        writer.Write()

        writer.SetFileName(pruned_skeleton_pd_name)
        writer.SetInputData(aorta_centerline.pruned_skeleton)
        writer.Write()

        return True
    return False


def compute_infrarenal_section_using_kidney_to_kidney_line(
    segm_folder,
    stats_folder,
    lm_folder,
    cl_folder,
    verbose,
    quiet,
    write_log_file,
    output_folder,
):
    """
    Compute the infrarenal part of the aorta based on the centerline
    and the position of the kidneys
    """
    lm_renal_name_cl = f"{lm_folder}renal_on_centerline.txt"
    sdf_name = f"{segm_folder}out_of_scan_sdf.nii.gz"
    total_in_name = f"{segm_folder}total.nii.gz"
    parts_stats = read_json_file(f"{stats_folder}aorta_parts.json")

    segm_id_kidney_l = 3
    segm_id_kidney_r = 2
    min_size = 100

    # This should go into a config setting
    check_partial_kidneys = False

    debug = False

    infrarenal_out = f"{cl_folder}infrarenal_section.json"
    if os.path.exists(infrarenal_out):
        if verbose:
            print(f"{infrarenal_out} already exists - skipping")
        return True

    if verbose:
        print(f"Computing {infrarenal_out}")

    if not os.path.exists(sdf_name):
        msg = f"Out-of-scan SDF {sdf_name} not found. Can not compute centerline landmarks for type 2"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    sdf_img = read_nifti_with_logging_cached(
        sdf_name, verbose, quiet, write_log_file, output_folder
    )
    if sdf_img is None:
        return False

    # try:
    #     sdf_img = sitk.ReadImage(sdf_name)
    # except RuntimeError as e:
    #     msg = f"Could not read {sdf_name}: {str(e)} got an exception"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     return False

    sdf_data = sitk.GetArrayFromImage(sdf_img)

    label_img = read_nifti_with_logging_cached(
        total_in_name, verbose, quiet, write_log_file, output_folder
    )
    if label_img is None:
        return False
    #
    #
    # try:
    #     label_img = sitk.ReadImage(total_in_name)
    # except RuntimeError as e:
    #     msg = f"Could not read {total_in_name}: {str(e)} got an exception"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     return False

    label_img_np = sitk.GetArrayFromImage(label_img)

    mask_np_l = label_img_np == segm_id_kidney_l
    if np.sum(mask_np_l) < min_size:
        msg = "Missing kidney left for infrarenal region computation"
        if verbose:
            print(msg)
        return True

    # This can be disabled if we do not want to check for partial kidneys
    if check_partial_kidneys:
        min_touch_dist = 0.50
        # Check out-of-field-dist to see if any part of the segmentation is close to the boundary
        sdf_filter = sdf_data[mask_np_l]
        min_dist = np.min(sdf_filter)
        if min_dist <= min_touch_dist:
            if verbose:
                msg = "Kidney left is touching out-of-scan for infrarenal region computation"
                print(msg)
            return True

    com_np_l = measurements.center_of_mass(mask_np_l)
    com_np_l = [com_np_l[2], com_np_l[1], com_np_l[0]]
    com_l = label_img.TransformIndexToPhysicalPoint(
        [int(com_np_l[0]), int(com_np_l[1]), int(com_np_l[2])]
    )

    # We assume that left and right kidney are in the same label img
    mask_np_r = label_img_np == segm_id_kidney_r
    if np.sum(mask_np_r) < 100:
        msg = "Missing kidney right for infrarenal region computation"
        if verbose:
            print(msg)
        return True

    # This can be disabled if we do not want to check for partial kidneys
    if check_partial_kidneys:
        min_touch_dist = 0.50
        # Check out-of-field-dist to see if any part of the segmentation is close to the boundary
        sdf_filter = sdf_data[mask_np_r]
        min_dist = np.min(sdf_filter)
        if min_dist <= min_touch_dist:
            msg = (
                "Kidney right is touching out-of-scan for infrarenal region computation"
            )
            if verbose:
                print(msg)
            return True

    com_np_r = measurements.center_of_mass(mask_np_r)
    com_np_r = [com_np_r[2], com_np_r[1], com_np_r[0]]
    com_r = label_img.TransformIndexToPhysicalPoint(
        [int(com_np_r[0]), int(com_np_r[1]), int(com_np_r[2])]
    )

    if parts_stats:
        n_aorta_parts = parts_stats["aorta_parts"]
    else:
        n_aorta_parts = 1

    if n_aorta_parts == 1:
        cl_file = f"{cl_folder}aorta_centerline.vtp"
    else:
        cl_file = f"{cl_folder}aorta_centerline_descending.vtp"

    pd = vtk.vtkXMLPolyDataReader()
    pd.SetFileName(cl_file)
    pd.Update()
    cl = pd.GetOutput()
    if cl.GetNumberOfPoints() < 10:
        msg = f"Centerline has less than 10 points in {cl_file}"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    # Find lower point
    # Distance should be set somewhere else
    low_dist = 10
    low_idx = -1
    for idx in range(cl.GetNumberOfPoints()):
        cl_dist = cl.GetPointData().GetScalars().GetValue(idx)
        if cl_dist > low_dist:
            low_idx = idx
            break

    if low_idx < 0:
        msg = "CL low idx < 0: something weird. No infrarenal point"
        if verbose:
            print(msg)
        # if write_log_file:
        #     write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
        return True

    low_infra_p = cl.GetPoint(low_idx)
    low_infra_dist = cl.GetPointData().GetScalars().GetValue(low_idx)

    low_normal = clutils.estimate_normal_from_centerline(cl, low_idx)
    if low_normal is None:
        msg = f"Could not estimate normal at renal point. Something wrong. No infrarenal points for {cl_file}"
        if verbose:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return True

    # Find closest point on centerline to the line that goes from the center of mass of the left kidney to the right
    min_line_dist = np.inf
    infra_idx = -1

    for idx in range(cl.GetNumberOfPoints()):
        cl_p = cl.GetPoint(idx)
        dist_l = vtk.vtkLine.DistanceToLine(cl_p, com_l, com_r)
        if dist_l < min_line_dist:
            min_line_dist = dist_l
            infra_idx = idx

    infra_dist = cl.GetPointData().GetScalars().GetValue(infra_idx)

    # Check for very low kidneys (sometimes at the hips)
    if infra_dist < low_dist:
        msg = f"Very low kidney(s) detected. Re-estiming renal point for {cl_file}"
        if verbose:
            print(msg)
        # if write_log_file:
        #     write_message_to_log_file(base_dir=output_folder, message=msg, level="warning")
        locator = vtk.vtkPointLocator()
        locator.SetDataSet(cl)
        locator.BuildLocator()

        # Find closest point on centerline from center of masses of the kidneys
        idx_l = locator.FindClosestPoint(com_l)
        idx_r = locator.FindClosestPoint(com_r)

        dist_l = cl.GetPointData().GetScalars().GetValue(idx_l)
        dist_r = cl.GetPointData().GetScalars().GetValue(idx_r)
        if debug:
            print(f"Dist l: {dist_l} dist r: {dist_r}")
        if dist_l > dist_r:
            if verbose:
                print("Using left kidney since right kidney probably very low")
            infra_idx = idx_l
            infra_dist = dist_l
        else:
            if verbose:
                print("Using right kidney since left kidney probably very low")
            infra_idx = idx_r
            infra_dist = dist_r

    # If both kidneys are very low. We can not estimat renal point
    if infra_dist < low_dist:
        msg = f"Can not compute infrarenal point. Probably both kidneys are too low. For {cl_file}"
        if verbose:
            print(msg)
        # if write_log_file:
        #     write_message_to_log_file(base_dir=output_folder, message=msg, level="warning")
        return True

    infra_p = cl.GetPoint(infra_idx)

    normal = clutils.estimate_normal_from_centerline(cl, infra_idx)
    if normal is None:
        msg = f"Could not estimate normal at renal point. Something wrong. No infrarenal point for {cl_file}"
        if verbose:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="warning"
            )
        return True

    f_p_out = open(lm_renal_name_cl, "w")
    f_p_out.write(f"{infra_p[0]} {infra_p[1]} {infra_p[2]}")
    f_p_out.close()

    infra_renal_stats = {
        "low_dist": low_infra_dist,
        "low_cl_idx": low_idx,
        "low_cl_pos": list(low_infra_p),
        "low_cl_normal": list(low_normal),
        "distance": infra_dist,
        "cl_idx": infra_idx,
        "cl_pos": list(infra_p),
        "cl_normal": list(normal),
    }
    json_object = json.dumps(infra_renal_stats, indent=4)
    with open(infrarenal_out, "w") as outfile:
        outfile.write(json_object)

    return True


def compute_aortic_arch_landmarks_on_centerline(
    lm_folder, cl_folder, verbose, quiet, write_log_file, output_folder
):
    """
    Compute the landmarks on the centerline of the aortic arch by using the landmarks from the three top arteries
    """
    aortic_arch_out = f"{cl_folder}aortic_arch.json"
    lm_carotid_name = f"{lm_folder}common_carotid_artery_left.txt"
    lm_trunc_name = f"{lm_folder}brachiocephalic_trunc.txt"
    lm_subclavian_name = f"{lm_folder}subclavian_artery_left.txt"

    lm_carotid_name_cl = f"{lm_folder}common_carotid_artery_left_on_centerline.txt"
    lm_trunc_name_cl = f"{lm_folder}brachiocephalic_trunc_on_centerline.txt"
    lm_subclavian_name_cl = f"{lm_folder}subclavian_artery_left_on_centerline.txt"
    cl_file = f"{cl_folder}aorta_centerline.vtp"

    # debug = False

    if os.path.exists(aortic_arch_out):
        if verbose:
            print(f"{aortic_arch_out} already exists - skipping")
        return True
    if verbose:
        print(f"Computing {aortic_arch_out}")

    required_files = [lm_trunc_name, lm_subclavian_name, cl_file]
    for f_r in required_files:
        if not os.path.exists(f_r):
            msg = f"Missing {f_r} for aortic arch computation"
            if verbose:
                print(msg)
            return True

    pd = vtk.vtkXMLPolyDataReader()
    pd.SetFileName(cl_file)
    pd.Update()
    cl = pd.GetOutput()
    if cl.GetNumberOfPoints() < 10:
        msg = f"Centerline has less than 10 points {cl_file}"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    p_carotid = clutils.read_landmarks(lm_carotid_name)
    p_trunc = clutils.read_landmarks(lm_trunc_name)
    p_subclavian = clutils.read_landmarks(lm_subclavian_name)

    locator = vtk.vtkPointLocator()
    locator.SetDataSet(cl)
    locator.BuildLocator()

    min_cl_dist = np.inf
    max_cl_dist = -np.inf

    # Find closest point on centerline
    if p_carotid is not None:
        idx_carotid = locator.FindClosestPoint(p_carotid)
        dist_carotid = cl.GetPointData().GetScalars().GetValue(idx_carotid)
        p_cl_carotid = cl.GetPoint(idx_carotid)
        if dist_carotid < min_cl_dist:
            min_cl_dist = dist_carotid
        if dist_carotid > max_cl_dist:
            max_cl_dist = dist_carotid

        p_out = open(lm_carotid_name_cl, "w")
        p_out.write(f"{p_cl_carotid[0]} {p_cl_carotid[1]} {p_cl_carotid[2]}")
        p_out.close()

    if p_trunc is not None:
        idx_trunc = locator.FindClosestPoint(p_trunc)
        dist_trunc = cl.GetPointData().GetScalars().GetValue(idx_trunc)
        p_cl_trunc = cl.GetPoint(idx_trunc)

        if dist_trunc < min_cl_dist:
            min_cl_dist = dist_trunc
        if dist_trunc > max_cl_dist:
            max_cl_dist = dist_trunc

        p_out = open(lm_trunc_name_cl, "w")
        p_out.write(f"{p_cl_trunc[0]} {p_cl_trunc[1]} {p_cl_trunc[2]}")
        p_out.close()

    if p_subclavian is not None:
        idx_subclavian = locator.FindClosestPoint(p_subclavian)
        dist_subclavian = cl.GetPointData().GetScalars().GetValue(idx_subclavian)
        p_cl_subclavian = cl.GetPoint(idx_subclavian)

        if dist_subclavian < min_cl_dist:
            min_cl_dist = dist_subclavian
        if dist_subclavian > max_cl_dist:
            max_cl_dist = dist_subclavian

        p_out = open(lm_subclavian_name_cl, "w")
        p_out.write(f"{p_cl_subclavian[0]} {p_cl_subclavian[1]} {p_cl_subclavian[2]}")
        p_out.close()

    # Translate along centerline.
    # TODO: Set translation value elsewhere.
    #  In 2010 ACCF/AHA/AATS/ACR/ASA/SCA/SCAI/SIR/STS/SVM
    #  Guidelines for the Diagnosis and Management of Patients With descending Aortic Disease
    # They mention 2 cm distal to the left subclavian artery
    trans_value_min = 20
    trans_value_max = 10
    min_idx, min_val = clutils.find_position_on_centerline_based_on_scalar(
        cl, min_cl_dist - trans_value_min
    )
    max_idx, max_val = clutils.find_position_on_centerline_based_on_scalar(
        cl, max_cl_dist + trans_value_max
    )
    min_pos = cl.GetPoint(min_idx)
    max_pos = cl.GetPoint(max_idx)
    min_normal = clutils.estimate_normal_from_centerline(cl, min_idx)
    max_normal = clutils.estimate_normal_from_centerline(cl, max_idx)

    if min_normal is None or max_normal is None:
        msg = f"Could not estimate normal at aortic arch point. Something wrong. No aortic arch point for {cl_file}"
        if verbose:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="warning"
            )
        return True

    aortic_arch_stats = {
        "min_cl_dist": min_val,
        "min_cl_idx": min_idx,
        "min_cl_pos": list(min_pos),
        "min_cl_normal": list(min_normal),
        "max_cl_dist": max_val,
        "max_cl_idx": max_idx,
        "max_cl_pos": list(max_pos),
        "max_cl_normal": list(max_normal),
    }
    json_object = json.dumps(aortic_arch_stats, indent=4)
    with open(aortic_arch_out, "w") as outfile:
        outfile.write(json_object)

    return True


def compute_diaphragm_point_on_centerline(
    lm_folder, cl_folder, stats_folder, verbose, quiet, write_log_file, output_folder
):
    """
    Compute the point on the centerline closest to the diaphragm.
    If we just use the naive closest point, it will be a point on the top of aorta.
    Instead we restrict it to be in between the bottom and the aortic arch
    """
    aortic_arch_in = f"{cl_folder}aortic_arch.json"
    diaphragm_out = f"{cl_folder}diaphragm.json"
    lm_diaphragm_name = f"{lm_folder}diaphragm.txt"
    lm_diaphragm_name_cl = f"{lm_folder}diaphragm_on_centerline.txt"
    infrarenal_in = f"{cl_folder}infrarenal_section.json"
    # debug = False

    if os.path.exists(diaphragm_out):
        if verbose:
            print(f"{diaphragm_out} already exists - skipping")
        return True

    parts_stats = read_json_file(f"{stats_folder}aorta_parts.json")
    if parts_stats:
        n_aorta_parts = parts_stats["aorta_parts"]
    else:
        n_aorta_parts = 1

    if n_aorta_parts == 1:
        cl_file = f"{cl_folder}aorta_centerline.vtp"
    else:
        cl_file = f"{cl_folder}aorta_centerline_descending.vtp"

    required_files = [lm_diaphragm_name, cl_file, infrarenal_in]
    for f_r in required_files:
        if not os.path.exists(f_r):
            msg = f"Missing {f_r} for diaphragm computation for {cl_file}"
            if verbose:
                print(msg)
            # no need to log this
            # if write_log_file:
            #     write_message_to_log_file(base_dir=output_folder, message=msg, level="warning")
            return True

    if verbose:
        print(f"Computing diaphragm {diaphragm_out}")

    arch_dist = np.inf
    if os.path.exists(aortic_arch_in):
        try:
            with open(aortic_arch_in, "r") as openfile:
                arch_stats = json.load(openfile)
        except IOError as e:
            print(f"I/O error({e.errno}): {e.strerror}: {aortic_arch_in}")
            return False

        arch_dist = arch_stats["min_cl_dist"]

    infra_stats = read_json_file(infrarenal_in)
    if infra_stats is None:
        msg = f"Could not read {infrarenal_in} for diaphragm computation for {cl_file}"
        if verbose:
            print(msg)
        # no need to log this
        # if write_log_file:
        #     write_message_to_log_file(base_dir=output_folder, message=msg, level="warning")
        return True

    infra_dist = infra_stats["distance"]

    pd = vtk.vtkXMLPolyDataReader()
    pd.SetFileName(cl_file)
    pd.Update()
    cl = pd.GetOutput()
    if cl.GetNumberOfPoints() < 10:
        msg = f"Centerline has less than 10 points in {cl_file}"
        if not quiet:
            print(msg)
        if write_message_to_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    p_diaphragm = clutils.read_landmarks(lm_diaphragm_name)

    min_dist = np.inf
    min_idx = -1
    for idx in range(cl.GetNumberOfPoints()):
        cl_dist = cl.GetPointData().GetScalars().GetValue(idx)
        if cl_dist > arch_dist:
            break
        p_cl = cl.GetPoint(idx)
        dist = np.linalg.norm(np.subtract(p_cl, p_diaphragm))
        if dist < min_dist:
            min_dist = dist
            min_idx = idx

    if min_idx < 0:
        msg = f"No matching point found on centerline for diaphragm computation for {cl_file}"
        if verbose:
            print(msg)
        # if write_log_file:
        #     write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
        return True

    idx_diaphragm = min_idx
    dist_diaphragm = cl.GetPointData().GetScalars().GetValue(idx_diaphragm)
    if dist_diaphragm < infra_dist:
        msg = f"Abdominal cl dist less than infrarenal cl dist. No abdominal point for {cl_file}"
        if verbose:
            print(msg)
        # if write_log_file:
        #     write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
        return True

    p_cl_diaphragm = cl.GetPoint(idx_diaphragm)

    diaphragm_normal = clutils.estimate_normal_from_centerline(cl, idx_diaphragm)
    if diaphragm_normal is None:
        msg = f"Could not estimate normal at diaphragm point. Something wrong. No diaphragm point for {cl_file}"
        if verbose:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return True

    f_p_out = open(lm_diaphragm_name_cl, "w")
    f_p_out.write(f"{p_cl_diaphragm[0]} {p_cl_diaphragm[1]} {p_cl_diaphragm[2]}")
    f_p_out.close()

    diaphragm_stats = {
        "diaphragm_cl_dist": dist_diaphragm,
        "diaphragm_cl_idx": idx_diaphragm,
        "diaphragm_cl_pos": list(p_cl_diaphragm),
        "diaphragm_cl_normal": list(diaphragm_normal),
    }
    json_object = json.dumps(diaphragm_stats, indent=4)
    with open(diaphragm_out, "w") as outfile:
        outfile.write(json_object)
    return True


def compute_ventricularoaortic_point_on_centerline(
    lm_folder, cl_folder, stats_folder, verbose, quiet, write_log_file, output_folder
):
    """
    Compute the ventricularoaortic point on the centerline by finding the intersection of the aorta and the ventricle on the centerline
    """
    ventri_out = f"{cl_folder}ventricularaortic.json"
    lm_ventri_name = f"{lm_folder}ventricularoaortic_point.txt"
    lm_ventri_name_cl = f"{lm_folder}ventricularoaortic_on_centerline.txt"
    # debug = False

    if os.path.exists(ventri_out):
        if verbose:
            print(f"{ventri_out} already exists - skipping")
        return True

    if verbose:
        print(f"Computing ventricularoaortic point on centerline {ventri_out}")

    parts_stats = read_json_file(f"{stats_folder}/aorta_parts.json")
    if parts_stats:
        n_aorta_parts = parts_stats["aorta_parts"]
    else:
        n_aorta_parts = 1

    if n_aorta_parts == 1:
        cl_file = f"{cl_folder}/aorta_centerline.vtp"
    else:
        cl_file = f"{cl_folder}/aorta_centerline_annulus.vtp"

    required_files = [lm_ventri_name, cl_file]
    for f_r in required_files:
        if not os.path.exists(f_r):
            msg = f"Missing {f_r} for ventricularoaortic computation for {cl_file}"
            if verbose:
                print(msg)
            # if write_log_file:
            #     write_message_to_log_file(base_dir=output_folder, message=msg, level="warning")
            return True

    pd = vtk.vtkXMLPolyDataReader()
    pd.SetFileName(cl_file)
    pd.Update()
    cl = pd.GetOutput()
    if cl.GetNumberOfPoints() < 10:
        msg = f"Centerline has less than 10 points in {cl_file}"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    p_ventri = clutils.read_landmarks(lm_ventri_name)

    locator = vtk.vtkPointLocator()
    locator.SetDataSet(cl)
    locator.BuildLocator()
    idx_ventri = locator.FindClosestPoint(p_ventri)
    dist_ventri = cl.GetPointData().GetScalars().GetValue(idx_ventri)
    p_cl_ventri = cl.GetPoint(idx_ventri)

    ventri_normal = clutils.estimate_normal_from_centerline(cl, idx_ventri)
    if ventri_normal is None:
        msg = f"Could not estimate normal at ventricularoaortic point. Something wrong. No ventricularoaortic point for {cl_file}"
        if verbose:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return True

    f_p_out = open(lm_ventri_name_cl, "w")
    f_p_out.write(f"{p_cl_ventri[0]} {p_cl_ventri[1]} {p_cl_ventri[2]}")
    f_p_out.close()

    ventri_stats = {
        "ventri_cl_dist": dist_ventri,
        "ventri_cl_idx": idx_ventri,
        "ventri_cl_pos": list(p_cl_ventri),
        "ventri_cl_normal": list(ventri_normal),
    }
    json_object = json.dumps(ventri_stats, indent=4)
    with open(ventri_out, "w") as outfile:
        outfile.write(json_object)
    return True


def sample_aorta_center_line_hu_stats(
    input_file, cl_folder, stats_folder, verbose, quiet, write_log_file, output_folder
):
    """
    Compute HU stats
    """
    ct_name = input_file
    out_file = f"{cl_folder}aorta_centerline_hu_vals.csv"
    out_file_stats = f"{cl_folder}aorta_centerline_hu_stats.json"

    # If theses exists, they are used to bound the sampling to not go into the ventricle etc
    infrarenal_in = f"{cl_folder}infrarenal_section.json"
    ventri_in = f"{cl_folder}ventricularaortic.json"
    # debug = False

    if os.path.exists(out_file_stats):
        if verbose:
            print(f"{out_file_stats} already computed. Reading from file.")
        stats = read_json_file(out_file_stats)
        return stats

    if verbose:
        print(f"Computing {out_file_stats}")

    parts_stats = read_json_file(f"{stats_folder}aorta_parts.json")
    if parts_stats:
        n_aorta_parts = parts_stats["aorta_parts"]
    else:
        n_aorta_parts = 1

    if n_aorta_parts == 1:
        cl_file = f"{cl_folder}/aorta_centerline.vtp"
    else:
        cl_file = f"{cl_folder}/aorta_centerline_descending.vtp"

    stats = {}
    stats["cl_count"] = -1
    stats["cl_mean"] = -1
    stats["cl_std"] = -1
    stats["cl_med"] = -1
    stats["cl_q01"] = -1
    stats["cl_q99"] = -1

    if not os.path.exists(cl_file) or not os.path.exists(ct_name):
        msg = f"Missing files for aorta center line stats {cl_file} or {ct_name}"
        if verbose:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return None

    pd = vtk.vtkXMLPolyDataReader()
    pd.SetFileName(cl_file)
    pd.Update()
    cl = pd.GetOutput()
    if cl.GetNumberOfPoints() < 10:
        msg = f"Centerline has less than 10 points in {cl_file}"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return None

    min_dist = -np.inf
    max_dist = np.inf
    if os.path.exists(infrarenal_in):
        if verbose:
            print("Using infra-renal section for lower bound for sampling")
        try:
            with open(infrarenal_in, "r") as openfile:
                infrarenal_stats = json.load(openfile)
        except IOError as e:
            print(f"I/O error({e.errno}): {e.strerror}: {infrarenal_in}")
            return None

        min_dist = infrarenal_stats["low_dist"]

    # TODO: Should use the scan type to do proper sampling instead of just checking the number of parts
    if n_aorta_parts == 1 and os.path.exists(ventri_in):
        if verbose:
            print("Using ventricularoaoartic point as last point")
        try:
            with open(ventri_in, "r") as openfile:
                ventri_stats = json.load(openfile)
        except IOError as e:
            print(f"I/O error({e.errno}): {e.strerror}: {ventri_in}")
            return None
        max_dist = ventri_stats["ventri_cl_dist"]

    img = read_nifti_with_logging_cached(
        ct_name, verbose, quiet, write_log_file, output_folder
    )
    if img is None:
        return None

    # try:
    #     img = sitk.ReadImage(ct_name)
    # except RuntimeError as e:
    #     msg = f"Could not read {ct_name}: Got an exception {str(e)}"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     return None

    use_mean_filter = False
    if use_mean_filter:
        if verbose:
            print("Mean filtering for value extraction")
        meanfilter = sitk.MeanImageFilter()
        meanfilter.SetRadius(1)
        meanfilter = meanfilter.Execute(img)
        i2 = sitk.GetArrayFromImage(meanfilter)
        if verbose:
            print("Mean filtering done")
    else:
        i2 = sitk.GetArrayFromImage(img)
    i2_np = i2.transpose(2, 1, 0)

    hu_values = []

    f = open(out_file, "w")

    shp = i2_np.shape

    for idx in range(cl.GetNumberOfPoints()):
        p = cl.GetPoint(idx)
        p_indx = img.TransformPhysicalPointToIndex(p)

        # The centerline can in rare cases, go outside the scan
        if (
            0 <= p_indx[0] < shp[0]
            and 0 <= p_indx[1] < shp[1]
            and 0 <= p_indx[2] < shp[2]
        ):
            cl_dist = cl.GetPointData().GetScalars().GetValue(idx)
            if min_dist < cl_dist < max_dist:
                val = i2_np[p_indx]
                hu_values.append(val)
                f.write(f"{cl_dist}, {val}\n")

    if len(hu_values) > 0:
        stats["cl_count"] = len(hu_values)
        stats["cl_mean"] = np.mean(hu_values)
        stats["cl_std"] = np.std(hu_values)
        stats["cl_med"] = np.median(hu_values)
        stats["cl_q01"] = np.percentile(hu_values, 1)
        stats["cl_q99"] = max_hu = np.percentile(hu_values, 99)

        # TODO: the default values of 0 and 1000 should be set elsewhere
        # We use 0 as the base min, since the max value is the most interesting
        min_hu = 0
        max_hu = min(max_hu, 1000)
        stats["img_window"] = max_hu - min_hu
        stats["img_level"] = (max_hu + min_hu) / 2.0
    else:
        msg = f"Did not manage to sample any centerline HU values for {ct_name}"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        f.close()
        os.remove(out_file)
        return None

    json_object = json.dumps(stats, indent=4)
    with open(out_file_stats, "w") as outfile:
        outfile.write(json_object)

    return stats


def compute_straightened_volume_using_cpr(
    input_file,
    segm_folder,
    cl_folder,
    stats_folder,
    verbose,
    quiet,
    write_log_file,
    output_folder,
    use_raw_segmentations=False,
):
    """
    Compute straightened volume using curved planar reformation (CPR)
    """
    file_name = input_file

    parts_stats = read_json_file(f"{stats_folder}aorta_parts.json")
    stats_file = f"{stats_folder}aorta_scan_type.json"

    if verbose:
        if use_raw_segmentations:
            print(
                "Computing straightened volume using CPR for raw segmentations from TotalSegmentator"
            )
        else:
            print("Computing straightened volume using CPR")

    if parts_stats:
        n_aorta_parts = parts_stats["aorta_parts"]
    else:
        n_aorta_parts = 1

    scan_type_stats = read_json_file(stats_file)
    if not scan_type_stats:
        msg = f"Could not read {stats_file} can not do CPR"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    scan_type = scan_type_stats["scan_type"]

    ct_img = read_nifti_with_logging_cached(
        input_file, verbose, quiet, write_log_file, output_folder
    )
    if ct_img is None:
        return False
    #
    # try:
    #     ct_img = sitk.ReadImage(input_file)
    # except RuntimeError as e:
    #     msg = f"Could not read {input_file} for straigthening: {str(e)} got an exception"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     return False

    if n_aorta_parts == 1:
        cl_file = f"{cl_folder}aorta_centerline.vtp"
        img_straight_name = f"{segm_folder}straight_aorta_img.nii.gz"
        if os.path.exists(img_straight_name):
            if verbose:
                print(f"{img_straight_name} already exists - skipping")
            return True

        cl_in = vtk.vtkXMLPolyDataReader()
        cl_in.SetFileName(cl_file)
        cl_in.Update()
        cl = cl_in.GetOutput()

        # For scan type 2, the aorta is not coupled with LV
        if scan_type == "2":
            if use_raw_segmentations:
                label_name = f"{segm_folder}aorta_lumen_hires_raw.nii.gz"
                label_straight_name = f"{segm_folder}straight_aorta_label_ts_org.nii.gz"
            else:
                label_name = f"{segm_folder}aorta_lumen.nii.gz"
                label_straight_name = f"{segm_folder}straight_aorta_label.nii.gz"
        else:
            if use_raw_segmentations:
                label_name = f"{segm_folder}aorta_lumen_extended_ts_org.nii.gz"
                label_straight_name = f"{segm_folder}straight_aorta_label_ts_org.nii.gz"
            else:
                label_name = f"{segm_folder}aorta_lumen_extended.nii.gz"
                label_straight_name = f"{segm_folder}straight_aorta_label.nii.gz"

        label_img = read_nifti_with_logging_cached(
            label_name, verbose, quiet, write_log_file, output_folder
        )
        if label_img is None:
            return False

        # try:
        #     label_img = sitk.ReadImage(label_name)
        # except RuntimeError as e:
        #     msg = f"Could not read {label_name} for straigthening: {str(e)} got an exception"
        #     if not quiet:
        #         print(msg)
        #     if write_log_file:
        #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
        #     return False

        if not clutils.compute_single_straightened_volume_using_cpr(
            cl, ct_img, label_img, img_straight_name, label_straight_name, verbose
        ):
            msg = f"Error computing straightened volume using CPR for {file_name}"
            if not quiet:
                print(msg)
            if write_log_file:
                write_message_to_log_file(
                    base_dir=output_folder, message=msg, level="error"
                )
            return False
    elif n_aorta_parts == 2:
        parts = ["annulus", "descending"]
        for part in parts:
            cl_file = f"{cl_folder}aorta_centerline_{part}.vtp"
            img_straight_name = f"{segm_folder}straight_aorta_{part}_img.nii.gz"
            if os.path.exists(img_straight_name):
                if verbose:
                    print(f"{img_straight_name} already exists - skipping")
                return True

            cl_in = vtk.vtkXMLPolyDataReader()
            cl_in.SetFileName(cl_file)
            cl_in.Update()
            cl = cl_in.GetOutput()

            if use_raw_segmentations:
                label_name = f"{segm_folder}aorta_lumen_{part}_raw.nii.gz"
                if part == "annulus":
                    label_name = f"{segm_folder}aorta_lumen_extended_ts_org.nii.gz"
                label_straight_name = (
                    f"{segm_folder}straight_aorta_{part}_label_ts_org.nii.gz"
                )
            else:
                label_name = f"{segm_folder}aorta_lumen_{part}.nii.gz"
                if part == "annulus":
                    label_name = f"{segm_folder}aorta_lumen_extended.nii.gz"
                label_straight_name = f"{segm_folder}straight_aorta_{part}_label.nii.gz"

            label_img = read_nifti_with_logging_cached(
                label_name, verbose, quiet, write_log_file, output_folder
            )
            if label_img is None:
                return False
            #
            # try:
            #     label_img = sitk.ReadImage(label_name)
            # except RuntimeError as e:
            #     msg = f"Could not read {label_name} for straigthening: {str(e)} got an exception"
            #     if not quiet:
            #         print(msg)
            #     if write_log_file:
            #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
            #     return False

            if not clutils.compute_single_straightened_volume_using_cpr(
                cl, ct_img, label_img, img_straight_name, label_straight_name, verbose
            ):
                msg = f"Error computing straightened volume using CPR for {file_name}"
                if not quiet:
                    print(msg)
                if write_log_file:
                    write_message_to_log_file(
                        base_dir=output_folder, message=msg, level="error"
                    )
                return False
    return True


def compute_cuts_along_straight_labelmaps(
    segm_folder, cl_folder, stats_folder, verbose, quiet, write_log_file, output_folder
):
    """
    Finds cross-sectional cuts on the surface of the aorta sampled along the centerline.
    Here the straightened version of the label map is used.
    We use the distance along the center line as a kind of ground truth, when comparing
    the straight versus the original centerline. The renal section is for example defined using CL distances.
    """
    # debug = False

    if verbose:
        print("Sampling along straight labelmap")

    parts_stats = read_json_file(f"{stats_folder}aorta_parts.json")
    if parts_stats:
        n_aorta_parts = parts_stats["aorta_parts"]
    else:
        n_aorta_parts = 1

    if n_aorta_parts == 1:
        cl_sampling_out = f"{cl_folder}straight_labelmap_sampling.csv"
        ventri_in = f"{cl_folder}ventricularaortic.json"
        straight_label_in = f"{segm_folder}straight_aorta_label.nii.gz"
        straight_volume_in = f"{segm_folder}straight_aorta_img.nii.gz"

        if os.path.exists(cl_sampling_out):
            if verbose:
                print(f"{cl_sampling_out} already exists - skipping")
            return True

        if not os.path.exists(straight_label_in) or not os.path.exists(
            straight_volume_in
        ):
            msg = f"Missing straightened files {straight_label_in} or {straight_volume_in} for sampling"
            if not quiet:
                print(msg)
            if write_log_file:
                write_message_to_log_file(
                    base_dir=output_folder, message=msg, level="error"
                )
            return False

        if verbose:
            print(f"Computing {cl_sampling_out}")

        ventri_max_dist = np.inf
        if os.path.exists(ventri_in):
            if verbose:
                print("Using ventricularoaoartic point as last point")
            try:
                with open(ventri_in, "r") as openfile:
                    ventri_stats = json.load(openfile)
            except IOError as e:
                print(f"I/O error({e.errno}): {e.strerror}: {ventri_in}")
                return False

            # We add a 1 cm buffer
            ventri_max_dist = ventri_stats["ventri_cl_dist"] + 10

        if not clutils.sample_along_single_straight_labelmap(
            straight_label_in,
            straight_volume_in,
            cl_sampling_out,
            min_cl_dist=0,
            max_cl_dist=ventri_max_dist,
            verbose=verbose,
        ):
            msg = f"Could not sample along CPR for {straight_label_in}"
            if not quiet:
                print(msg)
            if write_log_file:
                write_message_to_log_file(
                    base_dir=output_folder, message=msg, level="error"
                )
            return False
    if n_aorta_parts == 2:
        cl_sampling_out = f"{cl_folder}straight_labelmap_sampling_annulus.csv"
        ventri_in = f"{cl_folder}ventricularaortic.json"
        straight_label_in = f"{segm_folder}straight_aorta_annulus_label.nii.gz"
        straight_volume_in = f"{segm_folder}straight_aorta_annulus_img.nii.gz"

        ventri_max_dist = np.inf
        if os.path.exists(ventri_in):
            if verbose:
                print("Using ventricularoaoartic point as last point")
            try:
                with open(ventri_in, "r") as openfile:
                    ventri_stats = json.load(openfile)
            except IOError as e:
                print(f"I/O error({e.errno}): {e.strerror}: {ventri_in}")
                return False

            # We add a 1 cm buffer
            ventri_max_dist = ventri_stats["ventri_cl_dist"] + 10

        if not clutils.sample_along_single_straight_labelmap(
            straight_label_in,
            straight_volume_in,
            cl_sampling_out,
            min_cl_dist=0,
            max_cl_dist=ventri_max_dist,
        ):
            msg = f"Could not sample along CPR for {straight_label_in}"
            if not quiet:
                print(msg)
            if write_log_file:
                write_message_to_log_file(
                    base_dir=output_folder, message=msg, level="error"
                )
            return False

        cl_sampling_out = f"{cl_folder}straight_labelmap_sampling_descending.csv"
        straight_label_in = f"{segm_folder}straight_aorta_descending_label.nii.gz"
        straight_volume_in = f"{segm_folder}straight_aorta_descending_img.nii.gz"

        if not clutils.sample_along_single_straight_labelmap(
            straight_label_in,
            straight_volume_in,
            cl_sampling_out,
            min_cl_dist=0,
            max_cl_dist=ventri_max_dist,
        ):
            msg = f"Could not sample along CPR for {straight_label_in}"
            if not quiet:
                print(msg)
            if write_log_file:
                write_message_to_log_file(
                    base_dir=output_folder, message=msg, level="error"
                )
            return False

    return True


def compute_sinutubular_junction_and_sinus_of_valsalva_from_max_and_min_cut_areas(
    cl_folder, lm_folder, stats_folder, verbose, quiet, write_log_file, output_folder
):
    """
    Compute the position of the sinutubular junction from the centerline cross areas
    """
    ventri_in = f"{cl_folder}ventricularaortic.json"
    # aortic_arch_in = f"{cl_dir}/aortic_arch.json"
    sinotubular_stats_out = f"{cl_folder}sinotubular_junction.json"
    valsalva_stats_out = f"{cl_folder}sinus_of_valsalva.json"
    sinotubular_point_out = f"{lm_folder}sinotubular_junction_on_centerline.txt"
    valsalva_point_out = f"{lm_folder}sinus_of_valsalva_on_centerline.txt"
    stats_file = f"{stats_folder}aorta_scan_type.json"
    # debug = False

    scan_type_stats = read_json_file(stats_file)
    if not scan_type_stats:
        msg = f"Could not read {stats_file}"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    scan_type = scan_type_stats["scan_type"]
    if scan_type not in ["1", "1b", "1c", "1d", "5"]:
        msg = f"We can not compute sinotubular junction and valsalva for scan type {scan_type}"
        if verbose:
            print(msg)
        return True

    n_aorta_parts = 1
    parts_stats = read_json_file(f"{stats_folder}aorta_parts.json")
    if parts_stats:
        n_aorta_parts = parts_stats["aorta_parts"]

    if n_aorta_parts == 2:
        cl_name = f"{cl_folder}aorta_centerline_annulus.vtp"
        cl_sampling_in = f"{cl_folder}straight_labelmap_sampling_annulus.csv"
    else:
        cl_name = f"{cl_folder}aorta_centerline.vtp"
        cl_sampling_in = f"{cl_folder}straight_labelmap_sampling.csv"

    if os.path.exists(sinotubular_stats_out):
        if verbose:
            print(f"{sinotubular_stats_out} already exists - skipping")
        return True

    if verbose:
        print(f"Computing {sinotubular_stats_out} and {valsalva_stats_out}")
    if not os.path.exists(cl_sampling_in):
        msg = f"Missing {cl_sampling_in} for sinutubular junction and valsalva"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    cl_sampling = np.loadtxt(cl_sampling_in, delimiter=",")
    if len(cl_sampling) < 1:
        msg = f"No samples in {cl_sampling_in} for sinutubular junction and valsalva"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    ventri_stats = read_json_file(ventri_in)
    if ventri_stats:
        if verbose:
            print("Using ventricularoaoartic point as last point")
        ventri_dist = ventri_stats["ventri_cl_dist"]
    else:
        msg = f"Could not read {ventri_in} for sinutubular junction and valsalva. It is needed."
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="warning"
            )
        return True

    cl_reader = vtk.vtkXMLPolyDataReader()
    cl_reader.SetFileName(cl_name)
    cl_reader.Update()
    cl = cl_reader.GetOutput()
    if cl.GetNumberOfPoints() < 10:
        msg = f"Centerline has less than 10 points in {cl_name}"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    n_slices = len(cl_sampling)

    # First find sinus of valsalva that is the maximum cut close to the annulus
    # Search length from ventri point the cl distances get smaller the going from the annulus along the aorta
    search_length = 20
    areas = []
    dists = []
    org_idx = []
    for idx in range(n_slices - 1, 0, -1):
        cl_dist = cl_sampling[idx][0]
        cl_area = cl_sampling[idx][1]
        out_of_scan_percent = cl_sampling[idx][2]
        if (
            out_of_scan_percent < 20
            and ventri_dist - search_length < cl_dist < ventri_dist
        ):
            areas.append(cl_area)
            dists.append(cl_dist)
            org_idx.append(idx)

    if len(areas) < 4:
        msg = "Can not find sinus of valsalva - probably to close to border of scan"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="warning"
            )
        return True

    max_idx = np.argmax(areas)
    valsalva_dist = dists[max_idx]
    valsalva_area = areas[max_idx]
    valsalva_cl_idx = org_idx[max_idx]

    cl_true_idx, _ = clutils.find_position_on_centerline_based_on_scalar(
        cl, valsalva_dist
    )
    if cl_true_idx is None:
        msg = "Could not find sinus of valsalva on centerline. Something wrong."
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return True

    valsalva_point = cl.GetPoint(cl_true_idx)
    f_p_out = open(valsalva_point_out, "w")
    f_p_out.write(f"{valsalva_point[0]} {valsalva_point[1]} {valsalva_point[2]}")
    f_p_out.close()

    valsalva_stats = {
        "valsalva_cl_dist": valsalva_dist,
        "valsalva_area": valsalva_area,
        "valsalva_cl_idx": valsalva_cl_idx,
    }
    json_object = json.dumps(valsalva_stats, indent=4)
    with open(valsalva_stats_out, "w") as outfile:
        outfile.write(json_object)

    # Now find sinotubular junction. This is further along the aorta than the sinus of valsalva.
    # we set a minimum distance of 0.5 cm
    search_length = 30
    min_dist = 5
    areas = []
    dists = []
    org_idx = []
    for idx in range(n_slices - 1, 0, -1):
        cl_dist = cl_sampling[idx][0]
        cl_area = cl_sampling[idx][1]
        out_of_scan_percent = cl_sampling[idx][2]
        if (
            out_of_scan_percent < 20
            and valsalva_dist - search_length < cl_dist < valsalva_dist - min_dist
        ):
            areas.append(cl_area)
            dists.append(cl_dist)
            org_idx.append(idx)

    if len(areas) < 2:
        msg = "Can not find sinotubular junction. Probably out of scan."
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="warning"
            )
        return True

    min_idx = np.argmin(areas)
    sinotubular_dist = dists[min_idx]
    sinotubular_area = areas[min_idx]
    sinotubular_idx = org_idx[min_idx]

    cl_true_idx, _ = clutils.find_position_on_centerline_based_on_scalar(
        cl, sinotubular_dist
    )
    if cl_true_idx is None:
        msg = "Could not find sinotubular junction on centerline. Something wrong."
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return True

    sinotubular_point = cl.GetPoint(cl_true_idx)
    f_p_out = open(sinotubular_point_out, "w")
    f_p_out.write(
        f"{sinotubular_point[0]} {sinotubular_point[1]} {sinotubular_point[2]}"
    )

    sinotubular_stats = {
        "sinotubular_cl_dist": sinotubular_dist,
        "sinotubular_area": sinotubular_area,
        "sinotubular_idx": sinotubular_idx,
    }
    json_object = json.dumps(sinotubular_stats, indent=4)
    with open(sinotubular_stats_out, "w") as outfile:
        outfile.write(json_object)

    return True


def identy_and_extract_samples_from_straight_volume_2_parts_aorta(
    cl_folder,
    segm_folder,
    verbose,
    quiet,
    write_log_file,
    output_folder,
    use_raw_segmentations=False,
):
    """
    Based on the sampled straight volume, we here extract maximum slices and extract their image.
    This function is for when the aorta is split in two
    """
    ext = ""
    if use_raw_segmentations:
        ext = "_ts_original"

    cl_sampling_in_annulus = f"{cl_folder}straight_labelmap_sampling_annulus.csv"
    cl_file_annulus = f"{cl_folder}aorta_centerline_annulus.vtp"
    cl_sampling_in_descending = f"{cl_folder}straight_labelmap_sampling_descending.csv"
    cl_file_descending = f"{cl_folder}aorta_centerline_descending.vtp"
    infrarenal_in = f"{cl_folder}infrarenal_section.json"
    ventri_in = f"{cl_folder}ventricularaortic.json"
    diaphragm_in = f"{cl_folder}diaphragm.json"
    aortic_arch_in = f"{cl_folder}aortic_arch.json"
    sinotubular_in = f"{cl_folder}sinotubular_junction.json"
    valsalva_in = f"{cl_folder}sinus_of_valsalva.json"

    cl_hu_stats_file = f"{cl_folder}aorta_centerline_hu_stats.json"

    straight_vol_in_annulus = f"{segm_folder}straight_aorta_annulus_img.nii.gz"
    straight_label_in_annulus = f"{segm_folder}straight_aorta_annulus_label{ext}.nii.gz"
    straight_vol_in_descending = f"{segm_folder}straight_aorta_descending_img.nii.gz"
    straight_label_in_descending = (
        f"{segm_folder}straight_aorta_descending_label{ext}.nii.gz"
    )

    debug = False

    check_file = f"{cl_folder}infrarenal_segment{ext}_max_slice_rgb_crop.png"
    check_file_2 = f"{cl_folder}ascending_segment{ext}_max_slice_rgb.png"

    if os.path.exists(check_file) or os.path.exists(check_file_2):
        if verbose:
            print(f"{check_file} already exists - skipping")
        return True

    # TODO: Extend check
    if not os.path.exists(straight_label_in_annulus):
        print(f"Missing files for sampling {straight_label_in_annulus}")
        msg = f"Missing files for sampling {straight_label_in_annulus}"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    if verbose:
        print(f"Identify and sample cuts using {straight_label_in_annulus}")

    # Use original centerline
    pd = vtk.vtkXMLPolyDataReader()
    pd.SetFileName(cl_file_descending)
    pd.Update()
    cl_descending = pd.GetOutput()
    if cl_descending.GetNumberOfPoints() < 2:
        msg = f"Centerline {cl_descending} is not good"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    pd_2 = vtk.vtkXMLPolyDataReader()
    pd_2.SetFileName(cl_file_annulus)
    pd_2.Update()
    cl_annulus = pd_2.GetOutput()
    if cl_annulus.GetNumberOfPoints() < 2:
        msg = f"Centerline {cl_annulus} is not good"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    label_img_annulus = read_nifti_with_logging_cached(
        straight_label_in_annulus, verbose, quiet, write_log_file, output_folder
    )
    if label_img_annulus is None:
        return False

    # try:
    #     label_img_annulus = sitk.ReadImage(straight_label_in_annulus)
    # except RuntimeError as e:
    #     msg = f"Could not read {straight_label_in_annulus}: Got an exception {str(e)}"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     return False

    straight_img_annulus = read_nifti_with_logging_cached(
        straight_vol_in_annulus, verbose, quiet, write_log_file, output_folder
    )
    if straight_img_annulus is None:
        return False

    # try:
    #     straight_img_annulus = sitk.ReadImage(straight_vol_in_annulus)
    # except RuntimeError as e:
    #     msg = f"Could not read {straight_vol_in_annulus}: Got an exception {str(e)}"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     return False

    straight_img_annulus_np = sitk.GetArrayFromImage(straight_img_annulus)
    straight_img_annulus_np = straight_img_annulus_np.transpose(2, 1, 0)

    label_img_annulus_np = sitk.GetArrayFromImage(label_img_annulus)
    label_img_annulus_np = label_img_annulus_np.transpose(2, 1, 0)

    label_img_descending = read_nifti_with_logging_cached(
        straight_label_in_descending, verbose, quiet, write_log_file, output_folder
    )
    if label_img_descending is None:
        return False
    #
    # try:
    #     label_img_descending = sitk.ReadImage(straight_label_in_descending)
    # except RuntimeError as e:
    #     msg = f"Could not read {straight_label_in_descending}: Got an exception {str(e)}"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     return False

    straight_img_descending = read_nifti_with_logging_cached(
        straight_vol_in_descending, verbose, quiet, write_log_file, output_folder
    )
    if straight_img_descending is None:
        return False

    # try:
    #     straight_img_descending = sitk.ReadImage(straight_vol_in_descending)
    # except RuntimeError as e:
    #     msg = f"Could not read {straight_vol_in_descending}: Got an exception {str(e)}"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     return False

    straight_img_descending_np = sitk.GetArrayFromImage(straight_img_descending)
    straight_img_descending_np = straight_img_descending_np.transpose(2, 1, 0)

    label_img_descending_np = sitk.GetArrayFromImage(label_img_descending)
    label_img_descending_np = label_img_descending_np.transpose(2, 1, 0)

    dims = label_img_annulus_np.shape
    spacing = label_img_annulus.GetSpacing()
    # n_slices = dims[2]
    # max_distance = spacing[2] * n_slices

    if not os.path.exists(cl_sampling_in_annulus):
        msg = f"Missing {cl_sampling_in_annulus} for sampling"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False
    cl_sampling_annulus = np.loadtxt(cl_sampling_in_annulus, delimiter=",")

    if not os.path.exists(cl_sampling_in_descending):
        msg = f"Missing {cl_sampling_in_descending} for sampling"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False
    cl_sampling_descending = np.loadtxt(cl_sampling_in_descending, delimiter=",")

    # Defaults
    img_window = 200
    img_level = 200

    cl_stats = read_json_file(cl_hu_stats_file)
    if cl_stats:
        img_window = cl_stats["img_window"]
        img_level = cl_stats["img_level"]
        if verbose:
            print(
                f"Using centerline HU stats for window {img_window:.2f} and level {img_level:.2f}"
            )

    min_renal_dist = -np.inf
    max_renal_dist = np.inf
    infrarenal_stats = read_json_file(infrarenal_in)
    if infrarenal_stats:
        if debug:
            print("Using infra-renal section")
        min_renal_dist = infrarenal_stats["low_dist"]
        max_renal_dist = infrarenal_stats["distance"]

    ventri_max_dist = np.inf
    ventri_stats = read_json_file(ventri_in)
    if ventri_stats:
        if verbose:
            print("Using ventricularoaoartic point as last point")
        ventri_max_dist = ventri_stats["ventri_cl_dist"]

    diaphragm_dist = np.inf
    diaphragm_stats = read_json_file(diaphragm_in)
    if diaphragm_stats:
        if verbose:
            print("Using diaphragm info")
        diaphragm_dist = diaphragm_stats["diaphragm_cl_dist"]

    aortic_arch_min_dist = -np.inf
    aortic_arch_max_dist = np.inf
    aortic_arch_stats = read_json_file(aortic_arch_in)
    if aortic_arch_stats:
        if verbose:
            print("Using aortic arch info")
        aortic_arch_min_dist = aortic_arch_stats["min_cl_dist"]
        aortic_arch_max_dist = aortic_arch_stats["max_cl_dist"]

    valsalva_dist = np.inf
    valsalva_stats = read_json_file(valsalva_in)
    if valsalva_stats:
        if verbose:
            print("Using valsalva info")
        valsalva_dist = valsalva_stats["valsalva_cl_dist"]

    sinotubular_dist = np.inf
    sinotubular_stats = read_json_file(sinotubular_in)
    if sinotubular_stats:
        if verbose:
            print("Using sinutubular  info")
        sinotubular_dist = sinotubular_stats["sinotubular_cl_dist"]

    # max_distance_annulus = cl_annulus.GetPointData().GetScalars().GetValue(cl_annulus.GetNumberOfPoints() - 1)
    max_distance_descending = (
        cl_descending.GetPointData()
        .GetScalars()
        .GetValue(cl_descending.GetNumberOfPoints() - 1)
    )

    segment_name = f"infrarenal_segment{ext}"
    start_cl_dist = min_renal_dist
    end_cl_dist = max_renal_dist
    result, msg = clutils.extract_max_cut_in_defined_section(
        cl_folder,
        segment_name,
        start_cl_dist,
        end_cl_dist,
        cl_sampling_descending,
        cl_descending,
        straight_img_descending_np,
        label_img_descending_np,
        img_window,
        img_level,
        spacing,
        dims,
    )
    if not result:
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="warning"
            )

    segment_name = f"abdominal_segment{ext}"
    start_cl_dist = max_renal_dist
    end_cl_dist = diaphragm_dist
    result, msg = clutils.extract_max_cut_in_defined_section(
        cl_folder,
        segment_name,
        start_cl_dist,
        end_cl_dist,
        cl_sampling_descending,
        cl_descending,
        straight_img_descending_np,
        label_img_descending_np,
        img_window,
        img_level,
        spacing,
        dims,
    )
    if not result:
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="warning"
            )

    segment_name = f"descending_segment{ext}"
    start_cl_dist = diaphragm_dist
    if start_cl_dist == np.inf or start_cl_dist == -np.inf:
        start_cl_dist = 0
    end_cl_dist = max_distance_descending
    result, msg = clutils.extract_max_cut_in_defined_section(
        cl_folder,
        segment_name,
        start_cl_dist,
        end_cl_dist,
        cl_sampling_descending,
        cl_descending,
        straight_img_descending_np,
        label_img_descending_np,
        img_window,
        img_level,
        spacing,
        dims,
    )
    if not result:
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="warning"
            )

    segment_name = f"aortic_arch_segment{ext}"
    start_cl_dist = aortic_arch_min_dist
    end_cl_dist = aortic_arch_max_dist
    result, msg = clutils.extract_max_cut_in_defined_section(
        cl_folder,
        segment_name,
        start_cl_dist,
        end_cl_dist,
        cl_sampling_descending,
        cl_descending,
        straight_img_descending_np,
        label_img_descending_np,
        img_window,
        img_level,
        spacing,
        dims,
    )
    if not result:
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="warning"
            )

    segment_name = f"ascending_segment{ext}"
    # start_cl_dist = aortic_arch_max_dist
    start_cl_dist = 0
    end_cl_dist = sinotubular_dist
    result, msg = clutils.extract_max_cut_in_defined_section(
        cl_folder,
        segment_name,
        start_cl_dist,
        end_cl_dist,
        cl_sampling_annulus,
        cl_annulus,
        straight_img_annulus_np,
        label_img_annulus_np,
        img_window,
        img_level,
        spacing,
        dims,
    )
    if not result:
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="warning"
            )

    segment_name = f"sinotubular_junction_segment{ext}"
    start_cl_dist = sinotubular_dist - 2
    end_cl_dist = sinotubular_dist + 2
    result, msg = clutils.extract_max_cut_in_defined_section(
        cl_folder,
        segment_name,
        start_cl_dist,
        end_cl_dist,
        cl_sampling_annulus,
        cl_annulus,
        straight_img_annulus_np,
        label_img_annulus_np,
        img_window,
        img_level,
        spacing,
        dims,
        find_minimum=True,
    )
    if not result:
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="warning"
            )

    segment_name = f"sinus_of_valsalva_segment{ext}"
    start_cl_dist = valsalva_dist - 2
    end_cl_dist = valsalva_dist + 2
    result, msg = clutils.extract_max_cut_in_defined_section(
        cl_folder,
        segment_name,
        start_cl_dist,
        end_cl_dist,
        cl_sampling_annulus,
        cl_annulus,
        straight_img_annulus_np,
        label_img_annulus_np,
        img_window,
        img_level,
        spacing,
        dims,
    )
    if not result:
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="warning"
            )

    segment_name = f"lvot_segment{ext}"
    start_cl_dist = ventri_max_dist - 1
    end_cl_dist = ventri_max_dist + 10
    result, msg = clutils.extract_max_cut_in_defined_section(
        cl_folder,
        segment_name,
        start_cl_dist,
        end_cl_dist,
        cl_sampling_annulus,
        cl_annulus,
        straight_img_annulus_np,
        label_img_annulus_np,
        img_window,
        img_level,
        spacing,
        dims,
        find_minimum=True,
    )
    if not result:
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="warning"
            )

    return True


def identy_and_extract_samples_from_straight_volume(
    cl_folder,
    segm_folder,
    stats_folder,
    verbose,
    quiet,
    write_log_file,
    output_folder,
    use_raw_segmentations=False,
):
    """
    Based on the sampled straight volume, we here extract maximum slices and extract their image
    """
    # debug = False
    n_aorta_parts = 1
    parts_stats = read_json_file(f"{stats_folder}/aorta_parts.json")
    if parts_stats:
        n_aorta_parts = parts_stats["aorta_parts"]

    if n_aorta_parts == 2:
        return identy_and_extract_samples_from_straight_volume_2_parts_aorta(
            cl_folder,
            segm_folder,
            verbose,
            quiet,
            write_log_file,
            output_folder,
            use_raw_segmentations,
        )

    cl_sampling_in = f"{cl_folder}straight_labelmap_sampling.csv"
    cl_file = f"{cl_folder}aorta_centerline.vtp"
    infrarenal_in = f"{cl_folder}infrarenal_section.json"
    ventri_in = f"{cl_folder}ventricularaortic.json"
    diaphragm_in = f"{cl_folder}diaphragm.json"
    aortic_arch_in = f"{cl_folder}aortic_arch.json"
    sinotubular_in = f"{cl_folder}sinotubular_junction.json"
    valsalva_in = f"{cl_folder}sinus_of_valsalva.json"
    # distensability_in = f"{cl_folder}/aortic_distensability_point.json"

    cl_hu_stats_file = f"{cl_folder}aorta_centerline_hu_stats.json"

    ext = ""
    if use_raw_segmentations:
        ext = "_ts_original"

    straight_vol_in = f"{segm_folder}straight_aorta_img.nii.gz"
    straight_label_in = f"{segm_folder}straight_aorta_label{ext}.nii.gz"
    check_file = f"{cl_folder}infrarenal_segment{ext}_max_slice_rgb_crop.png"
    check_file_2 = f"{cl_folder}ascending_segment{ext}_max_slice_rgb.png"

    if os.path.exists(check_file):
        if verbose:
            print(f"{check_file} already exists - skipping")
        return True

    if os.path.exists(check_file_2):
        if verbose:
            print(f"{check_file_2} already exists - skipping")
        return True

    # TODO: Extend check
    if not os.path.exists(straight_label_in):
        msg = f"Missing files for sampling {straight_label_in}"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    if verbose:
        print(f"Identify and sample cuts using {straight_label_in}")

    # Use original centerline
    pd = vtk.vtkXMLPolyDataReader()
    pd.SetFileName(cl_file)
    pd.Update()
    cl = pd.GetOutput()
    if cl.GetNumberOfPoints() < 10:
        msg = f"Centerline is not good in {cl_file}"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    label_img = read_nifti_with_logging_cached(
        straight_label_in, verbose, quiet, write_log_file, output_folder
    )
    if label_img is None:
        return False

    # try:
    #     label_img = sitk.ReadImage(straight_label_in)
    # except RuntimeError as e:
    #     msg = f"Got an exception {str(e)} reading {straight_label_in}"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     return False

    straight_img = read_nifti_with_logging_cached(
        straight_vol_in, verbose, quiet, write_log_file, output_folder
    )
    if straight_img is None:
        return False

    # try:
    #     straight_img = sitk.ReadImage(straight_vol_in)
    # except RuntimeError as e:
    #     msg = f"Got an exception {str(e)} reading {straight_vol_in}"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     return False

    straight_img_np = sitk.GetArrayFromImage(straight_img)
    straight_img_np = straight_img_np.transpose(2, 1, 0)

    label_img_np = sitk.GetArrayFromImage(label_img)
    label_img_np = label_img_np.transpose(2, 1, 0)
    dims = label_img_np.shape

    spacing = label_img.GetSpacing()
    # n_slices = dims[2]
    # max_cl_dist = spacing * n_slices

    if not os.path.exists(cl_sampling_in):
        msg = f"Could not read {cl_sampling_in}"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    cl_sampling = np.loadtxt(cl_sampling_in, delimiter=",")

    # Defaults
    img_window = 200
    img_level = 200

    cl_stats = read_json_file(cl_hu_stats_file)
    if cl_stats:
        img_window = cl_stats["img_window"]
        img_level = cl_stats["img_level"]
        if verbose:
            print(
                f"Using centerline HU stats for window {img_window:.2f} and level {img_level:.2f}"
            )

    max_cl_distance = (
        cl.GetPointData().GetScalars().GetValue(cl.GetNumberOfPoints() - 1)
    )

    min_renal_dist = -np.inf
    max_renal_dist = np.inf
    infrarenal_stats = read_json_file(infrarenal_in)
    if infrarenal_stats:
        if verbose:
            print("Using infra-renal section")
        min_renal_dist = infrarenal_stats["low_dist"]
        max_renal_dist = infrarenal_stats["distance"]

    ventri_max_dist = np.inf
    ventri_stats = read_json_file(ventri_in)
    if ventri_stats:
        if verbose:
            print("Using ventricularoaoartic point as last point")
        ventri_max_dist = ventri_stats["ventri_cl_dist"]

    diaphragm_dist = np.inf
    diaphragm_stats = read_json_file(diaphragm_in)
    if diaphragm_stats:
        if verbose:
            print("Using diaphragm info")
        diaphragm_dist = diaphragm_stats["diaphragm_cl_dist"]

    aortic_arch_min_dist = -np.inf
    aortic_arch_max_dist = np.inf
    aortic_arch_stats = read_json_file(aortic_arch_in)
    if aortic_arch_stats:
        if verbose:
            print("Using aortic arch info")
        aortic_arch_min_dist = aortic_arch_stats["min_cl_dist"]
        aortic_arch_max_dist = aortic_arch_stats["max_cl_dist"]

    valsalva_dist = np.inf
    valsalva_stats = read_json_file(valsalva_in)
    if valsalva_stats:
        if verbose:
            print("Using valsalva info")
        valsalva_dist = valsalva_stats["valsalva_cl_dist"]

    sinotubular_dist = np.inf
    sinotubular_stats = read_json_file(sinotubular_in)
    if sinotubular_stats:
        if verbose:
            print("Using sinutubular  info")
        sinotubular_dist = sinotubular_stats["sinotubular_cl_dist"]

    segment_name = f"infrarenal_segment{ext}"
    start_cl_dist = min_renal_dist
    end_cl_dist = max_renal_dist
    result, msg = clutils.extract_max_cut_in_defined_section(
        cl_folder,
        segment_name,
        start_cl_dist,
        end_cl_dist,
        cl_sampling,
        cl,
        straight_img_np,
        label_img_np,
        img_window,
        img_level,
        spacing,
        dims,
    )
    if not result:
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="warning"
            )

    segment_name = f"abdominal_segment{ext}"
    start_cl_dist = max_renal_dist
    end_cl_dist = diaphragm_dist
    result, msg = clutils.extract_max_cut_in_defined_section(
        cl_folder,
        segment_name,
        start_cl_dist,
        end_cl_dist,
        cl_sampling,
        cl,
        straight_img_np,
        label_img_np,
        img_window,
        img_level,
        spacing,
        dims,
    )
    if not result:
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="warning"
            )

    segment_name = f"descending_segment{ext}"
    start_cl_dist = diaphragm_dist
    end_cl_dist = aortic_arch_min_dist
    if start_cl_dist == np.inf or start_cl_dist == -np.inf:
        start_cl_dist = min_renal_dist
        if start_cl_dist == np.inf or start_cl_dist == -np.inf:
            start_cl_dist = 0
    if end_cl_dist == np.inf or end_cl_dist == -np.inf:
        end_cl_dist = max_cl_distance
    result, msg = clutils.extract_max_cut_in_defined_section(
        cl_folder,
        segment_name,
        start_cl_dist,
        end_cl_dist,
        cl_sampling,
        cl,
        straight_img_np,
        label_img_np,
        img_window,
        img_level,
        spacing,
        dims,
    )
    if not result:
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="warning"
            )

    segment_name = f"aortic_arch_segment{ext}"
    start_cl_dist = aortic_arch_min_dist
    end_cl_dist = aortic_arch_max_dist
    result, msg = clutils.extract_max_cut_in_defined_section(
        cl_folder,
        segment_name,
        start_cl_dist,
        end_cl_dist,
        cl_sampling,
        cl,
        straight_img_np,
        label_img_np,
        img_window,
        img_level,
        spacing,
        dims,
    )
    if not result:
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="warning"
            )

    segment_name = f"ascending_segment{ext}"
    start_cl_dist = aortic_arch_max_dist
    end_cl_dist = sinotubular_dist
    result, msg = clutils.extract_max_cut_in_defined_section(
        cl_folder,
        segment_name,
        start_cl_dist,
        end_cl_dist,
        cl_sampling,
        cl,
        straight_img_np,
        label_img_np,
        img_window,
        img_level,
        spacing,
        dims,
    )
    if not result:
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="warning"
            )

    # TODO: Make sure to find the minimum
    segment_name = f"sinotubular_junction_segment{ext}"
    start_cl_dist = sinotubular_dist - 2
    end_cl_dist = sinotubular_dist + 2
    result, msg = clutils.extract_max_cut_in_defined_section(
        cl_folder,
        segment_name,
        start_cl_dist,
        end_cl_dist,
        cl_sampling,
        cl,
        straight_img_np,
        label_img_np,
        img_window,
        img_level,
        spacing,
        dims,
        find_minimum=True,
    )
    if not result:
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="warning"
            )

    segment_name = f"sinus_of_valsalva_segment{ext}"
    start_cl_dist = valsalva_dist - 2
    end_cl_dist = valsalva_dist + 2
    result, msg = clutils.extract_max_cut_in_defined_section(
        cl_folder,
        segment_name,
        start_cl_dist,
        end_cl_dist,
        cl_sampling,
        cl,
        straight_img_np,
        label_img_np,
        img_window,
        img_level,
        spacing,
        dims,
    )
    if not result:
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="warning"
            )

    segment_name = f"lvot_segment{ext}"
    start_cl_dist = ventri_max_dist - 1
    end_cl_dist = ventri_max_dist + 10
    result, msg = clutils.extract_max_cut_in_defined_section(
        cl_folder,
        segment_name,
        start_cl_dist,
        end_cl_dist,
        cl_sampling,
        cl,
        straight_img_np,
        label_img_np,
        img_window,
        img_level,
        spacing,
        dims,
        find_minimum=True,
    )
    if not result:
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="warning"
            )

    return True


def combine_cross_section_images_into_one(cl_dir, verbose=False):
    """
    Combine precomputed images into one image
    """
    out_combined = f"{cl_dir}combined_cuts.png"
    if os.path.exists(out_combined):
        if verbose:
            print(f"{out_combined} already exists - skipping")
        return True

    if verbose:
        print(f"Combining cross section images into {out_combined}")

    cut_imgs = [
        {"file": "lvot_segment_max_slice_rgb_crop", "rgb": [200, 255, 100]},
        {"file": "sinus_of_valsalva_segment_max_slice_rgb_crop", "rgb": [0, 128, 255]},
        {
            "file": "sinotubular_junction_segment_max_slice_rgb_crop",
            "rgb": [128, 0, 128],
        },
        {"file": "distensability_segment_avg_slice_rgb_crop", "rgb": [255, 128, 128]},
        {"file": "ascending_segment_max_slice_rgb_crop", "rgb": [0, 255, 255]},
        {"file": "aortic_arch_segment_max_slice_rgb_crop", "rgb": [255, 0, 0]},
        {"file": "descending_segment_max_slice_rgb_crop", "rgb": [0, 255, 0]},
        {"file": "abdominal_segment_max_slice_rgb_crop", "rgb": [255, 128, 0]},
        {"file": "infrarenal_segment_max_slice_rgb_crop", "rgb": [255, 255, 0]},
    ]

    valid_imgs = []
    widths = []
    heights = []
    rgbs = []
    for idx in range(len(cut_imgs)):
        basename = cut_imgs[idx]["file"]
        img_name = f"{cl_dir}{basename}.png"
        if os.path.exists(img_name):
            img_in = skimage.io.imread(img_name)
            shp = img_in.shape
            rgb = cut_imgs[idx]["rgb"]
            if not (shp[0] == 0 or shp[1] == 0):
                valid_imgs.append(img_in)
                widths.append(shp[1])
                heights.append(shp[0])
                rgbs.append(rgb)

    if len(valid_imgs) < 1:
        if verbose:
            print("No valid cut sections found")
        return False

    spacing = 5
    max_width = max(widths)
    total_height = sum(heights) + len(valid_imgs) * spacing

    new_image = np.zeros([total_height, max_width, 3]).astype(np.uint8)
    cur_row = 0
    for idx in range(len(valid_imgs)):
        img_in = valid_imgs[idx]
        shp = img_in.shape
        side_pad = (max_width - shp[1]) // 2

        new_image[cur_row : cur_row + shp[0], 0 + side_pad : shp[1] + side_pad] = (
            img_in[0 : shp[0], 0 : shp[1]]
        )
        # Color sides
        new_image[cur_row : cur_row + shp[0], 0:3] = rgbs[idx]
        new_image[cur_row : cur_row + shp[0], max_width - 4 : max_width - 1] = rgbs[idx]
        cur_row += shp[0] + spacing

    imageio.imwrite(out_combined, new_image)
    return True


def create_two_long_slices(label_img_np, straight_img_np, dims, img_window, img_level):
    # dims = label_img_np.shape
    mid_id = dims[0] // 2
    single_slice_np = label_img_np[mid_id, :, :]

    mid_id_2 = dims[1] // 2
    single_slice_np_2 = label_img_np[:, mid_id_2, :]

    #  only keep one connected component in the slice. The one that is in the middle
    slice_components = label(single_slice_np)
    mid_label = slice_components[dims[1] // 2, dims[2] // 2]
    largest_cc = slice_components == mid_label

    slice_components = label(single_slice_np_2)
    mid_label = slice_components[dims[1] // 2, dims[2] // 2]
    largest_cc_2 = slice_components == mid_label

    # contour = find_contours(img_as_ubyte(largest_cc), 0.5)[0]
    boundary = find_boundaries(largest_cc, mode="thick")
    boundary_2 = find_boundaries(largest_cc_2, mode="thick")

    # skimage.io.imsave(max_slice_boundary_out, boundary)
    single_slice_np_img = straight_img_np[mid_id, :, :]
    single_slice_np_img_2 = straight_img_np[:, mid_id_2, :]

    single_slice_np_img = clutils.set_window_and_level_on_single_slice(
        single_slice_np_img, img_window, img_level
    )
    single_slice_np_img_2 = clutils.set_window_and_level_on_single_slice(
        single_slice_np_img_2, img_window, img_level
    )
    scaled_ubyte = img_as_ubyte(single_slice_np_img)
    scaled_ubyte_2 = img_as_ubyte(single_slice_np_img_2)
    # skimage.io.imsave(max_slice_out, scaled_ubyte)

    scaled_2_rgb = color.gray2rgb(scaled_ubyte)
    scaled_2_rgb_2 = color.gray2rgb(scaled_ubyte_2)
    rgb_boundary = [255, 0, 0]

    # Draw boundary last for visual style
    scaled_2_rgb[boundary > 0] = rgb_boundary
    scaled_2_rgb_2[boundary_2 > 0] = rgb_boundary

    return scaled_2_rgb, scaled_2_rgb_2


def create_longitudinal_figure_from_straight_volume_from_2_part_aort(
    cl_folder, segm_folder, verbose, quiet, write_log_file, output_folder
):
    infrarenal_in = f"{cl_folder}infrarenal_section.json"
    ventri_in = f"{cl_folder}ventricularaortic.json"
    diaphragm_in = f"{cl_folder}diaphragm.json"
    aortic_arch_in = f"{cl_folder}aortic_arch.json"

    infrarenal_max_in = f"{cl_folder}infrarenal_segment_max_slice_info.json"
    abdominal_max_in = f"{cl_folder}abdominal_segment_max_slice_info.json"
    aortic_arch_max_in = f"{cl_folder}aortic_arch_segment_max_slice_info.json"
    ascending_max_in = f"{cl_folder}ascending_segment_max_slice_info.json"
    distensability_avg_in = f"{cl_folder}distensability_segment_avg_slice_info.json"
    sinotubular_max_in = f"{cl_folder}sinotubular_junction_segment_max_slice_info.json"
    sinus_of_valsalva_max_in = (
        f"{cl_folder}sinus_of_valsalva_segment_max_slice_info.json"
    )
    lvot_max_in = f"{cl_folder}lvot_segment_max_slice_info.json"
    descending_max_in = f"{cl_folder}descending_segment_max_slice_info.json"

    cl_hu_stats_file = f"{cl_folder}aorta_centerline_hu_stats.json"

    straight_vol_in_annulus = f"{segm_folder}straight_aorta_annulus_img.nii.gz"
    straight_label_in_annulus = f"{segm_folder}straight_aorta_annulus_label.nii.gz"
    out_file_annulus = f"{cl_folder}annulus_straight_volume_mid_cut.png"
    out_file_2_annulus = f"{cl_folder}annulus_straight_volume_mid_cut_2.png"

    straight_vol_in_descending = f"{segm_folder}straight_aorta_descending_img.nii.gz"
    straight_label_in_descending = (
        f"{segm_folder}straight_aorta_descending_label.nii.gz"
    )
    out_file_descending = f"{cl_folder}descending_straight_volume_mid_cut.png"
    out_file_2_descending = f"{cl_folder}descending_straight_volume_mid_cut_2.png"

    out_file_combined_1 = f"{cl_folder}straight_volume_mid_cut.png"
    out_file_combined_2 = f"{cl_folder}straight_volume_mid_cut_2.png"
    # debug = False

    if os.path.exists(out_file_combined_1):
        if verbose:
            print(f"{out_file_combined_1} already exists - skipping")
        return True

    if not os.path.exists(straight_label_in_annulus):
        msg = f"Missing files for sampling {straight_label_in_annulus}"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    if verbose:
        print(
            f"Creating long figure axis using {straight_label_in_annulus} and {straight_label_in_descending}"
        )

    label_img_annulus = read_nifti_with_logging_cached(
        straight_label_in_annulus, verbose, quiet, write_log_file, output_folder
    )
    if label_img_annulus is None:
        return False
    #
    # try:
    #     label_img_annulus = sitk.ReadImage(straight_label_in_annulus)
    # except RuntimeError as e:
    #     msg = f"Got an exception {str(e)}"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     return False

    straight_img_annulus = read_nifti_with_logging_cached(
        straight_vol_in_annulus, verbose, quiet, write_log_file, output_folder
    )
    if straight_img_annulus is None:
        return False

    # try:
    #     straight_img_annulus = sitk.ReadImage(straight_vol_in_annulus)
    # except RuntimeError as e:
    #     msg = f"Got an exception {str(e)}"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     return False

    straight_img_np_annulus = sitk.GetArrayFromImage(straight_img_annulus)
    straight_img_np_annulus = straight_img_np_annulus.transpose(2, 1, 0)

    label_img_np_annulus = sitk.GetArrayFromImage(label_img_annulus)
    label_img_np_annulus = label_img_np_annulus.transpose(2, 1, 0)

    label_img_descending = read_nifti_with_logging_cached(
        straight_label_in_descending, verbose, quiet, write_log_file, output_folder
    )
    if label_img_descending is None:
        return False

    straight_img_descending = read_nifti_with_logging_cached(
        straight_vol_in_descending, verbose, quiet, write_log_file, output_folder
    )
    if straight_img_descending is None:
        return False

    # try:
    #     label_img_descending = sitk.ReadImage(straight_label_in_descending)
    # except RuntimeError as e:
    #     msg = f"Got an exception {str(e)}"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     return False

    # try:
    #     straight_img_descending = sitk.ReadImage(straight_vol_in_descending)
    # except RuntimeError as e:
    #     msg = f"Got an exception {str(e)}"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     return False

    straight_img_np_descending = sitk.GetArrayFromImage(straight_img_descending)
    straight_img_np_descending = straight_img_np_descending.transpose(2, 1, 0)

    label_img_np_descending = sitk.GetArrayFromImage(label_img_descending)
    label_img_np_descending = label_img_np_descending.transpose(2, 1, 0)

    dims_annulus = label_img_np_annulus.shape
    dims_descending = label_img_np_descending.shape
    spacing = label_img_annulus.GetSpacing()
    # n_slices = dims[2]

    # Defaults
    img_window = 200
    img_level = 200

    cl_stats = read_json_file(cl_hu_stats_file)
    if cl_stats:
        if verbose:
            print("Using centerline HU stats")
        img_window = cl_stats["img_window"]
        img_level = cl_stats["img_level"]

    single_slice_annulus_1, single_slice_annulus_2 = create_two_long_slices(
        label_img_np_annulus,
        straight_img_np_annulus,
        dims_annulus,
        img_window,
        img_level,
    )

    single_slice_descending_1, single_slice_descending_2 = create_two_long_slices(
        label_img_np_descending,
        straight_img_np_descending,
        dims_descending,
        img_window,
        img_level,
    )

    line_stats = []
    infrarenal_stats = read_json_file(infrarenal_in)
    if infrarenal_stats:
        line_stats.append(
            {
                "part": "descending",
                "name": "infrarenal_low",
                "dist": infrarenal_stats["low_dist"],
                "rgb": [255, 255, 255],
            }
        )
        line_stats.append(
            {
                "part": "descending",
                "name": "infrarenal",
                "dist": infrarenal_stats["distance"],
                "rgb": [255, 255, 255],
            }
        )

    ventri_stats = read_json_file(ventri_in)
    if ventri_stats:
        line_stats.append(
            {
                "part": "annulus",
                "name": "ventricularoaoartic",
                "dist": ventri_stats["ventri_cl_dist"],
                "rgb": [255, 255, 255],
            }
        )

    diaphragm_stats = read_json_file(diaphragm_in)
    if diaphragm_stats:
        line_stats.append(
            {
                "part": "descending",
                "name": "diaphragm",
                "dist": diaphragm_stats["diaphragm_cl_dist"],
                "rgb": [255, 255, 255],
            }
        )

    aortic_arch_stats = read_json_file(aortic_arch_in)
    if aortic_arch_stats:
        line_stats.append(
            {
                "part": "annulus",
                "name": "arch_start",
                "dist": aortic_arch_stats["min_cl_dist"],
                "rgb": [255, 255, 255],
            }
        )
        line_stats.append(
            {
                "part": "annulus",
                "name": "arch_end",
                "dist": aortic_arch_stats["max_cl_dist"],
                "rgb": [255, 255, 255],
            }
        )

    max_stats = read_json_file(infrarenal_max_in)
    if max_stats:
        line_stats.append(
            {
                "part": "descending",
                "name": "infrarenal_max",
                "dist": max_stats["cl_dist"],
                "rgb": [255, 255, 0],
            }
        )
    max_stats = read_json_file(abdominal_max_in)
    if max_stats:
        line_stats.append(
            {
                "part": "descending",
                "name": "abdominal_max_in",
                "dist": max_stats["cl_dist"],
                "rgb": [255, 128, 0],
            }
        )
    max_stats = read_json_file(aortic_arch_max_in)
    if max_stats:
        line_stats.append(
            {
                "part": "annulus",
                "name": "aortic_arch_max_in",
                "dist": max_stats["cl_dist"],
                "rgb": [255, 0, 0],
            }
        )
    max_stats = read_json_file(ascending_max_in)
    if max_stats:
        line_stats.append(
            {
                "part": "annulus",
                "name": "ascending_max_in",
                "dist": max_stats["cl_dist"],
                "rgb": [0, 255, 255],
            }
        )
    avg_stats = read_json_file(distensability_avg_in)
    if avg_stats:
        line_stats.append(
            {
                "part": "annulus",
                "name": "distensability_avg_in",
                "dist": avg_stats["cl_dist"],
                "rgb": [255, 128, 128],
            }
        )
    max_stats = read_json_file(sinotubular_max_in)
    if max_stats:
        line_stats.append(
            {
                "part": "annulus",
                "name": "sinotubular_max_in",
                "dist": max_stats["cl_dist"],
                "rgb": [128, 0, 128],
            }
        )
    max_stats = read_json_file(sinus_of_valsalva_max_in)
    if max_stats:
        line_stats.append(
            {
                "part": "annulus",
                "name": "sinus_of_valsalva_max_in",
                "dist": max_stats["cl_dist"],
                "rgb": [0, 128, 255],
            }
        )
    max_stats = read_json_file(descending_max_in)
    if max_stats:
        line_stats.append(
            {
                "part": "descending",
                "name": "descending_max_in",
                "dist": max_stats["cl_dist"],
                "rgb": [0, 255, 0],
            }
        )
    max_stats = read_json_file(lvot_max_in)
    if max_stats:
        line_stats.append(
            {
                "part": "annulus",
                "name": "lvot_max_in",
                "dist": max_stats["cl_dist"],
                "rgb": [200, 255, 100],
            }
        )

    for lstats in line_stats:
        dist = lstats["dist"]
        rgb = lstats["rgb"]
        part = lstats["part"]

        n_straight_idx = int(dist // spacing[2])
        if part == "annulus":
            if 0 <= n_straight_idx < dims_annulus[2]:
                if n_straight_idx > 0:
                    single_slice_annulus_1[:, n_straight_idx - 1, :] = rgb
                    single_slice_annulus_1[:, n_straight_idx - 1, :] = rgb
                single_slice_annulus_1[:, n_straight_idx, :] = rgb
                if n_straight_idx < dims_annulus[2] - 1:
                    single_slice_annulus_2[:, n_straight_idx + 1, :] = rgb
                    single_slice_annulus_2[:, n_straight_idx + 1, :] = rgb
                single_slice_annulus_2[:, n_straight_idx, :] = rgb
        elif part == "descending":
            if 0 <= n_straight_idx < dims_descending[2]:
                if n_straight_idx > 0:
                    single_slice_descending_1[:, n_straight_idx - 1, :] = rgb
                    single_slice_descending_1[:, n_straight_idx - 1, :] = rgb
                single_slice_descending_1[:, n_straight_idx, :] = rgb
                if n_straight_idx < dims_descending[2] - 1:
                    single_slice_descending_2[:, n_straight_idx + 1, :] = rgb
                    single_slice_descending_2[:, n_straight_idx + 1, :] = rgb
                single_slice_descending_2[:, n_straight_idx, :] = rgb

    skimage.io.imsave(
        out_file_annulus, np.flipud(single_slice_annulus_1.transpose(1, 0, 2))
    )
    skimage.io.imsave(
        out_file_2_annulus, np.flipud(single_slice_annulus_2.transpose(1, 0, 2))
    )
    skimage.io.imsave(
        out_file_descending, np.flipud(single_slice_descending_1.transpose(1, 0, 2))
    )
    skimage.io.imsave(
        out_file_2_descending, np.flipud(single_slice_descending_2.transpose(1, 0, 2))
    )

    # Combine into two images
    single_slice_annulus_1 = np.flipud(single_slice_annulus_1.transpose(1, 0, 2))
    single_slice_descending_1 = np.flipud(single_slice_descending_1.transpose(1, 0, 2))
    shp_annulus_1 = single_slice_annulus_1.shape
    shp_descending_1 = single_slice_descending_1.shape

    spacing = 10
    new_width = max(shp_annulus_1[1], shp_descending_1[1])
    new_height = shp_annulus_1[0] + shp_descending_1[0] + spacing
    new_image = np.zeros([new_height, new_width, 3]).astype(np.uint8)
    new_image[0 : shp_annulus_1[0], 0 : shp_annulus_1[1]] = single_slice_annulus_1
    new_image[shp_annulus_1[0] + spacing :, 0 : shp_descending_1[1]] = (
        single_slice_descending_1
    )

    imageio.imwrite(out_file_combined_1, new_image)

    single_slice_annulus_2 = np.flipud(single_slice_annulus_2.transpose(1, 0, 2))
    single_slice_descending_2 = np.flipud(single_slice_descending_2.transpose(1, 0, 2))
    shp_annulus_2 = single_slice_annulus_2.shape
    shp_descending_2 = single_slice_descending_2.shape

    spacing = 10
    new_width = max(shp_annulus_2[1], shp_descending_2[1])
    new_height = shp_annulus_2[0] + shp_descending_2[0] + spacing
    new_image = np.zeros([new_height, new_width, 3]).astype(np.uint8)
    new_image[0 : shp_annulus_2[0], 0 : shp_annulus_2[1]] = single_slice_annulus_2
    new_image[shp_annulus_2[0] + spacing :, 0 : shp_descending_2[1]] = (
        single_slice_descending_2
    )

    imageio.imwrite(out_file_combined_2, new_image)
    return True


def create_longitudinal_figure_from_straight_volume(
    cl_folder, segm_folder, stats_folder, verbose, quiet, write_log_file, output_folder
):
    """
    Based on the sampled straight volume, we here extract the samples along the long axis
    """
    n_aorta_parts = 1
    parts_stats = read_json_file(f"{stats_folder}aorta_parts.json")
    if parts_stats:
        n_aorta_parts = parts_stats["aorta_parts"]

    if n_aorta_parts == 2:
        return create_longitudinal_figure_from_straight_volume_from_2_part_aort(
            cl_folder, segm_folder, verbose, quiet, write_log_file, output_folder
        )
    infrarenal_in = f"{cl_folder}infrarenal_section.json"
    ventri_in = f"{cl_folder}ventricularaortic.json"
    diaphragm_in = f"{cl_folder}diaphragm.json"
    aortic_arch_in = f"{cl_folder}aortic_arch.json"

    infrarenal_max_in = f"{cl_folder}infrarenal_segment_max_slice_info.json"
    abdominal_max_in = f"{cl_folder}abdominal_segment_max_slice_info.json"
    aortic_arch_max_in = f"{cl_folder}aortic_arch_segment_max_slice_info.json"
    ascending_max_in = f"{cl_folder}ascending_segment_max_slice_info.json"
    sinotubular_max_in = f"{cl_folder}sinotubular_junction_segment_max_slice_info.json"
    sinus_of_valsalva_max_in = (
        f"{cl_folder}sinus_of_valsalva_segment_max_slice_info.json"
    )
    lvot_max_in = f"{cl_folder}lvot_segment_max_slice_info.json"
    descending_max_in = f"{cl_folder}descending_segment_max_slice_info.json"

    cl_hu_stats_file = f"{cl_folder}aorta_centerline_hu_stats.json"

    straight_vol_in = f"{segm_folder}straight_aorta_img.nii.gz"
    straight_label_in = f"{segm_folder}straight_aorta_label.nii.gz"

    out_file = f"{cl_folder}straight_volume_mid_cut.png"
    out_file_2 = f"{cl_folder}straight_volume_mid_cut_2.png"

    if os.path.exists(out_file) or os.path.exists(out_file_2):
        if verbose:
            print(f"{out_file} already exists - skipping")
        return True

    if not os.path.exists(straight_label_in):
        msg = f"Missing files for sampling {straight_label_in}"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    if verbose:
        print(f"Computing long axis figures from {straight_label_in}")

    label_img = read_nifti_with_logging_cached(
        straight_label_in, verbose, quiet, write_log_file, output_folder
    )
    if label_img is None:
        return False
    #
    # try:
    #     label_img = sitk.ReadImage(straight_label_in)
    # except RuntimeError as e:
    #     msg = f"Got an exception {str(e)} reading {straight_label_in}"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     return False

    straight_img = read_nifti_with_logging_cached(
        straight_vol_in, verbose, quiet, write_log_file, output_folder
    )
    if straight_img is None:
        return False
    #
    # try:
    #     straight_img = sitk.ReadImage(straight_vol_in)
    # except RuntimeError as e:
    #     msg = f"Got an exception {str(e)} reading {straight_vol_in}"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     return False

    straight_img_np = sitk.GetArrayFromImage(straight_img)
    straight_img_np = straight_img_np.transpose(2, 1, 0)

    label_img_np = sitk.GetArrayFromImage(label_img)
    label_img_np = label_img_np.transpose(2, 1, 0)
    # dims = label_img_np.shape

    spacing = label_img.GetSpacing()
    # n_slices = dims[2]

    # Defaults
    img_window = 200
    img_level = 200

    cl_stats = read_json_file(cl_hu_stats_file)
    if cl_stats:
        # print(f"Using centerline HU stats")
        img_window = cl_stats["img_window"]
        img_level = cl_stats["img_level"]

    dims = label_img_np.shape
    mid_id = dims[0] // 2
    single_slice_np = label_img_np[mid_id, :, :]

    mid_id_2 = dims[1] // 2
    single_slice_np_2 = label_img_np[:, mid_id_2, :]

    #  only keep one connected component in the slice. The one that is in the middle
    slice_components = label(single_slice_np)
    mid_label = slice_components[dims[1] // 2, dims[2] // 2]
    largest_cc = slice_components == mid_label

    slice_components = label(single_slice_np_2)
    mid_label = slice_components[dims[1] // 2, dims[2] // 2]
    largest_cc_2 = slice_components == mid_label

    # contour = find_contours(img_as_ubyte(largest_cc), 0.5)[0]
    boundary = find_boundaries(largest_cc, mode="thick")
    boundary_2 = find_boundaries(largest_cc_2, mode="thick")

    # skimage.io.imsave(max_slice_boundary_out, boundary)
    single_slice_np_img = straight_img_np[mid_id, :, :]
    single_slice_np_img_2 = straight_img_np[:, mid_id_2, :]

    single_slice_np_img = clutils.set_window_and_level_on_single_slice(
        single_slice_np_img, img_window, img_level
    )
    single_slice_np_img_2 = clutils.set_window_and_level_on_single_slice(
        single_slice_np_img_2, img_window, img_level
    )
    scaled_ubyte = img_as_ubyte(single_slice_np_img)
    scaled_ubyte_2 = img_as_ubyte(single_slice_np_img_2)
    # skimage.io.imsave(max_slice_out, scaled_ubyte)

    scaled_2_rgb = color.gray2rgb(scaled_ubyte)
    scaled_2_rgb_2 = color.gray2rgb(scaled_ubyte_2)
    rgb_boundary = [255, 0, 0]

    # Draw boundary last for visual style
    scaled_2_rgb[boundary > 0] = rgb_boundary
    scaled_2_rgb_2[boundary_2 > 0] = rgb_boundary

    line_stats = []
    infrarenal_stats = read_json_file(infrarenal_in)
    if infrarenal_stats:
        line_stats.append(
            {
                "name": "infrarenal_low",
                "dist": infrarenal_stats["low_dist"],
                "rgb": [255, 255, 255],
            }
        )
        line_stats.append(
            {
                "name": "infrarenal",
                "dist": infrarenal_stats["distance"],
                "rgb": [255, 255, 255],
            }
        )

    ventri_stats = read_json_file(ventri_in)
    if ventri_stats:
        line_stats.append(
            {
                "name": "ventricularoaoartic",
                "dist": ventri_stats["ventri_cl_dist"],
                "rgb": [255, 255, 255],
            }
        )

    diaphragm_stats = read_json_file(diaphragm_in)
    if diaphragm_stats:
        line_stats.append(
            {
                "name": "diaphragm",
                "dist": diaphragm_stats["diaphragm_cl_dist"],
                "rgb": [255, 255, 255],
            }
        )

    aortic_arch_stats = read_json_file(aortic_arch_in)
    if aortic_arch_stats:
        line_stats.append(
            {
                "name": "arch_start",
                "dist": aortic_arch_stats["min_cl_dist"],
                "rgb": [255, 255, 255],
            }
        )
        line_stats.append(
            {
                "name": "arch_end",
                "dist": aortic_arch_stats["max_cl_dist"],
                "rgb": [255, 255, 255],
            }
        )

    max_stats = read_json_file(infrarenal_max_in)
    if max_stats:
        line_stats.append(
            {
                "name": "infrarenal_max",
                "dist": max_stats["cl_dist"],
                "rgb": [255, 255, 0],
            }
        )
    max_stats = read_json_file(abdominal_max_in)
    if max_stats:
        line_stats.append(
            {
                "name": "abdominal_max_in",
                "dist": max_stats["cl_dist"],
                "rgb": [255, 128, 0],
            }
        )
    max_stats = read_json_file(aortic_arch_max_in)
    if max_stats:
        line_stats.append(
            {
                "name": "aortic_arch_max_in",
                "dist": max_stats["cl_dist"],
                "rgb": [255, 0, 0],
            }
        )
    max_stats = read_json_file(ascending_max_in)
    if max_stats:
        line_stats.append(
            {
                "name": "ascending_max_in",
                "dist": max_stats["cl_dist"],
                "rgb": [0, 255, 255],
            }
        )
    max_stats = read_json_file(sinotubular_max_in)
    if max_stats:
        line_stats.append(
            {
                "name": "sinotubular_max_in",
                "dist": max_stats["cl_dist"],
                "rgb": [128, 0, 128],
            }
        )
    max_stats = read_json_file(sinus_of_valsalva_max_in)
    if max_stats:
        line_stats.append(
            {
                "name": "sinus_of_valsalva_max_in",
                "dist": max_stats["cl_dist"],
                "rgb": [0, 128, 255],
            }
        )
    max_stats = read_json_file(descending_max_in)
    if max_stats:
        line_stats.append(
            {
                "name": "descending_max_in",
                "dist": max_stats["cl_dist"],
                "rgb": [0, 255, 0],
            }
        )
    max_stats = read_json_file(lvot_max_in)
    if max_stats:
        line_stats.append(
            {
                "name": "lvot_max_in",
                "dist": max_stats["cl_dist"],
                "rgb": [200, 255, 100],
            }
        )

    for lstats in line_stats:
        dist = lstats["dist"]
        rgb = lstats["rgb"]

        n_straight_idx = int(dist // spacing[2])
        if 0 <= n_straight_idx < dims[2]:
            if n_straight_idx > 0:
                scaled_2_rgb[:, n_straight_idx - 1, :] = rgb
                scaled_2_rgb_2[:, n_straight_idx - 1, :] = rgb
            scaled_2_rgb[:, n_straight_idx, :] = rgb
            if n_straight_idx < dims[2] - 1:
                scaled_2_rgb[:, n_straight_idx + 1, :] = rgb
                scaled_2_rgb_2[:, n_straight_idx + 1, :] = rgb
            scaled_2_rgb_2[:, n_straight_idx, :] = rgb

    skimage.io.imsave(out_file, np.flipud(scaled_2_rgb.transpose(1, 0, 2)))
    skimage.io.imsave(out_file_2, np.flipud(scaled_2_rgb_2.transpose(1, 0, 2)))

    return True


def gather_whole_heart_volumes(segm_folder, stats):
    segm_name_hc = f"{segm_folder}heartchambers_highres.nii.gz"
    if not os.path.exists(segm_name_hc):
        return False

    segm_data, spacing, size = read_nifti_itk_to_numpy(segm_name_hc)

    if segm_data is None:
        return False

    vox_volume = spacing[0] * spacing[1] * spacing[2]

    segments = [
        "heart_myocardium",
        "heart_left_atrium",
        "heart_left_ventricle",
        "heart_right_atrium",
        "heart_right_ventricle",
    ]
    segment_ids = [1, 2, 3, 4, 5]

    for i in range(len(segment_ids)):
        segm_id = segment_ids[i]
        segm_name = segments[i]
        n_vox = np.sum(segm_data == segm_id)
        if n_vox > 0:
            stats[f"vol_{segm_name}"] = n_vox * vox_volume

    return True


def compile_aorta_cut_statistics(cl_folder, compare_with_raw_segmentations=False):
    cl_dir = cl_folder
    ext = ""
    if compare_with_raw_segmentations:
        ext = "_ts_original"

    # currently we do not use the colors
    cut_infos = [
        {"file": f"lvot_segment{ext}_max_slice_info", "rgb": [200, 255, 100]},
        {
            "file": f"sinus_of_valsalva_segment{ext}_max_slice_info",
            "rgb": [0, 128, 255],
        },
        {
            "file": f"sinotubular_junction_segment{ext}_max_slice_info",
            "rgb": [128, 0, 128],
        },
        {"file": f"ascending_segment{ext}_max_slice_info", "rgb": [0, 255, 255]},
        {"file": f"aortic_arch_segment{ext}_max_slice_info", "rgb": [255, 0, 0]},
        {"file": f"descending_segment{ext}_max_slice_info", "rgb": [0, 255, 0]},
        {"file": f"abdominal_segment{ext}_max_slice_info", "rgb": [255, 128, 0]},
        {"file": f"infrarenal_segment{ext}_max_slice_info", "rgb": [255, 255, 0]},
    ]

    # cut_stats = []
    cut_stats = {}
    for ci in cut_infos:
        file_name = f"{cl_dir}/{ci['file']}.json"
        if os.path.exists(file_name):
            slice_info = read_json_file(file_name)
            if slice_info:
                name = ci["file"].replace(f"_segment{ext}_max_slice_info", ext)
                area = slice_info["area"]
                cl_dist = slice_info["cl_dist"]
                min_diameter = slice_info["min_diameter"]
                max_diameter = slice_info["max_diameter"]
                cut_stats[f"{name}_area"] = area
                cut_stats[f"{name}_cl_dist"] = cl_dist
                cut_stats[f"{name}_cl_min_diameter"] = min_diameter
                cut_stats[f"{name}_cl_max_diameter"] = max_diameter
    return cut_stats


def compile_aortic_aneurysm_sac_statistics(stats_folder):
    stats_file = f"{stats_folder}aorta_aneurysm_sac_stats.json"

    out_stats = {}
    stats = read_json_file(stats_file)
    if not stats:
        return None

    out_stats["sac_aorta_lumen_volume"] = stats["aorta_lumen"]
    out_stats["sac_original_lumen_volume"] = stats["original_aorta_volume"]
    out_stats["sac_calcification_volume"] = stats["calcification_volume"]
    out_stats["sac_aorta_ratio"] = stats["aorta_ratio"]
    out_stats["sac_q95_distance"] = stats["q95_distances"]
    return out_stats


def compile_aortic_calcification_statistics(stats_folder):
    calcification_stats_file = f"{stats_folder}aorta_calcification_stats.json"

    out_stats = {}

    stats = read_json_file(calcification_stats_file)
    if not stats:
        return None

    calc_volume = stats["calcification_volume"]
    aorta_lumen_volume = stats["aorta_lumen_volume"]
    if aorta_lumen_volume > 0:
        ratio = calc_volume / aorta_lumen_volume
    else:
        ratio = 0
    out_stats["calcification_volume"] = calc_volume
    out_stats["calcification_pure_lumen_volume"] = aorta_lumen_volume
    out_stats["calcification_ratio"] = ratio

    return out_stats


def compute_aortic_tortuosity_statistics(
    input_file,
    cl_folder,
    lm_folder,
    stats_folder,
    verbose,
    quiet,
    write_log_file,
    output_folder,
):
    stats_file = f"{stats_folder}aorta_tortuosity_stats.json"

    if os.path.exists(stats_file):
        if verbose:
            print(f"{stats_file} already exists - not recomputing")
        return True
    if verbose:
        print(f"Computing {stats_file}")

    ati_stats = clutils.compute_tortuosity_index_based_on_scan_type(
        cl_folder,
        lm_folder,
        stats_folder,
        verbose,
        quiet,
        write_log_file,
        output_folder,
    )

    if ati_stats is None:
        msg = f"Could not compute aortic tortuosity index for {input_file}"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    try:
        with Path(stats_file).open("wt") as handle:
            json.dump(ati_stats, handle, indent=4, sort_keys=False)
    except IOError as e:
        print(f"I/O error({e.errno}): {e.strerror}: {stats_file}")
    return True


def compute_all_aorta_statistics(
    input_file,
    cl_folder,
    segm_folder,
    stats_folder,
    verbose,
    quiet,
    write_log_file,
    output_folder,
):
    """
    Compute aorta statistics from scan.
    including HU distributions, volume and surface
    """
    stats_file = f"{stats_folder}/aorta_statistics.json"
    scan_type_file = f"{stats_folder}/aorta_scan_type.json"
    ati_stats_file = f"{stats_folder}aorta_tortuosity_stats.json"
    skeleton_stats_file = f"{stats_folder}aorta_skeleton_hu_stats.json"
    skeleton_stats_file_a = f"{stats_folder}aorta_skeleton_annulus_hu_stats.json"
    skeleton_stats_file_d = f"{stats_folder}aorta_skeleton_descending_hu_stats.json"

    gather_wh_stats = True

    if os.path.exists(stats_file):
        if verbose:
            print(f"{stats_file} already exists - not recomputing")
        return True

    if verbose:
        print(f"Computing {stats_file}")

    # Get pure name of input file without path and extension
    scan_id = os.path.basename(input_file)
    scan_id = os.path.splitext(scan_id)[0]
    if scan_id.endswith(".nii"):
        scan_id = os.path.splitext(scan_id)[0]

    stats = {"scan_name": input_file, "base_name": scan_id}

    ao_version = importlib.metadata.version("AortaExplorer")
    stats["aorta_explorer_version"] = ao_version

    last_error_message = get_last_error_message()
    if last_error_message:
        stats["last_error_message"] = last_error_message

    n_aorta_parts = 1
    parts_stats = read_json_file(f"{stats_folder}/aorta_parts.json")
    if parts_stats:
        n_aorta_parts = parts_stats["aorta_parts"]
        stats["n_aorta_parts"] = n_aorta_parts


    segm_name = f"{segm_folder}aorta_lumen.nii.gz"
    aorta_segment_id = 1

    # stats = None
    img_data, spacing, size = read_nifti_itk_to_numpy(input_file)
    if img_data is None:
        msg = f"Cannot read {input_file}"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        # return False

    vol_dims = size
    vox_volume = spacing[0] * spacing[1] * spacing[2]

    stats["spacing"] = spacing
    stats["volume_dims"] = vol_dims
    stats["volume_size"] = [
        vol_dims[0] * spacing[0],
        vol_dims[1] * spacing[1],
        vol_dims[2] * spacing[2],
    ]

    segm_data, _, _ = read_nifti_itk_to_numpy(segm_name)
    if segm_data is None:
        msg = f"Cannot read {segm_name}"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        # return False
    else:
        if segm_data.sum() == 0:
            msg = f"No aorta voxels in {segm_name}"
            if not quiet:
                print(msg)
            if write_log_file:
                write_message_to_log_file(
                    base_dir=output_folder, message=msg, level="error"
                )
        else:
            hu_values = img_data[segm_data == aorta_segment_id]
            stats["avg_hu"] = np.average(hu_values)
            stats["std_hu"] = np.std(hu_values)
            stats["med_hu"] = np.median(hu_values)
            stats["q99_hu"] = np.percentile(hu_values, 99)
            stats["q01_hu"] = np.percentile(hu_values, 1)
            stats["tot_vol"] = len(hu_values) * vox_volume

    # Somewhat hacky to do this here:
    if n_aorta_parts == 2:
        segm_name_asc = f"{segm_folder}aorta_lumen_annulus.nii.gz"
        segm_data, _, _ = read_nifti_itk_to_numpy(segm_name_asc)
        if segm_data is None:
            msg = f"Cannot read {segm_name_asc}"
            if not quiet:
                print(msg)
            if write_log_file:
                write_message_to_log_file(
                    base_dir=output_folder, message=msg, level="error"
                )
        else:
            vox_volume = spacing[0] * spacing[1] * spacing[2]
            n_vox = np.sum(segm_data == aorta_segment_id)
            stats["vol_ascending_aorta"] = n_vox * vox_volume

        segm_name_desc = f"{segm_folder}aorta_lumen_descending.nii.gz"
        segm_data, _, _ = read_nifti_itk_to_numpy(segm_name_desc)
        if segm_data is None:
            msg = f"Cannot read {segm_name_desc}"
            if not quiet:
                print(msg)
            if write_log_file:
                write_message_to_log_file(
                    base_dir=output_folder, message=msg, level="error"
                )
        else:
            vox_volume = spacing[0] * spacing[1] * spacing[2]
            n_vox = np.sum(segm_data == aorta_segment_id)
            stats["vol_descending_aorta"] = n_vox * vox_volume

    skeleton_hu_files = [skeleton_stats_file, skeleton_stats_file_a, skeleton_stats_file_d]
    for sh_file in skeleton_hu_files:
        sh_stats = read_json_file(sh_file)
        if sh_stats is not None:
            stats = {**stats, **sh_stats}

    if gather_wh_stats:
        gather_whole_heart_volumes(segm_folder, stats)

    scan_type = read_json_file(scan_type_file)
    if scan_type:
        stats["scan_type"] = scan_type["scan_type"]
        stats["scan_type_desc"] = scan_type["scan_type_desc"]

    surfutils.aorta_volume_properties(
        segm_folder, stats_folder, quiet, write_log_file, output_folder, stats
    )

    cl_stats = sample_aorta_center_line_hu_stats(
        input_file,
        cl_folder,
        stats_folder,
        verbose,
        quiet,
        write_log_file,
        output_folder,
    )
    if cl_stats is not None:
        # https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression-in-python
        stats = {**stats, **cl_stats}

    cut_stats = compile_aorta_cut_statistics(cl_folder, False)
    if cut_stats is not None:
        stats = {**stats, **cut_stats}

    ati_stats = read_json_file(ati_stats_file)
    if ati_stats is not None:
        stats = {**stats, **ati_stats}

    sac_stats = compile_aortic_aneurysm_sac_statistics(stats_folder)
    if sac_stats:
        stats = {**stats, **sac_stats}

    calcification_stats = compile_aortic_calcification_statistics(stats_folder)
    if calcification_stats:
        stats = {**stats, **calcification_stats}

    # TODO: Implement a method to add patient metadata

    try:
        with Path(stats_file).open("wt") as handle:
            json.dump(stats, handle, indent=4, sort_keys=False)
    except IOError as e:
        print(f"I/O error({e.errno}): {e.strerror}: {stats_file}")
    return True


def aorta_visualization(
    input_file,
    cl_folder,
    segm_folder,
    stats_folder,
    vis_folder,
    verbose,
    params,
    save_to_file=True,
):
    mask_with_body_segmentation = True
    vis_file = f"{vis_folder}aorta_visualization.png"
    win_size = params.get("rendering_window_size", [1600, 1200])

    if os.path.exists(vis_file):
        if verbose:
            print(f"{vis_file} already exists - skipping")
        return True

    if verbose:
        print(f"Creating aorta visualization {vis_file}")

    # Get pure name of input file without path and extension
    scan_id = os.path.basename(input_file)
    scan_id = os.path.splitext(scan_id)[0]
    if scan_id.endswith(".nii"):
        scan_id = os.path.splitext(scan_id)[0]


    render_aorta_data = RenderAortaData(
        win_size, scan_id, save_to_file, stats_folder, segm_folder, cl_folder
    )

    segm_name = None
    if mask_with_body_segmentation:
        segm_name = f"{segm_folder}body.nii.gz"

    render_aorta_data.set_sitk_image_file(input_file, segm_name)
    render_aorta_data.set_aorta_statistics(stats_folder)

    render_aorta_data.set_plot_data(stats_folder, cl_folder)
    render_aorta_data.set_precomputed_slice(cl_folder)
    render_aorta_data.set_all_max_cut_data(cl_folder)
    render_aorta_data.set_cut_statistics(cl_folder)
    render_aorta_data.set_aortic_tortuosity_index_statistics(stats_folder)
    render_aorta_data.set_aortic_aneurysm_sac_statistics(stats_folder)
    render_aorta_data.set_aortic_calcification_statistics(stats_folder)
    render_aorta_data.set_precomputed_straight_longitudinal_slices(cl_folder)

    render_aorta_data.render_to_file(vis_file)
    # render_aorta_data.render_interactive()
    return True


def do_aorta_analysis(
    verbose, quiet, write_log_file, params, output_folder, input_file
):
    """
    Compute aorta data
    input_file: input CT file with path
    """
    # Get pure name of input file without path and extension
    scan_id = os.path.basename(input_file)
    scan_id = os.path.splitext(scan_id)[0]
    if scan_id.endswith(".nii"):
        scan_id = os.path.splitext(scan_id)[0]

    segm_folder = f"{output_folder}{scan_id}/segmentations/"
    stats_folder = f"{output_folder}{scan_id}/statistics/"
    surface_folder = f"{output_folder}{scan_id}/surfaces/"
    lm_folder = f"{output_folder}{scan_id}/landmarks/"
    cl_folder = f"{output_folder}{scan_id}/centerlines/"
    vis_folder = f"{output_folder}{scan_id}/visualization/"

    total_in_name = f"{segm_folder}total.nii.gz"
    use_org_ts_segmentations = params.get(
        "compute_centerline_from_ts_segmentation", True
    )

    # It is possible to compare with the results that the raw TotalSegmentator segmentations would give
    # This is mainly for research purposes to see how much the results differ
    compare_with_raw_ts_segmentations = False

    Path(stats_folder).mkdir(parents=True, exist_ok=True)
    Path(surface_folder).mkdir(parents=True, exist_ok=True)
    Path(lm_folder).mkdir(parents=True, exist_ok=True)
    Path(cl_folder).mkdir(parents=True, exist_ok=True)
    Path(vis_folder).mkdir(parents=True, exist_ok=True)

    # Do not inherit any previous error message
    clear_last_error_message()
    setup_vtk_error_handling(output_folder)

    if not os.path.exists(input_file):
        msg = f"Could not find {input_file} for aorta analysis"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    ts_total_exists = os.path.exists(total_in_name)
    # ts_hc_exists = os.path.exists(hc_in_name)
    if not ts_total_exists:
        msg = f"Could not find TotalSegmentator segmentations {total_in_name} for aorta analysis"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return False

    success = True
    success = compute_body_segmentation(
        input_file, segm_folder, verbose, quiet, write_log_file, output_folder
    )
    if success:
        success = compute_out_scan_field_segmentation_and_sdf(
            input_file,
            segm_folder,
            params,
            verbose,
            quiet,
            write_log_file,
            output_folder,
        )
    if success:
        success = extract_pure_aorta_lumen_start_by_finding_parts(
            input_file,
            params,
            segm_folder,
            stats_folder,
            verbose,
            quiet,
            write_log_file,
            output_folder,
        )
    if success:
        success = extract_top_of_iliac_arteries(
            input_file, segm_folder, verbose, quiet, write_log_file, output_folder
        )
    if success:
        success = extract_aortic_calcifications(
            input_file,
            params,
            segm_folder,
            stats_folder,
            verbose,
            quiet,
            write_log_file,
            output_folder,
        )
    if success:
        success = check_for_aneurysm_sac(
            segm_folder, stats_folder, verbose, quiet, write_log_file, output_folder
        )

    # TODO: Issue warning if large difference between TotalSegmentator and lumen segmentations is found
    # and the centerline computation is based on TotalSegmentator segmentations
    # This typically happens if there are large sac-like aneurysms

    if success:
        success = compute_ventricularoaortic_landmark(
            segm_folder,
            lm_folder,
            stats_folder,
            verbose,
            quiet,
            write_log_file,
            output_folder,
        )
    if success:
        success = combine_aorta_and_left_ventricle_and_iliac_arteries(
            input_file,
            segm_folder,
            lm_folder,
            stats_folder,
            verbose,
            quiet,
            write_log_file,
            output_folder,
            use_ts_org_segmentations=False,
        )
    if success:
        success = combine_aorta_and_left_ventricle_and_iliac_arteries(
            input_file,
            segm_folder,
            lm_folder,
            stats_folder,
            verbose,
            quiet,
            write_log_file,
            output_folder,
            use_ts_org_segmentations=True,
        )
    if success:
        success = compute_aortic_arch_landmarks(
            segm_folder, lm_folder, verbose, quiet, write_log_file, output_folder
        )
    if success:
        success = compute_diaphragm_landmarks_from_surfaces(
            segm_folder, lm_folder, verbose, quiet, write_log_file, output_folder
        )
    if success:
        success = compute_aorta_scan_type(
            input_file,
            segm_folder,
            lm_folder,
            stats_folder,
            verbose,
            quiet,
            write_log_file,
            output_folder,
        )
    if success:
        success = extract_surfaces_for_centerlines(
            segm_folder,
            stats_folder,
            surface_folder,
            verbose,
            quiet,
            write_log_file,
            output_folder,
            use_ts_org_segmentations=use_org_ts_segmentations,
        )
    if success:
        success = compute_centerline_landmarks_based_on_scan_type(
            segm_folder,
            lm_folder,
            stats_folder,
            verbose,
            quiet,
            write_log_file,
            output_folder,
            use_ts_org_segmentations=use_org_ts_segmentations,
        )
    if success:
        success = compute_center_line_using_skeleton(segm_folder, stats_folder, lm_folder, surface_folder, cl_folder,
                                                     verbose, quiet, write_log_file, output_folder, use_ts_org_segmentations=True)
    if success:
        success = compute_center_line(
            stats_folder,
            lm_folder,
            surface_folder,
            cl_folder,
            verbose,
            quiet,
            write_log_file,
            output_folder,
        )
    if success:
        success = compute_infrarenal_section_using_kidney_to_kidney_line(
            segm_folder,
            stats_folder,
            lm_folder,
            cl_folder,
            verbose,
            quiet,
            write_log_file,
            output_folder,
        )
    if success:
        success = compute_aortic_arch_landmarks_on_centerline(
            lm_folder, cl_folder, verbose, quiet, write_log_file, output_folder
        )
    if success:
        success = compute_diaphragm_point_on_centerline(
            lm_folder,
            cl_folder,
            stats_folder,
            verbose,
            quiet,
            write_log_file,
            output_folder,
        )
    if success:
        success = compute_ventricularoaortic_point_on_centerline(
            lm_folder,
            cl_folder,
            stats_folder,
            verbose,
            quiet,
            write_log_file,
            output_folder,
        )
    if success:
        success = (
            sample_aorta_center_line_hu_stats(
                input_file,
                cl_folder,
                stats_folder,
                verbose,
                quiet,
                write_log_file,
                output_folder,
            )
            is not None
        )
    if success:
        success = compute_straightened_volume_using_cpr(
            input_file,
            segm_folder,
            cl_folder,
            stats_folder,
            verbose,
            quiet,
            write_log_file,
            output_folder,
            use_raw_segmentations=False,
        )
    if compare_with_raw_ts_segmentations:
        if success:
            success = compute_straightened_volume_using_cpr(
                input_file,
                segm_folder,
                cl_folder,
                stats_folder,
                verbose,
                quiet,
                write_log_file,
                output_folder,
                use_raw_segmentations=True,
            )
    if success:
        success = compute_cuts_along_straight_labelmaps(
            segm_folder,
            cl_folder,
            stats_folder,
            verbose,
            quiet,
            write_log_file,
            output_folder,
        )
    if success:
        success = compute_sinutubular_junction_and_sinus_of_valsalva_from_max_and_min_cut_areas(
            cl_folder,
            lm_folder,
            stats_folder,
            verbose,
            quiet,
            write_log_file,
            output_folder,
        )
    if success:
        success = identy_and_extract_samples_from_straight_volume(
            cl_folder,
            segm_folder,
            stats_folder,
            verbose,
            quiet,
            write_log_file,
            output_folder,
            use_raw_segmentations=False,
        )
    if success:
        success = combine_cross_section_images_into_one(cl_folder, verbose)
    if success:
        success = create_longitudinal_figure_from_straight_volume(
            cl_folder,
            segm_folder,
            stats_folder,
            verbose,
            quiet,
            write_log_file,
            output_folder,
        )
    if success:
        success = compute_aortic_tortuosity_statistics(
            input_file,
            cl_folder,
            lm_folder,
            stats_folder,
            verbose,
            quiet,
            write_log_file,
            output_folder,
        )

    # Not matter the success, we still gather statistics. At least we will get the last error message
    success = compute_all_aorta_statistics(
        input_file,
        cl_folder,
        segm_folder,
        stats_folder,
        verbose,
        quiet,
        write_log_file,
        output_folder,
    )

    # Also try to create visualization of the things we achieved even in erro
    success = aorta_visualization(
        input_file, cl_folder, segm_folder, stats_folder, vis_folder, verbose, params
    )

    if verbose:
        image_reader_cache_info = read_nifti_with_logging_cached.cache_info()
        print(f"Image reader cache info: {image_reader_cache_info}")
    read_nifti_with_logging_cached.cache_clear()

    return True


def computer_process(
    verbose, quiet, write_log_file, params, output_folder, process_queue, process_id
):
    while not process_queue.empty():
        q_size = process_queue.qsize()
        input_file = process_queue.get()
        if verbose:
            print(
                f"Process {process_id} running aorta analysis on: {input_file} - {q_size} left"
            )
        local_start_time = time.time()
        do_aorta_analysis(
            verbose, quiet, write_log_file, params, output_folder, input_file
        )
        elapsed_time = time.time() - local_start_time
        q_size = process_queue.qsize()
        est_time_left = q_size * elapsed_time
        if verbose:
            print(
                f"Process {process_id} done with {input_file} - took {elapsed_time:.1f} s. Time left {est_time_left:.1f} s"
            )


def aorta_analysis(
    in_files,
    output_folder,
    params=None,
    nr_tg=1,
    verbose=False,
    quiet=False,
    write_log_file=True,
):
    if verbose:
        print(
            f"Computing aorta data with max {nr_tg} processes on {len(in_files)} files. Output to {output_folder}"
        )

    num_processes = nr_tg
    # no need to spawn more processes than files
    num_processes = min(num_processes, len(in_files))

    # no need to do multiprocessing for one file
    if len(in_files) == 1:
        input_file = in_files[0].strip()
        if verbose:
            print(f"Running aorta analysis on: {input_file}")
        local_start_time = time.time()
        do_aorta_analysis(
            verbose, quiet, write_log_file, params, output_folder, input_file
        )
        elapsed_time = time.time() - local_start_time
        if verbose:
            print(f"Done with {input_file} - took {elapsed_time:.1f} s.")

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
                    params,
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
