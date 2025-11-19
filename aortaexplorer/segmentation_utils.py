import numpy as np
from skimage.measure import label
from scipy.ndimage import measurements
import edt
import SimpleITK as sitk
from aortaexplorer.io_utils import read_nifti_file_robustly


def get_components_over_certain_size(
    segmentation, min_size=5000, max_number_of_components=2
):
    debug = False
    labels = label(segmentation)
    bin_c = np.bincount(labels.flat, weights=segmentation.flat)
    # probably extremely unefficient
    comp_ids = []
    for c in range(max_number_of_components):
        idx = np.argmax(bin_c)
        if bin_c[idx] > min_size:
            comp_ids.append(idx)
            bin_c[idx] = 0

    if len(comp_ids) < 1:
        if debug:
            print(f"No connected components with size above {min_size} found")
        return None, None
    largest_cc = labels == comp_ids[0]
    for idx in range(1, len(comp_ids)):
        largest_cc = np.bitwise_or(largest_cc, labels == comp_ids[idx])

    return largest_cc, len(comp_ids)


def get_components_over_certain_size_as_individual_volumes(
    segmentation, min_size=5000, max_number_of_components=2
):
    labels = label(segmentation)
    bin_c = np.bincount(labels.flat, weights=segmentation.flat)
    debug = False
    # probably extremely unefficient
    comp_ids = []
    for c in range(max_number_of_components):
        idx = np.argmax(bin_c)
        if debug:
            print(f"{bin_c[idx]}")
        if bin_c[idx] > min_size:
            comp_ids.append(idx)
            bin_c[idx] = 0

    if len(comp_ids) < 1:
        if debug:
            print(f"No connected components with size above {min_size} found")
        return None

    components = []
    for idx in range(0, len(comp_ids)):
        largest_cc = labels == comp_ids[idx]
        components.append(largest_cc)

    return components


def close_cavities_in_segmentations(segmentation):
    """
    Close cavities in segmentations by finding the largest connected component of the background
    """
    background = segmentation == 0
    labels = label(background)
    bin_c = np.bincount(labels.flat, weights=background.flat)
    n_comp = np.count_nonzero(bin_c)
    idx = np.argmax(bin_c)

    connected_background = labels == idx
    closed_segm = np.bitwise_not(connected_background)

    return closed_segm, n_comp


def edt_based_opening(segmentation, spacing, radius):
    sdf_mask = -edt.sdf(
        segmentation, anisotropy=[spacing[0], spacing[1], spacing[2]], parallel=8
    )
    eroded_mask = sdf_mask < -radius
    sdf_mask = -edt.sdf(
        eroded_mask, anisotropy=[spacing[0], spacing[1], spacing[2]], parallel=8
    )
    opened_mask = sdf_mask <= radius
    return opened_mask


def edt_based_closing(segmentation, spacing, radius):
    sdf_mask = -edt.sdf(
        segmentation, anisotropy=[spacing[0], spacing[1], spacing[2]], parallel=8
    )
    dilated_mask = sdf_mask <= radius
    sdf_mask = -edt.sdf(
        dilated_mask, anisotropy=[spacing[0], spacing[1], spacing[2]], parallel=8
    )
    closed_mask = sdf_mask < -radius
    return closed_mask


def edt_based_dilation(segmentation, spacing, radius):
    sdf_mask = -edt.sdf(
        segmentation, anisotropy=[spacing[0], spacing[1], spacing[2]], parallel=8
    )
    dilated_mask = sdf_mask <= radius
    return dilated_mask


def edt_based_erosion(segmentation, spacing, radius):
    sdf_mask = -edt.sdf(
        segmentation, anisotropy=[spacing[0], spacing[1], spacing[2]], parallel=8
    )
    eroded_mask = sdf_mask < -radius
    return eroded_mask


def edt_based_overlap(segmentation_1, segmentation_2, spacing, radius):
    """
    Compute the overlap between two segmentations using the Euclidean distance transform
    """
    sdf_mask_1 = -edt.sdf(
        segmentation_1, anisotropy=[spacing[0], spacing[1], spacing[2]], parallel=8
    )
    sdf_mask_2 = -edt.sdf(
        segmentation_2, anisotropy=[spacing[0], spacing[1], spacing[2]], parallel=8
    )
    overlap_mask = (sdf_mask_1 < radius) & (sdf_mask_2 < radius)

    # overlap_mask = np.bitwise_and(sdf_mask_1 < radius, sdf_mask_2 < radius)
    return overlap_mask


def edt_based_compute_landmark_from_segmentation_overlap(
    segmentation_1,
    segmentation_2,
    radius,
    segm_sitk_img,
    overlap_name,
    lm_name,
    min_size_mm3 = 2000,
    only_larges_components=True,
    debug=False,
):
    if debug:
        print(f"Computing {overlap_name} and {lm_name}")

    spacing = segm_sitk_img.GetSpacing()
    spc_trans = [spacing[2], spacing[1], spacing[0]]
    min_size_vox = min_size_mm3 / (spacing[0] * spacing[1] * spacing[2])

    overlap_mask = edt_based_overlap(segmentation_1, segmentation_2, spc_trans, radius)
    if only_larges_components:
        overlap_mask, n_comp = get_components_over_certain_size(overlap_mask, min_size_vox, 1)
        if overlap_mask is None or n_comp < 1:
            if debug:
                print(f"No components found in {overlap_name}")
            return False

    if np.sum(overlap_mask) == 0:
        if debug:
            print(f"No overlap found for {overlap_name}")
        return False

    com_np = measurements.center_of_mass(overlap_mask)
    com_np = [com_np[2], com_np[1], com_np[0]]

    com_phys = segm_sitk_img.TransformIndexToPhysicalPoint(
        [int(com_np[0]), int(com_np[1]), int(com_np[2])]
    )
    if debug:
        img_o = sitk.GetImageFromArray(overlap_mask.astype(int))
        img_o.CopyInformation(segm_sitk_img)

        print(f"saving {overlap_name}")
        sitk.WriteImage(img_o, overlap_name)

    end_p_out = open(lm_name, "w")
    end_p_out.write(f"{com_phys[0]} {com_phys[1]} {com_phys[2]}")
    end_p_out.close()
    return True


def compute_segmentation_volume(segmentation_file, segm_id):
    """
    Compute the volume of a segmentation
    """
    segmentation, _ = read_nifti_file_robustly(segmentation_file)
    if segmentation is None:
        return 0
    #
    # try:
    #     segmentation = sitk.ReadImage(segmentation_file)
    # except RuntimeError as e:
    #     print(f"Got an exception {str(e)}")
    #     print(f"Error reading {segmentation_file}")
    #     return 0

    spacing = segmentation.GetSpacing()

    segmentation_np = sitk.GetArrayFromImage(segmentation)
    volume = np.sum(segmentation_np == segm_id)
    volume = volume * spacing[0] * spacing[1] * spacing[2]
    return volume


def read_nifti_itk_to_numpy(file_name):
    img, _ = read_nifti_file_robustly(file_name)
    if img is None:
        return None, None, None
    #
    # try:
    #     img = sitk.ReadImage(file_name)
    # except RuntimeError as e:
    #     print(f"Got an exception {str(e)}")
    #     print(f"Error reading {file_name}")
    #     return None, None, None

    i2 = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    size = img.GetSize()
    return i2, spacing, size


def check_if_segmentation_hit_sides_of_scan(segmentation, segm_id, n_slices_to_check=8):
    """
    Return the sides (if any) that the segmentation hits
    """
    shp = segmentation.shape
    bin_segm = segmentation == segm_id

    down = np.sum(bin_segm[0:n_slices_to_check, :, :])
    up = np.sum(bin_segm[shp[0] - 1 - n_slices_to_check : shp[0] - 1, :, :])
    left = np.sum(bin_segm[:, 0:n_slices_to_check, :])
    right = np.sum(bin_segm[:, shp[1] - 1 - n_slices_to_check : shp[1] - 1, :])
    front = np.sum(bin_segm[:, :, 0:n_slices_to_check])
    back = np.sum(bin_segm[:, :, shp[2] - 1 - n_slices_to_check : shp[2] - 1])

    sides = set()
    if up > 0:
        sides.add("up")
    if down > 0:
        sides.add("down")
    if left > 0:
        sides.add("left")
    if right > 0:
        sides.add("right")
    if front > 0:
        sides.add("front")
    if back > 0:
        sides.add("back")

    return sides
