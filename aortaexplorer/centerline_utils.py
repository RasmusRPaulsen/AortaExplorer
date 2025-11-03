import os.path
import numpy as np
import vtk
from scipy.interpolate import UnivariateSpline
from aortaexplorer.curvedreformat_utils import CurvedPlanarReformat
from aortaexplorer.general_utils import write_message_to_log_file, read_json_file
from aortaexplorer.io_utils import read_nifti_file_robustly
import SimpleITK as sitk
import json
from skimage.measure import label, regionprops, find_contours
from skimage.segmentation import find_boundaries
import skimage.io
from skimage import color
from skimage.util import img_as_ubyte
from skimage.exposure import rescale_intensity
from skimage.draw import line

# VMTK only works with a very specific environment
try:
    from vmtk import vmtkscripts
except ImportError as e:
    print(f"Failed to import vmtk name {e.name} and path {e.path}")
    pass


def read_landmarks(filename):
    if not os.path.exists(filename):
        return None

    x, y, z = 0, 0, 0
    with open(filename) as f:
        for line in f:
            if len(line) > 1:
                temp = line.split()  # Remove whitespaces and line endings and so on
                x, y, z = np.double(temp)
    return x, y, z


def add_distances_from_landmark_to_centerline(in_center, lm_in):
    """
    Add scalar values to a center line where each value is the accumulated distance to the start point
    :param in_center: vtk center line
    :param lm_in: start landmark
    :return: centerline with scalar values
    """
    n_points = in_center.GetNumberOfPoints()
    # print(f"Number of points in centerline: {n_points}")

    cen_p_start = in_center.GetPoint(0)
    cen_p_end = in_center.GetPoint(n_points - 1)
    dist_start = np.linalg.norm(np.subtract(lm_in, cen_p_start))
    dist_end = np.linalg.norm(np.subtract(lm_in, cen_p_end))
    # print(f"Dist start: {dist_start} end: {dist_end}")

    start_idx = 0
    inc = 1
    # Go reverse
    if dist_start > dist_end:
        start_idx = n_points - 1
        inc = -1

    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    scalars = vtk.vtkDoubleArray()
    scalars.SetNumberOfComponents(1)

    idx = start_idx
    p_1 = in_center.GetPoint(idx)
    accumulated_dist = 0
    pid = points.InsertNextPoint(p_1)
    scalars.InsertNextValue(accumulated_dist)

    while 0 < idx <= n_points:
        idx += inc
        p_2 = in_center.GetPoint(idx)
        dist = np.linalg.norm(np.subtract(p_1, p_2))
        # pid = points.InsertNextPoint(p_1)
        # scalars.InsertNextValue(accumulated_dist)
        accumulated_dist += dist
        pid_2 = points.InsertNextPoint(p_2)
        scalars.InsertNextValue(accumulated_dist)
        lines.InsertNextCell(2)
        lines.InsertCellPoint(pid)
        lines.InsertCellPoint(pid_2)
        p_1 = p_2
        pid = pid_2

    pd = vtk.vtkPolyData()
    pd.SetPoints(points)
    del points
    pd.SetLines(lines)
    del lines
    pd.GetPointData().SetScalars(scalars)
    del scalars

    # print(f"Number of points in refined centerline: {pd.GetNumberOfPoints()}")
    return pd


def add_start_and_end_point_to_centerline(in_center, start_point, end_point):
    """
    Add start and end point centerline
    """
    n_points = in_center.GetNumberOfPoints()
    # print(f"Number of points in centerline: {n_points}")

    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    pid = points.InsertNextPoint(start_point)

    for idx in range(n_points):
        p_2 = in_center.GetPoint(idx)
        pid_2 = points.InsertNextPoint(p_2)
        lines.InsertNextCell(2)
        lines.InsertCellPoint(pid)
        lines.InsertCellPoint(pid_2)
        pid = pid_2

    pid_2 = points.InsertNextPoint(end_point)
    lines.InsertNextCell(2)
    lines.InsertCellPoint(pid)
    lines.InsertCellPoint(pid_2)

    pd = vtk.vtkPolyData()
    pd.SetPoints(points)
    del points
    pd.SetLines(lines)
    del lines

    # print(f"Number of points in refined centerline: {pd.GetNumberOfPoints()}")
    return pd


def compute_spline_from_path(
    cl_in, cl_out_file, spline_smoothing_factor=20, sample_spacing=0.25
):
    sum_dist = 0
    n_points = cl_in.GetNumberOfPoints()

    x = []
    y_1 = []
    y_2 = []
    y_3 = []
    p = cl_in.GetPoint(n_points - 1)
    p_old = p
    # Compute the three individual components of the path
    # it is parameterised using the length along the path

    for idx in range(n_points):
        # We go backwards to fix a reverse distance problem
        p = cl_in.GetPoint(n_points - idx - 1)
        d = np.linalg.norm(np.array(p) - np.array(p_old))
        sum_dist += d
        p_old = p
        x.append(sum_dist)
        y_1.append(p[0])
        y_2.append(p[1])
        y_3.append(p[2])

    min_x = 0
    max_x = sum_dist

    spl_1 = UnivariateSpline(x, y_1)
    spl_2 = UnivariateSpline(x, y_2)
    spl_3 = UnivariateSpline(x, y_3)
    spl_1.set_smoothing_factor(spline_smoothing_factor)
    spl_2.set_smoothing_factor(spline_smoothing_factor)
    spl_3.set_smoothing_factor(spline_smoothing_factor)

    # self.spline_parameters = {
    #     "min_x": self.min_x,
    #     "max_x": self.max_x,
    #     "spl_1": self.spl_1,
    #     "spl_2": self.spl_2,
    #     "spl_3": self.spl_3
    # }

    # def compute_sampled_spline_curve(self):
    samp_space = sample_spacing
    spline_n_points = int(max_x / samp_space)
    # print(f"Computing sampled spline path with length {max_x:.1f} and sample spacing {samp_space} "
    #      f"resulting in {spline_n_points} samples for smoothing")

    # Compute a polydata object with the spline points
    xs = np.linspace(min_x, max_x, spline_n_points)

    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    scalars = vtk.vtkDoubleArray()
    scalars.SetNumberOfComponents(1)

    current_idx = 0
    sum_dist = 0
    sp = [spl_1(xs[current_idx]), spl_2(xs[current_idx]), spl_3(xs[current_idx])]
    pid = points.InsertNextPoint(sp)
    scalars.InsertNextValue(sum_dist)
    current_idx += 1

    while current_idx < spline_n_points:
        p_1 = [spl_1(xs[current_idx]), spl_2(xs[current_idx]), spl_3(xs[current_idx])]
        sum_dist += np.linalg.norm(np.array(p_1) - np.array(sp))
        lines.InsertNextCell(2)
        pid_2 = points.InsertNextPoint(p_1)
        scalars.InsertNextValue(sum_dist)
        lines.InsertCellPoint(pid)
        lines.InsertCellPoint(pid_2)
        pid = pid_2
        sp = p_1
        current_idx += 1

    vtk_spline = vtk.vtkPolyData()
    vtk_spline.SetPoints(points)
    del points
    vtk_spline.SetLines(lines)
    del lines
    vtk_spline.GetPointData().SetScalars(scalars)
    del scalars

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(vtk_spline)
    writer.SetFileName(cl_out_file)
    writer.Write()


def compute_single_center_line(surface_name, cl_name, start_p_name, end_p_name):
    debug = False

    cl_org_name = os.path.splitext(cl_name)[0] + "_original.vtp"
    cl_spline_name = cl_name

    # cl_spline_name = os.path.splitext(cl_name)[0] + "_spline.vtk"
    # cl_surface_name = os.path.splitext(cl_name)[0] + "_surface_in.vtp"
    voronoi_name = os.path.splitext(cl_name)[0] + "_voronoi.vtp"

    if not os.path.exists(surface_name):
        print(f"Could not read {surface_name}")
        return False
    # print(f"Computing centerline from {surface_name}")
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(surface_name)
    reader.Update()
    surface = reader.GetOutput()

    start_point = read_landmarks(start_p_name)
    end_point = read_landmarks(end_p_name)
    if start_point is None or end_point is None:
        print(f"Could not read landmarks {start_p_name} or {end_p_name}")
        return False

    # computes the centerlines using vmtk
    centerlinePolyData = vmtkscripts.vmtkCenterlines()
    centerlinePolyData.Surface = surface
    centerlinePolyData.SeedSelectorName = "pointlist"
    centerlinePolyData.SourcePoints = start_point
    # endpoints needs to be: len(end_points) mod 3 = 0
    centerlinePolyData.TargetPoints = end_point
    # centerlinePolyData.AppendEndPoints = 1
    try:
        centerlinePolyData.Execute()
    except RuntimeError as e:
        print(f"Got an exception {str(e)}")
        print(f"When computing cl on {surface_name}")
        return False

    if debug:
        assignAttribute_v = vtk.vtkAssignAttribute()
        assignAttribute_v.SetInputData(centerlinePolyData.VoronoiDiagram)
        assignAttribute_v.Assign(
            "MaximumInscribedSphereRadius",
            vtk.vtkDataSetAttributes.SCALARS,
            vtk.vtkAssignAttribute.POINT_DATA,
        )
        assignAttribute_v.Update()

        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(assignAttribute_v.GetOutput())
        writer.SetFileName(voronoi_name)
        writer.SetFileTypeToBinary()
        writer.Write()

    if centerlinePolyData.Centerlines.GetNumberOfPoints() < 10:
        print("Something wrong with centerline")
        return False

    # Adding start and end points did not work very well. Too many errors at the ends
    # cl_added = add_start_and_end_point_to_centerline(centerlinePolyData.Centerlines, end_point, start_point)
    # cl_dists = add_distances_from_landmark_to_centerline(cl_added, start_point)
    cl_dists = add_distances_from_landmark_to_centerline(
        centerlinePolyData.Centerlines, start_point
    )

    # saves the centerlines in new vtk file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(cl_dists)
    writer.SetFileName(cl_org_name)
    writer.Write()

    cl_added = add_start_and_end_point_to_centerline(
        centerlinePolyData.Centerlines, end_point, start_point
    )
    compute_spline_from_path(cl_added, cl_spline_name)

    return True


def get_tangent_from_centerline(cl, idx):
    """
    Get centerline tanget at given point idx.
    The tangents are assumed to be precomputed and stored in the "Tangents" array.
    Typically they are computed using a spline fit to the centerline.
    """
    if cl.GetNumberOfPoints() < 2:
        print("Centerline has less than 2 points. Can not estimate tangent.")
        return None

    tangents = cl.GetPointData().GetArray("Tangents")
    if tangents is None:
        print("Centerline has no Tangents array. Can not estimate tanget.")
        return None

    t = tangents.GetTuple(idx)
    return list(t)


# def estimate_normal_from_centerline(cl, idx, n_samples=4):
#     """
#     Estimate centerline normal at given point idx.
#     n_samples is the number of points to use in the estimation
#     """
#     if cl.GetNumberOfPoints() < 2:
#         print("Centerline has less than 2 points. Can not estimate normal.")
#         return None
#         # return [1, 0, 0]
#     elif cl.GetNumberOfPoints() < n_samples:
#         print(f"Centerline has only {cl.GetNumberOfPoints()} points. Using them")
#         idx_1 = 0
#         idx_2 = cl.GetNumberOfPoints() - 1
#     elif idx == 0:
#         idx_1 = 0
#         idx_2 = n_samples
#     elif idx >= cl.GetNumberOfPoints() - 1:
#         idx_1 = cl.GetNumberOfPoints() - 1 - n_samples
#         idx_2 = cl.GetNumberOfPoints() - 1
#     else:
#         idx_1 = max(0, idx - n_samples // 2)
#         idx_2 = min(cl.GetNumberOfPoints() - 1, idx + n_samples // 2)
#
#     p_1 = cl.GetPoint(idx_1)
#     p_2 = cl.GetPoint(idx_2)
#     normal = np.subtract(p_1, p_2)
#     norm_length = np.linalg.norm(normal)
#     if norm_length < 0.001:
#         print("Normal estimation issue. Normal length too small")
#         return None
#         # normal = [1, 0, 0]
#     else:
#         normal = np.divide(normal, norm_length)
#     return normal

def find_position_on_centerline_based_on_scalar(cl, value):
    """
    Find the position on the centerline that is closest to the given scalar value
    The scalar values are assumed to be monotonically increasing or decreasing
    """
    for idx in range(0, cl.GetNumberOfPoints() - 1):
        cl_val_1 = cl.GetPointData().GetScalars().GetValue(idx)
        cl_val_2 = cl.GetPointData().GetScalars().GetValue(idx + 1)
        if value == cl_val_1:
            return idx, cl_val_1
        if value == cl_val_2:
            return idx + 1, cl_val_2
        if cl_val_1 < value <= cl_val_2:
            if abs(cl_val_1 - value) < abs(cl_val_2 - value):
                return idx, cl_val_1
            else:
                return idx, cl_val_2

    cl_val_1 = cl.GetPointData().GetScalars().GetValue(0)
    cl_val_2 = cl.GetPointData().GetScalars().GetValue(cl.GetNumberOfPoints() - 1)
    if abs(cl_val_1 - value) < abs(cl_val_2 - value):
        print(f"Could not find value {value} on centerline. Returning closest point")
        return 0, cl_val_1
    else:
        print(f"Could not find value {value} on centerline. Returning closest point")
        return cl.GetNumberOfPoints() - 1, cl_val_2


def compute_single_straightened_volume_using_cpr(
    cl, ct_img, label_img, img_straight_name, label_straight_name, verbose=False
):
    """ """
    # slice_resolution = [1.0, 1.0, 1.0]
    slice_resolution = [0.5, 0.5, 0.5]
    # slice_resolution = [0.25, 0.25, 0.25]
    slice_size_mm = [70.0, 70.0]
    # outputSpacingMm = 1.0
    outputSpacingMm = 0.5
    # outputSpacingMm = 0.25
    convert_system = False

    if verbose:
        print(
            f"Computing straightened volume with spacing {slice_resolution} and size {slice_size_mm} mm and output spacing {outputSpacingMm} mm"
        )
    cpr = CurvedPlanarReformat(slice_resolution, slice_size_mm, convert_system)
    transform = cpr.compute_straightening_transform(
        cl, slice_size_mm, outputSpacingMm, convert_system
    )

    if verbose:
        print("Performing straightening")

    cpr.straighten_volume(
        ct_img,
        transform,
        slice_resolution,
        isLabelmap=False,
        file_name=img_straight_name,
    )
    cpr.straighten_volume(
        label_img,
        transform,
        slice_resolution,
        isLabelmap=True,
        file_name=label_straight_name,
    )

    return True


def sample_along_single_straight_labelmap(
    straight_label_in,
    straight_volume_in,
    cl_sampling_out,
    min_cl_dist=None,
    max_cl_dist=None,
    verbose=False,
):
    """
    Finds cross-sectional cuts on the surface of the aorta sampled along the centerline.
    Here the straightened version of the label map is used.
    We use the distance along the center line as a kind of ground truth, when comparing
    the straight versus the original centerline. The renal section is for example defined using CL distances.
    We use the original volume to check for out-of-scan values (typically -2048)
    """
    label_img, _ = read_nifti_file_robustly(straight_label_in)
    if label_img is None:
        return False
    ct_img, _ = read_nifti_file_robustly(straight_volume_in)
    if ct_img is None:
        return False

    # try:
    #     label_img = sitk.ReadImage(straight_label_in)
    # except RuntimeError as e:
    #     print(f"Got an exception {str(e)}")
    #     print(f"Error reading {straight_label_in}")
    #     return False
    #
    # try:
    #     ct_img = sitk.ReadImage(straight_volume_in)
    # except RuntimeError as e:
    #     print(f"Got an exception {str(e)}")
    #     print(f"Error reading {straight_volume_in}")
    #     return False

    label_img_np = sitk.GetArrayFromImage(label_img)
    label_img_np = label_img_np.transpose(2, 1, 0)
    ct_img_np = sitk.GetArrayFromImage(ct_img)
    ct_img_np = ct_img_np.transpose(2, 1, 0)

    dims = label_img_np.shape

    spacing = label_img.GetSpacing()
    n_slices = dims[2]

    cl_sampling_out_file = open(cl_sampling_out, "w")

    if min_cl_dist is None:
        min_cl_dist = 0
    if max_cl_dist is None:
        max_cl_dist = (n_slices + 1) * spacing[2]

    for idx in range(n_slices):
        # moving along the center
        cl_dist = idx * spacing[2]

        if min_cl_dist <= cl_dist <= max_cl_dist:
            single_slice_np = label_img_np[:, :, idx]
            single_slice_ct = ct_img_np[:, :, idx]

            # only keep one connected component in the slice. The one that is in the middle
            slice_components = label(single_slice_np)
            mid_p = [dims[0] // 2, dims[1] // 2]
            mid_label = slice_components[mid_p[0], mid_p[1]]
            # The biggest label is not even centered...ignore it
            if mid_label == 0:
                cut_area = 0
            else:
                largest_cc = slice_components == mid_label
                cut_area = np.sum(largest_cc) * spacing[0] * spacing[1]

            # We use the value -2048 in the sample function to mark out of scan
            # also check if org value is less that -2000 for out of scan
            out_of_scan_region = (single_slice_np < 0) | (single_slice_ct < -2000)
            out_of_scan_area = np.sum(out_of_scan_region)
            out_of_scan_percent = out_of_scan_area / (dims[0] * dims[1]) * 100.0

            cl_sampling_out_file.write(
                f"{cl_dist}, {cut_area}, {out_of_scan_percent}\n"
            )

    cl_sampling_out_file.close()
    return True


def compute_diameters_from_contour(contour, org, pix_spacing):
    n_points = len(contour)

    min_diameter = np.inf
    max_diameter = -np.inf
    min_d_idx_1 = -1
    min_d_idx_2 = -1
    max_d_idx_1 = -1
    max_d_idx_2 = -1

    for idx in range(n_points):
        p = contour[idx]
        # l_1 = np.subtract(p, org)
        # l_2_end = np.subtract(org, 10 * l_1)

        dist_p_to_org = np.linalg.norm(p - org)

        # Find closest point to the line from origin to the opposite point
        # this is the opposite point to the line going through the center
        min_dist = np.inf
        min_idx = -1
        for idx_2 in range(n_points):
            p_2 = contour[idx_2]
            # line = vtk.vtkLine()
            # t = vtk.reference(0)
            # cp = vtk.reference(0)
            # dist = line.DistanceToLine(p_2, org, l_2_end, t, cp)

            # Only consider points on opposite side
            dist_p_p_2 = np.linalg.norm(p - p_2)
            if dist_p_p_2 > dist_p_to_org:
                # https://stackoverflow.com/questions/39840030/distance-between-point-and-a-line-from-two-points
                dist = np.abs(np.cross(org - p, org - p_2)) / np.linalg.norm(org - p)

                if dist < min_dist:
                    min_dist = dist
                    min_idx = idx_2

        p_opp = contour[min_idx]
        diameter = np.linalg.norm(np.subtract(p, p_opp))
        if diameter > max_diameter:
            max_diameter = diameter
            max_d_idx_1 = idx
            max_d_idx_2 = min_idx
        if diameter < min_diameter:
            min_diameter = diameter
            min_d_idx_1 = idx
            min_d_idx_2 = min_idx

    # Turn into physical measures
    max_diameter = max_diameter * pix_spacing
    min_diameter = min_diameter * pix_spacing

    diam_stats = {
        "min_diameter": min_diameter,
        "max_diameter": max_diameter,
        "max_d_p1": list(contour[max_d_idx_1]),
        "max_d_p2": list(contour[max_d_idx_2]),
        "min_d_p1": list(contour[min_d_idx_1]),
        "min_d_p2": list(contour[min_d_idx_2]),
    }
    return diam_stats


def set_window_and_level_on_single_slice(img_in, img_window, img_level):
    out_min = 0
    out_max = 1
    in_min = img_level - img_window / 2
    in_max = img_level + img_window / 2
    # in_max = 800

    # https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.rescale_intensity
    out_img = rescale_intensity(
        img_in, in_range=(in_min, in_max), out_range=(out_min, out_max)
    )

    return out_img


def extract_max_cut_in_defined_section(
    cl_dir,
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
    find_minimum=False,
    verbose=False,
):
    max_slice_out_rgb = f"{cl_dir}{segment_name}_max_slice_rgb.png"
    max_slice_out_rgb_crop = f"{cl_dir}{segment_name}_max_slice_rgb_crop.png"
    max_slice_out_info = f"{cl_dir}{segment_name}_max_slice_info.json"
    debug = False

    if (
        start_cl_dist == np.inf
        or start_cl_dist == -np.inf
        or end_cl_dist == np.inf
        or end_cl_dist == -np.inf
    ):
        if verbose:
            print(f"Can not compute {segment_name} since distances are not defined.")
        return True, ""

    n_slices = len(cl_sampling)

    max_a = -np.inf
    max_idx = -1
    max_dist = -1
    if find_minimum:
        max_a = np.inf

    for idx in range(n_slices):
        cl_dist = idx * spacing[2]
        cl_dist_check = cl_sampling[idx][0]
        if cl_dist_check != cl_dist:
            msg = f"Something wrong in sampling: cl_dist {cl_dist} and cl_dist_check {cl_dist_check}"
            return False, msg

        area = cl_sampling[idx][1]
        out_of_scan_percent = cl_sampling[idx][2]
        if out_of_scan_percent < 20 and start_cl_dist <= cl_dist <= end_cl_dist:
            if find_minimum:
                if area < max_a:
                    max_a = area
                    max_idx = idx
                    max_dist = cl_dist
            else:
                if area > max_a:
                    max_a = area
                    max_idx = idx
                    max_dist = cl_dist

    max_slice_info = {"area": max_a, "cl_dist": max_dist, "straight_idx": max_idx}
    if max_idx < 0:
        msg = f"Could not find max cut for {segment_name} in range {start_cl_dist} to {end_cl_dist}"
        return False, msg
    else:
        if debug:
            print(
                f"Found max {segment_name} area {max_a} at idx {max_idx} dist {max_dist}"
            )

    # Extract single slice
    single_slice_np = label_img_np[:, :, max_idx]
    # # only keep one connected component in the slice. The one that is in the middle
    slice_components = label(single_slice_np)
    mid_p = [dims[0] // 2, dims[1] // 2]
    mid_label = slice_components[mid_p[0], mid_p[1]]
    largest_cc = slice_components == mid_label
    if np.sum(largest_cc) < 1:
        msg = f"Could not find contour in {segment_name} the biggest label is not centered."
        return False, msg

    contours = find_contours(img_as_ubyte(largest_cc), 0.5)
    if len(contours) < 1:
        msg = f"Could not find contour in {segment_name}"
        return False, msg

    # In rare cases there can be more than one contour found (probably du to cavities) - we use the longest
    max_l = -np.inf
    contour = None
    for i in range(len(contours)):
        if len(contours[i]) > max_l:
            max_l = len(contours[i])
            contour = contours[i]

    # contour = find_contours(img_as_ubyte(largest_cc), 0.5)[0]
    # contour = contours[0]
    org = mid_p
    pix_spacing = spacing[0]
    if spacing[0] != spacing[1]:
        msg = f"Slice spacing not isotropic: {spacing[0]} and {spacing[1]}"
        return False, msg

    diam_stats = compute_diameters_from_contour(contour, org, pix_spacing)
    max_slice_info["min_diameter"] = diam_stats["min_diameter"]
    max_slice_info["max_diameter"] = diam_stats["max_diameter"]

    no_check = [
        "lvot_segment",
        "sinus_of_valsalva_segment",
        "sinotubular_junction_segment",
        "lvot_segment_ts_original",
        "sinus_of_valsalva_segment_ts_original",
        "sinotubular_junction_segment_ts_original",
    ]

    if segment_name not in no_check:
        if diam_stats["min_diameter"] == 0:
            msg = (
                f"Minimum diameter for cut for {segment_name} is 0 - something is wrong"
            )
            max_slice_info["diameter_ratio"] = 0
            return False, msg
        else:
            diam_ratio = diam_stats["max_diameter"] / diam_stats["min_diameter"]
            max_slice_info["diameter_ratio"] = diam_ratio
            if diam_ratio > 1.5:
                msg = f"Diameter ratio for cut for {segment_name} is {diam_ratio:.2f} - we do not use it"
                return False, msg

    boundary = find_boundaries(largest_cc, mode="outer")
    # skimage.io.imsave(max_slice_boundary_out, boundary)
    single_slice_np_img = straight_img_np[:, :, max_idx]

    single_slice_np_img = set_window_and_level_on_single_slice(
        single_slice_np_img, img_window, img_level
    )
    scaled_ubyte = img_as_ubyte(single_slice_np_img)
    # skimage.io.imsave(max_slice_out, scaled_ubyte)

    scaled_2_rgb = color.gray2rgb(scaled_ubyte)
    rgb_boundary = [255, 0, 0]
    rgb_line = [255, 255, 0]

    rr, cc = line(
        int(diam_stats["max_d_p1"][0]),
        int(diam_stats["max_d_p1"][1]),
        int(diam_stats["max_d_p2"][0]),
        int(diam_stats["max_d_p2"][1]),
    )
    scaled_2_rgb[rr, cc] = rgb_line

    rr, cc = line(
        int(diam_stats["min_d_p1"][0]),
        int(diam_stats["min_d_p1"][1]),
        int(diam_stats["min_d_p2"][0]),
        int(diam_stats["min_d_p2"][1]),
    )
    scaled_2_rgb[rr, cc] = rgb_line

    # Draw boundary last for visual style
    scaled_2_rgb[boundary > 0] = rgb_boundary
    skimage.io.imsave(max_slice_out_rgb, scaled_2_rgb)

    region_p = regionprops(img_as_ubyte(boundary))
    if len(region_p) < 1:
        msg = f"No regions found in boundary for {segment_name}"
        return False, msg
    bbox = list(region_p[0].bbox)

    shp = boundary.shape
    # Extend bbox range.
    # TODO set value elsewhere
    extend = 10
    bbox[0] = max(0, bbox[0] - extend)
    bbox[1] = max(0, bbox[1] - extend)
    bbox[2] = min(shp[0], bbox[2] + extend)
    bbox[3] = min(shp[1], bbox[3] + extend)

    scaled_2_rgb_crop = scaled_2_rgb[bbox[0] : bbox[2], bbox[1] : bbox[3]]
    skimage.io.imsave(max_slice_out_rgb_crop, scaled_2_rgb_crop)

    cl_idx, cl_val = find_position_on_centerline_based_on_scalar(cl, max_dist)
    cl_point = cl.GetPoint(cl_idx)
    cl_normal = get_tangent_from_centerline(cl, cl_idx)
    max_slice_info["origin"] = list(cl_point)
    max_slice_info["normal"] = list(cl_normal)

    json_object = json.dumps(max_slice_info, indent=4)
    with open(max_slice_out_info, "w") as outfile:
        outfile.write(json_object)

    return True, ""


def compute_tortuosity_index_based_on_scan_type(
    cl_folder, lm_folder, stats_folder, verbose, quiet, write_log_file, output_folder
):
    """
    Compute tortuosity index based on scan type, where the type is defined here:
    https://github.com/RasmusRPaulsen/AortaExplorer
    """
    ventri_in = f"{cl_folder}ventricularaortic.json"
    infrarenal_in = f"{cl_folder}infrarenal_section.json"
    diaphragm_in = f"{cl_folder}diaphragm.json"
    aortic_arch_in = f"{cl_folder}aortic_arch.json"
    stats_file = f"{stats_folder}aorta_scan_type.json"
    start_p_name = f"{lm_folder}aorta_start_point.txt"
    end_p_name = f"{lm_folder}aorta_end_point.txt"
    # debug = False

    if verbose:
        print("computing tortuosity index")
    ati_stats = {}

    if not os.path.exists(stats_file):
        msg = f"Missing file {stats_file} - cannot compute tortuosity"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return None

    scan_type_stats = read_json_file(stats_file)
    if not scan_type_stats:
        msg = f"Could not read file {stats_file} - cannot compute tortuosity"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(
                base_dir=output_folder, message=msg, level="error"
            )
        return None

    scan_type = scan_type_stats["scan_type"]
    scan_type_desc = scan_type_stats["scan_type_desc"]
    if verbose:
        print(f"Computing tortuosity index for scan type {scan_type}: {scan_type_desc}")

    # Full aorta
    if scan_type in ["1", "1b", "1c"]:
        cl_file = f"{cl_folder}aorta_centerline.vtp"
        if not os.path.exists(cl_file):
            msg = f"Missing file {cl_file} - cannot compute tortuosity"
            if not quiet:
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
        if cl.GetNumberOfPoints() < 2:
            msg = f"Centerline in {cl_file} has less than 2 points - cannot compute tortuosity"
            if not quiet:
                print(msg)
            if write_log_file:
                write_message_to_log_file(
                    base_dir=output_folder, message=msg, level="error"
                )
            return None

        # The start point of the centerline
        start_cl_p = cl.GetPoint(0)

        # The start point between the iliac artery and the aorta
        start_p = read_landmarks(start_p_name)
        add_distance_start = np.linalg.norm(np.array(start_p) - np.array(start_cl_p))

        # The point at the annulus (the ventrial-aortic junction)
        ventri = read_json_file(ventri_in)
        if not ventri:
            msg = f"Missing file {ventri_in} - cannot compute tortuosity"
            if not quiet:
                print(msg)
            if write_log_file:
                write_message_to_log_file(
                    base_dir=output_folder, message=msg, level="error"
                )
            return ati_stats

        ventri_cl_dist = ventri["ventri_cl_dist"]
        ventri_pos = ventri["ventri_cl_pos"]
        geometric_length = np.linalg.norm(np.array(start_p) - np.array(ventri_pos))
        aortic_length = ventri_cl_dist + add_distance_start

        if geometric_length > 0 and aortic_length > 0:
            aortic_tortuosity_index = aortic_length / geometric_length
            if verbose:
                print(f"Computed Annulus tortuosity index: {aortic_tortuosity_index}")
            ati_stats["annulus_aortic_tortuosity_index"] = aortic_tortuosity_index
            ati_stats["annulus_aortic_length"] = aortic_length
            ati_stats["annulus_geometric_length"] = geometric_length

        arch = read_json_file(aortic_arch_in)
        if not arch:
            msg = f"Missing file {aortic_arch_in} - cannot compute tortuosity"
            if not quiet:
                print(msg)
            if write_log_file:
                write_message_to_log_file(
                    base_dir=output_folder, message=msg, level="error"
                )
            return None
        ascending_cl_dist = arch["max_cl_dist"]
        ascending_pos = arch["max_cl_pos"]
        geometric_length = np.linalg.norm(
            np.array(ascending_pos) - np.array(ventri_pos)
        )
        aortic_length = ventri_cl_dist - ascending_cl_dist

        if geometric_length > 0 and aortic_length > 0:
            aortic_tortuosity_index = aortic_length / geometric_length
            if verbose:
                print(f"Computed ascending tortuosity index: {aortic_tortuosity_index}")
            ati_stats["ascending_aortic_tortuosity_index"] = aortic_tortuosity_index
            ati_stats["ascending_aortic_length"] = aortic_length
            ati_stats["ascending_geometric_length"] = geometric_length

        infrarenal = read_json_file(infrarenal_in)
        if infrarenal:
            infrarenal_cl_dist = infrarenal["distance"]
            infrarenal_pos = infrarenal["cl_pos"]
            geometric_length = np.linalg.norm(
                np.array(start_p) - np.array(infrarenal_pos)
            )
            # We add the start of the centerline to the true start point of the aorta
            aortic_length = infrarenal_cl_dist + add_distance_start
            if geometric_length > 0 and aortic_length > 0:
                aortic_tortuosity_index = aortic_length / geometric_length
                if verbose:
                    print(
                        f"Computed infrarenal tortuosity index: {aortic_tortuosity_index}"
                    )
                ati_stats["infrarenal_aortic_tortuosity_index"] = (
                    aortic_tortuosity_index
                )
                ati_stats["infrarenal_aortic_length"] = aortic_length
                ati_stats["infrarenal_geometric_length"] = geometric_length

        # Abdominal ATI from diaphragm to iliac bifurcation
        diaphragm = read_json_file(diaphragm_in)
        if diaphragm:
            diaphragm_cl_dist = diaphragm["diaphragm_cl_dist"]
            diaphragm_pos = diaphragm["diaphragm_cl_pos"]
            geometric_length = np.linalg.norm(
                np.array(start_p) - np.array(diaphragm_pos)
            )
            # We add the start of the centerline to the true start point of the aorta
            aortic_length = diaphragm_cl_dist + add_distance_start
            if geometric_length > 0 and aortic_length > 0:
                aortic_tortuosity_index = aortic_length / geometric_length
                if verbose:
                    print(
                        f"Computed abdominal tortuosity index: {aortic_tortuosity_index}"
                    )
                ati_stats["abdominal_aortic_tortuosity_index"] = aortic_tortuosity_index
                ati_stats["abdominal_aortic_length"] = aortic_length
                ati_stats["abdominal_geometric_length"] = geometric_length

            # Descending from diaphragm to subclavian artery (arch min dist)
            arch_min_cl_dist = arch["min_cl_dist"]
            arch_min_pos = arch["min_cl_pos"]
            geometric_length = np.linalg.norm(
                np.array(diaphragm_pos) - np.array(arch_min_pos)
            )
            aortic_length = arch_min_cl_dist - diaphragm_cl_dist

            if geometric_length > 0 and aortic_length > 0:
                aortic_tortuosity_index = aortic_length / geometric_length
                if verbose:
                    print(
                        f"Computed descending tortuosity index: {aortic_tortuosity_index}"
                    )
                ati_stats["descending_aortic_tortuosity_index"] = (
                    aortic_tortuosity_index
                )
                ati_stats["descending_aortic_length"] = aortic_length
                ati_stats["descending_geometric_length"] = geometric_length

            # IB-ARCH from iliac bifurcation to subclavian artery (arch min dist)
            geometric_length = np.linalg.norm(
                np.array(start_p) - np.array(arch_min_pos)
            )
            aortic_length = arch_min_cl_dist + add_distance_start

            if geometric_length > 0 and aortic_length > 0:
                aortic_tortuosity_index = aortic_length / geometric_length
                if verbose:
                    print(
                        f"Computed iliac bifucation to arch tortuosity index: {aortic_tortuosity_index}"
                    )
                ati_stats["ib_arch_aortic_tortuosity_index"] = aortic_tortuosity_index
                ati_stats["ib_arch_aortic_length"] = aortic_length
                ati_stats["ib_arch_geometric_length"] = geometric_length

    # 2: Lower aorta with iliac bifurcation
    elif scan_type == "2":
        cl_file = f"{cl_folder}aorta_centerline.vtp"
        if not os.path.exists(cl_file):
            msg = f"Missing file {cl_file} - cannot compute tortuosity"
            if not quiet:
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
        if cl.GetNumberOfPoints() < 2:
            msg = f"Centerline in {cl_file} has less than 2 points - cannot compute tortuosity"
            if not quiet:
                print(msg)
            if write_log_file:
                write_message_to_log_file(
                    base_dir=output_folder, message=msg, level="error"
                )
            return None

        # The start point of the centerline
        start_cl_p = cl.GetPoint(0)
        end_cl_p = cl.GetPoint(cl.GetNumberOfPoints() - 1)

        # The start point between the iliac artery and the aorta
        start_p = read_landmarks(start_p_name)

        # The end point is where the aorta hits the border of the scan
        end_p = read_landmarks(end_p_name)
        add_distance_start = np.linalg.norm(np.array(start_p) - np.array(start_cl_p))
        add_distance_end = np.linalg.norm(np.array(end_p) - np.array(end_cl_p))

        infrarenal = read_json_file(infrarenal_in)
        if infrarenal:
            infrarenal_cl_dist = infrarenal["distance"]
            infrarenal_pos = infrarenal["cl_pos"]
            geometric_length = np.linalg.norm(
                np.array(start_p) - np.array(infrarenal_pos)
            )
            # We add the start of the centerline to the true start point of the aorta
            aortic_length = infrarenal_cl_dist + add_distance_start
            if geometric_length > 0 and aortic_length > 0:
                aortic_tortuosity_index = aortic_length / geometric_length
                if verbose:
                    print(
                        f"Computed infrarenal tortuosity index: {aortic_tortuosity_index}"
                    )
                ati_stats["infrarenal_aortic_tortuosity_index"] = (
                    aortic_tortuosity_index
                )
                ati_stats["infrarenal_aortic_length"] = aortic_length
                ati_stats["infrarenal_geometric_length"] = geometric_length

        # Abdominal ATI: From IB to diaphragm if present else to top of scan
        diaphragm = read_json_file(diaphragm_in)
        if diaphragm:
            diaphragm_cl_dist = diaphragm["diaphragm_cl_dist"]
            diaphragm_pos = diaphragm["diaphragm_cl_pos"]
            geometric_length = np.linalg.norm(
                np.array(start_p) - np.array(diaphragm_pos)
            )
            # We add the start of the centerline to the true start point of the aorta
            aortic_length = diaphragm_cl_dist + add_distance_start
            if geometric_length > 0 and aortic_length > 0:
                aortic_tortuosity_index = aortic_length / geometric_length
                if verbose:
                    print(
                        f"Computed abdominal tortuosity index: {aortic_tortuosity_index}"
                    )
                ati_stats["abdominal_aortic_tortuosity_index"] = aortic_tortuosity_index
                ati_stats["abdominal_aortic_length"] = aortic_length
                ati_stats["abdominal_geometric_length"] = geometric_length
        else:
            aortic_length = (
                cl.GetPointData().GetScalars().GetValue(cl.GetNumberOfPoints() - 1)
                + add_distance_start
                + add_distance_end
            )
            geometric_length = np.linalg.norm(np.array(start_p) - np.array(end_p))
            if geometric_length > 0 and aortic_length > 0:
                aortic_tortuosity_index = aortic_length / geometric_length
                if verbose:
                    print(
                        f"Computed abdominal tortuosity index (no diaphragm): {aortic_tortuosity_index}"
                    )
                ati_stats["abdominal_aortic_tortuosity_index"] = aortic_tortuosity_index
                ati_stats["abdominal_aortic_length"] = aortic_length
                ati_stats["abdominal_geometric_length"] = geometric_length
    # Two parts (cardiac)
    elif scan_type == "5":
        # Start point is the top of the scan
        start_p_name = f"{lm_folder}aorta_start_point_annulus.txt"
        cl_file = f"{cl_folder}aorta_centerline_annulus.vtp"
        if not os.path.exists(cl_file):
            print(f"Missing {cl_file} - cannot compute tortuosity")
            return None

        pd = vtk.vtkXMLPolyDataReader()
        pd.SetFileName(cl_file)
        pd.Update()
        cl = pd.GetOutput()
        if cl.GetNumberOfPoints() < 2:
            print("Centerline with too few points - cannot compute tortuosity")
            return None

        # The start point of the centerline (close to top of scan)
        start_cl_p = cl.GetPoint(0)

        # The start point between the iliac artery and the aorta
        start_p = read_landmarks(start_p_name)
        add_distance_start = np.linalg.norm(np.array(start_p) - np.array(start_cl_p))

        # The point at the annulus (the ventrial-aortic junction)
        ventri = read_json_file(ventri_in)
        if not ventri:
            print("Missing ventricular-aortic junction - cannot compute tortuosity")
            return ati_stats

        ventri_cl_dist = ventri["ventri_cl_dist"]
        ventri_pos = ventri["ventri_cl_pos"]
        geometric_length = np.linalg.norm(np.array(start_p) - np.array(ventri_pos))
        aortic_length = ventri_cl_dist + add_distance_start

        if geometric_length > 0 and aortic_length > 0:
            aortic_tortuosity_index = aortic_length / geometric_length
            if verbose:
                print(f"Computed ascending tortuosity index: {aortic_tortuosity_index}")
            ati_stats["ascending_aortic_tortuosity_index"] = aortic_tortuosity_index
            ati_stats["ascending_aortic_length"] = aortic_length
            ati_stats["ascending_geometric_length"] = geometric_length

        # Second part - the descending part
        # Start point is at the bottom of the scan
        start_p_name = f"{lm_folder}aorta_start_point_descending.txt"
        end_p_name = f"{lm_folder}aorta_end_point_descending.txt"

        cl_file = f"{cl_folder}aorta_centerline_descending.vtp"
        if not os.path.exists(cl_file):
            print(f"Missing {cl_file} - cannot compute tortuosity")
            return None

        pd = vtk.vtkXMLPolyDataReader()
        pd.SetFileName(cl_file)
        pd.Update()
        cl = pd.GetOutput()
        if cl.GetNumberOfPoints() < 2:
            print("Centerline with too few points - cannot compute tortuosity")
            return None

        # The start point at the bottom of the scan
        start_p = read_landmarks(start_p_name)

        # The start point of the centerline (close to bottom of scan)
        start_cl_p = cl.GetPoint(0)
        # The end point of the centerline (close to top of scan)
        end_cl_p = cl.GetPoint(cl.GetNumberOfPoints() - 1)

        end_p = read_landmarks(end_p_name)
        add_distance_start = np.linalg.norm(np.array(start_p) - np.array(start_cl_p))
        add_distance_end = np.linalg.norm(np.array(end_p) - np.array(end_cl_p))

        # Descending from top of scan to diaphragm if present else to bottom to scn
        diaphragm = read_json_file(diaphragm_in)
        if diaphragm:
            diaphragm_cl_dist = diaphragm["diaphragm_cl_dist"]
            diaphragm_pos = diaphragm["diaphragm_cl_pos"]
            geometric_length = np.linalg.norm(np.array(end_p) - np.array(diaphragm_pos))
            aortic_length = (
                cl.GetPointData().GetScalars().GetValue(cl.GetNumberOfPoints() - 1)
                + add_distance_end
                - diaphragm_cl_dist
            )
            if geometric_length > 0 and aortic_length > 0:
                aortic_tortuosity_index = aortic_length / geometric_length
                if verbose:
                    print(
                        f"Computed descending tortuosity index: {aortic_tortuosity_index}"
                    )
                ati_stats["descending_aortic_tortuosity_index"] = (
                    aortic_tortuosity_index
                )
                ati_stats["descending_aortic_length"] = aortic_length
                ati_stats["descending_geometric_length"] = geometric_length
        else:
            geometric_length = np.linalg.norm(np.array(start_p) - np.array(end_p))
            aortic_length = (
                cl.GetPointData().GetScalars().GetValue(cl.GetNumberOfPoints() - 1)
                + add_distance_start
                + add_distance_end
            )

            if geometric_length > 0 and aortic_length > 0:
                aortic_tortuosity_index = aortic_length / geometric_length
                if verbose:
                    print(
                        f"Computed descending tortuosity index: {aortic_tortuosity_index}"
                    )
                ati_stats["descending_aortic_tortuosity_index"] = (
                    aortic_tortuosity_index
                )
                ati_stats["descending_aortic_length"] = aortic_length
                ati_stats["descending_geometric_length"] = geometric_length
    # Two parts (cardiac) but also bottom of aorta
    elif scan_type == "4":
        # Start point is the top of the scan
        start_p_name = f"{lm_folder}aorta_start_point_annulus.txt"
        cl_file = f"{cl_folder}aorta_centerline_annulus.vtp"
        if not os.path.exists(cl_file):
            print(f"Missing {cl_file} - cannot compute tortuosity")
            return None

        pd = vtk.vtkXMLPolyDataReader()
        pd.SetFileName(cl_file)
        pd.Update()
        cl = pd.GetOutput()
        if cl.GetNumberOfPoints() < 2:
            print("Centerline with too few points - cannot compute tortuosity")
            return None

        # The start point of the centerline (close to top of scan)
        start_cl_p = cl.GetPoint(0)

        start_p = read_landmarks(start_p_name)
        add_distance_start = np.linalg.norm(np.array(start_p) - np.array(start_cl_p))

        # The point at the annulus (the ventrial-aortic junction)
        ventri = read_json_file(ventri_in)
        if not ventri:
            print("Missing ventricular-aortic junction - cannot compute tortuosity")
            return ati_stats

        ventri_cl_dist = ventri["ventri_cl_dist"]
        ventri_pos = ventri["ventri_cl_pos"]
        geometric_length = np.linalg.norm(np.array(start_p) - np.array(ventri_pos))
        aortic_length = ventri_cl_dist + add_distance_start

        if geometric_length > 0 and aortic_length > 0:
            aortic_tortuosity_index = aortic_length / geometric_length
            if verbose:
                print(f"Computed ascending tortuosity index: {aortic_tortuosity_index}")
            ati_stats["ascending_aortic_tortuosity_index"] = aortic_tortuosity_index
            ati_stats["ascending_aortic_length"] = aortic_length
            ati_stats["ascending_geometric_length"] = geometric_length

        # Second part - the descending part
        start_p_name = f"{lm_folder}aorta_start_point_descending.txt"
        end_p_name = f"{lm_folder}aorta_end_point_descending.txt"

        cl_file = f"{cl_folder}aorta_centerline_descending.vtp"
        if not os.path.exists(cl_file):
            print(f"Missing {cl_file} - cannot compute tortuosity")
            return None

        pd = vtk.vtkXMLPolyDataReader()
        pd.SetFileName(cl_file)
        pd.Update()
        cl = pd.GetOutput()
        if cl.GetNumberOfPoints() < 2:
            print("Centerline with too few points - cannot compute tortuosity")
            return None

        # The start point of the centerline
        start_cl_p = cl.GetPoint(0)
        end_cl_p = cl.GetPoint(cl.GetNumberOfPoints() - 1)

        # The start point between the iliac artery and the aorta
        start_p = read_landmarks(start_p_name)

        # The end point is where the aorta hits the border of the scan
        end_p = read_landmarks(end_p_name)
        add_distance_start = np.linalg.norm(np.array(start_p) - np.array(start_cl_p))
        add_distance_end = np.linalg.norm(np.array(end_p) - np.array(end_cl_p))

        # geometric_length = np.linalg.norm(np.array(start_p) - np.array(end_p))
        # aortic_length = cl.GetPointData().GetScalars().GetValue(cl.GetNumberOfPoints() - 1) + \
        #                 add_distance_start + add_distance_end
        #
        # if geometric_length > 0 and aortic_length > 0:
        #     aortic_tortuosity_index = aortic_length / geometric_length
        #     print(f"Computed descending tortuosity index: {aortic_tortuosity_index}")
        #     ati_stats["descending_aortic_tortuosity_index"] = aortic_tortuosity_index
        #     ati_stats["descending_aortic_length"] = aortic_length
        #     ati_stats["descending_geometric_length"] = geometric_length

        infrarenal = read_json_file(infrarenal_in)
        if infrarenal:
            infrarenal_cl_dist = infrarenal["distance"]
            infrarenal_pos = infrarenal["cl_pos"]
            geometric_length = np.linalg.norm(
                np.array(start_p) - np.array(infrarenal_pos)
            )
            # We add the start of the centerline to the true start point of the aorta
            aortic_length = infrarenal_cl_dist + add_distance_start
            if geometric_length > 0 and aortic_length > 0:
                aortic_tortuosity_index = aortic_length / geometric_length
                if verbose:
                    print(
                        f"Computed infrarenal tortuosity index: {aortic_tortuosity_index}"
                    )
                ati_stats["infrarenal_aortic_tortuosity_index"] = (
                    aortic_tortuosity_index
                )
                ati_stats["infrarenal_aortic_length"] = aortic_length
                ati_stats["infrarenal_geometric_length"] = geometric_length

        diaphragm = read_json_file(diaphragm_in)
        if diaphragm:
            diaphragm_cl_dist = diaphragm["diaphragm_cl_dist"]
            diaphragm_pos = diaphragm["diaphragm_cl_pos"]
            geometric_length = np.linalg.norm(
                np.array(start_p) - np.array(diaphragm_pos)
            )
            # We add the start of the centerline to the true start point of the aorta
            aortic_length = diaphragm_cl_dist + add_distance_start
            if geometric_length > 0 and aortic_length > 0:
                aortic_tortuosity_index = aortic_length / geometric_length
                if verbose:
                    print(
                        f"Computed diaphragm tortuosity index: {aortic_tortuosity_index}"
                    )
                ati_stats["diaphragm_aortic_tortuosity_index"] = aortic_tortuosity_index
                ati_stats["diaphragm_aortic_length"] = aortic_length
                ati_stats["diaphragm_geometric_length"] = geometric_length

            # geometric_length = np.linalg.norm(np.array(start_p) - np.array(end_p))
            # aortic_length = cl.GetPointData().GetScalars().GetValue(cl.GetNumberOfPoints() - 1) + \
            #                 add_distance_start + add_distance_end
            #
            # if geometric_length > 0 and aortic_length > 0:
            #     aortic_tortuosity_index = aortic_length / geometric_length
            #     print(f"Computed descending tortuosity index: {aortic_tortuosity_index}")
            #     ati_stats["descending_aortic_tortuosity_index"] = aortic_tortuosity_index
            #     ati_stats["descending_aortic_length"] = aortic_length
            #     ati_stats["descending_geometric_length"] = geometric_length

            # Descending from diaphragm to top of scan
            top_cl_dist = (
                cl.GetPointData().GetScalars().GetValue(cl.GetNumberOfPoints() - 1)
                + add_distance_start
                + add_distance_end
            )
            top_pos = end_p
            # arch_min_cl_dist = arch["min_cl_dist"]
            # arch_min_pos = arch["min_cl_pos"]
            geometric_length = np.linalg.norm(
                np.array(diaphragm_pos) - np.array(top_pos)
            )
            aortic_length = top_cl_dist - diaphragm_cl_dist

            if geometric_length > 0 and aortic_length > 0:
                aortic_tortuosity_index = aortic_length / geometric_length
                print(
                    f"Computed descending tortuosity index: {aortic_tortuosity_index}"
                )
                ati_stats["descending_aortic_tortuosity_index"] = (
                    aortic_tortuosity_index
                )
                ati_stats["descending_aortic_length"] = aortic_length
                ati_stats["descending_geometric_length"] = geometric_length
    else:
        print(
            f"Scan type {scan_type}: {scan_type_desc} not supported for tortuosity index"
        )
        return ati_stats

    return ati_stats
