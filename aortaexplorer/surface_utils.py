import numpy as np
import vtk
from vtk.vtkCommonCore import vtkMath
import SimpleITK as sitk
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
import aortaexplorer.general_utils as gu
from aortaexplorer.io_utils import read_nifti_file_robustly


def sitk2vtk(img, flip_for_volume_rendering=False, debugOn=False):
    """Convert a SimpleITK image to a VTK image, via numpy."""

    size = list(img.GetSize())
    origin = list(img.GetOrigin())
    spacing = list(img.GetSpacing())
    ncomp = img.GetNumberOfComponentsPerPixel()
    direction = img.GetDirection()

    # convert the SimpleITK image to a numpy array
    i2 = sitk.GetArrayFromImage(img)
    if debugOn:
        i2_string = i2.tostring()
        print("data string address inside sitk2vtk", hex(id(i2_string)))

    vtk_image = vtk.vtkImageData()

    # VTK expects 3-dimensional parameters
    if len(size) == 2:
        size.append(1)

    if len(origin) == 2:
        origin.append(0.0)

    if len(spacing) == 2:
        spacing.append(spacing[0])

    if len(direction) == 4:
        direction = [
            direction[0],
            direction[1],
            0.0,
            direction[2],
            direction[3],
            0.0,
            0.0,
            0.0,
            1.0,
        ]

    vtk_image.SetDimensions(size)
    vtk_image.SetSpacing(spacing)
    vtk_image.SetOrigin(origin)
    vtk_image.SetExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)

    if vtk.vtkVersion.GetVTKMajorVersion() < 9:
        print("Warning: VTK version <9.  No direction matrix.")
    else:
        vtk_image.SetDirectionMatrix(direction)

    # TODO: Volume rendering does not support direction matrices (27/5-2023)
    # so sometimes the volume rendering is mirrored
    # this a brutal hack to avoid that
    if flip_for_volume_rendering:
        if direction[4] < 0:
            i2 = np.fliplr(i2)

    # depth_array = numpy_support.numpy_to_vtk(i2.ravel(), deep=True,
    #                                          array_type = vtktype)
    depth_array = numpy_to_vtk(i2.ravel(), deep=True)
    depth_array.SetNumberOfComponents(ncomp)
    vtk_image.GetPointData().SetScalars(depth_array)

    vtk_image.Modified()
    #
    if debugOn:
        print("Volume object inside sitk2vtk")
        print(vtk_image)
        #        print("type = ", vtktype)
        print("num components = ", ncomp)
        print(size)
        print(origin)
        print(spacing)
        print(vtk_image.GetScalarComponentAsFloat(0, 0, 0, 0))

    return vtk_image


def filter_image_with_segmentation(img, mask_img_name, fill_val=-1000):
    i2 = sitk.GetArrayFromImage(img)

    mask, _ = read_nifti_file_robustly(mask_img_name)
    if mask is None:
        print(f"Error reading {mask_img_name}")
        return img

    # try:
    #     mask = sitk.ReadImage(mask_img_name)
    # except RuntimeError as e:
    #     print(f"Got an exception {str(e)}")
    #     print(f"Error reading {mask_img_name}")
    #     return img

    m_np = sitk.GetArrayFromImage(mask)
    m_mask = m_np > 0.5
    i2[~m_mask] = fill_val
    img_o = sitk.GetImageFromArray(i2)
    img_o.CopyInformation(img)

    return img_o


def read_nifti_itk_to_vtk(
    file_name, img_mask_name=None, flip_for_volume_rendering=None
):
    img, _ = read_nifti_file_robustly(file_name)
    if img is None:
        return None

    # try:
    #     img = sitk.ReadImage(file_name)
    # except RuntimeError as e:
    #     print(f"Got an exception {str(e)}")
    #     print(f"Error reading {file_name}")
    #     return None

    if img_mask_name is not None:
        img = filter_image_with_segmentation(img, img_mask_name)

    vtk_image = sitk2vtk(img, flip_for_volume_rendering)
    return vtk_image


def convert_sitk_image_to_surface(sitk_image):
    vtk_img = sitk2vtk(sitk_image)
    if vtk_img is None:
        return None

    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(vtk_img)
    mc.SetValue(0, 0.5)
    mc.Update()

    if mc.GetOutput().GetNumberOfPoints() < 10:
        print("No isosurface found")
        return None

    surface = mc.GetOutput()
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(surface)
    cleaner.Update()
    surface = cleaner.GetOutput()

    return surface


def convert_label_map_to_surface(
    label_name, reset_direction_matrix=False, segment_id=1, only_largest_component=False
):
    debug = False
    vtk_img = read_nifti_itk_to_vtk(label_name)
    if vtk_img is None:
        return None

    # Check if there is any data
    vol_np = vtk_to_numpy(vtk_img.GetPointData().GetScalars())
    # remember that label = -2048 denotes out of scan
    if np.sum(vol_np > 0) < 1:
        if debug:
            print(f"No valid labels in {label_name}")
        return None

    if reset_direction_matrix:
        direction = [1, 0, 0.0, 0, 1, 0.0, 0.0, 0.0, 1.0]
        vtk_img.SetDirectionMatrix(direction)

    mc = vtk.vtkDiscreteMarchingCubes()
    mc.SetInputData(vtk_img)
    mc.SetNumberOfContours(1)
    mc.SetValue(0, segment_id)
    mc.Update()

    if mc.GetOutput().GetNumberOfPoints() < 10:
        if debug:
            print(f"No isosurface found in {label_name} for segment {segment_id}")
        return None

    surface = mc.GetOutput()
    if only_largest_component:
        conn = vtk.vtkConnectivityFilter()
        conn.SetInputConnection(mc.GetOutputPort())
        conn.SetExtractionModeToLargestRegion()
        conn.Update()
        surface = conn.GetOutput()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(surface)
    cleaner.Update()
    surface = cleaner.GetOutput()

    return surface


def compute_min_and_max_z_landmark(surface):
    """
    Takes VTK surface and returns the two points that have the smallest and largest z-value
    """
    # We only want to look at the largest structure (in case of for example split aorta)
    conn = vtk.vtkConnectivityFilter()
    conn.SetInputData(surface)
    conn.SetExtractionModeToLargestRegion()
    conn.Update()

    # Connectivity filter do not remove underlying points only cells
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(conn.GetOutputPort())
    cleaner.Update()

    surface = cleaner.GetOutput()

    n_points = surface.GetNumberOfPoints()
    if n_points < 2:
        print("Not enough points in surface")
        return None, None

    # There can be more than one minimum point if is a flat end surface
    min_p_list = []
    max_p_list = []

    min_z = np.inf
    max_z = -np.inf
    for i in range(n_points):
        p = surface.GetPoint(i)
        z = p[2]
        if z < min_z:
            min_z = z
            min_p_list = [surface.GetPoint(i)]
        elif z == min_z:
            min_p_list.append(surface.GetPoint(i))
        if z > max_z:
            max_z = z
            max_p_list = [surface.GetPoint(i)]
        elif z == max_z:
            max_p_list.append(surface.GetPoint(i))

    min_p = np.mean(np.stack(min_p_list), axis=0)
    max_p = np.mean(np.stack(max_p_list), axis=0)

    return min_p, max_p


def find_closests_points_on_two_surfaces_with_start_point(
    surface_1, surface_2, start_point_surface_1
):
    """
    Find the two points that are closest on each other on the two surfaces
    """
    min_dist = np.inf

    locator_1 = vtk.vtkPointLocator()
    locator_1.SetDataSet(surface_1)
    locator_1.BuildLocator()

    locator_2 = vtk.vtkPointLocator()
    locator_2.SetDataSet(surface_2)
    locator_2.BuildLocator()
    p_1 = start_point_surface_1

    idx_1 = -1
    idx_2 = -1

    stop = False
    while not stop:
        idx_2 = locator_2.FindClosestPoint(p_1)
        p_2 = surface_2.GetPoint(idx_2)
        idx_1 = locator_1.FindClosestPoint(p_2)
        p_1 = surface_1.GetPoint(idx_1)
        dist_squared = vtkMath.Distance2BetweenPoints(p_1, p_2)
        if dist_squared < min_dist:
            min_dist = dist_squared
        else:
            stop = True

    p_1 = surface_1.GetPoint(idx_1)
    p_2 = surface_2.GetPoint(idx_2)
    avg_p = np.mean(np.stack((p_1, p_2)), axis=0)

    return idx_1, idx_2, avg_p, np.sqrt(min_dist)


def preprocess_surface_for_centerline_extraction(vtk_in):
    conn = vtk.vtkConnectivityFilter()
    conn.SetInputData(vtk_in)
    conn.SetExtractionModeToLargestRegion()
    conn.Update()

    # print("Filling holes")
    fill_holes = vtk.vtkFillHolesFilter()
    fill_holes.SetInputData(conn.GetOutput())
    fill_holes.SetHoleSize(1000.0)
    fill_holes.Update()

    # print("Triangle filter")
    triangle = vtk.vtkTriangleFilter()
    triangle.SetInputData(fill_holes.GetOutput())
    triangle.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(triangle.GetOutput())
    cleaner.Update()

    smooth_filter = vtk.vtkSmoothPolyDataFilter()
    smooth_filter.SetInputData(cleaner.GetOutput())
    smooth_filter.SetNumberOfIterations(100)
    smooth_filter.SetRelaxationFactor(0.1)
    smooth_filter.FeatureEdgeSmoothingOff()
    smooth_filter.BoundarySmoothingOn()
    smooth_filter.Update()

    decimate = vtk.vtkDecimatePro()
    decimate.SetInputData(smooth_filter.GetOutput())
    decimate.SetTargetReduction(0.90)
    decimate.PreserveTopologyOn()
    decimate.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(decimate.GetOutput())
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOn()
    normals.SplittingOff()
    normals.Update()
    return normals.GetOutput()


def aorta_volume_properties(
    segm_folder, stats_folder, quiet, write_log_file, output_folder, stats
):
    """
    Compute volume properties of aorta including volume and surface area
    """
    n_aorta_parts = 1
    parts_stats = gu.read_json_file(f"{stats_folder}aorta_parts.json")
    if parts_stats:
        n_aorta_parts = parts_stats["aorta_parts"]

    if n_aorta_parts == 1:
        aorta_segm_file = f"{segm_folder}aorta_lumen.nii.gz"
        aorta_surface = convert_label_map_to_surface(
            aorta_segm_file,
            reset_direction_matrix=False,
            segment_id=1,
            only_largest_component=True,
        )
        if aorta_surface is None:
            msg = f"Could not extract aorta surface from {aorta_segm_file}"
            if write_log_file:
                gu.write_message_to_log_file(
                    base_dir=output_folder, message=msg, level="error"
                )
            if not quiet:
                print(msg)
            return False

        mass = vtk.vtkMassProperties()
        mass.SetInputData(aorta_surface)
        mass.Update()

        stats["surface_volume"] = mass.GetVolume()
        stats["surface_area"] = mass.GetSurfaceArea()
    else:
        aorta_segm_file = f"{segm_folder}aorta_lumen_annulus.nii.gz"
        aorta_surface = convert_label_map_to_surface(
            aorta_segm_file,
            reset_direction_matrix=False,
            segment_id=1,
            only_largest_component=True,
        )
        if aorta_surface is None:
            msg = f"Could not extract aorta surface from {aorta_segm_file}"
            if write_log_file:
                gu.write_message_to_log_file(
                    base_dir=output_folder, message=msg, level="error"
                )
            if not quiet:
                print(msg)
            return False

        mass = vtk.vtkMassProperties()
        mass.SetInputData(aorta_surface)
        mass.Update()

        stats["annulus_surface_volume"] = mass.GetVolume()
        stats["annulus_surface_area"] = mass.GetSurfaceArea()

        aorta_segm_file = f"{segm_folder}aorta_lumen_descending.nii.gz"
        aorta_surface = convert_label_map_to_surface(
            aorta_segm_file,
            reset_direction_matrix=False,
            segment_id=1,
            only_largest_component=True,
        )
        if aorta_surface is None:
            msg = f"Could not extract aorta surface from {aorta_segm_file}"
            if write_log_file:
                gu.write_message_to_log_file(
                    base_dir=output_folder, message=msg, level="error"
                )
            if not quiet:
                print(msg)
            return False

        mass = vtk.vtkMassProperties()
        mass.SetInputData(aorta_surface)
        mass.Update()

        stats["descending_surface_volume"] = mass.GetVolume()
        stats["descending_surface_area"] = mass.GetSurfaceArea()

    return True
