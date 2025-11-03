from typing import Optional, Sequence, Tuple, List
import numpy as np
import pathlib
from aortaexplorer.general_utils import write_message_to_log_file
import numpy as np
import SimpleITK as sitk
import time
import edt
# import cupy as cp
# from cupyx.scipy.ndimage import distance_transform_edt
# from cupyx.scipy.ndimage import binary_closing
# import scipy.ndimage
from skimage.morphology import skeletonize
import vtk
from skimage.measure import label
from scipy.interpolate import UnivariateSpline


class AortaCenterliner:
    """
    Class for aorta centerline extraction.

    Responsibilities (to be implemented):
    - Takes as input a label_map as a SimpleITK image
    - aorta_type as a string
    """

    def __init__(self, label_map, aorta_type, start_point, end_point, output_folder, verbose=False,  quiet=False, write_log_file=True ):
        """
        Args:
            image: SimpleITK image of the segmented aorta.
            aorta_type: string identifier for aorta type (e.g. 'type1', 'type2').
            start_point: tuple of (z, y, x) in physical coordinates.
            end_point: tuple of (z, y, x) in physical coordinates.
        """
        self.label_map = label_map
        self.aorta_type = aorta_type
        self.start_point = start_point
        self.end_point = end_point
        self.output_folder = output_folder
        self.verbose = verbose
        self.quiet = quiet
        self.write_log_file = write_log_file
        self.spacing = None
        self.skeleton_polydata = None
        self.pruned_skeleton = None
        self.dijkstra_map = None
        self.dijkstra_path = None
        self.spline_parameters = None
        self.cl_polydata = None

    def report_error(self, message: str, level: str = "error"):
        """
        Reports an error message based on verbosity and logging settings.
        """
        if not self.quiet:
            print(message)
        if self.write_log_file:
            write_message_to_log_file(base_dir=self.output_folder, message=message, level=level)

    def compute_centerline(self):
        """
        Computes the centerline
        """
        if self.label_map is None:
            report_error(f"No label map provided for centerline computation.")
            return False
        if not self.compute_skeleton_and_vtk_from_segmentation():
            return False
        if not self.iterative_pruning():
            return False
        if not self.dijkstra_on_skeleton():
            return False
        if not self.compute_spline_from_dijkstra_path():
            return False
        return True


    def save(self, path: pathlib.Path):
        """
        Save centerline
        """
        return True


    def load(self, path: pathlib.Path):
        """
        Load centerline
        """
        return True

    def spatial_resample_scan(self, image, desired_spacing, is_label_map=False):
        """
        Resample a scan to iso-tropic pixel spacing
        :param image: Original image with potentially anisotropic spacing
        :param desired_spacing: desired voxel spacing
        :param is_label_map: whether the image is a label map (affects interpolation method)
        :return: resampled image
        """
        current_n_vox = image.GetWidth()
        current_spacing = image.GetSpacing()
        new_n_vox_in_slice: int = int(current_n_vox * current_spacing[0] / desired_spacing)

        # voxel size in the direction of the patient
        depth_spacing = current_spacing[2]
        n_vox_depth = image.GetDepth()
        new_n_vox_depth = int(n_vox_depth * depth_spacing / desired_spacing)

        new_volume_size = [new_n_vox_in_slice, new_n_vox_in_slice, new_n_vox_depth]

        # Create new image with desired properties
        new_image = sitk.Image(new_volume_size, image.GetPixelIDValue())
        new_image.SetOrigin(image.GetOrigin())
        new_image.SetSpacing([desired_spacing, desired_spacing, desired_spacing])
        new_image.SetDirection(image.GetDirection())

        # Make translation with no offset, since sitk.Resample needs this arg.
        translation = sitk.TranslationTransform(3)
        translation.SetOffset((0, 0, 0))

        interpolator = sitk.sitkLinear
        if is_label_map:
            interpolator = sitk.sitkNearestNeighbor

        # Create final resampled image
        resampled_image = sitk.Resample(image, new_image, translation, interpolator)
        return resampled_image


    def extend_axial_with_copy(self, image, extension_cm=3.0):
        extension_mm = extension_cm * 10.0
        spacing = image.GetSpacing()

        extension_slices = int(round(extension_mm / spacing[2]))

        # Convert image to numpy array (z, y, x)
        image_array = sitk.GetArrayFromImage(image)

        # Copy first and last slices
        first_slice = image_array[0]
        last_slice = image_array[-1]

        # Get the logical or of the first and last five slices
        for i in range(1, 5):
            first_slice = np.logical_or(first_slice, image_array[i])
            last_slice = np.logical_or(last_slice, image_array[-(i + 1)])

        # Repeat slices to create extensions
        top_extension = np.repeat(first_slice[np.newaxis, :, :], extension_slices, axis=0)
        bottom_extension = np.repeat(last_slice[np.newaxis, :, :], extension_slices, axis=0)

        # Concatenate extensions
        extended_array = np.concatenate([top_extension, image_array, bottom_extension], axis=0)

        # Convert back to SimpleITK image
        extended_image = sitk.GetImageFromArray(extended_array)
        extended_image.SetSpacing(spacing)
        extended_image.SetDirection(image.GetDirection())

        # Get direction matrix
        direction = image.GetDirection()
        sign_z = direction[2]

        # Adjust origin to account for top extension
        new_origin = list(image.GetOrigin())
        if sign_z > 0:
            new_origin[2] -= extension_mm
        else:
            new_origin[2] += extension_mm
        extended_image.SetOrigin(new_origin)

        return extended_image


    def compute_skeleton_and_vtk_from_segmentation(self):
        # self.spacing = self.label_map.GetSpacing()
        # vox_space = min(self.spacing)
        # TODO: This is for testing only - need to make it more flexible
        vox_space = 1.0  # mm

        start_time = time.time()
        if self.verbose:
            print(f"Resampling to isotropic spacing: {vox_space} mm")
        image_res = self.spatial_resample_scan(self.label_map, vox_space, is_label_map=True)
        elapsed_time = time.time() - start_time
        if self.verbose:
            print(f"Resampling took {elapsed_time:.2f} seconds")

        if self.aorta_type in ["2", "5"]:
            if self.verbose:
                print(f"Extending axial slices by copying for type {self.aorta_type} aorta")
            start_time = time.time()
            image_res = self.extend_axial_with_copy(image_res, extension_cm=3.0)
            elapsed_time = time.time() - start_time
            if self.verbose:
                print(f"Axial extension took {elapsed_time:.2f} seconds")

        label_img_np = sitk.GetArrayFromImage(image_res)
        mask_np = label_img_np == 1
        start_time = time.time()
        if self.verbose:
            print(f"Skeletonization input shape: {mask_np.shape}, dtype: {mask_np.dtype}")
        skeleton = skeletonize(mask_np)
        elapsed_time = time.time() - start_time
        if self.verbose:
            print(f"Skeletonization took {elapsed_time:.2f} seconds")

        # print(f"Skeletonization output shape: {skeleton.shape}")
        skeleton_img = sitk.GetImageFromArray(skeleton.astype(np.uint8))
        skeleton_img.CopyInformation(image_res)

        # sitk.WriteImage(skeleton_img, out_name)

        # Get coordinates of skeleton voxels
        coords = np.argwhere(skeleton > 0)
        if self.verbose:
            print(f"Number of skeleton voxels: {len(coords)}")

        if self.verbose:
            print(f"Converting points to physical space")
        # Convert voxel coordinates to physical space
        index_to_point = {}
        for coord in coords:
            index = (int(coord[2]), int(coord[1]), int(coord[0]))  # (x, y, z)
            physical_point = image_res.TransformIndexToPhysicalPoint(index)
            index_to_point[tuple(coord)] = physical_point

        if self.verbose:
            print(f"Creating VTK polydata with points and lines")
        # Create VTK points and map voxel coordinates to point IDs
        points = vtk.vtkPoints()
        coord_to_id = {}
        for i, coord in enumerate(coords):
            pt = index_to_point[tuple(coord)]
            points.InsertNextPoint(pt)
            coord_to_id[tuple(coord)] = i

        if self.verbose:
            print(f"Connecting points to form lines")
        # Create VTK lines connecting each voxel to its 26-connected neighbors
        lines = vtk.vtkCellArray()
        neighbor_offsets = [(i, j, k) for i in [-1, 0, 1]
                                      for j in [-1, 0, 1]
                                      for k in [-1, 0, 1] if not (i == j == k == 0)]

        connections = set()
        coord_set = set(map(tuple, coords))
        for coord in coords:
            for offset in neighbor_offsets:
                neighbor = tuple(np.array(coord) + np.array(offset))
                if neighbor in coord_set:
                    pt_id1 = coord_to_id[tuple(coord)]
                    pt_id2 = coord_to_id[neighbor]
                    if (pt_id1, pt_id2) in connections:
                        # print(f"Skipping already connected points: {pt_id1}, {pt_id2}")
                        continue
                    line = vtk.vtkLine()
                    line.GetPointIds().SetId(0, pt_id1)
                    line.GetPointIds().SetId(1, pt_id2)
                    lines.InsertNextCell(line)
                    connections.add(tuple([pt_id1, pt_id2]))
                    connections.add(tuple([pt_id2, pt_id1]))

        if self.verbose:
            print(f"Total lines created: {lines.GetNumberOfCells()}")
        # Create VTK polydata
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(lines)

        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(polydata)
        cleaner.Update()
        clean_polydata = cleaner.GetOutput()
        if self.verbose:
            print(f"After cleaning: {clean_polydata.GetNumberOfPoints()} points, {clean_polydata.GetNumberOfLines()} lines")
        self.skeleton_polydata = clean_polydata
        #
        # # Write to VTK file
        # writer = vtk.vtkXMLPolyDataWriter()
        # writer.SetFileName(out_name_vtp)
        # writer.SetInputData(polydata)
        # writer.Write()
        return True


    def prune(self, min_endpoints=2):

        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(self.pruned_skeleton)
        cleaner.Update()
        polydata = cleaner.GetOutput()

        n_points = polydata.GetNumberOfPoints()
        branch_point_ids = []
        end_point_ids = []
        for i in range(n_points):
            cell_ids = vtk.vtkIdList()
            polydata.GetPointCells(i, cell_ids)
            n_cells = cell_ids.GetNumberOfIds()
            if n_cells > 2:
                branch_point_ids.append(i)
            elif n_cells == 1:
                # Check if endpoint (only one connected cell)
                end_point_ids.append(i)

        # Now compute the length of all branches. A branch goes from an end point down to a branch point
        # make new VTK polydata with vertices at end points with a scalar value of the branch length
        n_endpoints = len(end_point_ids)
        if self.verbose:
            print(f"Number of points in skeleton: {polydata.GetNumberOfPoints()}")
            print(f"Number of branch points: {len(branch_point_ids)}")
            print(f"Number of end points: {n_endpoints}")

        if len(end_point_ids) <= min_endpoints:
            if self.verbose:
                print(f"Stopping since only {n_endpoints} end points left and we should stop at {min_endpoints}")
            return True

        if self.verbose:
            print(f"Computing branch lengths")

        branches = []
        branch_lengths_arr = []
        for end_id in end_point_ids:
            cur_branch ={"end_id": end_id}
            visited = set()
            current_id = end_id
            length = 0.0
            cur_branch_points = []
            while True:
                if current_id not in branch_point_ids:
                    cur_branch_points.append(current_id)

                visited.add(current_id)
                cell_ids = vtk.vtkIdList()
                polydata.GetPointCells(current_id, cell_ids)
                if cell_ids.GetNumberOfIds() == 0:
                    break  # No more cells, should not happen
                # Get connected points
                connected_point_ids = set()
                for j in range(cell_ids.GetNumberOfIds()):
                    cell_id = cell_ids.GetId(j)
                    cell = polydata.GetCell(cell_id)
                    point_ids = cell.GetPointIds()
                    for k in range(point_ids.GetNumberOfIds()):
                        pid = point_ids.GetId(k)
                        if pid != current_id and pid not in visited:
                            connected_point_ids.add(pid)
                if len(connected_point_ids) == 0:
                    break  # Dead end, should not happen
                # Move to the next point (there should be only one unvisited connected point in a branch)
                next_id = connected_point_ids.pop()
                pt1 = np.array(polydata.GetPoint(current_id))
                pt2 = np.array(polydata.GetPoint(next_id))
                segment_length = np.linalg.norm(pt2 - pt1)
                length += segment_length
                current_id = next_id
                if current_id in branch_point_ids:
                    cur_branch["branch_id"] = current_id

                # Check if we reached a branch point
                cell_ids_next = vtk.vtkIdList()
                polydata.GetPointCells(current_id, cell_ids_next)
                if cell_ids_next.GetNumberOfIds() > 2:
                    break  # Reached a branch point

            cur_branch["points"] = cur_branch_points
            cur_branch["length"] = length
            branches.append(cur_branch)

            branch_lengths_arr.append(length)

        # Sort list of dictionaries by the length key
        sorted_branches = sorted(branches, key=lambda b: b["length"], reverse=False)
        visited_branch_ids = set()
        # delete points in branches starting with shortest first. Do not delete a branch that has the same branch point id
        # as a previusly deleted branch
        removed_branches = 0
        kept_branches = 0
        for branch in sorted_branches:
            # print(f"Processing branch {branch['branch_id']} with length {branch['length']}")
            if branch["branch_id"] not in visited_branch_ids:
                removed_branches += 1
                for pid in branch["points"]:
                    cell_ids = vtk.vtkIdList()
                    polydata.GetPointCells(pid, cell_ids)
                    for i in range(cell_ids.GetNumberOfIds()):
                        polydata.DeleteCell(cell_ids.GetId(i))
                    polydata.RemoveDeletedCells()
                    # polydata.DeletePoint(pid)
                visited_branch_ids.add(branch["branch_id"])
                if (n_endpoints - removed_branches) <= min_endpoints:
                    print(f"Removed {removed_branches} branches out of {n_endpoints} stopping now")
                    kept_branches += min_endpoints
                    break
            else:
                kept_branches += 1

        if self.verbose:
            print(f"Removed branches: {removed_branches} and kept branches: {kept_branches}")

        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(polydata)
        cleaner.Update()
        self.pruned_skeleton.DeepCopy(cleaner.GetOutput())

        if removed_branches == 0:
            return True

        return False


    def iterative_pruning(self):
        # in_name = f"C:/data/SkeletonTest/{scan_id}/skeleton_isotropic.vtp"
        min_endpoints = 3
        if self.aorta_type == "2":
            min_endpoints = 3  # Top of the aorta + two iliac arteries
        if self.aorta_type == "5":
            min_endpoints = 2  # Just the aorta in the ascending and descending parts

        self.pruned_skeleton = vtk.vtkPolyData()
        self.pruned_skeleton.DeepCopy(self.skeleton_polydata)

        it = 0
        stop = False
        while not stop:
            stop = self.prune(min_endpoints)
            it += 1
            if it > 5:
                stop = True
        return True


    def dijkstra_on_skeleton(self):
        self.dijkstra_map = vtk.vtkPolyData()
        self.dijkstra_map.DeepCopy(self.pruned_skeleton)

        # Find the closest point in the skeleton to the start landmark using a vtkPointLocator
        point_locator = vtk.vtkPointLocator()
        point_locator.SetDataSet(self.dijkstra_map)
        point_locator.BuildLocator()
        start_point_id = point_locator.FindClosestPoint(self.start_point)
        #print(f"Start point ID in skeleton: {start_point_id}")
        end_point_id = point_locator.FindClosestPoint(self.end_point)
        # print(f"End point ID in skeleton: {end_point_id}")
        dijkstra = vtk.vtkDijkstraGraphGeodesicPath()
        dijkstra.SetInputData(self.dijkstra_map)
        dijkstra.SetStartVertex(start_point_id)
        dijkstra.Update()
        weights = vtk.vtkDoubleArray()
        dijkstra.GetCumulativeWeights(weights)

        self.dijkstra_map.GetPointData().SetScalars(weights)

        dijkstra.SetEndVertex(end_point_id)
        dijkstra.Update()

        self.dijkstra_path = vtk.vtkPolyData()
        self.dijkstra_path.DeepCopy(dijkstra.GetOutput())
        return True


    def compute_spline_from_dijkstra_path(self, spline_smoothing_factor=20, sample_spacing=0.25):
        cl_in = self.dijkstra_path
        # spline_smoothing_factor = 50

        sum_dist = 0
        n_points = cl_in.GetNumberOfPoints()

        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_splrep.html#scipy.interpolate.make_splrep
        spline_smoothing_factor = n_points / 3.0

        if self.verbose:
            print(f"Computing spline from Dijkstra path with {n_points} "
                  f"points and smoothing factor {spline_smoothing_factor:.1f}")

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

        self.spline_parameters = {
            "min_x": min_x,
            "max_x": max_x,
            "spl_1": spl_1,
            "spl_2": spl_2,
            "spl_3": spl_3
        }
        return True


    def get_centerline_as_polydata(self, sample_spacing=0.25):
        min_x = self.spline_parameters["min_x"]
        max_x = self.spline_parameters["max_x"]
        spl_1 = self.spline_parameters["spl_1"]
        spl_2 = self.spline_parameters["spl_2"]
        spl_3 = self.spline_parameters["spl_3"]

        der_1 = spl_1.derivative()
        der_2 = spl_2.derivative()
        der_3 = spl_3.derivative()

        samp_space = sample_spacing
        spline_n_points = int(max_x / samp_space)
        if self.verbose:
            print(f"Computing sampled spline path with length {max_x:.1f} and sample spacing {samp_space} resulting in {spline_n_points} samples for smoothing")

        # Compute a polydata object with the spline points
        xs = np.linspace(min_x, max_x, spline_n_points)

        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        scalars = vtk.vtkDoubleArray()
        scalars.SetNumberOfComponents(1)

        tangents = vtk.vtkDoubleArray()
        tangents.SetNumberOfComponents(3)
        tangents.SetName("Tangents")  # Optional but useful for identification

        current_idx = 0
        sum_dist = 0
        sp = [spl_1(xs[current_idx]), spl_2(xs[current_idx]), spl_3(xs[current_idx])]
        t = [der_1(xs[current_idx]), der_2(xs[current_idx]), der_3(xs[current_idx])]
        vtk.vtkMath.Normalize(t)

        pid = points.InsertNextPoint(sp)
        scalars.InsertNextValue(sum_dist)
        tangents.InsertNextTuple(t)
        current_idx += 1

        while current_idx < spline_n_points:
            p_1 = [spl_1(xs[current_idx]), spl_2(xs[current_idx]), spl_3(xs[current_idx])]
            t_1 = [der_1(xs[current_idx]), der_2(xs[current_idx]), der_3(xs[current_idx])]
            vtk.vtkMath.Normalize(t_1)

            sum_dist += np.linalg.norm(np.array(p_1) - np.array(sp))
            lines.InsertNextCell(2)
            pid_2 = points.InsertNextPoint(p_1)
            scalars.InsertNextValue(sum_dist)
            tangents.InsertNextTuple(t_1)
            lines.InsertCellPoint(pid)
            lines.InsertCellPoint(pid_2)
            pid = pid_2
            sp = p_1
            current_idx += 1

        self.cl_polydata = vtk.vtkPolyData()
        self.cl_polydata.SetPoints(points)
        del points
        self.cl_polydata.SetLines(lines)
        del lines
        self.cl_polydata.GetPointData().SetScalars(scalars)
        del scalars
        self.cl_polydata.GetPointData().AddArray(tangents)
        del tangents

        return self.cl_polydata

