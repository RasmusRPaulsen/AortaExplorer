import os.path
import vtk
import numpy as np
import csv
from aortaexplorer.general_utils import read_json_file
import aortaexplorer.surface_utils as surfutils
from aortaexplorer.surface_utils import read_nifti_itk_to_vtk


class RenderTotalSegmentatorData:
    """
    Can render data from TotalsSegmentator.
    Can also dump rendering to an image file.

    This is a super class that should be inherited.
    """

    def __init__(self, win_size=(1600, 800), render_to_file=True):
        self.ren_win = vtk.vtkRenderWindow()
        self.ren_win.SetOffScreenRendering(render_to_file)
        self.win_size = win_size
        self.ren_win.SetSize(win_size)
        self.ren_win.SetWindowName("Segmentation view")

        self.vtk_image = None
        self.ren_volume = None
        self.ren_text = None
        self.ren_patient_text = None
        self.ren_warning_text = None

        self.viewport_volume = [0.60, 0.0, 1.0, 1.0]
        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.ren_win)
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.iren.SetInteractorStyle(style)

        self.actors = []
        self.segments = []
        self.landmarks = []
        self.centerlines = []

        # The text message that will be showed in the text renderer
        self.message_text = ""
        self.patient_text = ""
        self.warning_text = ""

    def render_interactive(self):
        """
        Creates an interactive renderwindow with the results
        """
        pos = (5, 5)
        font_size = 12
        self.add_text_to_render(
            self.ren_text,
            self.message_text,
            color=(1.0, 1.0, 1.0),
            position=pos,
            font_size=font_size,
        )
        self.add_text_to_render(
            self.ren_patient_text,
            self.patient_text,
            color=(0.0, 1.0, 0.0),
            position=pos,
            font_size=font_size,
        )
        self.add_text_to_render(
            self.ren_warning_text,
            self.warning_text,
            color=(1.0, 1.0, 0.0),
            position=pos,
            font_size=font_size,
        )
        self.iren.Start()

    def render_to_file(self, file_name):
        """
        Write the renderwindow to an image file
        :param file_name: Image file name (.png)
        """
        # viewport_size = self.ren_text.GetSize()
        # print(f"Viewport size {size}")
        # pos = (5, viewport_size[1] - 50)
        pos = (5, 5)
        self.add_text_to_render(self.ren_text, self.message_text, color=(1.0, 1.0, 1.0), position=pos)
        self.add_text_to_render(
            self.ren_patient_text,
            self.patient_text,
            color=(0.0, 1.0, 0.0),
            position=pos,
        )
        self.add_text_to_render(
            self.ren_warning_text,
            self.warning_text,
            color=(1.0, 1.0, 0.0),
            position=pos,
        )

        self.ren_win.SetOffScreenRendering(1)
        # print(f"Writing visualization to {file_name}")
        self.ren_win.SetSize(self.win_size)
        self.ren_win.Render()
        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(self.ren_win)
        writer_png = vtk.vtkPNGWriter()
        writer_png.SetInputConnection(w2if.GetOutputPort())
        writer_png.SetFileName(file_name)
        writer_png.Write()

    def set_sitk_image_file(self, input_file, img_mask=None):
        """
        Add a simple ITK image to the renderer using a volume renderer.
        If a mask is provided, the volume data is first masked. This can for example remove scanner beds etc.
        """
        self.vtk_image = read_nifti_itk_to_vtk(input_file, img_mask, flip_for_volume_rendering=True)
        if self.vtk_image is None:
            return

        vtk_dim = self.vtk_image.GetDimensions()
        vtk_spc = self.vtk_image.GetSpacing()

        img_txt = f"Spacing: ({vtk_spc[0]:.2f}, {vtk_spc[1]:.2f}, {vtk_spc[2]:.2f}) mm\nDimensions: ({vtk_dim[0]}, {vtk_dim[1]}, {vtk_dim[2]}) vox\nSize: ({vtk_spc[0] * vtk_dim[0] / 10.0:.1f}, {vtk_spc[1] * vtk_dim[1] / 10.0:.1f}, {vtk_spc[2] * vtk_dim[2] / 10.0:.1f}) cm\n"
        self.message_text += img_txt

        # Get direction to set camera (not needed when we do the brutal flip of image data in the load routine)
        # dir_mat = self.vtk_image.GetDirectionMatrix().GetData()
        # print(f"Dir mat: {dir_mat}")
        # dir_val = dir_mat[4]
        dir_val = 1

        # Reset direction matrix, since volume render do not cope good with it
        direction = [1, 0, 0.0, 0, 1, 0.0, 0.0, 0.0, 1.0]
        self.vtk_image.SetDirectionMatrix(direction)

        volume_mapper = vtk.vtkSmartVolumeMapper()
        volume_mapper.SetInputData(self.vtk_image)
        volume = vtk.vtkVolume()
        volume.SetMapper(volume_mapper)

        self.ren_volume = vtk.vtkRenderer()
        self.ren_volume.SetViewport(self.viewport_volume)

        volume_color = vtk.vtkColorTransferFunction()
        volume_color.AddRGBPoint(-2048, 0.0, 0.0, 0.0)
        volume_color.AddRGBPoint(145, 0.0, 0.0, 0.0)
        volume_color.AddRGBPoint(145, 0.62, 0.0, 0.015)
        volume_color.AddRGBPoint(192, 0.91, 0.45, 0.0)
        volume_color.AddRGBPoint(217, 0.97, 0.81, 0.61)
        volume_color.AddRGBPoint(384, 0.91, 0.91, 1.0)
        volume_color.AddRGBPoint(478, 0.91, 0.91, 1.0)
        volume_color.AddRGBPoint(3660, 1, 1, 1.0)

        volume_scalar_opacity = vtk.vtkPiecewiseFunction()
        volume_scalar_opacity.AddPoint(-2048, 0.00)
        volume_scalar_opacity.AddPoint(143, 0.00)
        volume_scalar_opacity.AddPoint(145, 0.12)
        volume_scalar_opacity.AddPoint(192, 0.56)
        volume_scalar_opacity.AddPoint(217, 0.78)
        volume_scalar_opacity.AddPoint(385, 0.83)
        volume_scalar_opacity.AddPoint(3660, 0.83)

        volume_gradient_opacity = vtk.vtkPiecewiseFunction()
        volume_gradient_opacity.AddPoint(0, 1.0)
        volume_gradient_opacity.AddPoint(255, 1.0)

        volume_property = vtk.vtkVolumeProperty()
        volume_property.SetColor(volume_color)
        volume_property.SetScalarOpacity(volume_scalar_opacity)
        volume_property.SetGradientOpacity(volume_gradient_opacity)
        volume_property.SetInterpolationTypeToLinear()
        volume_property.ShadeOn()
        volume_property.SetAmbient(0.2)
        volume_property.SetDiffuse(1.0)
        volume_property.SetSpecular(0.0)

        volume.SetProperty(volume_property)
        self.ren_volume.AddViewProp(volume)

        self.ren_win.AddRenderer(self.ren_volume)

        c = volume.GetCenter()
        view_offsets = [500, -1000, 0]
        # Hack to handle direction matrices
        view_offsets[0] *= dir_val
        view_offsets[1] *= dir_val
        self.ren_volume.GetActiveCamera().SetParallelProjection(1)
        self.ren_volume.GetActiveCamera().SetViewUp(0, 0, 1)
        self.ren_volume.GetActiveCamera().SetPosition(c[0] + view_offsets[0], c[1] + view_offsets[1], c[2] + view_offsets[2])
        self.ren_volume.GetActiveCamera().SetFocalPoint(c[0], c[1], c[2])
        # ren_volume.ResetCamera()
        self.ren_volume.ResetCameraScreenSpace()
        self.ren_volume.GetActiveCamera().Zoom(1.2)

    @staticmethod
    def add_text_to_render(ren, message, color=(1, 1, 1), position=(5, 5), font_size=10):
        if ren is None or message == "":
            return
        txt = vtk.vtkTextActor()
        txt.SetInput(message)
        # txt.SetTextScaleModeToNone()
        # txt.SetTextScaleModeToViewport()
        txt.SetTextScaleModeToProp()
        txtprop = txt.GetTextProperty()
        # txtprop.SetFontFamilyToArial()
        # txtprop.SetFontSize(font_size)
        txtprop.SetColor(color)
        # txt.SetDisplayPosition(position[0], position[1])

        # txtprop.SetJustificationToLeft()
        txtprop.SetVerticalJustificationToTop()
        txt.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        # txt.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
        # txt.GetPositionCoordinate().SetValue(0.005, 0.99)
        txt.SetPosition(0.05, 0.05)
        txt.SetPosition2(0.95, 0.95)
        # txt.GetPositionCoordinate().SetValue(0.0, 0.0)
        # txt.GetPositionCoordinate2().SetValue(1.0, 1.0)
        # txt.SetTextScaleModeToViewport()
        ren.AddActor(txt)

    @staticmethod
    def generate_actor_from_surface(surface, color=np.array([1, 0, 0]), opacity=1.0, smooth="heavy"):
        n_points = surface.GetNumberOfPoints()
        if n_points < 2:
            print("Not enough points in surface")
            return None

        # https://kitware.github.io/vtk-examples/site/Python/Modelling/SmoothDiscreteMarchingCubes/
        if smooth == "light":
            feature_angle = 120.0
            pass_band = 0.001
            smoothing_iterations = 10
        elif smooth == "heavy":
            feature_angle = 120.0
            pass_band = 0.001
            smoothing_iterations = 50
        else:
            feature_angle = 120.0
            pass_band = 0.001
            smoothing_iterations = 20

        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputData(surface)
        smoother.SetNumberOfIterations(smoothing_iterations)
        smoother.FeatureEdgeSmoothingOff()
        smoother.BoundarySmoothingOff()
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.SetFeatureAngle(feature_angle)
        smoother.SetPassBand(pass_band)
        smoother.Update()

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(smoother.GetOutputPort())
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOn()
        normals.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(normals.GetOutputPort())
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetColor(float(color[0]), float(color[1]), float(color[2]))

        return actor

    def generate_actors_from_segment_file_name(self, segm_name, surface_name, color_name, opacity, smooth="heavy"):
        surface = surfutils.convert_label_map_to_surface(segm_name)
        if surface is not None:
            rgba = [0.0, 0.0, 0.0, 0.0]
            vtk.vtkNamedColors().GetColor(color_name, rgba)
            actor = self.generate_actor_from_surface(surface, rgba, opacity, smooth=smooth)
            if actor is not None:
                self.actors.append(actor)

    def generate_actors_from_centerlines(self, cl_folder):
        for cl in self.centerlines:
            cl_file = f"{cl_folder}{cl['file']}"
            if not os.path.exists(cl_file):
                print(f"No {cl_file}")
                cl_vtk = None
            else:
                reader = vtk.vtkXMLPolyDataReader()
                reader.SetFileName(cl_file)
                reader.Update()
                cl_vtk = reader.GetOutput()
                n_points = cl_vtk.GetNumberOfPoints()
                if n_points < 2:
                    print(f"No points in {cl_file}")
                    cl_vtk = None

            if cl_vtk is not None:
                size = cl["size"]
                color = cl["color"]
                opacity = cl["opacity"]
                rgba = [0.0, 0.0, 0.0, 0.0]
                vtk.vtkNamedColors().GetColor(color, rgba)

                vtk_tube_filter = vtk.vtkTubeFilter()
                vtk_tube_filter.SetInputData(cl_vtk)
                vtk_tube_filter.SetNumberOfSides(16)
                vtk_tube_filter.SetRadius(size)
                vtk_tube_filter.Update()

                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(vtk_tube_filter.GetOutputPort())
                mapper.ScalarVisibilityOff()

                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetOpacity(opacity)
                actor.GetProperty().SetColor(rgba[0], rgba[1], rgba[2])

                self.actors.append(actor)


class RenderAortaData(RenderTotalSegmentatorData):
    def __init__(self, win_size, render_to_file, stats_folder, segm_folder, cl_folder):
        super().__init__(win_size, render_to_file)
        # print(f"Initialising aorta renderer")

        n_aorta_parts = 1
        parts_stats = read_json_file(f"{stats_folder}aorta_parts.json")
        if parts_stats:
            n_aorta_parts = parts_stats["aorta_parts"]

        if n_aorta_parts == 1:
            segm_name_aorta = f"{segm_folder}aorta_lumen.nii.gz"
            # surf_name_aorta = f"{surf_output_dir}computed_aorta_lumen_surface.vtk"
            self.generate_actors_from_segment_file_name(segm_name_aorta, None, "Crimson", 0.8)
        elif n_aorta_parts == 2:
            segm_name_aorta = f"{segm_folder}aorta_lumen_annulus.nii.gz"
            # surf_name_aorta = f"{surf_output_dir}aorta_lumen_annulus.vtk"
            self.generate_actors_from_segment_file_name(segm_name_aorta, None, "Crimson", 0.8)
            segm_name_aorta = f"{segm_folder}aorta_lumen_descending.nii.gz"
            # surf_name_aorta = f"{surf_output_dir}aorta_lumen_descending.vtk"
            self.generate_actors_from_segment_file_name(segm_name_aorta, None, "Crimson", 0.8)

        segm_name_calc = f"{segm_folder}aorta_calcification_raw.nii.gz"
        # surf_name_calc = f"{surf_output_dir}aorta_calcification.vtk"
        self.generate_actors_from_segment_file_name(segm_name_calc, None, "Ivory", 1.0, smooth="light")

        segm_name = f"{segm_folder}iliac_artery_left_top.nii.gz"
        # surf_name = f"{surf_output_dir}computed_iliac_artery_left_top.vtk"
        self.generate_actors_from_segment_file_name(segm_name, None, "DarkSalmon", 1.0, smooth="heavy")

        segm_name = f"{segm_folder}iliac_artery_right_top.nii.gz"
        # surf_name = f"{surf_output_dir}computed_iliac_artery_right_top.vtk"
        self.generate_actors_from_segment_file_name(segm_name, None, "PaleVioletRed", 1.0, smooth="heavy")

        # TODO: THis is hacky and should be updated
        if n_aorta_parts == 1:
            aneurysm_sac_stats_file = f"{stats_folder}aorta_aneurysm_sac_stats.json"
            aneurysm_sac_stats = read_json_file(aneurysm_sac_stats_file)
            if aneurysm_sac_stats:
                aneurysm_sac_ratio = aneurysm_sac_stats["aorta_ratio"]
                q95_dists = aneurysm_sac_stats["q95_distances"]
                if aneurysm_sac_ratio > 1.2 and q95_dists > 5:
                    self.segments.append(
                        {
                            "task": "total",
                            "id": 52,
                            "segment": "aorta",
                            "max_components": 2,
                            "min_component_size": 5000,
                            "color": "OldLace",
                            "opacity": 0.6,
                            "visible": True,
                        }
                    )

        # self.generate_actors_from_combined_segments_with_tasks()

        if n_aorta_parts == 1:
            self.centerlines.append(
                {
                    "name": "aorta_center_line",
                    "file": "aorta_centerline.vtp",
                    "size": 1,
                    "color": "white",
                    "opacity": 1.0,
                }
            )
        elif n_aorta_parts == 2:
            self.centerlines.append(
                {
                    "name": "aorta_center_line_annulus",
                    "file": "aorta_centerline_annulus.vtp",
                    "size": 1,
                    "color": "white",
                    "opacity": 1.0,
                }
            )
            self.centerlines.append(
                {
                    "name": "aorta_center_line_descending",
                    "file": "aorta_centerline_descending.vtp",
                    "size": 1,
                    "color": "white",
                    "opacity": 1.0,
                }
            )

        self.generate_actors_from_centerlines(cl_folder)

        # Split the screen into viewports
        # xmin, ymin, xmax, ymax (range 0-1)
        self.viewport_text = [0.0, 0.1, 0.20, 1.0]
        self.viewport_3d_1 = [0.20, 0.2, 0.35, 1.0]
        self.viewport_3d_2 = [0.35, 0.2, 0.50, 1.0]
        self.viewport_straight_1 = [0.50, 0.2, 0.60, 1.0]
        self.viewport_straight_2 = [0.60, 0.2, 0.70, 1.0]
        self.viewport_slice = [0.70, 0.2, 0.8, 1.0]
        self.viewport_volume = [0.80, 0.2, 1.0, 1.0]
        self.viewport_plot = [0.2, 0.0, 1.0, 0.2]

        self.ren_3d_1 = None
        self.ren_3d_2 = None
        self.ren_slice = None
        # Straightened volumes
        self.ren_straight_1 = None
        self.ren_straight_2 = None

        self.setup_renderers()

    def setup_renderers(self):
        self.ren_3d_1 = vtk.vtkRenderer()
        # xmin, ymin, xmax, ymax (range 0-1)
        # self.ren_3d_1.SetViewport(0.0, 0.0, 0.20, 1.0)
        self.ren_3d_1.SetViewport(self.viewport_3d_1)
        self.ren_win.AddRenderer(self.ren_3d_1)
        self.ren_3d_2 = vtk.vtkRenderer()
        self.ren_3d_2.SetViewport(self.viewport_3d_2)
        # self.ren_3d_2.SetViewport(0.20, 0.0, 0.4, 1.0)
        self.ren_win.AddRenderer(self.ren_3d_2)

        for actor in self.actors:
            self.ren_3d_1.AddActor(actor)
            self.ren_3d_2.AddActor(actor)

        actor_bounds = self.ren_3d_1.ComputeVisiblePropBounds()
        c = [
            (actor_bounds[0] + actor_bounds[1]) / 2,
            (actor_bounds[2] + actor_bounds[3]) / 2,
            (actor_bounds[4] + actor_bounds[5]) / 2,
        ]

        self.ren_3d_1.GetActiveCamera().SetParallelProjection(1)
        self.ren_3d_1.GetActiveCamera().SetViewUp(0, 0, 1)
        self.ren_3d_1.GetActiveCamera().SetPosition(c[0], c[1] - 400, c[2])
        self.ren_3d_1.GetActiveCamera().SetFocalPoint(c[0], c[1], c[2])
        self.ren_3d_1.ResetCameraScreenSpace()

        self.ren_3d_2.GetActiveCamera().SetParallelProjection(1)
        self.ren_3d_2.GetActiveCamera().SetFocalPoint(c[0], c[1], c[2])
        self.ren_3d_2.GetActiveCamera().SetViewUp(0, 0, 1)
        self.ren_3d_2.GetActiveCamera().SetPosition(c[0] + 400, c[1], c[2])
        self.ren_3d_2.ResetCameraScreenSpace()

        # Renderer for text
        self.ren_text = vtk.vtkRenderer()
        self.ren_win.AddRenderer(self.ren_text)
        self.ren_text.SetViewport(self.viewport_text)

        # Renderer for plot
        self.ren_plot = vtk.vtkRenderer()
        self.ren_win.AddRenderer(self.ren_plot)
        self.ren_plot.SetViewport(self.viewport_plot)

        # Renderer for slice
        self.ren_slice = vtk.vtkRenderer()
        self.ren_win.AddRenderer(self.ren_slice)
        self.ren_slice.SetViewport(self.viewport_slice)

        # Renderer for straight volume
        self.ren_straight_1 = vtk.vtkRenderer()
        self.ren_win.AddRenderer(self.ren_straight_1)
        self.ren_straight_1.SetViewport(self.viewport_straight_1)

        self.ren_straight_2 = vtk.vtkRenderer()
        self.ren_win.AddRenderer(self.ren_straight_2)
        self.ren_straight_2.SetViewport(self.viewport_straight_2)

    def set_aorta_statistics(self, stats_folder):
        stats_file = f"{stats_folder}aorta_statistics.json"
        scan_type_file = f"{stats_folder}aorta_scan_type.json"

        aorta_stats = read_json_file(stats_file)
        if aorta_stats is None:
            print(f"Found no stats file {stats_file}")
            return

        scan_type_stats = read_json_file(scan_type_file)
        if scan_type_stats is None:
            print(f"Found no scan type file {scan_type_file}")
            return

        scan_type = scan_type_stats["scan_type"]

        # # TODO: add more types: https://github.com/RasmusRPaulsen/DTUHeartCenter/tree/main/Aorta/figs
        # if scan_type == "1":
        #     cl_length = aorta_stats.get("annulus_aortic_length",0)
        # elif scan_type == "2":
        #     cl_length = aorta_stats.get("descending_aortic_length", 0)
        #     # cl_length = aorta_stats["descending_aortic_length"]
        # else:
        #     cl_length = 0

        aorta_txt = f"\nAorta HU avg: {aorta_stats['avg_hu']:.0f} ({aorta_stats['cl_mean']:.0f})\nstd.dev: {aorta_stats['std_hu']:.0f} ({aorta_stats['cl_std']:.0f})\nmedian: {aorta_stats['med_hu']:.0f} ({aorta_stats['cl_med']:.0f})\n99%: {aorta_stats['q99_hu']:.0f} ({aorta_stats['cl_q99']:.0f})\n1%: {aorta_stats['q01_hu']:.0f} ({aorta_stats['cl_q01']:.0f})\nAorta vol: {aorta_stats['tot_vol'] / 1000.0:.0f} cm3\nscan type: {scan_type}\nAorta Surface area: {aorta_stats['surface_area'] / 100.0:.1f} cm2\n"
        # f'Centerline length: {cl_length / 10.0:.1f} cm\n' \

        self.message_text += aorta_txt

    def set_plot_data(self, stats_folder, cl_folder):
        n_aorta_parts = 1
        parts_stats = read_json_file(f"{stats_folder}/aorta_parts.json")
        if parts_stats:
            n_aorta_parts = parts_stats["aorta_parts"]

        if n_aorta_parts == 2:
            data_file = f"{cl_folder}straight_labelmap_sampling_annulus.csv"
            max_in_files = [
                "aortic_arch_segment_max_slice_info",
                "ascending_segment_max_slice_info",
                "distensability_segment_avg_slice_info",
                "sinotubular_junction_segment_max_slice_info",
                "sinus_of_valsalva_segment_max_slice_info",
                "lvot_segment_max_slice_info",
            ]
            max_rgb = [
                [255, 0, 0],
                [0, 255, 255],
                [255, 128, 128],
                [128, 0, 128],
                [0, 128, 255],
                [200, 255, 100],
            ]
            max_rgb = np.divide(max_rgb, 255.0)
        elif n_aorta_parts == 1:
            data_file = f"{cl_folder}straight_labelmap_sampling.csv"
            max_in_files = [
                "infrarenal_segment_max_slice_info",
                "abdominal_segment_max_slice_info",
                "aortic_arch_segment_max_slice_info",
                "ascending_segment_max_slice_info",
                "sinotubular_junction_segment_max_slice_info",
                "sinus_of_valsalva_segment_max_slice_info",
                "descending_segment_max_slice_info",
                "lvot_segment_max_slice_info",
            ]
            max_rgb = [
                [255, 255, 0],
                [255, 128, 0],
                [255, 0, 0],
                [0, 255, 255],
                [128, 0, 128],
                [0, 128, 255],
                [0, 255, 0],
                [200, 255, 100],
            ]
            max_rgb = np.divide(max_rgb, 255.0)
        else:
            return

        if not os.path.exists(data_file):
            print(f"Could not open {data_file}")
            return

        PlotData = vtk.vtkPolyData()
        PlotPoints = vtk.vtkPoints()
        PlotScalars = vtk.vtkFloatArray()
        PlotData.SetPoints(PlotPoints)
        PlotData.GetPointData().SetScalars(PlotScalars)

        try:
            file = open(data_file, "r")
        except IOError:
            print(f"Cannot read {data_file}")
            return

        measurements = csv.reader(file, delimiter=",")

        for elem in measurements:
            dist = float(elem[0])
            area = float(elem[1])
            PlotPoints.InsertNextPoint(dist, 0, 0)
            PlotScalars.InsertNextValue(area)

        if PlotPoints.GetNumberOfPoints() < 1:
            print(f"No valid data in {data_file}")
            return

        xyplot = vtk.vtkXYPlotActor()
        xyplot.AddDataSetInput(PlotData)
        xyplot.SetPlotLabel(0, "Aorta")

        plot_idx = 1
        for idx, pfile in enumerate(max_in_files):
            file_name = f"{cl_folder}{pfile}.json"
            cut_plane_stats = read_json_file(file_name)
            if cut_plane_stats:
                area = cut_plane_stats["area"]
                cl_dist = cut_plane_stats["cl_dist"]
                rgb = max_rgb[idx]

                PlotData_2 = vtk.vtkPolyData()
                PlotPoints_2 = vtk.vtkPoints()
                PlotScalars_2 = vtk.vtkFloatArray()
                PlotData_2.SetPoints(PlotPoints_2)
                PlotData_2.GetPointData().SetScalars(PlotScalars_2)

                PlotPoints_2.InsertNextPoint(cl_dist, 0, 0)
                PlotScalars_2.InsertNextValue(area)

                xyplot.AddDataSetInput(PlotData_2)
                xyplot.SetPlotGlyphType(plot_idx, 8)
                xyplot.SetPlotColor(plot_idx, rgb)
                plot_idx += 1

        xyplot.GetPositionCoordinate().SetValue(0.05, 0.05, 0.0)
        xyplot.GetPosition2Coordinate().SetValue(0.95, 0.95, 0.0)  # relative to Position
        xyplot.SetXValuesToValue()
        xyplot.SetYLabelFormat("%-#6.0f")
        xyplot.SetNumberOfXLabels(6)
        xyplot.SetNumberOfYLabels(6)
        xyplot.SetXTitle("Distance")
        xyplot.SetYTitle("A")
        xyplot.GetProperty().SetLineWidth(1)
        self.ren_plot.AddActor2D(xyplot)

    @staticmethod
    def create_cylinder_actor(pos, normal, radius=30, height=1, rgba=None, opacity=1.0):
        """ """
        if rgba is None:
            rgba = [0.0, 1.0, 0.0, 0.0]

        cylinder = vtk.vtkCylinderSource()
        cylinder.SetResolution(100)
        cylinder.CappingOn()
        cylinder.SetRadius(radius)
        cylinder.SetHeight(height)

        y_axis = [0.0, 1.0, 0.0]
        axis = np.cross(y_axis, normal)
        angle = np.arccos(np.dot(y_axis, normal))
        transform = vtk.vtkTransform()
        transform.Translate(pos)
        transform.RotateWXYZ(np.degrees(angle), *axis)
        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetTransform(transform)
        transform_filter.SetInputConnection(cylinder.GetOutputPort())
        transform_filter.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(transform_filter.GetOutputPort())
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetColor(rgba[0], rgba[1], rgba[2])
        return actor

    def set_all_max_cut_data(self, cl_folder):
        """
        Generate a disk where max cut is found
        """
        cl_dir = cl_folder
        max_in_files = [
            "infrarenal_segment_max_slice_info",
            "abdominal_segment_max_slice_info",
            "aortic_arch_segment_max_slice_info",
            "ascending_segment_max_slice_info",
            "distensability_segment_avg_slice_info",
            "sinotubular_junction_segment_max_slice_info",
            "sinus_of_valsalva_segment_max_slice_info",
            "descending_segment_max_slice_info",
            "lvot_segment_max_slice_info",
        ]
        max_rgb = [
            [255, 255, 0],
            [255, 128, 0],
            [255, 0, 0],
            [0, 255, 255],
            [255, 128, 128],
            [128, 0, 128],
            [0, 128, 255],
            [0, 255, 0],
            [200, 255, 100],
        ]
        max_rgb = np.divide(max_rgb, 255.0)

        for idx, pfile in enumerate(max_in_files):
            file_name = f"{cl_dir}{pfile}.json"
            cut_plane_stats = read_json_file(file_name)
            if cut_plane_stats:
                pos = cut_plane_stats["origin"]
                normal = cut_plane_stats["normal"]
                max_diam = cut_plane_stats["max_diameter"]
                radius = max_diam / 2 * 1.10
                rgb = max_rgb[idx]

                actor_cut = self.create_cylinder_actor(pos, normal, radius=radius, height=1.5, rgba=rgb, opacity=1.0)

                self.ren_3d_1.AddActor(actor_cut)
                self.ren_3d_2.AddActor(actor_cut)

    def set_precomputed_slice(self, cl_folder):
        """
        Here the slices are precomputed as png files
        """
        plane_file = f"{cl_folder}combined_cuts.png"

        png_reader = vtk.vtkPNGReader()
        if not png_reader.CanReadFile(plane_file):
            print(f"Can not read {plane_file}")
            return
        png_reader.SetFileName(plane_file)
        png_reader.Update()

        # https://www.weiy.city/2021/12/scale-image-displayed-by-vtkactor2d-object/
        # extent = png_reader.GetOutput().GetExtent()
        # origin = png_reader.GetOutput().GetOrigin()
        # spacing = png_reader.GetOutput().GetSpacing()
        # size = [extent[1], extent[3]]
        # new_size = [5 * size[0], 5 * size[1], 1]
        # interpolator = vtk.vtkImageSincInterpolator()
        # resizer = vtk.vtkImageResize()
        # resizer.SetInputConnection(png_reader.GetOutputPort())
        # resizer.SetInterpolator(interpolator)
        # resizer.SetOutputDimensions(new_size)
        # resizer.InterpolateOn()
        # resizer.Update()

        image_viewer = vtk.vtkImageViewer2()
        # image_viewer.SetInputConnection(resizer.GetOutputPort())
        image_viewer.SetInputConnection(png_reader.GetOutputPort())
        image_viewer.SetRenderWindow(self.ren_win)
        image_viewer.SetRenderer(self.ren_slice)

        image_viewer.Render()
        self.ren_slice.GetActiveCamera().ParallelProjectionOn()
        self.ren_slice.ResetCameraScreenSpace()

    def set_cut_statistics(self, cl_folder):
        cl_dir = cl_folder
        cut_stats = [
            {"name": "LVOT", "file": "lvot_segment_max_slice_info.json"},
            {
                "name": "Sinus of Valsalve",
                "file": "sinus_of_valsalva_segment_max_slice_info.json",
            },
            {
                "name": "Sinutubular junction",
                "file": "sinotubular_junction_segment_max_slice_info.json",
            },
            {"name": "Ascending", "file": "ascending_segment_max_slice_info.json"},
            {"name": "Aortic arch", "file": "aortic_arch_segment_max_slice_info.json"},
            {"name": "Descending", "file": "descending_segment_max_slice_info.json"},
            {"name": "Abdominal", "file": "abdominal_segment_max_slice_info.json"},
            {"name": "Infrarenal", "file": "infrarenal_segment_max_slice_info.json"},
        ]
        # self.message_text += f"Max cross sectional area: {cut_area:.0f} mm2\n" \
        #                      f"Diameters: {min_diam:.0f} and {max_diam:.0f} mm\n"

        self.message_text += "\nMax cross sectional areas:\n"

        for c in cut_stats:
            name = c["name"]
            file = c["file"]
            file = f"{cl_dir}{file}"

            stats = read_json_file(file)
            if stats:
                cut_area = stats["area"]
                self.message_text += f"{name}: {cut_area} mm2\n"

    def set_aortic_tortuosity_index_statistics(self, stats_folder):
        stats_file = f"{stats_folder}/aorta_statistics.json"
        ati_stats = read_json_file(stats_file)
        if not ati_stats:
            print(f"Could not read {stats_file}")
            return

        if "annulus_aortic_length" in ati_stats:
            cl_length = ati_stats["annulus_aortic_length"]
            self.message_text += f"\nTotal aortic length: {cl_length / 10.0:.1f} cm\n"
        self.message_text += "\nAortic tortuosity index:\n"
        if "annulus_aortic_tortuosity_index" in ati_stats:
            ati = ati_stats["annulus_aortic_tortuosity_index"]
            self.message_text += f"Annulus: {ati:.2f}\n"
        ati = ati_stats.get("ascending_aortic_tortuosity_index", None)
        if ati:
            self.message_text += f"Ascending: {ati:.2f}\n"
        ati = ati_stats.get("descending_aortic_tortuosity_index", None)
        if ati:
            self.message_text += f"Descending: {ati:.2f}\n"
        if "diaphragm_aortic_tortuosity_index" in ati_stats:
            ati = ati_stats["diaphragm_aortic_tortuosity_index"]
            self.message_text += f"Diaphragm: {ati:.2f}\n"
        if "infrarenal_aortic_tortuosity_index" in ati_stats:
            ati = ati_stats["infrarenal_aortic_tortuosity_index"]
            self.message_text += f"Infrarenal: {ati:.2f}\n"

    def set_aortic_aneurysm_sac_statistics(self, stats_folder):
        type_file = f"{stats_folder}/aorta_scan_type.json"
        stats_file = f"{stats_folder}/aorta_aneurysm_sac_stats.json"
        if not os.path.exists(type_file):
            print(f"Missing file {type_file}")
            return

        scan_type_stats = read_json_file(type_file)
        if not scan_type_stats:
            print(f"Missing file {type_file}")
            return
        scan_type = scan_type_stats["scan_type"]

        if scan_type in ["1", "2", "4", "5"]:
            stats = read_json_file(stats_file)
            if not stats:
                print(f"Could not read {stats_file}")
                return

            original_aorta_volume = stats["original_aorta_volume"]
            aorta_lumen = stats["aorta_lumen"]
            calcification_volume = stats["calcification_volume"]
            if aorta_lumen < 1:
                print(f"Something wrong with {stats_file} aorta_lumen={aorta_lumen}")
                return

            tot_vol = aorta_lumen + calcification_volume
            ratio = original_aorta_volume / tot_vol
            dif_percent = abs(original_aorta_volume - tot_vol) / tot_vol * 100.0
            q95_dists = stats["q95_distances"]
            self.message_text += "\nAneurysm sac info:\n"
            self.message_text += f"Ratio: {ratio:.2f}\n"
            self.message_text += f"Enlarged percent: {dif_percent:.0f}%\n"
            self.message_text += f"Q95 distances: {q95_dists:0.1f} mm\n"
        else:
            print(f"Can not set aortic aneurysm sac statistics for scan type: {scan_type}")

    def set_aortic_calcification_statistics(self, stats_folder):
        type_file = f"{stats_folder}aorta_scan_type.json"
        stats_file = f"{stats_folder}aorta_calcification_stats.json"

        if not os.path.exists(type_file):
            print(f"Missing file {type_file}")
            return

        scan_type_stats = read_json_file(type_file)
        if not scan_type_stats:
            print(f"Missing file {type_file}")
            return
        scan_type = scan_type_stats["scan_type"]

        if scan_type in ["1", "2", "5", "4"]:
            stats = read_json_file(stats_file)
            if not stats:
                print(f"Could not read {stats_file}")
                return

            # aorta_stats = pt.read_json_file(tot_stats_file)
            # if aorta_stats is None:
            #     print(f"Found no stats file {tot_stats_file}")
            #     return

            calcification_volume = stats["calcification_volume"]
            aorta_lumen_volume = stats["aorta_lumen_volume"]
            # total_volume = aorta_stats["tot_vol"]

            self.message_text += "\nCalcification info:\n"
            self.message_text += f"Calcified volume: {calcification_volume / 1000.0:.1f} cm3\n"
            self.message_text += f"Lumen volume: {aorta_lumen_volume / 1000.0:.1f} cm3\n"
            if aorta_lumen_volume > 0:
                ratio = calcification_volume / aorta_lumen_volume * 100
                self.message_text += f"Percent of total: {ratio:.2f}%\n"
        else:
            print(f"Can not set calcification statistics for scan type: {scan_type}")

    def set_precomputed_straight_longitudinal_slices(self, cl_folder):
        """
        Here the longitudinal are precomputed as png files in the vmtk processing script
        """
        plane_file_1 = f"{cl_folder}straight_volume_mid_cut.png"
        plane_file_2 = f"{cl_folder}straight_volume_mid_cut_2.png"

        png_reader = vtk.vtkPNGReader()
        if not png_reader.CanReadFile(plane_file_1):
            print(f"Can not read {plane_file_1}")
            return
        png_reader.SetFileName(plane_file_1)
        png_reader.Update()

        image_viewer = vtk.vtkImageViewer2()
        image_viewer.SetInputConnection(png_reader.GetOutputPort())
        image_viewer.SetRenderWindow(self.ren_win)
        image_viewer.SetRenderer(self.ren_straight_1)
        image_viewer.Render()

        self.ren_straight_1.GetActiveCamera().ParallelProjectionOn()
        self.ren_straight_1.ResetCameraScreenSpace()

        png_reader = vtk.vtkPNGReader()
        if not png_reader.CanReadFile(plane_file_2):
            print(f"Can not read {plane_file_2}")
            return
        png_reader.SetFileName(plane_file_2)
        png_reader.Update()

        image_viewer = vtk.vtkImageViewer2()
        image_viewer.SetInputConnection(png_reader.GetOutputPort())
        image_viewer.SetRenderWindow(self.ren_win)
        image_viewer.SetRenderer(self.ren_straight_2)
        image_viewer.Render()

        self.ren_straight_2.GetActiveCamera().ParallelProjectionOn()
        self.ren_straight_2.ResetCameraScreenSpace()
