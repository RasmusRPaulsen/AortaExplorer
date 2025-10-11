import os.path
import vtk
import numpy as np
import csv
import json
from vtk.util.numpy_support import vtk_to_numpy
from scipy import ndimage
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
        self.ren_win.SetWindowName('Segmentation view')

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
        self.add_text_to_render(self.ren_text, self.message_text, color=(1.0, 1.0, 1.0), position=pos,
                                font_size=font_size)
        self.add_text_to_render(self.ren_patient_text, self.patient_text, color=(0.0, 1.0, 0.0), position=pos,
                                font_size=font_size)
        self.add_text_to_render(self.ren_warning_text, self.warning_text, color=(1.0, 1.0, 0.0), position=pos,
                                font_size=font_size)
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
        self.add_text_to_render(self.ren_patient_text, self.patient_text, color=(0.0, 1.0, 0.0), position=pos)
        self.add_text_to_render(self.ren_warning_text, self.warning_text, color=(1.0, 1.0, 0.0), position=pos)

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

    def set_patient_id_from_translation_table(self, settings):
        translation_table_name = settings.get("translation_table", None)
        if translation_table_name is None or translation_table_name == "":
            print(f"No translation table given")
            return
        if not os.path.exists(translation_table_name):
            print(f"Translation table file {translation_table_name} not found")
            return

        translation_table_file = open(translation_table_name, "r")
        translation_table = csv.DictReader(translation_table_file, delimiter=",", skipinitialspace=True)
        img_file = settings["input_file"]

        # patient_id,number,zero_number,pseudo_id
        for patient in translation_table:
            # print(patient)
            patient_id = patient["patient_id"]
            pseudo_id = patient["pseudo_id"]

            if pseudo_id in img_file:
                self.patient_text = f"{patient_id}"
                id_len = len(self.patient_text)
                if id_len == 10:
                    bdate = self.patient_text[0:6]
                    num4 = self.patient_text[6:10]
                    self.patient_text = f"{bdate}-{num4}"
                elif id_len == 9:
                    bdate = self.patient_text[0:5]
                    num4 = self.patient_text[5:9]
                    self.patient_text = f"0{bdate}-{num4}"
                elif id_len == 11:
                    # All good. CPR number with dash
                    pass
                else:
                    print(f"Could not recognise patientid {self.patient_text}")
                return


    def set_sitk_image_file(self, input_file, img_mask=None):
        """
        Add a simple ITK image to the renderer using a volume renderer.
        If a mask is provided, the volume data is first masked. This can for example remove scanner beds etc.
        """
        self.vtk_image = read_nifti_itk_to_vtk(input_file, img_mask, flip_for_volume_rendering=True)
        if self.vtk_image is None:
            return

        # vtk_ext = self.vtk_image.GetExtent()
        # vtk_org = self.vtk_image.GetOrigin()
        vtk_dim = self.vtk_image.GetDimensions()
        vtk_spc = self.vtk_image.GetSpacing()


        img_txt = f'Spacing: ({vtk_spc[0]:.2f}, {vtk_spc[1]:.2f}, {vtk_spc[2]:.2f}) mm\n' \
                  f'Dimensions: ({vtk_dim[0]}, {vtk_dim[1]}, {vtk_dim[2]}) vox\n' \
                  f'Size: ({vtk_spc[0] * vtk_dim[0] / 10.0:.1f}, {vtk_spc[1] * vtk_dim[1] / 10.0:.1f}, ' \
                  f'{vtk_spc[2] * vtk_dim[2] / 10.0:.1f}) cm\n' \

        self.message_text += img_txt

        # vtk_image_no_dir = self.vtk_image
        # vtk_image_no_dir.DeepCopy(self.vtk_image)

        # Get direction to set camera (not needed when we do the brutal flip of image data in the load routine)
        # dir_mat = self.vtk_image.GetDirectionMatrix().GetData()
        # print(f"Dir mat: {dir_mat}")
        # dir_val = dir_mat[4]
        dir_val = 1

        # Reset direction matrix, since volume render do not cope good with it
        direction = [1, 0, 0.0, 0, 1, 0.0, 0.0, 0.0, 1.0]
        self.vtk_image.SetDirectionMatrix(direction)

        # img_data = self.vtk_image.GetPointData().GetScalars()
        # vol_np = vtk_to_numpy(self.vtk_image.GetPointData().GetScalars())

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
        # We make a copy to avoid changing the original offsets using the *= operator
        view_offsets =  [500, -1000, 0]
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

    def add_text_to_render(self, ren, message, color=(1, 1, 1), position=(5, 5), font_size=10):
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


    def convert_label_map_to_surface(self, label_name, output_file):
        if not os.path.exists(label_name):
            print(f"Can not find {label_name}")
            return False
        vtk_img = pt.read_nifti_itk_to_vtk(label_name)
        if vtk_img is None:
            return False

        print(f"Generating: {output_file}")
        mc = vtk.vtkDiscreteMarchingCubes()
        mc.SetInputData(vtk_img)
        mc.SetNumberOfContours(1)
        mc.SetValue(0, 1)
        mc.Update()

        if mc.GetOutput().GetNumberOfPoints() < 10:
            print(f"No isosurface found in {label_name}")
            return False

        writer = vtk.vtkPolyDataWriter()
        writer.SetInputConnection(mc.GetOutputPort())
        writer.SetFileTypeToBinary()
        writer.SetFileName(output_file)
        writer.Write()

        return True

    def generate_actor_from_surface_file(self, surf_file, color=np.array([1, 0, 0]), opacity=1.0, smooth="heavy"):
        if not os.path.exists(surf_file):
            return None

        print(f"Generating actor from: {surf_file}")

        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(surf_file)
        reader.Update()
        surface = reader.GetOutput()

        n_points = surface.GetNumberOfPoints()
        if n_points < 2:
            print(f"Not enough points in {surf_file}")
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
        smoother.SetInputConnection(reader.GetOutputPort())
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
        actor.GetProperty().SetColor(color[0], color[1], color[2])

        return actor


    def generate_actor_from_surface(self, surface, color=np.array([1, 0, 0]), opacity=1.0, smooth="heavy"):
        n_points = surface.GetNumberOfPoints()
        if n_points < 2:
            print(f"Not enough points in surface")
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
        actor.GetProperty().SetColor(color[0], color[1], color[2])

        return actor

    def generate_actors_from_segments(self):
        segm_base_dir = self.settings["segment_base_dir"]
        surf_output_dir = self.settings["surface_dir"]

        for idx, segment_dict in enumerate(self.segments):
            task = segment_dict["task"]
            segment = segment_dict["segment"]
            color_name = segment_dict["color"]
            rgba = [0.0, 0.0, 0.0, 0.0]
            vtk.vtkNamedColors().GetColor(color_name, rgba)
            opacity = segment_dict["opacity"]
            segm_name = f"{segm_base_dir}{task}/{segment}.nii.gz"
            surface_name = f"{surf_output_dir}{task}_{segment}_surface.vtk"
            if not os.path.exists(surface_name):
                self.convert_label_map_to_surface(segm_name, surface_name)

            actor = self.generate_actor_from_surface_file(surface_name, rgba, opacity, smooth="heavy")
            if actor is not None:
                self.actors.append(actor)

    def generate_actors_from_segment_file_name_o1(self, segm_name, surface_name, color_name, opacity, smooth="heavy"):
        if not os.path.exists(surface_name):
            self.convert_label_map_to_surface(segm_name, surface_name)

        rgba = [0.0, 0.0, 0.0, 0.0]
        vtk.vtkNamedColors().GetColor(color_name, rgba)
        actor = self.generate_actor_from_surface_file(surface_name, rgba, opacity, smooth=smooth)
        if actor is not None:
            self.actors.append(actor)

    def generate_actors_from_segment_file_name(self, segm_name, surface_name, color_name, opacity, smooth="heavy"):
        surface = surfutils.convert_label_map_to_surface(segm_name)
        if surface is not None:
            rgba = [0.0, 0.0, 0.0, 0.0]
            vtk.vtkNamedColors().GetColor(color_name, rgba)
            actor = self.generate_actor_from_surface(surface, rgba, opacity, smooth=smooth)
            if actor is not None:
                self.actors.append(actor)


    def generate_actors_from_combined_segments(self):
        segm_base_dir = self.settings["segment_base_dir"]
        surf_output_dir = self.settings["surface_dir"]
        scan_id = self.settings["scan_id"]
        segmentation_model = self.settings["segmentation_model"]
        if segmentation_model == "TotalSegmentator":
            # TODO: Define task somewhere else and make different tasks
            task = "total"
            label_name = os.path.join(segm_base_dir, f"{task}/{task}.nii.gz")
        elif segmentation_model == "nnunet_abdominal_1":
            # TODO: Define task somewhere else and make different tasks
            task = "abdominal_1"
            label_name = os.path.join(segm_base_dir, f"{scan_id}_{task}.nii.gz")
        else:
            print(f"Segmentation model: {segmentation_model} not known")
            return False

        vtk_img = pt.read_nifti_itk_to_vtk(label_name)
        if vtk_img is None:
            return False

        for idx, segment_dict in enumerate(self.segments):
            segm_id = segment_dict["id"]
            segment = segment_dict["segment"]
            color_name = segment_dict["color"]
            rgba = [0.0, 0.0, 0.0, 0.0]
            vtk.vtkNamedColors().GetColor(color_name, rgba)
            opacity = segment_dict["opacity"]
            # segm_name = f"{segm_base_dir}{task}/{segment}.nii.gz"
            surface_name = f"{surf_output_dir}{task}_{segment}_surface.vtk"
            if os.path.exists(surface_name):
                print(f"{surface_name} exists - not recomputing")
            else:
                print(f"Generating: {surface_name}")
                mc = vtk.vtkDiscreteMarchingCubes()
                mc.SetInputData(vtk_img)
                mc.SetNumberOfContours(1)
                mc.SetValue(0, float(segm_id))
                mc.Update()

                if mc.GetOutput().GetNumberOfPoints() < 10:
                    print(f"No isosurface found in {label_name}")
                else:
                    writer = vtk.vtkPolyDataWriter()
                    writer.SetInputConnection(mc.GetOutputPort())
                    writer.SetFileTypeToBinary()
                    writer.SetFileName(surface_name)
                    writer.Write()

            actor = self.generate_actor_from_surface_file(surface_name, rgba, opacity, smooth="heavy")
            if actor is not None:
                self.actors.append(actor)

    def generate_actors_from_combined_segments_with_tasks(self):
        segm_base_dir = self.settings["segment_base_dir"]
        surf_output_dir = self.settings["surface_dir"]
        scan_id = self.settings["scan_id"]
        segmentation_model = self.settings["segmentation_model"]

        # To avoid reading the same label file many times
        old_task = None
        vtk_img = None
        for idx, segment_dict in enumerate(self.segments):
            segm_id = segment_dict["id"]
            segment = segment_dict["segment"]
            color_name = segment_dict["color"]
            rgba = [0.0, 0.0, 0.0, 0.0]
            vtk.vtkNamedColors().GetColor(color_name, rgba)
            opacity = segment_dict["opacity"]
            task = segment_dict["task"]
            visible = segment_dict["visible"]

            if visible:
                # Read label image if need task is coming in
                if old_task != task:
                    if segmentation_model == "TotalSegmentator":
                        # label_name = os.path.join(segm_base_dir, f"{task}/{task}.nii.gz")
                        ts_base_dir = self.settings["totalsegmentator_dir"]
                        label_name = f"{ts_base_dir}{scan_id}/segmentations/{task}/{task}.nii.gz"
                    elif segmentation_model == "nnunet_abdominal_1":
                        # task = "abdominal_1"
                        label_name = os.path.join(segm_base_dir, f"{scan_id}_{task}.nii.gz")
                    elif segmentation_model == "CustomImageCAS":
                        label_name = os.path.join(segm_base_dir, f"image_cas_ts_laa_combined.nii.gz")
                    elif segmentation_model == "BartholinatorV1":
                        segm_folder = self.settings["bartholinator_dir"]
                        label_name = f"{segm_folder}{scan_id}_labels.nii.gz"
                    else:
                        ts_base_dir = self.settings["totalsegmentator_dir"]
                        # print(f"Segmentation model: {segmentation_model} is not known. We try our best...")
                        label_name = f"{ts_base_dir}{scan_id}/segmentations/{task}/{task}.nii.gz"

                    vtk_img = pt.read_nifti_itk_to_vtk(label_name)
                    if vtk_img is None:
                        pt.write_message_to_log_file(self.settings, f"Visualization: "
                                                                    f"Could not find {label_name}", scan_id, "error")
                        print(f"Could not convert {label_name} to VTK")
                    old_task = task

                if vtk_img is not None:
                    surface_name = f"{surf_output_dir}{task}_{segment}_surface.vtk"
                    if os.path.exists(surface_name):
                        print(f"{surface_name} exists - not recomputing")
                    else:
                        print(f"Generating: {surface_name}")
                        mc = vtk.vtkDiscreteMarchingCubes()
                        mc.SetInputData(vtk_img)
                        mc.SetNumberOfContours(1)
                        mc.SetValue(0, float(segm_id))
                        mc.Update()

                        if mc.GetOutput().GetNumberOfPoints() < 10:
                            print(f"No isosurface found in {label_name} for {surface_name}")
                        else:
                            writer = vtk.vtkPolyDataWriter()
                            writer.SetInputConnection(mc.GetOutputPort())
                            writer.SetFileTypeToBinary()
                            writer.SetFileName(surface_name)
                            writer.Write()

                    # actor = self.generate_actor_from_surface_file(surface_name, rgba, opacity, smooth="heavy")
                    actor = self.generate_actor_from_surface_file(surface_name, rgba, opacity, smooth="none")
                    if actor is not None:
                        self.actors.append(actor)

    def generate_actors_from_landmarks(self):
        lm_base_dir = self.settings["landmark_dir"]
        for lm in self.landmarks:
            lm_file = f'{lm_base_dir}{lm["file"]}'
            try:
                point = pt.read_landmarks(lm_file)
            except:
                print(f"Could not read landmark file: {lm_file}")
                point = None

            size = lm["size"]
            color = lm["color"]
            opacity = lm["opacity"]
            rgba = [0.0, 0.0, 0.0, 0.0]
            vtk.vtkNamedColors().GetColor(color, rgba)
            if point is not None:
                sphere = vtk.vtkSphereSource()
                sphere.SetCenter(point)
                sphere.SetRadius(size)
                sphere.SetThetaResolution(20)
                sphere.SetPhiResolution(20)
                sphere.Update()

                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(sphere.GetOutputPort())
                mapper.ScalarVisibilityOff()

                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetOpacity(opacity)
                actor.GetProperty().SetColor(rgba[0], rgba[1], rgba[2])

                self.actors.append(actor)

    def generate_actors_from_centerlines(self, cl_folder):
        for cl in self.centerlines:
            cl_file = f'{cl_folder}{cl["file"]}'
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

    def add_message_text(self, text):
        self.message_text += text


class RenderAortaData(RenderTotalSegmentatorData):
    def __init__(self, win_size, render_to_file, stats_folder, segm_folder, cl_folder):
        super().__init__(win_size, render_to_file)
        # print(f"Initialising aorta renderer")

        n_aorta_parts = 1
        parts_stats = read_json_file(f'{stats_folder}aorta_parts.json')
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
        self.generate_actors_from_segment_file_name(segm_name_calc, None, "Ivory", 1.0,
                                                    smooth="light")

        segm_name = f"{segm_folder}iliac_artery_left_top.nii.gz"
        # surf_name = f"{surf_output_dir}computed_iliac_artery_left_top.vtk"
        self.generate_actors_from_segment_file_name(segm_name, None, "DarkSalmon", 1.0,
                                                    smooth="heavy")

        segm_name = f"{segm_folder}iliac_artery_right_top.nii.gz"
        # surf_name = f"{surf_output_dir}computed_iliac_artery_right_top.vtk"
        self.generate_actors_from_segment_file_name(segm_name, None, "PaleVioletRed", 1.0,
                                                    smooth="heavy")

        # TODO: THis is hacky and should be updated
        if n_aorta_parts == 1:
            aneurysm_sac_stats_file = f'{stats_folder}aorta_aneurysm_sac_stats.json'
            aneurysm_sac_stats = read_json_file(aneurysm_sac_stats_file)
            if aneurysm_sac_stats:
                aneurysm_sac_ratio = aneurysm_sac_stats["aorta_ratio"]
                q95_dists = aneurysm_sac_stats["q95_distances"]
                if aneurysm_sac_ratio > 1.2 and q95_dists > 5:
                    self.segments.append({"task": "total", "id": 52, "segment": "aorta",
                     "max_components": 2, "min_component_size": 5000, "color": "OldLace", "opacity": 0.6,
                     "visible": True})

        # self.generate_actors_from_combined_segments_with_tasks()

        if n_aorta_parts == 1:
            self.centerlines.append({"name": "aorta_center_line", "file": f"aorta_centerline.vtp", "size": 1,
                                     "color": "white", "opacity": 1.0})
        elif n_aorta_parts == 2:
            self.centerlines.append({"name": "aorta_center_line_annulus", "file": f"aorta_centerline_annulus.vtp",
                                     "size": 1, "color": "white", "opacity": 1.0})
            self.centerlines.append({"name": "aorta_center_line_descending", "file": f"aorta_centerline_descending.vtp",
                                     "size": 1, "color": "white", "opacity": 1.0})

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
        c = [(actor_bounds[0] + actor_bounds[1]) / 2, (actor_bounds[2] + actor_bounds[3]) / 2,
                   (actor_bounds[4] + actor_bounds[5]) / 2]

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
        stats_file = f'{stats_folder}aorta_statistics.json'
        scan_type_file = f'{stats_folder}aorta_scan_type.json'

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

        aorta_txt = f'\nAorta HU avg: {aorta_stats["avg_hu"]:.0f} ({aorta_stats["cl_mean"]:.0f})\n' \
                    f'std.dev: {aorta_stats["std_hu"]:.0f} ({aorta_stats["cl_std"]:.0f})\n' \
                    f'median: {aorta_stats["med_hu"]:.0f} ({aorta_stats["cl_med"]:.0f})\n' \
                    f'99%: {aorta_stats["q99_hu"]:.0f} ({aorta_stats["cl_q99"]:.0f})\n' \
                    f'1%: {aorta_stats["q01_hu"]:.0f} ({aorta_stats["cl_q01"]:.0f})\n' \
                    f'Aorta vol: {aorta_stats["tot_vol"]/1000.0:.0f} cm3\n' \
                    f'scan type: {scan_type}\n' \
                    f'Aorta Surface area: {aorta_stats["surface_area"] / 100.0:.1f} cm2\n'
                    # f'Centerline length: {cl_length / 10.0:.1f} cm\n' \

        self.message_text += aorta_txt

    def set_plot_data(self, stats_folder, cl_folder):
        n_aorta_parts = 1
        parts_stats = read_json_file(f'{stats_folder}/aorta_parts.json')
        if parts_stats:
            n_aorta_parts = parts_stats["aorta_parts"]

        if n_aorta_parts == 2:
            data_file = f'{cl_folder}straight_labelmap_sampling_annulus.csv'
            max_in_files = ["aortic_arch_segment_max_slice_info",
                            "ascending_segment_max_slice_info",
                            "distensability_segment_avg_slice_info",
                            "sinotubular_junction_segment_max_slice_info",
                            "sinus_of_valsalva_segment_max_slice_info",
                            "lvot_segment_max_slice_info"]
            max_rgb = [[255, 0, 0], [0, 255, 255], [255, 128, 128], [128, 0, 128], [0, 128, 255],
                       [200, 255, 100]]
            max_rgb = np.divide(max_rgb, 255.0)
        elif n_aorta_parts == 1:
            data_file = f'{cl_folder}straight_labelmap_sampling.csv'
            max_in_files = ["infrarenal_segment_max_slice_info",
                            "abdominal_segment_max_slice_info",
                            "aortic_arch_segment_max_slice_info",
                            "ascending_segment_max_slice_info",
                            "sinotubular_junction_segment_max_slice_info",
                            "sinus_of_valsalva_segment_max_slice_info",
                            "descending_segment_max_slice_info",
                            "lvot_segment_max_slice_info"]
            max_rgb = [[255, 255, 0], [255, 128, 0], [255, 0, 0], [0, 255, 255], [128, 0, 128], [0, 128, 255], [0, 255, 0],
                       [200, 255, 100]]
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
            cut_plane_stats = self.read_json_file(file_name)
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


    def set_slice_data_with_image_reslicer(self, settings):
        """
        This version is currently not working on all files due to issues with VTK setdirections
        """
        file_name = settings["input_file"]
        # plane_file = f'{settings["centerline_dir"]}cl_max_plane.txt'
        plane_file = f'{settings["centerline_dir"]}cl_max_cut_stats.json'
        cut_file = f'{settings["centerline_dir"]}cl_max_cut.vtk'
        infrarenal_in = f'{settings["centerline_dir"]}/infrarenal_section.json'

        if not os.path.exists(plane_file):
            print(f"Could not open {plane_file}")
            return

        if not os.path.exists(file_name):
            print(f"Could not open {file_name}")
            return

        if not os.path.exists(cut_file):
            print(f"Could not open {cut_file}")
            return

        try:
            with open(plane_file, 'r') as openfile:
                cut_plan_stats = json.load(openfile)
        except IOError as e:
            print(f"I/O error({e.errno}): {e.strerror}: {plane_file}")
            return

        x, y, z = cut_plan_stats["origin"]
        nx, ny, nz = cut_plan_stats["normal"]
        cut_area = cut_plan_stats["area"]

        self.message_text += f"Max cross sectional area: {cut_area:.0f} mm2"

        if os.path.exists(infrarenal_in):
            try:
                with open(infrarenal_in, 'r') as openfile:
                    infrarenal_stats = json.load(openfile)
            except IOError as e:
                print(f"I/O error({e.errno}): {e.strerror}: {infrarenal_in}")
                return

            pos = infrarenal_stats["cl_pos"]
            normal = infrarenal_stats["cl_normal"]

            # renal point actor
            disc = vtk.vtkDiscSource()
            disc.SetCenter(pos)
            disc.SetNormal(normal)
            disc.SetOuterRadius(40)
            # sphere = vtk.vtkSphereSource()
            # sphere.SetCenter(pos)
            # sphere.SetRadius(5.0)
            # sphere.SetThetaResolution(20)
            # sphere.SetPhiResolution(20)
            # sphere.Update()

            mapper = vtk.vtkPolyDataMapper()
            # mapper.SetInputConnection(sphere.GetOutputPort())
            mapper.SetInputConnection(disc.GetOutputPort())
            mapper.ScalarVisibilityOff()

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetOpacity(1.0)
            actor.GetProperty().SetColor(0, 0, 1)

            self.ren_3d_1.AddActor(actor)
            self.ren_3d_2.AddActor(actor)


        #
        #
        # # f = open(plane_file, "r")
        # with open(plane_file) as f:
        #     line = f.readline()
        #     x, y, z = np.double(line.split(" "))
        #     line = f.readline()
        #     nx, ny, nz = np.double(line.split(" "))

        vtk_img = pt.read_nifti_itk_to_vtk(file_name)

        im = vtk.vtkImageResliceMapper()
        im.SetInputData(vtk_img)
        im.SliceFacesCameraOn()
        im.SliceAtFocalPointOn()
        im.BorderOff()

        ip = vtk.vtkImageProperty()
        ip.SetColorWindow(600)
        ip.SetColorLevel(200)
        ip.SetAmbient(0.0)
        ip.SetDiffuse(1.0)
        ip.SetOpacity(1.0)
        ip.SetInterpolationTypeToLinear()

        ia = vtk.vtkImageSlice()
        ia.SetMapper(im)
        ia.SetProperty(ip)

        # self.ren_slice.AddViewProp(ia)
        self.ren_slice.AddViewProp(ia)

        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(cut_file)
        reader.Update()
        cut_vtk = reader.GetOutput()
        n_points = cut_vtk.GetNumberOfPoints()
        if n_points < 2:
            print(f"No points in {cut_file}")
            return

        vtk_tube_filter = vtk.vtkTubeFilter()
        vtk_tube_filter.SetInputData(cut_vtk)
        vtk_tube_filter.SetNumberOfSides(16)
        vtk_tube_filter.SetRadius(0.3)
        vtk_tube_filter.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(vtk_tube_filter.GetOutputPort())
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(1.0)
        actor.GetProperty().SetColor(0, 1, 0)

        bounds = actor.GetBounds()
        # diagonal length of bounds
        l_diag = np.linalg.norm((bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]))
        w_size = l_diag * 1.2 / 2.0

        self.ren_slice.AddActor(actor)

        # mid point actor
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter((x, y, z))
        sphere.SetRadius(1.0)
        sphere.SetThetaResolution(20)
        sphere.SetPhiResolution(20)
        sphere.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(1.0)
        actor.GetProperty().SetColor(1, 0, 0)

        self.ren_slice.AddActor(actor)

        cam1 = self.ren_slice.GetActiveCamera()
        cam1.ParallelProjectionOn()
        cam1.SetParallelScale(w_size)

        # https://public.kitware.com/pipermail/vtkusers/2011-September/070130.html
        d = 1000
        cam1.SetFocalPoint(x, y, z)
        cam1.SetPosition(x + d * nx, y + d * ny, z + d * nz)
        cam1.Roll(180.0)
        # cam1.SetViewUp(0, 0, 1)
        self.ren_slice.ResetCameraClippingRange()

        # self.ren_volume.SetActiveCamera(self.ren_3d_1.GetActiveCamera())

    def create_disk_actor(self, pos, normal):
        """
        For some kind of weird reason, SetCenter and SetNormal does not work for DiskSource
        We have made a hack following:
        https://stackoverflow.com/questions/53244072/rotate-vtkdisksource-vtkpolydatamapper-by-normals
        """
        disk = vtk.vtkDiskSource()
        disk.SetCircumferentialResolution(100)
        # disk.SetCenter(pos[0], pos[1], pos[2])
        # disk.SetNormal(normal)
        disk.SetOuterRadius(30)

        z_axis = [0., 0., 1.]
        axis = np.cross(z_axis, normal)
        angle = np.arccos(np.dot(z_axis, normal))
        transform = vtk.vtkTransform()
        # Put the disks a bit above the mesh, otherwise they might be partially burried
        transform.Translate(pos)
        transform.RotateWXYZ(np.degrees(angle), *axis)
        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetTransform(transform)
        transform_filter.SetInputConnection(disk.GetOutputPort())
        transform_filter.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(transform_filter.GetOutputPort())
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(0.95)
        actor.GetProperty().SetColor(0.9, 0.9, 1)
        return actor

    def create_cylinder_actor(self, pos, normal, radius=30, height=1, rgba = [0.0, 1.0, 0.0, 0.0], opacity=1.0):
        """
        """
        cylinder = vtk.vtkCylinderSource()
        cylinder.SetResolution(100)
        cylinder.CappingOn()
        cylinder.SetRadius(radius)
        cylinder.SetHeight(height)

        y_axis = [0., 1., 0.]
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

    def set_renal_info(self, settings):
        """
        Visualise disk for the renal section on the 3D view
        """
        infrarenal_in = f'{settings["centerline_dir"]}/infrarenal_section.json'
        if os.path.exists(infrarenal_in):
            try:
                with open(infrarenal_in, 'r') as openfile:
                    infrarenal_stats = json.load(openfile)
            except IOError as e:
                print(f"I/O error({e.errno}): {e.strerror}: {infrarenal_in}")
                return

            pos = infrarenal_stats["cl_pos"]
            normal = infrarenal_stats["cl_normal"]
            # actor_high = self.create_disk_actor(pos, normal)
            actor_high = self.create_cylinder_actor(pos, normal, radius=20, height=0.5, rgba=[1, 1, 0], opacity=0.9)

            pos = infrarenal_stats["low_cl_pos"]
            normal = infrarenal_stats["low_cl_normal"]
            # actor_low = self.create_disk_actor(pos, normal)
            actor_low = self.create_cylinder_actor(pos, normal, radius=20, height=0.5, rgba=[1, 1, 0], opacity=0.9)

            self.ren_3d_1.AddActor(actor_high)
            self.ren_3d_2.AddActor(actor_high)
            self.ren_3d_1.AddActor(actor_low)
            self.ren_3d_2.AddActor(actor_low)
        else:
            print("Infrarenal data not yet computed")

    def set_aortic_arch_info(self, settings):
        """
        Visualise disk for the aortic arch the 3D view
        """
        arch_in = f'{settings["centerline_dir"]}/aortic_arch.json'
        if os.path.exists(arch_in):
            try:
                with open(arch_in, 'r') as openfile:
                    arch_stats = json.load(openfile)
            except IOError as e:
                print(f"I/O error({e.errno}): {e.strerror}: {arch_in}")
                return

            pos = arch_stats["max_cl_pos"]
            normal = arch_stats["max_cl_normal"]
            actor_high = self.create_cylinder_actor(pos, normal, radius=30, height=0.5, rgba=[1, 1, 1], opacity=0.9)

            pos = arch_stats["min_cl_pos"]
            normal = arch_stats["min_cl_normal"]
            actor_low = self.create_cylinder_actor(pos, normal, radius=30, height=0.5, rgba=[1, 1, 1], opacity=0.9)

            self.ren_3d_1.AddActor(actor_high)
            self.ren_3d_2.AddActor(actor_high)
            self.ren_3d_1.AddActor(actor_low)
            self.ren_3d_2.AddActor(actor_low)
        else:
            print("Aortic arch data not yet computed")

    def set_diaphragm_info(self, settings):
        """
        Visualise disk for the diaphragm level on 3D view
        """
        arch_in = f'{settings["centerline_dir"]}/diaphragm.json'
        if os.path.exists(arch_in):
            try:
                with open(arch_in, 'r') as openfile:
                    dia_stats = json.load(openfile)
            except IOError as e:
                print(f"I/O error({e.errno}): {e.strerror}: {arch_in}")
                return

            pos = dia_stats["diaphragm_cl_pos"]
            normal = dia_stats["diaphragm_cl_normal"]
            actor = self.create_cylinder_actor(pos, normal, radius=30, height=0.5, rgba=[1, 1, 1], opacity=0.9)

            self.ren_3d_1.AddActor(actor)
            self.ren_3d_2.AddActor(actor)
        else:
            print("Diaphragm data not yet computed")


    def line_actor_from_points(self, p_1, p_2):
        line = vtk.vtkLineSource()
        line.SetPoint1(p_1)
        line.SetPoint2(p_2)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(line.GetOutputPort())
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.GetProperty().SetRenderLinesAsTubes(1)
        actor.GetProperty().SetLineWidth(2)
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(1)
        actor.GetProperty().SetColor(0.0, 1.0, 1.0)
        return actor

    def read_json_file(self, json_name):
        if os.path.exists(json_name):
            try:
                with open(json_name, 'r') as openfile:
                    json_stuff = json.load(openfile)
                    return json_stuff
            except IOError as e:
                print(f"I/O error({e.errno}): {e.strerror}: {json_name}")
                return None
        return None

    def set_all_max_cut_data(self, cl_folder):
        """
        Generate a disk where max cut is found
        """
        cl_dir = cl_folder
        max_in_files = ["infrarenal_segment_max_slice_info",
                        "abdominal_segment_max_slice_info",
                        "aortic_arch_segment_max_slice_info",
                        "ascending_segment_max_slice_info",
                        "distensability_segment_avg_slice_info",
                        "sinotubular_junction_segment_max_slice_info",
                        "sinus_of_valsalva_segment_max_slice_info",
                        "descending_segment_max_slice_info",
                        "lvot_segment_max_slice_info"]
        max_rgb = [[255, 255, 0], [255, 128, 0], [255, 0, 0], [0, 255, 255], [255, 128, 128], [128, 0, 128], [0, 128, 255], [0, 255, 0],
                   [200, 255, 100]]
        max_rgb = np.divide(max_rgb, 255.0)

        for idx, pfile in enumerate(max_in_files):
            file_name = f"{cl_dir}{pfile}.json"
            cut_plane_stats = self.read_json_file(file_name)
            if cut_plane_stats:
                pos = cut_plane_stats["origin"]
                normal = cut_plane_stats["normal"]
                max_diam = cut_plane_stats["max_diameter"]
                radius = max_diam / 2 * 1.10
                rgb = max_rgb[idx]

                actor_cut = self.create_cylinder_actor(pos, normal, radius=radius, height=1.5, rgba=rgb,
                                                       opacity=1.0)

                self.ren_3d_1.AddActor(actor_cut)
                self.ren_3d_2.AddActor(actor_cut)


    def set_max_cut_data(self, settings):
        """
        Generate a disk where max cut is found
        """
        plane_file = f'{settings["centerline_dir"]}cl_max_cut_stats.json'
        if not os.path.exists(plane_file):
            print(f"Could not open {plane_file}")
            return

        try:
            with open(plane_file, 'r') as openfile:
                cut_plane_stats = json.load(openfile)
        except IOError as e:
            print(f"I/O error({e.errno}): {e.strerror}: {plane_file}")
            return

        pos = cut_plane_stats["origin"]
        normal = cut_plane_stats["normal"]
        actor_cut = self.create_cylinder_actor(pos, normal, radius=30, height=0.5, rgba=[0, 0, 1], opacity=0.9)

        self.ren_3d_1.AddActor(actor_cut)
        self.ren_3d_2.AddActor(actor_cut)
        # if self.ren_volume:
        #     self.ren_volume.AddActor(actor_cut)



        #
        # # generate max cut actor
        # cl_file = f'{settings["centerline_dir"]}cl_max_cut.vtk'
        # if not os.path.exists(cl_file):
        #     print(f"No {cl_file}")
        #     cl_vtk = None
        # else:
        #     reader = vtk.vtkXMLPolyDataReader()
        #     reader.SetFileName(cl_file)
        #     reader.Update()
        #     cl_vtk = reader.GetOutput()
        #     n_points = cl_vtk.GetNumberOfPoints()
        #     if n_points < 2:
        #         print(f"No points in {cl_file}")
        #         cl_vtk = None
        #
        # if cl_vtk is not None:
        #     size = cl["size"]
        #     color = cl["color"]
        #     opacity = cl["opacity"]
        #     rgba = [0.0, 0.0, 0.0, 0.0]
        #     vtk.vtkNamedColors().GetColor(color, rgba)
        #
        #     vtk_tube_filter = vtk.vtkTubeFilter()
        #     vtk_tube_filter.SetInputData(cl_vtk)
        #     vtk_tube_filter.SetNumberOfSides(16)
        #     vtk_tube_filter.SetRadius(size)
        #     vtk_tube_filter.Update()
        #
        #     mapper = vtk.vtkPolyDataMapper()
        #     mapper.SetInputConnection(vtk_tube_filter.GetOutputPort())
        #     mapper.ScalarVisibilityOff()
        #
        #     actor = vtk.vtkActor()
        #     actor.SetMapper(mapper)
        #     actor.GetProperty().SetOpacity(opacity)
        #     actor.GetProperty().SetColor(rgba[0], rgba[1], rgba[2])
        #
        #     # if self.ren_volume:
        #     #     self.ren_volume.AddActor(actor)
        #
        #     self.actors.append(actor)

    def set_precomputed_slice(self, cl_folder):
        """
        Here the slices are precomputed as png files
        """
        plane_file = f'{cl_folder}combined_cuts.png'

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
        cut_stats = [{"name": "LVOT", "file": "lvot_segment_max_slice_info.json"},
                     {"name": "Sinus of Valsalve", "file": "sinus_of_valsalva_segment_max_slice_info.json"},
                     {"name": "Sinutubular junction", "file": "sinotubular_junction_segment_max_slice_info.json"},
                     {"name": "Ascending", "file": "ascending_segment_max_slice_info.json"},
                     {"name": "Aortic arch", "file": "aortic_arch_segment_max_slice_info.json"},
                     {"name": "Descending", "file": "descending_segment_max_slice_info.json"},
                     {"name": "Abdominal", "file": "abdominal_segment_max_slice_info.json"},
                     {"name": "Infrarenal", "file": "infrarenal_segment_max_slice_info.json"}]
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
        stats_file = f'{stats_folder}/aorta_statistics.json'
        ati_stats = read_json_file(stats_file)
        if not ati_stats:
            print(f"Could not read {stats_file}")
            return

        if "annulus_aortic_length" in ati_stats:
            cl_length = ati_stats["annulus_aortic_length"]
            self.message_text += f"\nTotal aortic length: {cl_length/10.0:.1f} cm\n"
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
        type_file = f'{stats_folder}/aorta_scan_type.json'
        stats_file = f'{stats_folder}/aorta_aneurysm_sac_stats.json'
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

            tot_vol = (aorta_lumen + calcification_volume)
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
        type_file = f'{stats_folder}aorta_scan_type.json'
        stats_file = f'{stats_folder}aorta_calcification_stats.json'
        # tot_stats_file = f'{settings["statistics_dir"]}/aorta_statistics.json'

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
            self.message_text += f"Calcified volume: {calcification_volume/1000.0:.1f} cm3\n"
            self.message_text += f"Lumen volume: {aorta_lumen_volume/1000.0:.1f} cm3\n"
            if aorta_lumen_volume > 0:
                ratio = calcification_volume / aorta_lumen_volume * 100
                self.message_text += f"Percent of total: {ratio:.2f}%\n"
        else:
            print(f"Can not set calcification statistics for scan type: {scan_type}")


    def set_slice_data(self, settings):
        # file_name = settings["input_file"]
        # plane_file = f'{settings["centerline_dir"]}cl_max_plane.txt'
        plane_file = f'{settings["centerline_dir"]}straight_cl_max_cut_stats.json'
        # cut_file = f'{settings["centerline_dir"]}cl_max_cut.vtk'
        # infrarenal_in = f'{settings["centerline_dir"]}/infrarenal_section.json'
        # vol_name = f'{settings["centerline_dir"]}straightened_volume.nii.gz'
        # surf_name = f'{settings["centerline_dir"]}straightened_labelmap_surface.vtk'
        cl_dir = settings["centerline_dir"]
        vol_name = f'{cl_dir}straightened_volume.nii.gz'
        surf_name = f'{cl_dir}straightened_labelmap_surface.vtk'
        hu_cl_file = f"{cl_dir}aorta_centerline_hu_vals.csv"

        if not os.path.exists(plane_file):
            print(f"Could not open {plane_file}")
            return

        if not os.path.exists(surf_name):
            print(f"Could not open {surf_name}")
            return

        if not os.path.exists(vol_name):
            print(f"Could not open {vol_name}")
            return

        # if not os.path.exists(cut_file):
        #     print(f"Could not open {cut_file}")
        #     return

        try:
            with open(plane_file, 'r') as openfile:
                cut_plan_stats = json.load(openfile)
        except IOError as e:
            print(f"I/O error({e.errno}): {e.strerror}: {plane_file}")
            return

        x, y, z = cut_plan_stats["origin"]
        nx, ny, nz = cut_plan_stats["normal"]
        cut_area = cut_plan_stats["area"]
        min_diam  = cut_plan_stats["min_diameter"]
        max_diam  = cut_plan_stats["max_diameter"]

        self.message_text += f"Max cross sectional area: {cut_area:.0f} mm2\n" \
                             f"Diameters: {min_diam:.0f} and {max_diam:.0f} mm\n"
        cl_dist = cut_plan_stats["cl_dist"]
        idx = cut_plan_stats["cl_idx"]

        vtk_img = pt.read_nifti_itk_to_vtk(vol_name)
        # print(f"VTK pixel type {vtk_img.GetScalarTypeAsString()}")

        # vtk_spc = vtk_img.GetSpacing()
        # vtk_dim = vtk_img.GetDimensions()
        # # print(f"VTK dim {vtk_dim}")
        # slice_n = int(cl_dist/vtk_spc[2])
        # print(f"Slice n : {slice_n}")
        slice_n = idx


        # # Slow and hacky - just to get HU stats
        # img_np, _, _ = pt.read_nifti_itk_to_numpy(vol_name)
        # min_hu = np.min(img_np)
        # max_hu = np.max(img_np)
        min_hu = 0
        max_hu = 500
        if os.path.exists(hu_cl_file):
            hu_vals = np.loadtxt(hu_cl_file, delimiter=",")
            hu_vals = hu_vals[:, 1]
            max_hu = np.percentile(hu_vals, 99)
            # min_hu = np.percentile(hu_vals, 1)

        print(f"Org HU min {min_hu} max {max_hu}")
        # Adjust to defaults
        min_hu = max(min_hu, 0)
        max_hu = min(max_hu, 1000)
        # https://discourse.vtk.org/t/relation-between-vtkimagedata-wl-and-vtkimageactor-wl/2934/20

        # print(f"HU min: {min_hu} max {max_hu}")
        img_window = max_hu - min_hu
        img_level = (max_hu + min_hu) / 2.0
        # img_window = 440
        # img_level = 92
        print(f"Computed window {img_window} level {img_level}")
        self.message_text += f"W/L: {img_window:.0f}/{img_level:.0f}\n"

        image_viewer = vtk.vtkImageViewer2()
        image_viewer.SetInputData(vtk_img)
        image_viewer.SetColorWindow(img_window)
        image_viewer.SetColorLevel(img_level)

        image_viewer.SetRenderWindow(self.ren_win)
        image_viewer.SetRenderer(self.ren_slice)

        # image_viewer.SetSliceOrientationToYZ()
        image_viewer.SetSlice(slice_n)
        # image_viewer.SetSliceOrientationToXZ()
        # image_viewer.SetWindowLevel(win_to_col)
        # bounds = image_viewer.GetImageActor().GetBounds()
        # # print(bounds)
        #
        # origin = ((bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2)
        # print(origin)
        # normal = [0, 0, 1]  # Normal mode
        # normal = [0, 1, 0] # xz
        # normal = [1, 0, 0]  # yz

        plane = vtk.vtkPlane()
        plane.SetOrigin(x, y, z)
        plane.SetNormal(nx, ny, nz)

        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(surf_name)
        reader.Update()

        cutter = vtk.vtkCutter()
        cutter.SetInputConnection(reader.GetOutputPort())
        cutter.SetCutFunction(plane)
        cutter.GenerateCutScalarsOff()
        cutter.Update()

        stripper = vtk.vtkStripper()
        stripper.SetInputConnection(cutter.GetOutputPort())
        stripper.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(stripper.GetOutputPort())
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.GetProperty().SetRenderLinesAsTubes(1)
        actor.GetProperty().SetLineWidth(2)
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(1)
        actor.GetProperty().SetColor(0.0, 0.0, 1.0)

        self.ren_slice.AddActor(actor)
        self.ren_straight_1.AddActor(actor)
        self.ren_straight_2.AddActor(actor)

        diameter_actor_min = self.line_actor_from_points(cut_plan_stats["min_d_p1"], cut_plan_stats["min_d_p2"])
        diameter_actor_max = self.line_actor_from_points(cut_plan_stats["max_d_p1"], cut_plan_stats["max_d_p2"])

        self.ren_slice.AddActor(diameter_actor_max)
        self.ren_slice.AddActor(diameter_actor_min)
        self.ren_slice.ResetCameraScreenSpace()

    def set_precomputed_straight_longitudinal_slices(self, cl_folder):
        """
        Here the longitudinal are precomputed as png files in the vmtk processing script
        """
        # plane_file = f'{settings["centerline_dir"]}infrarenal_segment_max_slice_rgb_crop.png'
        plane_file_1 = f'{cl_folder}straight_volume_mid_cut.png'
        plane_file_2 = f'{cl_folder}straight_volume_mid_cut_2.png'

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


    def set_straightened_data(self, settings):
        """
        Enable viewing of straightened resliced volume
        """
        cl_dir = settings["centerline_dir"]
        vol_name = f'{cl_dir}straightened_volume.nii.gz'
        surf_name = f'{cl_dir}straightened_labelmap_surface.vtk'
        hu_cl_file = f"{cl_dir}aorta_centerline_hu_vals.csv"

        # plane_file = f'{settings["centerline_dir"]}cl_max_plane.txt'
        # plane_file = f'{settings["centerline_dir"]}cl_max_cut_stats.json'
        # cut_file = f'{settings["centerline_dir"]}straightened_volume.nii.gz'
        # infrarenal_in = f'{settings["centerline_dir"]}/infrarenal_section.json'

        if not os.path.exists(surf_name):
            print(f"Could not open {surf_name}")
            return

        if not os.path.exists(vol_name):
            print(f"Could not open {vol_name}")
            return

        vtk_img = pt.read_nifti_itk_to_vtk(vol_name)
        # print(f"VTK pixel type {vtk_img.GetScalarTypeAsString()}")

        vtk_dim = vtk_img.GetDimensions()
        # print(f"VTK dim {vtk_dim}")
        slice_n = int(vtk_dim[0] / 2)

        min_hu = 0
        max_hu = 500
        if os.path.exists(hu_cl_file):
            hu_vals = np.loadtxt(hu_cl_file, delimiter=",")
            hu_vals = hu_vals[:, 1]
            max_hu = np.percentile(hu_vals, 99)
            # min_hu = np.percentile(hu_vals, 1)

        # # Slow and hacky - just to get HU stats
        # img_np, _, _ = pt.read_nifti_itk_to_numpy(vol_name)
        # min_hu = np.min(img_np)
        # max_hu = np.max(img_np)
        print(f"Org HU min {min_hu} max {max_hu}")
        # Adjust to defaults
        min_hu = max(min_hu, 0)
        max_hu = min(max_hu, 1000)
        # https://discourse.vtk.org/t/relation-between-vtkimagedata-wl-and-vtkimageactor-wl/2934/20

        # print(f"HU min: {min_hu} max {max_hu}")
        img_window = max_hu - min_hu
        img_level = (max_hu + min_hu) / 2.0
        # img_window = 440
        # img_level = 92
        # print(f"Computed window {img_window} level {img_level}")

        image_viewer = vtk.vtkImageViewer2()
        image_viewer.SetInputData(vtk_img)
        image_viewer.SetColorWindow(img_window)
        image_viewer.SetColorLevel(img_level)

        image_viewer.SetRenderWindow(self.ren_win)
        image_viewer.SetRenderer(self.ren_straight_1)

        image_viewer.SetSliceOrientationToYZ()
        image_viewer.SetSlice(slice_n)
        # image_viewer.SetSliceOrientationToXZ()
        # image_viewer.SetWindowLevel(win_to_col)
        bounds = image_viewer.GetImageActor().GetBounds()
        # print(bounds)

        origin = ((bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2)
        # print(origin)
        # normal = [0, 0, 1]  # Normal mode
        # normal = [0, 1, 0] # xz
        normal = [1, 0, 0]  # yz

        plane = vtk.vtkPlane()
        plane.SetOrigin(origin)
        plane.SetNormal(normal)

        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(surf_name)
        reader.Update()

        cutter = vtk.vtkCutter()
        cutter.SetInputConnection(reader.GetOutputPort())
        cutter.SetCutFunction(plane)
        cutter.GenerateCutScalarsOff()
        cutter.Update()

        stripper = vtk.vtkStripper()
        stripper.SetInputConnection(cutter.GetOutputPort())
        stripper.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(stripper.GetOutputPort())
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.GetProperty().SetRenderLinesAsTubes(1)
        actor.GetProperty().SetLineWidth(2)
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(1)
        actor.GetProperty().SetColor(1.0, 0.0, 0.0)

        self.ren_straight_1.AddActor(actor)
        self.ren_straight_1.ResetCameraScreenSpace()

        # Second straight view
        image_viewer_2 = vtk.vtkImageViewer2()
        image_viewer_2.SetInputData(vtk_img)
        image_viewer_2.SetColorWindow(img_window)
        image_viewer_2.SetColorLevel(img_level)

        image_viewer_2.SetRenderWindow(self.ren_win)
        image_viewer_2.SetRenderer(self.ren_straight_2)

        image_viewer_2.SetSliceOrientationToXZ()
        image_viewer_2.SetSlice(slice_n)
        # image_viewer.SetSliceOrientationToXZ()
        # image_viewer.SetWindowLevel(win_to_col)
        bounds = image_viewer_2.GetImageActor().GetBounds()
        # print(bounds)

        origin = ((bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2)
        # print(origin)
        # normal = [0, 0, 1]  # Normal mode
        normal = [0, 1, 0] # xz
        # normal = [1, 0, 0]  # yz

        plane_2 = vtk.vtkPlane()
        plane_2.SetOrigin(origin)
        plane_2.SetNormal(normal)

        # reader = vtk.vtkXMLPolyDataReader()
        # reader.SetFileName(surf_name)
        # reader.Update()

        cutter_2 = vtk.vtkCutter()
        cutter_2.SetInputConnection(reader.GetOutputPort())
        cutter_2.SetCutFunction(plane_2)
        cutter_2.GenerateCutScalarsOff()
        cutter_2.Update()

        stripper_2 = vtk.vtkStripper()
        stripper_2.SetInputConnection(cutter_2.GetOutputPort())
        stripper_2.Update()

        mapper_2 = vtk.vtkPolyDataMapper()
        mapper_2.SetInputConnection(stripper_2.GetOutputPort())
        mapper_2.ScalarVisibilityOff()

        actor_2 = vtk.vtkActor()
        actor_2.GetProperty().SetRenderLinesAsTubes(1)
        actor_2.GetProperty().SetLineWidth(2)
        actor_2.SetMapper(mapper_2)
        actor_2.GetProperty().SetOpacity(1)
        actor_2.GetProperty().SetColor(1.0, 0.0, 0.0)

        self.ren_straight_2.AddActor(actor_2)
        self.ren_straight_2.ResetCameraScreenSpace()

