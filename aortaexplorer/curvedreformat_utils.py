import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import math
import numpy as np
import vtk
import SimpleITK as sitk
from vtk import vtkMatrix4x4
from vtk import vtkMatrix3x3
import vtk.util.numpy_support


def points_to_poly(points):
    numberOfPoints = points.GetNumberOfPoints()
    polyLine = vtk.vtkPolyLine()
    polyLine.GetPointIds().SetNumberOfIds(numberOfPoints)

    for i in range(0,numberOfPoints):
        polyLine.GetPointIds().SetId(i,i)

    cells = vtk.vtkCellArray()
    cells.InsertNextCell(polyLine)

    polyData = vtk.vtkPolyData()
    polyData.SetPoints(points)
    polyData.SetLines(cells)


    return polyData

def array_from_vtk_matrix(vmatrix):
    '''
    From Slicer discord
    Returns vtkMatrix 4x4 or vtkMatrix3x3 elements as numpy array.
    The returned array is a copy, and modifications to the output should not impact to the input arrays
    '''
    if isinstance(vmatrix, vtkMatrix4x4):
        matrixSize = 4
    elif isinstance(vmatrix, vtkMatrix3x3):
        matrixSize = 3
    else:
        raise RuntimeError("Input must be vtk.vtkMatrix3x3 or vtk.vtkMatrix4x4")
    narray = np.eye(matrixSize)
    vmatrix.DeepCopy(narray.ravel(),vmatrix)
    return narray

def vtk_matrix_from_array(narray):
    '''
    From Slicer.util.py:
    Create VTK matrix from a 3x3 or 4x4 numpy array.

    :param narray: input numpy array
    :raises RuntimeError: in case of failure

    The returned matrix is just a copy and so any modification in the array will not affect the output matrix.
    To set numpy array from VTK matrix, use :py:meth:`arrayFromVTKMatrix`.
    '''
    narrayshape = narray.shape
    if narrayshape == (4,4):
        vmatrix = vtkMatrix4x4()
        update_vtk_matrix_from_array(vmatrix, narray)
        return vmatrix
    elif narrayshape == (3,3):
        vmatrix = vtkMatrix3x3()
        update_vtk_matrix_from_array(vmatrix, narray)
        return vmatrix
    else:
        raise RuntimeError("Unsupported numpy array shape: "+str(narrayshape)+" expected (4,4)")

def update_vtk_matrix_from_array(vmatrix, narray):
    """
    From Slicer.util.py:
    Update VTK matrix values from a numpy array.

    :param vmatrix: VTK matrix (vtkMatrix4x4 or vtkMatrix3x3) that will be update
    :param narray: input numpy array
    :raises RuntimeError: in case of failure

    To set numpy array from VTK matrix, use :py:meth:`arrayFromVTKMatrix`.
    """
    if isinstance(vmatrix, vtkMatrix4x4):
        matrixSize = 4
    elif isinstance(vmatrix, vtkMatrix3x3):
        matrixSize = 3
    else:
        raise RuntimeError("Output vmatrix must be vtk.vtkMatrix3x3 or vtk.vtkMatrix4x4")
    if narray.shape != (matrixSize, matrixSize):
        raise RuntimeError("Input narray size must match output vmatrix size ({0}x{0})".format(matrixSize))
    vmatrix.DeepCopy(narray.ravel())

def array_from_grid_transform(displacementGrid):
    nshape = tuple(reversed(displacementGrid.GetDimensions()))
    nshape = nshape + (3,)
    import vtk.util.numpy_support
    narray = vtk.util.numpy_support.vtk_to_numpy(displacementGrid.GetPointData().GetScalars()).reshape(nshape)
    return narray


class CurvedPlanarReformat:
    '''
    Compute straightened volume for visualization of curved vessels.
    Based on implementation: https://github.com/PerkLab/SlicerSandbox/blob/master/CurvedPlanarReformat/CurvedPlanarReformat.py
    '''
    def __init__(self, sliceResolution = [0.5, 0.5, 1.0], sliceSizeMm = [40.0, 40.0], converCoordinatSystem = False):
        self.transformSpacingFactor = 5.0
        self.MinimumDistance = 1e-3 # Minimum distance for comuting initial tangent direction
        self.PreferredInitialNormalVector = np.array([1.0, 0.0, 0.0 ])
        self.PreferredInitialBinormalVector = np.zeros(3)
        self.Tolerance = 1e-6 # Tolerance value used for checking that a value is non-zero.

        self.sliceResolution = sliceResolution
        self.sliceSize = sliceSizeMm
        self.converCoordinatSystem = converCoordinatSystem # Flip between RAS/LPS compared to input

        # Save outputs in
        # self.transformOutputDir = transformOutputDir
        # self.volumeOutputDir = volumeOutputDir
        self.resampledCurveNode = None


# ==============================================================================
# Helper-methods
# ==============================================================================

    def resample_points(self, originalPoints, sampledPoints, samplingDistance):
        '''
        Resamples given original points into sampledPoints using given samplingDistance

        originalPoints (vtkPoints): Points to be resampled
        sampledPoints (vtkPoints): Obj to store points in after resampling
        samplingDistance (double): Distance between points after sampling

        Given points must be an open curve (closed curve not implemented)
        Simplified version of resampling-function found in https://github.com/Slicer/Slicer/blob/master/Modules/Loadable/Markups/MRML/vtkMRMLMarkupsCurveNode.cxx

        '''

        if (originalPoints.GetNumberOfPoints() < 2):
            sampledPoints.DeepCopy(originalPoints)
            return True

        distanceFromLastSampledPoint = 0.0
        remainingSegmentLength = 0.0
        previousCurvePoint = np.array([0.0, 0.0, 0.0])
        originalPoints.GetPoint(0, previousCurvePoint)
        sampledPoints.Reset()
        sampledPoints.InsertNextPoint(previousCurvePoint)

        numberOfOriginalPoints = originalPoints.GetNumberOfPoints()

        currentCurvePoint = None

        for originalPointIndex in range(numberOfOriginalPoints):
            currentCurvePoint = originalPoints.GetPoint(originalPointIndex)

            segmentLength = math.sqrt(vtk.vtkMath.Distance2BetweenPoints(currentCurvePoint, previousCurvePoint))
            if segmentLength <= 0.0:
                continue
            remainingSegmentLength = distanceFromLastSampledPoint + segmentLength
            if (remainingSegmentLength >= samplingDistance):
                segmentDirectionVector = np.array([
                    (currentCurvePoint[0] - previousCurvePoint[0]) / segmentLength,
                    (currentCurvePoint[1] - previousCurvePoint[1]) / segmentLength,
                    (currentCurvePoint[2] - previousCurvePoint[2]) / segmentLength])

                #distance of new sampled point from previous curve point
                distanceFromLastInterpolatedPoint = samplingDistance - distanceFromLastSampledPoint

                while (remainingSegmentLength >= samplingDistance):
                    newSampledPoint = np.array([
                        previousCurvePoint[0] + segmentDirectionVector[0] * distanceFromLastInterpolatedPoint,
                        previousCurvePoint[1] + segmentDirectionVector[1] * distanceFromLastInterpolatedPoint,
                        previousCurvePoint[2] + segmentDirectionVector[2] * distanceFromLastInterpolatedPoint])

                    sampledPoints.InsertNextPoint(newSampledPoint)
                    distanceFromLastSampledPoint = 0
                    distanceFromLastInterpolatedPoint += samplingDistance

                    remainingSegmentLength -= samplingDistance
                    
                distanceFromLastSampledPoint = remainingSegmentLength

            else:
                distanceFromLastSampledPoint += segmentLength

            previousCurvePoint[0] = currentCurvePoint[0]
            previousCurvePoint[1] = currentCurvePoint[1]
            previousCurvePoint[2] = currentCurvePoint[2]

        # Curve is assumed to be open (original handled closed curves as well)

        # Ideally, remainingSegmentLength would be equal to 0.
        # if (remainingSegmentLength > samplingDistance * 0.5):
        # # last segment would be much longer than the sampling distance, so add an extra point
        #     secondLastPointPosition = np.array([0.0, 0.0, 0.0])
        #     secondLastPointOriginalPointIndex = 0
        #     print('I should have implemented this part too...')

        # else last segment is only slightly longer than the sampling distance
        # so we just adjust the position of last point
        sampledPoints.SetPoint(sampledPoints.GetNumberOfPoints() - 1,
                               originalPoints.GetPoint(originalPoints.GetNumberOfPoints() - 1))

        return True

# ------------------------------------------------------------------------------
# Helper functions for coordinate system calculations
# Note that this framework is modified to only work within one coordinate system,
# making transforms between systems obsolete.

    def rotate_vector(self, in_vector, out_vector, axis, angle):
        '''
        Rotates a given vector a given angle.
        Helper-function for ComputeAxisDirections

        in_vector (np array (dtype = float64)): 3D vector to be rotated
        out_vector (np array (dtype = float64)): 3D vector to save result in
        axis (np array (dtype = float64)): Rotation axis
        angle (double): angle in_vector is rotated

        Modified version of the one found in vtkParallelTransportFrame in vtkAddonmaster
        '''
        UdotN = vtk.vtkMath.Dot(in_vector, axis)
        NcrossU = np.zeros(3)
        vtk.vtkMath.Cross(axis, in_vector, NcrossU)


        for comp in range(0,3):
            out_vector[comp] = math.cos(angle) * in_vector[comp] + (1 - math.cos(angle)) * UdotN * axis[comp] + math.sin(angle) * NcrossU[comp]

        return True

    def compute_axis_directions(self, curve, idx, tangents, normals, binormals):
        '''
        Helper-function for RequestData()

        curve (vtkPolyData): Curve to compute a axis directions on
        idx (int): Point index at which axis directions are computed

        tangents (vtkDoubleArray): Obj to store computed tangents
        normals (vtkDoubleArray): Obj to store computed normals
        binormals (vtkDoubleArray): Obj to store computed binormals

        Modified version of the one found in vtkParallelTransportFrame in vtkAddonmaster
        '''

        polyLine = curve.GetCell(idx)

        if (not polyLine):
            print("Cannot compute axis directions, something wrong with curve")
            return False

        numberOfPointsInCell = polyLine.GetNumberOfPoints()
        if (numberOfPointsInCell < 2):
            print("Only 2 points in cell")
            return False

        tangent0 = np.zeros(3)
        pointId0 = polyLine.GetPointId(0)
        pointPosition0 = np.zeros(3)
        curve.GetPoint(pointId0, pointPosition0)

        # Find tangent by direction vector by moving a minimal distance from the initial point
        for pointIndex in range(1,numberOfPointsInCell):
            pointId1 = polyLine.GetPointId(pointIndex)
            pointPosition1 = np.zeros(3)
            curve.GetPoint(pointId1, pointPosition1)
            tangent0[0] = pointPosition1[0] - pointPosition0[0]
            tangent0[1] = pointPosition1[1] - pointPosition0[1]
            tangent0[2] = pointPosition1[2] - pointPosition0[2]

            if (vtk.vtkMath.Norm(tangent0) >= self.MinimumDistance):
                break

        vtk.vtkMath.Normalize(tangent0)

        # Compute initial normal and binormal directions from the initial tangent and preferred normal/binormal directions.
        normal0 = np.zeros(3)
        binormal0 = np.zeros(3)
        vtk.vtkMath.Cross(tangent0, self.PreferredInitialNormalVector, binormal0)
        if (vtk.vtkMath.Norm(binormal0) > self.Tolerance):
            vtk.vtkMath.Normalize(binormal0)
            vtk.vtkMath.Cross(binormal0,tangent0,normal0)
        else:
            vtk.vtkMath.Cross(self.PreferredInitialBinormalVector, tangent0, normal0)
            vtk.vtkMath.Normalize(normal0)
            vtk.vtkMath.Cross(tangent0, normal0, binormal0)

        tangents.SetTuple(pointId0, tangent0)
        normals.SetTuple(pointId0, normal0)
        binormals.SetTuple(pointId0, binormal0)

        pointId2 = -1
        tangent1 = np.array([tangent0[0],tangent0[1], tangent0[2] ])
        normal1 = np.array([normal0[0],normal0[1], normal0[2] ])
        binormal1 = np.array([binormal0[0],binormal0[1], binormal0[2] ])

        for i in range(1,numberOfPointsInCell-1):
            pointId1 = polyLine.GetPointId(i)
            pointId2 = polyLine.GetPointId(i+1)
            pointPosition1 = np.zeros(3)
            pointPosition2 = np.zeros(3)
            curve.GetPoint(pointId1, pointPosition1)
            curve.GetPoint(pointId2, pointPosition2)

            tangent1[0] = pointPosition2[0] - pointPosition1[0]
            tangent1[1] = pointPosition2[1] - pointPosition1[1]
            tangent1[2] = pointPosition2[2] - pointPosition1[2]

            vtk.vtkMath.Normalize(tangent0)
            vtk.vtkMath.Normalize(tangent1)

            dot = vtk.vtkMath.Dot(tangent0, tangent1)
            theta = 0.0
            if ((1 - dot) < self.Tolerance):
                theta = 0.0
            else:
                theta = math.acos(dot)

            rotationAxis = np.zeros(3)
            vtk.vtkMath.Cross(tangent0, tangent1, rotationAxis)

            self.rotate_vector(normal0, normal1, rotationAxis, theta)
            dot = vtk.vtkMath.Dot(tangent1, normal1)
            normal1[0] -= dot * tangent1[0]
            normal1[1] -= dot * tangent1[1]
            normal1[2] -= dot * tangent1[2]

            vtk.vtkMath.Normalize(normal1)
            vtk.vtkMath.Cross(tangent1, normal1, binormal1)

            tangents.SetTuple(pointId1, tangent1)
            normals.SetTuple(pointId1, normal1)
            binormals.SetTuple(pointId1, binormal1)

            # Save current data for next iteration
            tangent0[0] = tangent1[0]
            tangent0[1] = tangent1[1]
            tangent0[2] = tangent1[2]
            normal0[0] = normal1[0]
            normal0[1] = normal1[1]
            normal0[2] = normal1[2]

        if (pointId2 >= 0):
            tangents.SetTuple(pointId2, tangent1)
            normals.SetTuple(pointId2, normal1)
            binormals.SetTuple(pointId2, binormal1)


        #narray = vtk.util.numpy_support.vtk_to_numpy(normals)
        #print("normal: ", narray)
        #print("normals: ", normals)

        return True

    def request_data(self, curve, output_curve):
        '''
        Calculates tangents, normals and binormals for all points in given curve

        curve (vtkPolyData): Curve to extend with additional data
        output_curve (vtkPolyData): Obj to save resulting curve in

        Modified version of the one found in vtkParallelTransportFrame in vtkAddonmaster.
        '''
        output_curve.DeepCopy(curve)

        tangentsArray = vtk.vtkDoubleArray()
        tangentsArray.SetNumberOfComponents(3)
        tangentsArray.SetName("Tangents")

        normalsArray = vtk.vtkDoubleArray()
        normalsArray.SetNumberOfComponents(3)
        normalsArray.SetName("Normals")

        binormalsArray = vtk.vtkDoubleArray()
        binormalsArray.SetNumberOfComponents(3)
        binormalsArray.SetName("Binormals")

        numberOfPoints = curve.GetNumberOfPoints()
        tangentsArray.SetNumberOfTuples(numberOfPoints)
        tangentsArray.Fill(0.0)
        normalsArray.SetNumberOfTuples(numberOfPoints)
        normalsArray.Fill(0.0)
        binormalsArray.SetNumberOfTuples(numberOfPoints)
        binormalsArray.Fill(0.0)

        numberOfCells = curve.GetNumberOfCells()
        if (numberOfCells == 0):
            print("Number of cells", numberOfCells)

        for cellIndex in range(0,numberOfCells):
            self.compute_axis_directions(curve, cellIndex, tangentsArray, normalsArray, binormalsArray)

        output_curve.GetPointData().AddArray(tangentsArray)
        output_curve.GetPointData().AddArray(normalsArray)
        output_curve.GetPointData().AddArray(binormalsArray)
        output_curve.GetPointData().Modified()

        return 1


    def get_curve_point_to_world_transform_at_point_index(self, curve, curvePointIndex, curvePointToWorld):
        '''
        From a given curve, calculate normal, binormal, tangent and position of curvePointIndex and save in curvePointToWorld

        curve (vtkpolydata): Resampled curve from which the coordinate system is calculated (NOT INCLUDED IN ORIGINAL)
        curvePointIndex (vtkIdType): idx of point on curve, for which to get the transform
        curvePointToWorld (4x4 vtkMatrix): Obj to save result in

        Modified version from vtkMRMLMarkupsCurveNode in Slicer-master
        Note that "WorldTransform" is obsolete due to simplification.
        '''

        # Skips pointer check since no pointers in python3
        curvePoly = vtk.vtkPolyData() # vtkPolyData storing of curve coordinate system world - in this case we only work with one coordinate system, thus it is just curve wrapped in a vtkPolyData object
        curvePoly.DeepCopy(curve)
        # Note that curvePoly is derived from CurveCoordinateSystemGeneratorWorld, which means that it might already contain tangens etc from having called RequestData
        # !!! Therefore - call RequestData!!

        self.request_data(curve, curvePoly) # TO DO: Only do if not already done?
        # Check that not empty
        if (not curvePoly):
            print("GetCurvePointToWorldTransformAtPointIndex() expects non-empty curve")
            return False

        n = curvePoly.GetNumberOfPoints()

        # Sanity check: the given idx has to be between 0 and the number of points on curve
        if (curvePointIndex < 0 or curvePointIndex >= n):
            print("Curve point idx wrong: Implement vtkError to handle this!")
            #vtkErrorMacro("vtkMRMLMarkupsCurveNode::GetCurvePointToWorldTransformAtPointIndex failed: Invalid curvePointIndex " << curvePointIndex << " (number of curve points: " << n << ")")
            return False

        pointData = curvePoly.GetPointData()
        if (not pointData):
            print('Something wrong with pointData')
            return False

        tangents = pointData.GetAbstractArray("Tangents")
        normals = pointData.GetAbstractArray("Normals")
        binormals = pointData.GetAbstractArray("Binormals")

        if ((not tangents) or (not normals) or (not binormals)):
            print("GetCurvePointToWorldTransformAtPointIndex() has something wrong with tangets, normals or binormals")
            return False

        normal = normals.GetTuple3(curvePointIndex)
        #print("Normal: ", normal)
        binormal = binormals.GetTuple3(curvePointIndex)
        tangent = tangents.GetTuple3(curvePointIndex)
        position = curvePoly.GetPoint(curvePointIndex)

        # Put result in given container
        for row in range(0,3):
            curvePointToWorld.SetElement(row, 0, normal[row])
            curvePointToWorld.SetElement(row, 1, binormal[row])
            curvePointToWorld.SetElement(row, 2, tangent[row])
            curvePointToWorld.SetElement(row, 3, position[row])

        return True

# ------------------------------------------------------------------------------
# Helper-functions for curve properties (for transformation grid)

    def get_curve_length(self, curvePoints, startCurvePointIndex = 0, numberOfCurvePoints = None):
        '''
        Measures length to end of given curve from a given point index.
        Assumes open curve!

        curvepoints (vtkPoints*) curve points to get length off
        startCurvePointIndex (vtkIdType): Point to start measuring at (optional, measures entire curve if no parameter given)
        numberOfCurvePoints (vtkIdType): Number of curvepoints to measure (optional, measures to end of curve if none given)
        '''
        if ((not curvePoints) or curvePoints.GetNumberOfPoints() < 2):
            return 0.0
        if (startCurvePointIndex < 0):
            print("Implement error handling in GetCurveLength please!")
            startCurvePointIndex = 0
        # Get last curve point idx
        lastCurvePointIndex = curvePoints.GetNumberOfPoints() - 1

        # If number of points to be measured are not the end of curve, update index of last point
        if numberOfCurvePoints >= 0 and (startCurvePointIndex + numberOfCurvePoints) < lastCurvePointIndex:
            lastCurvePointIndex = startCurvePointIndex + numberOfCurvePoints - 1

        # Init
        length = 0.0
        previousPoint = np.array([0.0, 0.0, 0.0])
        nextPoint = np.array([0.0, 0.0, 0.0])

        curvePoints.GetPoint(startCurvePointIndex, previousPoint) # Save start point in "previous point"
        for curvePointIndex in range(startCurvePointIndex + 1,lastCurvePointIndex + 1):
            curvePoints.GetPoint(curvePointIndex,nextPoint)
            length += math.sqrt(vtk.vtkMath.Distance2BetweenPoints(previousPoint, nextPoint))
            previousPoint[0] = nextPoint[0]
            previousPoint[1] = nextPoint[1]
            previousPoint[2] = nextPoint[2]

        # Add length of closing segment omitted, since open curve is assumed
        return length


    def get_curve_center(self, points):
        '''
        points (vtkPolyData): Contains points to find center
        '''
        # Check input
        if ((not points) or  (points.GetNumberOfPoints() < 1)):
            print("GetCurveCenter() inputs wrong")
            return False

        numberOfPoints = points.GetNumberOfPoints()

        pointCoords = np.zeros((3,numberOfPoints)) # To be filled with all the points to fit plane to
        point = [0,0,0] # Preallocation

        for pointIndex in range(0,numberOfPoints):
            points.GetPoint(pointIndex,point)
            pointCoords[0,pointIndex] = point[0]
            pointCoords[1,pointIndex] = point[1]
            pointCoords[2,pointIndex] = point[2]

        # Define centroid
        centroid = np.array([pointCoords[0,:].mean(),pointCoords[1,:].mean(),pointCoords[2,:].mean()])

        return centroid


# ==============================================================================
# Helper functions for actual reformatting

# ==============================================================================
# Main methods handling CPR logic

    #def computeStraighteningTransform(self, transformToStraightenedNode, curve, sliceSizeMm = [40.0, 40,0], outputSpacingMm = 1.0):
    def compute_straightening_transform(self, curve, sliceSizeMm = [40.0, 40, 0], outputSpacingMm = 1.0,
                                        convertCoordinatSystem = False):
        '''
        Computes transform used to map each slice of volume during straightening/reformatting.
        transformToStraightenedNode: Container to store result (deprecated)
        curve (vtkPolyData): Curve used for reformatting
        sliceSizeMm: Gridspacing of transform in mm (Example FOV = [40.0, 40.0])
        outputSpacingMm (double): Spacing in mm, default 1 mm
        '''
        ### Create a temporary resampled curve
        resamplingCurveSpacing = outputSpacingMm * self.transformSpacingFactor
        #originalCurvePoints = curveNode.GetCurvePointsWorld()
        points = curve.GetPoints() # get points from curve in vtk format
        originalCurvePoints = vtk.vtkPoints()
        originalCurvePoints.DeepCopy(points) # make copy of points because lack of pointers confuses me

        sampledPoints = vtk.vtkPoints()

        # Resample points - todo: insert try-catch?
        self.resample_points(originalCurvePoints, sampledPoints, resamplingCurveSpacing)
        self.resampledCurveNode = points_to_poly(sampledPoints)

        numberOfSlices = self.resampledCurveNode.GetNumberOfPoints()

         # Z axis (from first curve point to last, this will be the straightened curve long axis)
        curveStartPoint = np.zeros(3)
        curveEndPoint = np.zeros(3)
        self.resampledCurveNode.GetPoint(0,curveStartPoint)


        self.resampledCurveNode.GetPoint(numberOfSlices-1,curveEndPoint)
        transformGridAxisZ = (curveEndPoint-curveStartPoint)/np.linalg.norm(curveEndPoint-curveStartPoint)


        # X axis = average X axis of curve, to minimize torsion (and so have a simple displacement field, which can be robustly inverted)
        sumCurveAxisX_RAS = np.zeros(3)
        for gridK in range(numberOfSlices): # gridK is slice idx
            curvePointToWorld = vtk.vtkMatrix4x4()
            self.get_curve_point_to_world_transform_at_point_index(self.resampledCurveNode, gridK, curvePointToWorld) # On assumption that the two index type are equivivalent
            curvePointToWorldArray = array_from_vtk_matrix(curvePointToWorld)# Convert to numpy
            curveAxisX_RAS = curvePointToWorldArray[0:3, 0]
            sumCurveAxisX_RAS += curveAxisX_RAS
        meanCurveAxisX_RAS = sumCurveAxisX_RAS/np.linalg.norm(sumCurveAxisX_RAS)
        transformGridAxisX = meanCurveAxisX_RAS

        # Y axis
        transformGridAxisY = np.cross(transformGridAxisZ, transformGridAxisX)
        transformGridAxisY = transformGridAxisY/np.linalg.norm(transformGridAxisY)

        # Make sure that X axis is orthogonal to Y and Z
        transformGridAxisX = np.cross(transformGridAxisY, transformGridAxisZ)
        transformGridAxisX = transformGridAxisX/np.linalg.norm(transformGridAxisX)

        # Origin (makes the grid centered at the curve)
        curveLength = self.get_curve_length(self.resampledCurveNode, numberOfCurvePoints = numberOfSlices)
        #curveNodePlane = vtk.vtkPlane()
        #self.GetBestFitPlane(resampledCurveNode, curveNodePlane)

        curveCentroid = self.get_curve_center(self.resampledCurveNode)
        #print("curveCentroid: ", curveCentroid)
        #print("Plane center: ", np.array(curveNodePlane.GetOrigin()))

        #transformGridOrigin = np.array(curveNodePlane.GetOrigin())
        transformGridOrigin = curveCentroid
        transformGridOrigin -= transformGridAxisX * sliceSizeMm[0]/2.0
        transformGridOrigin -= transformGridAxisY * sliceSizeMm[1]/2.0
        transformGridOrigin -= transformGridAxisZ * curveLength/2.0

        # Create grid transform
        # Each corner of each slice is mapped from the original volume's reformatted slice
        # to the straightened volume slice.
        # The grid transform contains one vector at the corner of each slice.
        # The transform is in the same space and orientation as the straightened volume.
        gridDimensions = [2, 2, numberOfSlices]
        gridSpacing = [sliceSizeMm[0], sliceSizeMm[1], resamplingCurveSpacing]
        gridDirectionMatrixArray = np.eye(4)
        gridDirectionMatrixArray[0:3, 0] = transformGridAxisX
        gridDirectionMatrixArray[0:3, 1] = transformGridAxisY
        gridDirectionMatrixArray[0:3, 2] = transformGridAxisZ
        # gridDirectionMatrix = vtk_matrix_from_array(gridDirectionMatrixArray)


        gridImage = vtk.vtkImageData()
        gridImage.SetOrigin(transformGridOrigin)
        gridImage.SetDimensions(gridDimensions)
        gridImage.SetSpacing(gridSpacing)
        gridImage.AllocateScalars(vtk.VTK_DOUBLE, 3)

        transformVTK = vtk.vtkGridTransform()
        transformVTK.SetDisplacementGridData(gridImage) #vtkTransform does not have orientation, but Slicer uses vtkAddons vtkOrientedGridTransform

        # orientedTransformVTK = (transformVTK, gridDirectionMatrix)

        # Compute displacements
        transformDisplacements_RAS = array_from_grid_transform(gridImage)

        for gridK in range(gridDimensions[2]):
            #print(gridK)
            curvePointToWorld = vtk.vtkMatrix4x4()
            # resampledCurveNode.GetCurvePointToWorldTransformAtPointIndex(resampledCurveNode.GetCurvePointIndexFromControlPointIndex(gridK), curvePointToWorld)
            # resampledCurveNode.GetCurvePointToWorldTransformAtPointIndex(resampledCurveNode.GetCurvePointIndexFromControlPointIndex(gridK), curvePointToWorld)
            self.get_curve_point_to_world_transform_at_point_index(self.resampledCurveNode, gridK, curvePointToWorld) # On assumption that the two index type are equivivalent
            curvePointToWorldArray = array_from_vtk_matrix(curvePointToWorld)
            curveAxisX_RAS = curvePointToWorldArray[0:3, 0]
            curveAxisY_RAS = curvePointToWorldArray[0:3, 1]
            curvePoint_RAS = curvePointToWorldArray[0:3, 3]

            for gridJ in range(gridDimensions[1]):
                for gridI in range(gridDimensions[0]):
                    straightenedVolume_RAS = (transformGridOrigin
                        + gridI*gridSpacing[0]*transformGridAxisX
                        + gridJ*gridSpacing[1]*transformGridAxisY
                        + gridK*gridSpacing[2]*transformGridAxisZ)
                    inputVolume_RAS = (curvePoint_RAS
                        + (gridI-0.5)*sliceSizeMm[0]*curveAxisX_RAS
                        + (gridJ-0.5)*sliceSizeMm[1]*curveAxisY_RAS)
                    transformDisplacements_RAS[gridK][gridJ][gridI] = inputVolume_RAS - straightenedVolume_RAS

        # Call modified manually to indicate modificatoin of np array (original calls arrayFromGridTransformModified)
        gridImage.GetPointData().GetScalars().Modified()
        gridImage.Modified()

        # Saving as sitk transform, since straightening is done with sitk
        direction = gridDirectionMatrixArray[0:3,0:3].flatten().tolist()

        displacement_image_RAS = sitk.GetImageFromArray(transformDisplacements_RAS)
        displacement_image_RAS.SetOrigin(transformGridOrigin)
        displacement_image_RAS.SetSpacing(gridSpacing)
        displacement_image_RAS.SetDirection(direction)
        itk_transform_RAS = sitk.DisplacementFieldTransform(displacement_image_RAS)

        # Saving gridData
        # if curve_id:
        #     transform_filename = curve_id + '_transform'
        # else:
        #     transform_filename = "straight_transform"
        #
        # sitk.WriteTransform(itk_transform_RAS, self.transformOutputDir + "/" + transform_filename + '.tfm')
        # sitk.WriteTransform(itk_transform_RAS, self.transformOutputDir + "/" + transform_filename + '.txt')


        # Converting to LPS:
        if convertCoordinatSystem:
            transformDisplacements_LPS = transformDisplacements_RAS * np.array([-1,-1,1])
            transformGridOrigin_LPS = transformGridOrigin * np.array([-1,-1,1])

            rot = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
            direction_LPS = np.dot(rot, gridDirectionMatrixArray[0:3,0:3]).flatten().tolist()

            displacement_image_LPS = sitk.GetImageFromArray(transformDisplacements_LPS)
            displacement_image_LPS.SetOrigin(transformGridOrigin_LPS)
            displacement_image_LPS.SetSpacing(gridSpacing)
            displacement_image_LPS.SetDirection(direction_LPS)

            itk_transform_LPS = sitk.DisplacementFieldTransform(displacement_image_LPS)

            # sitk.WriteTransform(itk_transform_LPS, self.transformOutputDir + "/" + transform_filename + '_LPS.tfm')
            # sitk.WriteTransform(itk_transform_LPS, self.transformOutputDir + "/" + transform_filename + '_LPS.txt')
            return itk_transform_LPS

        return itk_transform_RAS

    def straighten_volume(self, volumeNode, straighteningTransformNode, outputStraightenedVolumeSpacing,
                          isLabelmap, file_name):
        """
        Compute straightened volume (useful for example for visualization of curved vessels)
        straighteningTransformNode (sitk gridTransform)
        """

        ### Get transformation grid geometry
        # Load grid
        transformGrid1 = straighteningTransformNode

        # Get parameters from grid
        FP = transformGrid1.GetFixedParameters()
        gridDimensions = FP[0:3]
        gridOrigin = FP[3:6]
        gridSpacing = FP[6:9]
        gridOrientation = FP[9:18]

        gridExtentMm = [gridSpacing[0]*(gridDimensions[0]-1), gridSpacing[1]*(gridDimensions[1]-1), gridSpacing[2]*(gridDimensions[2]-1)]

        # Reshape direction/orientation to 3x3 matrix, since it is stored flat
        gridDirection = np.array(gridOrientation).reshape(3,3)

        ### Compute IJK to RAS matrix of output volume -
        # Save grid axis directions as 4x4 array
        straightenedVolumeIJKToLPSArray = np.zeros([4,4])
        straightenedVolumeIJKToLPSArray[0:3,0:3] = gridDirection
        straightenedVolumeIJKToLPSArray[3,3] = 1

        # Apply scaling
        straightenedVolumeIJKToLPSArray = np.dot(straightenedVolumeIJKToLPSArray,
          np.diag([outputStraightenedVolumeSpacing[0], outputStraightenedVolumeSpacing[1], outputStraightenedVolumeSpacing[2], 1]))

        # Set origin
        straightenedVolumeIJKToLPSArray[0:3,3] = gridOrigin #?

        # Calculate dimensions of output volume
        outputDimensions = [int(gridExtentMm[0]/outputStraightenedVolumeSpacing[0]), int(gridExtentMm[1]/outputStraightenedVolumeSpacing[1]), int(gridExtentMm[2]/outputStraightenedVolumeSpacing[2])]

        # Use nearest neighbor interpolation for label volumes (to avoid incorrect labels at boundaries)
        # and higher-order (bspline) interpolation for scalar volumes.
        interpolationType = sitk.sitkNearestNeighbor if isLabelmap else sitk.sitkBSpline

        default_ct_value = -2048
        # computes the acual transformed volume
        resampled = sitk.ResampleImageFilter()
        resampled.SetReferenceImage(volumeNode)
        resampled.SetInterpolator(interpolationType)
        resampled.SetDefaultPixelValue(default_ct_value)
        resampled.SetOutputDirection(gridOrientation)
        resampled.SetOutputOrigin(gridOrigin)
        resampled.SetOutputSpacing(outputStraightenedVolumeSpacing)
        resampled.SetSize(outputDimensions)
        resampled.SetTransform(transformGrid1)

        outResampled = resampled.Execute(volumeNode)

        writer = sitk.ImageFileWriter()
        writer.SetFileName(file_name)
        writer.Execute(outResampled)

        return outResampled
