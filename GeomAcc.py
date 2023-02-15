#!/usr/bin/env python

# This code is developed to be used as part of an analysis module within the WADQC software
# The GeomAcc class expects MR data from the geometric fidelity phantom
# Data can be loaded using GeomAccDicomUtil function
# It performs marker localization and gives some statistics on the deviations
# 
# It has been adapted from the original code of Erik van der Bijl
# The WAD Software can be found on https://bitbucket.org/MedPhysNL/wadqc/wiki/Home
#  
#
# Changelog:
#
# 20210623: initial version
# 20210820: remove obsolete code

import numpy as np
from scipy import optimize
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import gaussian_filter
import logging
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from datetime import datetime,date,time
from scipy.cluster.vq import kmeans2
from multiprocessing import Pool

#initiate logger
logger = logging.getLogger(__name__)
handler = logging.FileHandler('GeomAcc')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

def _parallel_cluster(points):
    result = []
    cur_points=np.copy(points)
    while len(cur_points) > 0:
        curPoint = cur_points[0]
        squaredDistanceToCurrentPoint = cdist([curPoint], cur_points, 'sqeuclidean')[0]
        pointsCloseToCurrent = squaredDistanceToCurrentPoint < 225
        result.append(np.mean(cur_points[pointsCloseToCurrent], axis=0))
        cur_points = cur_points[np.logical_not(pointsCloseToCurrent)]
    return np.array(result)


class GeomAcc():

    def __init__(self):
        
        # can possibly distinguish between different configations/setups with new Defaults file
        import GeomAccDefaultsMRL as GeomAccDefaults

        #Define default constants
        self.LR = GeomAccDefaults.LR
        self.AP = GeomAccDefaults.AP
        self.CC = GeomAccDefaults.CC

        #Properties of the study
        self.studyDate = None
        self.studyTime = None
        self.studyScanner = None

        #results for this study
        self.rigid_transformation_setup = [0, 0, 0, 0, 0, 0]
        self.measurementsPerTablePos = {}
        self.measurementTablePositions = []

        #Phantom type dependent default values
        self.NEMA_PAIRS = np.array(GeomAccDefaults.NEMA_PAIRS)
        self.CENTER_POSITIONS = np.array(GeomAccDefaults.CENTER_POSITIONS)
        self.BODY_CTR_POSITIONS = np.array(GeomAccDefaults.BODY_CTR_POSITIONS)
        self.ALL_POSITIONS = np.array(GeomAccDefaults.markerPositions_LR_AP,dtype=int)

        #Constants/Config

        self.TRANSVERSE_ORIENTATION = GeomAccDefaults.TRANSVERSEORIENTATION
        self.MARKER_THRESHOLD_AFTER_FILTERING = GeomAccDefaults.MARKER_THRESHOLD_AFTER_FILTERING
        self.CLUSTERSIZE = GeomAccDefaults.CLUSTERSIZE
        self.LIMIT_CC_SEPARATION_FROM_CC_POSITION=GeomAccDefaults.LIMIT_CC_SEPARATION_FROM_CC_POSITION

        self.positions_CC = np.sort(np.array(GeomAccDefaults.marker_positions_CC))
        self.positions_LR_AP = np.array(GeomAccDefaults.markerPositions_LR_AP,dtype=float)

        self.degLimit = GeomAccDefaults.LIMITFITDEGREES
        self.transLimit = GeomAccDefaults.LIMITFITTRANS

        self.expectedMarkerPositions = self._expected_marker_positions(self.positions_CC, self.positions_LR_AP)

        #Results
        self.detectedMarkerPositions  = None
        self.correctedMarkerPositions = None
        self.closestExpectedMarkerIndices = None
        self.differencesCorrectedExpected = None

    def _expected_marker_positions(self, marker_positions_cc, marker_positions_LR_AP):
        #create a complete list of marker positions by copying the AP_LR
        # list to every cc position that is scanned
        expected_marker_positions= np.vstack([self.__marker_positions_at_cc_pos(cc_pos,marker_positions_LR_AP) for cc_pos in marker_positions_cc])
        return expected_marker_positions

    def __marker_positions_at_cc_pos(self, cc_pos, marker_positions_LR_AP):
        return np.hstack((marker_positions_LR_AP,
                          np.ones((len(marker_positions_LR_AP), 1)) * cc_pos))

    def loadSeries(self,dcmSeries):
        logger.log(logging.INFO, 'loading data from series')

        #Read Dicom header data
        self.readHeaderData(dcmSeries)

        #Get all points (3D) with high contrast
        detectedPoints=self._getHighIntensityPoints(dcmSeries)

        #Cluster high contrast points into markerpositions
        self.detectedMarkerPositions=self._createClusters(detectedPoints,size=self.CLUSTERSIZE)
        logger.log(logging.INFO,self.detectedMarkerPositions)

    def readHeaderData(self,dcmSeries):

        self.studyDate=datetime.strptime(dcmSeries.header.AcquisitionDate,"%Y%m%d").date()
        self.studyTime=datetime.strptime(dcmSeries.header.AcquisitionTime,"%H%M%S.%f").time()
        self.studyScanner=dcmSeries.header.PerformedStationAETitle
        self.seriesDescription = dcmSeries.header.SeriesDescription

    def _getHighIntensityPoints(self, dcmSeries):

        self.origin = dcmSeries.origin
        self.spacing = dcmSeries.voxel_spacing
        self.axs = dcmSeries.axs

        highVoxels = self._getHighVoxelsFromImageData(np.swapaxes(dcmSeries.voxel_data,1,2))
        highPoints = self.index_to_coords(highVoxels)
        print(highPoints.shape)

        return highPoints

    def index_to_coords(self,ix):
        return self.origin+np.dot((ix.T*self.spacing),self.axs)

    def _getHighVoxelsFromImageData(self, imageData):
        logger.log(logging.INFO, "Filtering dataset")
        
        sigma = 2.5
        dataCube = gaussian_filter(imageData, sigma, truncate=5)/ np.sqrt(2*np.pi * sigma**2)
        idx = np.argwhere(dataCube > np.percentile(dataCube,99.5)).T
        #idx = np.argwhere(imageData > self.MARKER_THRESHOLD_AFTER_FILTERING).T
        return idx


    def _createClusters(self, points, size):
        logger.log(logging.INFO, "Finding clusters of high intensity voxels")

        #k-means clustering to separate distinct marker planes
        # Cluster according to expected CC positions of phantomslabs
        logger.log(logging.INFO,points.T[self.CC])
        centroids,cluster_id= kmeans2(points.T[self.CC], self.positions_CC)

        points_per_cc = [points[cluster_id == n_cluster] for n_cluster in np.arange(len(self.positions_CC))]

        workpool = Pool(6)
        clusters = np.concatenate(workpool.map(_parallel_cluster,points_per_cc))
        return clusters

    def indices_cc_pos(self, positions, cc_pos):
        return np.abs(positions.T[self.CC] - cc_pos) < self.LIMIT_CC_SEPARATION_FROM_CC_POSITION

    def setCorrectedMarkerPositions(self,transformation):
        self.correctedMarkerPositions = self.rigidTransform(self.detectedMarkerPositions,transformation[0:3],[transformation[3], transformation[4], transformation[5]])
        self.closestExpectedMarkerIndices = self.closestLocations(self.correctedMarkerPositions,self.expectedMarkerPositions)
        self.differencesCorrectedExpected = self.getdifferences(self.correctedMarkerPositions,self.expectedMarkerPositions)

    def closestLocations(self, detectedMarkerPositions, expectedMarkerPositions):
        distances = np.sum(np.power((detectedMarkerPositions - expectedMarkerPositions[:, np.newaxis]), 2), axis=2)
        return np.argmin(distances, axis=0)

    def getdifferences(self, markerPositions,expectedMarkerPositions):
        return markerPositions - expectedMarkerPositions[self.closestLocations(markerPositions,expectedMarkerPositions)]

    def _findMarkerIndex(self, xyz):
        x,y,z=zip*(xyz)
        xPosMarkers, yPosMarkers,zPosMarkers = self.expectedMarkerPositions.T
        xMatch = (x == xPosMarkers)
        yMatch = (y == yPosMarkers)
        zMatch = (z == zPosMarkers)
        return np.argwhere(np.logical_and(np.logical_and(xMatch, yMatch),zMatch))[0]


    def correctDetectedClusterPositionsForSetup(self):
        """
        This function adds corrected clusterpositions to the measurements based on the
        calculated setup rotation and translation in the measurement at tableposition 0
        :return:
        """
        logger.log(logging.INFO, "Correcting for phantom setup")

        #Select only markers within 100 mm of isoc
        detected_markers_at_isoc_plane = self.detectedMarkerPositions[self.indices_cc_pos(self.detectedMarkerPositions,cc_pos=0.0)]
        dist_to_isoc_2d = cdist([[0.0, 0.0]], detected_markers_at_isoc_plane[:,:-1], metric='euclidean')[0]
        ix=dist_to_isoc_2d<100

        detected_markers_at_isoc_plane=detected_markers_at_isoc_plane[ix]
        markers_at_isoc_plane = self.__marker_positions_at_cc_pos(0.0, self.positions_LR_AP)

        self.rigid_transformation_setup = self._findRigidTransformation(detected_markers_at_isoc_plane,markers_at_isoc_plane)
        self.setCorrectedMarkerPositions(self.rigid_transformation_setup)

    def _findRigidTransformation(self, detectedMarkerPositions, expectedMarkerPositions):
        logger.log(logging.INFO, "Determining setup translation and rotation for tableposition 0")

        # average detected cc position
        init_CC = np.mean(detectedMarkerPositions, axis=0)[self.CC]

        # optimization init
        optimization_initial_guess = np.zeros(6)
        optimization_initial_guess[self.CC] = init_CC

        # optimization bounds
        optimization_bounds = [(-self.transLimit, self.transLimit),
                               (-self.transLimit, self.transLimit),
                               (-self.transLimit, self.transLimit),
                               (-self.degLimit, self.degLimit),
                               (-self.degLimit, self.degLimit),
                               (-self.degLimit, self.degLimit)]

        def penaltyFunction(transRot):
            opt_pos = self.rigidTransform(detectedMarkerPositions, translation=transRot[0:3], eulerAngles=transRot[3:6])
            differences = self.getdifferences(opt_pos, expectedMarkerPositions)
            penalty = np.sum(np.sum(np.power(differences, 2)))
            return penalty

        opt_result = optimize.minimize(fun=penaltyFunction,
                                       x0=optimization_initial_guess,
                                       bounds=optimization_bounds,
                                       tol=.0001)

        return opt_result.x

    def rigidRotation(self, markerPositions, eulerAngles):
        s0 = np.sin(eulerAngles[0])
        c0 = np.cos(eulerAngles[0])
        s1 = np.sin(eulerAngles[1])
        c1 = np.cos(eulerAngles[1])
        s2 = np.sin(eulerAngles[2])
        c2 = np.cos(eulerAngles[2])

        m00 = c1 * c2
        m01 = c0 * s2 + s0 * s1 * c2
        m02 = s0 * s2 - c0 * s1 * c2
        m10 = -c1 * s2
        m11 = c0 * c2 - s0 * s1 * s2
        m12 = s0 * c2 + c0 * s1 * s2
        m20 = s1
        m21 = -s0 * c1
        m22 = c0 * c1

        rotationMatrix = np.array([
            [m00, m01, m02],
            [m10, m11, m12],
            [m20, m21, m22]])

        return np.dot(markerPositions, rotationMatrix)

    def rigidTranslation(self, markerPositions, Translation):
        return markerPositions + Translation

    def rigidTransform(self, positions, translation, eulerAngles):
        rotated_positions = self.rigidRotation(positions, eulerAngles)
        transformed_positions = self.rigidTranslation(rotated_positions, translation)
        return transformed_positions

    def calculateDistanceToDifference(self, markerIdxs, limit_in_mm):
        """
        This function calculates the distance from to origin (0,0,cc_pos) to the 
        closest marker location which has a certain defined deviation.
        Calculation is per slice (cc_pos)
        
        Input: 
        markerIdxs: the indices of the markers to consider 
        (obtained eg: markerIdxs = GeomAcc.indices_cc_pos(GeomAcc.correctedMarkerPositions,cc_position))
        limit_in_mm: the defined deviation, function will locate the first marker 
        that exceeds this deviation in its detected position.
        
        Output:
        distance_to_difference: the distance from the origin to the found marker
        """
        
        # calculate distance to origin and sort based on this
        detected = self.correctedMarkerPositions[markerIdxs]
        detected_dist_to_origin = np.sqrt(np.power(detected[:,0], 2)+np.power(detected[:,1], 2))
        detected_dist_to_origin_sortix = np.argsort(detected_dist_to_origin)
        
        # sort the differences the same way and find the first difference (distance)
        # that exceeds the limit
        differences = self.differencesCorrectedExpected[markerIdxs]
        distlength = np.sqrt(np.sum(np.power(differences, 2), axis=1))     
        limit_ix = np.argmax(distlength[detected_dist_to_origin_sortix] > limit_in_mm)
        
        expectedMarkerPositions = self.correctedMarkerPositions[markerIdxs] - self.differencesCorrectedExpected[markerIdxs]
        markerpos = expectedMarkerPositions[detected_dist_to_origin_sortix][limit_ix]
        markerpos_ix = np.where((expectedMarkerPositions[:,0] == markerpos[0]) & (expectedMarkerPositions[:,1] == markerpos[1]) & (expectedMarkerPositions[:,2] == markerpos[2]))[0][0]
        dist_origin = np.sqrt(np.power(markerpos[0], 2)+np.power(markerpos[1], 2))
        
        return dist_origin, markerpos_ix
    
    def calculateDistanceToDifference3D(self, limit_in_mm):
        """
        This function calculates the distance from to origin (0,0,0) to the 
        closest marker location which has a certain defined deviation.
        Calculation is in 3D for entire volume
        
        Input: 
        markerIdxs: the indices of the markers to consider 
        limit_in_mm: the defined deviation, function will locate the first marker 
        that exceeds this deviation in its detected position.
        
        Output:
        distance_to_difference: the distance from the origin to the found marker
        """
        
        # calculate distance to origin and sort based on this
        detected = self.correctedMarkerPositions
        origin = [[0.0,0.0,0.0]]
        detected_dist_to_origin = cdist(detected,origin).flatten()
        detected_dist_to_origin_sortix = np.argsort(detected_dist_to_origin)
        
        # calculate the difference lengths and find the first difference length 
        # that exceeds the limit (after sorting the list using the previous indices)
        difference = self.differencesCorrectedExpected
        difference_length = np.sqrt( difference[:,0]**2 + difference[:,1]**2 + difference[:,2]**2 )
        limit_ix = np.argmax(difference_length[detected_dist_to_origin_sortix] > limit_in_mm)
        
        # take the index for the first marker to exceed limit
        # and calculate its distance from the origin
        markerposition = detected[detected_dist_to_origin_sortix][limit_ix]
        marker_dist_to_origin = np.sqrt( markerposition[0]**2 + markerposition[1]**2 + markerposition[2]**2 )

        # Also get the marker that gives the radius for which 98% of the markers are within the limit
        # (according to longitudinal QA article)
        # For this, consider the max we just found to be 98% instead of the max (100%)
        ix_98 = np.int0(np.floor(100*limit_ix/98))
        markerposition_98 = detected[detected_dist_to_origin_sortix][ix_98]
        marker98_dist_to_origin = np.sqrt( markerposition_98[0]**2 + markerposition_98[1]**2 + markerposition_98[2]**2 )
                
        return marker_dist_to_origin, marker98_dist_to_origin
    
    
    def calculateStatisticsForDSV3D(self, diameter):
        """
        Calculate some statistics for the markers within a specified sphere (DSV)
        
        Markers are selected based on their detected position
        
        ----------

        diameter : float
            diameter of the DSV (mm)

        Returns
        -------
        dsv_max: Max deviation length (3D) within specified DSV.
        dsv_p98: 98th percentile value of all deviations (3D length) within specified DSV

        """
        detected = self.correctedMarkerPositions
        differences = self.differencesCorrectedExpected
        
        detected_dist_to_origin = np.sqrt(np.power(detected[:,0], 2)+np.power(detected[:,1], 2)+np.power(detected[:,2], 2))
        dsv_ix = detected_dist_to_origin < diameter/2
        
        # for every detected marker in the given diameter, take 3D distance length of difference with expected
        distlength = np.sqrt(np.sum(np.power(differences[dsv_ix], 2), axis=1))  
        
        dsv_max = np.max(distlength)
        dsv_p98 = np.percentile(distlength,98)
        dsv_p95 = np.percentile(distlength,95)
        
        return dsv_max, dsv_p98, dsv_p95
    
    def calculateStatisticsForDSV(self, markerIdxs, diameter):
        """
        Calculate some statistics for the markers within a specified circle
        (analogous to DSV, but consider an in-plane circle for every slab position)
        
        Markersare selected based on their detected position
        
        ----------
        markerIdxs : TYPE
            the indices of the markers to consider 
        (obtained eg: markerIdxs = GeomAcc.indices_cc_pos(GeomAcc.correctedMarkerPositions,cc_position)).
        diameter : float
            diameter of the circle (mm)

        Returns
        -------
        None.

        """
        detected = self.correctedMarkerPositions[markerIdxs]
        differences = self.differencesCorrectedExpected[markerIdxs]
        
        detected_dist_to_origin = np.sqrt(np.power(detected[:,0], 2)+np.power(detected[:,1], 2))
        dsv_ix = detected_dist_to_origin < diameter/2
        
        # for every detected marker in the given diameter, take 3D distance length of difference with expected
        distlength = np.sqrt(np.sum(np.power(differences[dsv_ix], 2), axis=1))  
        
        dsv_max = np.max(distlength)
        dsv_mean = np.mean(distlength)
        
        return dsv_max, dsv_mean
        
    def calculateStatisticsForExpectedMarkerPositions(self, cc_pos, markerPositions):
        indices = np.array([self._findMarkerIndex((posx, posy, cc_pos)) for (posx, posy) in markerPositions]).flatten()
        indices_of_differences = []
        No_Missing_Points = 0
        for index in indices:
            try:
                indices_of_differences.append(np.argwhere(index == self.closestExpectedMarkerIndices)[0, 0])
            except:
                No_Missing_Points += 1
        differences_defined_by_indices = self.differencesCorrectedExpected[indices_of_differences]
        mean = np.mean(differences_defined_by_indices, axis=0)
        rms = np.std(differences_defined_by_indices, axis=0)
        return mean, rms, No_Missing_Points

    def calculateMaxDistanceWithinLimit(self, cc_pos, limit_in_mm):

        #select relevant phantom slab
        ix = self.indices_cc_pos(self.correctedMarkerPositions, cc_pos)
        detectedMarkerPositions = self.correctedMarkerPositions[ix]
        differences = self.differencesCorrectedExpected[ix]

        #Find distances of found markers and ordering according to distance to isoc
        correctedDistancesToIsoc = cdist([[0.0,0.0,0.0]],detectedMarkerPositions,metric='euclidean')[0]
        order_distances=np.argsort(correctedDistancesToIsoc)

        #find the first exceeding the limit
        idx_first_exceeding = np.argmax(cdist([[0.0,0.0,0.0]],differences[order_distances],metric='euclidean')[0]>limit_in_mm)
        return (correctedDistancesToIsoc[order_distances])[idx_first_exceeding]

    def save_images_to_wad(self,results):
        logger.log(logging.INFO, "Creating and saving figures")
        fileName = "positions.jpg"
        self.createDeviationFigure(fileName)
        results.addObject("detectedPositions",fileName)

        fileName = "Histograms.jpg"
        self.createHistogramsFigure(fileName)
        results.addObject("Histograms", fileName)


    def save_logfile_to_wad(self, results):
        results.addObject(description="Logfile", value='GeomAcc', level=2)

    def createDeviationFigure(self,fileName=None):
        fig, axs = plt.subplots(ncols=2, nrows=4, sharey=True, sharex=True,figsize=(8, 12))
        title = self.studyDate.strftime("%Y-%m-%d ") +self.studyTime.strftime("%H:%M:%S ") +" "+str(self.studyScanner)
        fig.suptitle(title,fontsize=20,x=0.75)

        self._createDeviationSubplot(ax=axs[0, 0], cc_position=self.positions_CC[3])
        self._createDeviationLegend (ax=axs[0, 1])

        self._createDeviationSubplot(ax=axs[1, 0], cc_position=self.positions_CC[2])
        self._createDeviationSubplot(ax=axs[1, 1], cc_position=self.positions_CC[4])

        self._createDeviationSubplot(ax=axs[2, 0], cc_position=self.positions_CC[1])
        self._createDeviationSubplot(ax=axs[2, 1], cc_position=self.positions_CC[5])

        self._createDeviationSubplot(ax=axs[3, 0], cc_position=self.positions_CC[0])
        self._createDeviationSubplot(ax=axs[3, 1], cc_position=self.positions_CC[6])

        # fig.tight_layout()
        if not fileName:
            fileName = 'GNL Deviations '+self.studyDate.strftime("%Y-%m-%d ")+'.pdf'
        logging.log(logging.INFO, 'saving file:' + fileName)
        plt.subplots_adjust(top=.95)
        fig.savefig(fileName,dpi=150)

    def _createDeviationSubplot(self, ax, cc_position):
        ix = self.indices_cc_pos(self.correctedMarkerPositions,cc_position)
        detectedPositions = self.correctedMarkerPositions[ix]
        differences = self.differencesCorrectedExpected[ix]
        distlength = np.sqrt(np.sum(np.power(differences, 2), axis=1))
        markerPositions = self.positions_LR_AP

        ax.set_facecolor('black')
        ax.set_aspect('equal')
        ax.set_title('Position: {0:.1f}'.format(cc_position))
        # ax.text(.02,.9,cc_position, ha='left', va='top', transform=ax.transAxes,color='white')
        ax.text(.98, .9, '{0:.1f} $\pm$ {1:.1f} mm'.format(np.mean(differences),np.std(differences)), ha='right', va='top', transform=ax.transAxes, color='white')

        ax.set_xlim(-275, 275)
        ax.set_ylim(-150, 275)
        ax.set_xticks([],[])
        ax.set_yticks([],[])

        ax.scatter(markerPositions[:, self.LR], - markerPositions[:,  self.AP], marker='x', c='blue')
        ax.scatter(detectedPositions[distlength > 2, self.LR], - detectedPositions[distlength > 2, self.AP], marker='o', c='red')
        ax.scatter(detectedPositions[distlength < 2, self.LR], - detectedPositions[distlength < 2, self.AP], marker='o', c='yellow')
        ax.scatter(detectedPositions[distlength < 1, self.LR], - detectedPositions[distlength < 1, self.AP], marker='o', c='green')
        
        #add origin
        ax.scatter(0,0,marker='x', c='white')
        
        # draw circles with radius with distance where closest marker with deviation larger than 1mm (white) & 2mm (orange) 
        #dist_origin, markerpos_ix = self.calculateDistanceToDifference(ix, 1)
        #draw_circle = plt.Circle((0,0),dist_origin,fill=False,color='yellowgreen',lw=3)
        #ax.add_artist(draw_circle)
        
        #dist_origin, markerpos_ix = self.calculateDistanceToDifference(ix, 2)
        #draw_circle = plt.Circle((0,0),dist_origin,fill=False,color='orange',lw=3)
        #ax.add_artist(draw_circle)
        
        # Alternative: draw circles based on the 3D DSV
        # the radius of the spheres at the intersections with the phantom slices are given by:
        # radius(z) = sqrt(R^2 - z^2), with R the radius of the current sphere
        radius_1mm, radius_98p_1mm = self.calculateDistanceToDifference3D(1.0)
        radius_2mm, radius_98p_2mm = self.calculateDistanceToDifference3D(2.0)
        
        # first check if the current cc_pos intersects with sphere
        if radius_1mm**2 - cc_position**2 > 0:
            radius_1mm_this_cc_pos = np.sqrt(radius_1mm**2 - cc_position**2)
            draw_circle = plt.Circle((0,0),radius_1mm_this_cc_pos,fill=False,color='yellowgreen',lw=3)
            ax.add_artist(draw_circle)
            
        if radius_2mm**2 - cc_position**2 > 0:
            radius_2mm_this_cc_pos = np.sqrt(radius_2mm**2 - cc_position**2)
            draw_circle = plt.Circle((0,0),radius_2mm_this_cc_pos,fill=False,color='orange',lw=3)
            ax.add_artist(draw_circle)
        

    def _createDeviationLegend(self,ax):

        blue_cross = mlines.Line2D([], [], color='blue', marker='x',
                                  markersize=15,linestyle='None', label='Expected marker')
        green = mlines.Line2D([], [], color='green', marker='o',
                                   markersize=15,linestyle='None', label=r'$\delta$ < 1 mm')
        yellow = mlines.Line2D([], [], color='yellow',linestyle='None', marker='o',
                                   markersize=15, label=r'1 mm < $\delta$ < 2 mm')
        red = mlines.Line2D([], [], color='red',linestyle='None', marker='o',
                                   markersize=15, label=r'$\delta$ > 2 mm')
        yellowgreen_line = mlines.Line2D([], [], color='yellowgreen', marker='_',
                                  markersize=15,linestyle='None', label='Deviation < 1 mm')
        orange_line = mlines.Line2D([], [], color='orange', marker='_',
                                  markersize=15,linestyle='None', label='Deviation < 2 mm')
        ax.legend(handles=[blue_cross,green,yellow,red,yellowgreen_line,orange_line],loc=10,title=r'Deviation $\delta$ from expected position',facecolor='gray')
        ax.set_axis_off()

    def createHistogramsFigure(self, fileName):
        fig, axs = plt.subplots(ncols=2, nrows=4, sharey=True, sharex=True,figsize=(8,12))

        title = self.studyDate.strftime("%Y-%m-%d ") + self.studyTime.strftime("%H:%M:%S ") + " " + str(self.studyScanner)
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=1)
        self._createHistogramPlot(ax=axs[0, 0], cc_position=self.positions_CC[3])
        self._createHistogramLegend(ax=axs[0,1])

        self._createHistogramPlot(ax=axs[1, 0], cc_position=self.positions_CC[2])
        self._createHistogramPlot(ax=axs[1, 1], cc_position=self.positions_CC[4])

        self._createHistogramPlot(ax=axs[2, 0], cc_position=self.positions_CC[1])
        self._createHistogramPlot(ax=axs[2, 1], cc_position=self.positions_CC[5])

        self._createHistogramPlot(ax=axs[3, 0], cc_position=self.positions_CC[0])
        axs[3, 0].set_xlabel('mm')
        self._createHistogramPlot(ax=axs[3, 1], cc_position=self.positions_CC[6])
        axs[3, 1].set_xlabel('mm')

        fig.tight_layout()
        # fig.xlabel('mm')
        logging.log(logging.INFO, 'saving file:' + fileName)
        plt.subplots_adjust(top=.95)
        fig.savefig(fileName,dpi=150)

    def _createHistogramPlot(self, ax, cc_position):
        ix = self.indices_cc_pos(self.correctedMarkerPositions, cc_position)
        differences = self.differencesCorrectedExpected[ix]

        bins = np.linspace(start=-3, stop=3, num=14)

        mu_diff = np.mean(differences, axis=0)

        sigma_diff = np.std(differences,axis=0)

        textstr = '(AP,LR,CC)\n'\
                  '$\mu$ = ({0:.1f}, {1:.1f}, {2:.1f}) mm \n' \
                  '$\sigma$ = ({3:.1f}, {4:.1f}, {5:.1f}) mm'.format(mu_diff[self.AP],mu_diff[self.LR],mu_diff[self.CC],
                                                                   sigma_diff[self.AP],sigma_diff[self.LR],sigma_diff[self.CC])

        textstrprops = dict(boxstyle='round', facecolor='white', alpha=0.5)

        ax.set_facecolor('white')
        ax.set_title(cc_position)

        ax.hist((differences[:, self.AP],differences[:, self.LR],differences[:, self.CC]), bins=bins, density=False,
                label=('AP', 'LR', 'CC'), color=['green', 'blue', 'red'],rwidth=0.66)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=textstrprops)

    def _createHistogramLegend(self,ax):
        green = mpatches.Patch(color='green',label='AP')
        blue = mpatches.Patch(color='blue',label='LR')
        red = mpatches.Patch(color='red', label='CC')

        ax.legend(handles=[green, blue, red], loc=10, title=r'Deviation $\delta$ from expected position')
        ax.set_axis_off()
