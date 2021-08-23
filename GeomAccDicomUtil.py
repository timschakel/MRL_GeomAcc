#!/usr/bin/env python

# This code is developed to be used as part of an analysis module within the WADQC software
# Its a helper function to load in dicom files and supply them to the GeomAcc
# class in the correct format.
# 
# It has been adapted from the original code of Erik van der Bijl
# The WAD Software can be found on https://bitbucket.org/MedPhysNL/wadqc/wiki/Home
#  
#
# Changelog:
#
# 20210623: initial version
# 20210820: rename and remove obsolete code

import pydicom as pydicom
import numpy as np
from operator import itemgetter

def thru_plane_position(dcm):
    """Gets spatial coordinate of image origin whose axis
    is perpendicular to image plane.
    """
    try:
        orientation = tuple((float(o) for o in dcm.ImageOrientationPatient))
        position = tuple((float(p) for p in dcm.ImagePositionPatient))
        rowvec, colvec = orientation[:3], orientation[3:]
        normal_vector = np.cross(rowvec, colvec)
        slice_pos = np.dot(position, normal_vector)
    except:
        print(dcm.ImageType)
    return slice_pos

class Dicom_Series():
    def __init__(self,dcm_filenames):
        self.load_data(dcm_filenames)

    def _get_cube_orientation(self, image_orientation):
        u_ori, v_ori = np.split(np.array(image_orientation, dtype=float), 2)
        plane_normal = np.cross(u_ori, v_ori)
        return np.array([plane_normal, u_ori, v_ori])

    def load_data(self,dcm_filenames):

        
        dcm_slices = [pydicom.read_file(fn) for fn in dcm_filenames]
        #load data
        self.header = dcm_slices[0]
        self.slices = dcm_slices
        
        # Extract position for each slice to sort and calculate slice spacing
        dcm_slices = [(dcm, thru_plane_position(dcm)) for dcm in dcm_slices]
        dcm_slices = sorted(dcm_slices, key=itemgetter(1))

        spacings = np.diff([dcm_slice[1] for dcm_slice in dcm_slices])
        slice_spacing = np.mean(spacings)

        # All slices will have the same in-plane shape
        shape = (int(dcm_slices[0][0].Columns), int(dcm_slices[0][0].Rows))
        self.nslices = len(dcm_slices)

        # Final 3D array will be N_Slices x Columns x Rows
        self.shape = (self.nslices, *shape)
        self.voxel_data = np.empty(self.shape, dtype='float32')
        slope = 1.0
        intercept = 1.0
        for idx, (dcm, _) in enumerate(dcm_slices):
            # Rescale and shift in order to get accurate pixel values
            try:

                # print(slope,intercept)
                slope = float(dcm.RealWorldValueMappingSequence[0].RealWorldValueSlope)
                intercept = float(dcm.RealWorldValueMappingSequence[0].RealWorldValueIntercept)
                # print('slope',slope)
                # print('intercept',intercept)
                # print(dcm.ImageType)
            except:

                # print(dcm)
                slope = float(dcm.RescaleSlope)
                intercept = float(dcm.RescaleIntercept)
                # print('slope', slope)
                # print('intercept', intercept)
                # print('not found')
            self.voxel_data[idx, ...] = dcm.pixel_array.astype('float32') * slope + intercept

        # Calculate size of a voxel in mm
        pixel_spacing = tuple(float(spac) for spac in dcm_slices[0][0].PixelSpacing)

        self.voxel_spacing = (slice_spacing, *pixel_spacing)
        self.origin = np.array(dcm_slices[0][0].ImagePositionPatient).astype('float32')

        self.axs = self._get_cube_orientation(self.header.ImageOrientationPatient)

