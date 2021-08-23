#!/usr/bin/env python

# This code is developed to be used as an analysis module within the WADQC software
# It functions as a wrapper around the GeomAcc class
# 
#
#
# The WAD Software can be found on https://bitbucket.org/MedPhysNL/wadqc/wiki/Home
#  
#
# Changelog:
#
# 20210623: initial version

from __future__ import print_function
import os
import sys

__version__ = '20210623'
__author__ = 'tschakel'

# for development it could be convenient to download the wadqc software locally and create a reference to it:
#sys.path.insert(0, 'C:/Users/Tim/Software/github/wadqc')

# import packages
from wad_qc.module import pyWADinput
from wad_qc.modulelibs import wadwrapper_lib
from wad_qc.module.moduledata import ModuleData
from wad_qc.module.moduleresults import ModuleResults
import numpy as np
import pydicom as dicom

# matplotlib has some specific workaround, this code is copied from a factory module:
if not 'MPLCONFIGDIR' in os.environ:
    import pkg_resources
    try:
        #only for matplotlib < 3 should we use the tmp work around, but it should be applied before importing matplotlib
        matplotlib_version = [int(v) for v in pkg_resources.get_distribution("matplotlib").version.split('.')]
        if matplotlib_version[0]<3:
            os.environ['MPLCONFIGDIR'] = "/tmp/.matplotlib" # if this folder already exists it must be accessible by the owner of WAD_Processor
    except:
        os.environ['MPLCONFIGDIR'] = "/tmp/.matplotlib" # if this folder already exists it must be accessible by the owner of WAD_Processor

import matplotlib
matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.

import GeomAccDicomUtil #GeomAcc uses the Dicom_Series class from here to load the data
import GeomAcc

def logTag():
    return "[MRL_GeomAcc_wadwrapper] "

def geomfidel_analysis(data, results, action):
    # Load dicoms from data
    dcmseries = GeomAccDicomUtil.Dicom_Series(data.series_filelist[0])

    # Initialize GeomAcc class
    GeomFidel = GeomAcc.GeomAcc()
    GeomFidel.loadSeries(dcmseries)
    GeomFidel.correctDetectedClusterPositionsForSetup()
    
    # Save images
    GeomFidel.save_images_to_wad(results)
    
    # Collect results
    reportkeyvals = []
    for cc_position in GeomFidel.positions_CC:
        idname = "_"+str(cc_position)
          
        # get the differences
        ix = GeomFidel.indices_cc_pos(GeomFidel.correctedMarkerPositions,cc_position)
        
        distance_to_1mm, pos1 = GeomFidel.calculateDistanceToDifference(ix,1)
        distance_to_2mm, pos2 = GeomFidel.calculateDistanceToDifference(ix,2)
        differences = GeomFidel.differencesCorrectedExpected[ix]
        
        means_cc_position = np.mean(differences)
        stdev_cc_position = np.std(differences)
        
        dsv400_max, dsv400_mean = GeomFidel.calculateStatisticsForDSV(ix,400)
        dsv200_max, dsv200_mean = GeomFidel.calculateStatisticsForDSV(ix,200)
        
        reportkeyvals.append( ("mean diff"+idname,means_cc_position) )
        reportkeyvals.append( ("std diff"+idname,stdev_cc_position) )
        reportkeyvals.append( ("distance to 1mm"+idname,distance_to_1mm) )
        reportkeyvals.append( ("distance to 2mm"+idname,distance_to_2mm) )
        reportkeyvals.append( ("DSV 400 max"+idname,dsv400_max) )
        reportkeyvals.append( ("DSV 200 max"+idname,dsv200_max) )
        
    for key,val in reportkeyvals:
        results.addFloat(key, val)
        
def acqdatetime_series(data, results, action):
    """
    Read acqdatetime from dicomheaders and write to IQC database

    Workflow:
        1. Read only headers
    """
    try:
        params = action['params']
    except KeyError:
        params = {}

    ## 1. read only headers
    dcmInfile = dicom.read_file(data.series_filelist[0][0], stop_before_pixels=True)

    dt = wadwrapper_lib.acqdatetime_series(dcmInfile)

    results.addDateTime('AcquisitionDateTime', dt)
    
#### main function
if __name__ == "__main__":
    data, results, config = pyWADinput()

    print(config)
    for name,action in config['actions'].items():

        if name == 'acqdatetime':
            acqdatetime_series(data, results, action)

        elif name == 'geomfidel_analysis':
            geomfidel_analysis(data, results, action)

    results.write()

