# MR_geomacc

## Summary
This module performs analysis of measurements of the Philips 7-slab Geometric Fideltiy phantom on the Unity MR-Linac.

## Status
Initial version released based on code provided by Erik van der Bijl. Some cleanup is still needed.

## Dependencies
pip:
- numpy
- pydicom
- matplotlib
- scipy
- skimage
- logging
- datetime
- multiprocessing
- operator

## Acquisition protocol
Use the recommended Philips protocol (3D GEOM QA). 

## Selector
A selector for this module should run on dcm_series level and takes data with SeriesDescription: "3D_GEOM_QA:3D,FFE"

### Input data
dcm_series

### Config
The config file is simple and does not provide any additional parameters
- MRL_GeometricFidelity_Unity.json

### Meta
Currently no limits are set for the results.

### Rules
StationName, PatientName, SeriesDescription

## Analysis
Implementation is based detecting marker positions and comparing them with their expected positions.

## Results
- Figure showing all slices of the phantom with: 
	- detected marker positions, colorcoded by their deviation from the expected position
	- circles from geometric center per slice where deviation is smaller than 1 and 2 mm respectively
- Mean, standarddeviation of deviations in position per slice
- Distance (radius of circle) to 1 and 2 mm deviation per slice
- Max deviation in DSV of 200/400 mm per slice
