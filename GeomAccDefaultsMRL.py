#settings for MRL
import numpy as np
TRANSVERSEORIENTATION = [1, 0, 0, 0, 1, 0]
MARKER_THRESHOLD_AFTER_FILTERING =750
CLUSTERSIZE = 225
LIMIT_Z_SEPARATION_FROM_TABLEPOSITION=25

#Labels of directions
AP=1
LR=0
CC=2

LIMITFITDEGREES = 4 * 3.14159265358979 / 180 #radians -> 4 degrees
LIMITFITTRANS = 30 #mm

DIST_TO_ISOC_RIGID = 5  # Take into account the (5x5+1)^2 closest positions to isoc
CLUSTER_SEPARATION = 25

LIMITMAXDISTANCE=2

LIMIT_CC_SEPARATION_FROM_CC_POSITION = 25.0
marker_positions_CC = [-165.0,-110.0,-55.0,0.0,55.0,110.0,165.0]
#marker_positions_CC = [-200.0,-130.0,-60.0,0.0,60.0,130.0,200.0]


BODY_CTR_POSITIONS=np.array([
   #( LR,AP)
    (-75, -125),
    (-50, -125),
    (-25, -125),
    (0, -125),
    (25, -125),
    (50, -125),
    (75, -125),
    (-125, -100),
    (-100, -100),
    (100, -100),
    (125, -100),
    (-150, -75),
    (-125, -75),
    (125, -75),
    (150, -75),
    (-175, -50),
    (-150, -50),
    (150, -50),
    (175, -50),
    (-175, -25),
    (175, -25),
    (-175, 0),
    (175, 0),
    (-175, 25),
    (175, 25),
    (-175, 50),
    (-150, 50),
    (150, 50),
    (175, 50),
    (-150, 75),
    (-125, 75),
    (125, 75),
    (150, 75)
])

CENTER_POSITIONS=np.array([
#   ( LR,AP)
    (-25, -50),
    (0, -50),
    (25, -50),
    (-50, -25),
    (-25, -25),
    (0, -25),
    (25, -25),
    (50, -25),
    (-50, 0),
    (-25, 0),
    (0, 0),
    (25, 0),
    (50, 0),
    (-50, 25),
    (-25, 25),
    (0, 25),
    (25, 25),
    (50, 25),
    (-25, 50),
    (0, 50),
    (25, 50)
])

NEMA_PAIRS=np.array([
    [(225, 0,0), (-225, 0,0)],
    [(225, 25,0), (-225, -25,0)],
    [(-225, 25,0), (225, -25,0)],
    [(0, -200,0), (0, 75,0)],
    [(25,-200,0 ), ( -25,75,0)],
    [(-25,-200,0), (25,75,0 )]
])

markerPositions_LR_AP=np.array([
                       (-250.000,-50.000),
                       (-250.000, -25.000),
                       (-250.000,  +0.000),
                       (-250.000, +25.000),
                       (-250.000, +50.000),
                       (-225.000,-100.000),
                       (-225.000, -75.000),
                       (-225.000, -50.000),
                       (-225.000, -25.000),
                       (-225.000,  +0.000),
                       (-225.000, +25.000),
                       (-225.000, +50.000),
                       (-225.000, +75.000),
                       (-225.000,+100.000),
                       (-225.000,+125.000),
                       (-200.000,-150.000),
                       (-200.000,-125.000),
                       (-200.000,-100.000),
                       (-200.000, -75.000),
                       (-200.000, -50.000),
                       (-200.000, -25.000),
                       (-200.000,  +0.000),
                       (-200.000, +25.000),
                       (-200.000, +50.000),
                       (-200.000, +75.000),
                       (-200.000,+100.000),
                       (-200.000,+125.000),
                       (-175.000,-175.000),
                       (-175.000,-150.000),
                       (-175.000,-125.000),
                       (-175.000,-100.000),
                       (-175.000, -75.000),
                       (-175.000, -50.000),
                       (-175.000, -25.000),
                       (-175.000,  +0.000),
                       (-175.000, +25.000),
                       (-175.000, +50.000),
                       (-175.000, +75.000),
                       (-175.000,+100.000),
                       (-175.000,+125.000),
                       (-150.000,-200.000),
                       (-150.000,-175.000),
                       (-150.000,-150.000),
                       (-150.000,-125.000),
                       (-150.000,-100.000),
                       (-150.000, -75.000),
                       (-150.000, -50.000),
                       (-150.000, -25.000),
                       (-150.000,  +0.000),
                       (-150.000, +25.000),
                       (-150.000, +50.000),
                       (-150.000, +75.000),
                       (-150.000,+100.000),
                       (-150.000,+125.000),
                       (-125.000,-200.000),
                       (-125.000,-175.000),
                       (-125.000,-150.000),
                       (-125.000,-125.000),
                       (-125.000,-100.000),
                       (-125.000, -75.000),
                       (-125.000, -50.000),
                       (-125.000, -25.000),
                       (-125.000,  +0.000),
                       (-125.000, +25.000),
                       (-125.000, +50.000),
                       (-125.000, +75.000),
                       (-125.000,+100.000),
                       (-125.000,+125.000),
                       (-100.000,-225.000),
                       (-100.000,-200.000),
                       (-100.000,-175.000),
                       (-100.000,-150.000),
                       (-100.000,-125.000),
                       (-100.000,-100.000),
                       (-100.000, -75.000),
                       (-100.000, -50.000),
                       (-100.000, -25.000),
                       (-100.000,  +0.000),
                       (-100.000, +25.000),
                       (-100.000, +50.000),
                       (-100.000, +75.000),
                       (-100.000,+100.000),
                       (-100.000,+125.000),
                       ( -75.000,-225.000),
                       ( -75.000,-200.000),
                       ( -75.000,-175.000),
                       ( -75.000,-150.000),
                       ( -75.000,-125.000),
                       ( -75.000,-100.000),
                       ( -75.000, -75.000),
                       ( -75.000, -50.000),
                       ( -75.000, -25.000),
                       ( -75.000,  +0.000),
                       ( -75.000, +25.000),
                       ( -75.000, +50.000),
                       ( -75.000, +75.000),
                       ( -75.000,+100.000),
                       ( -75.000,+125.000),
                       ( -50.000,-250.000),
                       ( -50.000,-225.000),
                       ( -50.000,-200.000),
                       ( -50.000,-175.000),
                       ( -50.000,-150.000),
                       ( -50.000,-125.000),
                       ( -50.000,-100.000),
                       ( -50.000, -75.000),
                       ( -50.000, -50.000),
                       ( -50.000, -25.000),
                       ( -50.000,  +0.000),
                       ( -50.000, +25.000),
                       ( -50.000, +50.000),
                       ( -50.000, +75.000),
                       ( -50.000,+100.000),
                       ( -50.000,+125.000),
                       ( -25.000,-250.000),
                       ( -25.000,-225.000),
                       ( -25.000,-200.000),
                       ( -25.000,-175.000),
                       ( -25.000,-150.000),
                       ( -25.000,-125.000),
                       ( -25.000,-100.000),
                       ( -25.000, -75.000),
                       ( -25.000, -50.000),
                       ( -25.000, -25.000),
                       ( -25.000,  +0.000),
                       ( -25.000, +25.000),
                       ( -25.000, +50.000),
                       ( -25.000, +75.000),
                       ( -25.000,+100.000),
                       ( -25.000,+125.000),
                       (  +0.000,-250.000),
                       (  +0.000,-225.000),
                       (  +0.000,-200.000),
                       (  +0.000,-175.000),
                       (  +0.000,-150.000),
                       (  +0.000,-125.000),
                       (  +0.000,-100.000),
                       (  +0.000, -75.000),
                       (  +0.000, -50.000),
                       (  +0.000, -25.000),
                       (  +0.000,  +0.000),
                       (  +0.000, +25.000),
                       (  +0.000, +50.000),
                       (  +0.000, +75.000),
                       (  +0.000,+100.000),
                       (  +0.000,+125.000),
                       ( +25.000,-250.000),
                       ( +25.000,-225.000),
                       ( +25.000,-200.000),
                       ( +25.000,-175.000),
                       ( +25.000,-150.000),
                       ( +25.000,-125.000),
                       ( +25.000,-100.000),
                       ( +25.000, -75.000),
                       ( +25.000, -50.000),
                       ( +25.000, -25.000),
                       ( +25.000,  +0.000),
                       ( +25.000, +25.000),
                       ( +25.000, +50.000),
                       ( +25.000, +75.000),
                       ( +25.000,+100.000),
                       ( +25.000,+125.000),
                       ( +50.000,-250.000),
                       ( +50.000,-225.000),
                       ( +50.000,-200.000),
                       ( +50.000,-175.000),
                       ( +50.000,-150.000),
                       ( +50.000,-125.000),
                       ( +50.000,-100.000),
                       ( +50.000, -75.000),
                       ( +50.000, -50.000),
                       ( +50.000, -25.000),
                       ( +50.000,  +0.000),
                       ( +50.000, +25.000),
                       ( +50.000, +50.000),
                       ( +50.000, +75.000),
                       ( +50.000,+100.000),
                       ( +50.000,+125.000),
                       ( +75.000,-225.000),
                       ( +75.000,-200.000),
                       ( +75.000,-175.000),
                       ( +75.000,-150.000),
                       ( +75.000,-125.000),
                       ( +75.000,-100.000),
                       ( +75.000, -75.000),
                       ( +75.000, -50.000),
                       ( +75.000, -25.000),
                       ( +75.000,  +0.000),
                       ( +75.000, +25.000),
                       ( +75.000, +50.000),
                       ( +75.000, +75.000),
                       ( +75.000,+100.000),
                       ( +75.000,+125.000),
                       (+100.000,-225.000),
                       (+100.000,-200.000),
                       (+100.000,-175.000),
                       (+100.000,-150.000),
                       (+100.000,-125.000),
                       (+100.000,-100.000),
                       (+100.000, -75.000),
                       (+100.000, -50.000),
                       (+100.000, -25.000),
                       (+100.000,  +0.000),
                       (+100.000, +25.000),
                       (+100.000, +50.000),
                       (+100.000, +75.000),
                       (+100.000,+100.000),
                       (+100.000,+125.000),
                       (+125.000,-200.000),
                       (+125.000,-175.000),
                       (+125.000,-150.000),
                       (+125.000,-125.000),
                       (+125.000,-100.000),
                       (+125.000, -75.000),
                       (+125.000, -50.000),
                       (+125.000, -25.000),
                       (+125.000,  +0.000),
                       (+125.000, +25.000),
                       (+125.000, +50.000),
                       (+125.000, +75.000),
                       (+125.000,+100.000),
                       (+125.000,+125.000),
                       (+150.000,-200.000),
                       (+150.000,-175.000),
                       (+150.000,-150.000),
                       (+150.000,-125.000),
                       (+150.000,-100.000),
                       (+150.000, -75.000),
                       (+150.000, -50.000),
                       (+150.000, -25.000),
                       (+150.000,  +0.000),
                       (+150.000, +25.000),
                       (+150.000, +50.000),
                       (+150.000, +75.000),
                       (+150.000,+100.000),
                       (+150.000,+125.000),
                       (+175.000,-175.000),
                       (+175.000,-150.000),
                       (+175.000,-125.000),
                       (+175.000,-100.000),
                       (+175.000, -75.000),
                       (+175.000, -50.000),
                       (+175.000, -25.000),
                       (+175.000,  +0.000),
                       (+175.000, +25.000),
                       (+175.000, +50.000),
                       (+175.000, +75.000),
                       (+175.000,+100.000),
                       (+175.000,+125.000),
                       (+200.000,-150.000),
                       (+200.000,-125.000),
                       (+200.000,-100.000),
                       (+200.000, -75.000),
                       (+200.000, -50.000),
                       (+200.000, -25.000),
                       (+200.000,  +0.000),
                       (+200.000, +25.000),
                       (+200.000, +50.000),
                       (+200.000, +75.000),
                       (+200.000,+100.000),
                       (+200.000,+125.000),
                       (+225.000,-100.000),
                       (+225.000, -75.000),
                       (+225.000, -50.000),
                       (+225.000, -25.000),
                       (+225.000,  +0.000),
                       (+225.000, +25.000),
                       (+225.000, +50.000),
                       (+225.000, +75.000),
                       (+225.000,+100.000),
                       (+225.000,+125.000),
                       (+250.000, -50.000),
                       (+250.000, -25.000),
                       (+250.000,  +0.000),
                       (+250.000, +25.000),
                       (+250.000, +50.000)])


