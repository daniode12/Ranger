import sys
import cv2
import numpy as np
import time
import math


def find_depth(center_right, center_left, frame_right, frame_left, baseline,f, alpha):

    # CONVERT FOCAL LENGTH f FROM [mm] TO [pixel]:
    height_right, width_right, depth_right = frame_right.shape
    height_left, width_left, depth_left = frame_left.shape

    if width_right == width_left:
        f_pixel = (width_right * 0.5) / np.tan(alpha * 0.5 * np.pi/180)

    else:
        print('Left and right camera frames do not have the same pixel width')

    x_right = center_right[0] - width_right/2
    x_left = center_left[0] - width_left/2

    # CALCULATE THE DISPARITY:
    disparity = abs(x_left-x_right)      #Displacement between left and right frames [pixels]

    # CALCULATE DEPTH z:
    zDepth = (baseline*f_pixel)/disparity             #Depth in [m]

    # CALCULATE x COORDINATE:
    xCoordinate = (x_right*zDepth)/f_pixel        #Represented in the right camera. The purpose is to know the relationship between pixels and meters

    # CALCULATE y COORDINATE:
    y_right = center_right[1] - height_right/2
    yCoordinate = (y_right*zDepth)/f_pixel

    # CALCULARE distance:
    distance = math.sqrt((xCoordinate*xCoordinate) + (yCoordinate*yCoordinate) + (zDepth*zDepth))

    # factor to convert pixels to meters
    pixels2meters = zDepth/f_pixel

    return distance, xCoordinate, yCoordinate, zDepth, pixels2meters