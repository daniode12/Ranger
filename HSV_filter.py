import sys
import cv2
import numpy as np
import time


def add_HSV_filter(frame):

    # Blurring the frame
    blur = cv2.GaussianBlur(frame,(5,5),0) 

    # Converting RGB to HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    l_b = np.array([0, 50, 50])
    u_b = np.array([180, 255, 255])

	# HSV-filter mask
    mask = cv2.inRange(hsv, l_b, u_b)


    # Morphological Operation - Opening - Erode followed by Dilate - Remove noise
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    return mask