import sys
import cv2
import numpy as np
import time
import imutils
from matplotlib import pyplot as plt

# Functions
import HSV_filter as hsv
import shape_recognition as shape
import triangulation as tri
import no_mask
import predict_real_time as pre
#import calibration as calib

# Open both cameras
frame_left = cv2.imread('/home/daniel/Escritorio/MSc AAI/IRP/blender projects/RANGER/Render/cube/Stereo Vision/cube_stereo_006_left.png')
frame_right = cv2.imread('/home/daniel/Escritorio/MSc AAI/IRP/blender projects/RANGER/Render/cube/Stereo Vision/cube_stereo_006_right.png')

B = 10.9878               #Distance between the cameras [m]
f = 25               #Camera lense's focal length [mm]
alpha = 71.5078        #Camera field of view in the horizontal plane [degrees]


################## CALIBRATION #########################################################

#frame_right, frame_left = calib.undistorted(frame_right, frame_left)

########################################################################################

# # APPLYING HSV-FILTER:
# mask_right = hsv.add_HSV_filter(frame_right)
# mask_left = hsv.add_HSV_filter(frame_left)

# # Result-frames after applying HSV-filter mask
# res_right = cv2.bitwise_and(frame_right, frame_right, mask=mask_right)
# res_left = cv2.bitwise_and(frame_left, frame_left, mask=mask_left) 

# # APPLYING SHAPE RECOGNITION:
# circles_right = shape.find_circles(frame_right, mask_right)
# circles_left  = shape.find_circles(frame_left, mask_left)

# # APPLYING SHAPE RECOGNITION WITHOUT THE MASK:
# circles_right = no_mask.find_circles(frame_right)
# circles_left  = no_mask.find_circles(frame_left)

# Hough Transforms can be used aswell or some neural network to do object detection
bounding_box_right = pre.find_bounding_box(frame_right)
bounding_box_left  = pre.find_bounding_box(frame_left)

        ################## CALCULATING BALL DEPTH #########################################################

# If no ball can be caught in one camera show text "TRACKING LOST"
if np.all(circles_right) == None or np.all(circles_left) == None:
    cv2.putText(frame_right, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
    cv2.putText(frame_left, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

else:
# Function to calculate depth of object. Outputs vector of all depths in case of several balls.
# All formulas used to find depth is in video presentaion
    depth = tri.find_depth(circles_right, circles_left, frame_right, frame_left, B, f, alpha)

    # cv2.putText(frame_right, "TRACKING", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124,252,0),2)
    # cv2.putText(frame_left, "TRACKING", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124,252,0),2)
    # cv2.putText(frame_right, "Distance: " + str(round(depth,3)), (200,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124,252,0),2)
    # cv2.putText(frame_left, "Distance: " + str(round(depth,3)), (200,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124,252,0),2)
    # Multiply computer value with 205.8 to get real-life depth in [cm]. The factor was found manually.
    cv2.putText(frame_right, "{0:.2f}m".format(depth),
        (frame_right.shape[1] - 400, frame_right.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
        2.0, (0, 255, 0), 3)
    cv2.putText(frame_left, "{0:.2f}m".format(depth),
        (frame_left.shape[1] - 400, frame_left.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
        2.0, (0, 255, 0), 3)
    print("Depth: ", depth)                                            

# Resize the frames
frame_right = cv2.resize(frame_right, (960, 540))
frame_left = cv2.resize(frame_left, (960, 540))
# mask_right = cv2.resize(mask_right, (960, 540))
# mask_left = cv2.resize(mask_left, (960, 540))

# Show the frames
# cv2.imshow("frame right", frame_right) 
# cv2.imshow("frame left", frame_left)
# # cv2.imshow("mask right", mask_right) 
# # cv2.imshow("mask left", mask_left)
# cv2.waitKey(0)

# cv2.destroyAllWindows()

cv2.imwrite('/home/daniel/Escritorio/MSc AAI/IRP/Results/Stereo Vision/cube_stereo_006_left.png', frame_left)
cv2.imwrite('/home/daniel/Escritorio/MSc AAI/IRP/Results/Stereo Vision/cube_stereo_006_right.png', frame_right)