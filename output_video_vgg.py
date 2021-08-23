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

# Save image paths
videoPath_left = '/home/daniel/Escritorio/MSc AAI/IRP/blender projects/RANGER/Render/videos/cassini001_L.mkv'
videoPath_right = '/home/daniel/Escritorio/MSc AAI/IRP/blender projects/RANGER/Render/videos/cassini001_R.mkv'

# Open both cameras
cap_right = cv2.VideoCapture(videoPath_right)                    
cap_left =  cv2.VideoCapture(videoPath_left)
frame_rate = 120    #Camera frame rate (maximum at 120 fps)

B = 5.13535               #Distance between the cameras [m]
f = 200               #Camera lense's focal length [mm]
alpha = 10.2855        #Camera field of view in the horizontal plane [degrees]


#Initial values
count = -1
img_array_right = []
img_array_left = []

while(True):
    count += 1

    ret_right, frame_right = cap_right.read()
    ret_left, frame_left = cap_left.read()

################## CALIBRATION #########################################################

    #frame_right, frame_left = calib.undistorted(frame_right, frame_left)

########################################################################################

    # If cannot catch any frame, break
    if ret_right==False or ret_left==False:                    
        break

    else:
        # # APPLYING HSV-FILTER:
        # mask_right = hsv.add_HSV_filter(frame_right, 1)
        # mask_left = hsv.add_HSV_filter(frame_left, 0)

        # # Result-frames after applying HSV-filter mask
        # res_right = cv2.bitwise_and(frame_right, frame_right, mask=mask_right)
        # res_left = cv2.bitwise_and(frame_left, frame_left, mask=mask_left) 

        # # APPLYING SHAPE RECOGNITION:
        # circles_right = shape.find_circles(frame_right, mask_right)
        # circles_left  = shape.find_circles(frame_left, mask_left)

        # # APPLYING SHAPE RECOGNITION WITHOUT THE MASK:
        # bounding_box_right = no_mask.find_bounding_box(frame_right)
        # bounding_box_left  = no_mask.find_bounding_box(frame_left)

        # Hough Transforms can be used aswell or some neural network to do object detection
        bounding_box_right = pre.find_bounding_box(frame_right)
        bounding_box_left  = pre.find_bounding_box(frame_left)

        ################## CALCULATING BALL DEPTH #########################################################

        # If no ball can be caught in one camera show text "TRACKING LOST"
        if np.all(bounding_box_right) == None or np.all(bounding_box_left) == None:
            cv2.putText(frame_right, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
            cv2.putText(frame_left, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

        else:
            # Function to calculate depth of object. Outputs vector of all depths in case of several balls.
            # All formulas used to find depth is in video presentaion
            depth = tri.find_depth(bounding_box_right, bounding_box_left, frame_right, frame_left, B, f, alpha)

            cv2.putText(frame_right, "TRACKING", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
            cv2.putText(frame_left, "TRACKING", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
            cv2.putText(frame_right, "Distance: " + str(round(depth,3)), (200,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
            cv2.putText(frame_left, "Distance: " + str(round(depth,3)), (200,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
            # Multiply computer value with 205.8 to get real-life depth in [cm]. The factor was found manually.
            print("Depth: ", depth)                                            


        height, width, layers = frame_right.shape
        size_right = (width,height)
        img_array_right.append(frame_right)

        height, width, layers = frame_left.shape
        size_left = (width,height)
        img_array_left.append(frame_left)


out_right = cv2.VideoWriter('output_videos/project_R.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size_right)
 
for i in range(len(img_array_right)):
    out_right.write(img_array_right[i])
out_right.release()


out_left = cv2.VideoWriter('output_videos/project_L.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size_left)
 
for i in range(len(img_array_left)):
    out_left.write(img_array_left[i])
out_left.release()

# Release and destroy all windows before termination
cap_right.release()
cap_left.release()

cv2.destroyAllWindows()