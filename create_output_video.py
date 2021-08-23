from collections import deque
import sys
import cv2
import numpy as np
import time
import imutils
import math
from matplotlib import pyplot as plt
from imutils.video import FPS

# Functions
import HSV_filter as hsv
import shape_recognition as shape
import triangulation as tri
import no_mask
import predict_real_time as pre
#import calibration as calib

# Save image paths
videoPath_left = '/home/daniel/Escritorio/MSc AAI/IRP/blender projects/RANGER/Render/videos/cassini004_L.mkv'
videoPath_right = '/home/daniel/Escritorio/MSc AAI/IRP/blender projects/RANGER/Render/videos/cassini004_R.mkv'

# Open both cameras
cap_right = cv2.VideoCapture(videoPath_right)                    
cap_left =  cv2.VideoCapture(videoPath_left)
frame_rate = 120    #Camera frame rate (maximum at 120 fps)

B = 5.13535               #Distance between the cameras [m]
f = 50               #Camera lense's focal length [mm]
alpha = 39.5978        #Camera field of view in the horizontal plane [degrees]

# initialize the list of tracked points, the frame counter,
# and the coordinate deltas
pts = deque(maxlen=32)
pts_left = deque(maxlen=32)

coordinates = deque(maxlen=32)

(dX, dY, dZ) = (0, 0, 0)
velocity = 0
direction = ""

# initialize the object trackers
tracker_right = cv2.TrackerCSRT_create()
tracker_left = cv2.TrackerCSRT_create()

# initialize the bounding box coordinates of the object we are going to track
initBB_right = None
initBB_left = None
# initialize the counter and the FPS throughput estimator
counter = -1
counter2 = -1
fps = None

time.sleep(2.0)

#Initial values
img_array_right = []
img_array_left = []

while(True):
    counter += 1

    ret_right, frame_right = cap_right.read()
    ret_left, frame_left = cap_left.read()

################## CALIBRATION #########################################################

    #frame_right, frame_left = calib.undistorted(frame_right, frame_left)

########################################################################################

    # If cannot catch any frame, break
    if ret_right==False or ret_left==False:                    
        break

    else:
        # check to see if we are already tracking an object
        if initBB_right is not None and initBB_left is not None:  

            # grab the new bounding box coordinates of the object
            (success_right, box_right) = tracker_right.update(frame_right)
            # check to see if the tracking was a success
            if success_right:
                (x, y, w, h) = [int(v) for v in box_right]
                cv2.rectangle(frame_right, (x, y), (x + w, y + h),
                    (0, 255, 0), 2)

                # Turn string to integer
                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)

                center_right = (int(x+w/2), int(y+h/2))
                pts.appendleft(center_right)

            # grab the new bounding box coordinates of the object
            (success_left, box_left) = tracker_left.update(frame_left)
            # check to see if the tracking was a success
            if success_left:
                (x, y, w, h) = [int(v) for v in box_left]
                cv2.rectangle(frame_left, (x, y), (x + w, y + h),
                    (0, 255, 0), 2)

                # Turn string to integer
                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)

                center_left = (int(x+w/2), int(y+h/2))
                pts_left.appendleft(center_right)

            # update the FPS counter
            fps.update()
            fps.stop()

            velocity_ms = velocity*fps.fps()

            # show fps
            cv2.putText(frame_right, "FPS: {0:.2f}".format(fps.fps()), (50, frame_right.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

            #show velocity
            cv2.putText(frame_right, "Velocity: {0:.2f} m/s".format(velocity_ms), (50, 180), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

        else:
            # APPLYING HSV-FILTER:
            mask_right = hsv.add_HSV_filter(frame_right)
            mask_left = hsv.add_HSV_filter(frame_left)

            # # APPLYING SHAPE RECOGNITION:
            center_right, initBB_right = shape.find_bounding_box(frame_right, mask_right, pts)
            center_left, initBB_left = shape.find_bounding_box(frame_left, mask_left, pts_left)

            # start the FPS throughput estimator
            # start OpenCV object tracker using the supplied bounding box
            # coordinates, then start the FPS throughput estimator as well
            tracker_right.init(frame_right, initBB_right)
            tracker_left.init(frame_left, initBB_left)  
            fps = FPS().start()

        ################## CALCULATING BALL DEPTH #########################################################

        # If no ball can be caught in one camera show text "TRACKING LOST"
        if np.all(center_right) == None or np.all(center_left) == None:
            cv2.putText(frame_right, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
            cv2.putText(frame_left, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

        else:
            counter2 += 1
            # Function to calculate depth of object. Outputs vector of all depths in case of several balls.
            # All formulas used to find depth is in video presentaion
            distance, xCoordinate, yCoordinate, depth, pixels2meters = tri.find_depth(center_right, center_left, frame_right, frame_left, B, f, alpha)

            cv2.putText(frame_right, "TRACKING", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
            cv2.putText(frame_left, "TRACKING", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
            cv2.putText(frame_right, "Distance: " + str(round(distance,3))+" m", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
            # cv2.putText(frame_left, "Distance: " + str(round(distance,3))+" m", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
            # Multiply computer value with 205.8 to get real-life depth in [cm]. The factor was found manually.
            print("Depth: ", distance)                                            

            coordinates.appendleft((xCoordinate, yCoordinate, depth))

        # loop over the set of tracked points
        for i in np.arange(1, len(pts)):
            # if either of the tracked points are None, ignore
            # them
            if pts[i - 1] is None or pts[i] is None:
                continue
            # check to see if enough points have been accumulated in
            # the buffer
            if counter >= 10 and i == 1 and len(pts)==32:
                # compute the difference between the x and y
                # coordinates and re-initialize the direction
                # text variables
                dX = coordinates[i][0] - coordinates[-10][0]
                dY = coordinates[i][1] - coordinates[-10][1]
                dZ = coordinates[i][2] - coordinates[-10][2]

                velocity = (math.sqrt((dX*dX) + (dY*dY) + (dZ*dZ)))/22

                (dirX, dirY, dirZ) = ("", "", "")
                # ensure there is significant movement in the
                # x-direction
                if np.abs(dX) > 20:
                    dirX = "Right" if np.sign(dX) == 1 else "Left"
                # ensure there is significant movement in the
                # y-direction
                if np.abs(dY) > 20:
                    dirY = "Down" if np.sign(dY) == 1 else "Up"
                # ensure there is significant movement in the
                # z-direction
                if np.abs(dZ) > 20:
                    dirZ = "Inside" if np.sign(dZ) == 1 else "Outside"
                # handle when both directions are non-empty
                if dirX != "" and dirY != "" and dirZ !="":
                    direction = "{}-{}-{}".format(dirX, dirY, dirZ)
                # otherwise, only one direction is non-empty
                else:
                    direction = dirX if dirX != "" else dirY

            # otherwise, compute the thickness of the line and
            # draw the connecting lines
            thickness = int(np.sqrt(32 / float(i + 1)) * 2.5)
            cv2.line(frame_right, pts[i - 1], pts[i], (0, 0, 255), thickness)

        # Calculate the direction in degrees
        angleXY=math.degrees(math.atan2(dY,dX))
        if angleXY < 0:
            angleXY = 180 + 180-abs(angleXY)

        angleXZ=math.degrees(math.atan2(dZ,dX))
        if angleXZ < 0:
            angleXZ = 180 + 180-abs(angleXZ)

        # calculate size of the object
        if counter2 == 0:
            initial_pixels2meters = pixels2meters
        width = initBB_right[2] * initial_pixels2meters
        height = initBB_right[3] * initial_pixels2meters

        # show the coordinates
        cv2.putText(frame_right, "X: {0:.2f} m, Y: {1:.2f} m, Z: {2:.2f} m".format(xCoordinate, yCoordinate, depth), (50, 130), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2)

        # show the movement deltas and the direction of movement on
        # the frame
        # cv2.putText(frame_right, "angleXY: {0:.2f}, angleXZ: {1:.2f}".format(angleXY, angleXZ), (10, 300), cv2.FONT_HERSHEY_SIMPLEX,
        #     0.7, (0, 255, 0), 2)
        # cv2.putText(frame_right, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        #     0.7, (0, 255, 0), 2)
        cv2.putText(frame_right, "dx: {0:.2f} m, dy: {1:.2f} m, dz: {2:.2f} m".format(dX, dY, dZ),
            (50, 210), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2)
        # cv2.putText(frame_right, "Velocity: {0:.2f}".format(velocity), (10, 500), cv2.FONT_HERSHEY_SIMPLEX,
        #     0.7, (0, 255, 0), 2)
        
        #show size of the object
        cv2.putText(frame_right, "Width: {0:.2f} m, Height {0:.2f} m".format(width, height), (50, 260), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)                                         


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