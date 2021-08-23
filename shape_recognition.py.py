import sys
import cv2
import numpy as np
import time
import imutils

def find_circles(frame, mask):

    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    center = None

    # Only proceed if at least one contour was found
    if len(contours) > 0:
        # Find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)       #Finds center point
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # Only proceed if the radius is greater than a minimum value
        if radius > 10:
            # Draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
            	(0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 0), -1)


    return center

def find_bounding_box(frame, mask, pts):

    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    center = None
    initBB = None

    # Only proceed if at least one contour was found
    if len(contours) > 0:
        # Find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(contours, key=cv2.contourArea)
        startX, startY, width, height = cv2.boundingRect(c)

        # Turn string to integer
        startX = int(startX)
        startY = int(startY)
        width = int(width)
        height = int(height)

        endX = startX + width
        endY = startY + height

        initBB = (startX, startY, width, height)
        center = (int((startX + endX) / 2), int((startY + endY) / 2))
        pts.appendleft(center)

        # draw the predicted bounding box and class label on the image
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(frame, 'Unidentified Object', (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 255, 0), 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY),
            (0, 255, 0), 2)

        # if width > 15 | height > 15:

        #     center = ((startX + endX) / 2, (startY + endY) / 2)

        #     # draw the predicted bounding box and class label on the image
        #     y = startY - 10 if startY - 10 > 10 else startY + 10
        #     cv2.putText(frame, 'Unidentified Object', (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
        #         0.65, (0, 255, 0), 2)
        #     cv2.rectangle(frame, (startX, startY), (endX, endY),
        #         (0, 255, 0), 2)  


    return center, initBB