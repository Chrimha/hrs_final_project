#!/usr/bin/env python
from __future__ import print_function
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import os
import random
import time

if __name__ == "__main__":

    #cv_image_color = cv2.imread('./templateImg.jpg', 1)
    lower_yellow = np.array([22, 93, 0])
    upper_yellow = np.array([45, 255, 255])

    #img_hsv = cv2.cvtColor(cv_image_color, cv2.COLOR_BGR2HSV)
    mask_yellow = cv2.inRange(HSV_image, lower_yellow, upper_yellow)

    (rows, cols, channels) = cv_image_color.shape
    if cols > 60 and rows > 60:
        cv2.circle(cv_image_color, (50, 50), 10, 255)

    #cv2.imshow("Input Image", cv_image_color)
    #cv2.waitKey()

    #cv2.imshow("Img HSV", img_hsv)
    #cv2.waitKey()

    # Show only yellow pixels
    #cv2.imshow("Color  Extraction", mask_yellow)
    #cv2.waitKey()

    # Image kernel
    kernel_er = np.ones((5, 5), 'uint8')

    # Erode + Dilate picture to form proper Blobs
    erode_img = cv2.erode(mask_yellow, kernel_er)
    kernel_dil = np.ones((3, 3), 'uint8')
    dilate_img = cv2.dilate(erode_img, kernel_dil, iterations=10)

    # Find Blobs using contours
    # Invert colors in image so no black blobs are detected
    threshold = cv2.threshold(dilate_img, 200, 255, cv2.THRESH_BINARY)[1]

    # Find contours in picture
    (cnts, _) = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find largest contour
    if cnts != []:
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(dilate_img, c, -1, (126, 255, 255), 3)
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # draw the contour and center of the shape on the image
        cv2.circle(dilate_img, (cX, cY), 3, (126, 255, 255), -1)
        cv2.putText(dilate_img, "center", (cX - 20, cY - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.2, (126, 255, 255), 2)

    cnt = c
    approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
    # draws boundary of contours.
    cv2.drawContours(dilate_img, [approx], 0, (0, 0, 255), 5)
    # Used to flatted the array containing
    # the co-ordinates of the vertices.
    n = approx.ravel()
    i = 0

    x = [0] * 100
    y = [0] * 100

    for j in n:
        if (i % 2 == 0):
            x[i] = n[i]
            y[i] = n[i + 1]
            # String containing the co-ordinates.
            string = str(x[i]) + " " + str(y[i])
            if (i == 0):
                # text on topmost co-ordinate.
                cv2.putText(dilate_img, "Arrow tip", (x[i], y[i]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (145, 200, 200))
            else:
                # text on remaining co-ordinates.
                cv2.putText(dilate_img, string, (x[i], y[i]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (145, 200, 200))
        i = i + 1

    cv2.imshow("Blob extraaction", dilate_img)
    cv2.waitKey()

    x = np.copy(x)
    y = np.copy(y)

    x_min = np.min(x[np.nonzero(x)])
    x_max = np.max(x[np.nonzero(x)])
    y_max = np.max(y[np.nonzero(y)])
    y_min = np.min(y[np.nonzero(y)])

    crop_img = cv_image_color[y_min:y_max, x_min:x_max]
    cv2.imshow("cropped", crop_img)
    cv2.waitKey()
