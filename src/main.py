#!/usr/bin/env python
from __future__ import print_function
import rospy
import cv2
from sensor_msgs.msg import Image
from sensor_msgs.msg import Range
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import naoqi_bridge_msgs.msg
import std_msgs.msg
import std_srvs.srv
import os


# The state of the traffic light (False = red,  True = green)
traffic_light = False


def initialize_ros():
    rospy.init_node('guidenao', anonymous=True)

    # Subscibers
    image_sub = rospy.Subscriber('/nao_robot/camera/top/camera/image_raw', Image, callback_image, queue_size=3)
    # angles = rospy.Subscriber('joint_states', JointState, callback_angles, queue_size=3)
    tactile_sub = rospy.Subscriber('/tactile_touch', naoqi_bridge_msgs.msg.HeadTouch, callback_tactile, queue_size=3)
    recog_sub = rospy.Subscriber('/word_recognized', naoqi_bridge_msgs.msg.WordRecognized, callback_recog, queue_size=3)
    footContact_sub = rospy.Subscriber("/foot_contact", std_msgs.msg.Bool, callback_footContact, queue_size=3)

    # Publisher
    # speech_pub = rospy.Publisher('/speech_action/goal', naoqi_bridge_msgs.SpeechWithFeedbackActionGoal, queue_size=3)
    # voc_params_pub = rospy.Publisher('/speech_vocabulary_action/goal',
    # naoqi_bridge_msgs.SetSpeechVocabularyActionGoal, queue_size=3)
    # walk_pub = rospy.Publisher('/cmd_pose', geometry_msgs.Pose2D, queue_size=3)

    # Services
    rospy.wait_for_service('/start_recognition')
    recognition_start_srv = rospy.ServiceProxy('/start_recognition', std_srvs.srv.Empty())
    rospy.wait_for_service('/stop_recognition')
    recognition_stop_srv = rospy.ServiceProxy('/stop_recognition', std_srvs.srv.Empty())
    rospy.wait_for_service('/stop_walk_srv')
    stop_walk_srv = rospy.ServiceProxy('/stop_walk_srv', std_srvs.srv.Empty())


def callback_image(data):
    bridge = CvBridge()
    cv_image_color = bridge.imgmsg_to_cv2(data, "bgr8")

    (rows_b, cols_b, channels_b) = cv_image_color.shape
    if cols_b > 60 and rows_b > 60:
        cv2.circle(cv_image_color, (50, 50), 10, 255)

    # canny_edges(cv_image_color)

    # Convert into HSV
    HSV_image = cv2.cvtColor(cv_image_color, cv2.COLOR_BGR2HSV)

    # Split image
    h, s, v = cv2.split(HSV_image)

    # Mask out low saturated pixels
    th, dst = cv2.threshold(s, 200, 255, cv2.THRESH_BINARY)

    # Erode + Dilate picture to form proper Blobs
    kernel_er = np.ones((4, 4), 'uint8')
    erode_img = cv2.erode(dst, kernel_er)
    kernel_dil = np.ones((3, 3), 'uint8')
    dilate_img = cv2.dilate(erode_img, kernel_dil, iterations=10)

    # Find Blobs using contours
    # Invert colors in image so no black blobs are detected
    threshold = cv2.threshold(dilate_img, 235, 255, cv2.THRESH_BINARY)[1]

    # Find contours in picture
    (_, cnts, _) = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find largest contour
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        areas = cv2.contourArea(c)
        if areas > 1000:
            cv2.drawContours(dilate_img, c, -1, (126, 255, 255), 3)

            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # draw the contour and center of the shape on the image
            cv2.circle(cv_image_color, (cX, cY), 7, (126, 255, 255), -1)
            cv2.putText(cv_image_color, "center", (cX - 20, cY - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (126, 255, 255), 2)

            # Detect color of traffic light
            mask = np.zeros(HSV_image.shape[:2], np.uint8)
            cv2.fillPoly(mask, pts=[c], color=(255, 255, 255))
            mean = cv2.mean(HSV_image, mask=mask)
            if 0 <= mean[0] <= 30:
                print("rot")
                traffic_light = False
            elif 40 <= mean[0] <= 90:
                print("green")
                traffic_light = True

    # Show images
    cv2.imshow("Camera", cv_image_color)

    cv2.waitKey(3)


def canny_edges(cv_image_color):
    edges = cv2.Canny(cv_image_color, 100, 200)
    (_, cnts, _) = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        rectangles = []
        for cnt in cnts:
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                rectangles.append(cnt)

        if cnts:
            c = max(rectangles, key=cv2.contourArea)

            # draw the contour and center of the shape on the image
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(edges, (x, y), (x + w, y + h), (100, 100, 100), 2)
            cv2.imshow("Canny", edges)


def callback_angles(data):
    print("callback_angles")


def callback_tactile(data):
    print("callback_tactile")


def callback_recog(data):
    print("callback_recog")


def callback_footContact(data):
    print("callback_footContact")


if __name__ == "__main__":
    cv_image_color = cv2.imwread('templateImg.jpg')

    callback_image(cv_image_color)

    # initialize_ros()
    # rospy.spin()
