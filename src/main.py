#!/usr/bin/env python
from __future__ import print_function
import rospy
import cv2
from sensor_msgs.msg import Image
from sensor_msgs.msg import Range
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import naoqi_bridge_msgs.msg
from naoqi import ALProxy
import std_msgs.msg
import std_srvs.srv
import os
import random
import time


robot = True
if robot:
    motion = ALProxy("ALMotion", "nao.local", 9559)
    speech = ALProxy("ALTextToSpeech", "nao.local", 9559)
    posture = ALProxy("ALRobotPosture", "nao.local", 9559)

# The state of the traffic light (False = red,  True = green)
traffic_light = False
contact = False
pose = False
end = False


def initialize_ros():
    rospy.init_node('guidenao', anonymous=True)

    # Subscibers
    image_sub = rospy.Subscriber('/nao_robot/camera/top/camera/image_raw', Image, callback_image, queue_size=3)
    # angles = rospy.Subscriber('joint_states', JointState, callback_angles, queue_size=3)
    tactile_sub = rospy.Subscriber('/tactile_touch', naoqi_bridge_msgs.msg.HeadTouch, callback_tactile, queue_size=3)
    recog_sub = rospy.Subscriber('/word_recognized', naoqi_bridge_msgs.msg.WordRecognized, callback_recog, queue_size=3)
    footContact_sub = rospy.Subscriber("/foot_contact", std_msgs.msg.Bool, callback_footcontact, queue_size=3)

    # Publisher
    speech_pub = rospy.Publisher('/speech_action/goal', naoqi_bridge_msgs.msg.SpeechWithFeedbackActionGoal, queue_size=3)
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

    speech.say("Hello, how are you?")


def initialize_motion():
    global pose
    for i in range(5):
        motion.setStiffnesses("Body", 1.0)

    posture.goToPosture("StandInit", 1.0)
    pose = True
    speech.say("lets go!")


def callback_image(data):
    global traffic_light
    bridge = CvBridge()
    cv_image_color = bridge.imgmsg_to_cv2(data, "bgr8")

    cv2.imwrite('arucoTemp4.jpg', cv_image_color)

    (rows_b, cols_b, channels_b) = cv_image_color.shape

    # canny_edges(cv_image_color)

    # Convert into HSV
    HSV_image = cv2.cvtColor(cv_image_color, cv2.COLOR_BGR2HSV)

    cv_image_color = check_color(HSV_image, cv_image_color)
    cv_image_color, tvec = detect_aruco(cv_image_color)

    print(tvec)

    # Show images
    cv2.imshow("Camera", cv_image_color)

    cv2.waitKey(3)


def detect_aruco(cv_image_color2):
    # Create numpy arrays containing dist. coeff. and cam. coeff. for estimatePoseSingleMarkers and Drawaxis
    distortion_coefficients = np.zeros((1, 5, 1), dtype="float")
    distortion_coefficients[0, 0] = -0.066494
    distortion_coefficients[0, 1] = 0.095481
    distortion_coefficients[0, 2] = -0.000279
    distortion_coefficients[0, 3] = 0.002292
    distortion_coefficients[0, 4] = 0.000000
    camera_coefficients = np.zeros((3, 3, 1), dtype="float")
    camera_coefficients[0, 0] = 551.543059
    camera_coefficients[0, 1] = 0.000000
    camera_coefficients[0, 2] = 327.382898
    camera_coefficients[1, 0] = 0.000000
    camera_coefficients[1, 1] = 553.736023
    camera_coefficients[1, 2] = 225.026380
    camera_coefficients[2, 0] = 0.000000
    camera_coefficients[2, 1] = 0.000000
    camera_coefficients[2, 2] = 1.000000

    # Detect Aruco Markers
    gray = cv2.cvtColor(cv_image_color2, cv2.COLOR_BGR2GRAY)
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
    arucoParam = cv2.aruco.DetectorParameters_create()
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, arucoDict, parameters=arucoParam)

    tvecs = []

    if np.all(ids is not None):  # If there are markers found by detector
        for i in range(0, len(ids)):  # Iterate in markers
            # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 9, camera_coefficients,
                                                                           distortion_coefficients)
            (rvec - tvec).any()  # get rid of that nasty numpy value array error
            cv2.aruco.drawDetectedMarkers(cv_image_color2, corners)  # Draw A square around the markers
            cv2.aruco.drawAxis(cv_image_color2, camera_coefficients, distortion_coefficients, rvec, tvec,
                               0.01)  # Draw Axis
            tvecs.append(tvec)

    return cv_image_color2, tvecs


def check_color(HSV_image, cv_image_color):
    global traffic_light
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
    return cv_image_color


def canny_edges(cv_image_color2):
    edges = cv2.Canny(cv_image_color2, 100, 200)
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
    global end
    if data.button == 1:
        speech.say("Time for a break.")
        motion.rest()
    else:
        end = True


def callback_recog(data):
    print("callback_recog")


def callback_footcontact(data):
    global contact
    contact = data.data


def recover():
    global pose
    if not contact:
        pose = False
        speech.say("I lost my footing")
        for j in range(5):
            motion.setStiffnesses("Body", 0.0)
        time.sleep(10)
        speech.say("I will try to stand up")
        time.sleep(5)
        if not contact:
            posture.goToPosture("StandInit", 1.0)
            pose = True
    else:
        if not pose:
            speech.say("Thank you for helping me up!")
            posture.goToPosture("StandInit", 1.0)
            pose = True


def main_loop():
    while not end:
        recover()

    motion.rest()


if __name__ == "__main__":
    if not robot:
        # Without ros
        image = cv2.imread('./arucoTemp.jpg')
        callback_image(image)
        rospy.spin()
    else:
        initialize_ros()
        initialize_motion()

        motion.moveTo(0.5, 0, 0)

        main_loop()

        speech.say("Good bye!")