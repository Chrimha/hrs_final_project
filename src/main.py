#!/usr/bin/env python
from __future__ import print_function

import math

import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import Range
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import std_msgs.msg
import std_srvs.srv
import os
import random
import time
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import threading
import tf
from rrt import RRT
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

import almath
import sys
from numpy.linalg import inv
import scipy
#from scipy.spatial.transform import Rotation as R
import math # Math library
import random

# Disable at home
import cv2
import naoqi_bridge_msgs.msg
from naoqi import ALProxy
import glob  # Used to get retrieve files that have a specified pattern

# Declaration of variables
robot_ip = "10.152.246.74"

#Real-time camera image
live_image = 0
robot = True
ros = True

# The state of the traffic light (False = red,  True = green)
traffic_light = False

contact = True
pose = False
end = False
obstacles = []

visual_pub = None
path_pub = None
path = []
world_frame_pub = None
world_frame_pos = []  # x, y, z
world_frame_rot = []
robot_pos = [0, 0, 0]  # x, y, theta
path_msg = None
marker_array = MarkerArray()
recalc_trajectory = False
old_num_obstacles = 1000000000
cal_im_counter = 0
world_search_counter = 0
gx = 0
gy = 0

world_marker_available = True

if robot:
    motion = ALProxy("ALMotion", robot_ip, 9559)
    speech = ALProxy("ALTextToSpeech", robot_ip, 9559)
    posture = ALProxy("ALRobotPosture", robot_ip, 9559)

def initialize_ros():
    global visual_pub
    global path_pub
    global world_frame_pub
    rospy.init_node('guidenao', anonymous=True)

    if robot:
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
        # rospy.wait_for_service('/start_recognition')
        # recognition_start_srv = rospy.ServiceProxy('/start_recognition', std_srvs.srv.Empty())
        # rospy.wait_for_service('/stop_recognition')
        # recognition_stop_srv = rospy.ServiceProxy('/stop_recognition', std_srvs.srv.Empty())
        # rospy.wait_for_service('/stop_walk_srv')
        # stop_walk_srv = rospy.ServiceProxy('/stop_walk_srv', std_srvs.srv.Empty())

        #speech.say("Hello, how are you?")

    # Publishers for visualisation in rviz
    path_pub = rospy.Publisher('/path', Path, queue_size=3)
    visual_pub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=3)
    world_frame_pub = tf.TransformBroadcaster()


def initialize_motion():
    global pose
    for i in range(5):
        motion.setStiffnesses("Body", 1.0)

    posture.goToPosture("StandInit", 1.0)
    pose = True
    #speech.say("lets go!")


def get_quaternion_from_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.

    Input
      :param roll: The roll (rotation around x-axis) angle in radians.
      :param pitch: The pitch (rotation around y-axis) angle in radians.
      :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
      :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return [qx, qy, qz, qw]


def callback_image(data):
    global traffic_light
    global live_image
    bridge = CvBridge()
    cv_image_color = bridge.imgmsg_to_cv2(data, "bgr8")

    # cv2.imwrite('arucoTemp4.jpg', cv_image_color)

    (rows_b, cols_b, channels_b) = cv_image_color.shape

    # canny_edges(cv_image_color)

    # Convert into HSV
    HSV_image = cv2.cvtColor(cv_image_color, cv2.COLOR_BGR2HSV)

    cv_image_color = check_color(HSV_image, cv_image_color)
    cv_image_color, tvec = detect_aruco(cv_image_color)

    # print(tvec)
    live_image = cv_image_color
    # Show images
    #cv2.imshow("Camera", cv_image_color)
    #cv2.waitKey(2)

    #global cal_im_counter
    #if cal_im_counter % 50 == 0 and cal_im_counter < 1500:
    #    filename = str(cal_im_counter) + "_cal.jpg"
    #    cv2.imwrite(filename, cv_image_color)
    #    print("saving image")

    #cal_im_counter = cal_im_counter + 1
    return


def measure_distance(crop_img):
    lower_yellow = np.array([10, 90, 220])
    upper_yellow = np.array([45, 255, 255])
    print("measure distance")
    pixel_width_1_5m = 35
    img_hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

    (rows, cols, channels) = crop_img.shape
    if cols > 60 and rows > 60:
        cv2.circle(crop_img, (50, 50), 10, 255)

    cv2.imshow("Img HSV 1", crop_img)
    cv2.waitKey()

    # Show only yellow pixels
    # cv2.imshow("Color  Extraction 1", mask_yellow)
    # cv2.waitKey()

    # Remove noise from cropped traffic light
    kernel = np.ones((2, 2), np.uint8)

    mask_yellow = cv2.erode(mask_yellow, kernel, iterations=1)
    mask_yellow = cv2.dilate(mask_yellow, kernel, iterations=1)

    cv2.imshow('Clean Image', mask_yellow)
    cv2.waitKey()

    height = mask_yellow.shape[0]
    width = mask_yellow.shape[1]

    num_of_white_pix = 0
    x_val_white = [0] * 1000000
    y_val_white = [0] * 1000000

    for x in range(mask_yellow.shape[1]):
        for y in range(mask_yellow.shape[0]):
            if mask_yellow[y][x]:
                x_val_white[num_of_white_pix] = x
                y_val_white[num_of_white_pix] = y
                num_of_white_pix = num_of_white_pix + 1

    x_val_white = np.copy(x_val_white)
    y_val_white = np.copy(y_val_white)

    x_min_mask = np.min(x_val_white[np.nonzero(x_val_white)])
    x_max_mask = np.max(x_val_white[np.nonzero(x_val_white)])
    y_max_mask = np.max(y_val_white[np.nonzero(y_val_white)])
    y_min_mask = np.min(y_val_white[np.nonzero(y_val_white)])

    print(x_max_mask - x_min_mask, "Width of traffic light")
    distance_to_robot = -0.05062*(x_max_mask - x_min_mask) + 3.269 - 0.2 #0,2 correction

    print("Traffic Light at (m): ", distance_to_robot)

    return distance_to_robot


def crop_img():

    global live_image
    cv_image_color = live_image
    lower_yellow = np.array([10, 90, 240])
    upper_yellow = np.array([45, 255, 255])

    #lower_yellow = np.array([10, 90, 220])
    #upper_yellow = np.array([45, 255, 255])

    #cv2.imshow("Img HSV2", cv_image_color)
    #cv2.waitKey()

    # lower_yellow = np.array([10, 140, 180])
    # upper_yellow = np.array([45, 255 , 255])

    # lower_yellow = np.array([10, 150, 180])
    # upper_yellow = np.array([45, 255 , 255])

    # lower_yellow = np.array([22, 93, 0])
    # upper_yellow = np.array([45, 255 , 255])
    img_hsv = cv2.cvtColor(cv_image_color, cv2.COLOR_BGR2HSV)
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

    (rows, cols, channels) = cv_image_color.shape
    if cols > 60 and rows > 60:
        cv2.circle(cv_image_color, (50, 50), 10, 255)
    print("now")

    # cv2.imshow("Input Image", cv_image_color)
    # cv2.waitKey()

    # cv2.imshow("Img HSV", img_hsv)
    # cv2.waitKey()

    # Show only yellow pixels

    #cv2.imshow("Color  Extraction", mask_yellow)
    #cv2.waitKey(2)

    # Image kernel
    kernel_er = np.ones((1, 1), 'uint8')

    # Erode + Dilate picture to form proper Blobs
    erode_img = cv2.erode(mask_yellow, kernel_er)
    kernel_dil = np.ones((3, 3), 'uint8')
    dilate_img = cv2.dilate(erode_img, kernel_dil, iterations=10)

    # Find Blobs using contours
    # Invert colors in image so no black blobs are detected
    threshold = cv2.threshold(dilate_img, 200, 255, cv2.THRESH_BINARY)[1]

    #cv2.imshow("threshold", threshold)
    #cv2.waitKey()
    # Find contours in picture
    (_, cnts, _) = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

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

    #cv2.imshow("Blob extraaction", dilate_img)
    #cv2.waitKey()

    x = np.copy(x)
    y = np.copy(y)
    borders = -5;
    x_min = np.min(x[np.nonzero(x)]) - borders
    x_max = np.max(x[np.nonzero(x)]) + borders
    y_max = np.max(y[np.nonzero(y)]) + borders
    y_min = np.min(y[np.nonzero(y)]) - borders

    crop_img = cv_image_color[y_min:y_max, x_min:x_max]
    # cv2.imshow("cropped", crop_img)
    # cv2.waitKey()

    # ToDo
    return measure_distance(crop_img), 0


def detect_aruco(cv_image_color2):
    global robot_pos
    global old_num_obstacles
    global recalc_trajectory
    global world_frame_pos
    global world_frame_rot
    global world_marker_available
    global world_search_counter
    # Create numpy arrays containing dist. coeff. and cam. coeff. for estimatePoseSingleMarkers and Drawaxis

    distortion_coefficients = np.zeros((1, 5, 1), dtype="float")
    distortion_coefficients[0, 0] = -0.0481869853715082
    distortion_coefficients[0, 1] = 0.0201858398559121
    distortion_coefficients[0, 2] = 0.0030362056699177
    distortion_coefficients[0, 2] = 0.0030362056699177
    distortion_coefficients[0, 3] = -0.00172241952442813
    distortion_coefficients[0, 4] = 0.000000
    camera_coefficients = np.zeros((3, 3, 1), dtype="float")
    camera_coefficients[0, 0] = 278.236008818534
    camera_coefficients[0, 1] = 0.000000
    camera_coefficients[0, 2] = 156.194471689706
    camera_coefficients[1, 0] = 0.000000
    camera_coefficients[1, 1] = 279.380102992049
    camera_coefficients[1, 2] = 126.007123836447
    camera_coefficients[2, 0] = 0.000000
    camera_coefficients[2, 1] = 0.000000
    camera_coefficients[2, 2] = 1.000000

    '''
    distortion_coefficients = np.zeros((1, 5, 1), dtype="float")
    distortion_coefficients[0, 0] = 0.17503185
    distortion_coefficients[0, 1] = -0.95341384
    distortion_coefficients[0, 2] = 0.00933572
    distortion_coefficients[0, 3] = -0.03518294
    distortion_coefficients[0, 4] = 0.84339923
    camera_coefficients = np.zeros((3, 3, 1), dtype="float")
    camera_coefficients[0, 0] = 212.10260868
    camera_coefficients[0, 1] = 0.000000
    camera_coefficients[0, 2] = 143.67715053
    camera_coefficients[1, 0] = 0.000000
    camera_coefficients[1, 1] = 203.93292893
    camera_coefficients[1, 2] = 121.06392618
    camera_coefficients[2, 0] = 0.000000
    camera_coefficients[2, 1] = 0.000000
    camera_coefficients[2, 2] = 1.000000
    '''

    # Detect Aruco Markers
    gray = cv2.cvtColor(cv_image_color2, cv2.COLOR_BGR2GRAY)
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
    arucoParam = cv2.aruco.DetectorParameters_create()
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, arucoDict, parameters=arucoParam)

    tvecs = []
    rvecs = []

    # If there are markers found by detector
    if np.all(ids is not None):
        if 5 in ids:
            world_marker_available = True
            i = np.where(ids == 5)

            rvec_old, tvec_old, markerPoints_old = cv2.aruco.estimatePoseSingleMarkers(corners[i[0][0]], 0.175, camera_coefficients,
                                                                           distortion_coefficients)
            tvec = tvec_old[0][0]

            if len(world_frame_pos) < 2:
                world_frame_pos = tvec
            robot_pos[0] = tvec[0]
            robot_pos[1] = -tvec[2]
            robot_pos[2] = 0

            # print(robot_pos)

            '''
            #print("camera", tvecs[i[0][0]][0],tvecs[i[0][0]][1],tvecs[i[0][0]][2])
            #print("Robot Position", getpositions(tvecs[i[0][0]][0], tvecs[i[0][0]][1], tvecs[i[0][0]][2]))
            #print("Found World Aruco")
            #print("rvec", rvecs[i[0][0]])
            #rotation_matrix = np.eye(4)
            #rotation_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvec[i][0]))[0]
            #r = R.from_matrix(rotation_matrix[0:3, 0:3])
            #quat = r.as_quat()
            #print("i",i)
            # Store the rotation information
            #rotation_matrix = np.eye(4)
            #rotation_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[i[0][0]]))[0]
            #r = R.from_dcm(rotation_matrix[0:3, 0:3])
            #quat = r.as_quat()

            # Quaternion format
            transform_rotation_x = quat[0]
            transform_rotation_y = quat[1]
            transform_rotation_z = quat[2]
            transform_rotation_w = quat[3]

            transform_translation_x = tvecs[i[0][0]][0]
            transform_translation_y = tvecs[i[0][0]][1]
            transform_translation_z = tvecs[i[0][0]][2]

            # Euler angle format in radians
            roll_x, pitch_y, yaw_z = euler_from_quaternion(transform_rotation_x,
                                                           transform_rotation_y,
                                                           transform_rotation_z,
                                                           transform_rotation_w)

            roll_x = math.degrees(roll_x)
            pitch_y = math.degrees(pitch_y)
            yaw_z = math.degrees(yaw_z)
            #print("transform_translation_x: {}".format(transform_translation_x))
            #print("transform_translation_y: {}".format(transform_translation_y))
            #print("transform_translation_z: {}".format(transform_translation_z))
            #print("roll_x: {}".format(roll_x))
            #print("pitch_y: {}".format(pitch_y))
            #print("yaw_z: {}".format(yaw_z))


            world_frame_rot = [roll_x,pitch_y,yaw_z]
            '''

            # print(world_frame_rot)
            # tvecs.pop(i[0][0])
            # rvecs.pop(i[0][0])
            ids = list(ids)
            ids.remove([5])
            # print(tvecs)
            world_search_counter = 0
        else:
            if world_search_counter > 10:
                world_marker_available = False
            else:
                world_search_counter += 1

        for i in range(0, len(ids)):  # Iterate in markers
            # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.087, camera_coefficients,
                                                                           distortion_coefficients)
            (rvec - tvec).any()  # get rid of that nasty numpy value array error
            cv2.aruco.drawDetectedMarkers(cv_image_color2, corners)  # Draw A square around the markers
            cv2.aruco.drawAxis(cv_image_color2, camera_coefficients, distortion_coefficients, rvec, tvec,
                               0.01)  # Draw Axis
            # rvec = tvec[0][0]
            tvec_pos = tvec[0][0]
            tvecs.append(tvec_pos)
            # rvecs.append(rvec)
            # tvec -> x(horizontal + 0.5), y (vertical) and z= (distance*2)
            # print(tvec)

        global obstacles
        if not obstacles and not robot_pos[0] == 0:
            for obstacle in tvecs:
                position = [0, 0, 0]
                position[0] = -obstacle[0]
                position[1] = obstacle[2]
                position[2] = 0
                print("create obstacle")
                # print(position)
                create_obstacle_marker(random.random(), 0.3, 0.3, position[0], position[1])

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
        #speech.say("Time for a break.")
        motion.rest()
    else:
        end = True


def callback_recog(data):
    print("callback_recog")


def callback_footcontact(data):
    global contact
    print(contact)
    contact = data.data


def calculate_trajectory(gy, gx, sx, sy):
    global obstacles
    global path_msg
    global recalc_trajectory
    # print(obstacles)

    # print("Calculating trajectory...")
    show_animation = False

    rrt = RRT(
        start=[sx, sy],
        goal=[gx, gy],
        max_iter=100,
        obstacle_list=obstacles
        )
    paths = []
    path_len = []
    path_lengths = []
    if show_animation:
        iterations = 1
    else:
        iterations = 500

    for i in range(iterations):
        path = rrt.planning(animation=show_animation)
        if path is not None:
            paths.append(path)
            path_len.append(len(path))
            path_lengths.append(calc_path_length(path))

    if path_lengths:
        path = paths[np.argmin(path_lengths)]

    # print(calc_path_length(path))
    # print(len(path))

    if path is None:
        print("Cannot find path")
    else:
        pass
        # print("Found path!!")

        # Draw final path
        if show_animation:
            rrt.draw_graph()
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
            plt.grid(True)
            plt.pause(0.01)
            plt.show()

    msg = Path()
    msg.header.frame_id = "world_frame"
    msg.header.stamp = rospy.Time.now()

    if path is not None:
        for wp in path:
            pos = PoseStamped()
            pos.pose.position.x = wp[0]
            pos.pose.position.y = wp[1]
            pos.pose.position.z = 0

            msg.poses.append(pos)
            path_msg = msg
        recalc_trajectory = False
        return path
    return None


def calc_path_length(path):
    x = 0
    for i in range(1, len(path)):
        x += np.linalg.norm(np.add(path[i], np.negative(path[i-1])))
    return x


def recover():
    global pose
    global contact

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
            contact = True
    else:
        if not pose:
            speech.say("Thank you for helping me up!")
            posture.goToPosture("StandInit", 1.0)
            pose = True


def camera_calibration():
    # Project: Camera Calibration Using Python and OpenCV
    # Date created: 12/19/2021
    # Python version: 3.8

    # Chessboard dimensions
    number_of_squares_X = 8  # Number of chessboard squares along the x-axis
    number_of_squares_Y = 6  # Number of chessboard squares along the y-axis
    nX = number_of_squares_X - 1  # Number of interior corners along x-axis
    nY = number_of_squares_Y - 1  # Number of interior corners along y-axis
    square_size = 0.03  # Size, in meters, of a square side

    # Set termination criteria. We stop either when an accuracy is reached or when
    # we have finished a certain number of iterations.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Define real world coordinates for points in the 3D coordinate frame
    # Object points are (0,0,0), (1,0,0), (2,0,0) ...., (5,8,0)
    object_points_3D = np.zeros((nX * nY, 3), np.float32)

    # These are the x and y coordinates
    object_points_3D[:, :2] = np.mgrid[0:nY, 0:nX].T.reshape(-1, 2)

    object_points_3D = object_points_3D * square_size

    # Store vectors of 3D points for all chessboard images (world coordinate frame)
    object_points = []

    # Store vectors of 2D points for all chessboard images (camera coordinate frame)
    image_points = []

    # Get the file path for images in the current directory
    images = glob.glob('*.jpg')
    print("amount img", len(images))
    # Go through each chessboard image, one by one
    for image_file in images:

        # Load the image
        print("loaded Image calibration")
        image = cv2.imread(image_file)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find the corners on the chessboard
        success, corners = cv2.findChessboardCorners(gray, (nY, nX), None)

        # If the corners are found by the algorithm, draw them
        if success == True:
            print("found chessboard")
            # Append object points
            object_points.append(object_points_3D)

            # Find more exact corner pixels
            corners_2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Append image points
            image_points.append(corners_2)

            # Draw the corners
            cv2.drawChessboardCorners(image, (nY, nX), corners_2, success)

            # Display the image. Used for testing.
            cv2.imshow("Image", image)

            # Display the window for a short period. Used for testing.
            cv2.waitKey(1000)

            # Perform camera calibration to return the camera matrix, distortion coefficients, rotation and translation vectors etc
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points,
                                                       image_points,
                                                       gray.shape[::-1],
                                                       None,
                                                       None)

    # Save parameters to a file
    cv_file = cv2.FileStorage('calibration_chessboard.yaml', cv2.FILE_STORAGE_WRITE)
    cv_file.write('K', mtx)
    cv_file.write('D', dist)
    cv_file.release()

    # Load the parameters from the saved file
    cv_file = cv2.FileStorage('calibration_chessboard.yaml', cv2.FILE_STORAGE_READ)
    mtx = cv_file.getNode('K').mat()
    dst = cv_file.getNode('D').mat()
    cv_file.release()

    # Display key parameter outputs of the camera calibration process
    print("Camera matrix:")
    print(mtx)

    print("\n Distortion coefficient:")
    print(dist)

    # Close all windows
    cv2.destroyAllWindows()


def main_loop():
    global marker_array
    global path_msg
    global end
    global world_frame_pub
    global robot_pos
    global recalc_trajectory
    global path
    global gx
    global gy

    path = calculate_trajectory(gx, gy, robot_pos[0], robot_pos[1])
    print("Trajectory calculation", gx, gy, robot_pos[0], robot_pos[1])
    if path and robot:
        path.reverse()
        thread_walking = threading.Thread(target=walk_path, args=[])
        thread_walking.start()

        # walk_path(path)
    else:
        end = True
    # motion.moveTo(1 * 0.83, 0, 0)
    try:
        while not end:
            if robot:
                recover()
                if robot_pos[1] > gx:
                    motion.move(0.0, 0.0, 0)
                    print("I have reached my goal")
                    print(gx, robot_pos[1])
                    end = True
                # print(robot_pos[1], gx)

            if marker_array:
                # ToDo: can lead to errors if obstacles are added between these three lines
                create_robot_marker()
                visual_pub.publish(marker_array)
                marker_array.markers.pop()

                path_pub.publish(path_msg)

                # ToDo: set world_frame to aruco. Once or everytime?
                # world_frame_pub.sendTransform((0.0, 0.0, 0.0), get_quaternion_from_euler(math.degrees(world_frame_rot[0]), math.degrees(world_frame_rot[1]), math.degrees(world_frame_rot[2])), rospy.Time.now(), "world_frame", "map")
                world_frame_pub.sendTransform((0.0, 0.0, 0.0), (0, 0, 0, 1), rospy.Time.now(), "world_frame", "map")
    except KeyboardInterrupt:
        print("Closing")

    if robot:
        motion.rest()


def walk_path():
    global world_marker_available
    global robot_pos
    global path
    global gx
    global gy

    print("walking")
    i = 1
    old_waypoint = []

    while i < len(path):
        if not world_marker_available:
            robot_pos[1] = old_waypoint[0]
            robot_pos[0] = old_waypoint[1]

        # print("robot position", robot_pos[1], robot_pos[0])
        waypoint = [path[i][1] - robot_pos[1], -(path[i][0] - robot_pos[0]), 0]
        # print("Waypoint", i, ": ", waypoint)
        # waypoint = [path[i][1] * 0.70 - robot_pos[1] * 0.7, -path[i][0] * 0.7 - robot_pos[0] * 0.7, 0]
        # print(path)
        # moveTo([forward,sidewards,0])
        # moveTo([1,2,0])
        # motion.moveTo([0.2*0.75,0.2,0])

        time_for_x = waypoint[0]/0.05
                                                #-0.049
        # motion.move(0.05, waypoint[1]/time_for_x, 0)
        rospy.sleep(time_for_x)
        motion.move(0.0, 0.0, 0)
        print("waiting for key")
        raw_input("Press Enter to continue...")
        print("key was pressed")
        calculate_trajectory(gx, gy, robot_pos[0], robot_pos[1])

        i += 1
        old_waypoint = waypoint
        create_marker(random.random(), 0.1, 0.1, old_waypoint[1], old_waypoint[0])
        # time.sleep(10)

        # path = calculate_trajectory(gx, gy, robot_pos[0], robot_pos[1])
    print("done walking")


def create_obstacle_marker(number, size_x, size_y, x, y):
    global robot_pos

    marker = Marker()
    marker.header.frame_id = "/world_frame"
    marker.header.stamp = rospy.Time.now()
    marker.type = 2
    marker.id = number
    marker.scale.x = size_x
    marker.scale.y = size_y
    marker.scale.z = 0.6
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    marker.pose.position.x = x + robot_pos[0]
    marker.pose.position.y = y + robot_pos[1]
    print(x, y, 0)
    print(robot_pos)
    print(marker.pose.position.x, marker.pose.position.y, 0)
    marker.pose.position.z = 0
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 5.0
    marker.pose.orientation.w = 1.0

    global marker_array
    global obstacles
    # ToDo: list of obstacles grows every iteration. Bug or feature?
    marker_array.markers.append(marker)
    obstacles.append([marker.pose.position.x, marker.pose.position.y, max(marker.scale.x, marker.scale.y)])


def create_marker(number, size_x, size_y, x, y):
    global robot_pos

    marker = Marker()
    marker.header.frame_id = "/world_frame"
    marker.header.stamp = rospy.Time.now()
    marker.type = 1
    marker.id = number
    marker.scale.x = size_x
    marker.scale.y = size_y
    marker.scale.z = 1.0
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    marker.pose.position.x = x + robot_pos[0]
    marker.pose.position.y = y + robot_pos[1]
    marker.pose.position.z = 0
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 5.0
    marker.pose.orientation.w = 1.0

    global marker_array
    global obstacles
    marker_array.markers.append(marker)


def create_robot_marker():
    global robot_pos

    marker = Marker()
    marker.header.frame_id = "/world_frame"
    marker.header.stamp = rospy.Time.now()
    marker.type = 2
    marker.id = -1
    marker.scale.x = 0.3
    marker.scale.y = 0.3
    marker.scale.z = 0.7
    marker.color.r = 0.0
    marker.color.g = 0.0
    marker.color.b = 1.0
    marker.color.a = 1.0
    marker.pose.position.x = robot_pos[0]
    marker.pose.position.y = robot_pos[1]
    marker.pose.position.z = 0
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = robot_pos[2]
    marker.pose.orientation.w = 1.0

    global marker_array
    marker_array.markers.append(marker)


def create_goal_marker():
    global gx
    global gy

    marker = Marker()
    marker.header.frame_id = "/world_frame"
    marker.header.stamp = rospy.Time.now()
    marker.type = 2
    marker.id = -2
    marker.scale.x = 0.2
    marker.scale.y = 0.2
    marker.scale.z = 0.2
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    marker.pose.position.x = gy
    marker.pose.position.y = gx
    marker.pose.position.z = 0
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = robot_pos[2]
    marker.pose.orientation.w = 1.0

    global marker_array
    marker_array.markers.append(marker)


def getpositions(Marker_x, Marker_y, Marker_z):

    frame = 0  # FRAME_TORSO = 0df
    useSensorValues = True

    # Homogeneous Coordinates of Aruco Marker in CameraBottom_optical-frame
    marker_cam = np.array([Marker_x, Marker_y, Marker_z, 1])

    # Homogenous transformation matrix from CameraBottom_optical-frame to CameraBottom-frame
    Hz = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    Hx = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

    Hom1 = Hz.dot(Hx)

    # Homogenous transformation matrix from CameraBottom-frame to Torso
    Mat1 = motion.getTransform("CameraBottom", frame, useSensorValues)
    Hom2 = np.zeros((4,4))
    for i in range(0, 4):
        for j in range(0, 4):
            Hom2[i][j] = Mat1[4*i + j]  # similar to np.reshape(Mat1,(4,4))

    # Homogenous transformation matrix from CameraBottom_optical-frame to Torso
    Hom3 = Hom2.dot(Hom1)

    # Homogeneous Coordinates of Aruco Marker in Torso-frame
    marker_torso = Hom3.dot(marker_cam)

    # response.markertorsox = marker_torso[0]
    # response.markertorsoy = marker_torso[1]
    # response.markertorsoz = marker_torso[2]
    return marker_torso[0], marker_torso[1], marker_torso[2]


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians

if __name__ == "__main__":
    # NAO_IP = "10.152.246.134" #  old robot
    NAO_IP = robot_ip
    if not robot and not ros:
        # Without ros
        print(cv2.__version__)
        image = cv2.imread('../arucoTemp.jpg')
        callback_image(image)
        rospy.spin()

    elif robot:
        initialize_ros()
        initialize_motion()
        rospy.sleep(3)
        # 2.3m : 21
        # 2m : 24
        # 1.5m : 33
        # 1.1m : 44
        gx, gy = crop_img()
        gx = gx + robot_pos[1]
        gy = gy + robot_pos[0]

        create_goal_marker()

        #motion.move(0.0,0.05 ,-0.049 )
        #rospy.spin()
        # 2.05412+-2.52185047506   0+0.488096629433
        # print(gx, robot_pos[1], gy,  robot_pos[0])
        main_loop()

        speech.say("bye!")

    elif ros and not robot:
        initialize_ros()
        create_obstacle_marker(0, 0.2, 0.1, 2, 3)  # (id, size_x, size_y, x, y)
        create_obstacle_marker(1, 0.2, 0.3, 8, 7)

        main_loop(3, 0)
