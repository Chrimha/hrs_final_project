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
import random
import cv2
import naoqi_bridge_msgs.msg
from naoqi import ALProxy
import glob

# Declaration of variables
robot_ip = "10.152.246.74"  # red robot
# robot_ip = "10.152.246.134"  # Orange
live_image = 0
robot = True
ros = True
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
obstacle_ids = [1023, 0, 21]
old_num_obstacles = 1000000000
cal_im_counter = 0
world_search_counter = 0
gx = 0
gy = 0
counter_lost_contact = 0
world_marker_available = True
customer = False
Array_roll_x = np.zeros(10)
Array_pitch_y = np.zeros(10)
Array_yaw_z = np.zeros(10)
status_button_1 = False
status_button_2 = False
status_button_3 = False
counter_lost_touch = 0
at_least_one_button = False
contact_lost = False
speech = True
goal_reached = False
Stop_Robot = False

# Create the ALProxys if the robot is present
if robot:
    # Create
    motion = ALProxy("ALMotion", robot_ip, 9559)
    speech = ALProxy("ALTextToSpeech", robot_ip, 9559)
    posture = ALProxy("ALRobotPosture", robot_ip, 9559)
    asr = ALProxy("ALSpeechRecognition", robot_ip, 9559)
    memProxy = ALProxy("ALMemory", robot_ip, 9559)


def initialize_ros():
    """
    Initialize the subscribers and publishers
    """
    global visual_pub
    global path_pub
    global world_frame_pub

    # Start the subscibers if the robot is present
    if robot:
        # Subscibers
        image_sub = rospy.Subscriber('/nao_robot/camera/top/camera/image_raw', Image, callback_image, queue_size=3)
        tactile_sub = rospy.Subscriber('/tactile_touch', naoqi_bridge_msgs.msg.HeadTouch, callback_tactile, queue_size=3)
        footContact_sub = rospy.Subscriber("/foot_contact", std_msgs.msg.Bool, callback_footcontact, queue_size=3)

        # Publisher
        speech_pub = rospy.Publisher('/speech_action/goal', naoqi_bridge_msgs.msg.SpeechWithFeedbackActionGoal, queue_size=3)

    # Publishers for visualisation in rviz
    path_pub = rospy.Publisher('/path', Path, queue_size=3)
    visual_pub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=3)
    world_frame_pub = tf.TransformBroadcaster()


def initialize_motion():
    """
    Prepare the robot for movement
    """
    global pose
    # Activate the motors of the robot
    for i in range(5):
        motion.setStiffnesses("Body", 1.0)
    # Go to the initial pose
    posture.goToPosture("StandInit", 1.0)
    pose = True


##########  VOICE   ##########
def voice():
    """
    Use voice control to detect a customer
    """
    # Pause the voice recognition if active
    asr.pause(True)

    # Remove the old data
    memProxy.removeData("WordRecognized")

    # Prepare the voice recognition
    asr.setLanguage("English")
    vocabulary = ["hand", "aps", "help", "cross"]
    asr.setVocabulary(vocabulary, False)
    asr.subscribe("Test_ASR")
    print('Speech recognition engine started')
    memProxy.subscribeToEvent('WordRecognized',robot_ip,'wordRecognized')
    words = ["Not a word"]
    # Start the voice recognition
    asr.pause(False)

    # Pause until the word help was detected
    while "help" not in words:
        print("Talk now")
        time.sleep(1)
        words = memProxy.getData("WordRecognized")
        if words is None:
            words = ["Not a word"]
    print(words[0])

    # Stop the voice recognition
    asr.unsubscribe("Test_ASR")
    asr.pause(True)


##########  VISION  ##########
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


def euler_from_quaternion(x, y, z, w):
    """
    Code from https://automaticaddison.com/how-to-perform-pose-estimation-using-an-aruco-marker/
    How to Perform Pose Estimation Using an ArUco Marker
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


def callback_image(data):
    """
    Callback from image_sub
    Transform the image to bgr and start the marker detection
    data: the data that is sent from ros
    """
    global live_image
    bridge = CvBridge()
    # Transform the image to bgr
    cv_image_color = bridge.imgmsg_to_cv2(data, "bgr8")
    (rows_b, cols_b, channels_b) = cv_image_color.shape
    # Detect the ArUco markers on the image
    cv_image_color, tvec = detect_aruco(cv_image_color)
    # Update the live_image
    live_image = cv_image_color
    return


def measure_distance(crop_img):
    """
    Measure the distance to the detected traffic light
    crop_img: the cropped image of the traffic light
    Output: the distance from the traffic light to the robot
    """
    # Set limits for the detected color
    lower_yellow = np.array([10, 90, 220])
    upper_yellow = np.array([45, 255, 255])

    # Prepare the images
    img_hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    (rows, cols, channels) = crop_img.shape
    if cols > 60 and rows > 60:
        cv2.circle(crop_img, (50, 50), 10, 255)

    cv2.imshow("Img HSV Meas", crop_img)
    cv2.waitKey(3)

    # Remove noise from cropped traffic light
    kernel = np.ones((2, 2), np.uint8)
    eroded_meas = cv2.erode(mask_yellow, kernel, iterations=1)
    dilated_meas = cv2.dilate(eroded_meas, kernel, iterations=1)
    cv2.imshow('Dilated Meas', dilated_meas)
    cv2.waitKey(3)

    # Get the height and width of the detected traffic light
    height = mask_yellow.shape[0]
    width = mask_yellow.shape[1]

    # Calculate number of pixels of the detected blob
    num_of_white_pix = 0
    x_val_white = [0] * 1000000
    y_val_white = [0] * 1000000
    for x in range(mask_yellow.shape[1]):
        for y in range(mask_yellow.shape[0]):
            if mask_yellow[y][x]:
                # Count the number of pixels
                x_val_white[num_of_white_pix] = x
                y_val_white[num_of_white_pix] = y
                num_of_white_pix = num_of_white_pix + 1
    x_val_white = np.copy(x_val_white)
    y_val_white = np.copy(y_val_white)

    # Get the min and max values of the mask
    x_min_mask = np.min(x_val_white[np.nonzero(x_val_white)])
    x_max_mask = np.max(x_val_white[np.nonzero(x_val_white)])
    y_max_mask = np.max(y_val_white[np.nonzero(y_val_white)])
    y_min_mask = np.min(y_val_white[np.nonzero(y_val_white)])

    # Calculate the distance using a formula that was found with calibration
    distance_to_robot = -0.05062*(x_max_mask - x_min_mask) + 3.269
    print("Traffic Light at (m): ", distance_to_robot)
    return distance_to_robot


def crop_img():
    """
    Crops the detected traffic light from the picture
    Output: measure_distance(crop_img): the measurred distance of the traffic light, the goal position in x dimension,
            0: the goal position in y dimension,
                set to zero because the same position of the robot on the other side of the street is the desired goal
            crop_img: the cropped image with the traffic light
    """
    global live_image
    cv_image_color = live_image

    # Set limits for the detected color
    lower_yellow = np.array([10, 90, 240])
    upper_yellow = np.array([45, 255, 255])

    # Prepare the images
    img_hsv = cv2.cvtColor(cv_image_color, cv2.COLOR_BGR2HSV)
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    (rows, cols, channels) = cv_image_color.shape
    if cols > 60 and rows > 60:
        cv2.circle(cv_image_color, (50, 50), 10, 255)

    # Image kernel
    kernel_er = np.ones((1, 1), 'uint8')

    # Erode and dilate picture to form proper Blobs
    erode_img = cv2.erode(mask_yellow, kernel_er)
    kernel_dil = np.ones((3, 3), 'uint8')
    dilate_img = cv2.dilate(erode_img, kernel_dil, iterations=10)

    # Find Blobs using contours
    # Apply threshold function to have only black or white pixels
    threshold = cv2.threshold(dilate_img, 200, 255, cv2.THRESH_BINARY)[1]

    # Find contours in picture
    (_, cnts, _) = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Find largest contour
    if cnts != []:
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(dilate_img, c, -1, (126, 255, 255), 3)
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

    cnt = c
    approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
    # Draws boundary of contours.
    cv2.drawContours(dilate_img, [approx], 0, (0, 0, 255), 5)


    # Used to flatted the array containing the co-ordinates of the vertices.
    n = approx.ravel()
    i = 0

    x = [0] * 100
    y = [0] * 100

    for j in n:
        if (i % 2 == 0):
            x[i] = n[i]
            y[i] = n[i + 1]
            # String containing the coordinates.
            string = str(x[i]) + " " + str(y[i])
            if (i == 0):
                # Text on topmost coordinate.
                cv2.putText(dilate_img, "Arrow tip", (x[i], y[i]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (145, 200, 200))
            else:
                # Text on remaining coordinates.
                cv2.putText(dilate_img, string, (x[i], y[i]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (145, 200, 200))
        i = i + 1

    # Set the borders of the crop
    x = np.copy(x)
    y = np.copy(y)
    borders = -5;
    x_min = np.min(x[np.nonzero(x)]) - borders
    x_max = np.max(x[np.nonzero(x)]) + borders
    y_max = np.max(y[np.nonzero(y)]) + borders
    y_min = np.min(y[np.nonzero(y)]) - borders

    # Crop the image
    crop_img = cv_image_color[y_min:y_max, x_min:x_max]

    return measure_distance(crop_img), 0 , crop_img


def detect_aruco(cv_image_color2):
    """
    Detect all ArUco markers in the picture
    cv_image_color2: the picture from the top camera
    Output: cv_image_color2: the image with the detected markers
            tvecs: the positions of the obstacle markers
    """
    global robot_pos
    global old_num_obstacles
    global world_frame_pos
    global world_frame_rot
    global world_marker_available
    global world_search_counter
    global Array_roll_x
    global Array_pitch_y
    global Array_yaw_z

    # Create variables
    j = 0
    Array_tvec_0_obst = np.zeros(11)
    Array_tvec_1_obst = np.zeros(11)
    Array_tvec_2_obst = np.zeros(11)

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

    # Detect ArUco Markers
    gray = cv2.cvtColor(cv_image_color2, cv2.COLOR_BGR2GRAY)
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
    arucoParam = cv2.aruco.DetectorParameters_create()
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, arucoDict, parameters=arucoParam)

    tvecs = []
    rvecs = []

    # If there are markers found by detector
    if np.all(ids is not None):
        # If the world coordinate marker is found
        if 5 in ids:
            # Print that the world marker is found
            if not world_marker_available:
                world_search_counter = 0
                print("World marker found")
            world_marker_available = True

            # Estimate the position of the marker
            i = np.where(ids == 5)
            rvec_old, tvec_old, markerPoints_old = cv2.aruco.estimatePoseSingleMarkers(corners[i[0][0]], 0.175 , camera_coefficients,
                                                                             distortion_coefficients)
            tvec = tvec_old[0][0]

            # Set the robot position
            if len(world_frame_pos) < 2:
                world_frame_pos = tvec
            robot_pos[0] = -tvec[0]
            robot_pos[1] = -tvec[2]

            # Code from https://automaticaddison.com/how-to-perform-pose-estimation-using-an-aruco-marker/
            # How to Perform Pose Estimation Using an ArUco Marker

            # Store the rotation information
            rotation_matrix = np.eye(4)
            rotation_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvec_old[0][0]))[0]
            r = R.from_dcm(rotation_matrix[0:3, 0:3])
            quat = r.as_quat()

            # Quaternion format
            transform_rotation_x = quat[0]
            transform_rotation_y = quat[1]
            transform_rotation_z = quat[2]
            transform_rotation_w = quat[3]

            # Euler angle format in radians
            roll_x, pitch_y, yaw_z = euler_from_quaternion(transform_rotation_x,
                                                           transform_rotation_y,
                                                           transform_rotation_z,
                                                           transform_rotation_w)

            roll_x = math.degrees(roll_x)
            pitch_y = math.degrees(pitch_y)
            yaw_z = math.degrees(yaw_z)

            # Update the arrays to calculate the median
            Array_roll_x = np.append(Array_roll_x, roll_x)
            Array_pitch_y = np.append(Array_pitch_y, pitch_y)
            Array_yaw_z = np.append(Array_yaw_z, yaw_z)
            Array_roll_x = np.delete(Array_roll_x, 0)
            Array_pitch_y = np.delete(Array_pitch_y, 0)
            Array_yaw_z = np.delete(Array_yaw_z, 0)

            # Get the rotation from the medians
            world_frame_rot = [np.percentile(Array_roll_x,50),np.percentile(Array_pitch_y,50),np.percentile(Array_yaw_z,50)]

            # Remove the world marker from the detected marker list
            ids = list(ids)
            ids.remove([5])
            corners.pop(i[0][0])

        # If the world marker is not found
        else:
            if world_search_counter > 60:
                if world_marker_available:
                    print("World marker not found")
                world_marker_available = False
            else:
                world_search_counter += 1

        # Estimate the positions of the other markers
        global obstacle_ids
        # Iterate in markers
        for i in range(0, len(ids)):
            # Remove already detected markers
            if ids[i] in obstacle_ids:
                corners.pop(i)
                break
            else:
                obstacle_ids.append(ids[i][0])

            # Estimate the position of the markers
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.087 , camera_coefficients,
                                                                      distortion_coefficients)

            # Get rid of that nasty numpy value array error
            (rvec - tvec).any()
            # Draw A square around the markers
            cv2.aruco.drawDetectedMarkers(cv_image_color2, corners)
            # Draw Axis
            cv2.aruco.drawAxis(cv_image_color2, camera_coefficients, distortion_coefficients, rvec, tvec,
                               0.01)
            tvec_pos = tvec[0][0]
            tvecs.append(tvec_pos)

        # Create the obstacles for the detected markers
        counter_marker = 3
        global obstacles
        if not robot_pos[0] == 0:
            for obstacle in tvecs:
                position = [0, 0, 0]
                position[0] = obstacle[0]
                position[1] = obstacle[2]
                position[2] = 0
                create_obstacle_marker(counter_marker, 0.35, 0.3, position[0], position[1])
                counter_marker = counter_marker + 4
    return cv_image_color2, tvecs


def check_color(cv_image_color):
    """
    Check the color of the traffic light
    cv_image_color: the cropped picture of the traffic light
    Output: the color of the traffic light
    """
    # Split image
    color =  None

    # Prepare the image
    HSV_image = cv2.cvtColor(cv_image_color, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(HSV_image)
    # Mask out low saturated pixels
    th, dst = cv2.threshold(s, 200, 255, cv2.THRESH_BINARY)
    # Erode and dilate picture to form proper Blobs
    kernel_er = np.ones((1, 1), 'uint8')
    erode_img = cv2.erode(dst, kernel_er)
    kernel_dil = np.ones((1, 1), 'uint8')
    dilate_img = cv2.dilate(erode_img, kernel_dil, iterations=10)

    # Find Blobs using contours
    # Apply threshold function to have only black or white pixels
    threshold = cv2.threshold(dilate_img, 200, 255, cv2.THRESH_BINARY)[1]

    # Find contours in picture
    (_, cnts, _) = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find largest contour
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        areas = cv2.contourArea(c)
        if areas > 1000:
            cv2.drawContours(dilate_img, c, -1, (126, 255, 255), 3)
            M = cv2.moments(c)

            # Detect color of traffic light
            mask = np.zeros(HSV_image.shape[:2], np.uint8)
            cv2.fillPoly(mask, pts=[c], color=(255, 255, 255))
            mean = cv2.mean(HSV_image, mask=mask)

            # Determine the color
            if 0 <= mean[0] <= 30:
                color = "red"
            elif 40 <= mean[0] <= 90:
                color = "green"
            else:
                color = "unknown"
    return color


########## PLANNING ##########
def calculate_path(gy, gx, sx, sy):
    """
    Find a path to the goal that avoids the obstacles
    gy: goal position in y
    gx: goal position in x
    sx: start position in x
    sy: start position in y
    Output: the calculated path or None
    """
    global obstacles
    global path_msg

    # Show the RRT animation
    show_animation = False

    # Initialize the RRT from "PythonRobotics"
    rrt = RRT(
        start=[sx, sy],
        goal=[gx, gy],
        max_iter=100,
        obstacle_list=obstacles
        )

    # Variables to compare the paths
    paths = []
    path_lengths = []

    # Calculate 500 paths and calculate their length
    if show_animation:
        iterations = 1
    else:
        iterations = 500
    for i in range(iterations):
        # Find a path
        path = rrt.planning(animation=show_animation)
        if path is not None:
            paths.append(path)
            path_lengths.append(calc_path_length(path))

    # Find the shortest path
    if path_lengths:
        path = paths[np.argmin(path_lengths)]

    # Draw the path if show_animation is True
    if path is None:
        print("Cannot find path")
    else:
        # Draw final path
        if show_animation:
            rrt.draw_graph()
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
            plt.grid(True)
            plt.pause(0.01)
            plt.show()

    # Send the path to rviz
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
        return path
    return None


def calc_path_length(new_path):
    """
    Calculate the length of the path to compare them
    """
    x = 0
    for i in range(1, len(new_path)):
        x += np.linalg.norm(np.add(new_path[i], np.negative(new_path[i - 1])))
    return x


def create_obstacle_marker(number, size_x, size_y, x, y):
    """
    Create the obstacle marker for rviz
    number: the id of the marker
    size_x: size in x dimension
    size_y: size in y dimension
    x: x position
    y: y position
    The marker is appended to the marker_array
    """
    global robot_pos

    # Create the obstacle marker object
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
    marker.pose.position.z = 0
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 5.0
    marker.pose.orientation.w = 1.0

    # Add the marker to the marker_array that is sent to rviz and the obstacle list for the path planning
    global marker_array
    global obstacles
    marker_array.markers.append(marker)
    obstacles.append([marker.pose.position.x, marker.pose.position.y, max(marker.scale.x, marker.scale.y)])


def create_robot_marker():
    """
    Create the robot marker for rviz
    It is appended to the marker_array
    """
    global robot_pos

    # Create the robot marker object
    marker = Marker()
    marker.header.frame_id = "/world_frame"
    marker.header.stamp = rospy.Time.now()
    marker.type = 2
    marker.id = -1
    marker.scale.x = 0.45
    marker.scale.y = 0.1
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

    # Add the marker to the marker_array that is sent to rviz
    global marker_array
    marker_array.markers.append(marker)


def create_goal_marker():
    """
    Create the goal marker for rviz
    It is appended to the marker_array
    """
    global gx
    global gy

    # Create the goal marker object
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

    # Add the marker to the marker_array that is sent to rviz
    global marker_array
    marker_array.markers.append(marker)


########## CONTROL  ##########
def walk_path():
    """
    Thread to move the robot
    Sets the variable contact_lost
    """
    global world_marker_available
    global robot_pos
    global path
    global gx
    global gy
    global contact_lost
    global end
    global speech
    i = 1
    old_waypoint = []

    # Move to the next waypoint as long as the robot has not reached the goal
    while i < len(path):
        # Stop moving if the contact to the customer is lost
        if contact_lost:
            motion.move(0.0, 0.0, 0)
            speech.say("Please come back, we are not there yet")

            # Tell the customer the location of the robot
            while contact_lost:
                speech.say("I'm here!")
                rospy.sleep(2)

            # Restart walking
            speech.say("Thanks for coming back, now lets go")
            rospy.sleep(4)
            pass

        # Assume the robot reached the waypoint, if the world marker cannot be found
        if not world_marker_available:
            robot_pos[1] = old_waypoint[0]
            robot_pos[0] = old_waypoint[1]

        # Move to the waypoint
        waypoint = [path[i][1] - robot_pos[1], -(path[i][0] - robot_pos[0]), 0]
        x_speed = 0.04  # should be 0,5
        time_for_x = waypoint[0] / x_speed
        motion.move(x_speed, waypoint[1]/(time_for_x * 0.4), 0.01)

        # Tell the customer the direction, in which they will move next
        if abs(waypoint[1]) < 0.1:
            direction = "."
        elif waypoint[1] < 0:
            direction = " and to the right."
        else:
            direction = " and to the left."
        if speech:
            speech.say("We will move froward" + direction)
        print("doing step", i)

        # Split delay to check if robot needs to stop
        for y in range(100):
            rospy.sleep(time_for_x/100)
            if contact_lost and not end and speech:
                motion.move(0.0, 0.0, 0)
                speech.say("Please come back, we are not there yet")
                while contact_lost:
                    speech.say("I'm here!")
                    rospy.sleep(2)
                speech.say("Thanks for coming back, now lets go")
                rospy.sleep(4)

            motion.move(x_speed, waypoint[1] / (time_for_x * 0.4), 0.01)

        # Stop the movement after the calculated time
        motion.move(0.0, 0.0, 0)

        # Prepare for the next iteration
        i += 1
        old_waypoint = waypoint

        print("I will wait")
        rospy.sleep(4)

        # Calculate a new path from the new position
        calculate_path(gx, gy, robot_pos[0], robot_pos[1])
    print("done walking")


def callback_footcontact(data):
    """
    Callback from footContact_sub
    Check if the robot has foot contact to the floor
    data: the data that is sent from ros
    Sets the variable contact
    """
    global contact
    global counter_lost_contact

    # Set the foot contact to False if it had no contact for more than 5 times in a row
    if data.data:
        contact = True
        counter_lost_contact = 0
    else:
        counter_lost_contact = counter_lost_contact + 1
        if counter_lost_contact > 5:
            contact = False


def callback_tactile(data):
    """
    Callback from tactile_sub
    Check if the buttons on the head are pressed
    data: the data that is sent from ros
    Sets the variable at_least_one_button
    """
    global status_button_1
    global status_button_2
    global status_button_3
    global at_least_one_button

    # Check if one of the buttons is pressed
    if data.button == 1:
        if data.state == 1:
            status_button_1 = True
        else:
            status_button_1 = False
    if data.button == 2:
        if data.state == 1:
            status_button_2 = True
        else:
            status_button_2 = False
    if data.button == 3:
        if data.state == 1:
            status_button_3 = True
        else:
            status_button_3 = False

    # Set the variable at_least_one_button if one button is pressed
    if (not status_button_1) and (not status_button_2) and (not status_button_3):
        at_least_one_button = False
    else:
        at_least_one_button = True


def thread_touch():
    """
    Check if the customer touches the robot
    Sets the variable contact_lost
    """
    global end

    # Check if the customer has contact to the buttons on the head of the robot while the robot moves
    while not end:
        global counter_lost_contact
        global contact_lost

        if at_least_one_button:
            contact_lost = False
            counter_lost_contact = 0
        else:
            counter_lost_contact = counter_lost_contact + 1
            # The contact only counts as lost if it happened for a longer time
            if counter_lost_contact > 5000:
                #contact_lost = True
                counter_lost_contact = 0


def recover():
    """
    Recover a standing position if the robot fell down
    """
    global pose
    global contact

    # If the foot sensors lost contact to the floor
    if not contact:
        pose = False
        # Warn the customer
        speech.say("I lost my footing")
        # Give the customer time to react
        time.sleep(5)
        # Stop the motors of the robot to prevent further damage
        for j in range(5):
            motion.setStiffnesses("Body", 0.0)
        # Give the customer time to react
        time.sleep(10)
        # Warn the customer that the robot is going to stand up
        speech.say("I will try to stand up")
        # Give the customer time to react
        time.sleep(5)

        # Stand up if the robot still has no foot contact to the floor
        if not contact:
            posture.goToPosture("StandInit", 1.0)
            pose = True
            contact = True
    else:
        # Thank the customer if the customer helped the robot on its feet
        if not pose:
            speech.say("Thank you for helping me up!")
            posture.goToPosture("StandInit", 1.0)
            pose = True


##############################
def main_loop():
    """
    Handle errors and send the map data to rviz
    """
    global marker_array
    global path_msg
    global end
    global world_frame_pub
    global robot_pos
    global path
    global gx
    global gy
    global goal_reached

    # Calculate a path to the goal
    path = calculate_path(gx, gy, robot_pos[0], robot_pos[1])
    speech.say("I have found a path, lets go")

    # Start the thread that takes care of the walking
    if path and robot:
        path.reverse()
        thread_walking = threading.Thread(target=walk_path, args=[])
        thread_walking.start()
    else:
        end = True

    # Start the main loop
    try:
        while not end:
            if robot:
                # Recover if the robot lost foot contact
                recover()

                # Check if the robot reached the goal
                if robot_pos[1] > gx:
                    motion.move(0.0, 0.0, 0)
                    print("I have reached my goal")
                    goal_reached = True
                    end = True

            # Publish the marker_array for rviz
            if marker_array:
                # Update the robot position
                create_robot_marker()
                visual_pub.publish(marker_array)
                marker_array.markers.pop()

                # Publish the marker_array
                path_pub.publish(path_msg)

                # Publish the world_frame coordinates for rviz
                world_frame_pub.sendTransform((0.0, 0.0, 0.0), (0, 0, 0, 1), rospy.Time.now(), "world_frame", "map")
    except KeyboardInterrupt:
        print("Closing")


if __name__ == "__main__":
    # This function is used to start the subsystems

    # Start the ros node
    NAO_IP = robot_ip
    rospy.init_node('guidenao', anonymous=True)

    # Speech recognition to start the interaction
    voice()
    rospy.sleep(3)

    # If the person asked for help
    initialize_ros()
    initialize_motion()
    rospy.sleep(3)

    # Detect the traffic light and calculate the distance
    gx, gy, cropped_light = crop_img()
    gx = gx + robot_pos[1]
    gy = gy + robot_pos[0]
    # Create the goal marker for rviz
    create_goal_marker()

    # Thread to check status of touch buttons
    thread_touch = threading.Thread(target=thread_touch, args=[])
    thread_touch.start()

    # Greet the customer
    speech.say("Hello, I am your assistant to cross the street")
    speech.say("Touch my head once you feel ready to go")
    rospy.sleep(3)

    # Check if the customer touches the robot
    while contact_lost:
        pass


    # Check color of traffic light and wait until it is green.
    # NOTE: from 1,5 m  and on is not reliable due to NAOs limited camera quality.
    #       Deactivate if traffic light is further away

    # speech.say("I will check the status of the traffic light and let you know once we can go")
    # gx1, gy2, cropped_light = crop_img()
    # color = check_color(cropped_light)
    # while color != "green":
        # gx1, gy2, cropped_light = crop_img()
        # color = check_color(cropped_light)

    speech.say("All right, we may go now, I will calculate the best way to go")
    rospy.sleep(3)

    # Start the main loop
    main_loop()

    # Send the customer in the direction of the sidewalk
    speech.say("We have reached our goal, you could go either left or right")
    speech.say("Have a nice day!")

    # Prohibit speech generation for all threads
    speech = False

    # Put the robot in a save resting position
    motion.rest()
