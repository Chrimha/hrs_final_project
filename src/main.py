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
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import threading
from rrt import RRT
import matplotlib.pyplot as plt

#Real-time camera image
live_image = 0
robot = True
ros = True
if robot:
    motion = ALProxy("ALMotion", "nao.local", 9559)
    speech = ALProxy("ALTextToSpeech", "nao.local", 9559)
    posture = ALProxy("ALRobotPosture", "nao.local", 9559)

# The state of the traffic light (False = red,  True = green)
traffic_light = False

contact = False
pose = False
end = False
obstacles = []

visual_pub = None
path_pub = None
path_msg = None
marker_array = MarkerArray()
if robot:
    motion = ALProxy("ALMotion", "nao.local", 9559)
    speech = ALProxy("ALTextToSpeech", "nao.local", 9559)
    posture = ALProxy("ALRobotPosture", "nao.local", 9559)


def initialize_ros():
    global visual_pub
    global path_pub
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

        speech.say("Hello, how are you?")

    # Publishers for visualisation in rviz
    path_pub = rospy.Publisher('/path', Path, queue_size=3)
    visual_pub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=3)


def initialize_motion():
    global pose
    for i in range(5):
        motion.setStiffnesses("Body", 1.0)

    posture.goToPosture("StandInit", 1.0)
    pose = True
    speech.say("lets go!")


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
    #cv2.waitKey()

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

    cv2.imshow("Img HSV 1", img_hsv)
    cv2.waitKey()

    # Show only yellow pixels
    cv2.imshow("Color  Extraction 1", mask_yellow)
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
    distance_to_robot = -0.05062*(x_max_mask - x_min_mask) + 3.269

    print("Traffic Light at (m): ", distance_to_robot)

    return distance_to_robot


def crop_img():

    global live_image
    cv_image_color = live_image
    # print(cv_image_color)
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

    #cv2.imshow("Input Image 1", cv_image_color)
    #cv2.waitKey()

    #cv2.imshow("Img HSV", img_hsv)
    #cv2.waitKey()

    # Show only yellow pixels
    #cv2.imshow("Color  Extraction", mask_yellow)
    #cv2.waitKey()

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
    cv2.imshow("cropped", crop_img)
    cv2.waitKey()

    # ToDo
    return measure_distance(crop_img), 0


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
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.09, camera_coefficients,
                                                                           distortion_coefficients)
            (rvec - tvec).any()  # get rid of that nasty numpy value array error
            cv2.aruco.drawDetectedMarkers(cv_image_color2, corners)  # Draw A square around the markers
            cv2.aruco.drawAxis(cv_image_color2, camera_coefficients, distortion_coefficients, rvec, tvec,
                               0.01)  # Draw Axis
            tvec = tvec[0][0]
            tvec[0] = -tvec[0] - 0.5
            tvec[1] = -tvec[1] - 0.5
            tvec[2] = tvec[2]/2
            tvecs.append(tvec)
            #tvec -> x(horizontal + 0.5), y (vertical and z= (distance*2)
            print(tvec)

            # ToDo
            # create_marker(i, 1.0, 1.0, tvec[0], tvec[1])

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


def calculate_trajectory(gy, gx):
    global obstacles
    global path_msg

    print("Calculating trajectory...")
    show_animation = False

    rrt = RRT(
        start=[0, 0],
        goal=[gx, gy],
        max_iter=100,
        obstacle_list=obstacles
        )
    path = rrt.planning(animation=show_animation)
    print(path)

    if path is None:
        print("Cannot find path")
    else:
        print("Found path!!")

        # Draw final path
        if show_animation:
            rrt.draw_graph()
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
            plt.grid(True)
            plt.pause(0.01)
            plt.show()

    msg = Path()
    msg.header.frame_id = "torso"
    msg.header.stamp = rospy.Time.now()

    if path is not None:
        for wp in path:
            pos = PoseStamped()
            pos.pose.position.x = wp[0]
            pos.pose.position.y = wp[1]
            pos.pose.position.z = 0

            #quaternion = tf.transformations.quaternion_from_euler(
            #    0, 0, -math.radians(wp[0].transform.rotation.yaw))
            #pos.pose.orientation.x = quaternion[0]
            #pos.pose.orientation.y = quaternion[1]
            #pos.pose.orientation.z = quaternion[2]
            #pos.pose.orientation.w = quaternion[3]
            msg.poses.append(pos)
            path_msg = msg
        return path
    return None


def recover():
    global pose
    global contact
    if not contact:
        pose = False
        speech.say("I lost my footing")
        print("shufshrwrhjfiwjijiejdfiejfweifiofehjiodfwefhwhfsirihrihwrahwrhgwagho")
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


def main_loop(gx, gy):
    global marker_array
    global path_msg
    global end

    path = calculate_trajectory(gx, gy)

    if path:
        path.reverse()
        x = threading.Thread(target=walk_path(path), args=(1,))
        # x.start()
        #walk_path(path)
    else:
        end = True
    # motion.moveTo(1 * 0.83, 0, 0)
    try:
        while not end:
            print("while")
            if robot:
                recover()

            if marker_array:
                visual_pub.publish(marker_array)
                path_pub.publish(path_msg)
    except KeyboardInterrupt:
        print("Closing")

    if robot:
        motion.rest()


def walk_path(path):
    motion.moveTo(path[0][1] * 0.83, -path[0][0] * 0.83, 0)

    calculate_trajectory(gx, gy)


def create_marker(number, size_x, size_y, x, y):
    marker = Marker()
    marker.header.frame_id = "torso"
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
    marker.pose.position.x = x
    marker.pose.position.y = y
    marker.pose.position.z = 0
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 5.0
    marker.pose.orientation.w = 1.0

    global marker_array
    global obstacles
    marker_array.markers.append(marker)
    obstacles.append([marker.pose.position.x, marker.pose.position.y, max(marker.scale.x, marker.scale.y) + 0.5])


if __name__ == "__main__":
    if not robot and not ros:
        # Without ros
        print(cv2.__version__)
        image = cv2.imread('../arucoTemp.jpg')
        callback_image(image)
        rospy.spin()

    elif robot:
        initialize_ros()
        initialize_motion()
        rospy.sleep(2)
        # 2.3m : 21
        # 2m : 24
        # 1.5m : 33
        # 1.1m : 44
        gx, gy = crop_img()

        # rospy.spin()
        main_loop(gx, gy)


        speech.say("Good bye!")

    elif ros and not robot:
        initialize_ros()
        create_marker(0, 0.2, 0.1, 2, 3)  # (id, size_x, size_y, x, y)
        create_marker(1, 0.2, 0.3, 8, 7)

        main_loop()
