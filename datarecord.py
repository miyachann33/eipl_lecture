#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#from types import NoneType
import rospy
from std_msgs.msg import String,Float64MultiArray
from torobo_sensor_msgs.msg import ToroboState
from sensor_msgs.msg import Image
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from gpio_msgs.msg import GpioStates
from tf2_msgs.msg import TFMessage
import csv
import cv2
from cv_bridge import CvBridge
import datetime
import numpy as np
import os
import tf
import sys

now = datetime.datetime.now()
current_time = now.strftime("%Y-%m-%d-%H-%M-%S")
tmp_dir_name = os.path.abspath(os.path.basename(__file__))
tmp_file_name = tmp_dir_name.split('/')[-1]
dir_for_output = "/home/d-airec/catkin_ws/src/lecture1026/dataset/"+ current_time#tmp_dir_name.split(tmp_file_name)[0] + "dataset/" + current_time

os.makedirs(dir_for_output)
os.makedirs(dir_for_output+"/fisheye")
os.makedirs(dir_for_output+"/color")
os.makedirs(dir_for_output+"/depth")

step = 0
tactile = [0]
angle = [0]
torque=[0]
command_angle_right = [0]
command_angle_left = [0]
external_torque=[0]
fisheye_image = 0
color_image = 0
depth_image = 0
port_names = []
analog_data = []
_tf_static_list = []
callback_count = 0
param_text = ""
servo_on = False      # 最新のサーボ状態（True: ON, False: OFF）
recording = False     # 記録中かどうか
ARM_JOINT_INDICES = [
    10,11,12,13,14,15,16,   # left_arm/joint_1〜7
    28,29,30,31,32,33,34    # right_arm/joint_1〜7
]
def TouchSensorCallback(data):
    global tactile
    tac = data.data
    # print(tac)
    tactile = tuple([tac[:]])

def AngleSensorCallback(data):
    global angle, torque, external_torque, servo_on

    ang = data.link_position
    tor = data.link_effort
    extor = data.external_effort

    angle = tuple([ang])
    torque = tuple([tor])
    external_torque = tuple([extor])

    servo = list(data.servo_state)

    # 腕のどれか1つでも servo_state == 1 なら腕サーボON
    servo_on = any(servo[i] == 1 for i in ARM_JOINT_INDICES)


def CommandAngleSensorCallback(data):
    print(data.joint_names[0])
    global command_angle_right
    global command_angle_left
    cang = data.points[0].positions
    if data.joint_names[0] == "left_arm/joint_1":
        command_angle_right = tuple([cang])
    elif data.joint_names[0] == "right_arm/joint_1":
        command_angle_left = tuple([cang])

def _tf_callback(msg):
    global _tf_static_list
    for transform in msg.transforms:
        _tf_static_list.append(transform.child_frame_id)
    rospy.loginfo('tf_static subscribed.')

def gpio_state_callback(msg):
        global port_names
        global analog_data
        global callback_count

        analog_data = []
        t = rospy.Time.now()

        if callback_count < 10:
            port_names = []
            for port_name, data in zip(msg.analog_io_port_names, msg.analog_io_data):
                if port_name+'/link' in _tf_static_list:
                    port_names.append(tuple([port_name]))
                    analog_data.append(tuple([data]))
            callback_count += 1

        for port_name, data in zip(msg.analog_io_port_names, msg.analog_io_data):
            if port_name+'/link' in _tf_static_list:
                analog_data.append(tuple([data]))
        
def fisheye_callback(msg):
    global fisheye_image
    try:
        bridge = CvBridge()
        fisheye_image =bridge.imgmsg_to_cv2(msg, "bgr8")

    except Exception as err:
        rospy.logerr(err)

def color_callback(msg):
    global color_image
    try:
        bridge = CvBridge()
        color_image =bridge.imgmsg_to_cv2(msg, "bgr8")
        cv2.imshow("color image", color_image)
        cv2.waitKey(1)

    except Exception as err:
        rospy.logerr(err)

def depth_callback(msg):
    global depth_image
    try:
        bridge = CvBridge()
        depth_image = bridge.imgmsg_to_cv2(msg, "16UC1")

    except Exception as err:
        rospy.logerr(err)


def _imp_callback(msg):
    global param_text
    param_text = msg.data
    
def timerCallback(event):
    rospy.loginfo("timer callback")


def timer():
    rate=rospy.Rate(20) #Hz
    rate.sleep()

if __name__ == '__main__':
    try:
        rospy.init_node('listener', anonymous=True)

        with open(dir_for_output+"/angle.csv", 'w') as f:
            writer = csv.writer(f)

        with open(dir_for_output+"/torque.csv", 'w') as f:
            writer = csv.writer(f)

        with open(dir_for_output+"/touch.csv", 'w') as f:
            writer = csv.writer(f)

        with open(dir_for_output+"/touch_namedata_list.csv", 'w') as f:
            writer = csv.writer(f)

        with open(dir_for_output+"/command_angle.csv", 'a') as f:
            writer = csv.writer(f)

        with open(dir_for_output+"/external_torque.csv", 'a') as f:
            writer = csv.writer(f)

        with open(dir_for_output+"/impedance_state.csv", 'a') as f:
            writer = csv.writer(f)

        rospy.Subscriber("/torobo/head/insta360/camera/color/image_raw", Image, fisheye_callback)
        rospy.Subscriber("/torobo/head/sr300/camera/color/image_raw", Image, color_callback)
        rospy.Subscriber("/torobo/head/sr300/camera/aligned_depth_to_color/image_raw", Image, depth_callback)
        rospy.Subscriber("/torobo/torobo_states",ToroboState,AngleSensorCallback)
        rospy.Subscriber("/torobo/gpio_state_controller/analog_io",Float64MultiArray,TouchSensorCallback)
        rospy.Subscriber("/torobo/gpio_state_controller/gpio_states",GpioStates,gpio_state_callback)
        rospy.Subscriber("/tf_static", TFMessage, _tf_callback)
        rospy.Subscriber("/imp_state", String, _imp_callback)

        # rospy.Subscriber("/torobo/torobo/torobo_perception_neuron_trajectory_filter_node/parameter_descriptions/command",JointTrajectory,CommandAngleSensorCallback)
        rospy.Subscriber("/torobo/online_joint_impedance_controller/command",JointTrajectory,CommandAngleSensorCallback)
        # rospy.Subscriber("/torobo/online_joint_trajectory_controller/command",JointTrajectory,CommandAngleSensorCallback)
        # control_msgs/JointTrajectoryControllerState

    except rospy.ROSInterruptException :
           sys.exit()



    command_angle_right = tuple([[0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    command_angle_left = tuple([[0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    # print(type(command_angle_left[0]),type(command_angle_right[0]))
    # print(command_angle_right[0])
    # while(1):
    #     try:
    #         if step > 1 and callback_count > 9:
    #             cv2.imwrite(dir_for_output+"/fisheye/"+str(step)+".png", fisheye_image)
    #             cv2.imwrite(dir_for_output+"/color/"+str(step)+".png", color_image)
    #             cv2.imwrite(dir_for_output+"/depth/"+str(step)+".png", depth_image)

    #             with open(dir_for_output+"/angle.csv", 'a') as f:
    #                 writer = csv.writer(f)
    #                 writer.writerow(np.array(angle[0]))

    #             with open(dir_for_output+"/torque.csv", 'a') as f:
    #                 writer = csv.writer(f)
    #                 writer.writerow(np.array(torque[0]))

    #             with open(dir_for_output+"/touch.csv", 'a') as f:
    #                 writer = csv.writer(f)
    #                 writer.writerow(np.array(tactile[0]))

    #                 with open(dir_for_output+"/touch_namedata_list.csv", 'a') as f:
    #                     writer = csv.writer(f)
    #                     if callback_count == 10:
    #                         writer.writerow(np.array(port_names)) 
    #                         callback_count += 1
    #                     writer.writerow(np.array(analog_data))
  
    #             with open(dir_for_output+"/command_angle.csv", 'a') as f:
    #                 writer = csv.writer(f)
    #                 # print(type(command_angle_left[0]),type(command_angle_right[0]))
    #                 command_angles = tuple(command_angle_left[0]) + tuple(command_angle_right[0])
    #                 writer.writerow(np.array(command_angles))

    #             with open(dir_for_output+"/external_torque.csv", 'a') as f:
    #                 writer = csv.writer(f)
    #                 writer.writerow(np.array(external_torque[0]))

    #     except KeyboardInterrupt:
    #         sys.exit()

    #     step += 1
    #     timer()
    while(1):
        try:
            #global recording

            # ★ 腕サーボON検出 → 記録開始 ★
            if servo_on and not recording:
                rospy.loginfo("腕のサーボONを検出。記録を開始します。")
                recording = True

            # ★ 腕サーボOFF検出 → 記録停止 & 終了 ★
            if not servo_on and recording:
                rospy.loginfo("腕のサーボOFFを検出。記録を停止して終了します。")
                # 必要ならここでファイルを閉じたり後処理もできる
                break  # while ループを抜けて終了

            # 記録中のときだけ保存処理を行う
            if recording and step > 1 and callback_count > 9:
                cv2.imwrite(dir_for_output+"/fisheye/"+str(step)+".png", fisheye_image)
                cv2.imwrite(dir_for_output+"/color/"+str(step)+".png", color_image)
                cv2.imwrite(dir_for_output+"/depth/"+str(step)+".png", depth_image)

                with open(dir_for_output+"/angle.csv", 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(np.array(angle[0]))

                with open(dir_for_output+"/torque.csv", 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(np.array(torque[0]))

                with open(dir_for_output+"/touch.csv", 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(np.array(tactile[0]))

                    with open(dir_for_output+"/touch_namedata_list.csv", 'a') as f:
                        writer = csv.writer(f)
                        if callback_count == 10:
                            writer.writerow(np.array(port_names)) 
                            callback_count += 1
                        writer.writerow(np.array(analog_data))
    
                with open(dir_for_output+"/command_angle.csv", 'a') as f:
                    writer = csv.writer(f)
                    command_angles = tuple(command_angle_left[0]) + tuple(command_angle_right[0])
                    writer.writerow(np.array(command_angles))

                with open(dir_for_output+"/external_torque.csv", 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(np.array(external_torque[0]))

                with open(dir_for_output+"/impedance_state.csv", 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(param_text)

        except KeyboardInterrupt:
            sys.exit()

        step += 1
        timer()

    # while を抜けたらROSノードを安全に終了
    rospy.loginfo("データ記録ノードを終了します。")
    sys.exit(0)

