#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import mediapipe as mp
import csv
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import os
import tf
import math
from geometry_msgs.msg import PointStamped, Twist, Pose
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from torobo_std_msgs.srv import SetStringArray
from controller_manager_msgs.srv import ListControllers, SwitchController, SwitchControllerRequest
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from collections import deque
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from moveit_msgs.msg import RobotState, PositionIKRequest
from moveit_msgs.srv import GetPositionFK, GetPositionIK
from sensor_msgs.msg import JointState
import dynamic_reconfigure.client
import matplotlib.animation as animation
from torobo_sensor_msgs.msg import ToroboState
import moveit_commander
from std_msgs.msg import Header
import subprocess
import signal

######
import threading
from nav_msgs.msg import Odometry
import tf
JOINT_OFFSETS = {
    11: np.array([0, 0, 0.03]),
    12: np.array([0, 0, 0.03]),
    23: np.array([0, 0, 0.075]),
    24: np.array([0, 0, 0.15]),
    25: np.array([0, 0, 0.04]),
}
current_base_pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}

def odom_callback(msg):
    """odomからロボットの現在位置を取得"""
    global current_base_pose
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    q = msg.pose.pose.orientation
    (_, _, yaw) = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
    current_base_pose = {"x": x, "y": y, "yaw": yaw}

def broadcast_base_link_tf():
    """odom→base_link の変換を常に上書きブロードキャスト"""
    global current_base_pose
    br = tf.TransformBroadcaster()
    rate = rospy.Rate(20)  # 更新頻度(Hz)

    while not rospy.is_shutdown():
        x = current_base_pose["x"]
        y = current_base_pose["y"]
        yaw = current_base_pose["yaw"]

        # ここが重要：odom→base_link の動的変換を送信
        br.sendTransform(
            (x, y, 0.0),
            tf.transformations.quaternion_from_euler(0, 0, yaw),
            rospy.Time.now(),
            "base_link",  # 子フレーム（MoveItが参照するbase_link）
            "odom"        # 親フレーム（外部基準）
        )
        rate.sleep()

def smooth_update(client, start_params, end_params, steps=5, delay=0.2):
    for i in range(1, steps + 1):
        params = {}
        for key in start_params.keys():
            params[key] = start_params[key] + (end_params[key] - start_params[key]) * (i / steps)
        client.update_configuration(params)
        rospy.sleep(delay)

#######


r1_m = 'right_arm/joint_1/mass'
r1_s = 'right_arm/joint_1/spring'
r1_d = 'right_arm/joint_1/damping'
r2_m = 'right_arm/joint_2/mass'
r2_s = 'right_arm/joint_2/spring'
r2_d = 'right_arm/joint_2/damping'
r3_m = 'right_arm/joint_3/mass'
r3_s = 'right_arm/joint_3/spring'
r3_d = 'right_arm/joint_3/damping'
r4_m = 'right_arm/joint_4/mass'
r4_s = 'right_arm/joint_4/spring'
r4_d = 'right_arm/joint_4/damping'
r5_m = 'right_arm/joint_5/mass'
r5_s = 'right_arm/joint_5/spring'
r5_d = 'right_arm/joint_5/damping'
r6_m = 'right_arm/joint_6/mass'
r6_s = 'right_arm/joint_6/spring'
r6_d = 'right_arm/joint_6/damping'
r7_m = 'right_arm/joint_7/mass'
r7_s = 'right_arm/joint_7/spring'
r7_d = 'right_arm/joint_7/damping'
l1_m = 'left_arm/joint_1/mass'
l1_s = 'left_arm/joint_1/spring'
l1_d = 'left_arm/joint_1/damping'
l2_m = 'left_arm/joint_2/mass'
l2_s = 'left_arm/joint_2/spring'
l2_d = 'left_arm/joint_2/damping'
l3_m = 'left_arm/joint_3/mass'
l3_s = 'left_arm/joint_3/spring'
l3_d = 'left_arm/joint_3/damping'
l4_m = 'left_arm/joint_4/mass'
l4_s = 'left_arm/joint_4/spring'
l4_d = 'left_arm/joint_4/damping'
l5_m = 'left_arm/joint_5/mass'
l5_s = 'left_arm/joint_5/spring'
l5_d = 'left_arm/joint_5/damping'
l6_m = 'left_arm/joint_6/mass'
l6_s = 'left_arm/joint_6/spring'
l6_d = 'left_arm/joint_6/damping'
l7_m = 'left_arm/joint_7/mass'
l7_s = 'left_arm/joint_7/spring'
l7_d = 'left_arm/joint_7/damping'

torque_data_right = []
torque_data_left = []
time_stamps = []

# 安定性判定に使う変数
target_joint_ids = [11, 12, 24, 23, 25]
stability_threshold = 0.05  # 単位: メートル (例: 1cm以内のブレ)
position_history = defaultdict(lambda: deque(maxlen=10))

second_round = True
# 初期化
bridge = CvBridge()
depth_image = None
latest_color_image = None
latest_pc = None
current_step = 1
A_R = np.zeros((4, 2))
head_joints = ['head/joint_1', 'head/joint_2', 'head/joint_3']
torso_joints = ['torso/joint_1', 'torso/joint_2', 'torso/joint_3']
right_arm_joints = ['right_arm/joint_1','right_arm/joint_2','right_arm/joint_3','right_arm/joint_4','right_arm/joint_5','right_arm/joint_6','right_arm/joint_7']
left_arm_joints = ['left_arm/joint_1','left_arm/joint_2','left_arm/joint_3','left_arm/joint_4','left_arm/joint_5','left_arm/joint_6','left_arm/joint_7']
right_hand_joints = [
    'right_hand_thumb/joint_1',
    'right_hand_thumb/joint_2',
    'right_hand_thumb/joint_3',
    'right_hand_first_finger/joint_1',
    'right_hand_second_finger/joint_1',
    'right_hand_third_finger/joint_1'
]
wheel_joints = ['base_front_left_wheel/joint', 'base_front_right_wheel/joint', 'base_rear_left_wheel/joint', 'base_rear_right_wheel/joint']

override_params = {
    "left_arm/joint_1_speed_override": 0.2,
    "left_arm/joint_2_speed_override": 0.2,
    "left_arm/joint_3_speed_override": 0.2,
    "left_arm/joint_4_speed_override": 0.2,
    "left_arm/joint_5_speed_override": 0.2,
    "left_arm/joint_6_speed_override": 0.2,
    "left_arm/joint_7_speed_override": 0.2,
    "right_arm/joint_1_speed_override": 0.2,
    "right_arm/joint_2_speed_override": 0.2,
    "right_arm/joint_3_speed_override": 0.2,
    "right_arm/joint_4_speed_override": 0.2,
    "right_arm/joint_5_speed_override": 0.2,
    "right_arm/joint_6_speed_override": 0.2,
    "right_arm/joint_7_speed_override": 0.2,
}

# カメラパラメータ（カメラキャリブレーション結果に基づいて設定）
mtx = np.array([[372.24736924, 0., 570.06115586],
                [0., 371.35032753, 534.888006],
                [0., 0., 1.]])
dist = np.array([[-0.27165737, 0.05936596, 0.0029973, -0.00281489, -0.00493731]])
h, w = 1088, 1088
scale = 0.56

# CSV準備
directory_path = '/home/d-airec/catkin_ws/src/posture_change/dataset/'
os.makedirs(directory_path, exist_ok=True)
csv_file_path = os.path.join(directory_path, 'tf_skeleton.csv')
csv_file = open(csv_file_path, mode='w')
csv_writer = csv.writer(csv_file)
header = ["frame"]
for i in range(33):  # Mediapipeの33関節
    header.extend([f"Joint_{i}_x", f"Joint_{i}_y", f"Joint_{i}_z"])
csv_writer.writerow(header)

# MediaPipe 初期化
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2)
def transform(a, b, c, d):
    a = np.array(a).flatten()
    b = np.array(b).flatten()
    c = np.array(c).flatten()
    d = np.array(d).flatten()

    origin = (c + d) / 2
    pz = (a + b)/2 - origin
    py = d - c

    ez = pz / np.linalg.norm(pz)
    ex = np.cross(py, pz) / np.linalg.norm(np.cross(py, pz))
    ey = np.cross(ez, ex)

    # ローカル→グローバル
    R = np.array([ex, ey, ez]).T
    R_LtoG = np.identity(4)
    R_LtoG[:3, :3] = R
    R_LtoG[:3, 3] = origin

    # グローバル→ローカル
    R_GtoL = np.identity(4)
    R_GtoL[:3, :3] = R.T
    R_GtoL[:3, 3] = -R.T.dot(origin)

    return R_GtoL, R_LtoG


def switch_controller(start_controllers=[],
                       stop_controllers=[],
                       timeout=3.0):

    service_name = '/torobo/controller_manager/switch_controller'
    try:
        rospy.loginfo('call switch_controller service')
        rospy.wait_for_service(service_name, timeout=timeout)
        service = rospy.ServiceProxy(service_name, SwitchController)

        request = SwitchControllerRequest()
        request.start_controllers = start_controllers
        request.stop_controllers = stop_controllers
        request.strictness = SwitchControllerRequest.STRICT
        request.start_asap = True
        request.timeout = timeout

        response = service(request)

        # Check whether send is succeeded
        rospy.loginfo('switch controller result: [%s], start_controllers: [%s], stop_controllers: [%s]'
                   % (str(response.ok), '. '.join(start_controllers), ', '.join(stop_controllers)))

    except rospy.ROSException as e:
        e = rospy.ROSInterruptException('switch_controller service is failed.')
        rospy.logerr(e)
        raise e

    if not response.ok:
        e = rospy.ROSInterruptException('switch_controller service is failed.')
        rospy.logerr(e)
        raise e

    rospy.loginfo('switch_controller service is succeeded.')

    return response.ok

def set_servo_on(joint_names):

    # Call service
    service_name = '/torobo/robot_hw/servo_on'
    rospy.wait_for_service(service_name)
    service = rospy.ServiceProxy(service_name, SetStringArray)
    rospy.loginfo('call set_servo_on service')
    response = service(joint_names)

    # Check whether send is succeeded
    if not response.success:
        e = rospy.ROSInterruptException('set_servo_on service is failed.')
        rospy.logerr(e)
        raise e

    rospy.loginfo('set_servo_on service is succeeded.')

def set_servo_off(joint_names):

    # Call service
    service_name = '/torobo/robot_hw/servo_off'
    rospy.wait_for_service(service_name)
    service = rospy.ServiceProxy(service_name, SetStringArray)
    rospy.loginfo('call set_servo_off service')
    response = service(joint_names)

    # Check whether send is succeeded
    if not response.success:
        e = rospy.ROSInterruptException('set_servo_off service is failed.')
        rospy.logerr(e)
        raise e

    rospy.loginfo('set_servo_off service is succeeded.')

def publish_joint_trajectory(publisher, joint_names, positions, time_from_start):

    # Creates a message.
    trajectory = JointTrajectory()
    trajectory.header.stamp = rospy.Time.now()
    trajectory.joint_names = joint_names
    point = JointTrajectoryPoint()
    point.positions = positions
    point.velocities = [0.0] * len(joint_names)
    point.accelerations = [0.0] * len(joint_names)
    point.effort = [0.0] * len(joint_names)
    point.time_from_start = rospy.Duration(time_from_start)
    trajectory.points.append(point)

    # Publish the message.
    publisher.publish(trajectory)


class Torobo:

    def __init__(self, movegroup, tooltip):

        rospy.Rate(1)
        rospy.sleep(1)
        rospy.Subscriber('/torobo/joint_states', JointState, self.joint_callback)


        self._robot = moveit_commander.MoveGroupCommander(movegroup, robot_description="/torobo/robot_description", ns="/torobo")
        self._robot.set_pose_reference_frame('base_link')
        self._active_joints = self._robot.get_active_joints()
        self._tooltip = tooltip

        rospy.loginfo("loading fk")
        rospy.wait_for_service('compute_fk')
        rospy.wait_for_service('compute_ik')
        try:
            self._moveit_fk = rospy.ServiceProxy('compute_fk', GetPositionFK)
            self._moveit_ik = rospy.ServiceProxy('compute_ik', GetPositionIK)
        except rospy.ServiceException as e:
            rospy.logerror("Service call failed: %s"%e)
    
    def joint_callback(self, msg):
        
        self.arm_t = np.array(msg.position)

    def compute_ik(self, now, x, y, z, roll, pitch, yaw):
        quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)

        # create request
        req = PositionIKRequest()
        req.timeout = rospy.Duration(2.0)
        req.ik_link_name = self._tooltip
        req.pose_stamped.header.frame_id = "base_link"
        req.pose_stamped.pose.position.x = x
        req.pose_stamped.pose.position.y = y
        req.pose_stamped.pose.position.z = z
        req.pose_stamped.pose.orientation.x = quaternion[0]
        req.pose_stamped.pose.orientation.y = quaternion[1]
        req.pose_stamped.pose.orientation.z = quaternion[2]
        req.pose_stamped.pose.orientation.w = quaternion[3]
        req.robot_state.joint_state.name = self._active_joints

        # シード値の設定
        if now is None:
            seed = self._robot.get_current_joint_values()
            req.robot_state.joint_state.position = seed
            print("current value: ", seed)
        else:
            seed = now
            req.robot_state.joint_state.position = seed
            print("now: ", now)

        req.group_name = self._robot.get_name()
        req.avoid_collisions = True

        # --- IK 実行 ---
        response = self._moveit_ik(req)
        ik_result = response.solution

        # --- IK 解が得られなかった場合の処理 ---
        if len(ik_result.joint_state.position) == 0:
            rospy.logwarn("⚠ IK failed. Returning seed joint values instead.")

            # seed から JointState を作成して返す
            fallback = JointState()
            fallback.name = self._active_joints
            fallback.position = seed
            return fallback

        # --- IK 解が得られたとき ---
        for i in range(len(ik_result.joint_state.name)):
            joint_name = ik_result.joint_state.name[i]
            if joint_name in self._active_joints:
                rospy.loginfo("    {0} : {1:.2f}".format(
                    joint_name,
                    np.degrees(ik_result.joint_state.position[i])
                ))

        return ik_result.joint_state



def cleanup():
    # ここでは明示的にサーボOFFしない。初期姿勢を維持したい場合は保持状態のままにする。
    # set_servo_off(head_joints + torso_joints + right_arm_joints + left_arm_joints)
    try:
        csv_file.close()
    except Exception:
        pass
    cv2.destroyAllWindows()

def is_joint_stable(p_history, joint_id, frame_count=10, threshold=stability_threshold):
    history = p_history[joint_id]
    valid_positions = [p for p in history if p is not None]

    if len(valid_positions) < frame_count:
        return False

    std_dev = np.std(valid_positions, axis=0)
    return np.all(std_dev < threshold)


def are_joints_stable(p_history, joint_ids, frame_count=10, threshold=stability_threshold):
    return all(is_joint_stable(p_history, jid, frame_count, threshold) for jid in joint_ids)

def main():
    global tf_listener, tf_broadcaster, latest_color_image, depth_image, second_round, current_step, position_history
    rospy.init_node('publish_skeleton_tf_node')
    ######
    # --- base_link の動的TFを有効化 ---
    rospy.Subscriber("/odom", Odometry, odom_callback)
    tf_thread = threading.Thread(target=broadcast_base_link_tf)
    tf_thread.daemon = True
    tf_thread.start()
    ######
    rospy.on_shutdown(cleanup)
    set_servo_on(head_joints + torso_joints + right_arm_joints + left_arm_joints)
    rospy.sleep(1.0)

    # 正しい online impedance controller を起動して指令を送る
    switch_controller(
        start_controllers=['online_joint_impedance_controller'],
        stop_controllers=['joint_impedance_controller'],
        timeout=5.0,
    )

    #計測のための顔の動き
    TOPIC_NAME = '/torobo/online_joint_impedance_controller/command'
    publisher = rospy.Publisher(TOPIC_NAME, JointTrajectory, queue_size=1)
    rospy.sleep(1)

    #初期姿勢
    target_positions = np.deg2rad([
        0, 50, 0,
        0, 0, 0,
        28.294, -33.122, -46.115, 69.776, -14.564, -19.385, 0,
        28.294, -33.122, -46.115, 69.776, -14.564, -19.385, 0,
    ])
    publish_joint_trajectory(
        publisher = publisher,
        joint_names = head_joints + torso_joints + right_arm_joints + left_arm_joints,
        positions = target_positions,
        time_from_start = 10.0
    )
    rospy.sleep(10)

    # 目標姿勢を保持し続けるため、shutdown で強制終了しない
    rospy.loginfo("Initial pose command sent and holding.")
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
