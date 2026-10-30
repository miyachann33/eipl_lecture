#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os

import cv2
import numpy as np
import rospy
import torch
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from eipl.model.SARNN_rgbd import SARNN_rgbd


RIGHT_ARM_NAMES = [
    "right_arm/joint_1",
    "right_arm/joint_2",
    "right_arm/joint_3",
    "right_arm/joint_4",
    "right_arm/joint_5",
    "right_arm/joint_6",
    "right_arm/joint_7",
]
LEFT_ARM_NAMES = [
    "left_arm/joint_1",
    "left_arm/joint_2",
    "left_arm/joint_3",
    "left_arm/joint_4",
    "left_arm/joint_5",
    "left_arm/joint_6",
    "left_arm/joint_7",
]
HEAD_NAMES = ["head/joint_1", "head/joint_2", "head/joint_3"]
TORSO_NAMES = ["torso/joint_1", "torso/joint_2", "torso/joint_3"]
CONTROL_NAMES = HEAD_NAMES + TORSO_NAMES + RIGHT_ARM_NAMES + LEFT_ARM_NAMES


def normalize(x, in_min, in_max, out_min, out_max):
    return (x - in_min) / (in_max - in_min) * (out_max - out_min) + out_min


def denormalize(x, in_min, in_max, out_min, out_max):
    return (x - out_min) / (out_max - out_min) * (in_max - in_min) + in_min


def preprocess_rgbd(color_bgr, depth_u16, img_size=32, vmin=0.1, vmax=0.9):
    if color_bgr is None or depth_u16 is None:
        raise ValueError("color/depth image is None")

    rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    depth = depth_u16.astype(np.float32)

    valid_mask = (depth >= 0.0) & (depth <= 2500.0)
    rgb = rgb * valid_mask[..., None]

    depth_norm = (depth - 0.0) / 2500.0
    depth_norm = np.clip(depth_norm, 0.0, 1.0)

    rgb_chw = np.transpose(rgb, (2, 0, 1))
    rgb_norm = (rgb_chw - 0.0) / 255.0
    rgb_norm = rgb_norm * (vmax - vmin) + vmin
    rgbd = np.concatenate([rgb_norm, depth_norm[None]], axis=0)

    rgbd_hw = np.transpose(rgbd, (1, 2, 0))
    rgbd_hw = cv2.resize(rgbd_hw, (img_size, img_size), interpolation=cv2.INTER_AREA)
    rgbd = np.transpose(rgbd_hw, (2, 0, 1))
    return rgbd.astype(np.float32)


class RobotMTRNNController:
    def __init__(self, model_path, device, img_size=32):
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self.bridge = CvBridge()

        self.model_path = os.path.abspath(model_path)
        self.model_dir = os.path.dirname(self.model_path)
        data_json = os.path.join(self.model_dir, "data.json")
        if not os.path.exists(data_json):
            raise FileNotFoundError(f"data.json が見つかりません: {data_json}")

        with open(data_json, "r") as f:
            hyper = json.load(f)
        self.joint_min = np.array(hyper["joint_min"], dtype=np.float32)
        self.joint_max = np.array(hyper["joint_max"], dtype=np.float32)

        checkpoint = torch.load(self.model_path, map_location=self.device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

        # SpatialSoftmax / InverseSpatialSoftmax are size-dependent.
        # The checkpoint was trained with a full image size such that the feature map
        # after the 3x3 conv stack is 58x58, i.e. full image size = 64x64.
        inferred_sub_size = None
        if "pos_encoder.6.pos_x" in state_dict:
            inferred_sub_size = int(np.sqrt(state_dict["pos_encoder.6.pos_x"].numel()))
        elif "issm.pos_xy" in state_dict:
            inferred_sub_size = state_dict["issm.pos_xy"].shape[-1]

        if inferred_sub_size is not None:
            # Model uses 3 Conv2d layers with kernel 3 and no padding, so
            # sub_im_size = full_im_size - 3 * (kernel_size - 1)
            self.img_size = int(inferred_sub_size + 3 * (3 - 1))
        else:
            self.img_size = int(img_size)

        rec_dim = state_dict["rec.weight_hh"].shape[0] // 4
        self.model = SARNN_rgbd(
            rec_dim=rec_dim,
            joint_dim=len(self.joint_min),
            k_dim=5,
            heatmap_size=0.1,
            temperature=1e-4,
            im_size=[self.img_size, self.img_size],
        ).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.state = None
        self.latest_color = None
        self.latest_depth = None
        self.latest_joint_state = None
        self.last_publish_time = None

        self.pub = rospy.Publisher(
            "/torobo/online_joint_impedance_controller/command",
            JointTrajectory,
            queue_size=1,
        )
        rospy.Subscriber("/torobo/head/sr300/camera/color/image_raw", Image, self.color_cb, queue_size=1)
        rospy.Subscriber("/torobo/head/sr300/camera/aligned_depth_to_color/image_raw", Image, self.depth_cb, queue_size=1)
        rospy.Subscriber("/torobo/joint_states", JointState, self.joint_cb, queue_size=1)

    def color_cb(self, msg):
        try:
            self.latest_color = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logwarn(f"color bridge error: {e}")

    def depth_cb(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            rospy.logwarn(f"depth bridge error: {e}")

    def joint_cb(self, msg):
        self.latest_joint_state = msg

    def build_joint_input(self):
        if self.latest_joint_state is None:
            raise RuntimeError("joint_states has not arrived yet")

        pos_map = dict(zip(self.latest_joint_state.name, self.latest_joint_state.position))
        right = np.array([pos_map[name] for name in RIGHT_ARM_NAMES], dtype=np.float32)
        left = np.array([pos_map[name] for name in LEFT_ARM_NAMES], dtype=np.float32)
        angle_part = np.concatenate([right, left], axis=0)
        torque_part = np.zeros(14, dtype=np.float32)
        joint_vec = np.concatenate([angle_part, torque_part], axis=0)

        joint_vec = normalize(joint_vec, self.joint_min, self.joint_max, 0.0, 1.0)
        return joint_vec.astype(np.float32)

    def build_command(self, pred_joint_28):
        pred_joint_28 = np.asarray(pred_joint_28, dtype=np.float32)
        pred_joint_28 = np.clip(pred_joint_28, 0.0, 1.0)
        arm_pred = denormalize(pred_joint_28[:14], self.joint_min[:14], self.joint_max[:14], 0.0, 1.0)
        right_pred = arm_pred[:7]
        left_pred = arm_pred[7:14]

        pos_map = dict(zip(self.latest_joint_state.name, self.latest_joint_state.position))
        target_positions = []
        for name in HEAD_NAMES + TORSO_NAMES:
            if name in pos_map:
                target_positions.append(pos_map[name])
            else:
                target_positions.append(0.0)

        target_positions.extend(right_pred.tolist())
        target_positions.extend(left_pred.tolist())

        return CONTROL_NAMES, np.asarray(target_positions, dtype=np.float32)

    def publish_target(self, joint_names, positions):
        traj = JointTrajectory()
        traj.header.stamp = rospy.Time.now()
        traj.joint_names = joint_names
        point = JointTrajectoryPoint()
        point.positions = list(float(v) for v in positions)
        point.velocities = [0.0] * len(joint_names)
        point.accelerations = [0.0] * len(joint_names)
        point.effort = [0.0] * len(joint_names)
        point.time_from_start = rospy.Duration(2.0)
        traj.points = [point]
        self.pub.publish(traj)

    def run(self, rate_hz=5.0, loop_count=None):
        rate = rospy.Rate(rate_hz)
        steps = 0

        while not rospy.is_shutdown():
            if self.latest_color is None or self.latest_depth is None or self.latest_joint_state is None:
                rate.sleep()
                continue

            img = preprocess_rgbd(self.latest_color, self.latest_depth, img_size=self.img_size)
            joint_vec = self.build_joint_input()

            x = torch.from_numpy(img[None, ...]).float().to(self.device)
            j = torch.from_numpy(joint_vec[None, ...]).float().to(self.device)

            with torch.no_grad():
                y_img, y_joint, _, _, self.state = self.model(x, j, self.state)

            pred_joint_28 = y_joint[0].detach().cpu().numpy()
            joint_names, positions = self.build_command(pred_joint_28)
            self.publish_target(joint_names, positions)

            steps += 1
            if loop_count is not None and steps >= loop_count:
                break
            rate.sleep()


def main():
    parser = argparse.ArgumentParser(description="Run a trained MTRNN RGBD model on a robot in real time.")
    parser.add_argument("--model", type=str, required=True, help="checkpoint path, e.g. /home/d-airec/model_MTRNN/MTRNN_20261027_1334/MTRNN1000.pth")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="torch device")
    parser.add_argument("--img_size", type=int, default=32, help="same image size used in training")
    parser.add_argument("--rate", type=float, default=1.0, help="publish rate [Hz], set lower for slower motion")
    parser.add_argument("--loop", type=int, default=None, help="how many cycles to run; None = infinite")
    args = parser.parse_args()

    rospy.init_node("mtrnn_robot_controller", anonymous=True)
    controller = RobotMTRNNController(args.model, args.device, img_size=args.img_size)
    rospy.loginfo("MTRNN robot controller initialized")
    controller.run(rate_hz=args.rate, loop_count=args.loop)


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
