#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import csv

import matplotlib.pyplot as plt
from matplotlib import animation

from IPython.display import HTML

from tqdm import tqdm
import time
import os
import sys
import cv2
import json
from PIL import Image
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import Dataset

import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import ConcatDataset

from eipl.model.SARNN_rgbd import SARNN_rgbd
import eipl.utils as eiplutil
from eipl.utils import EarlyStopping #, check_args, set_logdir, tensor2numpy, deprocess_img

# === 追加: Mediapipe Pose ===
import mediapipe as mp
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
from ultralytics import YOLO

yolo_seg = YOLO("yolov8n-seg.pt")


start_end = int(os.getenv("MTRNN_FRAME_LIMIT", "250"))  # 使用するフレーム数

img_size = int(os.getenv("MTRNN_IMG_SIZE", "64"))
FAST_BATCH_SIZE = int(os.getenv("MTRNN_BATCH_SIZE", "4"))
FAST_EPOCHS = int(os.getenv("MTRNN_EPOCHS", "1200"))
FAST_REC_DIM = int(os.getenv("MTRNN_REC_DIM", "32"))
FAST_MAX_TRAIN_SEQS = int(os.getenv("MTRNN_MAX_TRAIN_SEQS", "4"))
FAST_MAX_TEST_SEQS = int(os.getenv("MTRNN_MAX_TEST_SEQS", "2"))
FAST_NUM_WORKERS = int(os.getenv("MTRNN_NUM_WORKERS", "4"))
window_tac = 10
losses = []
losses_test = []

def make_rgbd_using_depth_only(
    color_img,
    depth_img,
    depth_min_mm=0,
    depth_max_mm=3800,
    vmin=0.1,
    vmax=0.9,
):
    if color_img is None or depth_img is None:
        return None

    # --- BGR -> RGB ---
    rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB).astype(np.float32)

    # --- depth ---
    depth_mm = depth_img.astype(np.float32)
    valid_mask = (depth_mm >= depth_min_mm) & (depth_mm <= depth_max_mm)

    # RGB / Depth ともに有効範囲のみ
    rgb *= valid_mask[..., None]

    depth_norm = np.zeros_like(depth_mm, dtype=np.float32)
    # if valid_mask.any():
    #     depth_norm[valid_mask] = (depth_mm[valid_mask] - depth_min_mm) / (depth_max_mm - depth_min_mm)
        #     depth_norm[valid_mask] = depth_norm[valid_mask] * (vmax - vmin) + vmin

    depth_norm = (depth_mm - depth_min_mm) / (depth_max_mm - depth_min_mm)
    depth_norm = np.clip(depth_norm, 0.0, 1.0)

    # --- RGB 正規化 ---
    rgb_chw = np.transpose(rgb, (2, 0, 1))
    rgb_norm = normalization(rgb_chw, (0.0, 255.0), (vmin, vmax))

    # --- 4ch 結合 ---
    rgbd = np.concatenate([rgb_norm, depth_norm[None]], axis=0)

    # --- 64x64 ---
    rgbd_hw = np.transpose(rgbd, (1, 2, 0))
    rgbd_hw = cv2.resize(rgbd_hw, (img_size, img_size), interpolation=cv2.INTER_AREA)
    rgbd = np.transpose(rgbd_hw, (2, 0, 1))

    return rgbd

def main():
    ########## directly setting ##########
    model_path = "../model_MTRNN/MTRNN_" + time.strftime("%Y%m%d_%H%M")
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    try:
        from libs.fullBPTT_rgbd import fullBPTTtrainer
    except:
        sys.path.append("./libs/")
        from fullBPTT_rgbd import fullBPTTtrainer

    ########## model setting ##########
    torch_device = "cuda"  # "cuda" or "cpu"
    data_num = 500
    batch_size = FAST_BATCH_SIZE
    n_epoch = FAST_EPOCHS
    vmax = 1.0
    vmin = 0.0
    stdev = 0.1
    loss_record = []
    min_loss = 10.0
    joint_loss = 1.0
    img_loss = 0.1
    pt_loss = 0.1
    log_dir = "log/"
    tag = "Tag name for snap/log sub directory"
    layer_size = {"in": 2, "out": 2, "io": 10, "cf": 10, "cs": 10}
    tau = {"tau_io": 2.0, "tau_cf": 5.0, "tau_cs": 10.0}
    open_rate = 1


    if torch_device == "cuda" and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    model = SARNN_rgbd(
        rec_dim=FAST_REC_DIM,
        joint_dim=28,  # angles14 + torques14 = 28
        k_dim=5,
        heatmap_size=0.1,
        temperature=1e-4,
        im_size=[img_size, img_size],
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    print(model)

    ########## Loading data ##########
    x_list = []
    y_list = []

    datasets = []
    images_multiple = []
    joint_multiple = []
    directory_path = "/home/d-airec/catkin_ws/src/lecture1026/dataset/train"
    folder_names = [
        entry.name for entry in os.scandir(directory_path) if entry.is_dir()
    ]
    folder_names = sorted(folder_names)
    if FAST_MAX_TRAIN_SEQS > 0:
        folder_names = folder_names[:FAST_MAX_TRAIN_SEQS]
    print(folder_names)
    directory_path_test = "/home/d-airec/catkin_ws/src/lecture1026/dataset/test"
    folder_names_test = [
        entry.name for entry in os.scandir(directory_path_test) if entry.is_dir()
    ]
    folder_names_test = sorted(folder_names_test)
    if FAST_MAX_TEST_SEQS > 0:
        folder_names_test = folder_names_test[:FAST_MAX_TEST_SEQS]

    print(
        f"[FAST TRAIN] img_size={img_size}, frame_limit={start_end}, rec_dim={FAST_REC_DIM}, "
        f"batch_size={batch_size}, epochs={n_epoch}, train_seqs={len(folder_names)}, "
        f"test_seqs={len(folder_names_test)}"
    )

    # ---- train data ----
    for folder_name in folder_names:
        images, joints, touches = load_data(directory_path, folder_name)
        images_multiple.append(images)
        joint_multiple.append(joints)
        #mask_multiple.append(masks)
        print("a")

    # フォルダごとの系列長が異なる場合は最小長にそろえる
    train_seq_lens = [arr.shape[0] for arr in images_multiple]
    min_train_len = min(train_seq_lens)
    if min_train_len <= 0:
        raise ValueError("Train data contains an empty sequence")
    if len(set(train_seq_lens)) != 1:
        print(f"[INFO] Train sequence lengths differ: {train_seq_lens} -> trim to {min_train_len}")
    images_multiple = [arr[:min_train_len] for arr in images_multiple]
    joint_multiple = [arr[:min_train_len] for arr in joint_multiple]

    images = np.stack(images_multiple, axis=0)
    joints = np.stack(joint_multiple, axis=0)
    #masks  = np.stack(mask_multiple, axis=0)
    joint_max = np.max(joints, axis=(0, 1))
    joint_min = np.min(joints, axis=(0, 1))

    # CSVファイルに保存（確認用）
    filename = os.path.join(model_path, "inputjointdata.csv")
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for joint in joints[0]:
            writer.writerow(joint)

    # normalization only for torque elements (28要素全部に適用している)
    for i in range(len(joints)):
        for j in range(len(joints[i])):
            for k in range(28):
                joints[i][j][k] = normalization(
                    joints[i][j][k],
                    (joint_min[k], joint_max[k]),
                    (vmin, vmax),
                )

    print(images.shape, joints.shape)

    # save hyper parameters
    hyperparam = {"joint_max": joint_max.tolist(), "joint_min": joint_min.tolist()}
    json_data = json.dumps(hyperparam, indent=1)
    file_path = os.path.join(model_path, "data.json")
    with open(file_path, "w") as json_file:
        json_file.write(json_data)

    images_multiple.clear
    joint_multiple.clear

    train_dataset = MultimodalDataset(images, joints, stdev=stdev, training=True)

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=(torch_device == "cuda" and torch.cuda.is_available()),
        num_workers=max(0, FAST_NUM_WORKERS),
        persistent_workers=(FAST_NUM_WORKERS > 0),
    )
    loss_weights = [img_loss, joint_loss, pt_loss]
    trainer = fullBPTTtrainer(
        model, optimizer, loss_weights=loss_weights, device=torch_device
    )

    # ---- test data ----
    images_multiple_test = []
    joint_multiple_test = []
    for folder_name in folder_names_test:
        images, joints, touches = load_data(directory_path_test, folder_name)
        images_multiple_test.append(images)
        joint_multiple_test.append(joints)
        print("a")

    # テストデータも同様に系列長をそろえる
    test_seq_lens = [arr.shape[0] for arr in images_multiple_test]
    min_test_len = min(test_seq_lens)
    if min_test_len <= 0:
        raise ValueError("Test data contains an empty sequence")
    if len(set(test_seq_lens)) != 1:
        print(f"[INFO] Test sequence lengths differ: {test_seq_lens} -> trim to {min_test_len}")
    images_multiple_test = [arr[:min_test_len] for arr in images_multiple_test]
    joint_multiple_test = [arr[:min_test_len] for arr in joint_multiple_test]

    images = np.stack(images_multiple_test, axis=0)
    joints = np.stack(joint_multiple_test, axis=0)

    # normalization only for torque elements
    for i in range(len(joints)):
        for j in range(len(joints[i])):
            for k in range(28):
                joints[i][j][k] = normalization(
                    joints[i][j][k],
                    (joint_min[k], joint_max[k]),
                    (vmin, vmax),
                )

    test_dataset = MultimodalDataset(images, joints, stdev=stdev, training=True)
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=(torch_device == "cuda" and torch.cuda.is_available()),
        num_workers=max(0, FAST_NUM_WORKERS),
        persistent_workers=(FAST_NUM_WORKERS > 0),
    )

    images_multiple_test.clear
    joint_multiple_test.clear
    del images, joints

    ########## training ##########
    start_time = time.time()

    save_name = os.path.join(model_path, "MTRNN.pth")
    writer = SummaryWriter(log_dir=model_path, flush_secs=30)
    early_stop = EarlyStopping(patience=3000)

    with tqdm(range(n_epoch)) as pbar_epoch:
        for epoch in pbar_epoch:
            train_loss = trainer.process_epoch(train_loader)
            losses.append(train_loss)
            test_loss = trainer.process_epoch(test_loader, training=False)
            writer.add_scalar("Loss/train_loss", train_loss, epoch)
            losses_test.append(test_loss)
            trainer.save(epoch, [train_loss, test_loss], save_name)
            pbar_epoch.set_postfix(OrderedDict(train_loss=train_loss))

    end_time = time.time()
    learning_time = end_time - start_time
    print("learning time : {:.2f}[sec]".format(learning_time))

    filename = os.path.join(model_path, "losses.csv")
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Epoch", "Loss"])
        for epoch, loss in enumerate(losses, 1):
            writer.writerow([epoch, loss])

    filename = os.path.join(model_path, "losses_test.csv")
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Epoch", "Loss"])
        for epoch, loss in enumerate(losses_test, 1):
            writer.writerow([epoch, loss])



def load_data(directory_path, folder_name, vmin=0.1, vmax=0.9):
    angle_path = os.path.join(directory_path, folder_name, "angle.csv")
    torque_path = os.path.join(directory_path, folder_name, "torque.csv")
    color_dir = os.path.join(directory_path, folder_name, "color")  # ★ fisheye → color
    depth_dir = os.path.join(directory_path, folder_name, "depth")

    # --- Joint data ---
    with open(angle_path) as f:
        angle_data = list(csv.reader(f))[0:start_end]
        angles = [
            [float(item) for item in row[10:17] + row[27:34]]
            for row in angle_data
        ]

    with open(torque_path) as f:
        torque_data = list(csv.reader(f))[0:start_end]
        torques = [
            [float(item) for item in row[10:17] + row[27:34]]
            for row in torque_data
        ]


    if len(angles) != len(torques) :
        raise ValueError(
            f"Mismatch: {len(angles)} angles vs {len(torques)} torques in {folder_name}"
        )
    if len(angles) == 0 or len(torques) == 0 :
        raise ValueError("One of the input lists is empty")
    joints_1D = [a + t for a, t in zip(angles, torques)]
    joints = np.stack(joints_1D, axis=0)

    # --- color 画像ファイル一覧 ---
    color_files = sorted(
        [f for f in os.listdir(color_dir) if f.endswith(".png") or f.endswith(".jpg")],
        key=lambda x: int(os.path.splitext(x)[0]),
    )

    images = []
    for image_file in color_files[0:start_end]:
        color_path = os.path.join(color_dir, image_file)
        depth_path = os.path.join(depth_dir, image_file)

        color_img = cv2.imread(color_path, cv2.IMREAD_COLOR)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        # fisheye 回転をそろえる
        # color_rot = cv2.rotate(color_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # depth_rot = cv2.rotate(depth_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        H, W = color_img.shape[:2]

        # RGB
        color_img = color_img.astype(np.float32)

        # Depth
        depth_img = depth_img.astype(np.float32)

        rgbd = make_rgbd_using_depth_only(
            color_img,
            depth_img,
            depth_min_mm=0,
            depth_max_mm=2500,
            vmin=vmin,
            vmax=vmax,
        )

        images.append(rgbd)


    images = np.stack(images, axis=0)
    print(f"[DEPTH-ONLY] Loaded {folder_name}: images={images.shape}, joints={joints.shape}")
    touches = []

    return images, joints, touches


def normalization(data, indataRange, outdataRange):
    data = (data - indataRange[0]) / (indataRange[1] - indataRange[0])
    data = data * (outdataRange[1] - outdataRange[0]) + outdataRange[0]
    return data


def reshape_array(arr):
    original_shape = arr.shape
    new_shape = (original_shape[0], original_shape[1], original_shape[2] // 2, 2)
    print(new_shape)
    reshaped_arr = np.empty(new_shape)
    for i in range(original_shape[0]):
        for j in range(original_shape[1]):
            for k in range(original_shape[2] // 2):
                reshaped_arr[i, j, k, 0] = arr[i, j, k]
                reshaped_arr[i, j, k, 1] = arr[i, j, k + original_shape[2] // 2]
    return reshaped_arr

class MultimodalDataset(Dataset):
    def __init__(self, images, joints, stdev=0.0, training=True):
        self.images = torch.from_numpy(images).float()
        self.joints = torch.from_numpy(joints).float()
        self.stdev = stdev
        self.training = training
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        y_img = self.images[idx]
        y_joint = self.joints[idx]

        if self.training:
            x_img = y_img + torch.normal(0, self.stdev, y_img.shape)
            x_joint = y_joint + torch.normal(0, self.stdev, y_joint.shape)
        else:
            x_img = y_img
            x_joint = y_joint

        return [[x_img, x_joint], [y_img, y_joint]]

if __name__ == "__main__":
    main()