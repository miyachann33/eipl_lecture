#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from eipl.model.SARNN_rgbd import SARNN_rgbd
from MTRNN_learning_rgbd import load_data, img_size  # 学習時と同じ関数を利用

pred_joints_denorm = []   # デノーマライズ後 joint の時系列
pred_images_list = []     # 予測画像 (正規化された4ch) の時系列

enc_pts_list = []         # エンコーダ注意点
dec_pts_list = []         # デコーダ注意点

state = None
y_image = None
y_joint = None


# ---------- ユーティリティ ----------

def normalize(x, in_min, in_max, out_min, out_max):
    """学習時と同じ線形正規化."""
    return (x - in_min) / (in_max - in_min) * (out_max - out_min) + out_min


def denormalize(x, in_min, in_max, out_min, out_max):
    """normalize の逆変換."""
    return (x - out_min) / (out_max - out_min) * (in_max - in_min) + in_min


def img_denorm(x, vmin=0.1, vmax=0.9):
    """
    load_data() で [vmin, vmax] に正規化された画像を
    [0,1] に戻して可視化用にする。
    x: (..., C, H, W)
    """
    x = (x - vmin) / (vmax - vmin)
    return np.clip(x, 0.0, 1.0)


# ---------- メイン ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename",
        type=str,
        required=True,
        help="学習済みモデル(MTRNN.pth)へのパス",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/home/d-airec/catkin_ws/src/lecture1026/dataset/",
        help="RGBD＋jointデータのフォルダ（各シーケンスの親ディレクトリ）",
    )
    parser.add_argument(
        "--folder_name",
        type=str,
        required=True,
        help="確認に使うシーケンスフォルダ名（例: 2025-07-25-13-59-41）",
    )
    parser.add_argument(
        "--input_param",
        type=float,
        default=1.0,
        help="閉ループ係数 (1.0=完全オープンループ, 0.0=完全クローズループ)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda または cpu",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---------- モデルパスと data.json の読み込み ----------
    ckpt_path = os.path.abspath(args.filename)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint が見つかりません: {ckpt_path}")

    model_dir = os.path.dirname(ckpt_path)
    data_json = os.path.join(model_dir, "data.json")

    if not os.path.exists(data_json):
        raise FileNotFoundError(f"data.json が見つかりません: {data_json}")

    with open(data_json, "r") as f:
        hyper = json.load(f)

    joint_min = np.array(hyper["joint_min"])  # shape: (28,)
    joint_max = np.array(hyper["joint_max"])

    print("joint_min:", joint_min)
    print("joint_max:", joint_max)

    # ---------- データ読み込み ----------
    directory_path = args.dataset_dir
    folder_name = args.folder_name

    # images_np: (T, 4, H, W)  (RGBD, [0.1, 0.9] に正規化済み)
    # joints_np: (T, 28)       (生値)
    images_np, joints_np, _ = load_data(directory_path, folder_name)
    T, C, H, W = images_np.shape
    print(f"Loaded {folder_name}: images={images_np.shape}, joints={joints_np.shape}")

    # ---------- 学習済み重み読み込み ----------
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    # ---------- モデル定義（チェックポイント形状から自動復元） ----------
    if "rec.weight_hh" not in state_dict:
        raise KeyError("checkpoint に rec.weight_hh がありません")

    rec_dim = state_dict["rec.weight_hh"].shape[0] // 4
    model = SARNN_rgbd(
        rec_dim=rec_dim,
        joint_dim=len(joint_min),
        k_dim=5,
        heatmap_size=0.1,
        temperature=1e-4,
        im_size=[img_size, img_size],
    ).to(device)

    # fullBPTTtrainer.save() の形式に合わせて 'model_state_dict' から読む
    model.load_state_dict(state_dict)

    model.eval()
    print("Loaded checkpoint from:", ckpt_path)

    # ---------- 推論ループ ----------
    vmin_joint, vmax_joint = 0.0, 1.0  # joint 正規化スケール
    vmin_img, vmax_img = 0.1, 0.9      # load_data で使っている画像のスケール

    pred_joints_denorm = []   # デノーマライズ後 joint の時系列
    pred_images_list = []     # 予測画像 (正規化された4ch) の時系列

    state = None
    y_image = None
    y_joint = None

    for t in range(T):
        # 画像: すでに [0.1,0.9] に正規化されているのでそのまま使用
        img_t_np = images_np[t]  # (4, H, W)
        img_t = torch.from_numpy(img_t_np[None, ...]).float().to(device)  # (1, 4, H, W)

        # joints: 生値 -> [0,1] に正規化してから入力
        joint_raw = joints_np[t]  # (28,)
        joint_norm = normalize(
            joint_raw,
            joint_min,
            joint_max,
            vmin_joint,
            vmax_joint,
        )
        joint_t = torch.from_numpy(joint_norm[None, ...]).float().to(device)  # (1, 28)

        # 閉ループ (input_param < 1 のとき)
        if t > 0 and args.input_param < 1.0:
            img_t = args.input_param * img_t + (1.0 - args.input_param) * y_image
            joint_t = args.input_param * joint_t + (1.0 - args.input_param) * y_joint

        # SARNN_rgbd の1ステップ推論
        # インターフェイス: (img, joint, state) -> y_image, y_joint, enc_pts, dec_pts, state
        with torch.no_grad():
            y_image, y_joint, enc_pts, dec_pts, state = model(img_t, joint_t, state)
        # 注意点（enc/dec）も保存
        enc_pts_list.append(
            enc_pts[0].detach().cpu().numpy()
        )  # 形状想定: (k_dim*2,) or (k_dim,2)
        dec_pts_list.append(
            dec_pts[0].detach().cpu().numpy()
        )
        # joint のデノーマライズ
        y_joint_np = y_joint[0].detach().cpu().numpy()  # (28,)
        joint_denorm = denormalize(
            y_joint_np,
            joint_min,
            joint_max,
            vmin_joint,
            vmax_joint,
        )
        pred_joints_denorm.append(joint_denorm)

        # 画像の予測（まだ [0.1,0.9] スケール）
        y_image_np = y_image[0].detach().cpu().numpy()  # (4, H, W)
        pred_images_list.append(y_image_np)

        # print(f"t={t}, pred_joint[0:7]={joint_denorm[:7]}")

    pred_joints_denorm = np.stack(pred_joints_denorm, axis=0)   # (T, 28)
    pred_images_np = np.stack(pred_images_list, axis=0)         # (T, 4, H, W)

    # ---------- 注意点（enc/dec）を (T, k_dim, 2) 座標に変換 ----------
    k_dim = 5  # SARNN_rgbd の初期化で k_dim=5 にしているのでそれに合わせる

    enc_pts_np = np.stack(enc_pts_list, axis=0)  # 形状: (T, k_dim*2) or (T, k_dim, 2)
    dec_pts_np = np.stack(dec_pts_list, axis=0)

    # もし (T, k_dim*2) なら (T, k_dim, 2) に並び替え
    enc_pts_np = enc_pts_np.reshape(-1, k_dim, 2) * img_size
    dec_pts_np = dec_pts_np.reshape(-1, k_dim, 2) * img_size

    enc_pts_np = np.clip(enc_pts_np, 0, img_size)
    dec_pts_np = np.clip(dec_pts_np, 0, img_size)

    # ---------- joint の CSV 保存 ----------
    out_csv = os.path.join(model_dir, f"traj_{folder_name}.csv")
    np.savetxt(out_csv, pred_joints_denorm, delimiter=",")
    print("Saved predicted joint trajectory to:", out_csv)

    # ---------- joint の簡易プロット（最初の数関節だけ） ----------
    time_idx = np.arange(T)

    # plt.figure(figsize=(10, 6))
    # for j in range(min(6, pred_joints_denorm.shape[1])):
    #     plt.plot(time_idx, joints_np[:, j], "k--", alpha=0.5, label=f"GT joint{j}" if j == 0 else None)
    #     plt.plot(time_idx, pred_joints_denorm[:, j], label=f"pred joint{j}")
    # plt.xlabel("Time step")
    # plt.ylabel("Joint value")
    # plt.title(f"GT vs Predicted joints ({folder_name})")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # ===============================
    # 右腕角度（0-6）
    # ===============================
    plt.figure(figsize=(10, 6))
    for j in range(7):
        plt.plot(time_idx, joints_np[:, j], "k--", alpha=0.4)
        plt.plot(time_idx, pred_joints_denorm[:, j], label=f"R joint{j}")

    plt.title(f"Right Arm Angles ({folder_name})")
    plt.xlabel("Time step")
    plt.ylabel("Angle")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # ===============================
    # 左腕角度（7-13）
    # ===============================
    plt.figure(figsize=(10, 6))
    for j in range(7):
        idx = j + 7
        plt.plot(time_idx, joints_np[:, idx], "k--", alpha=0.4)
        plt.plot(time_idx, pred_joints_denorm[:, idx], label=f"L joint{j}")

    plt.title(f"Left Arm Angles ({folder_name})")
    plt.xlabel("Time step")
    plt.ylabel("Angle")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # ===============================
    # インピーダンス（29次元目がある場合のみ）
    # ===============================
    if joints_np.shape[1] > 28 and pred_joints_denorm.shape[1] > 28:
        plt.figure(figsize=(8, 4))

        plt.plot(time_idx, joints_np[:, 28], "k--", label="GT impedance")
        plt.plot(time_idx, pred_joints_denorm[:, 28], label="Pred impedance")

        plt.title(f"Impedance ({folder_name})")
        plt.xlabel("Time step")
        plt.ylabel("Impedance")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    # ---------- 画像の可視化（2段構成: 上段 RGB, 下段 Depth(gray)） ----------
    # 入力画像・予測画像を [0,1] に戻して可視化用に整形
    images_vis = img_denorm(images_np, vmin=vmin_img, vmax=vmax_img)      # (T, 4, H, W)
    pred_vis   = img_denorm(pred_images_np, vmin=vmin_img, vmax=vmax_img) # (T, 4, H, W)

    # RGB: ch=0..2, Depth: ch=3
    images_rgb   = np.transpose(images_vis[:, 0:3, :, :], (0, 2, 3, 1))  # (T, H, W, 3)
    pred_rgb     = np.transpose(pred_vis[:, 0:3, :, :], (0, 2, 3, 1))
    images_depth = images_vis[:, 3, :, :]                                # (T, H, W)
    pred_depth   = pred_vis[:, 3, :, :]
    all_depth = np.concatenate([images_depth.reshape(-1), pred_depth.reshape(-1)])
    d_min = float(all_depth.min())
    d_max = float(all_depth.max())
    eps = 1e-8

    images_depth_vis = (images_depth - d_min) / (d_max - d_min + eps)
    pred_depth_vis   = (pred_depth   - d_min) / (d_max - d_min + eps)

    T_anim = images_rgb.shape[0]
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), dpi=60)

    def anim_update(i):
        # 全てのAxisをクリア
        for r in range(2):
            for c in range(2):
                ax[r, c].cla()

        # 上段：RGB
        ax[0, 0].imshow(images_rgb[i])
        ax[0, 0].set_title("Input RGB")
        ax[0, 0].axis("off")

        ax[0, 1].imshow(pred_rgb[i])
        ax[0, 1].set_title("Predicted RGB")
        ax[0, 1].axis("off")

        # ---- ここから 注意点を重ね描き ----
        for j in range(k_dim):
            # enc_pts_np は Encoder の注意点（青丸）
            ax[0, 0].plot(
                enc_pts_np[i, j, 0], enc_pts_np[i, j, 1],
                "bo", markersize=4
            )
            # dec_pts_np は Decoder の注意点（赤バツ）
            ax[0, 1].plot(
                dec_pts_np[i, j, 0], dec_pts_np[i, j, 1],
                "rx", markersize=6, markeredgewidth=2
            )
        # 下段：Depth (gray)
        ax[1, 0].imshow(images_depth_vis[i], cmap="gray", vmin=0.0, vmax=1.0)
        ax[1, 0].set_title("Input Depth")
        ax[1, 0].axis("off")

        ax[1, 1].imshow(pred_depth_vis[i], cmap="gray", vmin=0.0, vmax=1.0)
        ax[1, 1].set_title("Predicted Depth")
        ax[1, 1].axis("off")

        fig.suptitle(f"Step {i+1}/{T_anim}")
        fig.tight_layout()

        return []

    ani = anim.FuncAnimation(
        fig,
        anim_update,
        frames=T_anim,
        interval=10,  # 100ms = 10fps くらい
        blit=False,
    )

    # 画面で見るだけならこれでOK
    plt.show()

    # GIF に保存したい場合はコメントアウト解除
    # gif_path = os.path.join(model_dir, f"rgbd_{folder_name}.gif")
    # ani.save(gif_path)
    # print("Saved RGBD animation to:", gif_path)


if __name__ == "__main__":
    main()
