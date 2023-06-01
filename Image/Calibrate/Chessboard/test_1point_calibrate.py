import argparse
import cv2
import numpy as np
import json
import os
from tqdm import tqdm
import math


def test_rotation_z(args):

    # load cameraMatrix
    cv_file = cv2.FileStorage(args.mtx_file, cv2.FILE_STORAGE_READ)
    cameraMatrix = cv_file.getNode("cameraMatrix").mat()
    cv_file.release()

    # load distCoeffs
    cv_file = cv2.FileStorage(args.dist_file, cv2.FILE_STORAGE_READ)
    distCoeffs = cv_file.getNode("distCoeffs").mat()
    cv_file.release()

    # json
    json_path = args.test_point_json_path
    with open(json_path, 'r', encoding='UTF-8') as fr:
        annotation = json.load(fr)

    # load cp_world & cp_img
    cp_world = np.zeros((len(args.point_1_corner_name), 3), dtype = np.float32)
    cp_img = np.zeros((len(args.point_1_corner_name), 1, 2), dtype = np.float32)

    for track in annotation['shapes']:
        label = track['label']
        points = np.array(track['points'])

        for idx in range(len(args.point_1_corner_name)):
            point_1_corner_name = args.point_1_corner_name[idx]

            if label != point_1_corner_name:
                continue
            
            # cp_world_idx
            if 'b' in label:
                cp_world_idx = [float(label.split('b')[1].split('_')[0]), float(label.split('b')[1].split('_')[1]), 0]
            elif 'c' in label:
                cp_world_idx = [float(label.split('c')[1].split('_')[0]), float(label.split('c')[1].split('_')[1]), 0]

            # cp_world
            cp_world[idx: idx+1] = np.array(cp_world_idx)
            
            # cp_img_idx
            cp_img_idx = list(points[0])

            # cp_img
            cp_img[idx: idx+1] = np.array(cp_img_idx).reshape((1,1,2))

    print(cp_world)
    print(cp_img)

    ##########################################################
    # 旋转矩阵归零 -> 使用一个点求俯仰角（无畸变）
    ##########################################################  
    # init，已知两个点（无畸变）
    cp_world_ref = np.ones((cp_world[[0]].shape[0], 4), dtype = np.float32)
    cp_world_ref[:, :3] = cp_world[[0]]

    # 像素坐标推到公式（无畸变）
    cp_img_ref = np.ones((cp_img[[0]].shape[0], 3), dtype = np.float32)
    cp_img_ref[:, :2] = cv2.undistortPoints(cp_img[[0]], cameraMatrix, distCoeffs, P=cameraMatrix)

    # 旋转矩阵
    wo_dist_rvecs = np.zeros((3, 1), dtype = np.float32)
    wo_dist_rvecs[0][0] = 10.0       # 任意值
    wo_dist_rvecs[1][0] = 0.0
    wo_dist_rvecs[2][0] = 0.0

    # 平移矩阵
    wo_dist_tvecs = np.zeros((3, 1), dtype = np.float32)
    wo_dist_tvecs[0][0] = 0.0
    wo_dist_tvecs[1][0] = args.tvecs_y
    wo_dist_tvecs[2][0] = args.tvecs_z

    # init，已知旋转矩阵（y轴、z轴）、平移矩阵
    rotation_m, _ = cv2.Rodrigues(wo_dist_rvecs)#罗德里格斯变换
    rotation_t = np.hstack([rotation_m, wo_dist_tvecs])
    print('rotation_t：\n', rotation_t)

    # 利用一个点，求x轴俯仰角（固定 y 轴 z 轴俯仰角，固定偏移量）
    a = np.dot(np.linalg.inv(cameraMatrix) , cp_img_ref[0])
    b = np.dot(rotation_t, cp_world_ref[0])
    a = a / (a[0] / b[0])
    rotation_z = math.acos((a[1] - wo_dist_tvecs[1]) / cp_world_ref[0][1])
    # rotation_z = math.asin((a[2] - wo_dist_tvecs[2]) / cp_world_ref[0][0])
    print(rotation_z)

    wo_dist_rvecs[0][0] = rotation_z
    print("rotation vectors:", wo_dist_rvecs)
    print("translation vectors:", wo_dist_tvecs)

    # img
    img = cv2.imread(args.test_point_img_path)

    # 投影
    img_point, _ = cv2.projectPoints(cp_world, wo_dist_rvecs, wo_dist_tvecs, cameraMatrix, distCoeffs)

    # 绘图
    for idy in range(len(img_point)):
        try:
            if int(img_point[idy][0][0]) >= 0 and int(img_point[idy][0][0]) <= img.shape[1] and int(img_point[idy][0][1]) >= 0 and int(img_point[idy][0][1]) <= img.shape[0]:
                cv2.circle(img, (int(img_point[idy][0][0]), int(img_point[idy][0][1])), 5, (0, 255, 0), 10)
                cv2.putText(img, "({:.1f},{:.1f})".format(float(cp_world[idy][0]), float(cp_world[idy][1])), (int(img_point[idy][0][0] + 10), int(img_point[idy][0][1]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except:
            pass

    return img, wo_dist_rvecs, wo_dist_tvecs, cameraMatrix, distCoeffs


def projectPoints(args, corner_name_list, img, rvecs, tvecs, cameraMatrix, distCoeffs):

    # json
    json_path = args.test_point_json_path
    with open(json_path, 'r', encoding='UTF-8') as fr:
        annotation = json.load(fr)

    # load calibrate_cp_world & calibrate_cp_img
    calibrate_cp_world = np.zeros((len(corner_name_list), 3), dtype = np.float32)
    calibrate_cp_img = np.zeros((len(corner_name_list), 1, 2), dtype = np.float32)

    for track in annotation['shapes']:
        label = track['label']
        points = np.array(track['points'])

        for idx in range(len(corner_name_list)):
            calibrate_corner_name = corner_name_list[idx]

            if label != calibrate_corner_name:
                continue
            
            # calibrate_cp_world_idx
            if 'b' in label:
                calibrate_cp_world_idx = [float(label.split('b')[1].split('_')[0]), float(label.split('b')[1].split('_')[1]), 0]
            elif 'c' in label:
                calibrate_cp_world_idx = [float(label.split('c')[1].split('_')[0]), float(label.split('c')[1].split('_')[1]), 0]

            # cp_world
            calibrate_cp_world[idx: idx+1] = np.array(calibrate_cp_world_idx)
            
            # calibrate_cp_img_idx
            calibrate_cp_img_idx = list(points[0])

            # calibrate_cp_img
            calibrate_cp_img[idx: idx+1] = np.array(calibrate_cp_img_idx).reshape((1,1,2))

    # print(calibrate_cp_world)
    # print(calibrate_cp_img)

    # 投影
    img_point, _ = cv2.projectPoints(calibrate_cp_world, rvecs, tvecs, cameraMatrix, distCoeffs)
    error = cv2.norm(calibrate_cp_img, img_point, cv2.NORM_L2) / len(img_point)
    print("error:", error)

    # 绘图
    for idy in range(len(img_point)):
        cv2.circle(img, (int(calibrate_cp_img[idy][0][0]), int(calibrate_cp_img[idy][0][1])), 5, (255, 0, 0), 10)
        if int(img_point[idy][0][0]) >= 0 and int(img_point[idy][0][0]) <= img.shape[1] and int(img_point[idy][0][1]) >= 0 and int(img_point[idy][0][1]) <= img.shape[0]:
            cv2.circle(img, (int(img_point[idy][0][0]), int(img_point[idy][0][1])), 5, (0, 0, 255), 10)
            cv2.line(img, (int(img_point[idy][0][0]), int(img_point[idy][0][1])), (int(calibrate_cp_img[idy][0][0]), int(calibrate_cp_img[idy][0][1])), (0, 0, 255), 1)
            cv2.putText(img, "({:.1f},{:.1f})".format(float(calibrate_cp_world[idy][0]), float(calibrate_cp_world[idy][1])), (int(img_point[idy][0][0] + 10), int(img_point[idy][0][1]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
    return img, error


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.file_dir = "/mnt/huanyuan2/data/image/Calibrate/Chessboard/BM1448/"
    args.mtx_file = os.path.join(args.file_dir, "cameraMatrix.xml" )
    args.dist_file = os.path.join(args.file_dir, "distCoeffs.xml" )

    # 公司棋盘格标定板
    args.chessboard_corner_shape = (5, 5)
    args.chessboard_size_per_grid = 1.0

    args.point_1_corner_name = ['b5_10']
    args.calibrate_corner_name_list = ['b5_0', 'b5_10', 'b0_10', 'b-5_10', 'b-5_0']
    args.feet_corner_name_list = [ ['c0_0', 'c0_5', 'c5_5', 'c-5_5'],
                                   ['b6_0', 'b6_5', 'b6_10', 'b-6_0', 'b-6_5', 'b-6_10'], 
                                   ['b13.6_0', 'b13.6_5', 'b13.6_10', 'b-13.8_0', 'b-13.8_5', 'b-13.8_10'],
                                   ['b14.2_5', 'b14.2_10', 'b-14.4_0', 'b-14.4_5', 'b-14.4_10'],
                                   ['b-11.2_0', 'b-11.2_5', 'b-11.2_10'] ]

    # y轴偏移
    args.tvecs_y = 1.1
    # z轴偏移
    args.tvecs_z = 2.5
    # args.tvecs_z = 2.9

    # args.image_name = "c28_250_high_angle"
    # args.image_name = "c28_250_low_angle"
    # args.image_name = "c28_290_high_angle"
    args.image_name = "c28_290_low_angle"
    args.test_point_img_path = os.path.join(args.file_dir, "test_jpg/point_img/{}.jpg".format(args.image_name))
    args.test_point_json_path = os.path.join(args.file_dir, "test_jpg/point_img/{}.json".format(args.image_name))

    # 利用棋盘格标定板，推到外参
    img, rvecs, tvecs, cameraMatrix, distCoeffs = test_rotation_z(args)
    output_test_solvePnP_img_path = os.path.join(args.file_dir, "test_jpg/1point_img_res/{}_solvePnP.jpg".format(args.image_name))
    cv2.imwrite(output_test_solvePnP_img_path, img)

    # 投影
    calibrate_corner_img, error = projectPoints(args, args.calibrate_corner_name_list, img.copy(), rvecs, tvecs, cameraMatrix, distCoeffs)
    output_test_solvePnP_calibrate_corner_img_path = os.path.join(args.file_dir, "test_jpg/1point_img_res/{}_solvePnP_calibrate_corner_{:.2f}.jpg".format(args.image_name, error))
    cv2.imwrite(output_test_solvePnP_calibrate_corner_img_path, calibrate_corner_img)

    # 投影
    for idx in range(len(args.feet_corner_name_list)):

        feet_corner_img, error = projectPoints(args, args.feet_corner_name_list[idx], img.copy(), rvecs, tvecs, cameraMatrix, distCoeffs)
        output_test_solvePnP_feet_corner_img_path = os.path.join(args.file_dir, "test_jpg/1point_img_res/{}_solvePnP_feet_corner_{}_{:.2f}.jpg".format(args.image_name, idx, error))
        cv2.imwrite(output_test_solvePnP_feet_corner_img_path, feet_corner_img)