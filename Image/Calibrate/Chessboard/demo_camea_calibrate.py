"""
Homework: Calibrate the Camera with ZhangZhengyou Method.
"""
import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm
import math

def calibrate(args):
    w, h = args.chessboard_corner_shape
    cp_int = np.zeros((w * h, 3), np.float32)
    cp_int[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
    cp_world = cp_int * args.chessboard_size_per_grid

    # init
    obj_points = []  # the points in world space
    img_points = []  # the points in image space (relevant to obj_points)
    
    # img list
    img_list = np.array(os.listdir(args.chessboard_img_dir))
    img_list = img_list[[img.endswith(args.suffix) for img in img_list]]
    img_list.sort()

    # load obj_points & img_points
    for idx in tqdm(range(len(img_list))):
        
        img_name = img_list[idx]
        img_path = os.path.join(args.chessboard_img_dir, img_name)

        # img
        img = cv2.imread(img_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # findChessboardCorners
        ret, cp_img = cv2.findChessboardCorners(gray_img, (w, h), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS)

        # if ret is True, save.
        if ret == True:
            obj_points.append(cp_world)
            img_points.append(cp_img)

            # view the corners
            cv2.drawChessboardCorners(img, (w, h), cp_img, ret)

            # mkdir 
            if not os.path.exists(args.output_chessboard_corners_img_dir):
                os.mkdir(args.output_chessboard_corners_img_dir)

            # output img
            output_img_path = os.path.join(args.output_chessboard_corners_img_dir, img_name)
            cv2.imwrite(output_img_path, img)
        else:
            print(img_name)
            os.remove(img_path)

    # calibrate the camera
    ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray_img.shape[::-1], None, None)

    print("ret:", ret)
    print("internal matrix:", cameraMatrix)
    print("distortion cofficients:", distCoeffs)  # in the form of (k_1,k_2,p_1,p_2,k_3)
    print("rotation vectors:", rvecs)
    print("translation vectors:", tvecs)

    # save cameraMatrix
    cv_file = cv2.FileStorage(args.mtx_file, cv2.FILE_STORAGE_WRITE)
    cv_file.write('cameraMatrix', cameraMatrix)
    cv_file.release()

    # save distCoeffs
    cv_file = cv2.FileStorage(args.dist_file, cv2.FILE_STORAGE_WRITE)
    cv_file.write('distCoeffs', distCoeffs)
    cv_file.release()

    # calculate the error of reproject
    total_error = 0
    for idx in tqdm(range(len(img_list))):

        img_name = img_list[idx]
        img_path = os.path.join(args.chessboard_img_dir, img_name)
        img = cv2.imread(img_path)

        img_points_repro, _ = cv2.projectPoints(obj_points[idx], rvecs[idx], tvecs[idx], cameraMatrix, distCoeffs)
        error = cv2.norm(img_points[idx], img_points_repro, cv2.NORM_L2) / len(img_points_repro)
        total_error += error

        # output img
        # cp_int = np.zeros((4 * w * 4 * h, 3), np.float32)
        # cp_int[:, :2] = np.mgrid[-2*w:2*w, -2*h:2*h].T.reshape(-1, 2)

        cp_int = np.zeros((2 * w * 2 * h, 3), np.float32)
        cp_int[:, :2] = np.mgrid[-1*w:1*w, -1*h:1*h].T.reshape(-1, 2)
        cp_int = cp_int + [int(w/2), int(h/2), 0]

        img_world = cp_int * args.chessboard_size_per_grid
        img_point, _ = cv2.projectPoints(img_world, rvecs[idx], tvecs[idx], cameraMatrix, distCoeffs)
        for idy in range(len(img_point)):
            if int(img_point[idy][0][0]) >= 0 and int(img_point[idy][0][0]) <= img.shape[1] and int(img_point[idy][0][1]) >= 0 and int(img_point[idy][0][1]) <= img.shape[0]:
                cv2.circle(img, (int(img_point[idy][0][0]), int(img_point[idy][0][1])), 5, (0, 0, 255), 10)

        # mkdir 
        if not os.path.exists(args.output_calibrate_img_dir):
            os.mkdir(args.output_calibrate_img_dir)

        img_name = img_list[idx]
        output_img_path = os.path.join(args.output_calibrate_img_dir, img_name)
        cv2.imwrite(output_img_path, img)

    print(("Average Error of Reproject: "), total_error / len(obj_points))


def solvePnP(args):

    w, h = args.chessboard_corner_shape
    cp_int = np.zeros((w * h, 3), np.float32)
    cp_int[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
    cp_world = cp_int * args.chessboard_size_per_grid

    # init
    obj_points = []  # the points in world space
    img_points = []  # the points in image space (relevant to obj_points)

    # img list
    img_list = np.array(os.listdir(args.chessboard_img_dir))
    img_list = img_list[[img.endswith(args.suffix) for img in img_list]]
    img_list.sort()

    # load obj_points & img_points
    for idx in tqdm(range(len(img_list))):
        
        img_name = img_list[idx]
        img_path = os.path.join(args.chessboard_img_dir, img_name)

        # img
        img = cv2.imread(img_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # findChessboardCorners
        ret, cp_img = cv2.findChessboardCorners(gray_img, (w, h), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS)

        # if ret is True, save.
        if ret == True:
            obj_points.append(cp_world)
            img_points.append(cp_img)
        else:
            print(img_name)
            # os.remove(img_name)

    # load cameraMatrix
    cv_file = cv2.FileStorage(args.mtx_file, cv2.FILE_STORAGE_READ)
    cameraMatrix = cv_file.getNode("cameraMatrix").mat()
    cv_file.release()

    # load distCoeffs
    cv_file = cv2.FileStorage(args.dist_file, cv2.FILE_STORAGE_READ)
    distCoeffs = cv_file.getNode("distCoeffs").mat()
    cv_file.release()

    # calculate the error of reproject
    total_error = 0
    for idx in tqdm(range(len(img_list))):

        img_name = img_list[idx]
        img_path = os.path.join(args.chessboard_img_dir, img_name)
        img = cv2.imread(img_path)

        # PnP 算法获取旋转矩阵和平移向量
        # # 计算外参：所有棋盘格点
        # _, rvecs, tvecs = cv2.solvePnP(obj_points[idx], img_points[idx], cameraMatrix, distCoeffs)
        # 计算外参：4个棋盘格点
        _, rvecs, tvecs = cv2.solvePnP(np.array(obj_points[idx][[27,28,29,38,39,40,49,50,51], :]), np.array(img_points[idx][[27,28,29,38,39,40,49,50,51], :]), cameraMatrix, distCoeffs)
        # # 计算外参：4个棋盘格点
        # _, rvecs, tvecs = cv2.solvePnP(np.array(obj_points[idx][[11,23,30,47], :]), np.array(img_points[idx][[11,23,30,47], :]), cameraMatrix, distCoeffs)
        # 计算外参：4个棋盘格点 P3P
        # _, rvecs, tvecs = cv2.solvePnP(np.array(obj_points[idx][[11,23,30,47], :]), np.array(img_points[idx][[11,23,30,47], :]), cameraMatrix, distCoeffs, flags = 3)

        img_points_repro, _ = cv2.projectPoints(obj_points[idx], rvecs, tvecs, cameraMatrix, distCoeffs)
        error = cv2.norm(img_points[idx], img_points_repro, cv2.NORM_L2) / len(img_points_repro)
        total_error += error

        # 罗德里格斯变换
        rotation_m, _ = cv2.Rodrigues(rvecs) 
        rotation_t = np.hstack([rotation_m, tvecs])
        rotation_t_Homogeneous_matrix = np.vstack([rotation_t, np.array([[0, 0, 0, 1]])])
        print('旋转矩阵是：\n', rvecs)
        print('平移矩阵是:\n', tvecs)
        print('Homogeneous_matrix:\n', rotation_t_Homogeneous_matrix)

        # output img
        # cp_int = np.zeros((4 * w * 4 * h, 3), np.float32)
        # cp_int[:, :2] = np.mgrid[-2*w:2*w, -2*h:2*h].T.reshape(-1, 2)
        
        cp_int = np.zeros((2 * w * 2 * h, 3), np.float32)
        cp_int[:, :2] = np.mgrid[-1*w:1*w, -1*h:1*h].T.reshape(-1, 2)
        cp_int = cp_int + [int(w/2), int(h/2), 0]

        img_world = cp_int * args.chessboard_size_per_grid
        img_point, _ = cv2.projectPoints(img_world, rvecs, tvecs, cameraMatrix, distCoeffs)
        for idy in range(len(img_point)):
            try:
                if int(img_point[idy][0][0]) >= 0 and int(img_point[idy][0][0]) <= img.shape[1] and int(img_point[idy][0][1]) >= 0 and int(img_point[idy][0][1]) <= img.shape[0]:
                    cv2.circle(img, (int(img_point[idy][0][0]), int(img_point[idy][0][1])), 5, (0, 0, 255), 10)
            except:
                pass
        
        # mkdir 
        if not os.path.exists(args.output_solvePnP_img_dir):
            os.mkdir(args.output_solvePnP_img_dir)

        img_name = img_list[idx]
        output_img_path = os.path.join(args.output_solvePnP_img_dir, img_name)
        cv2.imwrite(output_img_path, img)

    print(("Average Error of Reproject: "), total_error / len(obj_points))


def test_solvePnP(args):

    w, h = args.chessboard_corner_shape
    cp_int = np.zeros((w * h, 3), np.float32)
    cp_int[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
    cp_world = cp_int * args.chessboard_size_per_grid

    # load cameraMatrix
    cv_file = cv2.FileStorage(args.mtx_file, cv2.FILE_STORAGE_READ)
    cameraMatrix = cv_file.getNode("cameraMatrix").mat()
    cv_file.release()

    # load distCoeffs
    cv_file = cv2.FileStorage(args.dist_file, cv2.FILE_STORAGE_READ)
    distCoeffs = cv_file.getNode("distCoeffs").mat()
    cv_file.release()

    # img
    img = cv2.imread(args.test_chessboard_img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # findChessboardCorners
    ret, cp_img = cv2.findChessboardCorners(gray_img, (w, h), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS)

    # output img
    cv2.drawChessboardCorners(img, (w, h), cp_img, ret)
    cv2.imwrite(args.output_test_chessboard_corners_img_path, img)

    # PnP 算法获取旋转矩阵和平移向量
    # 计算外参：所有棋盘格点
    # _, rvecs, tvecs = cv2.solvePnP(cp_world, cp_img, cameraMatrix, distCoeffs)
    # 计算外参：4个棋盘格点
    _, rvecs, tvecs = cv2.solvePnP(np.array(cp_world[[27,28,29,38,39,40,49,50,51], :]), np.array(cp_img[[27,28,29,38,39,40,49,50,51], :]), cameraMatrix, distCoeffs)
    # 计算外参：4个棋盘格点
    # _, rvecs, tvecs = cv2.solvePnP(np.array(cp_world[[11,23,30,47], :]), np.array(cp_img[[11,23,30,47], :]), cameraMatrix, distCoeffs)
    # 计算外参：4个棋盘格点 P3P
    # _, rvecs, tvecs = cv2.solvePnP(np.array(cp_world[[11,23,30,47], :]), np.array(cp_img[[11,23,30,47], :]), cameraMatrix, distCoeffs, flags = 3)
    # print(rvecs)
    # print(tvecs)

    # output img
    # cp_int = np.zeros((4 * w * 4 * h, 3), np.float32)
    # cp_int[:, :2] = np.mgrid[-2*w:2*w, -2*h:2*h].T.reshape(-1, 2)

    cp_int = np.zeros((2 * w * 2 * h, 3), np.float32)
    cp_int[:, :2] = np.mgrid[-1*w:1*w, -1*h:1*h].T.reshape(-1, 2)
    cp_int = cp_int + [int(w/2), int(h/2), 0]
    
    img_world = cp_int * args.chessboard_size_per_grid
    img_point, _ = cv2.projectPoints(img_world, rvecs, tvecs, cameraMatrix, distCoeffs)
    for idy in range(len(img_point)):
        try:
            if int(img_point[idy][0][0]) >= 0 and int(img_point[idy][0][0]) <= img.shape[1] and int(img_point[idy][0][1]) >= 0 and int(img_point[idy][0][1]) <= img.shape[0]:
                cv2.circle(img, (int(img_point[idy][0][0]), int(img_point[idy][0][1])), 5, (0, 0, 255), 10)
        except:
            pass
    
    cv2.imwrite(args.output_test_solvePnP_img_path, img)


def test_undistort(args):

    w, h = args.chessboard_corner_shape
    cp_int = np.zeros((w * h, 3), np.float32)
    cp_int[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
    cp_world = cp_int * args.chessboard_size_per_grid

    # load cameraMatrix
    cv_file = cv2.FileStorage(args.mtx_file, cv2.FILE_STORAGE_READ)
    cameraMatrix = cv_file.getNode("cameraMatrix").mat()
    cv_file.release()

    # load distCoeffs
    cv_file = cv2.FileStorage(args.dist_file, cv2.FILE_STORAGE_READ)
    distCoeffs = cv_file.getNode("distCoeffs").mat()
    cv_file.release()

    # img
    img = cv2.imread(args.test_chessboard_img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # findChessboardCorners
    ret, cp_img = cv2.findChessboardCorners(gray_img, (w, h), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS)

    # 去畸变
    img = cv2.undistort(img, cameraMatrix, distCoeffs)
    cp_img_undistort = cv2.undistortPoints(cp_img[[27,28,29,38,39,40,49,50,51]], cameraMatrix, distCoeffs, P=cameraMatrix)
    # cp_img_undistort = cv2.undistortPoints(cp_img[[11,23,30,47]], cameraMatrix, distCoeffs, P=cameraMatrix)
    for idy in range(len(cp_img_undistort)):
        cv2.circle(img, (int(cp_img_undistort[idy][0][0]), int(cp_img_undistort[idy][0][1])), 5, (0, 0, 255), 10)

    # output img
    cv2.imwrite(args.output_test_undistort_img_path, img)

    ##########################################################
    # test 畸变系数归零 -> 推算外参（与带畸变，外参一致）
    ##########################################################
    wo_dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1,5)
    _, wo_dist_rvecs, wo_dist_tvecs = cv2.solvePnP(np.array(cp_world[[27,28,29,38,39,40,49,50,51], :]), np.array(cp_img_undistort), cameraMatrix, wo_dist)
    # _, wo_dist_rvecs, wo_dist_tvecs = cv2.solvePnP(np.array(cp_world[[11,23,30,47], :]), np.array(cp_img_undistort), cameraMatrix, wo_dist)
    print(wo_dist_rvecs)
    print(wo_dist_tvecs)

    ##########################################################
    # test 手推畸变公式
    ##########################################################
    cp_img_undistort_one = np.ones((3, 1), dtype = np.float32)
    cp_img_undistort_one[:2, :] = cp_img_undistort[1][0][:, np.newaxis]
    x = np.dot(np.linalg.inv(cameraMatrix), cp_img_undistort_one)
    r = np.linalg.norm(x[:2, :])
    x_0 = x.copy()
    # x_0=x(1+k_1r^2+k_2r^4+k_3r^6)
    # y_0=y(1+k_1r^2+k_2r^4+k_3r^6)
    x_0[0] = x[0] * (1 + distCoeffs[0][0]*pow(r,2) + distCoeffs[0][1]*pow(r,4) + distCoeffs[0][4]*pow(r,6))
    x_0[1] = x[1] * (1 + distCoeffs[0][0]*pow(r,2) + distCoeffs[0][1]*pow(r,4) + distCoeffs[0][4]*pow(r,6))

    # x_0=2p_1xy+p_2(r^2+2x^2)+x
    # y_0=p_2(r^2+2y^2)+2p_2xy+y
    x_1 = x_0.copy()
    x_1[0] = 2*distCoeffs[0][2]*x_0[0]*x_0[1] + distCoeffs[0][3]*(pow(r,2)+ 2*pow(x_0[0],2)) + x_0[0]
    x_1[1] = distCoeffs[0][2]*(pow(r,2)+ 2*pow(x_0[1],2)) + 2*distCoeffs[0][3]*x_0[0]*x_0[1] + x_0[1]

    cp_img_one = np.ones((3, 1), dtype = np.float32)
    cp_img_one[:2, :] = cp_img[[7]][0].T
    x_ref = np.dot(np.linalg.inv(cameraMatrix), cp_img_one)
    print(x_1)
    print(x_ref)


def test_rotation_z(args):

    w, h = args.chessboard_corner_shape
    cp_int = np.zeros((w * h, 3), np.float32)
    cp_int[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
    cp_world = cp_int * args.chessboard_size_per_grid

    # load cameraMatrix
    cv_file = cv2.FileStorage(args.mtx_file, cv2.FILE_STORAGE_READ)
    cameraMatrix = cv_file.getNode("cameraMatrix").mat()
    cv_file.release()

    # load distCoeffs
    cv_file = cv2.FileStorage(args.dist_file, cv2.FILE_STORAGE_READ)
    distCoeffs = cv_file.getNode("distCoeffs").mat()
    cv_file.release()

    # img
    img = cv2.imread(args.test_chessboard_img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # findChessboardCorners
    ret, cp_img = cv2.findChessboardCorners(gray_img, (w, h), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS)

    # 去畸变
    cp_img_undistort = cv2.undistortPoints(cp_img[[27,28,29,38,39,40,49,50,51]], cameraMatrix, distCoeffs, P=cameraMatrix)
    # cp_img_undistort = cv2.undistortPoints(cp_img[[11,23,30,47]], cameraMatrix, distCoeffs, P=cameraMatrix)

    ##########################################################
    # test 畸变系数归零 -> 推算外参（与带畸变，外参一致）
    ##########################################################
    wo_dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1,5)
    _, wo_dist_rvecs, wo_dist_tvecs = cv2.solvePnP(np.array(cp_world[[27,28,29,38,39,40,49,50,51], :]), np.array(cp_img_undistort), cameraMatrix, wo_dist)
    # _, wo_dist_rvecs, wo_dist_tvecs = cv2.solvePnP(np.array(cp_world[[11,23,30,47], :]), np.array(cp_img_undistort), cameraMatrix, wo_dist)
    print('旋转矩阵是：\n', wo_dist_rvecs)
    print('平移矩阵是:\n', wo_dist_tvecs)

    ##########################################################
    # test 旋转矩阵归零 -> 使用一个点求俯仰角（无畸变）
    ##########################################################   
    # init，已知两个点（无畸变）
    world_ones = np.ones((cp_world[[27]].shape[0], 4), dtype = np.float32)
    world_ones[:, :3] = cp_world[[27]]
    # 像素坐标推到公式（无畸变）
    cp_ref = np.ones((cp_img[[27]].shape[0], 3), dtype = np.float32)
    cp_ref[:, :2] = cv2.undistortPoints(cp_img[[27]], cameraMatrix, distCoeffs, P=cameraMatrix)

    # 利用一个点，求x轴俯仰角（固定 y 轴 z 轴俯仰角，固定偏移量）
    # wo_dist_rvecs[0][0] = 0.0
    wo_dist_rvecs[1][0] = 0.0
    wo_dist_rvecs[2][0] = 0.0

    # init，已知旋转矩阵（y轴、z轴）、平移矩阵
    rotation_m, _ = cv2.Rodrigues(wo_dist_rvecs)#罗德里格斯变换
    rotation_t = np.hstack([rotation_m, wo_dist_tvecs])
    print('rotation_t：\n', rotation_t)
    
    # 利用一个点，求x轴俯仰角（固定 y 轴 z 轴俯仰角，固定偏移量）
    a = np.dot(np.linalg.inv(cameraMatrix) , cp_ref[0])
    b = np.dot(rotation_t, world_ones[0])
    a = a / (a[0] / b[0])
    # rotation_z = math.acos((a[1] - wo_dist_tvecs[1]) / world_ones[0][1])
    rotation_z = math.asin((a[2] - wo_dist_tvecs[2]) / world_ones[0][0])
    # print(rotation_z)

    # cp_ref = np.dot( np.dot(cameraMatrix, rotation_t), world_ones.T).T
    # cp_ref = cp_ref / cp_ref[:, 2][:, np.newaxis]

    # # 利用一个点，求x轴俯仰角（固定 y 轴 z 轴俯仰角，固定偏移量）
    # a = np.dot(np.linalg.inv(cameraMatrix) , cp_ref[0])
    # b = np.dot(rotation_t, world_ones[0])
    # a = a / (a[0] / b[0])
    # # rotation_z = math.acos((a[1] - wo_dist_tvecs[1]) / world_ones[0][1])
    # rotation_z = math.asin((a[2] - wo_dist_tvecs[2]) / world_ones[0][0])
    # print(rotation_z)

    # 绘制映射矩阵
    wo_dist_rvecs[0][0] = rotation_z
    wo_dist_rvecs[1][0] = 0.0
    wo_dist_rvecs[2][0] = 0.0

    cp_int = np.zeros((2 * w * 2 * h, 3), np.float32)
    cp_int[:, :2] = np.mgrid[-1*w:1*w, -1*h:1*h].T.reshape(-1, 2)
    cp_int = cp_int + [int(w/2), int(h/2), 0]
    
    img_world = cp_int * args.chessboard_size_per_grid
    img_point, _ = cv2.projectPoints(img_world, wo_dist_rvecs, wo_dist_tvecs, cameraMatrix, distCoeffs)
    for idy in range(len(img_point)):
        try:
            if int(img_point[idy][0][0]) >= 0 and int(img_point[idy][0][0]) <= img.shape[1] and int(img_point[idy][0][1]) >= 0 and int(img_point[idy][0][1]) <= img.shape[0]:
                cv2.circle(img, (int(img_point[idy][0][0]), int(img_point[idy][0][1])), 5, (0, 0, 255), 10)
        except:
            pass
    
    cv2.imwrite(args.output_test_rotation_z_img_path, img)


def test_project_points(args):
    
    w, h = args.chessboard_corner_shape
    cp_int = np.zeros((2 * w * 2 * h, 3), np.float32)
    cp_int[:, :2] = np.mgrid[-1*w:1*w, -1*h:1*h].T.reshape(-1, 2)
    cp_int = cp_int + [int(w/2), int(h/2), 0]
    
    # load cameraMatrix
    cv_file = cv2.FileStorage(args.mtx_file, cv2.FILE_STORAGE_READ)
    cameraMatrix = cv_file.getNode("cameraMatrix").mat()
    cv_file.release()

    # load distCoeffs
    cv_file = cv2.FileStorage(args.dist_file, cv2.FILE_STORAGE_READ)
    distCoeffs = cv_file.getNode("distCoeffs").mat()
    cv_file.release()

    rvecs = np.array([[-1.0, 0.0, 0.0]])
    tvecs = np.array([[0.0, 0.0, 2.0]])

    # img
    img = cv2.imread(args.test_chessboard_img_path)

    # img_world = cp_int * args.chessboard_size_per_grid
    img_world = cp_int * args.chessboard_size_per_grid * 5
    img_point, _ = cv2.projectPoints(img_world, rvecs, tvecs, cameraMatrix, distCoeffs)
    for idy in range(len(img_point)):
        try:
            if int(img_point[idy][0][0]) >= 0 and int(img_point[idy][0][0]) <= img.shape[1] and int(img_point[idy][0][1]) >= 0 and int(img_point[idy][0][1]) <= img.shape[0]:
                cv2.circle(img, (int(img_point[idy][0][0]), int(img_point[idy][0][1])), 5, (0, 0, 255), 10)
        except:
            pass
    
    cv2.imwrite(args.output_test_project_points_img_path, img)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # args.file_dir = "/mnt/huanyuan2/data/image/Calibrate/Chessboard/预研/"
    args.file_dir = "/mnt/huanyuan2/data/image/Calibrate/Chessboard/BM1448/"
    args.mtx_file = os.path.join(args.file_dir, "cameraMatrix.xml" )
    args.dist_file = os.path.join(args.file_dir, "distCoeffs.xml" )

    ##########################################################
    # 棋盘格标定
    ##########################################################

    # 自定义棋盘格标定板
    # args.chessboard_corner_shape = (8, 6)
    # args.chessboard_size_per_grid = 0.021

    # 公司棋盘格标定板
    args.chessboard_corner_shape = (11, 8)
    args.chessboard_size_per_grid = 0.025
    
    args.chessboard_img_dir = os.path.join(args.file_dir, "chessboard_img/" )
    args.output_chessboard_corners_img_dir = os.path.join(args.file_dir, "chessboard_corners_img/")
    args.output_calibrate_img_dir = os.path.join(args.file_dir, "calibrate_img/")
    args.output_solvePnP_img_dir = os.path.join(args.file_dir, "solvePnP_img/")
    args.suffix = ".jpg"

    # # 标定相机参数（棋盘格标定板）
    # calibrate(args)

    # # solvePnP
    # solvePnP(args)


    ##########################################################
    # 标定板求外参
    ##########################################################

    # # 自定义棋盘格标定板
    # args.chessboard_corner_shape = (8, 6)
    # args.chessboard_size_per_grid = 0.021

    # 公司棋盘格标定板
    args.chessboard_corner_shape = (11, 8)
    args.chessboard_size_per_grid = 0.025

    args.test_chessboard_img_path = os.path.join(args.file_dir, "test_chessboard_img.jpg")
    args.output_test_chessboard_corners_img_path = os.path.join(args.file_dir, "test_chessboard_corners_img.jpg")
    args.output_test_solvePnP_img_path = os.path.join(args.file_dir, "test_solvePnP_img.jpg")
    args.output_test_undistort_img_path = os.path.join(args.file_dir, "test_undistort_img.jpg")
    args.output_test_rotation_z_img_path = os.path.join(args.file_dir, "test_rotation_z_img.jpg")
    args.output_test_project_points_img_path = os.path.join(args.file_dir, "test_project_points_img.jpg")

    # 利用棋盘格标定板，推到外参
    # test solvePnP
    test_solvePnP(args)

    # test undistort 去畸变
    test_undistort(args)

    # # 单点求俯仰角
    # test_rotation_z(args)

    # # 已知俯仰角，映射结果
    # test_project_points(args)
