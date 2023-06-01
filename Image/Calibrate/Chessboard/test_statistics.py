import argparse
import argparse
import cv2
import numpy as np
import json
import os


def test_statistics(args, corner_name_list):

    # img
    img = cv2.imread(args.test_point_img_path)

    # cp_img
    cp_world = np.zeros((len(corner_name_list), 3), dtype = np.float32)
    cp_img = np.zeros((len(corner_name_list), 1, 2), dtype = np.float32)

    # json
    json_path = args.test_point_json_path
    with open(json_path, 'r', encoding='UTF-8') as fr:
        annotation = json.load(fr)

    for track in annotation['shapes']:
        label = track['label']
        points = np.array(track['points'])

        for idx in range(len(corner_name_list)):
            calibrate_corner_name = corner_name_list[idx]

            if label != calibrate_corner_name:
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

    # cp_ref_img
    cp_ref_img = np.zeros((len(corner_name_list), 1, 2), dtype = np.float32)

    # json_ref
    json_ref_path = args.test_point_json_ref_path
    with open(json_ref_path, 'r', encoding='UTF-8') as fr:
        annotation = json.load(fr)

    for track in annotation['shapes']:
        label = track['label']
        points = np.array(track['points'])

        for idx in range(len(corner_name_list)):
            calibrate_corner_name = corner_name_list[idx]

            if label != calibrate_corner_name:
                continue
            
            # cp_img_idx
            cp_img_idx = list(points[0])

            # cp_ref_img
            cp_ref_img[idx: idx+1] = np.array(cp_img_idx).reshape((1,1,2))

    # error
    error = cv2.norm(cp_img[:, 0, 0], cp_ref_img[:, 0, 0], cv2.NORM_L2) / len(cp_img)
    print("error:", error)

    # 绘图
    for idy in range(len(cp_img)):
        cv2.circle(img, (int(cp_img[idy][0][0]), int(cp_img[idy][0][1])), 5, (255, 0, 0), 10)
        cv2.circle(img, (int(cp_ref_img[idy][0][0]), int(cp_ref_img[idy][0][1])), 5, (0, 0, 255), 10)
        cv2.line(img, (int(cp_img[idy][0][0]), int(cp_img[idy][0][1])), (int(cp_ref_img[idy][0][0]), int(cp_ref_img[idy][0][1])), (0, 0, 255), 1)
        cv2.putText(img, "({:.1f},{:.1f})".format(float(cp_world[idy][0]), float(cp_world[idy][1])), (int(cp_img[idy][0][0] + 10), int(cp_img[idy][0][1]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return img, error


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.file_dir = "/mnt/huanyuan2/data/image/Calibrate/Chessboard/BM1448/"

    args.feet_corner_name_list = [ ['b6_0', 'b6_5', 'b6_10'], 
                                   ['b-6_0', 'b-6_5', 'b-6_10'], 
                                   ['b13.6_0', 'b13.6_5', 'b13.6_10'], 
                                   ['b-13.8_0', 'b-13.8_5', 'b-13.8_10'],
                                   ['b14.2_0', 'b14.2_5', 'b14.2_10'],
                                   ['b-14.4_0', 'b-14.4_5', 'b-14.4_10'] ]
    
    # args.image_name = "c28_250_high_angle"
    # args.image_name = "c28_250_low_angle"
    args.image_name = "c28_290_high_angle"
    # args.image_name = "c28_290_low_angle"

    # args.image_ref_name = "c28_250_high_angle"
    args.image_ref_name = "c28_290_high_angle"

    args.test_point_img_path = os.path.join(args.file_dir, "test_jpg/point_img/{}.jpg".format(args.image_name))
    args.test_point_json_path = os.path.join(args.file_dir, "test_jpg/point_img/{}.json".format(args.image_name))
    args.test_point_json_ref_path = os.path.join(args.file_dir, "test_jpg/point_img/{}.json".format(args.image_ref_name))

    for idx in range(len(args.feet_corner_name_list)):
        img, error = test_statistics(args, args.feet_corner_name_list[idx])
        output_test_statistics_img_path = os.path.join(args.file_dir, "test_jpg/statistic_point_img_res/{}_statistics_ref_{}_{}_{:.2f}.jpg".format(args.image_name, args.image_ref_name, idx, error))
        cv2.imwrite(output_test_statistics_img_path, img)