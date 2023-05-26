import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm
import sys 

# sys.path.insert(0, '/home/huanyuan/code/demo/Image')
sys.path.insert(0, '/yuanhuan/code/demo/Image/')
from Basic.script.json.json_write import write_json
from Basic.utils.folder_tools import *


def tranform(args):
    
    imgs_path = []
    words = []

    # file
    with open(args.input_txt_path, "r") as f:

        lines = f.readlines()

        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)

    # write json
    pbar = tqdm(zip(imgs_path, words), desc='Scanning images', total=len(imgs_path))
    for i, (im, lb) in enumerate(pbar):
        
        # img
        img = cv2.imread(os.path.join(args.output_img_dir, im))

        # json
        json_bboxes = {}

        for idx, label in enumerate(lb):
            
            if args.tarin_bool:

                w = label[2]
                h = label[3]

                points_list = []
                points_list.append([label[0], label[1]])    # x1 y1
                points_list.append([label[0] + label[2], label[1] + label[3]])    # x2 y2

                # degree
                landmark_list = []

                # left eye, right eye, nose, left mouth, right mouth
                landmark_list.extend([label[4], label[5], label[7], label[8], label[10], label[11], label[13], label[14], label[16], label[17]])

                if landmark_list[0] == -1 or landmark_list[1] == -1 or \
                    landmark_list[2] == -1 or landmark_list[3] == -1 or \
                    landmark_list[4] == -1 or landmark_list[5] == -1 or \
                    landmark_list[6] == -1 or landmark_list[7] == -1 or \
                    landmark_list[8] == -1 or landmark_list[9] == -1:
                    points_list.append([-1.0, -1.0])
                else:
                    
                    up_line_k = float(landmark_list[1] - landmark_list[3]) / float(landmark_list[0] - landmark_list[2] + 1e-5);
                    up_line_b = float(landmark_list[1] - landmark_list[0] * up_line_k);
                    up_down_k = float(landmark_list[7] - landmark_list[9]) / float(landmark_list[6] - landmark_list[8] + 1e-5);
                    up_down_b = float(landmark_list[7] - landmark_list[6] * up_down_k);

                    center_up_x = (landmark_list[0] + landmark_list[2]) / 2
                    center_up_y = (landmark_list[1] + landmark_list[3]) / 2
                    center_down_x = (landmark_list[6] + landmark_list[8]) / 2
                    center_down_y = (landmark_list[7] + landmark_list[9]) / 2
                    # y = kx + b
                    up_donw_line_k = float(center_up_y - center_down_y) / float(center_up_x - center_down_x + 1e-5);
                    up_donw_line_b = float(center_up_y - center_up_x * up_donw_line_k);

                    left_line_k = float(landmark_list[1] - landmark_list[7]) / float(landmark_list[0] - landmark_list[6] + 1e-5);
                    left_line_b = float(landmark_list[1] - landmark_list[0] * left_line_k);
                    right_line_k = float(landmark_list[3] - landmark_list[9]) / float(landmark_list[2] - landmark_list[8] + 1e-5);
                    right_line_b = float(landmark_list[3] - landmark_list[2] * right_line_k);

                    cenert_left_x = (landmark_list[0] + landmark_list[6]) / 2
                    cenert_left_y = (landmark_list[1] + landmark_list[7]) / 2
                    cenert_right_x = (landmark_list[2] + landmark_list[8]) / 2
                    cenert_right_y = (landmark_list[3] + landmark_list[9]) / 2
                    # y = kx + b
                    left_right_line_k = float(cenert_left_y - cenert_right_y) / float(cenert_left_x - cenert_right_x + 1e-5);
                    left_right_line_b = float(cenert_left_y - cenert_left_x * left_right_line_k);
                
                    nose_k = left_right_line_k
                    nose_b = float(landmark_list[5] - landmark_list[4] * nose_k);
                    
                    up_donw_intersect_x = (up_donw_line_b - nose_b)/(nose_k - up_donw_line_k)
                    up_donw_intersect_y = nose_k * up_donw_intersect_x + nose_b

                    left_intersect_x = (left_line_b - nose_b)/(nose_k - left_line_k)
                    left_intersect_y = nose_k * left_intersect_x + nose_b

                    right_intersect_x = (right_line_b - nose_b)/(nose_k - right_line_k)
                    right_intersect_y = nose_k * right_intersect_x + nose_b

                    dist_nose = np.linalg.norm(np.array([up_donw_intersect_x - landmark_list[4], up_donw_intersect_y - landmark_list[5]]))
                    dist_left = np.linalg.norm(np.array([up_donw_intersect_x - left_intersect_x, up_donw_intersect_y - left_intersect_y]))
                    dist_right = np.linalg.norm(np.array([up_donw_intersect_x - right_intersect_x, up_donw_intersect_y - right_intersect_y]))

                    if landmark_list[4] < up_donw_intersect_x:
                        degree = min((dist_nose / (dist_left + 1e-5)), 2.0) / 2.0
                    else:
                        degree = min((dist_nose / (dist_right+ 1e-5)), 2.0) / 2.0

                    points_list.append([degree, degree])

                    if args.show_bool:
                        cv2.line(img, (int(center_up_x), int(center_up_y)), (int(center_down_x), int(center_down_y)), (0, 255, 0), 2)
                        cv2.line(img, (int(cenert_left_x), int(cenert_left_y)), (int(cenert_right_x), int(cenert_right_y)), (0, 255, 0), 2)
                        cv2.line(img, (int(left_intersect_x), int(left_intersect_y)), (int(up_donw_intersect_x), int(up_donw_intersect_y)), (0, 0, 255), 2)
                        cv2.line(img, (int(right_intersect_x), int(right_intersect_y)), (int(up_donw_intersect_x), int(up_donw_intersect_y)), (0, 0, 255), 2)
                        cv2.circle(img, (int(landmark_list[4]), int(landmark_list[5])), 3, (0, 0, 255), 4)
                        img = cv2.putText(img, "{:.2f}".format( degree ), (int(landmark_list[4] + 10), int(landmark_list[5] - 10)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            else:

                w = label[2]
                h = label[3]

                points_list = []
                points_list.append([label[0], label[1]])    # x1 y1
                points_list.append([label[0] + label[2], label[1] + label[3]])    # x2 y2

                # degree
                points_list.append([-1.0])          # degree

            # class_name
            class_name = args.set_class_name

            if w < args.width_threshold or h < args.height_threshold:
                class_name = args.filter_set_class_name
                
            if class_name not in json_bboxes:
                json_bboxes[class_name] = []     
            json_bboxes[class_name].append(points_list)
        
        if args.show_bool:
            output_img_path = os.path.join(args.output_show_img_dir, im)
            create_folder(os.path.dirname(output_img_path))
            cv2.imwrite(output_img_path, img)

        # output 
        output_json_path = os.path.join(args.output_json_dir, im.replace('.jpg', '.json'))
        create_folder(os.path.dirname(output_json_path))
        write_json(output_json_path, im, img.shape, json_bboxes, "polygon")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.set_class_name = "face"
    args.filter_set_class_name = "face_o"

    args.width_threshold = 10
    args.height_threshold = 10

    args.input_txt_path = "/yuanhuan/data/image/Open_Source/Wider_Face/original/retinaface_gt_v1.1/train/label.txt"
    args.output_dir = "/yuanhuan/data/image/Open_Source/Wider_Face/original/WIDER_train/"
    args.output_img_dir = os.path.join(args.output_dir, "images")
    args.output_json_dir = os.path.join(args.output_dir, "json_degree")
    args.output_show_img_dir = os.path.join(args.output_dir, "show_images")
    args.tarin_bool = True
    args.show_bool = True

    tranform(args)    