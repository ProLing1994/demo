import numpy as np

def landmark2degree(landmark_list):
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
    
    return degree