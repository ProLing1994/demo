def intersect(box_a, box_b):
    inter_x1 = max(box_a[0], box_b[0])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y1 = max(box_a[1], box_b[1])
    inter_y2 = min(box_a[3], box_b[3])
    inter =  max(inter_x2 - inter_x1, 0.0) * max(inter_y2 - inter_y1, 0.0) 
    return inter


def bool_box_in_roi(box, roi):
    bool_in_w = True if box[0] >= roi[0] and box[2] <= roi[2] else False
    bool_in_h = True if box[1] >= roi[1] and box[3] <= roi[3] else False
    return bool_in_w * bool_in_h


def match_bbox_iou(input_roi, match_roi_list):
    # init
    matched_roi_list = []
    max_intersect_iou = 0.0
    max_intersect_iou_idx = 0

    for idx in range(len(match_roi_list)):
        match_roi_idx = match_roi_list[idx][0:4]
        intersect_iou = intersect(input_roi, match_roi_idx)

        if intersect_iou > max_intersect_iou:
            max_intersect_iou = intersect_iou
            max_intersect_iou_idx = idx
        
    if max_intersect_iou > 0.0:
        matched_roi_list.append(match_roi_list[max_intersect_iou_idx])
    
    return matched_roi_list


def match_car_license_plate(car_roi, license_plate_list):
    # sort_key
    def sort_key(data):
        return data[-1]

    # init
    matched_roi_list = []

    for idx in range(len(license_plate_list)):
        match_roi_idx = license_plate_list[idx][0:4]

        # # 方案一：使用 IOU 判断
        # intersect_iou = intersect(car_roi, match_roi_idx)
        # # 计算车牌检测框与车辆检测框的交集区域，大于 0.0 则认为该车牌属于该车辆
        # if intersect_iou > 0.0:
        #     # 默认车牌均是在车辆的下沿
        #     matched_roi_list.append(license_plate_list[idx])
        #     if (car_roi[1] + car_roi[3] / 2.0) < (match_roi_idx[1] + match_roi_idx[3] / 2.0):
        #         matched_roi_list.append(license_plate_list[idx])

        # 方案二：计算车牌框完全位于车框内
        bool_in = bool_box_in_roi(match_roi_idx, car_roi)
        if bool_in:
            # 默认车牌均是在车辆的下沿
            matched_roi_list.append(license_plate_list[idx])
            # if (car_roi[1] + car_roi[3] / 2.0) < (match_roi_idx[1] + match_roi_idx[3] / 2.0):
            #     matched_roi_list.append(license_plate_list[idx])

    matched_roi_list.sort(key=sort_key, reverse=True)

    return matched_roi_list