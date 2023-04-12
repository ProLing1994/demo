import cv2 

color_dict = {
                "car": (0, 255, 0), 
                "car_alarm": (0, 0, 255), 
                "intersect_point_alarm": (255, 0, 0), 
                "roi_capture_area": (255, 255, 255),
            }

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


def cv_plot_rectangle(img, bbox, mode='xywh', color=(255, 255, 255), thickness=2):
    if mode == 'xywh':
        x, y, w, h = bbox
        xmin, ymin, xmax, ymax = x, y, w + x, h + y
    elif mode == 'ltrb':
        xmin, ymin, xmax, ymax = bbox
    else:
        print("Unknown plot mode")
        return None

    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
    img_p = img.copy()

    return cv2.rectangle(img_p, (xmin, ymin), (xmax, ymax), color=color, thickness=thickness)


def draw_bbox_tracker(img, tracker_bboxes, mode='ltrb'):
    
    for idx in range(len(tracker_bboxes)):
        tracker_bbox = tracker_bboxes[idx]

        id = tracker_bbox[-1]
        loc = tracker_bbox[0:4]
        color = get_color(abs(id))

        img = cv_plot_rectangle(img, loc, mode=mode, color=color)
        img = cv2.putText(img, "id: {}".format(int(id)), (int(loc[0]), int(loc[1])), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
        
    return img


def draw_bbox_info(img, bbox_info, capture_points, mode='xywh'):
    
    for idx in range(len(bbox_info)):
        bbox_info_idx = bbox_info[idx]

        # car
        bbox_info_idx['loc'] = [int(b + 0.5) for b in bbox_info_idx['loc'][:4]]
        if bbox_info_idx['alarm_flage']:
            img = cv_plot_rectangle(img, bbox_info_idx['loc'], mode=mode, color=color_dict["car_alarm"])
        else:
            img = cv_plot_rectangle(img, bbox_info_idx['loc'], mode=mode, color=color_dict["car"])
        
        img = cv2.putText(img, "{}_{}_{}_{}_{}_{}_{:.1f}_{:.1f}".format( bbox_info_idx['id'], bbox_info_idx['frame_num'], bbox_info_idx['up_down_state'], bbox_info_idx['up_down_state_frame_num'], bbox_info_idx['left_right_state'], bbox_info_idx['left_right_state_frame_num'], bbox_info_idx['up_down_speed'], bbox_info_idx['left_right_speed'] ), (bbox_info_idx['loc'][0], bbox_info_idx['loc'][1] - 10), 
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        
        # 车辆线
        car_line_k = bbox_info_idx['alarm_car_line'][0]
        car_line_b = bbox_info_idx['alarm_car_line'][1]
        point_1 = int((bbox_info_idx['loc'][1] - car_line_b) / car_line_k), bbox_info_idx['loc'][1]
        point_2 = int((bbox_info_idx['loc'][3] - car_line_b) / car_line_k), bbox_info_idx['loc'][3]
        cv2.line(img, point_1, point_2, color_dict["car"], 2)

        if bbox_info_idx['alarm_flage']:
            # 报警线
            point_1 =  capture_points[ 2 * bbox_info_idx['alarm_capture_line_id']] 
            point_2 =  capture_points[ 2 * bbox_info_idx['alarm_capture_line_id'] + 1]
            cv2.line(img, point_1, point_2, color_dict["car_alarm"], 2)

            # 报警交点
            cv2.circle(img, (int(bbox_info_idx['alarm_intersect_point'][0]), int(bbox_info_idx['alarm_intersect_point'][1])), 3, color_dict["intersect_point_alarm"], 6)
    
    return img


def draw_bbox_state(img, bbox_state_map):
    
    for key, bbox_state_idy in bbox_state_map.items():
        
        color = get_color(abs(bbox_state_idy['id']))
        
        for idx in range(len( bbox_state_idy['center_point_list'] )):
            x = int( bbox_state_idy['center_point_list'][idx][0] + 0.5 )
            y = int( bbox_state_idy['center_point_list'][idx][1] + 0.5 )

            cv2.circle(img, (x, y), 1, color, 2)

    return img


def draw_capture_line(img, capture_points):
    
    for idx in range(int(len(capture_points) / 2)):
        point_1 = tuple( capture_points[ 2 * idx] )
        point_2 = tuple( capture_points[ 2 * idx + 1] ) 
        cv2.line(img, point_1, point_2, color_dict["roi_capture_area"], 2)
        img = cv2.putText(img, "{}".format( 2 * idx ), (point_1[0] + 10, point_1[1] - 10), 
                            cv2.FONT_HERSHEY_COMPLEX, 1, color_dict["roi_capture_area"], 2)
        img = cv2.putText(img, "{}".format( 2 * idx + 1 ), (point_2[0] + 10, point_2[1] - 10), 
                            cv2.FONT_HERSHEY_COMPLEX, 1, color_dict["roi_capture_area"], 2)

    return img