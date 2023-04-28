import cv2 

color_dict = {
                "car": (0, 255, 0), 
                "car_warning": (255, 0, 0), 
                "car_capture": (12, 149, 255), 
                "plate": (0, 0, 255),
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


def draw_bbox_info(img, bbox_info, mode='xywh'):
    
    for idx in range(len(bbox_info)):
        bbox_info_idx = bbox_info[idx]

        # car
        bbox_info_idx['loc'] = [int(b + 0.5) for b in bbox_info_idx['loc'][:4]]
        if bbox_info_idx['alarm_flage']:
            img = cv_plot_rectangle(img, bbox_info_idx['loc'], mode=mode, color=color_dict["car_capture"])
        elif bbox_info_idx['warning_flage']:
            img = cv_plot_rectangle(img, bbox_info_idx['loc'], mode=mode, color=color_dict["car_warning"])
        else:
            img = cv_plot_rectangle(img, bbox_info_idx['loc'], mode=mode, color=color_dict["car"])

        img = cv2.putText(img, "{}_{}_{}_{}_{}_{}_{:.1f}_{:.1f}_{:.1f}_{:.1f}_{:.1f}_{:.1f}".format( bbox_info_idx['id'], bbox_info_idx['frame_num'], bbox_info_idx['up_down_state'], bbox_info_idx['up_down_state_frame_num'], bbox_info_idx['left_right_state'], bbox_info_idx['left_right_state_frame_num'], bbox_info_idx['up_down_speed'], bbox_info_idx['left_right_speed'], bbox_info_idx['dist_capture_line_left'], bbox_info_idx['dist_capture_line_right'], bbox_info_idx['dist_capture_line_left_top'], bbox_info_idx['dist_capture_line_right_top'] ), (bbox_info_idx['loc'][0], bbox_info_idx['loc'][1] - 10), 
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        img = cv2.putText(img, "{}_{}_{}_{}_{}".format( int(bbox_info_idx['left_alarm_flage']), int(bbox_info_idx['right_alarm_flage']), int(bbox_info_idx['top_alarm_flage']), int(bbox_info_idx['warning_flage']), int(bbox_info_idx['alarm_flage'])), (bbox_info_idx['loc'][0], bbox_info_idx['loc'][1] + 20), 
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        img = cv2.putText(img, "{}_{}_{}_{}".format( int(bbox_info_idx['left_in_alarm_flage']), int(bbox_info_idx['left_out_alarm_flage']), int(bbox_info_idx['right_in_alarm_flage']), int(bbox_info_idx['right_out_alarm_flage'])), (bbox_info_idx['loc'][0], bbox_info_idx['loc'][1] + 40), 
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        img = cv2.putText(img, "{}_{}".format( int(bbox_info_idx['lane_line_state']), int(bbox_info_idx['lane_line_state_frame_num'])), (bbox_info_idx['loc'][0], bbox_info_idx['loc'][1] + 60), 
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        
        # 与报警线的交线
        for idy in range(len(bbox_info_idx['lane_line_info'])):
            
            # 车辆在区域内，输出交线
            if (bbox_info_idx['loc'][3] > bbox_info_idx['lane_line_info'][0][-1][1]) and (bbox_info_idx['loc'][3] < bbox_info_idx['lane_line_info'][-1][-1][1]):
                point_1 = tuple( bbox_info_idx['lane_line_info'][idy][4] )
                if idy == len(bbox_info_idx['lane_line_info'])-1 :
                    point_2 = tuple( bbox_info_idx['lane_line_info'][0][4] ) 
                else:
                    point_2 = tuple( bbox_info_idx['lane_line_info'][idy+1][4] ) 

                cv2.line(img, point_1, point_2, color_dict["car"], 2)

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
    
    for idx in range(len(capture_points)):

        point_1 = tuple( capture_points[idx] )
        if idx == len(capture_points) - 1:
            point_2 = tuple( capture_points[0] )
        else:
            point_2 = tuple( capture_points[idx + 1] ) 
        
        # line
        cv2.line(img, point_1, point_2, color_dict["roi_capture_area"], 2)

        img = cv2.putText(img, "{}".format( idx ), (point_1[0] + 10, point_1[1] - 10), 
                            cv2.FONT_HERSHEY_COMPLEX, 1, color_dict["roi_capture_area"], 2)

    return img