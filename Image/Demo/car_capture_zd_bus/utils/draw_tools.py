import cv2 

color_dict = {
                "car": (0, 255, 0), 
                "car_waring_10m": (255, 0, 0), 
                "car_alarm_5m": (12, 149, 255), 
                "car_alarm_3m": (0, 0, 255), 
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
        if bbox_info_idx['alarm_line_3m_flage']:
            img = cv_plot_rectangle(img, bbox_info_idx['loc'], mode=mode, color=color_dict["car_alarm_3m"])
        elif bbox_info_idx['alarm_line_5m_flage']:
            img = cv_plot_rectangle(img, bbox_info_idx['loc'], mode=mode, color=color_dict["car_alarm_5m"])
        elif bbox_info_idx['waring_line_10m_flage']:
            img = cv_plot_rectangle(img, bbox_info_idx['loc'], mode=mode, color=color_dict["car_waring_10m"])
        else:
            img = cv_plot_rectangle(img, bbox_info_idx['loc'], mode=mode, color=color_dict["car"])
        
        img = cv2.putText(img, "{}_{}_{}_{}_{}_{}_{:.1f}_{:.1f}".format( bbox_info_idx['id'], bbox_info_idx['frame_num'], bbox_info_idx['up_down_state'], bbox_info_idx['up_down_state_frame_num'], bbox_info_idx['left_right_state'], bbox_info_idx['left_right_state_frame_num'], bbox_info_idx['up_down_speed'], bbox_info_idx['left_right_speed'] ), (bbox_info_idx['loc'][0], bbox_info_idx['loc'][1] - 10), 
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        img = cv2.putText(img, "{:.1f}_{:.1f}_{:.1f}".format( bbox_info_idx['dist_waring_line_10m'], bbox_info_idx['dist_alarm_line_5m'], bbox_info_idx['dist_alarm_line_3m'] ), (bbox_info_idx['loc'][0], bbox_info_idx['loc'][1] + 20), 
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        img = cv2.putText(img, "{}_{}_{}".format( int(bbox_info_idx['waring_line_10m_flage']), int(bbox_info_idx['alarm_line_5m_flage']), int(bbox_info_idx['alarm_line_3m_flage'])), (bbox_info_idx['loc'][0], bbox_info_idx['loc'][1] + 50), 
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        img = cv2.putText(img, "{}_{}".format( int(bbox_info_idx['lane_line_state']), int(bbox_info_idx['lane_line_state_frame_num'])), (bbox_info_idx['loc'][0], bbox_info_idx['loc'][1] + 80), 
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
    
    # 预警线（10m）
    point_1 = tuple( capture_points[1] )
    point_2 = tuple( capture_points[2] ) 
    cv2.line(img, point_1, point_2, color_dict["roi_capture_area"], 2)
    img = cv2.putText(img, "{}".format( 1 ), (point_1[0] + 10, point_1[1] - 10), 
                        cv2.FONT_HERSHEY_COMPLEX, 1, color_dict["roi_capture_area"], 2)

    point_1 = tuple( capture_points[2] )
    point_2 = tuple( capture_points[3] ) 
    cv2.line(img, point_1, point_2, color_dict["roi_capture_area"], 2)
    img = cv2.putText(img, "{}".format( 2 ), (point_1[0] + 10, point_1[1] - 10), 
                        cv2.FONT_HERSHEY_COMPLEX, 1, color_dict["roi_capture_area"], 2)
    
    point_1 = tuple( capture_points[3] )
    point_2 = tuple( capture_points[4] ) 
    cv2.line(img, point_1, point_2, color_dict["roi_capture_area"], 2)
    img = cv2.putText(img, "{}".format( 3 ), (point_1[0] + 10, point_1[1] - 10), 
                        cv2.FONT_HERSHEY_COMPLEX, 1, color_dict["roi_capture_area"], 2)
    
    point_1 = tuple( capture_points[4] )
    point_2 = tuple( capture_points[5] ) 
    cv2.line(img, point_1, point_2, color_dict["roi_capture_area"], 2)
    img = cv2.putText(img, "{}".format( 4 ), (point_1[0] + 10, point_1[1] - 10), 
                        cv2.FONT_HERSHEY_COMPLEX, 1, color_dict["roi_capture_area"], 2)

    point_1 = tuple( capture_points[5] )
    point_2 = tuple( capture_points[6] ) 
    cv2.line(img, point_1, point_2, color_dict["roi_capture_area"], 2)
    img = cv2.putText(img, "{}".format( 5 ), (point_1[0] + 10, point_1[1] - 10), 
                        cv2.FONT_HERSHEY_COMPLEX, 1, color_dict["roi_capture_area"], 2)
    
    point_1 = tuple( capture_points[6] )
    point_2 = tuple( capture_points[1] ) 
    cv2.line(img, point_1, point_2, color_dict["roi_capture_area"], 2)
    img = cv2.putText(img, "{}".format( 6 ), (point_1[0] + 10, point_1[1] - 10), 
                        cv2.FONT_HERSHEY_COMPLEX, 1, color_dict["roi_capture_area"], 2)
    
    # 报警线（3m）
    point_1 = tuple( capture_points[0] )
    point_2 = tuple( capture_points[11] ) 
    cv2.line(img, point_1, point_2, color_dict["roi_capture_area"], 2)
    img = cv2.putText(img, "{}".format( 0 ), (point_1[0] + 10, point_1[1] - 10), 
                        cv2.FONT_HERSHEY_COMPLEX, 1, color_dict["roi_capture_area"], 2)

    point_1 = tuple( capture_points[11] )
    point_2 = tuple( capture_points[10] ) 
    cv2.line(img, point_1, point_2, color_dict["roi_capture_area"], 2)
    img = cv2.putText(img, "{}".format( 11 ), (point_1[0] + 10, point_1[1] - 10), 
                        cv2.FONT_HERSHEY_COMPLEX, 1, color_dict["roi_capture_area"], 2)
    
    point_1 = tuple( capture_points[10] )
    point_2 = tuple( capture_points[9] ) 
    cv2.line(img, point_1, point_2, color_dict["roi_capture_area"], 2)
    img = cv2.putText(img, "{}".format( 10 ), (point_1[0] + 10, point_1[1] - 10), 
                        cv2.FONT_HERSHEY_COMPLEX, 1, color_dict["roi_capture_area"], 2)
    
    point_1 = tuple( capture_points[9] )
    point_2 = tuple( capture_points[8] ) 
    cv2.line(img, point_1, point_2, color_dict["roi_capture_area"], 2)
    img = cv2.putText(img, "{}".format( 9 ), (point_1[0] + 10, point_1[1] - 10), 
                        cv2.FONT_HERSHEY_COMPLEX, 1, color_dict["roi_capture_area"], 2)
    
    point_1 = tuple( capture_points[8] )
    point_2 = tuple( capture_points[7] ) 
    cv2.line(img, point_1, point_2, color_dict["roi_capture_area"], 2)
    img = cv2.putText(img, "{}".format( 8 ), (point_1[0] + 10, point_1[1] - 10), 
                        cv2.FONT_HERSHEY_COMPLEX, 1, color_dict["roi_capture_area"], 2)
    
    point_1 = tuple( capture_points[7] )
    point_2 = tuple( capture_points[0] ) 
    cv2.line(img, point_1, point_2, color_dict["roi_capture_area"], 2)
    img = cv2.putText(img, "{}".format( 7 ), (point_1[0] + 10, point_1[1] - 10), 
                        cv2.FONT_HERSHEY_COMPLEX, 1, color_dict["roi_capture_area"], 2)
    
    # 标定线（5m）
    point_1 = tuple( capture_points[12] )
    point_2 = tuple( capture_points[13] ) 
    cv2.line(img, point_1, point_2, color_dict["roi_capture_area"], 2)
    img = cv2.putText(img, "{}".format( 12 ), (point_1[0] + 10, point_1[1] - 10), 
                        cv2.FONT_HERSHEY_COMPLEX, 1, color_dict["roi_capture_area"], 2)
    
    point_1 = tuple( capture_points[13] )
    point_2 = tuple( capture_points[14] ) 
    cv2.line(img, point_1, point_2, color_dict["roi_capture_area"], 2)
    img = cv2.putText(img, "{}".format( 13 ), (point_1[0] + 10, point_1[1] - 10), 
                        cv2.FONT_HERSHEY_COMPLEX, 1, color_dict["roi_capture_area"], 2)
    
    point_1 = tuple( capture_points[14] )
    point_2 = tuple( capture_points[15] ) 
    cv2.line(img, point_1, point_2, color_dict["roi_capture_area"], 2)
    img = cv2.putText(img, "{}".format( 14 ), (point_1[0] + 10, point_1[1] - 10), 
                        cv2.FONT_HERSHEY_COMPLEX, 1, color_dict["roi_capture_area"], 2)

    point_1 = tuple( capture_points[15] )
    point_2 = tuple( capture_points[16] ) 
    cv2.line(img, point_1, point_2, color_dict["roi_capture_area"], 2)
    img = cv2.putText(img, "{}".format( 15 ), (point_1[0] + 10, point_1[1] - 10), 
                        cv2.FONT_HERSHEY_COMPLEX, 1, color_dict["roi_capture_area"], 2)

    point_1 = tuple( capture_points[16] )
    point_2 = tuple( capture_points[17] ) 
    cv2.line(img, point_1, point_2, color_dict["roi_capture_area"], 2)
    img = cv2.putText(img, "{}".format( 16 ), (point_1[0] + 10, point_1[1] - 10), 
                        cv2.FONT_HERSHEY_COMPLEX, 1, color_dict["roi_capture_area"], 2)

    point_1 = tuple( capture_points[17] )
    point_2 = tuple( capture_points[12] ) 
    cv2.line(img, point_1, point_2, color_dict["roi_capture_area"], 2)
    img = cv2.putText(img, "{}".format( 17 ), (point_1[0] + 10, point_1[1] - 10), 
                        cv2.FONT_HERSHEY_COMPLEX, 1, color_dict["roi_capture_area"], 2)
    
    #  连接线
    point_1 = tuple( capture_points[0] )
    point_2 = tuple( capture_points[1] ) 
    cv2.line(img, point_1, point_2, color_dict["roi_capture_area"], 2)
    point_1 = tuple( capture_points[11] )
    point_2 = tuple( capture_points[2] ) 
    cv2.line(img, point_1, point_2, color_dict["roi_capture_area"], 2)
    point_1 = tuple( capture_points[10] )
    point_2 = tuple( capture_points[3] ) 
    cv2.line(img, point_1, point_2, color_dict["roi_capture_area"], 2)
    point_1 = tuple( capture_points[9] )
    point_2 = tuple( capture_points[4] ) 
    cv2.line(img, point_1, point_2, color_dict["roi_capture_area"], 2)
    point_1 = tuple( capture_points[8] )
    point_2 = tuple( capture_points[5] ) 
    cv2.line(img, point_1, point_2, color_dict["roi_capture_area"], 2)
    point_1 = tuple( capture_points[7] )
    point_2 = tuple( capture_points[6] ) 
    cv2.line(img, point_1, point_2, color_dict["roi_capture_area"], 2)

    return img