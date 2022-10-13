import cv2 

color_dict = {
                "plate": (0, 255, 0), 
                "plate_capture": (12, 149, 255), 
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


def draw_bbox_info(img, bbox_info, capture_list=None, mode='xywh'):
    
    for idx in range(len(bbox_info)):
        bbox_info_idx = bbox_info[idx]

        capture_bool = False
        if capture_list:
            for idy, _ in capture_list.items():
                if bbox_info_idx['id'] == idy:
                    capture_bool = True

        # license_plate
        bbox_info_idx['loc'] = [int(b + 0.5) for b in bbox_info_idx['loc'][:4]]
        if not capture_bool:
            img = cv_plot_rectangle(img, bbox_info_idx['loc'], mode=mode, color=color_dict["plate"])
            img = cv_plot_rectangle(img, bbox_info_idx['kind_loc'], mode=mode, color=color_dict["plate"])
            img = cv_plot_rectangle(img, bbox_info_idx['num_loc'], mode=mode, color=color_dict["plate"])
        else:
            img = cv_plot_rectangle(img, bbox_info_idx['loc'], mode=mode, color=color_dict["plate_capture"])
            img = cv_plot_rectangle(img, bbox_info_idx['kind_loc'], mode=mode, color=color_dict["plate_capture"])
            img = cv_plot_rectangle(img, bbox_info_idx['num_loc'], mode=mode, color=color_dict["plate_capture"])

        img = cv2.putText(img, "{}_{}_{}_{}_{}_{}_{}_{:.2f}".format( bbox_info_idx['id'], bbox_info_idx['country'], bbox_info_idx['city'], bbox_info_idx['car_type'], bbox_info_idx['column'], bbox_info_idx['kind'], bbox_info_idx['num'], bbox_info_idx['score'] ), \
                            (bbox_info_idx['loc'][0], bbox_info_idx['loc'][1] - 10), 
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        img = cv2.putText(img, "{}_{}_{}_{}_{}_{}_{:.2f}_{:.2f}".format( bbox_info_idx['frame_num'], bbox_info_idx['lpr_num'], bbox_info_idx['up_down_state'], bbox_info_idx['up_down_state_frame_num'], bbox_info_idx['left_right_state'], bbox_info_idx['left_right_state_frame_num'], bbox_info_idx['up_down_speed'], bbox_info_idx['left_right_speed']), \
                            (bbox_info_idx['loc'][0], bbox_info_idx['loc'][3] + 10), 
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    return img


def draw_bbox_state(img, bbox_state_map):
    
    for key, bbox_state_idy in bbox_state_map.items():
        
        color = get_color(abs(bbox_state_idy['id']))
        
        for idx in range(len( bbox_state_idy['center_point_list'] )):
            x = int( bbox_state_idy['center_point_list'][idx][0] + 0.5 )
            y = int( bbox_state_idy['center_point_list'][idx][1] + 0.5 )

            cv2.circle(img, (x, y), 1, color, 2)

    return img


def draw_capture_line(img, capture_line_up_down, capture_line_left_right, mode='xywh'):
    
    image_width = img.shape[1]
    image_height = img.shape[0]

    for idx in range(len(capture_line_up_down)):

        capture_line_idx = int(capture_line_up_down[idx])

        # line
        cv2.line(img, (0, capture_line_idx), (image_width, capture_line_idx), color_dict["roi_capture_area"], 2)

    for idx in range(len(capture_line_left_right)):
    
        capture_line_idx = int(capture_line_left_right[idx])

        # line
        cv2.line(img, (capture_line_idx, 0), (capture_line_idx, image_height), color_dict["roi_capture_area"], 2)

    return img