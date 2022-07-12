import cv2 
import numpy as np

color_dict = {
                "car": (0, 255, 0), 
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


def draw_bbox_info(img, bbox_info, capture_dict=None, mode='xywh'):
    
    for idx in range(len(bbox_info)):
        bbox_info_idx = bbox_info[idx]
        
        capture_bool = False
        if capture_dict:
            for idy, _ in capture_dict.items():
                if bbox_info_idx['id'] == idy:
                    capture_bool = True

        # car
        color = get_color(abs(bbox_info_idx['id']))
        bbox_info_idx['loc'] = [int(b + 0.5) for b in bbox_info_idx['loc'][:4]]
        if not capture_bool:
            img = cv_plot_rectangle(img, bbox_info_idx['loc'], mode=mode, color=color_dict["car"])
        else:
            img = cv_plot_rectangle(img, bbox_info_idx['loc'], mode=mode, color=color_dict["car_capture"])
        
        # bbox_info_idx['stable_loc'] = [int(b + 0.5) for b in bbox_info_idx['stable_loc'][:4]]
        # img = cv_plot_rectangle(img, bbox_info_idx['stable_loc'], mode=mode, color=color)

        img = cv2.putText(img, "{}_{}_{}_{}_{}_{:.2f}".format( bbox_info_idx['id'], bbox_info_idx['frame_num'], bbox_info_idx['state'], bbox_info_idx['state_frame_num'], bbox_info_idx['attri'], bbox_info_idx['speed'] ), (bbox_info_idx['loc'][0], bbox_info_idx['loc'][1] - 10), 
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        # license_plate
        if len(bbox_info_idx['plate_loc']):
            
            bbox_info_idx['plate_loc'] = [int(b + 0.5) for b in bbox_info_idx['plate_loc'][:4]]
            img = cv_plot_rectangle(img, bbox_info_idx['plate_loc'], mode=mode, color=color_dict["plate"])

            img = cv2.putText(img, "{}".format(bbox_info_idx['plate_ocr']), (bbox_info_idx['plate_loc'][0], bbox_info_idx['plate_loc'][1] - 10), 
                                cv2.FONT_HERSHEY_COMPLEX, 1, color_dict["plate"], 2)
                                
    return img


def draw_bbox_state(img, bbox_state_map, type='face'):
    
    img_width = img.shape[1]
    img_height = img.shape[0]
    for key, bbox_state_idy in bbox_state_map.items():
        
        color = get_color(abs(bbox_state_idy['id']))

        y_max = np.array(bbox_state_idy['speed_list']).max()

        for idx in range( len( bbox_state_idy['speed_list'] )):
            x = img_width + 2 * ( idx + 1 - len( bbox_state_idy['speed_list']) )
            y = int( 500 - (5 * bbox_state_idy['speed_list'][idx]) + 0.5 )
            
            if bbox_state_idy['speed_list'][idx] == y_max and abs(y_max) > 5.0:
                cv2.circle(img, (x, y), 1, color, 20)
            else:
                cv2.circle(img, (x, y), 1, color, 2)
        
        for idx in range(len( bbox_state_idy['center_point_list'] )):
            x = int( bbox_state_idy['center_point_list'][idx][0] + 0.5 )
            y = int( bbox_state_idy['center_point_list'][idx][1] + 0.5 )

            cv2.circle(img, (x, y), 1, color, 2)

    return img


def draw_capture_line(img, capture_line, mode='xywh'):
    
    image_width = img.shape[1]

    for idx in range(len(capture_line)):

        capture_line_idx = int(capture_line[idx])

        # line
        cv2.line(img, (0, capture_line_idx), (image_width, capture_line_idx), color_dict["roi_capture_area"], 2)
    
    return img