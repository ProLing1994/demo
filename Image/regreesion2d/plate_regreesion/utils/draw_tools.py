import cv2 


color_dict = {
                "car": (0, 255, 0), 
                "car_capture": (12, 149, 255), 
                "bus": (255, 0, 0), 
                "truck": (0, 255, 255), 
                "plate": (0, 0, 255),
                "ocr_ignore_plate": (12, 149, 255),
                "height_ignore_plate": (57, 104, 205),
                "roi_ignore_area": (255, 255, 255),
                "roi_capture_area": (255, 255, 255),
            }

def cv_plot_rectangle(img, bbox, mode='xywh', color=None, thickness=2):
    if color is None:
        color = color_dict["plate"]
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
    return cv2.rectangle(img_p, (xmin, ymin),
                         (xmax, ymax), color=color, thickness=thickness)


def draw_detection_result(img, bboxes, mode='xywh', color=None):

    for key, values in bboxes.items():
        if key in color_dict:
            color = color_dict[key]
        else:
            color = color_dict["plate"]

        for box in values:

            if len(box) == 5:
                if isinstance(box[0], float):
                    box = [int(b + 0.5) for b in box[:4]]
                img = cv2.putText(img, f"{key}:" + "%.3f" % box[-1], (box[0], box[1]),
                                  cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
                img = cv_plot_rectangle(img, box[:4], mode=mode, color=color)
            else:
                if isinstance(box[0], float):
                    box = [int(b + 0.5) for b in box]
                img = cv2.putText(img, f"{key}", (box[0], box[1] - 10), 
                                  cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
                img = cv_plot_rectangle(img, box, mode=mode, color=color)
    return img

def draw_bbox_info(img, bbox_info, capture_id_list, mode='xywh'):

    for idx in range(len(bbox_info)):
        bbox_info_idx = bbox_info[idx]
        
        capture_bool = bbox_info_idx['id'] in capture_id_list
        # car
        if isinstance(bbox_info_idx['loc'][0], float):
            bbox_info_idx['loc'] = [int(b + 0.5) for b in bbox_info_idx['loc'][:4]]
        img = cv2.putText(img, "{}_{}_{}_{}".format('car', bbox_info_idx['id'], bbox_info_idx['frame_num'], bbox_info_idx['state']), (bbox_info_idx['loc'][0], bbox_info_idx['loc'][1] - 10), 
                            cv2.FONT_HERSHEY_COMPLEX, 1, color_dict["car"], 2)
        if not capture_bool:
            img = cv_plot_rectangle(img, bbox_info_idx['loc'], mode=mode, color=color_dict["car"])
        else:
            img = cv_plot_rectangle(img, bbox_info_idx['loc'], mode=mode, color=color_dict["car_capture"])

        # license_plate
        if len(bbox_info_idx['plate_loc']):
            if isinstance(bbox_info_idx['plate_loc'][0], float):
                bbox_info_idx['plate_loc'] = [int(b + 0.5) for b in bbox_info_idx['plate_loc'][:4]]
            img = cv2.putText(img, "{}".format(bbox_info_idx['plate_ocr']), (bbox_info_idx['plate_loc'][0], bbox_info_idx['plate_loc'][1] - 10), 
                                cv2.FONT_HERSHEY_COMPLEX, 1, color_dict["plate"], 2)
            img = cv_plot_rectangle(img, bbox_info_idx['plate_loc'], mode=mode, color=color_dict["plate"])
    
    return img

def draw_capture_line(img, capture_line, mode='xywh'):

    image_width = img.shape[1]

    for idx in range(len(capture_line)):
        capture_line_idx = int(capture_line[idx])

        # line
        cv2.line(img, (0, capture_line_idx), (image_width, capture_line_idx), color_dict["roi_capture_area"], 2)
    
    return img
    