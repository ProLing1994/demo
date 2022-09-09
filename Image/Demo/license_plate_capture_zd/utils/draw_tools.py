import cv2 

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


def draw_bbox_info(img, bbox_info, mode='xywh'):
    
    for idx in range(len(bbox_info)):
        bbox_info_idx = bbox_info[idx]
        
        # car
        bbox_info_idx['loc'] = [int(b + 0.5) for b in bbox_info_idx['loc'][:4]]
        img = cv_plot_rectangle(img, bbox_info_idx['loc'], mode=mode, color=color_dict["car"])

        img = cv2.putText(img, "{}".format( bbox_info_idx['id'] ), (bbox_info_idx['loc'][0], bbox_info_idx['loc'][1] - 10), 
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        # license_plate
        if len(bbox_info_idx['plate_loc']):
            
            bbox_info_idx['plate_loc'] = [int(b + 0.5) for b in bbox_info_idx['plate_loc'][:4]]
            img = cv_plot_rectangle(img, bbox_info_idx['plate_loc'], mode=mode, color=color_dict["plate"])

            img = cv2.putText(img, "{}".format("none"), (bbox_info_idx['plate_loc'][0], bbox_info_idx['plate_loc'][1] - 10), 
                                cv2.FONT_HERSHEY_COMPLEX, 1, color_dict["plate"], 2)
                                
    return img
