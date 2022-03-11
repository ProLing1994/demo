import cv2 


color_dict = {
                "car": (0, 0, 255), 
                "bus": (255, 0, 0), 
                "truck": (0, 255, 255), 
                "plate": (0, 255, 0),
                "ocr_ignore_plate": (12, 149, 255),
                "height_ignore_plate": (57, 104, 205),
            }

def cv_plot_rectangle(img, bbox, mode='xywh', color=None,thickness=2):
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