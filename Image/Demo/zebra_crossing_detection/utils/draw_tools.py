import cv2 
import numpy as np
import os
import json


color_dict = {
                "car_bus_truck": (0, 255, 0), 
                "non_motorized": (255, 0, 0), 
                "person": (0, 0, 255),
                "zebra": (255, 182, 193),
            }


def NiceBox(img,rect,line_color,thickness=3,mask=True,mask_chn=2):
    width=rect[2]-rect[0]
    height=rect[3]-rect[1]
    line_len=max(10,min(width*0.15,height*0.15))
    line_len=int(line_len)
    cv2.line(img,(rect[0],rect[1]),(rect[0]+line_len,rect[1]),line_color,thickness=thickness)
    cv2.line(img,(rect[2]-line_len,rect[1]),(rect[2],rect[1]),line_color,thickness=thickness)
    cv2.line(img,(rect[0],rect[3]),(rect[0]+line_len,rect[3]),line_color,thickness=thickness)
    cv2.line(img,(rect[2]-line_len,rect[3]),(rect[2],rect[3]),line_color,thickness=thickness)
    cv2.line(img,(rect[0],rect[1]),(rect[0],rect[1]+line_len),line_color,thickness=thickness)
    cv2.line(img,(rect[0],rect[3]-line_len),(rect[0],rect[3]),line_color,thickness=thickness)
    cv2.line(img,(rect[2],rect[1]),(rect[2],rect[1]+line_len),line_color,thickness=thickness)
    cv2.line(img,(rect[2],rect[3]-line_len),(rect[2],rect[3]),line_color,thickness=thickness)
    if(mask):
        mask=np.zeros(img.shape[:2],dtype=np.uint8)
        coordinate=[[[rect[0],rect[1]],[rect[2],rect[1]],[rect[2],rect[3]],[rect[0],rect[3]]]]
        coordinate=np.array(coordinate)
        mask=cv2.fillPoly(mask,coordinate,100)
        mask_pos=mask.astype(np.bool)
        #mask_color=np.array(mask_color,dtype=np.uint8)
        img1=img[:,:,mask_chn]
        img1[np.where(mask!=0)] = mask[np.where(mask!=0)]
        img[:,:,mask_chn]=img1
    return img


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

        bbox_info_idx['loc'] = [int(b + 0.5) for b in bbox_info_idx['loc'][:4]]
        # img = cv_plot_rectangle(img, bbox_info_idx['loc'], mode=mode, color=color_dict[bbox_info_idx['label']])
        img = NiceBox(img, bbox_info_idx['loc'], color_dict[bbox_info_idx['label']], mask=False, thickness=3)
        
        label = ''
        if bbox_info_idx['label'] == 'car_bus_truck':
            label = 'car'
        elif bbox_info_idx['label'] == 'non_motorized':
            label = 'non_motorized'
        elif bbox_info_idx['label'] == 'person':
            label = 'person'

        # img = cv2.putText(img, "{}_{}".format( bbox_info_idx['id'], label ), (bbox_info_idx['loc'][0], bbox_info_idx['loc'][1] - 10), 
        #                     cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        img = cv2.putText(img, "{}".format( int(bbox_info_idx['id'])), (bbox_info_idx['loc'][0] - 2, bbox_info_idx['loc'][1] - 10), 
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    # info
    

    return img


def draw_zebra(img, load_zebra_json_dir, json_name):

    json_path = os.path.join(load_zebra_json_dir, json_name)

    # read json
    with open(json_path, 'r', encoding='UTF-8') as fr:
        try:
            annotation = json.load(fr)
        except:
            print(json_path)
    
    for track in annotation['shapes']:

        label = track['label']
        points = np.array(track['points'], dtype=np.int32)

        mask = np.zeros(img.shape, np.int32)
        cv2.fillPoly(mask, [points], color_dict["zebra"], 4)
        
        img = img.astype(np.uint8)
        mask = mask.astype(np.uint8)
        img = cv2.addWeighted(src1=img, alpha=1.0, src2=mask, beta=1.0, gamma=0.)
    
    return img