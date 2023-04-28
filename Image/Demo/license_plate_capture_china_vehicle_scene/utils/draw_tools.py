import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


color_dict = {
                "car": (0, 255, 0), 
                "car_capture": (0, 0, 255), 
                "plate": (0, 255, 0), 
                "plate_capture": (0, 0, 255), 
                "text_capture": (255, 0, 0), 
                "roi_capture_area": (255, 255, 255),
            }

draw_alarm_frame_num_threshold = 25
draw_frame_num_threshold = 50
dispare_frame_num_threshold = 1000

capture_draw_dict = {}
capture_draw_dict['id'] = 0
capture_draw_dict['plate_ocr'] = 0
capture_draw_dict['draw_frame_num'] = 0

capture_draw_container = {}


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


def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    # 判断是否OpenCV图片类型
    if (isinstance(img, np.ndarray)):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype("font/simsun.ttc", textSize, encoding="utf-8")

    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)

    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


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

    # update
    pop_key_list = []
    for key, capture_draw_idy in capture_draw_container.items():
        # pop
        if capture_draw_idy['draw_frame_num'] > dispare_frame_num_threshold:
            pop_key_list.append(key)
        capture_draw_idy['draw_frame_num'] += 1
    # pop
    for idx in range(len(pop_key_list)):
        capture_draw_container.pop(pop_key_list[idx])

    # 抓怕结果
    if capture_dict:
        for idy, capture_list_idy in capture_dict.items():
            if capture_list_idy['id'] == 'None':
                continue
            if capture_list_idy['id'] not in capture_draw_container:
                capture_draw_dict = {}
                capture_draw_dict['id'] = idy
                capture_draw_dict['plate_ocr'] = capture_list_idy['plate_ocr']
                capture_draw_dict['draw_frame_num'] = 0
                capture_draw_container[capture_list_idy['id']] = capture_draw_dict

    # 绘制报警信息
    x, y = 850, 50 
    for key, capture_draw_idy in capture_draw_container.items():

        if capture_draw_idy['draw_frame_num'] >= draw_frame_num_threshold:
            continue

        # 报警状态
        capture_bool = False
        if capture_draw_idy['draw_frame_num'] < draw_alarm_frame_num_threshold and \
            not str(capture_draw_idy['plate_ocr']).startswith("0") and len(str(capture_draw_idy['plate_ocr'])) > 0:
            capture_bool = True

        if str(capture_draw_idy['plate_ocr']).startswith("0"):
            continue

        text = "抓拍结果：{}".format( capture_draw_idy['plate_ocr'] )
        text_size = cv2.getTextSize(text, 0, 3, 2)
        cv2.rectangle(img, (x, y), (x + int(text_size[0][0] * 0.8), y + text_size[0][1] + text_size[1]), (255,255,255), -1)
        # img = NiceBox(img, (x, y, x + text_size[0][0] + 1 * text_size[1], y + text_size[0][1] + text_size[1]), (255,255,255), thickness=5)
        
        if not capture_bool:
            # img = cv2.putText(img, text, (x, y + text_size[0][1]), cv2.FONT_HERSHEY_COMPLEX, 3, color_dict["plate"], 2)
            img = cv2ImgAddText(img, text, x, y, textColor=color_dict["plate"], textSize=100)
        else:
            # img = cv2.putText(img, text, (x, y + text_size[0][1]), cv2.FONT_HERSHEY_COMPLEX, 3, color_dict["plate_capture"], 2)
            img = cv2ImgAddText(img, text, x, y, textColor=color_dict["text_capture"], textSize=100)

        y += 100

    # draw
    for idx in range(len(bbox_info)):
        bbox_info_idx = bbox_info[idx]
        
        # 报警状态
        capture_bool = False
        for key, capture_draw_idy in capture_draw_container.items():
            if bbox_info_idx['id'] == capture_draw_idy['id'] and capture_draw_idy['draw_frame_num'] < draw_alarm_frame_num_threshold and \
            not str(capture_draw_idy['plate_ocr']).startswith("0") and len(str(capture_draw_idy['plate_ocr'])) > 0:
                capture_bool = True

        # car
        bbox_info_idx['loc'] = [int(b + 0.5) for b in bbox_info_idx['loc'][:4]]
        if not capture_bool:
            # img = cv_plot_rectangle(img, bbox_info_idx['loc'], mode=mode, color=color_dict["car"])
            img = NiceBox(img, bbox_info_idx['loc'], color_dict["car"], thickness=3, mask=False)
        else:
            # img = cv_plot_rectangle(img, bbox_info_idx['loc'], mode=mode, color=color_dict["car_capture"])
            img = NiceBox(img, bbox_info_idx['loc'], color_dict["car_capture"], thickness=3, mask=False)

        # img = cv2.putText(img, "{}_{}_{}_{}_{}_{}_{}_{:.2f}_{:.2f}".format( bbox_info_idx['id'], bbox_info_idx['frame_num'], bbox_info_idx['up_down_state'], bbox_info_idx['up_down_state_frame_num'], bbox_info_idx['left_right_state'], bbox_info_idx['left_right_state_frame_num'], bbox_info_idx['attri'], bbox_info_idx['up_down_speed'], bbox_info_idx['left_right_speed'] ), (bbox_info_idx['loc'][0], bbox_info_idx['loc'][1] - 10), 
        #                     cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        # license_plate
        if len(bbox_info_idx['plate_loc']):
            
            bbox_info_idx['plate_loc'] = [int(b + 0.5) for b in bbox_info_idx['plate_loc'][:4]]
            text = "{}".format( bbox_info_idx['plate_ocr'])
            text_size = cv2.getTextSize(text, 0, 3, 2)
            if not capture_bool:
                # img = cv_plot_rectangle(img, bbox_info_idx['plate_loc'], mode=mode, color=color_dict["plate"], thickness=3)
                img = NiceBox(img, bbox_info_idx['plate_loc'], color_dict["plate"], thickness=3, mask=False)

                # img = cv2.putText(img, "{}".format( bbox_info_idx['plate_ocr']), \
                #                     (bbox_info_idx['plate_loc'][0], bbox_info_idx['plate_loc'][1] - 15), 
                #                     cv2.FONT_HERSHEY_COMPLEX, 3, color_dict["plate"], 2)
                # img = cv2ImgAddText(img, text, bbox_info_idx['plate_loc'][0], bbox_info_idx['plate_loc'][1] - 15 - text_size[0][1], textColor=color_dict["plate"], textSize=60)
            else:
                # img = cv_plot_rectangle(img, bbox_info_idx['plate_loc'], mode=mode, color=color_dict["plate_capture"], thickness=3)
                img = NiceBox(img, bbox_info_idx['plate_loc'], color_dict["plate_capture"], thickness=3, mask=False)

                # img = cv2.putText(img, "{}".format( bbox_info_idx['plate_ocr']), \
                #                     (bbox_info_idx['plate_loc'][0], bbox_info_idx['plate_loc'][1] - 15), 
                #                     cv2.FONT_HERSHEY_COMPLEX, 3, color_dict["plate_capture"], 2)
                # img = cv2ImgAddText(img, text, bbox_info_idx['plate_loc'][0], bbox_info_idx['plate_loc'][1] - 15 - text_size[0][1], textColor=color_dict["plate_capture"], textSize=60)
                                
    return img


def draw_bbox_state(img, bbox_state_map, type='face'):
    
    img_width = img.shape[1]
    img_height = img.shape[0]
    for key, bbox_state_idy in bbox_state_map.items():
        
        color = get_color(abs(bbox_state_idy['id']))
        
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