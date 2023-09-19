import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


draw_alarm_frame_num_threshold = 25
draw_frame_num_threshold = 50

color_dict = {
                "car": (0, 255, 0), 
                "car_capture": (0, 0, 255), 
                "plate": (0, 255, 0), 
                "plate_capture": (0, 0, 255), 
                "plate_capture_reverse": (255, 0, 0), 
                "intersect_point_alarm": (255, 0, 0), 
                "roi_capture_area": (255, 255, 255),
                "roi_capture_area": (255, 255, 255),
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


class DrawApi():
    """
    DrawApi
    """

    def __init__(self, demo_type, country_type):
        self.demo_type = demo_type
        self.country_type = country_type


    def draw_bbox_info(self, img, bbox_info, capture_container=None, capture_res_container=None, mode='xywh'):


        # 绘制报警信息
        x, y = 50, 50 
        for _, capture_res_idy in capture_res_container.items():

            if capture_res_idy['capture']['capture_frame_num'] >= draw_frame_num_threshold:
                continue

            # 报警状态
            capture_bool = False
            if capture_res_idy['capture']['capture_frame_num'] < draw_alarm_frame_num_threshold:
                capture_bool = True

            if self.demo_type == "lpr":
                
                text = "抓拍结果：{}_{}".format( capture_res_idy['capture_occupying']['cap_lpr_num'], capture_res_idy['capture_occupying']['cap_lpr_color'] )
                
                text_size = cv2.getTextSize(text, 0, 3, 2)
                cv2.rectangle(img, (x, y), (x + int(text_size[0][0] * 0.8), y + text_size[0][1] + text_size[1]), (255,255,255), -1)
                # img = NiceBox(img, (x, y, x + text_size[0][0] + 1 * text_size[1], y + text_size[0][1] + text_size[1]), (255,255,255), thickness=5)
                
                if not capture_bool:
                    # img = cv2.putText(img, text, (x, y + text_size[0][1]), cv2.FONT_HERSHEY_COMPLEX, 3, color_dict["plate"], 2)
                    img = cv2ImgAddText(img, text, x, y, textColor=color_dict["plate"], textSize=100)
                else:
                    # img = cv2.putText(img, text, (x, y + text_size[0][1]), cv2.FONT_HERSHEY_COMPLEX, 3, color_dict["plate_capture"], 2)
                    img = cv2ImgAddText(img, text, x, y, textColor=color_dict["plate_capture_reverse"], textSize=100)

                y += 100

        # draw
        for idx in range(len(bbox_info)):
            bbox_info_idx = bbox_info[idx]
            
            # 报警状态
            capture_bool = False
            for _, capture_res_idy in capture_res_container.items():
                if bbox_info_idx['track_id'] == capture_res_idy['track_id'] and capture_res_idy['capture']['capture_frame_num'] < draw_alarm_frame_num_threshold:
                    capture_bool = True

            if self.demo_type == "lpr":
                # car
                bbox_info_idx['car_info']['roi'] = [int(b + 0.5) for b in bbox_info_idx['car_info']['roi'][:4]]
                if not capture_bool and len(bbox_info_idx['car_info']['roi']):
                    img = cv_plot_rectangle(img, bbox_info_idx['car_info']['roi'], mode=mode, color=color_dict["car"])
                    img = NiceBox(img, bbox_info_idx['car_info']['roi'], color_dict["car"], thickness=3, mask=False)

                    img = cv2.putText(img, "{}_{}_{}".format( int(bbox_info_idx['track_id']), int(bbox_info_idx['state_occupying']['frame_num']), int(bbox_info_idx['state_occupying']['disappear_frame_num'])), (bbox_info_idx['car_info']['roi'][0], bbox_info_idx['car_info']['roi'][1] - 10), 
                                        cv2.FONT_HERSHEY_COMPLEX, 1, color_dict["car"], 2)
                elif len(bbox_info_idx['car_info']['roi']):
                    img = cv_plot_rectangle(img, bbox_info_idx['car_info']['roi'], mode=mode, color=color_dict["car_capture"])
                    img = NiceBox(img, bbox_info_idx['car_info']['roi'], color_dict["car_capture"], thickness=3, mask=False)

                    img = cv2.putText(img, "{}_{}_{}".format( bbox_info_idx['track_id'], bbox_info_idx['state_occupying']['frame_num'], bbox_info_idx['state_occupying']['disappear_frame_num']), (bbox_info_idx['car_info']['roi'][0], bbox_info_idx['car_info']['roi'][1] - 10), 
                                        cv2.FONT_HERSHEY_COMPLEX, 1, color_dict["car_capture"], 2)

                # 车辆线
                car_line_k = bbox_info_idx['state_occupying']['car_line'][0]
                car_line_b = bbox_info_idx['state_occupying']['car_line'][1]
                point_1 = int((bbox_info_idx['car_info']['roi'][1] - car_line_b) / car_line_k), bbox_info_idx['car_info']['roi'][1]
                point_2 = int((bbox_info_idx['car_info']['roi'][3] - car_line_b) / car_line_k), bbox_info_idx['car_info']['roi'][3]
                cv2.line(img, point_1, point_2, color_dict["car"], 2)

                # 占道区域交点
                cv2.circle(img, (int(bbox_info_idx['state_occupying']['intersect_point'][0]), int(bbox_info_idx['state_occupying']['intersect_point'][1])), 3, color_dict["intersect_point_alarm"], 6)

                # license_plate
                if len(bbox_info_idx['plate_info']['roi']):
                    
                    bbox_info_idx['plate_info']['roi'] = [int(b + 0.5) for b in bbox_info_idx['plate_info']['roi'][:4]]
                    if self.country_type == "zd":
                        text = "{}#{}_{}_{}_{}_{}_{}_{:.2f}".format( bbox_info_idx['plate_info']['kind'], bbox_info_idx['plate_info']['num'], bbox_info_idx['plate_info']['country'], bbox_info_idx['plate_info']['city'], bbox_info_idx['plate_info']['car_type'], bbox_info_idx['plate_info']['color'], bbox_info_idx['plate_info']['column'], bbox_info_idx['plate_info']['score'])
                    else:                        
                        # text = "{}{}_{}_{}_{}_{}".format( bbox_info_idx['plate_info']['kind'], bbox_info_idx['plate_info']['num'], bbox_info_idx['plate_info']['color'], int(bbox_info_idx['plate_info']['dist_left_lane_line']), int(bbox_info_idx['plate_info']['dist_right_lane_line']), bbox_info_idx['state']['obj_num'])
                        text = "{}{}_{}_{}".format( bbox_info_idx['plate_info']['kind'], bbox_info_idx['plate_info']['num'], bbox_info_idx['plate_info']['color'], bbox_info_idx['state']['obj_num'])

                    text_size = cv2.getTextSize(text, 0, 3, 2)
                    if not capture_bool:
                        img = cv_plot_rectangle(img, bbox_info_idx['plate_info']['roi'], mode=mode, color=color_dict["plate"], thickness=3)
                        # img = NiceBox(img, bbox_info_idx['plate_info']['roi'], color_dict["plate"], thickness=3, mask=False)

                        # img = cv2.putText(img, "{}".format( bbox_info_idx['plate_info']['num']), \
                        #                     (bbox_info_idx['plate_info']['roi'][0], bbox_info_idx['plate_info']['roi'][1] - 15), 
                        #                     cv2.FONT_HERSHEY_COMPLEX, 3, color_dict["plate"], 2)
                        img = cv2ImgAddText(img, text, bbox_info_idx['plate_info']['roi'][0], bbox_info_idx['plate_info']['roi'][1] - 15 - text_size[0][1], textColor=color_dict["plate"], textSize=60)
                    else:
                        img = cv_plot_rectangle(img, bbox_info_idx['plate_info']['roi'], mode=mode, color=color_dict["plate_capture"], thickness=3)
                        # img = NiceBox(img, bbox_info_idx['plate_info']['roi'], color_dict["plate_capture"], thickness=3, mask=False)

                        # img = cv2.putText(img, "{}".format( bbox_info_idx['plate_info']['num']), \
                        #                     (bbox_info_idx['plate_info']['roi'][0], bbox_info_idx['plate_info']['roi'][1] - 15), 
                        #                     cv2.FONT_HERSHEY_COMPLEX, 3, color_dict["plate_capture"], 2)
                        img = cv2ImgAddText(img, text, bbox_info_idx['plate_info']['roi'][0], bbox_info_idx['plate_info']['roi'][1] - 15 - text_size[0][1], textColor=color_dict["plate_capture_reverse"], textSize=60)

                    if len(bbox_info_idx['plate_info']['kind_roi']):
                        img = cv_plot_rectangle(img, bbox_info_idx['plate_info']['kind_roi'], mode=mode, color=color_dict["plate"], thickness=3)
                    if len(bbox_info_idx['plate_info']['num_roi']):
                        img = cv_plot_rectangle(img, bbox_info_idx['plate_info']['num_roi'], mode=mode, color=color_dict["plate"], thickness=3)

        return img


    def draw_bbox_tracker(self, img, tracker_bboxes, mode='ltrb'):
        
        for idx in range(len(tracker_bboxes)):
            tracker_bbox = tracker_bboxes[idx]

            id = tracker_bbox[-1]
            loc = tracker_bbox[0:4]
            color = get_color(abs(id))

            img = cv_plot_rectangle(img, loc, mode=mode, color=color)
            img = cv2.putText(img, "id: {}".format(int(id)), (int(loc[0]), int(loc[1])), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
            
        return img
    

    def draw_bbox_state(self, img, bbox_state_container):
        
        for _, bbox_state_idy in bbox_state_container.items():
            
            color = get_color(abs(bbox_state_idy['track_id']))
            
            for idx in range(len( bbox_state_idy['state']['center_point_list'] )):
                x = int( bbox_state_idy['state']['center_point_list'][idx][0] + 0.5 )
                y = int( bbox_state_idy['state']['center_point_list'][idx][1] + 0.5 )

                cv2.circle(img, (x, y), 1, color, 2)

        return img


    def draw_capture_line(self, img, capture_line_points, mode='xywh'):
        
        for idx in range(len(capture_line_points)):

            if idx % 2 == 0:

                # line
                cv2.line(img, capture_line_points[idx], capture_line_points[idx + 1], color_dict["roi_capture_area"], 2)

        return img