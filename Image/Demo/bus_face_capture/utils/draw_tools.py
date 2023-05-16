import cv2 
import numpy as np
import os 


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


def cv_plot_rectangle(img, bbox, mode='xywh', color=None, thickness=2):
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


def draw_roi(img, roi_area, mode='ltrb'):

    img = cv_plot_rectangle(img, roi_area, mode=mode, color=(255, 255, 255))

    return img

def draw_bbox_dets(img, bbox_dets, mode='ltrb'):
    
    for key, bbox_det_idx in bbox_dets.items():

        for bbox_idy in range(len(bbox_det_idx)):
            
            img = cv_plot_rectangle(img, bbox_det_idx[bbox_idy][:4], mode=mode, color=(255, 255, 255))

    return img

def draw_bbox_tracker(img, tracker_bboxes, mode='ltrb'):

    for idx in range(len(tracker_bboxes)):
        tracker_bbox = tracker_bboxes[idx]

        id = tracker_bbox[-1]
        loc = tracker_bbox[0:4]
        color = get_color(abs(id))

        img = cv_plot_rectangle(img, loc, mode=mode, color=color)
        img = cv2.putText(img, "id: {}".format(int(id)), (int(loc[0]), int(loc[1])), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
        
    return img


def draw_bbox_info(img, bbox_info, mode='ltrb', type='face'):

    for idx in range(len(bbox_info)):
        bbox_info_idx = bbox_info[idx]

        if type == "face":

            color = get_color(abs(bbox_info_idx['id']))
            bbox_info_idx['loc'] = [int(b + 0.5) for b in bbox_info_idx['loc'][:4]]
            img = cv_plot_rectangle(img, bbox_info_idx['loc'], mode=mode, color=(255, 255, 255))

            bbox_info_idx['stable_loc'] = [int(b + 0.5) for b in bbox_info_idx['stable_loc'][:4]]
            img = cv_plot_rectangle(img, bbox_info_idx['stable_loc'], mode=mode, color=color)

            img = cv2.putText(img, "{}_{}_{}_{}_{:.2f}_{:.2f}".format(bbox_info_idx['id'], bbox_info_idx['frame_num'], bbox_info_idx['state'], bbox_info_idx['state_frame_num'], bbox_info_idx['score'], bbox_info_idx['speed']), (bbox_info_idx['loc'][0], bbox_info_idx['loc'][1] - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        
        elif type == "p3d":

            color = get_color(abs(bbox_info_idx['id']))
            bbox_info_idx['loc'] = [int(b + 0.5) for b in bbox_info_idx['loc'][:4]]

            if bbox_info_idx['captue_bool']:
                img = cv_plot_rectangle(img, bbox_info_idx['loc'], mode=mode, color=(0, 0, 255), thickness=10)
            else:
                img = cv_plot_rectangle(img, bbox_info_idx['loc'], mode=mode, color=color)
            
            img = cv2.putText(img, "{}_{}".format(bbox_info_idx['id'], bbox_info_idx['frame_num']), (bbox_info_idx['loc'][0], bbox_info_idx['loc'][1] - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    return img


def draw_bbox_state(img, bbox_state_map, type='face'):

    img_width = img.shape[1]
    img_height = img.shape[0]
    for key, bbox_state_idy in bbox_state_map.items():
        
        if type == "face":

            color = get_color(abs(bbox_state_idy['id']))

            y_max = np.array(bbox_state_idy['speed_list']).max()

            # print( "face key: {}, len(speed_list): {}".format( bbox_state_idy['id'], len( bbox_state_idy['speed_list'] )) )

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
        
        elif type == "p3d":

            color = get_color(abs(bbox_state_idy['id']))

            # print( "p3d key: {}, len(captue_bool_list): {}".format( bbox_state_idy['id'], len( bbox_state_idy['captue_bool_list'] )) )

            for idx in range( len( bbox_state_idy['captue_bool_list'] )):

                x0 = img_width + 2 * ( idx + 1 - len( bbox_state_idy['captue_bool_list']) )
                x1 = img_width + 2 * ( idx + 1 - len( bbox_state_idy['captue_bool_list']) )

                y0 = int( 500 - 100 )
                y1 = int( 500 + 100 )

                if idx > 0 and  bbox_state_idy['captue_bool_list'][idx] and not bbox_state_idy['captue_bool_list'][idx - 1]:
                    
                    cv2.line(img, (x0,y0), (x1,y1), (255,255,255), 2)
    
    return img

def draw_match_dict(match_dict, output_capture_folder, mode='ltrb'):

    for key, match_dict_idy in match_dict.items():
        
        face_id = match_dict_idy['face_id']
        face_loc = match_dict_idy['face_loc']
        face_frame_idx = match_dict_idy['face_frame_idx']
        face_img = match_dict_idy['face_img']

        face_img = cv_plot_rectangle(face_img, face_loc, mode=mode, color=(0, 0, 255))

        output_face_img_path = os.path.join(output_capture_folder, "{}_{}_{}_face.jpg".format(str(key), str(face_id), str(face_frame_idx)))
        cv2.imwrite(output_face_img_path, face_img)

        p3d_id = match_dict_idy['p3d_id']
        p3d_loc = match_dict_idy['p3d_loc']
        p3d_frame_idx = match_dict_idy['p3d_frame_idx']
        p3d_img = match_dict_idy['p3d_img']

        p3d_img = cv_plot_rectangle(p3d_img, p3d_loc, mode=mode, color=(0, 0, 255))

        output_p3d_img_path = os.path.join(output_capture_folder, "{}_{}_{}_p3d.jpg".format(str(key), str(p3d_id), str(p3d_frame_idx)))
        cv2.imwrite(output_p3d_img_path, p3d_img)
