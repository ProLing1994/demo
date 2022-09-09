import json
import numpy as np


def load_bbox_json(annotation_path):

    bbox_list = [] # [{'label': label, 'bbox': [x1, y1, x2, y2]}]

    # json
    with open(annotation_path, 'r', encoding='UTF-8') as fr:
        annotation = json.load(fr)

        # info
        img_width = annotation['width']
        img_height = annotation['height']

        # load annotation
        for track in annotation['shapes']:
            label = track['label']
            points = np.array(track['points'])
            type = track['type']

            if not type == 'rectangle':
                continue

            x1 = max(int(points[0]) + 1, 1)
            y1 = max(int(points[1]) + 1, 1)
            x2 = min(int(points[2]) + 1, img_width) 
            y2 = min(int(points[3]) + 1, img_height)

            bbox_list.append({"label": label, "bbox":[x1, y1, x2, y2]})
    
    return bbox_list


def load_mask_json(annotation_path):

    mask_list = [] # [{'label': label, 'corner': [points]}]

    # json
    with open(annotation_path, 'r', encoding='UTF-8') as fr:
        annotation = json.load(fr)

        # info
        img_width = annotation['width']
        img_height = annotation['height']

        # load annotation
        for track in annotation['shapes']:
            label = track['label']
            points = np.array(track['points'])
            type = track['type']

            if not type == 'polygon':
                continue
            
            points_list = []
            
            for idx in range(0, len(points), 2):

                points_idx = [points[idx], points[idx + 1]]
                points_idx[0] = min(max(int(points_idx[0]) + 1, 1), img_width)
                points_idx[1] = min(max(int(points_idx[1]) + 1, 1), img_height)

                points_list.append(points_idx)
                
            mask_list.append({"label": label, "corner":points_list})
    
    return mask_list