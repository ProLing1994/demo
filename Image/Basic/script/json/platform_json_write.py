from copy import deepcopy
import numpy as np
import json
from shapely.geometry import Polygon


class PlatformJsonWriter(object):

    def __init__(self):

        self.NewShape_rec = {
            "label": "",
            "type": "rectangle",
            "points": [],
            "hiddenPoints": [],
            "occluded": 0,
            "source": "auto",
            "group": 0,
            "z_order": 0,
            "attributes": [], 
        }
        self.NewShape_poly = {
            "label": "",
            "type": "polygon",
            "points": [],
            "hiddenPoints": [],
            "occluded": 0,
            "source": "auto",
            "group": 0,
            "z_order": 0,
            "attributes": [], 
        }
        self.NewJson = {
            "version": "v1.0",
            "frame_num": 0,
            "img_name": "",
            "width": 0,
            "height": 0,
            "shapes": [], 
        }


    def write_meta_json(self, meta_json_path, task_name="task", label_list=[]):
        out_dict = {
            "task": {
                "name": task_name,
                "mode": "annotation",
                "overlap": "0",
                "bugtracker": "",
                "labels": label_list
            },
            "jobs": 1
        }
        with open(meta_json_path, 'w') as f:
            f.write(json.dumps(out_dict, ensure_ascii=False, indent=1))


    def write_json(self, imgwidth, imgheight, imgname, jsonpath, frame_num=None, rect_list=None, polygon_dict=None):
        
        json_file = deepcopy(self.NewJson)
        json_file['shapes'] = []
        
        json_file['width'] = int(imgwidth)
        json_file['height'] = int(imgheight)
        json_file['img_name'] = imgname
        
        if frame_num is not None:
            json_file['frame_num'] = frame_num
        
        if rect_list is not None:
            self.add_det_tag(json_file, rect_list)

        if polygon_dict is not None:
            self.add_seg_tag(json_file, polygon_dict)

        with open(jsonpath, "w") as f:
            f.write(json.dumps(json_file, ensure_ascii=False, indent=4, separators=(',', ':')))
        return


    def add_det_tag(self, json_file, rect_list):
        for i in range(len(rect_list)):
            try:
                x1, y1, x2, y2, name = rect_list[i]
                new_shape = deepcopy(self.NewShape_rec)
                new_shape['label'] = name
                new_shape['points'] = np.array([x1,y1,x2,y2], dtype=int).reshape(-1).tolist()
                json_file['shapes'].append(new_shape)
            except:
                pass


    def add_seg_tag(self, json_file, polygon_dict):
        for key, values in polygon_dict.items():
            for points_list in values:
                try:                
                    new_shape = deepcopy(self.NewShape_poly)
                    new_shape['label'] = key
                    new_shape['points'] = points_list
                    json_file['shapes'].append(new_shape)
                except:
                    pass