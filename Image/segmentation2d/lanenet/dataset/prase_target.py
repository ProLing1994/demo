import xml.etree.ElementTree as ET
import numpy as np
import json
import cv2


class BaseTargetParser(object):
    def __init__(self, class_names=None, point_nums=11) -> None:
        super().__init__()
        self.class_names = class_names
        self.point_nums = point_nums
        if class_names is not None:
            self.obj_instance_count = [0] * len(self.class_names)
        self.obj_instance_dicts = {}
        self.tree = None

    def parse(self, label_file):
        if label_file.endswith('.xml'):
            r = self.parse_xml(label_file, self.class_names)
        elif label_file.endswith('keypoints.json'):
            r = self.parse_keypoints(label_file, self.class_names)
        elif label_file.endswith('.json'):
            r = self.parse_json(label_file, self.class_names)
        return r

    def parse_xml(self, label_file, class_names):
        with open(label_file, encoding='utf-8') as f:
            tree = ET.parse(f)
        self.tree = tree
        r = {
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []
        for obj in tree.findall("object"):
            cls = obj.find("name").text
            bbox = obj.find("bndbox")
            bbox = [int(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]

            # assert 0 <= bbox[0] <= r["width"]
            # assert 0 <= bbox[2] <= r["width"]
            # assert 0 <= bbox[1] <= r["height"]
            # assert 0 <= bbox[3] <= r["height"]
            size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            # print(size)
            # assert size > 10

            if class_names is not None:
                self.obj_instance_count[class_names.index(cls)] += 1

            if cls not in self.obj_instance_dicts:
                self.obj_instance_dicts[cls] = 1
            else:
                self.obj_instance_dicts[cls] += 1

            instances.append(
                {
                    "category_id": class_names.index(cls) if (class_names is not None) else None,
                    "class_name": cls,
                    "bbox": bbox,
                }
            )

        # assert len(instances) > 0
        r["annotations"] = instances
        return r

    def parse_json(self, label_file, class_names):
        with open(label_file, 'r') as f:
            tree = json.load(f)
        self.tree = tree
        r = {
            "height": tree["imageHeight"],
            "width": tree["imageWidth"],
        }
        instances = []
        for line_dict in tree['shapes']:
            cls = line_dict['label']
            points = []

            for point in line_dict['points']:
                x, y = point
                # assert 0 <= x <= r["width"]
                # assert 0 <= y <= r["height"]
                points.append([x, y])


            rect = cv2.boundingRect(np.array(points))

            assert rect[2] * rect[3] > 10, "area is not suitable"
            if class_names is not None:
                self.obj_instance_count[class_names.index(cls)] += 1

            if cls not in self.obj_instance_dicts:
                self.obj_instance_dicts[cls] = 0
            else:
                self.obj_instance_dicts[cls] += 1

            instances.append(
                {
                    "category_id": class_names.index(cls) if (class_names is not None) else None,
                    "class_name": cls,
                    "points": points,
                }
            )

        r["annotations"] = instances

        return r

    def parse_keypoints(self, label_file, class_names):
        with open(label_file, 'r') as f:
            tree = json.load(f)
        self.tree = tree
        r = {
            "height": tree["imageHeight"],
            "width": tree["imageWidth"],
        }
        instances = []
        for line_dict in tree['shapes']:
            cls = line_dict['label']
            final_points = []

            points = line_dict['points']
            point_types = line_dict['point_type']

            assert len(points) == len(point_types) == self.point_nums, "point nums is not correct"

            for point, point_type in zip(points, point_types):
                x, y = point
                if x == 0 or y == 0 or x == r["width"] - 1 or y == r["height"] - 1:
                    point_type = 0
                else:
                    point_type = 2
                # assert 0 <= x <= r["width"]
                # assert 0 <= y <= r["height"]
                final_points.append([x, y, point_type])

            if class_names is not None:
                self.obj_instance_count[class_names.index(cls)] += 1

            if cls not in self.obj_instance_dicts:
                self.obj_instance_dicts[cls] = 0
            else:
                self.obj_instance_dicts[cls] += 1

            instances.append(
                {
                    "category_id": class_names.index(cls) if (class_names is not None) else None,
                    "class_name": cls,
                    "points": final_points,
                }
            )

        r["annotations"] = instances

        return r

class PlatformTargetParser(object):
    def __init__(self, seg_classes, box_classes) -> None:
        super().__init__()
        self.seg_classes = seg_classes
        self.box_classes = box_classes
        self.seg_instance_count = [0] * len(self.seg_classes)
        self.box_instance_count = [0] * len(self.box_classes)

    def parse(self, label_file):
        with open(label_file, 'r', encoding="utf-8") as f:
            tree = json.load(f)
        r = {
            "height": tree["height"],
            "width": tree["width"],
        }
        instances = []
        for line_dict in tree['shapes']:
            flag = 1
            
            cls = line_dict['label']
            if line_dict["type"] == "polygon":
                
                # assert cls in self.seg_classes, "{} not in specific seg_class".format(cls)
                if cls in self.seg_classes:
                    self.seg_instance_count[self.seg_classes.index(cls)] += 1
                points = []

                for point in np.array(line_dict['points']).reshape(-1, 2):
                    x, y = point
                    x = max(0, min(x, tree["width"] - 1))
                    y = max(0, min(y, tree["height"] - 1))
                    points.append([x, y])
                rect = cv2.boundingRect(np.array(points, dtype=int))
                assert rect[2] * rect[3] > 10, "area is not suitable"
                
            elif line_dict["type"] == "rectangle":
                # assert cls in self.box_classes, "{} not in specific box_class".format(cls)
                if cls in self.box_classes:
                    self.box_instance_count[self.box_classes.index(cls)] += 1
                points = []

                for point in np.array(line_dict['points']).reshape(-1, 2):
                    x, y = point
                    x = max(0, min(x, tree["width"] - 1))
                    y = max(0, min(y, tree["height"] - 1))
                    points.append([x, y])
                size = (points[1][0] - points[0][0]) * (points[1][1] - points[0][1])
                # assert size > 0
                if size <= 5:
                    flag = 0

            else:
                print(line_dict["type"])
                raise

            if flag:
                instances.append(
                    {
                        "class_name": cls,
                        "points": points,
                        "type": line_dict["type"], 
                        "source": line_dict["source"], 
                    }
                )
        # assert len(instances) > 0
        r["annotations"] = instances

        return r
