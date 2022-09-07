import numpy as np
import json


class JsonWriter(object):
    def __init__(self, filename, image_shape):
        self.json_dict = {}
        self.json_dict["shapes"] = []
        self.json_dict["imageHeight"] = int(image_shape[0])
        self.json_dict["imageWidth"] = int(image_shape[1])
        self.json_dict["version"] = "20C.03"
        self.json_dict["flags"] = {}
        self.json_dict["imagePath"] = str(filename)
        self.json_dict["imageData"] = None


    def insert_object(self, key, box, bboxes_type):
        # box: [x, y, w, h]
        # 目前只支持多边形 polygon
        assert bboxes_type in ['polygon']

        obj_dict = {}
        obj_dict["label"] = key
        obj_dict["line_color"] = [int(0), int(255), int(0), int(128)]
        obj_dict["fill_color"] = [int(255), int(0), int(0), int(128)]

        x1 = int(box[0])
        x2 = int(box[0] + box[2])
        y1 = int(box[1])
        y2 = int(box[1] + box[3])
        obj_dict["points"] = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]

        obj_dict["z_order"] = int(len(self.json_dict["shapes"]))
        obj_dict["shape_type"] = bboxes_type

        self.json_dict["shapes"].append(obj_dict)
    

    def write_json(self, save_path):
        with open(save_path, "w") as f:
            json.dump(self.json_dict, f)


def write_json(output_json_path, filename, image_shape, bboxes, bboxes_type):
    json_writer = JsonWriter(filename, image_shape)

    for key, values in bboxes.items():
        for box in values:
            json_writer.insert_object(key, box, bboxes_type)

    json_writer.write_json(output_json_path)


if __name__ == '__main__':
    output_json_path = "/yuanhuan/data/test/test.json"
    filename = "20220819_000002_UAE_DUBAI_none_T#72058.jpg"
    image_shape = np.array([110, 170])
    bboxes = {'kind': [[ 14, 15, 50, 25]]}          # { 'label': [[x, y, w, h]] }
    bboxes_type = "polygon"
    write_json(output_json_path, filename, image_shape, bboxes, bboxes_type)
