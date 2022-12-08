import xml.etree.ElementTree as ET


def load_bbox_xml(annotation_path):

    bbox_list = [] # [{'label': label, 'bbox': [x1, y1, x2, y2]}]

    tree = ET.parse(annotation_path)
    root = tree.getroot()

    for obj in root.findall('object'):
        name = obj.find('name').text
        bnd_box = obj.find('bndbox')
        bbox = [
            int(bnd_box.find('xmin').text),
            int(bnd_box.find('ymin').text),
            int(bnd_box.find('xmax').text),
            int(bnd_box.find('ymax').text)
        ]

        bbox_list.append({"label": name, "bbox":bbox})
    
    return bbox_list