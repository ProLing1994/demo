import os
from xml.dom.minidom import Document


class XmlWriter(object):
    def __init__(self, filename, image_shape=None):
        self.doc = Document()
        self.annotation = self.doc.createElement('annotation')
        self.doc.appendChild(self.annotation)
        if image_shape is None:
            image_shape = [1080, 1920, 3]
        self.image_shape = image_shape
        self.filename = filename
        self._init_xml(filename)

    def _init_xml(self, image_name):

        folder = self.doc.createElement('folder')
        folder.appendChild(self.doc.createTextNode("VOC2017"))
        self.annotation.appendChild(folder)

        filename = self.doc.createElement('filename')
        filename.appendChild(self.doc.createTextNode(image_name))
        self.annotation.appendChild(filename)

        source = self.doc.createElement('source')
        database = self.doc.createElement('database')
        database.appendChild(self.doc.createTextNode('Unknown'))
        source.appendChild(database)
        self.annotation.appendChild(source)

        size = self.doc.createElement('size')
        width = self.doc.createElement('width')
        width.appendChild(self.doc.createTextNode(str(self.image_shape[1])))
        size.appendChild(width)
        height = self.doc.createElement('height')
        height.appendChild(self.doc.createTextNode(str(self.image_shape[0])))
        size.appendChild(height)
        depth = self.doc.createElement('depth')
        depth.appendChild(self.doc.createTextNode(str(self.image_shape[2])))
        size.appendChild(depth)
        self.annotation.appendChild(size)

        segmented = self.doc.createElement('segmented')
        segmented.appendChild(self.doc.createTextNode(str(0)))
        self.annotation.appendChild(segmented)

    def insert_object(self, datas):
        obj = self.doc.createElement('object')
        name = self.doc.createElement('name')
        name.appendChild(self.doc.createTextNode(datas[0]))
        obj.appendChild(name)
        fixratio = self.doc.createElement('fixratio')
        fixratio.appendChild(self.doc.createTextNode('0'))
        obj.appendChild(fixratio)
        pose = self.doc.createElement('pose')
        pose.appendChild(self.doc.createTextNode('Unspecified'))
        obj.appendChild(pose)
        truncated = self.doc.createElement('truncated')
        truncated.appendChild(self.doc.createTextNode(str(0)))
        obj.appendChild(truncated)
        difficult = self.doc.createElement('difficult')
        difficult.appendChild(self.doc.createTextNode(str(0)))
        obj.appendChild(difficult)
        bndbox = self.doc.createElement('bndbox')

        xmin = self.doc.createElement('xmin')
        xmin.appendChild(self.doc.createTextNode(str(datas[1])))
        bndbox.appendChild(xmin)
        ymin = self.doc.createElement('ymin')
        ymin.appendChild(self.doc.createTextNode(str(datas[2])))
        bndbox.appendChild(ymin)
        xmax = self.doc.createElement('xmax')
        xmax.appendChild(self.doc.createTextNode(str(datas[3])))
        bndbox.appendChild(xmax)
        ymax = self.doc.createElement('ymax')
        ymax.appendChild(self.doc.createTextNode(str(datas[4])))
        bndbox.appendChild(ymax)
        obj.appendChild(bndbox)
        self.annotation.appendChild(obj)

    def write_xml(self, save_path=None):
        if save_path is None:
            save_path = os.path.splitext(self.filename)[0] + ".xml"
        with open(save_path, "wb") as f:
            f.write(self.doc.toprettyxml(indent='    ', encoding='utf-8'))


def write_xml(output_xml_path, image_path, boxes, image_shape=None):
    if "/" in image_path:
        image_path = os.path.split(image_path)[-1]
    xml_writer = XmlWriter(image_path, image_shape)
    for key, values in boxes.items():
        for box in values:
            xml_writer.insert_object([key] + box)
    xml_writer.write_xml(output_xml_path)