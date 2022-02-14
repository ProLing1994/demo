 
from lxml import etree

  
class XmlAdder(object):
    def __init__(self, filename):
        self.parser = etree.XMLParser(remove_blank_text=True)
        self.root = etree.parse(filename, self.parser)

    def add_object(self, label, bbox):
        
        for idx in range(len(bbox)):

            xmin = bbox[idx][0]
            ymin = bbox[idx][1]
            xmax = bbox[idx][2]
            ymax = bbox[idx][3]
            
            object = etree.Element("object")

            namen = etree.SubElement(object, "name")
            namen.text = label
            object.append(namen)

            fixratio = etree.SubElement(object, "fixratio")
            fixratio.text = str(0)
            object.append(fixratio)

            bndbox = etree.SubElement(object,"bndbox")
            xminn = etree.SubElement(bndbox,"xmin")
            xminn.text = str(xmin)
            bndbox.append(xminn)
            yminn = etree.SubElement(bndbox,"ymin")
            yminn.text = str(ymin)
            bndbox.append(yminn)
            xmaxn = etree.SubElement(bndbox,"xmax")
            xmaxn.text = str(xmax)
            bndbox.append(xmaxn)
            ymaxn = etree.SubElement(bndbox,"ymax")
            ymaxn.text = str(ymax)

            pose = etree.SubElement(object,"pose")
            pose.text = "Unspecified"
            object.append(pose)

            truncated = etree.SubElement(object,"truncated")
            truncated.text = str(0)
            object.append(truncated)

            difficult = etree.SubElement(object,"difficult")
            difficult.text = str(0)
            object.append(difficult)

            self.root.getroot().append(object)

    def write_xml(self, output_xml_path):
        self.tree = etree.ElementTree(self.root.getroot())
        self.tree.write(output_xml_path, pretty_print=True, xml_declaration=False, encoding='utf-8')