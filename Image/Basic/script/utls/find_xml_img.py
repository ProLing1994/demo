import os
import shutil


if __name__ == "__main__":

    jpg_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/Brazil/Brazil/Brazil_all/JPEGImages_test/"
    xml_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/Brazil/Brazil/Brazil_all/Annotations_CarBusTruckMotorcyclePlateMotoplate_w_fuzzy/"
    find_xml_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/Brazil/Brazil/Brazil_all/JPEGImages_test/"

    jpg_path = os.path.join(jpg_dir, '%s.jpg')
    xml_path = os.path.join(xml_dir, '%s.xml')
    xml_out_path = os.path.join(find_xml_dir, '%s.xml')

    if not os.path.exists(find_xml_dir):
        os.makedirs(find_xml_dir)

    file_list = os.listdir(jpg_dir)
    file_list = [str(jpg).replace('.jpg', '') for jpg in file_list]

    for idx in range(len(file_list)):
        shutil.copy(xml_path % (file_list[idx]), xml_out_path % (file_list[idx]))
