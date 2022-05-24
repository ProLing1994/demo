import os
import shutil


if __name__ == "__main__":

    jpg_dir = ""
    xml_dir = ""
    find_img_dir = ""

    jpg_path = os.path.join(jpg_dir, '%s.jpg')
    xml_path = os.path.join(xml_dir, '%s.xml')
    jpg_out_path = os.path.join(find_img_dir, '%s.jpg')

    if not os.path.exists(find_img_dir):
        os.makedirs(find_img_dir)

    file_list = os.listdir(xml_dir)
    file_list = [str(xml).replace('.xml', '') for xml in file_list]

    for idx in range(len(file_list)):
        shutil.copy(jpg_path % (file_list[idx]), jpg_out_path % (file_list[idx]))
