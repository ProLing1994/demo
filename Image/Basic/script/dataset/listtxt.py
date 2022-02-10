import os


xml_list = "/mnt/huanyuan2/data/image/Recording/ZG_ZHJYZ_detection/ImageSets/Main/trainval.txt"
jpg_dir = "/mnt/huanyuan2/data/image/Recording/ZG_ZHJYZ_detection/JPEGImages/"
xml_dir = "/mnt/huanyuan2/data/image/Recording/ZG_ZHJYZ_detection/Annotations/"

# mkdir 
if not os.path.exists(os.path.dirname(xml_list)):
    os.makedirs(os.path.dirname(xml_list))

jpg_list = os.listdir(jpg_dir)
with open(xml_list, "w") as f:
    for jpg in jpg_list:
        xml = os.path.join(xml_dir, jpg.replace(".jpg", ".xml"))
        assert os.path.exists(xml), xml

        f.write(jpg.replace(".jpg", ""))
        f.write("\n")