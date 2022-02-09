import argparse
import os


def trainval_txt(args):

    # mkdir 
    if not os.path.exists(os.path.dirname(args.trainval_path)):
        os.makedirs(os.path.dirname(args.trainval_path))

    jpg_list = os.listdir(args.jpg_dir)
    with open(args.trainval_path, "w") as f:
        for jpg in jpg_list:

            if not jpg.endswith('.jpg'):
                continue

            xml = os.path.join(args.xml_dir, jpg.replace(".jpg", ".xml"))
            assert os.path.exists(xml), xml

            f.write(jpg.replace(".jpg", ""))
            f.write("\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # args.input_dir = "/mnt/huanyuan2/data/image/LicensePlate/China/"
    # args.input_dir = "/mnt/huanyuan2/data/image/LicensePlate/Europe/"
    # args.input_dir = "/mnt/huanyuan2/data/image/LicensePlate/Mexico/"
    args.input_dir = "/mnt/huanyuan2/data/image/RM_ADAS_AllInOne/allinone/"
    args.trainval_path = args.input_dir + "ImageSets/Main/trainval.txt"
    args.jpg_dir =  args.input_dir + "JPEGImages/"
    args.xml_dir =  args.input_dir + "XML/"

    trainval_txt(args)
