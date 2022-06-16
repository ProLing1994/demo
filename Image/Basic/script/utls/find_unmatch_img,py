import os
import shutil

if __name__ == "__main__":
    
    jpg_dir = "/yuanhuan/data/image/ZG_BMX_detection/daminghu_new/daminghu/JPEGImages/"
    mathch_jpg_dir = "/yuanhuan/data/image/ZG_BMX_detection/daminghu/JPEGImages/"
    jpg_out_dir = "/yuanhuan/data/image/ZG_BMX_detection/daminghu_new/daminghu/unmath_JPEGImages/"

    jpg_path = os.path.join(jpg_dir, '%s.jpg')
    mathch_jpg_path = os.path.join(mathch_jpg_dir, '%s.jpg')
    jpg_out_path = os.path.join(jpg_out_dir, '%s.jpg')

    if not os.path.exists(jpg_out_dir):
        os.makedirs(jpg_out_dir)

    file_list = os.listdir(jpg_dir)
    file_list = [str(jpg).replace('.jpg', '') for jpg in file_list]

    for idx in range(len(file_list)):
        if not os.path.exists(mathch_jpg_path % (file_list[idx])):
            shutil.copy(jpg_path % (file_list[idx]), jpg_out_path % (file_list[idx]))