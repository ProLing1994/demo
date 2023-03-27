import glob
import os

if __name__ == '__main__':
    input_dir = "/yuanhuan/data/image/RM_C28_detection/america_new/JPEGImages/"
    # input_dir = "/yuanhuan/data/image/RM_C28_detection/america_new/Json/"
    # input_dir = "/yuanhuan/data/image/RM_C28_detection/america_new/Annotations/"
    old_file_format = '9雨天1'
    new_file_format = 'rain_015'
    file_type = ".jpg"
    # file_type = ".json"
    # file_type = ".xml"

    file_list = glob.glob(os.path.join(input_dir, '*' + file_type))
    file_list.sort()

    for idx in range(len(file_list)):
        file_path = file_list[idx]
        file_name = os.path.basename(file_path)

        # 自定义重命名规则
        if old_file_format in file_name:
            rename_path = os.path.join(os.path.dirname(file_path), file_name.replace(old_file_format, new_file_format))

            print(file_path, '->', rename_path)
            os.rename(file_path, rename_path)
        