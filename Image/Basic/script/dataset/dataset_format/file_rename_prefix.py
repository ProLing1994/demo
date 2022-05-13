import glob
import os

if __name__ == '__main__':
    input_dir = "/mnt/huanyuan2/data/image/ZG_AHHBGS_detection/kakou4/JPEGImages/"
    old_file_format = '卡口4'
    new_file_format = 'kakou4_'
    file_type = ".jpg"
    # file_type = ".json"

    file_list = glob.glob(os.path.join(input_dir, '*' + file_type))
    file_list.sort()

    for idx in range(len(file_list)):
        file_path = file_list[idx]
        file_name = os.path.basename(file_path)

        # 自定义重命名规则
        if file_name.startswith(old_file_format):
            rename_path = os.path.join(os.path.dirname(file_path), file_name.replace(old_file_format, new_file_format))

            print(file_path, '->', rename_path)
            os.rename(file_path, rename_path)
        