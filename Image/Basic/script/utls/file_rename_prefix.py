import glob
import os

if __name__ == '__main__':
    input_dir = "/mnt/huanyuan/temp/卡口2/2022-03-31/avi/"
    file_type = '.avi'
    prefix = '卡口2'
    to_prefix = "type2-"
    
    file_list = glob.glob(os.path.join(input_dir, '*' + file_type))
    file_list.sort()

    for idx in range(len(file_list)):
        file_path = file_list[idx]

        # 自定义重命名规则
        rename_path = os.path.join(os.path.dirname(file_path), os.path.basename(file_path).replace(prefix, to_prefix))

        print(file_path, '->', rename_path)
        # os.rename(file_path, rename_path)
        