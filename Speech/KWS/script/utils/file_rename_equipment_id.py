import glob
import os

if __name__ == '__main__':
    input_dir = "/mnt/huanyuan/data/TODO/re_xiaorui/wav/"
    old_file_format = 'RM_KWS_XIAORUI_xiaorui_re_'
    new_file_format = 'RM_KWS_XIAORUI_xiaoruitransfer_'
    equipment_id = '11'
    file_type = ".wav"

    file_list = glob.glob(os.path.join(input_dir, '*' + file_type))
    file_list.sort()

    for idx in range(len(file_list)):
        file_path = file_list[idx]
        file_name = os.path.basename(file_path)

        # 自定义重命名规则
        if file_name.startswith(old_file_format):
            new_file_name = file_name.split('_')[-1].split('.')[0]
            new_file_name = new_file_name[:7]+ equipment_id + new_file_name[9:]
            rename_path = os.path.join(os.path.dirname(file_path), "{}{}{}".format(new_file_format, new_file_name, file_type))

            print(file_path, '->', rename_path)
            # os.rename(file_path, rename_path)
        