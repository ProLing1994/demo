import glob
import os

if __name__ == '__main__':
    input_dir = "/mnt/huanyuan2/data/speech/original/Recording/MTA_Truck_Platformalarm/Gorila/ADPLUS2_0_8k/2023-2024/0808-0830/0823/"
    new_file_format = 'RM_KWS_GORLIA_platform_alarm_20230823T'
    file_type = ".wav"

    file_list = glob.glob(os.path.join(input_dir, '*' + file_type))
    file_list.sort()

    for idx in range(len(file_list)):
        file_path = file_list[idx]
        file_name = os.path.basename(file_path)

        # # 自定义重命名规则
        # if file_name.startswith(old_file_format):
        #     rename_path = os.path.join(os.path.dirname(file_path), "{}{}{}".format(new_file_format, file_name.split('_')[-1].split('.')[0], file_type))

        #     print(file_path, '->', rename_path)
        #     # os.rename(file_path, rename_path)

        # 自定义重命名规则
        rename_path = os.path.join(os.path.dirname(file_path), "{}{:0>3d}{}".format(new_file_format, idx, file_type))

        print(file_path, '->', rename_path)
        os.rename(file_path, rename_path)
        