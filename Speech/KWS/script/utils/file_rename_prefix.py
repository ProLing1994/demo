import glob
import os

if __name__ == '__main__':
    input_dir = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/test_dataset/海外同事录制_0425/路边场景/场景二"
    # old_file_format = 'RM_KWS_ACTIVATEBWC_asr_'
    # new_file_format = 'RM_KWS_ACTIVATEBWC_ovweseas_asr_'
    old_file_format = 'RM_KWS_ACTIVATEBWC_ori_'
    new_file_format = 'RM_KWS_ACTIVATEBWC_ovweseas_ori_'
    file_type = ".wav"
    # file_type = ".txt"

    file_list = glob.glob(os.path.join(input_dir, '*' + file_type))
    file_list.sort()

    for idx in range(len(file_list)):
        file_path = file_list[idx]
        file_name = os.path.basename(file_path)

        # 自定义重命名规则
        if file_name.startswith(old_file_format):
            rename_path = os.path.join(os.path.dirname(file_path), "{}{}{}".format(new_file_format, file_name.split('_')[-1].split('.')[0], file_type))

            print(file_path, '->', rename_path)
            os.rename(file_path, rename_path)
        