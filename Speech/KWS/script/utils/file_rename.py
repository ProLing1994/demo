import glob
import os

if __name__ == '__main__':
    input_dir = "/mnt/huanyuan/model/tts/chinese_tts/sv2tts_chinese_new_tacotron2_singlespeaker_prosody_py_1_3_diff_feature_11292021/wavs_test/"
    file_type = ".h5"

    file_list = glob.glob(os.path.join(input_dir, '*' + file_type))
    file_list.sort()

    for idx in range(len(file_list)):
        file_path = file_list[idx]
        file_name = os.path.basename(file_path)

        # 自定义重命名规则
        rename_path = os.path.join(os.path.dirname(file_path), "{}{}".format(file_name.split('.')[0], file_type))

        print(file_path, '->', rename_path)
        # os.rename(file_path, rename_path)
        