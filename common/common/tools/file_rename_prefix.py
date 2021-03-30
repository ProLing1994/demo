import glob
import os

if __name__ == '__main__':
    input_dir = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/experimental_dataset/KwsEnglishDataset/activate/"
    file_format = 'RM_KWS_ACTIVATEBWC_activate_'
    file_type = ".wav"

    file_list = glob.glob(os.path.join(input_dir, '*' + file_type))
    file_list.sort()

    for idx in range(len(file_list)):
        file_path = file_list[idx]

        # 自定义重命名规则
        rename_path = os.path.join(os.path.dirname(file_path), "{}{}{}".format(file_format, file_path.split('_')[-1].split('.')[0], file_type))

        print(file_path, '->', rename_path)
        # os.rename(file_path, rename_path)
        