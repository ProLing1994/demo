import glob
import os

if __name__ == '__main__':
    input_dir = "/mnt/huanyuan2/model/audio_model/novt_model/kws_xiaoan8k_tc_resnet14/image_48_196/"
    file_type = ".jpg"
    rename = 'pic_'

    file_list = glob.glob(os.path.join(input_dir, '*' + file_type))
    file_list.sort()

    for idx in range(len(file_list)):
        file_path = file_list[idx]
        file_name = os.path.basename(file_path)

        # 自定义重命名规则
        rename_path = os.path.join(os.path.dirname(file_path), "{}{}{}".format(rename, idx, file_type))

        print(file_path, '->', rename_path)
        os.rename(file_path, rename_path)
        