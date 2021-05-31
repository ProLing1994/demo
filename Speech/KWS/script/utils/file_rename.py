import glob
import os

if __name__ == '__main__':
    # input_dir = "/mnt/huanyuan/model/kws_model/asr_english/image_296_64/"
    # input_dir = "/mnt/huanyuan/model/audio_model/hisi_model/kws_xiaoan8k_tc_resnet14/image_48_146/"
    # input_dir = "/mnt/huanyuan/model/audio_model/hisi_model/asr_mandarin_taxi_16k/image_296_64/"
    input_dir = "/mnt/huanyuan/model/audio_model/hisi_model/kws_xiaorui16k_tc_resnet14/image_64_192/"
    file_format = 'pic_'
    file_type = ".jpg"
    start_id = 1

    file_list = glob.glob(os.path.join(input_dir, '*' + file_type))
    file_list.sort()

    with open(os.path.join(input_dir, "output.txt"), "w") as f :
        for idx in range(len(file_list)):
            file_path = file_list[idx]

            # 自定义重命名规则
            # rename_path = os.path.join(os.path.dirname(file_path), "{}{:0>5d}{}".format(file_format, (start_id + idx), file_type))
            rename_path = os.path.join(os.path.dirname(file_path), "{}{}{}".format(file_format, (start_id + idx), file_type))

            print(file_path, '->', rename_path)
            os.rename(file_path, rename_path)
            f.write("{}\n".format(os.path.basename(rename_path)))
        