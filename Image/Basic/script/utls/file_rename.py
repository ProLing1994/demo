import glob
import os
import random

if __name__ == '__main__':
    input_dir = "/mnt/huanyuan2/data/image/ZG_Face/JPEGImages_val/"
    file_format = 'pic_'
    file_type = ".jpg"
    start_id = 1

    file_list = glob.glob(os.path.join(input_dir, '*' + file_type))
    file_list.sort()
    random.shuffle(file_list)

    with open(os.path.join(input_dir, "output.txt"), "w") as f :
        for idx in range(len(file_list)):
            file_path = file_list[idx]

            # 自定义重命名规则
            rename_path = os.path.join(os.path.dirname(file_path), "{}{:0>5d}{}".format(file_format, (start_id + idx), file_type))
            # rename_path = os.path.join(os.path.dirname(file_path), "{}{}{}".format(file_format, (start_id + idx), file_type))
            
            # # 自定义重命名规则
            # basename = "_".join(os.path.basename(file_path).split('.')[0].split('_')[-3:])
            # rename_path = os.path.join(os.path.dirname(file_path), "{:0>5d}_{}{}{}".format((start_id + idx), file_format, basename, file_type))

            print(file_path, '->', rename_path)
            os.rename(file_path, rename_path)
            f.write("{}\n".format(os.path.basename(rename_path)))
            # f.write("{}\n".format('./ocr_image/' + os.path.basename(rename_path)))
        