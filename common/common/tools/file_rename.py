import glob
import os

if __name__ == '__main__':
    # input_dir = "/mnt/huanyuan/data/speech/Negative_sample/QingTingFM/history/baijiajiangtan/"
    # file_format = 'baijiajiangtan_'
    # input_dir = "/mnt/huanyuan/data/speech/Negative_sample/QingTingFM/history/jinpingmei/"
    # file_format = 'jinpingmei_'
    # input_dir = "/mnt/huanyuan/data/speech/Negative_sample/QingTingFM/history/yeshimiwen/"
    # file_format = 'yeshimiwen_'
    # input_dir = "/mnt/huanyuan/data/speech/Negative_sample/QingTingFM/history/zhongdongwangshi/"
    # file_format = 'zhongdongwangshi_'
    # input_dir = "/mnt/huanyuan/data/speech/Negative_sample/QingTingFM/news/cishicike/"
    # file_format = 'cishicike_'
    # input_dir = "/mnt/huanyuan/data/speech/Negative_sample/QingTingFM/music_station/qingtingkongzhongyinyuebang/"
    # file_format = 'qingtingkongzhongyinyuebang_'
    # input_dir = "/mnt/huanyuan/data/speech/Negative_sample/QingTingFM/novel/douluodalu/"
    # file_format = 'douluodalu_'
    # input_dir = "/mnt/huanyuan/data/speech/Negative_sample/QingTingFM/music/xingetuijian/"
    # file_format = 'xingetuijian_'
    # input_dir = "/mnt/huanyuan/data/speech/Negative_sample/QingTingFM/music_station/naxienainzhuiguodege/"
    # file_format = 'naxienainzhuiguodege_'
    # input_dir = "/mnt/huanyuan/data/speech/Negative_sample/QingTingFM/music_station/quanqiuliuxingyinyuebang/"
    # file_format = 'quanqiuliuxingyinyuebang_'
    # input_dir = "/mnt/huanyuan/data/speech/Negative_sample/QingTingFM/music_station/shanggantingting/"
    # file_format = 'shanggantingting_'
    # input_dir = "/mnt/huanyuan/data/speech/Negative_sample/QingTingFM/music_station/suiyueruge/"
    # file_format = 'suiyueruge_'
    # input_dir = "/mnt/huanyuan/data/speech/Negative_sample/QingTingFM/news/jiaodianshike/"
    # file_format = 'jiaodianshike_'
    input_dir = "/mnt/huanyuan/data/speech/Recording_sample/Radio_sample/QingTingFM/news/全国新闻联播/"
    file_format = '全国新闻联播_'
    file_type = ".m4a"
    start_id = 1

    file_list = glob.glob(os.path.join(input_dir, '*' + file_type))
    file_list.sort()
    for idx in range(len(file_list)):
        file_path = file_list[idx]

        # 自定义重命名规则
        # rename_path = os.path.join(os.path.dirname(file_path), "{}{:0>5d}{}".format(file_format, (start_id + idx), file_type))
        
        basename = "_".join(os.path.basename(file_path).split('.')[0].split('_')[-3:])
        rename_path = os.path.join(os.path.dirname(file_path), "{:0>5d}_{}{}{}".format((start_id + idx), file_format, basename, file_type))

        print(file_path, '->', rename_path)
        # os.rename(file_path, rename_path)
