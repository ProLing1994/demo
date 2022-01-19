import argparse
import numpy as np


def analysis_log(args):

    time_info_dict = {}
    with open(args.log, "r") as f :
        for line in f:  
            time_str = line.strip().split(' - INFO - ')[1]
            time_str_list = time_str.split(',')
            for idx in range(len(time_str_list)):
                if ':' in time_str_list[idx]:
                    key = time_str_list[idx].split(':')[0]
                    value = time_str_list[idx].split(':')[1]

                    if '/' in value:
                        meav_value = value.split('/')[0]
                        std_value = value.split('/')[1]

                        if key in time_info_dict:
                            time_info_dict[key].append(float(meav_value))
                            # time_info_dict[key].append(float(std_value))
                        else:
                            time_info_dict[key] = []
                            time_info_dict[key].append(float(meav_value))
                    else:
                        if key in time_info_dict:
                            time_info_dict[key].append(float(value))
                        else:
                            time_info_dict[key] = []
                            time_info_dict[key].append(float(value))
    
    total_time = 0.0
    for key in time_info_dict.keys():
        total_time += np.sum(np.array(time_info_dict[key]))/1000
        print('key: {}, len: {}, total: {:.2f}s, mean: {:.2f}ms, std: {:.2f}ms '.format(key, len(time_info_dict[key]), np.sum(np.array(time_info_dict[key]))/1000, np.mean(np.array(time_info_dict[key])), np.std(np.array(time_info_dict[key])) ))

    print('total_time: {:.2f}h, {:.2f}m, {:.2f}s'.format(total_time/3600, total_time/60, total_time))
    return 


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-l', '--log', type=str, default="/mnt/huanyuan/model/kws/kws_xiaoan/test_time_cq/logging/train_time_log.txt")
    # parser.add_argument('-l', '--log', type=str, default="/mnt/huanyuan/model/kws/kws_xiaoan/test_time_cq_2/logging/train_time_log.txt")
    parser.add_argument('-l', '--log', type=str, default="/mnt/huanyuan/model/kws/kws_xiaoan/test_time_cq_3/logging/train_time_log.txt")
    # parser.add_argument('-l', '--log', type=str, default="/mnt/huanyuan/model/kws/kws_xiaoan/test_time_sz/logging/train_time_log.txt")
    args = parser.parse_args()

    analysis_log(args)


if __name__ == "__main__":
    main()
