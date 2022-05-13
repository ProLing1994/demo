import argparse
import os
import pandas as pd
import re


def cut_log(args):
    log_list = []
    with open(args.log_path, "rb") as f:
        for line in f:
            if 'jsonStr.c_str' in line.decode('utf-8'):
                res = re.findall(r'\"VN\":\".*"}]', line.decode('utf-8'))
                if len(res):
                    log_list.append(res[0][6:-3])
                else:
                    print(line.decode('utf-8'))

    # out csv
    csv_data_pd = pd.DataFrame(log_list)
    csv_data_pd.to_csv(args.csv_path, index=False, encoding="utf_8_sig")


def main():
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.log_path = "/mnt/huanyuan2/data/image/log_0512_03.log"
    args.csv_path = "/mnt/huanyuan2/data/image/log_0512_03.csv"

    cut_log(args)


if __name__ == '__main__':
    main()