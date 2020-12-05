#!/bin/bash

# 该脚本进行模型测试

stage=1

# init
# xiaorui
# config_file=/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaorui.py

# xiaoyu
config_file=/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaoyu.py

echo "test.sh"

# test model
if [ $stage -le 1 ];then
    python infer_longterm_audio_mean_min.py --input $config_file || exit 1
    python infer_longterm_audio_average_duration_ms.py --input $config_file || exit 1
fi

echo "test.sh succeeded"