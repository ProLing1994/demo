#!/bin/bash

# 该脚本进行模型测试，针对帧对齐的方式

stage=1

# init
# xiaoyu
# config_file=/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_align_xiaoyu.py
config_file=/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_align_pretrain.py

echo "test.sh"

# test model
if [ $stage -le 1 ];then
    python ./infer/infer.py --input $config_file --epoch 3000 --result_mode "double_edge_triggered_detecting" --timeshift_ms 30 --average_window_duration_ms 1500 || exit 1
fi

echo "test.sh succeeded"