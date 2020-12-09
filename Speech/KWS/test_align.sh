#!/bin/bash

# 该脚本进行模型测试，针对帧对齐的方式

stage=1

# init
# xiaoyu
config_file=/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_align_xiaoyu.py

echo "test.sh"

# test model
if [ $stage -le 1 ];then
    python ./infer/infer.py --input $config_file --result_mode "double_edge_triggered_detecting" || exit 1
fi

echo "test.sh succeeded"