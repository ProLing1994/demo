#!/bin/bash

# 该脚本进行模型测试

stage=1

# init
# xiaorui
config_file=/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaorui.py
model_epoch=7500

# xiaoyu
# config_file=/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaoyu.py

echo "test.sh"

# test model
if [ $stage -le 1 ];then
    python ./infer/infer.py --input $config_file --epoch $model_epoch --result_mode "min" || exit 1
    python ./infer/infer.py --input $config_file --epoch $model_epoch --result_mode "average_duration_ms" || exit 1
fi

echo "test.sh succeeded"